# =============================================================================
# Der-AI | Football Quant Desk — V1
#
# One click on "Analyse" runs the whole pipeline:
#   1. PYTHON collects every fixture of the chosen day from API-Football (api-sports.io)
#      -> top leagues, mid leagues, cups, international matches, friendlies.
#   2. PYTHON enriches a budget-aware shortlist: predictions, last-10 form, H2H, injuries,
#      lineups, standings, odds (multi-bookmaker), corners/shots/xG history and individual
#      player ratings (batched fixtures?ids= calls keep the request count low).
#   3. PYTHON runs a Monte Carlo per match (Poisson goals + shared tempo variance +
#      Dixon-Coles low-score correction + first-half split + negative-binomial corners) for:
#      1 / X / 2, 1X, X2, 12, BTTS Yes/No, Over/Under 0.5-4.5, team totals, 1H goals,
#      total corners. Model probabilities are blended with de-vigged bookmaker prices.
#   4. PYTHON scores every match for "bookmaker trap" risk and builds a safe-leg menu.
#   5. AI (Groq, gpt-oss-120b -> gpt-oss-20b -> qwen3-32b) acts as bookmaker + quant, reads a
#      compact evidence pack and returns 3 tickets x 5 legs (each ticket odds >= 3.5).
#      The whole AI step is hard-capped at 8,000 tokens (prompt + completion, all attempts).
#   6. PYTHON verifies the AI output (real ids, real odds, 5 distinct matches, odds >= 3.5),
#      repairs it if needed, falls back to a deterministic optimiser if the AI is unavailable,
#      and pushes the tickets + reasoning to Telegram.
# =============================================================================

import os, re, json, math, time, html, hashlib, threading, random, traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── Configuration ───────────────────────────────────────────────────────────
APIF_BASE = "https://v3.football.api-sports.io"
DEFAULT_TZ = "Africa/Kampala"
CACHE_DIR = os.environ.get("DERAI_FB_CACHE", ".derai_fb_cache")
STATE_PATH = os.environ.get("DERAI_FB_STATE", "derai_football_state.json")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT = 90
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3-32b"]
GROQ_MODEL_CONFIG = {
    "openai/gpt-oss-120b": {"max_completion_tokens": 4000, "reasoning_effort": "low", "supports_reasoning_effort": True},
    "openai/gpt-oss-20b": {"max_completion_tokens": 4000, "reasoning_effort": "low", "supports_reasoning_effort": True},
    "qwen/qwen3-32b": {"max_completion_tokens": 3000, "reasoning_effort": "none", "supports_reasoning_effort": True},
}
GROQ_DEFAULT_CFG = {"max_completion_tokens": 2000, "reasoning_effort": None, "supports_reasoning_effort": False}
NON_RETRYABLE = {400, 401, 403, 404, 422}

# ── Token budget (HARD LIMIT per analysis: prompt + completion, across ALL attempts) ──
TOTAL_TOKEN_BUDGET = 8000
OUTPUT_RESERVE = 2200        # default completion tokens reserved (includes hidden reasoning on gpt-oss)
EFFORT_RESERVE = {"low": 2200, "medium": 2900, "high": 3500}   # more thinking = more room reserved = smaller prompt
TOKEN_SAFETY = 100
MIN_COMPLETION = 900         # below this a call cannot finish the JSON -> don't start it
DEFAULT_CHARS_PER_TOKEN = 3.0  # deliberately conservative for dense numeric text

# ── Ticket rules ────────────────────────────────────────────────────────────
N_TICKETS, LEGS_PER_TICKET, MIN_TICKET_ODDS = 3, 5, 3.5
MIN_LEG_P, MIN_LEG_ODDS, MAX_LEG_ODDS = 0.62, 1.15, 2.00
MIN_AGREE_P = 0.60   # BOTH the raw model and the bookmaker's fair price must clear this for a leg to be eligible
BASIS_TAG = {"season+recent": "S+R", "recent": "R", "season": "S", "prior": "P"}
TICKET_LABELS = ["Ticket 1 - Safest", "Ticket 2 - Balanced", "Ticket 3 - Value"]

# ── League intelligence (API-Football league ids; unknown leagues fall into LOW) ──
TOP_LEAGUES = {39: "Premier League", 140: "La Liga", 135: "Serie A", 78: "Bundesliga", 61: "Ligue 1",
               2: "Champions League", 3: "Europa League", 848: "Conference League", 1: "World Cup",
               4: "Euro", 9: "Copa America", 5: "Nations League"}
MID_LEAGUES = {88, 94, 40, 144, 203, 179, 71, 128, 262, 253, 307, 13, 11, 41, 42, 79, 136, 141, 62, 89, 95,
               145, 204, 180, 72, 218, 207, 119, 103, 113, 106, 197, 235, 98, 292, 188, 45, 48, 81, 66, 143, 137}
CAT_WEIGHT = {"TOP": 100, "INTL": 80, "MID": 70, "LOW": 50}
CAT_QUOTA = {"TOP": 0.40, "INTL": 0.20, "MID": 0.30, "LOW": 0.10}
LEAGUE_EXCLUDE_RE = re.compile(r"\b(U-?1[5-9]|U-?2[0-3]|Youth|Reserve|Reserves|Women|Feminine|Femenin|Femminile|Frauen|Junior|Juniors|Development)\b", re.I)
LOWER_DIV_RE = re.compile(r"amateur|regionalliga|serie d\b|division [3-9]|liga alef|norrland|friendlies clubs|oberliga|landesliga|primera federaci|segunda federaci|tercera|copa federacion", re.I)
TEAM_EXCLUDE_RE = re.compile(r"\s(II|B|U-?\d{2}|Youth|Women|W)$", re.I)
PLAYED_OK = {"FT", "AET", "PEN"}
UPCOMING_OK = {"NS", "TBD"}

GOAL_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]
TEAM_LINES = [0.5, 1.5, 2.5]
HT_LINES = [0.5, 1.5]
CORNER_LINES = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]


# ═════════════════════════════════════════════════════════════════════════════
# Small helpers
# ═════════════════════════════════════════════════════════════════════════════
def get_secret(name, default=""):
    try:
        v = st.secrets.get(name, None)
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get(name, default)


def _f(x, d=None):
    try:
        if x is None or x == "":
            return d
        if isinstance(x, str):
            x = x.replace("%", "").strip()
        return float(x)
    except Exception:
        return d


def _esc(t):
    return "" if t is None else html.escape(str(t), quote=False)


def _pct(p):
    return int(round(100 * p))


def _short(name, n=16):
    name = str(name or "")
    return name if len(name) <= n else name[: n - 1] + "."


# ═════════════════════════════════════════════════════════════════════════════
# API-Football client (thread-safe, rate-limited, disk + memory cache, quota aware)
# ═════════════════════════════════════════════════════════════════════════════
class RateLimiter:
    def __init__(self, per_minute):
        self.interval = 60.0 / max(1, per_minute)
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.interval
        if t > now:
            time.sleep(t - now)


class ApiFootball:
    def __init__(self, key, per_minute=10, cache_dir=CACHE_DIR):
        self.session = requests.Session()
        self.session.headers.update({"x-apisports-key": key})
        self.rate = RateLimiter(per_minute)
        self.lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0
        self.remaining = None
        self.limit = None
        self.errors: List[str] = []
        self.mem: Dict[str, Any] = {}
        self.quota_out = False
        self.blocked = set()       # parameters the plan refuses (e.g. 'last' on Free)
        self.blocked_ep = set()    # endpoints/seasons the plan refuses
        self.err_counts = {}
        self.ep_fail, self.ep_ok = {}, {}
        self.no_history = False    # plan cannot serve team fixture history for the current season
        self.cache_dir = cache_dir
        try:
            os.makedirs(cache_dir, exist_ok=True)
            for fn in os.listdir(cache_dir):
                p = os.path.join(cache_dir, fn)
                if time.time() - os.path.getmtime(p) > 2 * 86400:
                    os.remove(p)
        except Exception:
            pass

    def _note_headers(self, r):
        with self.lock:
            self.calls += 1
            rem = r.headers.get("x-ratelimit-requests-remaining")
            lim = r.headers.get("x-ratelimit-requests-limit")
            if rem is not None and str(rem).isdigit():
                self.remaining = int(rem)
            if lim is not None and str(lim).isdigit():
                self.limit = int(lim)

    def _note_error(self, endpoint, msg):
        key = f"{endpoint} {msg}"
        with self.lock:
            self.err_counts[key] = self.err_counts.get(key, 0) + 1
        self.errors.append(key)
        m = re.search(r"access to the (\w+) parameter", msg, re.I)
        if m:
            self.blocked.add(m.group(1).lower())
        elif "plan" in msg.lower() and "access" in msg.lower():
            with self.lock:                       # block an endpoint only after 3 plan failures and zero successes
                self.ep_fail[endpoint] = self.ep_fail.get(endpoint, 0) + 1
                if self.ep_fail[endpoint] >= 3 and not self.ep_ok.get(endpoint):
                    self.blocked_ep.add(endpoint)

    def error_summary(self):
        with self.lock:
            return [f"{k}  (x{v})" if v > 1 else k for k, v in self.err_counts.items()]

    def status(self):
        """/status does not consume quota; used to read plan + remaining calls."""
        try:
            r = self.session.get(f"{APIF_BASE}/status", timeout=20)
            d = r.json()
            resp = d.get("response") or {}
            if isinstance(resp, list):
                resp = resp[0] if resp else {}
            req = resp.get("requests") or {}
            sub = resp.get("subscription") or {}
            cur, lim = req.get("current"), req.get("limit_day")
            if lim is not None:
                self.limit = int(lim)
                self.remaining = int(lim) - int(cur or 0)
            return {"plan": sub.get("plan"), "active": sub.get("active"), "used": cur, "limit": lim,
                    "errors": d.get("errors")}
        except Exception as e:
            return {"plan": None, "errors": str(e)}

    def get(self, endpoint, params=None, ttl=3600, bypass=False):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        ck = endpoint + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        h = hashlib.md5(ck.encode()).hexdigest()
        path = os.path.join(self.cache_dir, h + ".json")
        if ttl > 0:
            with self.lock:
                mem = self.mem.get(h)
            if mem and time.time() - mem[0] < ttl:
                with self.lock:
                    self.cache_hits += 1
                return mem[1]
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as fh:
                        obj = json.load(fh)
                    if time.time() - obj["t"] < ttl:
                        with self.lock:
                            self.cache_hits += 1
                            self.mem[h] = (obj["t"], obj["d"])
                        return obj["d"]
            except Exception:
                pass
        if self.quota_out:
            return {"response": [], "paging": {}, "errors": {"quota": "daily quota exhausted"}}
        if any(k in self.blocked for k in params) or (endpoint in self.blocked_ep and endpoint != "/fixtures" and not bypass):
            return {"response": [], "paging": {}, "errors": {"blocked": "plan restriction (skipped, no call spent)"}}
        last_err = ""
        for attempt in range(3):
            self.rate.wait()
            try:
                r = self.session.get(APIF_BASE + endpoint, params=params, timeout=30)
            except Exception as e:
                last_err = str(e)
                time.sleep(1.5 * (attempt + 1))
                continue
            self._note_headers(r)
            if r.status_code == 429:
                time.sleep(6 * (attempt + 1))
                continue
            if r.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code != 200:
                self.errors.append(f"{endpoint} HTTP {r.status_code}")
                return {"response": [], "paging": {}, "errors": {"http": r.status_code}}
            try:
                data = r.json()
            except Exception:
                last_err = "bad json"
                continue
            errs = data.get("errors")
            if errs:
                msg = json.dumps(errs)[:300]
                low = msg.lower()
                if "ratelimit" in low or ("rate" in low and "limit" in low and "minute" in low):
                    time.sleep(8)
                    continue
                if "request limit" in low or "requests" in low and "day" in low:
                    self.quota_out = True
                self._note_error(endpoint, msg)
                data.setdefault("response", [])
                return data
            self.ep_ok[endpoint] = True
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump({"t": time.time(), "d": data}, fh)
            except Exception:
                pass
            with self.lock:
                self.mem[h] = (time.time(), data)
            return data
        self.errors.append(f"{endpoint} failed: {last_err}")
        return {"response": [], "paging": {}, "errors": {"network": last_err}}

    def resp(self, endpoint, params=None, ttl=3600):
        return self.get(endpoint, params, ttl).get("response") or []

    def pages(self, endpoint, params, ttl=1200, max_pages=4):
        out, page = [], 1
        while page <= max_pages:
            d = self.get(endpoint, dict(params, page=page), ttl)
            out.extend(d.get("response") or [])
            pg = d.get("paging") or {}
            if int(pg.get("current", page) or page) >= int(pg.get("total", 1) or 1):
                break
            page += 1
        return out


# ═════════════════════════════════════════════════════════════════════════════
# Parsers
# ═════════════════════════════════════════════════════════════════════════════
def league_category(lid, country):
    if lid in TOP_LEAGUES:
        return "TOP"
    if (country or "").lower() == "world":
        return "INTL"
    if lid in MID_LEAGUES:
        return "MID"
    return "LOW"


def parse_fixture(f):
    fx, lg, tm = f.get("fixture", {}), f.get("league", {}), f.get("teams", {})
    return {
        "id": fx.get("id"), "ts": fx.get("timestamp"), "status": (fx.get("status") or {}).get("short"),
        "league_id": lg.get("id"), "league": lg.get("name"), "country": lg.get("country"),
        "season": lg.get("season"), "round": lg.get("round"),
        "home_id": (tm.get("home") or {}).get("id"), "home": (tm.get("home") or {}).get("name"),
        "away_id": (tm.get("away") or {}).get("id"), "away": (tm.get("away") or {}).get("name"),
        "cat": league_category(lg.get("id"), lg.get("country")),
    }


def _norm_line(txt):
    m = re.match(r"\s*(over|under)\s*([\d.]+)", txt.lower())
    if not m:
        return None
    line = float(m.group(2))
    if abs(line * 2 - round(line * 2)) > 1e-9 or int(line * 2) % 2 == 0:
        return None
    return ("O" if m.group(1) == "over" else "U") + f"{line:g}"


def canonical_market_key(name, value):
    n, v = name.lower().strip(), str(value).strip().lower()
    if n == "match winner":
        return {"home": "1", "draw": "X", "away": "2"}.get(v)
    if n == "double chance":
        return {"home/draw": "1X", "draw/away": "X2", "home/away": "12"}.get(v)
    if n == "both teams score":
        return {"yes": "BTTS_Y", "no": "BTTS_N"}.get(v)
    if n == "goals over/under":
        k = _norm_line(v)
        return k if k and float(k[1:]) in GOAL_LINES else None
    if n.startswith("goals over/under") and ("first half" in n or "1st half" in n):
        k = _norm_line(v)
        return "HT_" + k if k and float(k[1:]) in HT_LINES else None
    if "half" in n or "1st" in n or "2nd" in n or "corner" in n and ("home" in n or "away" in n):
        return None
    bad = ("card", "shot", "handicap", "race", "exact", "odd/even", "result", "minute", "first", "last")
    if "total" in n and any(w in n for w in ("home", "away")) and "corner" not in n and not any(b in n for b in bad):
        side = "H_" if "home" in n else "A_"
        k = _norm_line(v)
        return side + k if k and float(k[1:]) in TEAM_LINES else None
    if "corner" in n and ("over" in n or "total" in n) and not any(b in n for b in ("handicap", "race", "asian")):
        k = _norm_line(v)
        return "C_" + k if k and float(k[1:]) in CORNER_LINES else None
    return None


def parse_odds_item(item, pref_bookmaker):
    store, pref = defaultdict(list), {}
    for bk in item.get("bookmakers", []) or []:
        for bet in bk.get("bets", []) or []:
            for v in bet.get("values", []) or []:
                key = canonical_market_key(str(bet.get("name", "")), str(v.get("value", "")))
                odd = _f(v.get("odd"))
                if not key or not odd or odd <= 1.0:
                    continue
                store[key].append(odd)
                if bk.get("id") == pref_bookmaker:
                    pref[key] = odd
    out = {}
    for k, lst in store.items():
        med = float(np.median(lst))
        out[k] = {"odds": float(pref.get(k, med)), "median": med, "best": float(max(lst)), "low": float(min(lst)), "n": len(lst)}
    return out


def parse_predictions(resp):
    if not resp:
        return None
    r = resp[0]
    pr, teams, comp = r.get("predictions") or {}, r.get("teams") or {}, r.get("comparison") or {}

    def side(s):
        t = teams.get(s) or {}
        L = t.get("league") or {}
        g = L.get("goals") or {}
        gf, ga = (g.get("for") or {}).get("average") or {}, (g.get("against") or {}).get("average") or {}
        pl = (L.get("fixtures") or {}).get("played") or {}
        l5 = t.get("last_5") or {}
        return {"gf_h": _f(gf.get("home")), "gf_a": _f(gf.get("away")), "ga_h": _f(ga.get("home")),
                "ga_a": _f(ga.get("away")), "played_h": _f(pl.get("home"), 0), "played_a": _f(pl.get("away"), 0),
                "l5_form": l5.get("form"), "season_form": L.get("form") or "", "l5_played": _f(l5.get("played"), 0),
                "l5_form_pct": _f(l5.get("form")), "l5_gf": _f(((l5.get("goals") or {}).get("for") or {}).get("average")),
                "l5_ga": _f(((l5.get("goals") or {}).get("against") or {}).get("average"))}

    pc = pr.get("percent") or {}
    return {"pct": [_f(pc.get("home")), _f(pc.get("draw")), _f(pc.get("away"))], "advice": pr.get("advice"),
            "home": side("home"), "away": side("away"),
            "comp": {k: (_f((comp.get(k) or {}).get("home")), _f((comp.get(k) or {}).get("away")))
                     for k in ("form", "att", "def", "total")},
            "h2h": r.get("h2h") or []}


def _ft_goals(f):
    sc = (f.get("score") or {}).get("fulltime") or {}
    g = f.get("goals") or {}
    h = sc.get("home") if sc.get("home") is not None else g.get("home")
    a = sc.get("away") if sc.get("away") is not None else g.get("away")
    return h, a


def parse_results(resp, team_id):
    out = []
    for f in resp or []:
        if ((f.get("fixture") or {}).get("status") or {}).get("short") not in PLAYED_OK:
            continue
        h, a = _ft_goals(f)
        if h is None or a is None:
            continue
        tm = f.get("teams") or {}
        is_home = (tm.get("home") or {}).get("id") == team_id
        gf, ga = (h, a) if is_home else (a, h)
        opp = tm.get("away" if is_home else "home") or {}
        out.append({"id": f["fixture"]["id"], "ts": f["fixture"]["timestamp"], "home": is_home, "gf": gf, "ga": ga,
                    "res": "W" if gf > ga else "D" if gf == ga else "L", "opp": opp.get("name"),
                    "league_id": (f.get("league") or {}).get("id")})
    out.sort(key=lambda x: -x["ts"])
    return out


def parse_h2h(fixtures, home_id):
    """Results from the perspective of the current home team."""
    out = []
    for f in fixtures or []:
        if ((f.get("fixture") or {}).get("status") or {}).get("short") not in PLAYED_OK:
            continue
        h, a = _ft_goals(f)
        if h is None or a is None:
            continue
        was_home = ((f.get("teams") or {}).get("home") or {}).get("id") == home_id
        gf, ga = (h, a) if was_home else (a, h)
        out.append({"ts": f["fixture"]["timestamp"], "gf": gf, "ga": ga})
    out.sort(key=lambda x: -x["ts"])
    return out[:8]


def parse_standings(resp):
    out = {}
    for item in resp or []:
        for group in (item.get("league") or {}).get("standings") or []:
            size = len(group)
            for row in group:
                tid = (row.get("team") or {}).get("id")
                if tid is not None and tid not in out:
                    out[tid] = {"rank": row.get("rank"), "pts": row.get("points"), "of": size,
                                "played": (row.get("all") or {}).get("played"), "desc": row.get("description")}
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Team profile (form, goals, corners, xG, player form, injuries, lineup)
# ═════════════════════════════════════════════════════════════════════════════
def _stat_val(stats_list, name):
    for s in stats_list or []:
        if s.get("type") == name:
            return _f(s.get("value"))
    return None


def team_history_stats(team_id, ids, hist):
    cor_f, cor_a, sot, xg_f, xg_a = [], [], [], [], []
    players = defaultdict(lambda: {"name": "", "pos": None, "r": [], "m": [], "g": 0, "ast": 0, "games": 0})
    for fid in ids:
        f = hist.get(fid)
        if not f:
            continue
        own = opp = None
        for s in f.get("statistics") or []:
            if (s.get("team") or {}).get("id") == team_id:
                own = s.get("statistics")
            else:
                opp = s.get("statistics")
        if own and opp:
            c1, c2 = _stat_val(own, "Corner Kicks"), _stat_val(opp, "Corner Kicks")
            if c1 is not None and c2 is not None:
                cor_f.append(c1)
                cor_a.append(c2)
            v = _stat_val(own, "Shots on Goal")
            if v is not None:
                sot.append(v)
            x1, x2 = _stat_val(own, "expected_goals"), _stat_val(opp, "expected_goals")
            if x1 is not None and x2 is not None:
                xg_f.append(x1)
                xg_a.append(x2)
        for tp in f.get("players") or []:
            if (tp.get("team") or {}).get("id") != team_id:
                continue
            for p in tp.get("players") or []:
                st_ = (p.get("statistics") or [{}])[0]
                gm = st_.get("games") or {}
                mins, rt = _f(gm.get("minutes"), 0), _f(gm.get("rating"))
                if not mins:
                    continue
                pid = (p.get("player") or {}).get("id")
                e = players[pid]
                e["name"] = (p.get("player") or {}).get("name") or e["name"]
                e["pos"] = gm.get("position") or e["pos"]
                e["m"].append(mins)
                if rt:
                    e["r"].append(rt)
                e["g"] += int((st_.get("goals") or {}).get("total") or 0)
                e["ast"] += int((st_.get("goals") or {}).get("assists") or 0)
                e["games"] += 1
    avg = lambda l: float(np.mean(l)) if l else None
    return {"cor_for": avg(cor_f), "cor_against": avg(cor_a), "cor_n": len(cor_f), "sot": avg(sot),
            "xg_for": avg(xg_f), "xg_against": avg(xg_a), "players": dict(players)}


def build_profile(team_id, results, hs, injuries, lineup):
    r10 = results[:10]
    w = np.array([0.88 ** i for i in range(len(r10))]) if r10 else np.array([])
    prof = {"n": len(r10)}
    if r10:
        gf = np.array([x["gf"] for x in r10], float)
        ga = np.array([x["ga"] for x in r10], float)
        pts = np.array([3 if x["res"] == "W" else 1 if x["res"] == "D" else 0 for x in r10], float)
        prof.update({
            "gf_w": float((w * gf).sum() / w.sum()), "ga_w": float((w * ga).sum() / w.sum()),
            "ppg5": float(pts[:5].mean()), "ppg10": float(pts.mean()),
            "form5": "".join(x["res"] for x in reversed(r10[:5])),
            "cs": float((ga == 0).mean()), "fts": float((gf == 0).mean()),
            "o25": float(((gf + ga) > 2.5).mean()), "o15": float(((gf + ga) > 1.5).mean()),
            "btts": float(((gf > 0) & (ga > 0)).mean()),
            "form_score": float(100 * (w * pts / 3).sum() / w.sum()),
        })
    prof.update({k: hs.get(k) for k in ("cor_for", "cor_against", "cor_n", "sot", "xg_for", "xg_against")})
    # ── individual players ──
    pl = hs.get("players") or {}
    ngames = max([e["games"] for e in pl.values()] + [1])
    rows = []
    for pid, e in pl.items():
        if not e["r"]:
            continue
        rows.append({"id": pid, "name": e["name"], "pos": e["pos"], "rating": float(np.mean(e["r"])),
                     "mins": float(np.mean(e["m"])), "games": e["games"], "goals": e["g"], "ast": e["ast"]})
    key = [r for r in rows if r["games"] >= max(2, int(0.6 * ngames)) and r["mins"] >= 55 and r["rating"] >= 6.7]
    key.sort(key=lambda r: -(r["rating"] * math.log(r["mins"] + 1)))
    prof["key"] = key[:9]
    prof["key_ids"] = {r["id"] for r in prof["key"]}
    prof["stars"] = sorted([r for r in rows if r["games"] >= 2], key=lambda r: -(r["rating"] + 0.25 * r["goals"] + 0.1 * r["ast"]))[:2]
    mm = [(r["rating"], r["mins"]) for r in rows if r["mins"] >= 30]
    prof["team_rating"] = float(sum(a * b for a, b in mm) / sum(b for _, b in mm)) if mm else None
    # ── injuries ──
    inj = []
    pos_by_name = {r["name"]: r["pos"] for r in rows}
    for i in injuries or []:
        p = i.get("player") or {}
        typ = p.get("type") or ""
        pid, nm = p.get("id"), p.get("name")
        is_key = pid in prof["key_ids"]
        inj.append({"name": nm, "type": typ, "reason": p.get("reason"), "key": is_key,
                    "pos": next((r["pos"] for r in rows if r["id"] == pid), pos_by_name.get(nm))})
    prof["inj"] = inj
    # ── lineup ──
    prof["lineup"] = None
    if lineup:
        xi = {((x.get("player") or {}).get("id")) for x in lineup.get("startXI") or []}
        kin = len(prof["key_ids"] & xi)
        prof["lineup"] = {"formation": lineup.get("formation"), "key_in_xi": kin, "key_n": len(prof["key_ids"]),
                          "rotation": bool(len(prof["key_ids"]) >= 5 and kin <= len(prof["key_ids"]) - 3)}
    return prof


def enrich_profile_from_l5(prof, side):
    """When team history is unavailable (Free plan), use the predictions endpoint's last-5 block as recent form."""
    if prof.get("n", 0) >= 4 or not side:
        return prof
    n, gf, ga, fp = int(side.get("l5_played") or 0), side.get("l5_gf"), side.get("l5_ga"), side.get("l5_form_pct")
    if n >= 3 and gf is not None and ga is not None:
        sf = "".join(ch for ch in (side.get("season_form") or "") if ch in "WDL")[-5:]
        ppg = (fp / 100 * 3) if fp is not None else 1.35
        prof.update({"n": n, "gf_w": gf, "ga_w": ga, "ppg5": ppg, "ppg10": ppg,
                     "form5": sf or (f"~{int(fp)}%" if fp is not None else "?"), "form_score": fp if fp is not None else 50.0,
                     "src": "api_last5"})
    return prof


def injury_impact(prof):
    att = dfn = 0.0
    for i in prof.get("inj") or []:
        wgt = 1.0 if i["type"] == "Missing Fixture" else 0.4
        base = 0.045 if i["key"] else 0.012
        pos = i.get("pos")
        if pos in ("F", "M", None):
            att += base * wgt * (1.0 if pos == "F" else 0.7 if pos == "M" else 0.5)
        if pos in ("D", "G", None):
            dfn += base * wgt * (1.2 if pos == "G" else 1.0 if pos == "D" else 0.5)
    return min(att, 0.14), min(dfn, 0.14)


# ═════════════════════════════════════════════════════════════════════════════
# Expected goals / corners
# ═════════════════════════════════════════════════════════════════════════════
def estimate_lambdas(pred, ph, pa):
    """Returns (lam_h, lam_a, injury_impacts, basis). `basis` records what real data the numbers rest on;
    'prior' means NO team-specific data was available (league-average defaults) and the match is unusable."""
    prior_h, prior_a = 1.45, 1.15
    season_h = season_a = None
    n_season = 0
    if pred:
        H, A = pred["home"], pred["away"]
        v = [H["gf_h"], H["ga_h"], A["gf_a"], A["ga_a"]]
        n_s = min(H["played_h"] or 0, A["played_a"] or 0)
        if None not in v and sum(v) > 0 and n_s >= 3:
            season_h, season_a, n_season = (v[0] + v[3]) / 2, (v[2] + v[1]) / 2, n_s
    rec_h = rec_a = None
    n_rec = min(ph.get("n", 0), pa.get("n", 0))
    if n_rec >= 4:
        rec_h = (ph["gf_w"] + pa["ga_w"]) / 2
        rec_a = (pa["gf_w"] + ph["ga_w"]) / 2
    if season_h is not None and rec_h is not None:
        ws = 0.5 if n_season >= 6 else 0.3
        lh, la, m, basis = ws * season_h + (1 - ws) * rec_h, ws * season_a + (1 - ws) * rec_a, max(n_season, n_rec), "season+recent"
    elif season_h is not None:
        lh, la, m, basis = season_h, season_a, n_season, "season"
    elif rec_h is not None:
        lh, la, m, basis = rec_h, rec_a, n_rec, "recent"
    else:
        lh, la, m, basis = prior_h, prior_a, 0, "prior"
    lh = (m * lh + 3 * prior_h) / (m + 3)
    la = (m * la + 3 * prior_a) / (m + 3)
    (ah, dh), (aa, da) = injury_impact(ph), injury_impact(pa)
    lh *= (1 - ah) * (1 + da)
    la *= (1 - aa) * (1 + dh)
    return float(np.clip(lh, 0.25, 3.8)), float(np.clip(la, 0.25, 3.8)), (ah, dh, aa, da), basis


def estimate_corners(ph, pa, lh, la):
    prior_h, prior_a = 5.0 + 0.9 * (lh - 1.45), 4.3 + 0.9 * (la - 1.15)
    n = 0
    mu_h, mu_a = prior_h, prior_a
    if None not in (ph.get("cor_for"), pa.get("cor_against"), pa.get("cor_for"), ph.get("cor_against")):
        n = min(ph.get("cor_n", 0), pa.get("cor_n", 0))
        mu_h = (n * (ph["cor_for"] + pa["cor_against"]) / 2 + 3 * prior_h) / (n + 3)
        mu_a = (n * (pa["cor_for"] + ph["cor_against"]) / 2 + 3 * prior_a) / (n + 3)
    return float(mu_h), float(mu_a), n


# ═════════════════════════════════════════════════════════════════════════════
# Monte Carlo
# ═════════════════════════════════════════════════════════════════════════════
def monte_carlo(lam_h, lam_a, mu_ch, mu_ca, n=40000, seed=0):
    rng = np.random.default_rng(seed)
    tempo = rng.gamma(28.0, 1 / 28.0, n)                        # shared game-tempo -> goal correlation/overdispersion
    lh = lam_h * tempo * np.exp(rng.normal(-0.005, 0.10, n))    # parameter uncertainty
    la = lam_a * tempo * np.exp(rng.normal(-0.005, 0.10, n))
    h1, h2 = rng.poisson(lh * 0.45), rng.poisson(lh * 0.55)
    a1, a2 = rng.poisson(la * 0.45), rng.poisson(la * 0.55)
    h, a, hh, ah = h1 + h2, a1 + a2, h1, a1
    rho = -0.06                                                 # Dixon-Coles low-score correction as sim weights
    w = np.ones(n)
    m00, m01, m10, m11 = (h == 0) & (a == 0), (h == 0) & (a == 1), (h == 1) & (a == 0), (h == 1) & (a == 1)
    w[m00] = 1 - lh[m00] * la[m00] * rho
    w[m01] = 1 + lh[m01] * rho
    w[m10] = 1 + la[m10] * rho
    w[m11] = 1 - rho
    w = np.clip(w, 0.2, None)
    ws = w.sum()
    P = lambda mask: float((w * mask).sum() / ws)
    tot, p = h + a, {}
    p["1"], p["X"], p["2"] = P(h > a), P(h == a), P(h < a)
    p["1X"], p["X2"], p["12"] = p["1"] + p["X"], p["X"] + p["2"], p["1"] + p["2"]
    p["BTTS_Y"] = P((h > 0) & (a > 0))
    p["BTTS_N"] = 1 - p["BTTS_Y"]
    for L in GOAL_LINES:
        p[f"O{L:g}"] = P(tot > L)
        p[f"U{L:g}"] = 1 - p[f"O{L:g}"]
    for L in TEAM_LINES:
        p[f"H_O{L:g}"], p[f"H_U{L:g}"] = P(h > L), 1 - P(h > L)
        p[f"A_O{L:g}"], p[f"A_U{L:g}"] = P(a > L), 1 - P(a > L)
    for L in HT_LINES:
        p[f"HT_O{L:g}"] = P((hh + ah) > L)
        p[f"HT_U{L:g}"] = 1 - p[f"HT_O{L:g}"]
    r = 35.0
    tc = rng.gamma(40.0, 1 / 40.0, n)
    ch = rng.negative_binomial(r, r / (r + mu_ch * tc))
    ca = rng.negative_binomial(r, r / (r + mu_ca * tc))
    ct = ch + ca
    for L in CORNER_LINES:
        p[f"C_O{L:g}"] = P(ct > L)
        p[f"C_U{L:g}"] = 1 - p[f"C_O{L:g}"]
    code = np.clip(h, 0, 9) * 10 + np.clip(a, 0, 9)
    cnt = np.bincount(code, weights=w, minlength=100)
    top = np.argsort(-cnt)[:3]
    return {"p": p, "exp_goals": float((w * tot).sum() / ws), "exp_corners": float((w * ct).sum() / ws),
            "top_scores": [f"{int(c // 10)}-{int(c % 10)}" for c in top]}


# ═════════════════════════════════════════════════════════════════════════════
# Market blending, trap analysis, legs
# ═════════════════════════════════════════════════════════════════════════════
def _devig(odds_map, keys):
    if all(k in odds_map for k in keys):
        inv = [1 / odds_map[k]["odds"] for k in keys]
        s = sum(inv)
        return {k: i / s for k, i in zip(keys, inv)}
    return None


def blend_probs(model_p, odds_map, w_model):
    p, mkt_1x2 = dict(model_p), None
    d = _devig(odds_map, ["1", "X", "2"])
    if d:
        b = {k: w_model * model_p[k] + (1 - w_model) * d[k] for k in ("1", "X", "2")}
        s = sum(b.values())
        b = {k: v / s for k, v in b.items()}
        p.update(b)
        p["1X"], p["X2"], p["12"] = b["1"] + b["X"], b["X"] + b["2"], b["1"] + b["2"]
        mkt_1x2 = d
    pairs = [("BTTS_Y", "BTTS_N")]
    for pre, lines in (("", GOAL_LINES), ("H_", TEAM_LINES), ("A_", TEAM_LINES), ("HT_", HT_LINES), ("C_", CORNER_LINES)):
        pairs += [(f"{pre}O{L:g}", f"{pre}U{L:g}") for L in lines]
    for a_, b_ in pairs:
        if a_ in model_p:
            dd = _devig(odds_map, [a_, b_])
            if dd:
                p[a_] = w_model * model_p[a_] + (1 - w_model) * dd[a_]
                p[b_] = 1 - p[a_]
    return p, mkt_1x2


def h2h_summary(h2h):
    if len(h2h) < 3:
        return None
    w = sum(1 for x in h2h if x["gf"] > x["ga"])
    d = sum(1 for x in h2h if x["gf"] == x["ga"])
    return {"n": len(h2h), "w": w, "d": d, "l": len(h2h) - w - d, "g": float(np.mean([x["gf"] + x["ga"] for x in h2h]))}


def trap_analysis(m):
    mp, mk = m["model_p"], m.get("mkt_1x2")
    ref = mk or {k: mp[k] for k in ("1", "X", "2")}
    fav = max(("1", "2"), key=lambda k: ref[k])
    fp = ref[fav]
    if fp < 0.52:
        return {"risk": 8, "fav": None, "reasons": ["no clear favourite"]}
    fav_home = fav == "1"
    pf, pu = (m["ph"], m["pa"]) if fav_home else (m["pa"], m["ph"])
    risk, why = 0, []
    gap = fp - mp[fav]
    if gap >= 0.14:
        risk += 35; why.append(f"mkt{_pct(fp)}/mdl{_pct(mp[fav])}")
    elif gap >= 0.08:
        risk += 22; why.append(f"mkt{_pct(fp)}/mdl{_pct(mp[fav])}")
    elif gap >= 0.05:
        risk += 10; why.append(f"mkt{_pct(fp)}/mdl{_pct(mp[fav])}")
    if pf.get("n", 0) >= 4 and pf["ppg5"] < 1.4 and fp >= 0.6:
        risk += 15; why.append(f"fav ppg{pf['ppg5']:.1f}")
    if pu.get("n", 0) >= 4 and pu["ppg5"] >= 1.8:
        risk += 10; why.append(f"dog ppg{pu['ppg5']:.1f}")
    keyout = [i for i in pf.get("inj", []) if i["key"] and i["type"] == "Missing Fixture"]
    if keyout:
        risk += min(20, 8 * len(keyout)); why.append(f"{len(keyout)} fav key out")
    if (pf.get("lineup") or {}).get("rotation"):
        risk += 15; why.append("fav rotated XI")
    h = m.get("h2h")
    if h and h["n"] >= 4:
        fav_unbeaten = (h["w"] + h["d"]) if fav_home else (h["l"] + h["d"])
        dog_unbeaten = h["n"] - (h["w"] if fav_home else h["l"])
        if dog_unbeaten / h["n"] >= 0.6:
            risk += 10; why.append(f"H2H dog unbeaten {dog_unbeaten}/{h['n']}")
    if not fav_home and fp >= 0.6:
        risk += 8; why.append("away fav")
    if mp["X"] >= 0.27:
        risk += 10; why.append(f"X{_pct(mp['X'])}%")
    lam_f = m["lam_h"] if fav_home else m["lam_a"]
    if lam_f < 1.5 and fp >= 0.62:
        risk += 10; why.append("fav low xG")
    dog_st = m.get("st_a") if fav_home else m.get("st_h")
    if fav_home is False and dog_st and dog_st.get("rank") and dog_st.get("of") and dog_st["rank"] > dog_st["of"] - 4:
        risk += 8; why.append("home dog in drop zone")
    lname = (m.get("league") or "").lower()
    if "friendl" in lname:
        risk += 20; why.append("friendly")
    elif "cup" in lname or "copa" in lname or "pokal" in lname or "coppa" in lname:
        risk += 8; why.append("cup rotation")
    return {"risk": int(min(100, risk)), "fav": fav, "reasons": why[:4]}


def data_quality(m):
    dq = 0.0
    dq += 0.25 if m.get("odds") else 0
    dq += 0.20 if m.get("pred") else 0
    dq += 0.20 if m["ph"].get("n", 0) >= 5 and m["pa"].get("n", 0) >= 5 else (0.10 if m["ph"].get("n", 0) >= 3 else 0)
    dq += 0.10 if m.get("h2h") else 0
    dq += 0.10 if m.get("corner_n", 0) >= 3 else 0
    dq += 0.05 if m.get("inj_known") else 0
    dq += 0.05 if (m["ph"].get("lineup") and m["pa"].get("lineup")) else 0
    dq += 0.05 if m.get("st_h") else 0
    if not m.get("odds"):          # odds availability is judged by the gate; here score only the team data (max 0.75)
        dq = dq / 0.75
    return round(min(1.0, dq), 2)


def leg_label(key, m):
    h, a = m["home"], m["away"]
    fixed = {"1": f"{h} to win", "X": "Draw", "2": f"{a} to win", "1X": f"{h} or Draw", "X2": f"Draw or {a}",
             "12": "Either team to win", "BTTS_Y": "Both teams score - Yes", "BTTS_N": "Both teams score - No"}
    if key in fixed:
        return fixed[key]
    m_ = re.match(r"(H_|A_|HT_|C_)?(O|U)([\d.]+)$", key)
    if m_:
        pre, ou, line = m_.groups()
        word = "Over" if ou == "O" else "Under"
        return {None: f"{word} {line} goals", "H_": f"{h} {word} {line} goals", "A_": f"{a} {word} {line} goals",
                "HT_": f"1st half {word} {line} goals", "C_": f"{word} {line} corners"}[pre]
    return key


def _family(key):
    if key in ("1", "X", "2", "1X", "X2", "12"):
        return "RES"
    if key.startswith("BTTS"):
        return "BTTS"
    pre = re.match(r"(H_|A_|HT_|C_)?", key).group(1) or "G"
    return pre + ("O" if "O" in key.split("_")[-1][:1] else "U")


def market_fair_prob(key, om):
    """Bookmaker's margin-free probability for a selection (de-vigged when the full market is known)."""
    d = _devig(om, ["1", "X", "2"])
    if d:
        if key in d:
            return d[key]
        if key in ("1X", "X2", "12"):
            return sum(d[c] for c in {"1X": "1X", "X2": "X2", "12": "12"}[key])
    mm = re.match(r"(H_|A_|HT_|C_)?(O|U)([\d.]+)$", key)
    if mm:
        pre, ou, line = mm.groups()
        partner = f"{pre or ''}{'U' if ou == 'O' else 'O'}{line}"
        dd = _devig(om, [key, partner])
        if dd:
            return dd[key]
    if key.startswith("BTTS"):
        dd = _devig(om, ["BTTS_Y", "BTTS_N"])
        if dd:
            return dd[key]
    return min(0.97, 1 / (om[key]["odds"] * 1.06))


def build_legs(m, allow_est=False, corners_ok_min=3):
    fav, risk = m["trap"]["fav"], m["trap"]["risk"]
    dq = m["dq"]
    legs = []
    for key, p in m["p"].items():
        if key.startswith("C_") and m.get("corner_n", 0) < corners_ok_min:
            continue
        o = m["odds"].get(key)
        src = "book"
        lo = hi = None
        if o:
            odds = o["odds"]
            lo, hi = o.get("low", odds), o.get("best", odds)
            p_mkt = market_fair_prob(key, m["odds"])
        elif allow_est and not m["odds"]:           # only estimate when the match has NO bookmaker prices at all
            odds, src = round(max(1.05, 0.93 / max(p, 0.05)), 2), "est"
            p_mkt = p
        else:
            continue
        if not (MIN_LEG_ODDS <= odds <= MAX_LEG_ODDS):
            continue
        p_model = m["model_p"].get(key, p)
        if min(p_model, p_mkt) < MIN_AGREE_P:      # two independent sources must both back the leg
            continue
        pen = 0.05 * (1 - dq)
        if fav and key == fav:
            pen += 0.25 * risk / 100
        elif fav and key == ("1X" if fav == "1" else "X2"):
            pen += 0.10 * risk / 100
        elif risk:
            pen += 0.03 * risk / 100
        if "friendl" in (m.get("league") or "").lower():
            pen += 0.03
        pen += 0.5 * max(0.0, abs(p_model - p_mkt) - 0.08)   # unexplained model/market disagreement
        p_adj = p - pen
        if p_adj < MIN_LEG_P:
            continue
        edge = p * odds - 1
        eadj = float(np.clip(p_adj * odds - 1, -0.2, 0.15))
        legs.append({"key": key, "label": leg_label(key, m), "p": float(p), "p_adj": float(p_adj), "p_mkt": float(p_mkt),
                     "p_model": float(p_model), "odds": float(odds), "odds_lo": lo, "odds_hi": hi, "edge": float(edge), "src": src,
                     "rank": float(p_adj + 0.5 * eadj), "fam": _family(key)})
    legs.sort(key=lambda x: -x["rank"])
    chosen, fams = [], set()
    for lg in legs:
        if lg["fam"] in fams:
            continue
        chosen.append(lg)
        fams.add(lg["fam"])
        if len(chosen) >= 6:
            break
    return chosen


def gate_reasons(m, cfg):
    """Evidence gate: a match may only feed tickets if it rests on real team data AND real prices."""
    why = []
    if m["basis"] == "prior":
        why.append("no team-specific data (league-average defaults only)")
    if not m["odds"] and not cfg.get("allow_est"):
        why.append("no real bookmaker odds")
    if m["dq"] < cfg.get("min_dq", 0.5):
        why.append(f"data quality {m['dq']} < {cfg.get('min_dq', 0.5)}")
    if not why and not m["legs"]:
        why.append("no leg clears the safety filters (model AND market >= 60%)")
    return why


# ═════════════════════════════════════════════════════════════════════════════
# Data collection pipeline
# ═════════════════════════════════════════════════════════════════════════════
def fetch_coverage(api, ttl_s=86400):
    """One call: which competitions the API covers (predictions/odds/injuries/stats) this season."""
    cov = {}
    for item in api.resp("/leagues", {"current": "true"}, ttl_s):
        lid = (item.get("league") or {}).get("id")
        seasons = item.get("seasons") or []
        cur = next((x for x in seasons if x.get("current")), seasons[-1] if seasons else None)
        c = (cur or {}).get("coverage") or {}
        fxc = c.get("fixtures") or {}
        cov[lid] = {"pred": bool(c.get("predictions")), "odds": bool(c.get("odds")), "inj": bool(c.get("injuries")),
                    "stats": bool(fxc.get("statistics_fixtures")), "lineups": bool(fxc.get("lineups")), "standings": bool(c.get("standings"))}
    return cov


def shortlist_matches(cands, max_n, force_leagues):
    buckets = defaultdict(list)
    forced = [c for c in cands if c["league_id"] in force_leagues]
    for c in cands:
        if c not in forced:
            buckets[c["cat"]].append(c)
    order = list(TOP_LEAGUES)
    for cat, lst in buckets.items():
        lst.sort(key=lambda c: (0 if c.get("cov_odds") else 1 if c.get("cov_odds") is None else 2,
                                order.index(c["league_id"]) if c["league_id"] in order else 99, c["ts"]))
    picked = list(forced)[:max_n]
    for cat in ("TOP", "INTL", "MID", "LOW"):
        take = math.ceil(CAT_QUOTA[cat] * max_n)
        for c in buckets[cat][:take]:
            if len(picked) < max_n:
                picked.append(c)
    for cat in ("TOP", "MID", "INTL", "LOW"):
        for c in buckets[cat]:
            if len(picked) >= max_n:
                break
            if c not in picked:
                picked.append(c)
    return picked


def estimate_calls(matches, deep):
    leagues = len({m["league_id"] for m in matches})
    return 2 + leagues * (2 if deep else 1) + len(matches) * (3.7 if deep else 3.2)


def detail_cost(ms, deep):
    return len(ms) * (3.9 if deep else 3.3) + (len({m["league_id"] for m in ms}) if deep else 0)


def team_results(api, team_id, season, date_str, tz, ttl_s):
    """Last-10 results. Free plans block `last` and current-season queries, so try fallbacks once, then stop."""
    if api.no_history:
        return []
    if "last" not in api.blocked:
        r = api.resp("/fixtures", {"team": team_id, "last": 10, "timezone": tz}, ttl_s)
        if r:
            return parse_results(r, team_id)
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    base = {"team": team_id, "from": (d - timedelta(days=240)).isoformat(), "to": (d - timedelta(days=1)).isoformat(), "timezone": tz}
    d1 = api.get("/fixtures", base, ttl_s)
    res = parse_results(d1.get("response") or [], team_id)
    if not res and d1.get("errors") and season:
        d2 = api.get("/fixtures", dict(base, season=season), ttl_s)
        res = parse_results(d2.get("response") or [], team_id)
        if not res and d2.get("errors"):
            api.no_history = True
    return res[:10]


def collect_day(api, cfg, log, progress):
    out = {"warnings": [], "matches": [], "excluded": [], "raw_count": 0, "cat_counts": {}, "plan": None, "reason": None}
    date_str, tz, force = cfg["date"], cfg["tz"], cfg["force"]
    ttl = lambda t: 0 if force else t
    stt = api.status()
    out["plan"] = stt
    if stt.get("errors") and not stt.get("plan"):
        out["warnings"].append(f"API status problem: {stt.get('errors')}")
    if api.remaining is not None and api.remaining <= 5:
        out["warnings"].append("API daily quota almost exhausted - only cached data can be used.")
    budget = min(cfg["max_calls"], api.remaining if api.remaining is not None else cfg["max_calls"])
    if api.remaining is not None and budget < 60:
        out["warnings"].append(f"Only {api.remaining} API calls are left today (plan limit {api.limit}), so this run was limited to fit. "
                               "The quota resets at 00:00 UTC (03:00 in Kampala); cached data from earlier runs is reused for free.")
    log(f"📥 Fetching all fixtures for {date_str} ({tz})...")
    fx_raw = api.resp("/fixtures", {"date": date_str, "timezone": tz}, ttl=ttl(900))
    out["raw_count"] = len(fx_raw)
    fixtures = [parse_fixture(f) for f in fx_raw]
    fixtures = [f for f in fixtures if f["status"] in UPCOMING_OK and f["home_id"] and f["away_id"]]
    if cfg["exclude_minor"]:
        fixtures = [f for f in fixtures if not LEAGUE_EXCLUDE_RE.search(f["league"] or "")
                    and not TEAM_EXCLUDE_RE.search(f["home"] or "") and not TEAM_EXCLUDE_RE.search(f["away"] or "")]
    if cfg.get("exclude_lower", True):
        fixtures = [f for f in fixtures if not LOWER_DIV_RE.search(f["league"] or "")]
    cov = fetch_coverage(api)
    if cov:
        for f in fixtures:
            c_ = cov.get(f["league_id"])
            f["cov_pred"], f["cov_odds"] = (c_["pred"], c_["odds"]) if c_ else (None, None)
        n0 = len(fixtures)
        fixtures = [f for f in fixtures if f["cov_pred"] is not False]
        log(f"🗂 Coverage filter: dropped {n0 - len(fixtures)} fixtures in competitions the API has no predictions for; "
            f"{sum(1 for f in fixtures if f['cov_odds'])} of {len(fixtures)} remaining have odds coverage")
    out["cat_counts"] = {k: int(v) for k, v in pd.Series([f["cat"] for f in fixtures]).value_counts().items()} if fixtures else {}
    log(f"   {out['raw_count']} fixtures on the day, {len(fixtures)} upcoming & eligible {out['cat_counts']}")
    if not fixtures:
        out["reason"] = "No eligible upcoming fixtures were returned by the API. " + ("; ".join(api.error_summary()[:3]) if api.errors else "")
        return out
    progress(0.06)

    # ── Stage 1: candidate pool ──
    pool_n = min(len(fixtures), int(cfg["max_matches"] * 1.6) + 4)
    cand = shortlist_matches(fixtures, pool_n, cfg["force_leagues"])
    odds_cap = max(6, int(budget * 0.30))
    while len({c["league_id"] for c in cand}) > odds_cap and pool_n > 8:
        pool_n -= 1
        cand = shortlist_matches(fixtures, pool_n, cfg["force_leagues"])
    odds_by = {}

    def fetch_league_odds(lid, s_):
        for params in ({"league": lid, "season": s_, "date": date_str, "timezone": tz}, {"league": lid, "season": s_, "date": date_str}):
            items, errs = [], False
            for page in (1, 2, 3):
                d = api.get("/odds", dict(params, page=page), ttl(1200))
                items += d.get("response") or []
                if d.get("errors"):
                    errs = True
                    break
                pg = d.get("paging") or {}
                if int(pg.get("current", page) or page) >= int(pg.get("total", 1) or 1):
                    break
            if items or not errs:
                return items
        return []

    def store_odds(items, fid=None):
        for item in items:
            f_ = fid or (item.get("fixture") or {}).get("id")
            if f_:
                odds_by[f_] = parse_odds_item(item, cfg["bookmaker"])

    # ── capability probe: a few calls to learn what this API plan allows, so we plan around it ──
    log("🔬 Probing what your API plan allows (a few calls)...")
    sample = cand[0]
    team_results(api, sample["home_id"], sample["season"], date_str, tz, ttl(3 * 3600))
    caps = {"history": not api.no_history, "league_odds": None, "fixture_odds": False}
    items = fetch_league_odds(sample["league_id"], sample["season"])
    store_odds(items)
    if items:
        caps["league_odds"] = True
    elif api.ep_fail.get("/odds", 0) > 0 and not api.ep_ok.get("/odds"):
        caps["league_odds"] = False
    if caps["league_odds"] is False:
        for c in cand[:2]:                                   # season-free per-fixture odds may still be allowed
            its = api.get("/odds", {"fixture": c["id"]}, ttl(1200), True).get("response") or []
            if its:
                store_odds(its, c["id"])
                caps["fixture_odds"] = True
                break
    out["caps"] = caps
    log(f"   team history: {'yes' if caps['history'] else 'NO (using predictions last-5 form instead)'} | "
        f"odds by league: {'yes' if caps['league_odds'] else 'NO' if caps['league_odds'] is False else 'unknown'} | "
        f"odds by fixture: {'yes' if caps['fixture_odds'] else 'no'}")

    log(f"💰 Stage 1: bookmaker odds for {len(cand)} candidate matches in {len({c['league_id'] for c in cand})} competitions...")
    if caps["league_odds"] is not False:
        todo = {(c["league_id"], c["season"]) for c in cand} - {(sample["league_id"], sample["season"])}
        with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
            for fu in as_completed([ex.submit(fetch_league_odds, lid, s_) for lid, s_ in todo]):
                try:
                    store_odds(fu.result())
                except Exception as e:
                    api.errors.append(f"odds: {e}")
        if not odds_by and caps["league_odds"] is None and (api.remaining is None or api.remaining > 20):
            for c in cand[1:4]:
                its = api.get("/odds", {"fixture": c["id"]}, ttl(1200), True).get("response") or []
                if its:
                    store_odds(its, c["id"])
                    caps["fixture_odds"] = True
    with_odds = [c for c in cand if c["id"] in odds_by]
    if caps["fixture_odds"] or (cfg["allow_est"] and not (caps["league_odds"] and len(with_odds) >= 10)):
        keep = list(cand)                                  # odds come per fixture later, or are estimated for matches without prices
    else:
        keep = with_odds
    keep_ids = {c["id"] for c in keep}
    for c in cand:
        if c["id"] not in keep_ids:
            out["excluded"].append({"league": c["league"], "match": f"{c['home']} v {c['away']}", "reason": "no bookmaker odds"})
    log(f"   {len(odds_by)} candidates have bookmaker prices so far -> {len(keep)} continue")
    progress(0.14)
    if not keep:
        errs = [e for e in api.error_summary() if "/odds" in e][:2] or api.error_summary()[-2:]
        out["reason"] = ("No bookmaker odds were returned for any candidate match. "
                         + ("API said: " + " | ".join(e[:170] for e in errs) + ". " if errs else "No error was reported - odds may not be published yet for this day. ")
                         + "Enable 'model-estimated odds' in Settings to continue with approximate prices.")
        return out

    # ── budget-aware cap on how many get deep data ──
    per = 1.0 + (2.0 if caps["history"] else 0.0) + (1.0 if caps["fixture_odds"] else 0.0)
    deep = caps["history"] and (cfg["depth"] == "Full" or (cfg["depth"] == "Auto" and budget >= 150))

    def cost(ms, dp):
        return len(ms) * (per + (0.9 if dp else 0.0)) + (len({m["league_id"] for m in ms}) if dp else 0)

    left = budget - api.calls
    n = min(cfg["max_matches"], len(keep))
    while n > 4 and cost(keep[:n], deep) > left * 0.95:
        n -= 1
    if deep and n < min(cfg["max_matches"], len(keep)) and n < 12:
        n_l = min(cfg["max_matches"], len(keep))
        while n_l > 4 and cost(keep[:n_l], False) > left * 0.95:
            n_l -= 1
        if n_l >= n + 4:
            deep, n = False, n_l
    sl = keep[:n]
    for c in keep[n:]:
        out["excluded"].append({"league": c["league"], "match": f"{c['home']} v {c['away']}", "reason": "beyond API call budget"})
    out["deep"], out["budget"] = deep, budget
    leagues = {(m["league_id"], m["season"]) for m in sl}
    est_calls = cost(sl, deep)
    log(f"🎯 Stage 2: data for {len(sl)} matches | depth={'FULL' if deep else 'LITE'} | est. ~{est_calls:.0f} more calls "
        f"(budget left ~{left:.0f}) | ~{est_calls * api.rate.interval / 60:.1f} min at your plan's rate limit")

    # ── injuries ──
    log("🩹 Fetching injuries & suspensions...")
    inj_by = defaultdict(list)
    inj_raw = api.resp("/injuries", {"date": date_str, "timezone": tz}, ttl=ttl(3600))
    for i in inj_raw:
        inj_by[((i.get("fixture") or {}).get("id"), (i.get("team") or {}).get("id"))].append(i)
    inj_known = bool(inj_raw)
    progress(0.18)

    # ── predictions + last-10 (both teams) ──
    log("📊 Fetching predictions, team form (last 10) and H2H...")
    core = {}

    def fetch_core(m):
        pred = parse_predictions(api.resp("/predictions", {"fixture": m["id"]}, ttl=ttl(6 * 3600)))
        rh = team_results(api, m["home_id"], m["season"], date_str, tz, ttl(3 * 3600))
        ra = team_results(api, m["away_id"], m["season"], date_str, tz, ttl(3 * 3600))
        h2h = parse_h2h(pred["h2h"], m["home_id"]) if pred else []
        return m["id"], {"pred": pred, "rh": rh, "ra": ra, "h2h": h2h}

    def viable(c):
        pr = c["pred"]
        if pr:
            H, A = pr["home"], pr["away"]
            if None not in (H["gf_h"], H["ga_h"], A["gf_a"], A["ga_a"]) and min(H["played_h"] or 0, A["played_a"] or 0) >= 3:
                return True
            if (H.get("l5_played") or 0) >= 3 and (A.get("l5_played") or 0) >= 3 and H.get("l5_gf") is not None and A.get("l5_gf") is not None:
                return True
        return len(c["rh"]) >= 4 and len(c["ra"]) >= 4

    # probe up to 4 matches from DIFFERENT competitions; abort only if none has usable team data
    seen_lg, probe = set(), []
    for m in sl:
        if m["league_id"] not in seen_lg:
            seen_lg.add(m["league_id"])
            probe.append(m)
        if len(probe) == 4:
            break
    for m in probe:
        fid, c = fetch_core(m)
        core[fid] = c
        if sum(1 for x in core.values() if viable(x)) >= 2:
            break
    if not any(viable(c) for c in core.values()):
        out["reason"] = ("Team form and season statistics could not be retrieved for the matches probed (see API errors below - "
                         "usually a plan restriction). Stopped early to protect your daily API quota; without team data any ticket would be a guess.")
        for m in sl:
            out["excluded"].append({"league": m["league"], "match": f"{m['home']} v {m['away']}", "reason": "no team-specific data retrievable"})
        return out
    rest = [m for m in sl if m["id"] not in core]
    done = len(core)
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
        futs = [ex.submit(fetch_core, m) for m in rest]
        for fu in as_completed(futs):
            try:
                fid, c = fu.result()
                core[fid] = c
            except Exception as e:
                api.errors.append(f"core: {e}")
            done += 1
            progress(0.18 + 0.37 * done / len(sl))
    if caps["fixture_odds"]:
        log("💰 Fetching per-fixture bookmaker odds...")
        with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
            futs = {ex.submit(lambda mm: api.get("/odds", {"fixture": mm["id"]}, ttl(1200), True).get("response") or [], m): m
                    for m in sl if m["id"] not in odds_by and m.get("cov_odds") is not False}
            for fu in as_completed(futs):
                try:
                    store_odds(fu.result(), futs[fu]["id"])
                except Exception as e:
                    api.errors.append(f"fixture odds: {e}")
    if deep:
        for m in sl:
            c = core.get(m["id"])
            if c and len(c["h2h"]) < 3 and "last" not in api.blocked and (api.remaining is None or api.remaining > 40) and not api.quota_out:
                c["h2h"] = parse_h2h(api.resp("/fixtures/headtohead", {"h2h": f"{m['home_id']}-{m['away_id']}", "last": 8}, ttl=ttl(24 * 3600)), m["home_id"])

    # ── batched history stats (corners, shots, xG, player ratings) + upcoming lineups + standings ──
    hist, upc, st_by = {}, {}, {}
    if deep:
        log("🧮 Batch-fetching statistics, player ratings & lineups (fixtures?ids=)...")
        hist_ids = set()
        for c in core.values():
            for r in (c["rh"][:5] + c["ra"][:5]):
                hist_ids.add(r["id"])

        def batch(ids, ttl_s):
            ids = sorted(ids)
            chunks = [ids[i:i + 20] for i in range(0, len(ids), 20)]
            res = {}
            with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
                for fu in as_completed([ex.submit(api.resp, "/fixtures", {"ids": "-".join(map(str, c_))}, ttl_s) for c_ in chunks]):
                    for f in fu.result():
                        res[f["fixture"]["id"]] = f
            return res
        hist = batch(hist_ids, ttl(6 * 3600))
        progress(0.75)
        upc = batch([m["id"] for m in sl], ttl(600))
        with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
            futs = {ex.submit(api.resp, "/standings", {"league": lid, "season": s_}, ttl(12 * 3600)): lid for lid, s_ in leagues}
            for fu in as_completed(futs):
                try:
                    st_by[futs[fu]] = parse_standings(fu.result())
                except Exception:
                    st_by[futs[fu]] = {}
    progress(0.85)

    # ── build match objects ──
    log("🧠 Building profiles and running Monte Carlo simulations...")
    for m in sl:
        c = core.get(m["id"]) or {"pred": None, "rh": [], "ra": [], "h2h": []}
        hs_h = team_history_stats(m["home_id"], [r["id"] for r in c["rh"][:5]], hist)
        hs_a = team_history_stats(m["away_id"], [r["id"] for r in c["ra"][:5]], hist)
        lus = {((l.get("team") or {}).get("id")): l for l in (upc.get(m["id"]) or {}).get("lineups") or []}
        ph = build_profile(m["home_id"], c["rh"], hs_h, inj_by.get((m["id"], m["home_id"])), lus.get(m["home_id"]))
        pa = build_profile(m["away_id"], c["ra"], hs_a, inj_by.get((m["id"], m["away_id"])), lus.get(m["away_id"]))
        enrich_profile_from_l5(ph, (c["pred"] or {}).get("home"))
        enrich_profile_from_l5(pa, (c["pred"] or {}).get("away"))
        lam_h, lam_a, _, basis = estimate_lambdas(c["pred"], ph, pa)
        mu_ch, mu_ca, cn = estimate_corners(ph, pa, lam_h, lam_a)
        mc = monte_carlo(lam_h, lam_a, mu_ch, mu_ca, n=cfg["sims"], seed=int(m["id"]) % (2 ** 31))
        odds = odds_by.get(m["id"], {})
        p_bl, mkt = blend_probs(mc["p"], odds, cfg["w_model"])
        rec = dict(m, ph=ph, pa=pa, pred=c["pred"], h2h=h2h_summary(c["h2h"]), lam_h=lam_h, lam_a=lam_a, basis=basis,
                   model_p=mc["p"], p=p_bl, mkt_1x2=mkt, odds=odds, exp_goals=mc["exp_goals"],
                   exp_corners=mc["exp_corners"], top_scores=mc["top_scores"], corner_n=cn,
                   st_h=(st_by.get(m["league_id"]) or {}).get(m["home_id"]),
                   st_a=(st_by.get(m["league_id"]) or {}).get(m["away_id"]), inj_known=inj_known)
        rec["trap"] = trap_analysis(rec)
        rec["dq"] = data_quality(rec)
        out["matches"].append(rec)
    thin = sum(1 for r in out["matches"] if r["basis"] == "prior")
    if thin:
        out["warnings"].append(f"{thin}/{len(out['matches'])} matches have NO team-specific data (plan restriction, early season or missing coverage) and are excluded from tickets.")
    progress(0.90)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Evidence pack for the AI (hard token budget)
# ═════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are two people at once: head of trading at a sharp sportsbook (you know exactly how prices are shaded to punish public money) and a professional quantitative analyst. Python already collected the day's football data and ran a 40,000-run Monte Carlo per match (Poisson goals, shared-tempo variance, Dixon-Coles correction, negative-binomial corners), then blended it with de-vigged bookmaker prices. Audit that evidence and build @NT@ separate accumulator(s) of 5 legs each.

DATA KEY: every match shown already passed Python's evidence gate (real bookmaker prices + team-specific data); S+R/R/S after dq = data behind xG (season+recent form / recent form / season stats). xG=model expected goals home-away; mdl=model 1X2 %; mkt=bookmaker 1/X/2 prices; API=third-party win %; BTTS/O/U/corners = blended probabilities %; H/A lines = form (oldest>newest), ppg=points per game (last5), gf/ga=weighted goals for/against, rt=avg player rating, rk=table rank; H2H=W-D-L from home team's view + goals/game; inj=missing/doubtful (*=key starter, F/M/D/G=position); lu=formation, ? = not released, ROT=key starters benched; star=in-form players; dq=data quality 0-1; TRAP=Python trap score 0-100 + reasons. LEGS: id@decimal_odds/p=blended probability %/m=bookmaker margin-free probability %/e=edge %. Distrust a leg when p and m differ by >8 pts unless you can explain it.

HARD RULES
1. Pick legs ONLY by exact id from LEGS lists. Never invent legs/odds/matches. Python recomputes odds and rejects invalid picks.
2. Exactly @NT@ ticket(s) x 5 legs; 5 different matches per ticket@REUSE@.
3. Each ticket's combined odds (product of leg odds) must be >= 3.50; aim 3.5-4.6. Reach it with well-priced SAFE legs, never by adding a fragile leg.
4. Maximise each ticket's win probability (product of p) subject to rule 3. Prefer p>=75; accept 65-74 only with strong evidence.
5. Ticket 1 = safest possible, Ticket 2 (if any) = balanced, Ticket 3 (if any) = best value (positive edge) but still solid. Quality over quantity: if a leg is merely mediocre, swap it for a safer one from the list.

HOW TO REASON
- Price vs model: if mkt and mdl differ by >8 pts, explain it with hard evidence (injuries, lineups, form, motivation, rotation). No evidence = trust the market. A big + edge on a short price is usually model error, not a gift.
- TRAP DETECTION: books profit most when the obvious favourite in a big game fails. TRAP>=45 = suspect, 30-44 = watch. Signals: favourite shorter than model says, favourite in poor form, key starters out or rotated, dangerous underdog (form/H2H), away favourite, draw prob >=27%, cups/friendlies/dead rubbers, relegation-threatened home side. For a suspect NEVER use the favourite-win leg: skip the match, or use a leg that survives the trap (draw-covering double chance on the underdog side, goals-market leg, Under/BTTS-No) only if p is high and evidence supports it. Apply your own judgement to matches Python scored low too.
- Failure test per leg: name the scenario that loses it (early red, 1-0 low block, rotation, tempo) and judge how common it is. Reject legs whose losing scenario is common.
- Goals logic: Over1.5 solid at total xG>=2.6; Over2.5 needs >=3.0; Under needs total xG<=2.3 with weak attacks; BTTS-No needs a side with xG<=0.9; team over1.5 needs own xG>=1.9. Corners follow attacking volume; use only with dq>=0.6.
- Diversify market types and leagues; avoid five Overs or five favourites; do not stack legs with the same failure mode. Categories: TOP=major leagues, MID, INTL=international/continental, LOW=thin data (needs extra margin). Low dq (<0.6), unreleased lineups near kickoff, friendlies need extra margin or exclusion.

OUTPUT: ONE JSON object only, no markdown, all reasoning inside these fields, terse:
{"tickets":[{"name":"Ticket 1 - Safest","legs":[{"id":"m3.1X","why":"<=14 words of evidence"}, ...5 legs],"logic":"<=40 words: why these five together, what could break it","risk":"LOW|MED"}, ...@NT@ tickets],"traps":[{"id":"m5","note":"<=20 words: trap read + action"}],"avoid":[{"id":"m9","why":"<=12 words"}],"summary":"<=50 words overall slate read"}"""


def system_prompt(n_tickets, reuse):
    return (SYSTEM_PROMPT.replace("@NT@", str(n_tickets))
            .replace("@REUSE@", "; a match may appear in more than one ticket only if unavoidable" if reuse else f"; {n_tickets * LEGS_PER_TICKET} different matches overall (no match reused)"))


def format_match_block(mid, m, n_legs, tz):
    p = m["p"]
    ko = datetime.fromtimestamp(m["ts"], ZoneInfo(tz)).strftime("%H:%M")
    ph, pa = m["ph"], m["pa"]
    mk = m.get("mkt_1x2")
    mk_txt = "/".join(f"{m['odds'][k]['odds']:.2f}" for k in ("1", "X", "2")) if all(k in m["odds"] for k in ("1", "X", "2")) else "n/a"
    api_txt = "/".join(str(int(x)) if x is not None else "?" for x in (m["pred"] or {}).get("pct", [None] * 3)) if m["pred"] else "n/a"

    def side(pr, lab):
        if not pr.get("n"):
            return f"{lab} n/a"
        rt = f" rt{pr['team_rating']:.1f}" if pr.get("team_rating") else ""
        return f"{lab} {pr['form5']} ppg{pr['ppg5']:.1f} gf{pr['gf_w']:.1f} ga{pr['ga_w']:.1f}{rt}"

    def inj(pr):
        items = [f"{_short((i['name'] or '?').split(' ')[-1], 10)}{'*' if i['key'] else ''}({i['pos'] or '?'}){'?' if i['type'] != 'Missing Fixture' else ''}" for i in pr.get("inj", [])[:4]]
        return ",".join(items) if items else "-"

    def lu(pr):
        l = pr.get("lineup")
        return "?" if not l else (l["formation"] or "ok") + ("ROT" if l["rotation"] else "")

    def star(pr):
        return ",".join(f"{_short(s['name'].split(' ')[-1], 9)}{s['rating']:.1f}" for s in pr.get("stars", [])[:1]) or "-"

    rk = f" rk{m['st_h']['rank']}/{m['st_a']['rank']}" if m.get("st_h") and m.get("st_a") else ""
    h2 = f"H2H{m['h2h']['n']} {m['h2h']['w']}-{m['h2h']['d']}-{m['h2h']['l']} g{m['h2h']['g']:.1f}" if m.get("h2h") else "H2H n/a"
    corn = f" | crn{m['exp_corners']:.1f} O9.5:{_pct(p['C_O9.5'])}" if m.get("corner_n", 0) >= 3 else ""
    t = m["trap"]
    trap = f"TRAP{t['risk']}: {', '.join(t['reasons'])}" if t["risk"] >= 25 else f"TRAP{t['risk']}"
    legs = " ; ".join(f"{mid}.{l['key']}@{l['odds']:.2f}/p{_pct(l['p'])}/m{_pct(l['p_mkt'])}/e{l['edge'] * 100:+.0f}" for l in m["legs"][:n_legs])
    return (f"#{mid} {m['cat']}|{_short(m['league'], 18)} {_short(m['home'], 15)} v {_short(m['away'], 15)} {ko} dq{m['dq']} {BASIS_TAG.get(m['basis'], '?')}\n"
            f"xG {m['lam_h']:.2f}-{m['lam_a']:.2f} mdl {_pct(m['model_p']['1'])}/{_pct(m['model_p']['X'])}/{_pct(m['model_p']['2'])} mkt {mk_txt} API {api_txt}"
            f" | BTTS{_pct(p['BTTS_Y'])} O1.5:{_pct(p['O1.5'])} O2.5:{_pct(p['O2.5'])} O3.5:{_pct(p['O3.5'])}{corn}\n"
            f"{side(ph, 'H')} | {side(pa, 'A')}{rk} | {h2}\n"
            f"inj H:{inj(ph)} A:{inj(pa)} | lu H:{lu(ph)} A:{lu(pa)} | star H:{star(ph)} A:{star(pa)}\n"
            f"{trap}\nLEGS {legs}")


def est_tokens(text, cpt):
    return int(len(text) / cpt) + 1


def build_ai_pack(matches, tz, cpt, n_ai, sys_prompt, need, reserve=OUTPUT_RESERVE):
    """Choose matches + legs-per-match so that prompt + reserved completion <= 8,000 tokens."""
    usable = [m for m in matches if m["legs"]]
    usable.sort(key=lambda m: -(sum(l["rank"] for l in m["legs"][:2]) / min(2, len(m["legs"]))))
    top = usable[:n_ai]
    flagged = sorted([m for m in usable if m["trap"]["risk"] >= 45 and m not in top], key=lambda m: -m["trap"]["risk"])[:2]
    prompt_cap = TOTAL_TOKEN_BUDGET - reserve - TOKEN_SAFETY
    sys_tok = est_tokens(sys_prompt, cpt)
    floor = min(need, len(top))
    for n_legs in (5, 4, 3, 2):
        for n in range(len(top), max(0, floor - 1), -1):
            sel = top[:n]
            if flagged and n > floor + len(flagged):
                sel = top[:n - len(flagged)] + flagged
            ids = {id(m): f"m{i + 1}" for i, m in enumerate(sel)}
            user = (f"SLATE: {len(sel)} pre-screened matches (of {len(matches)} analysed). Build the ticket(s) from these LEGS only.\n\n"
                    + "\n\n".join(format_match_block(ids[id(m)], m, n_legs, tz) for m in sel))
            if sys_tok + est_tokens(user, cpt) <= prompt_cap:
                return sel, {ids[id(m)]: m for m in sel}, user, n_legs
    sel = top[:max(floor, 1)]
    ids = {id(m): f"m{i + 1}" for i, m in enumerate(sel)}
    user = "SLATE:\n\n" + "\n\n".join(format_match_block(ids[id(m)], m, 2, tz) for m in sel)
    return sel, {ids[id(m)]: m for m in sel}, user, 2


# ═════════════════════════════════════════════════════════════════════════════
# Groq (budgeted, model fallback)
# ═════════════════════════════════════════════════════════════════════════════
class TokenBudget:
    def __init__(self, total):
        self.total, self.used = total, 0

    @property
    def remaining(self):
        return self.total - self.used


def _parse_json(content):
    content = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.I)
    content = re.sub(r"\s*```$", "", content).strip()
    for cand in (content, content[content.find("{"): content.rfind("}") + 1] if "{" in content else ""):
        if not cand:
            continue
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", cand))
        except Exception:
            continue
    return None


def _retry_after(text):
    m = re.search(r"try again in ([0-9hms.]+)", text or "")
    if not m:
        return None
    sec = 0.0
    for val, unit in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1)):
        v = float(val)
        sec += v / 1000 if unit == "ms" else v * 3600 if unit == "h" else v * 60 if unit == "m" else v
    return sec


def call_groq_budgeted(system_prompt, user_prompt, budget, cpt, effort=None):
    """Calls Groq inside the hard token budget. The prompt estimate is padded, the completion cap is bounded by the chosen
    reasoning effort, and 413/429 errors are parsed so the request is corrected instead of blindly retried."""
    api_key = get_secret("GROQ_API_KEY", "").strip()
    log = []
    if not api_key:
        return None, {"status": "MISSING_KEY", "attempts": [{"model": "-", "status": "GROQ_API_KEY not set"}], "prompt_tokens": 0, "completion_tokens": 0}
    chars = len(system_prompt + user_prompt)
    est_prompt = est_tokens(system_prompt + user_prompt, cpt)
    info = {"attempts": log, "model": None, "prompt_tokens": 0, "completion_tokens": 0}
    too_large = False
    for model in GROQ_MODELS:
        cfg = GROQ_MODEL_CONFIG.get(model, GROQ_DEFAULT_CFG)
        eff = effort or cfg.get("reasoning_effort")
        upper = cfg["max_completion_tokens"]
        if model.startswith("openai/") and eff in EFFORT_RESERVE:
            upper = min(upper, EFFORT_RESERVE[eff] + 100)
        limit = TOTAL_TOKEN_BUDGET               # per-request token limit; refined from error messages
        p_est = int(est_prompt * 1.15) + 60      # pessimistic prompt size until the API tells us the real one
        for attempt in range(3):
            cap = min(upper, budget.remaining - p_est - 60, limit - p_est - 60)
            if cap < MIN_COMPLETION:
                log.append({"model": model, "status": f"SKIPPED_BUDGET(prompt~{p_est}, cap={cap})"})
                too_large = True
                break
            payload = {"model": model, "temperature": 0.2, "max_completion_tokens": int(cap),
                       "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}
            if cfg.get("supports_reasoning_effort") and eff:
                payload["reasoning_effort"] = eff if model.startswith("openai/") or eff == "none" else "none"
                if model.startswith("openai/"):
                    payload["include_reasoning"] = False
            try:
                r = requests.post(GROQ_API_URL, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=GROQ_TIMEOUT)
            except Exception as e:
                log.append({"model": model, "status": f"NETWORK {str(e)[:80]}"})
                time.sleep(2)
                continue
            if r.status_code == 200:
                d = r.json()
                u = d.get("usage") or {}
                pt, ct = u.get("prompt_tokens", est_prompt), u.get("completion_tokens", 0)
                budget.used += u.get("total_tokens", pt + ct)
                info["prompt_tokens"] += pt
                info["completion_tokens"] += ct
                info["actual_cpt"] = chars / max(1, pt)
                ch = (d.get("choices") or [{}])[0]
                content = (ch.get("message") or {}).get("content") or ""
                parsed = _parse_json(content) if content else None
                if parsed and parsed.get("tickets"):
                    log.append({"model": model, "status": "SUCCESS"})
                    info["model"], info["status"] = model, "SUCCESS"
                    return parsed, info
                log.append({"model": model, "status": f"BAD_OUTPUT finish={ch.get('finish_reason')} (completion {ct} of cap {cap})"})
                break
            body = (r.text or "")[:240]
            low = body.lower()
            if r.status_code == 413 or "too large" in low:
                lim, req = re.search(r"Limit (\d+)", body), re.search(r"Requested (\d+)", body)
                if lim and req:
                    limit = int(lim.group(1))
                    real_prompt = max(1, int(req.group(1)) - int(cap))
                    p_est = real_prompt + 20
                    info["actual_cpt"] = chars / real_prompt
                    log.append({"model": model, "status": f"HTTP_413 limit {limit}, real prompt ~{real_prompt} tokens -> recomputing cap"})
                else:
                    p_est = int(p_est * 1.2)
                    log.append({"model": model, "status": f"HTTP_413 {body[:120]}"})
                continue
            if r.status_code == 429:
                wait = _retry_after(body)
                if "per minute" in low and wait is not None and wait <= 30 and attempt < 2:
                    log.append({"model": model, "status": f"RATE_LIMIT_429 per-minute, waiting {wait:.0f}s"})
                    time.sleep(wait + 0.5)
                    continue
                log.append({"model": model, "status": f"RATE_LIMIT_429 {body[:140]}"})
                break
            if r.status_code in NON_RETRYABLE:
                log.append({"model": model, "status": f"HTTP_{r.status_code} {body[:140]}"})
                break
            log.append({"model": model, "status": f"HTTP_{r.status_code} {body[:100]} (retrying)"})
            time.sleep(2.5 * (attempt + 1))
    info["status"] = "PROMPT_TOO_LARGE" if too_large and not info["model"] else "FAILED"
    return None, info


# ═════════════════════════════════════════════════════════════════════════════
# Ticket validation / repair / deterministic optimiser
# ═════════════════════════════════════════════════════════════════════════════
def _tprod(legs, k="odds"):
    return float(np.prod([l[k] for l in legs])) if legs else 0.0


def optimize_tickets(match_legs, n_tickets=N_TICKETS, per=LEGS_PER_TICKET, min_odds=MIN_TICKET_ODDS, iters=6000, seed=11):
    """Deterministic fallback: maximise joint probability subject to ticket odds >= min_odds."""
    mids = [m for m, l in match_legs.items() if l]
    mids.sort(key=lambda m: -match_legs[m][0]["rank"])
    if len(mids) < per:
        return []
    need = n_tickets * per
    strict = len(mids) >= need
    pool = mids[: need + 6] if strict else mids
    rnd = random.Random(seed)
    tickets = []
    for t in range(n_tickets):
        if strict:
            tickets.append([[pool[i], 0] for i in range(t, need, n_tickets)][:per])
        else:
            tickets.append([[pool[(t * per + j) % len(pool)], 0] for j in range(per)])
    used = {e[0] for t in tickets for e in t}
    def tscore(t):
        legs = [match_legs[m][i] for m, i in t]
        lp = sum(math.log(l["p_adj"]) for l in legs)
        lo = sum(math.log(l["odds"]) for l in legs)
        fam = defaultdict(int)
        for l in legs:
            fam[l["fam"]] += 1
        div = -0.06 * sum(max(0, c - 2) for c in fam.values())
        return lp - 40 * max(0.0, math.log(min_odds) - lo) + div
    def total():
        return sum(tscore(t) for t in tickets)
    best = total()
    for _ in range(iters):
        mv = rnd.random()
        a = rnd.randrange(len(tickets))
        ia = rnd.randrange(len(tickets[a]))
        old = [list(map(list, t)) for t in tickets]
        if mv < 0.4 and len(tickets) > 1:
            b = rnd.randrange(len(tickets))
            ib = rnd.randrange(len(tickets[b]))
            tickets[a][ia], tickets[b][ib] = tickets[b][ib], tickets[a][ia]
            if len({e[0] for e in tickets[a]}) < len(tickets[a]) or len({e[0] for e in tickets[b]}) < len(tickets[b]):
                tickets = old
                continue
        elif mv < 0.75:
            m_ = tickets[a][ia][0]
            tickets[a][ia][1] = rnd.randrange(len(match_legs[m_]))
        else:
            free = [m for m in pool if m not in used] if strict else [m for m in pool if m not in {e[0] for e in tickets[a]}]
            if not free:
                continue
            new = rnd.choice(free)
            tickets[a][ia] = [new, 0]
        sc = total()
        if sc >= best:
            best = sc
            used = {e[0] for t in tickets for e in t}
        else:
            tickets = old
    return [[dict(match_legs[m][i], mid=m) for m, i in t] for t in tickets]


def enforce_min_odds(legs, match_legs, used, min_odds=MIN_TICKET_ODDS):
    fixed = False
    for _ in range(12):
        if _tprod(legs) >= min_odds:
            break
        cur = {l["mid"] for l in legs}
        best, best_s = None, 0
        for i, l in enumerate(legs):
            opts = [dict(x, mid=l["mid"]) for x in match_legs.get(l["mid"], []) if x["odds"] > l["odds"]]
            for mid, lst in match_legs.items():
                if mid not in used and mid not in cur:
                    opts += [dict(x, mid=mid) for x in lst if x["odds"] > l["odds"]]
            for o in opts:
                gain = math.log(o["odds"] / l["odds"])
                loss = max(1e-4, math.log(l["p_adj"]) - math.log(o["p_adj"])) + 0.02
                s = gain / loss
                if s > best_s:
                    best, best_s = (i, o), s
        if not best:
            break
        legs[best[0]] = dict(best[1], why=legs[best[0]].get("why", ""), swapped=True)
        fixed = True
    return legs, fixed


def finalize_tickets(ai, leg_index, match_legs, n_tickets=N_TICKETS, per=LEGS_PER_TICKET, distinct=True):
    raw = (ai or {}).get("tickets") or []
    fallback = None
    used, tickets = set(), []
    for ti in range(n_tickets):
        t = raw[ti] if ti < len(raw) and isinstance(raw[ti], dict) else {}
        legs, repaired = [], False
        for it in t.get("legs") or []:
            lid = (it.get("id") if isinstance(it, dict) else str(it)) or ""
            lg = leg_index.get(lid.strip().lower())
            if not lg or lg["mid"] in used or lg["mid"] in {l["mid"] for l in legs}:
                repaired = True
                continue
            legs.append(dict(lg, why=(it.get("why", "") if isinstance(it, dict) else "")))
        legs = legs[:per]
        if len(legs) < per:
            repaired = True
            if fallback is None:
                fallback = optimize_tickets(match_legs, n_tickets=n_tickets)
            pool_ = [l for tk in fallback for l in tk]
            pool_ += [dict(x, mid=k) for k, v in match_legs.items() for x in v[:3]]
            while len(legs) < per:
                cur_m = {x["mid"] for x in legs}
                avail = [c for c in pool_ if c["mid"] not in used and c["mid"] not in cur_m]
                if not avail:
                    break
                cur, need = max(_tprod(legs), 1.0), per - len(legs)
                target = (MIN_TICKET_ODDS * 1.03 / cur) ** (1 / need) if cur < MIN_TICKET_ODDS * 1.03 else 1.15
                c = max(avail, key=lambda x: x["rank"] - 0.4 * max(0.0, math.log(x["odds"] / (target * 1.15))))
                legs.append(dict(c, why="Python fill-in (AI leg invalid/duplicate)"))
        legs, up = enforce_min_odds(legs, match_legs, used)
        repaired = repaired or up
        if distinct:
            used |= {l["mid"] for l in legs}
        tickets.append({"name": t.get("name") or TICKET_LABELS[ti], "legs": legs, "logic": t.get("logic", ""),
                        "risk": t.get("risk", ""), "repaired": repaired, "odds": _tprod(legs),
                        "p_joint": _tprod(legs, "p"), "valid": len(legs) == per and _tprod(legs) >= MIN_TICKET_ODDS})
    return tickets


def python_only_tickets(match_legs, n_tickets=N_TICKETS):
    tks = optimize_tickets(match_legs, n_tickets=n_tickets)
    out = []
    for i, legs in enumerate(tks):
        for l in legs:
            l["why"] = f"p{_pct(l['p'])}% odds {l['odds']:.2f}"
        out.append({"name": TICKET_LABELS[i], "legs": legs, "risk": "", "repaired": False, "odds": _tprod(legs),
                    "p_joint": _tprod(legs, "p"), "valid": _tprod(legs) >= MIN_TICKET_ODDS,
                    "logic": "Deterministic Python optimiser (AI unavailable): maximises joint probability with trap/data-quality penalties and odds >= 3.5."})
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Telegram
# ═════════════════════════════════════════════════════════════════════════════
def send_telegram(text):
    tok, chat = get_secret("TELEGRAM_BOT_TOKEN"), get_secret("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return False, "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing"
    chunks, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) + 2 > 3800:
            chunks.append(cur)
            cur = ""
        cur += (("\n\n" if cur else "") + para)
    chunks.append(cur)
    for c in chunks:
        try:
            r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                              json={"chat_id": chat, "text": c, "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=15)
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:
            return False, str(e)
        time.sleep(0.4)
    return True, "sent"


def build_telegram_message(res, tz):
    nums = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    L = [f"⚽ <b>Der-AI Football Quant Desk</b>\n📅 {_esc(res['date'])} | {res['n_analysed']} matches analysed | {res['n_ai']} sent to AI"]
    if res.get("est_mode"):
        L.append("⚠️ <b>Approximate odds:</b> bookmaker prices were unavailable for some legs, so odds are model-estimated. Bookmakers differ slightly - confirm the price on yours.")
    for t in res["tickets"]:
        ok = "✅" if t["valid"] else "⚠️"
        s = f"🎟 <b>{_esc(t['name'])}</b> {ok}\nOdds <b>{t['odds']:.2f}</b> | joint prob ~{_pct(t['p_joint'])}% | risk {_esc(t.get('risk') or '-')} | evidence {t.get('grade', '?')}"
        for i, l in enumerate(t["legs"]):
            m = l["match"]
            ko = datetime.fromtimestamp(m["ts"], ZoneInfo(tz)).strftime("%H:%M")
            s += (f"\n{nums[i]} <i>{_esc(_short(m['league'], 22))}</i> {ko}\n   {_esc(m['home'])} v {_esc(m['away'])}\n"
                  f"   ➜ <b>{_esc(l['label'])}</b> @{l['odds']:.2f}{' (books ' + format(l['odds_lo'], '.2f') + '-' + format(l['odds_hi'], '.2f') + ')' if l.get('odds_lo') else ''} (p {_pct(l['p'])}%{' est.odds' if l.get('src') == 'est' else ''})")
            if l.get("why"):
                s += f"\n   💡 {_esc(l['why'])}"
        if t.get("logic"):
            s += f"\n\n🧠 {_esc(t['logic'])}"
        if t.get("repaired"):
            s += "\n🔧 Python adjusted this ticket (invalid/duplicate leg or odds<3.5)."
        L.append(s)
    ai = res.get("ai") or {}
    if ai.get("traps"):
        L.append("🪤 <b>Trap watch</b>\n" + "\n".join(f"• {_esc(res['id_map'][x['id']]['home'])} v {_esc(res['id_map'][x['id']]['away'])}: {_esc(x.get('note'))}"
                                                    for x in ai["traps"][:4] if x.get("id") in res["id_map"]))
    if ai.get("avoid"):
        L.append("🚫 <b>Avoid</b>\n" + "\n".join(f"• {_esc(res['id_map'][x['id']]['home'])} v {_esc(res['id_map'][x['id']]['away'])}: {_esc(x.get('why'))}"
                                                 for x in ai["avoid"][:4] if x.get("id") in res["id_map"]))
    if ai.get("summary"):
        L.append(f"📝 <b>Slate read</b>\n{_esc(ai['summary'])}")
    if not res.get("model"):
        L.append("ℹ️ <b>AI reasoning was not applied</b> - these tickets come from the Python Monte Carlo optimiser only."
                 + (f" ({_esc(res.get('ai_note'))[:180]})" if res.get("ai_note") else ""))
    foot = (f"🤖 {_esc(res.get('model') or 'Python optimiser')} | tokens {res.get('tokens', 0)}/{TOTAL_TOKEN_BUDGET} | "
            f"API calls {res.get('api_calls', 0)} | ⚠️ Odds shown are bookmaker snapshots; betting carries risk.")
    L.append(foot)
    return "\n\n".join(L)


def build_nobet_message(res, tz):
    reasons = defaultdict(int)
    for e in res.get("excluded", []):
        reasons[e["reason"]] += 1
    lines = "\n".join(f"• {_esc(k)}: {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:6])
    return (f"⚽ <b>Der-AI Football Quant Desk</b>\n📅 {_esc(res['date'])}\n\n🛑 <b>NO BET TODAY</b>\n{_esc(res.get('reason'))}"
            + (f"\n\n<b>Why matches were excluded</b>\n{lines}" if lines else "")
            + f"\n\nAPI calls used: {res.get('api_calls', 0)}. No tickets are better than tickets built on guesses.")


# ═════════════════════════════════════════════════════════════════════════════
# Ticket ledger (settlement + calibration)
# ═════════════════════════════════════════════════════════════════════════════
def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"history": [], "cpt": DEFAULT_CHARS_PER_TOKEN}


def save_state(s):
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as fh:
            json.dump(s, fh)
    except Exception:
        pass


def evaluate_leg(key, f):
    h, a = _ft_goals(f)
    if h is None or a is None:
        return None
    tot = h + a
    if key == "1": return h > a
    if key == "X": return h == a
    if key == "2": return h < a
    if key == "1X": return h >= a
    if key == "X2": return h <= a
    if key == "12": return h != a
    if key == "BTTS_Y": return h > 0 and a > 0
    if key == "BTTS_N": return not (h > 0 and a > 0)
    mm = re.match(r"(H_|A_|HT_|C_)?(O|U)([\d.]+)$", key)
    if not mm:
        return None
    pre, ou, line = mm.groups()
    line = float(line)
    if pre == "HT_":
        ht = (f.get("score") or {}).get("halftime") or {}
        if ht.get("home") is None:
            return None
        val = ht["home"] + ht["away"]
    elif pre == "C_":
        c = []
        for s in f.get("statistics") or []:
            v = _stat_val(s.get("statistics"), "Corner Kicks")
            if v is None:
                return None
            c.append(v)
        if len(c) < 2:
            return None
        val = sum(c)
    else:
        val = {None: tot, "H_": h, "A_": a}[pre]
    return val > line if ou == "O" else val < line


def settle_history(api, state):
    ids = sorted({l["fid"] for e in state["history"] if e.get("status") != "SETTLED" for t in e["tickets"] for l in t["legs"] if l.get("result") is None})
    fx = {}
    for i in range(0, len(ids), 20):
        for f in api.resp("/fixtures", {"ids": "-".join(map(str, ids[i:i + 20]))}, ttl=0):
            if ((f.get("fixture") or {}).get("status") or {}).get("short") in PLAYED_OK:
                fx[f["fixture"]["id"]] = f
    for e in state["history"]:
        alive = False
        for t in e["tickets"]:
            for l in t["legs"]:
                if l.get("result") is None and l["fid"] in fx:
                    l["result"] = evaluate_leg(l["key"], fx[l["fid"]])
            res = [l.get("result") for l in t["legs"]]
            t["outcome"] = "LOST" if any(r is False for r in res) else "WON" if all(r is True for r in res) else "PENDING"
            alive |= t["outcome"] == "PENDING"
        e["status"] = "PENDING" if alive else "SETTLED"
    return state


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator (runs on the Analyse click)
# ═════════════════════════════════════════════════════════════════════════════
def run_full_analysis(cfg, log, progress):
    t0 = time.time()
    api_key = get_secret("API_FOOTBALL_KEY") or get_secret("APISPORTS_KEY")
    if not api_key:
        return {"error": "API_FOOTBALL_KEY is not set (get one at https://dashboard.api-football.com)."}
    api = ApiFootball(api_key, per_minute=10)
    stt = api.status()
    plan = (stt.get("plan") or "Free")
    per_min = 10 if str(plan).lower() == "free" else 280
    api.rate = RateLimiter(cfg["rate_override"] or per_min)
    cfg = dict(cfg, workers=3 if per_min <= 10 else 8)
    log(f"🔑 API-Football plan: {plan} | used {stt.get('used')}/{stt.get('limit')} today")

    data = collect_day(api, cfg, log, progress)
    matches = data["matches"]

    def result(**kw):
        base = {"date": cfg["date"], "tickets": [], "ai": None, "model": None, "info": {"attempts": []}, "tokens": 0,
                "n_analysed": len(matches), "n_ai": 0, "matches": matches, "id_map": {}, "user_prompt": "",
                "api_calls": api.calls, "cache_hits": api.cache_hits, "warnings": data["warnings"], "est_mode": False,
                "cat_counts": data["cat_counts"], "plan": data["plan"], "deep": data.get("deep"),
                "api_errors": api.error_summary(), "seconds": round(time.time() - t0, 1),
                "excluded": data["excluded"], "no_bet": False, "reason": "", "caps": data.get("caps"), "quota": (api.remaining, api.limit)}
        base.update(kw)
        return base

    if not matches:
        return result(no_bet=True, reason=data.get("reason") or "No matches could be analysed.")

    # ── legs + evidence gate ──
    for m in matches:
        m["legs"] = build_legs(m, allow_est=cfg["allow_est"])
    elig = []
    for m in matches:
        why = gate_reasons(m, cfg)
        m["eligible"], m["excl"] = (not why), "; ".join(why)
        if why:
            data["excluded"].append({"league": m["league"], "match": f"{m['home']} v {m['away']}", "reason": m["excl"]})
        else:
            elig.append(m)
    log(f"🛡 Evidence gate: {len(elig)}/{len(matches)} matches qualify (real prices + team-specific data + safe legs)")
    n_t = min(N_TICKETS, len(elig) // LEGS_PER_TICKET)
    if cfg["allow_reuse"] and len(elig) >= LEGS_PER_TICKET:
        n_t = N_TICKETS
    if n_t == 0:
        tip = (f" Only ~{data.get('budget')} API calls were available this run - retry after the daily quota resets (00:00 UTC) or raise the limit."
               if (data.get("budget") or 999) < 60 else "")
        return result(no_bet=True, reason=f"Only {len(elig)} match(es) passed the evidence gate; a 5-leg ticket needs at least 5. "
                                          "Betting on thin data would just be guessing, so nothing is recommended." + tip)
    if n_t < N_TICKETS:
        data["warnings"].append(f"Only {len(elig)} matches qualified, so {n_t} ticket(s) were built instead of {N_TICKETS} (no match is reused).")

    state = load_state()
    cpt = min(max(float(state.get("cpt", DEFAULT_CHARS_PER_TOKEN)) * 0.95, 2.4), 3.6)
    sys_p = system_prompt(n_t, cfg["allow_reuse"])
    eff = cfg.get("effort") if cfg.get("effort") in EFFORT_RESERVE else "low"

    def make_pack(cpt_):
        for eff_try in [e for e in ("high", "medium", "low") if EFFORT_RESERVE[e] <= EFFORT_RESERVE[eff]]:
            reserve = EFFORT_RESERVE[eff_try]
            sel_, id_map_, user_, n_legs_ = build_ai_pack(elig, cfg["tz"], cpt_, cfg["n_ai"], sys_p, n_t * LEGS_PER_TICKET, reserve)
            if est_tokens(sys_p + user_, cpt_) <= TOTAL_TOKEN_BUDGET - reserve - TOKEN_SAFETY:
                break
        ml_, li_ = {}, {}
        for mid, m in id_map_.items():
            lst = []
            for l in m["legs"]:
                l2 = dict(l, mid=mid, fid=m["id"], match=m, id=f"{mid}.{l['key']}")
                lst.append(l2)
                li_[l2["id"].lower()] = l2
            ml_[mid] = lst
        return sel_, id_map_, user_, n_legs_, eff_try, ml_, li_

    sel, id_map, user_prompt, n_legs, eff_try, match_legs, leg_index = make_pack(cpt)
    if eff_try != eff:
        log(f"   ℹ️ '{eff}' effort needs more thinking room than the 8,000-token cap allows for this many matches - using '{eff_try}'.")
    log(f"📦 AI pack: {len(sel)} matches x {n_legs} legs | est. prompt {est_tokens(sys_p + user_prompt, cpt)} tokens (hard cap {TOTAL_TOKEN_BUDGET} incl. completion)")
    progress(0.93)

    ai, info, model_used = None, {"attempts": []}, None
    budget = TokenBudget(TOTAL_TOKEN_BUDGET)
    if cfg["use_ai"]:
        log("🤖 AI bookmaker/quant is reasoning over the evidence pack...")
        ai, info = call_groq_budgeted(sys_p, user_prompt, budget, cpt, eff_try)
        if not ai and info.get("status") == "PROMPT_TOO_LARGE" and info.get("actual_cpt"):
            new_cpt = min(max(info["actual_cpt"] * 0.97, 1.8), cpt)
            log(f"   ℹ️ The real prompt was bigger than estimated ({info['actual_cpt']:.2f} chars/token) - rebuilding a smaller pack and retrying.")
            sel, id_map, user_prompt, n_legs, eff_try, match_legs, leg_index = make_pack(new_cpt)
            prev = info["attempts"]
            ai, info = call_groq_budgeted(sys_p, user_prompt, budget, new_cpt, eff_try)
            info["attempts"] = prev + info["attempts"]
            cpt = new_cpt
        model_used = info.get("model")
        if info.get("actual_cpt"):
            state["cpt"] = round(0.7 * cpt + 0.3 * info["actual_cpt"], 3)
    if ai:
        tickets = finalize_tickets(ai, leg_index, match_legs, n_tickets=n_t, distinct=not cfg["allow_reuse"])
    else:
        tickets = python_only_tickets(match_legs, n_t)
        if cfg["use_ai"]:
            why_ = " | ".join(f"{a['model'].split('/')[-1]}: {a['status'][:110]}" for a in info.get("attempts", [])[-3:])
            data["warnings"].append(f"AI unavailable ({info.get('status')}): {why_}. Tickets were built by the deterministic Python optimiser instead.")
            res_note = why_
    for t in tickets:
        dqm = min([l["match"]["dq"] for l in t["legs"]] or [0])
        t["grade"] = "A" if dqm >= 0.75 and not t["repaired"] else "B" if dqm >= 0.6 else "C"
    progress(0.97)

    est_mode = any(l.get("src") == "est" for t in tickets for l in t["legs"])
    if est_mode:
        data["warnings"].append("Some legs use MODEL-ESTIMATED odds (fair odds x 0.93). Verify real prices before betting.")
    res = result(tickets=tickets, ai=ai, model=model_used, info=info, tokens=budget.used, n_ai=len(sel), id_map=id_map,
                 user_prompt=user_prompt, est_mode=est_mode, ai_note=(locals().get("res_note") or ""))
    state.setdefault("history", []).append({
        "created": datetime.now(timezone.utc).isoformat(), "date": cfg["date"], "status": "PENDING",
        "tickets": [{"name": t["name"], "odds": round(t["odds"], 2), "outcome": "PENDING", "grade": t["grade"],
                     "legs": [{"fid": l["fid"], "key": l["key"], "label": l["label"], "odds": round(l["odds"], 2),
                               "p": round(l["p"], 3), "match": f"{l['match']['home']} v {l['match']['away']}", "result": None}
                              for l in t["legs"]]} for t in tickets]})
    state["history"] = state["history"][-60:]
    save_state(state)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═════════════════════════════════════════════════════════════════════════════
def render_results(res, tz):
    for w in res.get("warnings", []):
        st.warning(w)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Matches analysed", res["n_analysed"])
    c2.metric("Qualified -> AI", res["n_ai"])
    c3.metric("AI tokens (cap 8000)", res["tokens"])
    c4.metric("API calls", res["api_calls"], f"cache {res['cache_hits']}")
    c5.metric("Model", (res.get("model") or "Python optimiser").split("/")[-1] if res["tickets"] else "-")
    if res.get("quota") and res["quota"][0] is not None:
        st.caption(f"📶 API quota left today: **{res['quota'][0]}** of {res['quota'][1]} (resets 00:00 UTC / 03:00 Kampala)")
    if res.get("caps"):
        cp = res["caps"]
        st.caption(f"API plan capabilities detected - team history: {'✅' if cp['history'] else '❌ (using predictions last-5 form)'} · "
                   f"odds by league: {'✅' if cp['league_odds'] else '❌' if cp['league_odds'] is False else '?'} · odds by fixture: {'✅' if cp['fixture_odds'] else '❌'}")
    if res.get("no_bet"):
        st.error(f"🛑 **NO BET** - {res.get('reason')}")
        if res.get("api_errors"):
            st.markdown("**API messages (send these if you need help):**")
            st.code("\n".join(res["api_errors"]), language="text")
    for t in res["tickets"]:
        st.markdown(f"### 🎟 {t['name']} — odds **{t['odds']:.2f}** {'✅' if t['valid'] else '⚠️ below 3.5'} · joint prob ~{_pct(t['p_joint'])}% · evidence grade **{t.get('grade', '?')}**")
        rows = []
        for l in t["legs"]:
            m = l["match"]
            rows.append({"Kick-off": datetime.fromtimestamp(m["ts"], ZoneInfo(tz)).strftime("%H:%M"), "League": m["league"],
                         "Match": f"{m['home']} v {m['away']}", "Pick": l["label"], "Odds": round(l["odds"], 2),
                         "Books range": (f"{l['odds_lo']:.2f}-{l['odds_hi']:.2f}" if l.get("odds_lo") else "est."),
                         "Model+Mkt %": _pct(l["p"]), "Market fair %": _pct(l["p_mkt"]), "Trap": m["trap"]["risk"],
                         "Data": BASIS_TAG.get(m["basis"]), "Why": l.get("why", "")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if t.get("logic"):
            st.info(t["logic"])
        if t.get("repaired"):
            st.caption("🔧 Python corrected this ticket (invalid/duplicate leg or odds below 3.5).")
    ai = res.get("ai") or {}
    for title, key, fld in (("🪤 Trap watch", "traps", "note"), ("🚫 Avoid", "avoid", "why")):
        if ai.get(key):
            st.markdown(f"#### {title}")
            for x in ai[key]:
                m = res["id_map"].get(x.get("id"))
                if m:
                    st.write(f"**{m['home']} v {m['away']}** — {x.get(fld)}")
    if ai.get("summary"):
        st.markdown(f"#### 📝 Slate read\n{ai['summary']}")
    if res.get("excluded"):
        with st.expander(f"🛡 Excluded by the evidence gate ({len(res['excluded'])})"):
            st.dataframe(pd.DataFrame(res["excluded"]), hide_index=True, width="stretch")
    with st.expander("📋 All analysed matches (Monte Carlo + market blend)"):
        rows = []
        for m in res["matches"]:
            p = m["p"]
            rows.append({"Cat": m["cat"], "League": m["league"], "Match": f"{m['home']} v {m['away']}",
                         "KO": datetime.fromtimestamp(m["ts"], ZoneInfo(tz)).strftime("%H:%M"),
                         "Data": BASIS_TAG.get(m["basis"]), "xG": f"{m['lam_h']:.2f}-{m['lam_a']:.2f}",
                         "1": _pct(p["1"]), "X": _pct(p["X"]), "2": _pct(p["2"]), "1X": _pct(p["1X"]), "X2": _pct(p["X2"]),
                         "BTTS": _pct(p["BTTS_Y"]), "O1.5": _pct(p["O1.5"]), "O2.5": _pct(p["O2.5"]),
                         "Corners": round(m["exp_corners"], 1) if m["corner_n"] >= 3 else None,
                         "Trap": m["trap"]["risk"], "DQ": m["dq"], "Qualified": "yes" if m.get("eligible") else "no",
                         "Likely": ",".join(m["top_scores"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if res.get("user_prompt"):
        with st.expander("🔎 Exact evidence pack sent to the AI"):
            st.code(res["user_prompt"], language="text")
    with st.expander("🛠 Model attempts / API errors"):
        for a in (res.get("info") or {}).get("attempts", []):
            st.caption(f"{a['model']} → {a['status']}")
        for e in res.get("api_errors", []):
            st.caption(f"API: {e}")


def main():
    st.set_page_config(page_title="Der-AI | Football Quant Desk", page_icon="⚽", layout="wide")
    st.markdown("""<style>
      .stButton>button{background:linear-gradient(90deg,#065f46 0%,#10b981 100%);color:white;border:none;padding:10px 24px;border-radius:8px;font-weight:bold;font-size:16px;width:100%}
      .stButton>button:hover{background:linear-gradient(90deg,#047857 0%,#059669 100%)}
    </style>""", unsafe_allow_html=True)
    for k, v in (("notifications", []), ("last_result", None)):
        st.session_state.setdefault(k, v)

    tab1, tab2, tab3, tab4 = st.tabs(["⚽ Match Analysis", "📜 Ticket Ledger", "🔔 Notifications", "⚙️ Settings"])

    with tab4:
        st.header("⚙️ Settings")
        st.info("Secrets needed: `API_FOOTBALL_KEY`, `GROQ_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.")
        s1, s2 = st.columns(2)
        tz = s1.text_input("Time zone", DEFAULT_TZ)
        max_matches = s1.slider("Max matches to collect deep data for", 10, 60, 30)
        n_ai = s1.slider("Max matches sent to AI (auto-trimmed to fit 8,000 tokens)", 15, 24, 18)
        max_calls = s1.number_input("Max API-Football calls per run", 30, 5000, 300, step=10,
                                    help="Free plan = 100/day. Pro = 7,500/day. The app auto-shrinks the shortlist to fit.")
        depth = s1.selectbox("Data depth", ["Auto", "Full", "Lite"], help="Full adds corners/shots/xG history, player ratings, lineups and standings via batched calls.")
        sims = s2.select_slider("Monte Carlo simulations per match", [10000, 20000, 40000, 80000], 40000)
        w_model = s2.slider("Weight of Monte Carlo model vs bookmaker price", 0.2, 0.8, 0.5, 0.05)
        bookmaker = s2.number_input("Preferred bookmaker id (8 = Bet365, 11 = 1xBet, 4 = Pinnacle)", 1, 500, 8)
        effort = s2.selectbox("AI reasoning effort (gpt-oss)", ["low", "medium", "high"], help="Higher effort thinks longer, so the evidence pack is automatically shrunk (fewer matches/legs) to keep prompt + completion within 8,000 tokens. If even that cannot fit, the app steps effort down.")
        exclude_minor = s2.checkbox("Exclude youth / women / reserve teams", True)
        exclude_lower = s2.checkbox("Exclude amateur / regional lower divisions & club friendlies", True)
        force_leagues = {int(x) for x in re.findall(r"\d+", s2.text_input("Always include league ids (comma separated)", ""))}
        rate_override = int(s2.number_input("Requests/minute override (0 = auto by plan)", 0, 900, 0))
        send_tg = st.checkbox("Send tickets to Telegram automatically", True)
        use_ai = st.checkbox("Use AI (Groq)", True)
        min_dq = st.slider("Minimum data-quality score for a match to qualify", 0.3, 0.9, 0.5, 0.05,
                           help="Matches below this (or with no team-specific data / no real odds) are excluded from tickets.")
        allow_reuse = st.checkbox("Allow the same match in more than one ticket (only if too few matches qualify)", False)
        allow_est = st.checkbox("If the API has no odds for a match, use model-estimated approximate odds (clearly labelled)", True,
                                help="Real bookmaker prices are always preferred (median of all bookmakers). Estimates only fill in when a match has none; trap and market-agreement checks are weaker for those matches.")
        st.markdown(f"**AI order:** {' → '.join(GROQ_MODELS)} · **Hard cap:** {TOTAL_TOKEN_BUDGET} tokens per analysis (prompt + completion, all attempts)")
        if st.button("🧹 Clear API cache"):
            try:
                for fn in os.listdir(CACHE_DIR):
                    os.remove(os.path.join(CACHE_DIR, fn))
                st.success("Cache cleared.")
            except Exception as e:
                st.error(str(e))

    with tab1:
        st.header("🚀 Football Quant Desk")
        local_today = datetime.now(ZoneInfo(tz)).date()
        d = st.date_input("Match day to analyse", local_today)
        force = st.checkbox("Ignore cache (fresh API data - costs more calls)", False)
        if st.button("🧠 Analyse Football Matches Now", type="primary"):
            cfg = {"date": d.strftime("%Y-%m-%d"), "tz": tz, "max_matches": max_matches, "n_ai": n_ai, "max_calls": int(max_calls),
                   "depth": depth, "sims": int(sims), "w_model": w_model, "bookmaker": int(bookmaker), "effort": effort,
                   "exclude_minor": exclude_minor, "exclude_lower": exclude_lower, "force_leagues": force_leagues, "rate_override": rate_override,
                   "force": force, "use_ai": use_ai, "min_dq": min_dq, "allow_reuse": allow_reuse, "allow_est": allow_est}
            bar = st.progress(0.0)
            box = st.status("Running full analysis...", expanded=True)
            def log(m): box.write(m)
            def prog(x): bar.progress(min(1.0, float(x)))
            try:
                res = run_full_analysis(cfg, log, prog)
            except Exception:
                res = {"error": "Unexpected error:\n" + traceback.format_exc()}
            bar.progress(1.0)
            if "error" in res:
                box.update(label="Analysis stopped", state="error")
                st.error(res["error"])
                for w in res.get("warnings", []) + res.get("api_errors", []):
                    st.caption(w)
            else:
                box.update(label=f"Done in {res['seconds']}s" + (" - NO BET" if res.get("no_bet") else ""), state="complete")
                st.session_state.last_result = res
                st.session_state.last_tz = tz
                if send_tg:
                    msg = build_nobet_message(res, tz) if res.get("no_bet") else build_telegram_message(res, tz)
                    ok, why = send_telegram(msg)
                    (st.success if ok else st.warning)("✅ Sent to Telegram" if ok else f"Telegram failed: {why}")
                    st.session_state.notifications.append({"time": datetime.now().strftime("%H:%M"), "ok": ok,
                        "msg": f"{cfg['date']}: " + ("NO-BET notice " if res.get("no_bet") else f"{len(res['tickets'])} ticket(s) ") + ("sent" if ok else f"not sent ({why})")})
        if st.session_state.last_result:
            render_results(st.session_state.last_result, st.session_state.get("last_tz", tz))

    with tab2:
        st.header("📜 Ticket Ledger & Calibration")
        state = load_state()
        if st.button("🔄 Settle finished matches"):
            key = get_secret("API_FOOTBALL_KEY")
            if key:
                save_state(settle_history(ApiFootball(key, 10), state))
                st.success("Ledger updated.")
                state = load_state()
        tk = [(e["date"], t) for e in state.get("history", []) for t in e["tickets"]]
        won, lost = sum(1 for _, t in tk if t.get("outcome") == "WON"), sum(1 for _, t in tk if t.get("outcome") == "LOST")
        legs_ = [l for _, t in tk for l in t["legs"] if l.get("result") is not None]
        a, b, c = st.columns(3)
        a.metric("Tickets won / lost", f"{won} / {lost}")
        b.metric("Ticket hit rate", f"{100 * won / (won + lost):.0f}%" if won + lost else "—")
        c.metric("Leg hit rate", f"{100 * sum(1 for l in legs_ if l['result']) / len(legs_):.0f}%" if legs_ else "—")
        if tk:
            st.dataframe(pd.DataFrame([{"Date": dt, "Ticket": t["name"], "Odds": t["odds"], "Outcome": t.get("outcome"),
                                        "Legs": " | ".join(f"{l['match']}: {l['label']}{'✅' if l.get('result') else '❌' if l.get('result') is False else '⏳'}" for l in t["legs"])}
                                       for dt, t in reversed(tk)]), hide_index=True, width="stretch")

    with tab3:
        st.header("🔔 Notifications")
        for n in reversed(st.session_state.notifications):
            (st.success if n["ok"] else st.warning)(f"[{n['time']}] {n['msg']}")
        if not st.session_state.notifications:
            st.info("No notifications yet.")


if __name__ == "__main__":
    main()
