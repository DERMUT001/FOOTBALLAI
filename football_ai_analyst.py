# =============================================================================
# Der-AI | Football Quant Desk — V2  (AI-first, quota-safe, free-source backed)
#
# One click on "Analyse" runs the whole pipeline:
#   1. PYTHON collects the day's fixtures from API-Football and spends the daily quota carefully
#      (disk cache, call-time budget guard, no probing when the answer is already known).
#   2. FREE SOURCES (zero API-Football calls) are used FIRST for what they cover and as a FALLBACK
#      for whatever API-Football cannot provide:
#        - football-data.co.uk : results, corners, shots, bookmaker 1X2 + O/U2.5 prices, H2H, table
#        - ClubElo            : team strength (independent 1X2 opinion for European clubs)
#        - ESPN public JSON   : fixtures + bookmaker prices (best effort)
#      If API-Football is unreachable / out of quota the app still analyses from free sources only.
#   3. PYTHON runs a Monte Carlo per match and blends it with de-vigged bookmaker prices.
#   4. PYTHON builds a WIDE leg menu (safe legs + clearly-labelled borderline legs) - the AI decides.
#   5. AI (Groq) works in short focused calls (scout, ONE desk call per ticket, audit), paced by a tokens-per-minute governor:
#        SCOUT  - reads a compact view of ALL qualified matches and picks the ones worth studying
#        DESK   - full evidence on the finalists, builds 3 tickets x 5 legs with a failure scenario per leg
#        AUDIT  - a risk officer tries to break every leg, swaps weak legs and fills any gap
#   6. PYTHON only verifies (real ids, distinct matches, odds >= 3.5, rule lint). Anything Python had
#      to change is labelled. If the AI is down, a labelled Python-only fallback still delivers tickets.
# =============================================================================

import os, re, json, math, time, html, hashlib, threading, random, traceback, csv, io, unicodedata, difflib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone, date as date_cls
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
CACHE_PURGE_DAYS = 8          # finished-match data never changes, so it may live long (TTLs still decide freshness)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT = 90
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]   # a model that answers 404 is remembered as unavailable and skipped
GROQ_MODEL_CONFIG = {
    "openai/gpt-oss-120b": {"max_completion_tokens": 4000, "reasoning_effort": "medium", "supports_reasoning_effort": True},
    "openai/gpt-oss-20b": {"max_completion_tokens": 4000, "reasoning_effort": "medium", "supports_reasoning_effort": True},
    "llama-3.3-70b-versatile": {"max_completion_tokens": 2500, "reasoning_effort": None, "supports_reasoning_effort": False},
}
GROQ_DEFAULT_CFG = {"max_completion_tokens": 2000, "reasoning_effort": None, "supports_reasoning_effort": False}
NON_RETRYABLE = {400, 401, 403, 404, 422}

# ── Token policy ────────────────────────────────────────────────────────────
# Groq's free tier allows ~9,000 tokens PER MINUTE. Every single request must fit in that, and several
# requests are spread over time by the TpmGovernor (it sleeps until the rolling 60-second window has room).
GROQ_TPM_DEFAULT = 8000      # Groq reported "Limit 8000" for your account: every single request (prompt + reply room) must fit in it
TPM_MARGIN = 350              # per-request ceiling = TPM limit - margin
WINDOW_MARGIN = 150           # rolling-window budget = TPM limit - margin (must stay > per-request ceiling)
RUN_TOKEN_CAP_DEFAULT = 45000  # all passes of one analysis together (protects the daily token allowance)
EFFORT_RESERVE = {"low": 2600, "medium": 3800, "high": 5000}   # completion room (incl. hidden reasoning) per effort
SCOUT_RESERVE, AUDIT_RESERVE = 1500, 2600
OUTPUT_RESERVE = 3000
TOKEN_SAFETY = 100
MIN_COMPLETION = 900
DEFAULT_CHARS_PER_TOKEN = 2.3  # measured: dense numeric evidence tokenizes at ~2.2 chars/token (the scout pass re-measures it every run)

# ── Ticket rules ────────────────────────────────────────────────────────────
N_TICKETS, LEGS_PER_TICKET, MIN_TICKET_ODDS = 3, 5, 3.5
MIN_LEG_P, MIN_LEG_ODDS, MAX_LEG_ODDS = 0.62, 1.15, 2.00
MIN_AGREE_P = 0.60   # SAFE legs: BOTH the raw model and the bookmaker's fair price must clear this
BORDER_AGREE_P, BORDER_MIN_P = 0.55, 0.57   # BORDERLINE legs (shown to the AI with a '~' flag, never silently dropped)
MAX_MENU_LEGS = 7
BASIS_TAG = {"season+recent": "S+R", "recent": "R", "season": "S", "elo": "E", "prior": "P"}
TICKET_LABELS = ["Ticket 1 - Safest", "Ticket 2 - Balanced", "Ticket 3 - Value"]

# ── League intelligence (API-Football league ids; unknown leagues fall into LOW) ──
TOP_LEAGUES = {39: "Premier League", 140: "La Liga", 135: "Serie A", 78: "Bundesliga", 61: "Ligue 1",
               2: "Champions League", 3: "Europa League", 848: "Conference League", 1: "World Cup",
               4: "Euro", 9: "Copa America", 5: "Nations League"}
MID_LEAGUES = {88, 94, 40, 144, 203, 179, 71, 128, 262, 253, 307, 13, 11, 41, 42, 79, 136, 141, 62, 89, 95,
               145, 204, 180, 72, 218, 207, 119, 103, 113, 106, 197, 235, 98, 292, 188, 45, 48, 81, 66, 143, 137}
CAT_WEIGHT = {"TOP": 100, "INTL": 80, "MID": 70, "LOW": 50}
CAT_QUOTA = {"TOP": 0.40, "INTL": 0.20, "MID": 0.30, "LOW": 0.10}
BREADTH_QUOTA = {"Focused": {"TOP": 0.50, "INTL": 0.20, "MID": 0.25, "LOW": 0.05}, "Balanced": CAT_QUOTA,
                 "Wide": {"TOP": 0.25, "INTL": 0.20, "MID": 0.30, "LOW": 0.25}}
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
        self.session.headers.update({"x-apisports-key": key or "none"})
        self.offline = not key
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
                if time.time() - os.path.getmtime(p) > CACHE_PURGE_DAYS * 86400:
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
        if self.offline:
            return {"plan": None, "errors": "no API-Football key configured"}
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
        if self.quota_out or self.offline:
            return {"response": [], "paging": {}, "errors": {"quota": "daily quota exhausted or no API key"}}
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
# FREE data sources - cost ZERO API-Football calls
#   football-data.co.uk : results + corners + shots + bookmaker prices (main leagues + 16 extra leagues)
#   ClubElo (clubelo.com): club strength ratings (European clubs)
#   ESPN public JSON    : fixtures + bookmaker prices (best effort, schema may change)
# Every method fails SOFT (returns None / [] and records the reason) so a dead source never stops an analysis.
# ═════════════════════════════════════════════════════════════════════════════
FD_BASE = "https://www.football-data.co.uk"
FD_MAIN = {39: "E0", 40: "E1", 41: "E2", 42: "E3", 140: "SP1", 141: "SP2", 135: "I1", 136: "I2", 78: "D1", 79: "D2",
           61: "F1", 62: "F2", 88: "N1", 94: "P1", 144: "B1", 203: "T1", 197: "G1", 179: "SC0", 180: "SC1"}
FD_NEW = {128: "ARG", 71: "BRA", 262: "MEX", 253: "USA", 98: "JPN", 169: "CHN", 357: "IRL", 103: "NOR", 113: "SWE",
          244: "FIN", 119: "DNK", 106: "POL", 283: "ROU", 235: "RUS", 207: "SWZ", 218: "AUT"}
FD_CODE_TO_LID = {v: k for k, v in FD_MAIN.items()}
FD_ALL = set(FD_MAIN) | set(FD_NEW)
ESPN_SLUG = {39: "eng.1", 40: "eng.2", 41: "eng.3", 42: "eng.4", 140: "esp.1", 141: "esp.2", 135: "ita.1", 136: "ita.2",
             78: "ger.1", 79: "ger.2", 61: "fra.1", 62: "fra.2", 88: "ned.1", 94: "por.1", 144: "bel.1", 203: "tur.1",
             179: "sco.1", 2: "uefa.champions", 3: "uefa.europa", 848: "uefa.europa.conf", 253: "usa.1", 262: "mex.1",
             71: "bra.1", 5: "uefa.nations", 10: "fifa.friendly", 128: "arg.1", 98: "jpn.1", 218: "aut.1", 207: "sui.1", 119: "den.1", 103: "nor.1",
             113: "swe.1", 197: "gre.1", 235: "rus.1", 169: "chn.1"}
ESPN_SLUG_TO_LID = {v: k for k, v in ESPN_SLUG.items()}
LEAGUE_INFO = {5: ("UEFA Nations League", "World"), 10: ("Friendlies", "World"), 39: ("Premier League", "England"), 40: ("Championship", "England"), 41: ("League One", "England"),
               42: ("League Two", "England"), 140: ("La Liga", "Spain"), 141: ("Segunda Division", "Spain"),
               135: ("Serie A", "Italy"), 136: ("Serie B", "Italy"), 78: ("Bundesliga", "Germany"),
               79: ("2. Bundesliga", "Germany"), 61: ("Ligue 1", "France"), 62: ("Ligue 2", "France"),
               88: ("Eredivisie", "Netherlands"), 94: ("Primeira Liga", "Portugal"), 144: ("Pro League", "Belgium"),
               203: ("Super Lig", "Turkey"), 197: ("Super League 1", "Greece"), 179: ("Premiership", "Scotland"),
               180: ("Championship", "Scotland"), 2: ("Champions League", "World"), 3: ("Europa League", "World"),
               848: ("Conference League", "World"), 253: ("MLS", "USA"), 262: ("Liga MX", "Mexico"),
               71: ("Serie A", "Brazil"), 128: ("Liga Profesional", "Argentina"), 98: ("J1 League", "Japan"),
               218: ("Bundesliga", "Austria"), 207: ("Super League", "Switzerland"), 119: ("Superliga", "Denmark"),
               103: ("Eliteserien", "Norway"), 113: ("Allsvenskan", "Sweden"), 197: ("Super League 1", "Greece"),
               235: ("Premier League", "Russia"), 169: ("Super League", "China"), 357: ("Premier Division", "Ireland"),
               244: ("Veikkausliiga", "Finland"), 106: ("Ekstraklasa", "Poland"), 283: ("Liga I", "Romania")}
ELO_COUNTRIES = {"england", "spain", "italy", "germany", "france", "netherlands", "portugal", "belgium", "turkey", "scotland",
                 "greece", "switzerland", "austria", "denmark", "norway", "sweden", "poland", "czech-republic", "croatia",
                 "serbia", "ukraine", "russia", "romania", "hungary", "cyprus", "israel", "slovakia", "slovenia", "bulgaria",
                 "wales", "ireland", "northern-ireland", "finland", "iceland", "bosnia", "belarus", "azerbaijan", "kazakhstan"}
ELO_LEAGUE_IDS = {2, 3, 848}
INTL_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
FREE_TTL_S = 6 * 3600
FREE_UA = {"User-Agent": "Mozilla/5.0 (compatible; DerAI-FootballDesk/2.0)"}

_STOP = {"fc", "cf", "afc", "sc", "ac", "as", "ss", "ssc", "fk", "sk", "bk", "if", "fsv", "vfb", "vfl", "sv", "cd", "ud", "sd",
         "ca", "club", "de", "the", "calcio", "and"}
_TOKEN_FIX = {"st": "saint", "utd": "united"}
_ALIAS = {"man united": "manchester united", "man utd": "manchester united", "man city": "manchester city",
          "nottm forest": "nottingham forest", "wolves": "wolverhampton wanderers", "spurs": "tottenham",
          "west brom": "west bromwich albion", "ath madrid": "atletico madrid", "ath bilbao": "athletic",
          "athletic bilbao": "athletic", "athletic club": "athletic", "sociedad": "real sociedad", "betis": "real betis",
          "vallecano": "rayo vallecano", "espanol": "espanyol", "celta": "celta vigo", "sp gijon": "sporting gijon",
          "inter": "internazionale", "inter milan": "internazionale", "mgladbach": "monchengladbach",
          "m gladbach": "monchengladbach", "ein frankfurt": "eintracht frankfurt", "bayern munich": "bayern munchen",
          "paris sg": "paris saint germain", "psg": "paris saint germain", "sp lisbon": "sporting cp",
          "sporting lisbon": "sporting cp", "sp braga": "braga", "olympique marseille": "marseille",
          "olympique lyonnais": "lyon", "nijmegen": "nec nijmegen",
          # national teams (API-Football naming -> international_results naming)
          "rep of ireland": "republic of ireland", "ireland": "republic of ireland", "czechia": "czech republic",
          "turkiye": "turkey", "usa": "united states", "united states of america": "united states",
          "korea republic": "south korea", "republic of korea": "south korea", "korea dpr": "north korea",
          "china pr": "china", "ir iran": "iran", "cote divoire": "ivory coast", "congo dr": "dr congo",
          "cape verde islands": "cape verde", "cabo verde": "cape verde", "us virgin islands": "united states virgin islands",
          "swaziland": "eswatini", "burma": "myanmar", "timor leste": "east timor", "macedonia": "north macedonia",
          "chinese taipei": "taiwan", "brunei darussalam": "brunei", "bosnia": "bosnia herzegovina",
          "saint kitts nevis": "saint kitts and nevis"}


def norm_team(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ").replace("'", "").replace(".", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = []
    for t in s.split():
        t = _TOKEN_FIX.get(t, t)
        if t in _STOP or t.isdigit():
            continue
        toks.append(t)
    out = " ".join(toks)
    return _ALIAS.get(out, out)


def _covers(a, b):
    """Every token of `a` appears in `b` either exactly or as an abbreviation of a b-token (man -> manchester)."""
    return all(any(t == u or (len(t) >= 3 and len(u) >= 4 and u.startswith(t)) for u in b) for t in a)


def _sim(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = a.split(), b.split()
    if len(ta) < len(tb):
        pairs = [(ta, tb)]
    elif len(tb) < len(ta):
        pairs = [(tb, ta)]
    else:
        pairs = [(ta, tb), (tb, ta)]
    for short, long_ in pairs:
        if short and _covers(short, long_) and (len(short) >= 2 or len(short[0]) >= 4):
            return 0.92 - 0.01 * (len(long_) - len(short))
    return difflib.SequenceMatcher(None, a, b).ratio()


def best_match(name, names, thr=0.86):
    """names = iterable of raw names. Returns the raw name that matches `name`, or None (ambiguous / too different)."""
    n = norm_team(name)
    best = second = 0.0
    arg = None
    for raw in names:
        s = _sim(n, norm_team(raw))
        if s > best:
            second, best, arg = best, s, raw
        elif s > second:
            second = s
    if arg is not None and best >= thr and (best - second >= 0.04 or best >= 0.999):
        return arg
    return None


def _fd_date(s):
    s = (s or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None


def fd_season_code(d, back=0):
    yy = (d.year if d.month >= 7 else d.year - 1) % 100
    yy = (yy - back) % 100
    return f"{yy:02d}{(yy + 1) % 100:02d}"


def _odds_entry(v):
    return {"odds": float(v), "median": float(v), "best": float(v), "low": float(v), "n": 1}


def fd_row_odds(row):
    """Bookmaker prices from one CSV row, always taking all outcomes from the SAME bookmaker."""
    out = {}
    for cols in (("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA"), ("PSH", "PSD", "PSA"), ("PH", "PD", "PA"), ("MaxH", "MaxD", "MaxA")):
        v = [_f(row.get(c)) for c in cols]
        if all(x and x > 1.0 for x in v):
            out.update({"1": _odds_entry(v[0]), "X": _odds_entry(v[1]), "2": _odds_entry(v[2])})
            break
    for cols in (("B365>2.5", "B365<2.5"), ("Avg>2.5", "Avg<2.5"), ("P>2.5", "P<2.5"), ("Max>2.5", "Max<2.5")):
        v = [_f(row.get(c)) for c in cols]
        if all(x and x > 1.0 for x in v):
            out.update({"O2.5": _odds_entry(v[0]), "U2.5": _odds_entry(v[1])})
            break
    return out


def _ts_of(d, tm="12:00"):
    try:
        hh, mm = [int(x) for x in (tm or "12:00").split(":")[:2]]
    except Exception:
        hh, mm = 12, 0
    return int(datetime(d.year, d.month, d.day, hh, mm, tzinfo=timezone.utc).timestamp())


def _int_or_none(x):
    v = _f(x)
    return None if v is None else int(v)


def fd_parse_main(text):
    """football-data.co.uk main-league CSV (also fixtures.csv). Rows without a full-time score are fixtures."""
    rows = []
    if not text or "HomeTeam" not in text[:600]:
        return rows
    for r in csv.DictReader(io.StringIO(text)):
        d, h, a = _fd_date(r.get("Date")), (r.get("HomeTeam") or "").strip(), (r.get("AwayTeam") or "").strip()
        if not d or not h or not a:
            continue
        rows.append({"div": (r.get("Div") or "").strip(), "date": d, "ts": _ts_of(d, r.get("Time")), "time": r.get("Time"),
                     "home": h, "away": a, "hg": _int_or_none(r.get("FTHG")), "ag": _int_or_none(r.get("FTAG")),
                     "hthg": _int_or_none(r.get("HTHG")), "htag": _int_or_none(r.get("HTAG")),
                     "hc": _f(r.get("HC")), "ac": _f(r.get("AC")), "hst": _f(r.get("HST")), "ast": _f(r.get("AST")),
                     "odds": fd_row_odds(r)})
    return rows


def fd_parse_new(text, since):
    """football-data.co.uk 'new leagues' CSV (Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PH,PD,PA,...)."""
    rows = []
    if not text or "Home" not in text[:600]:
        return rows
    for r in csv.DictReader(io.StringIO(text)):
        d, h, a = _fd_date(r.get("Date")), (r.get("Home") or "").strip(), (r.get("Away") or "").strip()
        if not d or not h or not a or d < since:
            continue
        rows.append({"div": (r.get("League") or "").strip(), "date": d, "ts": _ts_of(d, r.get("Time")), "time": r.get("Time"),
                     "home": h, "away": a, "hg": _int_or_none(r.get("HG")), "ag": _int_or_none(r.get("AG")),
                     "hthg": None, "htag": None, "hc": None, "ac": None, "hst": None, "ast": None, "odds": fd_row_odds(r)})
    return rows


def american_to_decimal(x):
    try:
        s = str(x).strip().upper()
        if s in ("EVEN", "EV", "PK"):
            return 2.0
        v = float(s.replace("+", ""))
        if v == 0:
            return None
        return 1 + v / 100.0 if v > 0 else 1 + 100.0 / abs(v)
    except Exception:
        return None


def espn_parse_odds(lst):
    """Defensive parse of ESPN's competition.odds block -> app odds map (1X2 and, if present, the O/U line)."""
    try:
        o = (lst or [{}])[0] or {}
        ml = lambda blk: american_to_decimal((blk or {}).get("moneyLine"))
        h, d, a = ml(o.get("homeTeamOdds")), ml(o.get("drawOdds")), ml(o.get("awayTeamOdds"))
        out = {}
        if h and d and a and min(h, d, a) > 1.0:
            out.update({"1": _odds_entry(h), "X": _odds_entry(d), "2": _odds_entry(a)})
        line = _f(o.get("overUnder"))
        ov, un = american_to_decimal(o.get("overOdds")), american_to_decimal(o.get("underOdds"))
        if line in GOAL_LINES and ov and un and ov > 1.0 and un > 1.0:
            out.update({f"O{line:g}": _odds_entry(ov), f"U{line:g}": _odds_entry(un)})
        return out
    except Exception:
        return {}


def elo_probs(eh, ea, hfa=65.0):
    """ClubElo ratings -> 1/X/2 probabilities (independent strength opinion)."""
    dr = eh + hfa - ea
    e = 1.0 / (1.0 + 10 ** (-dr / 400.0))
    px = max(0.12, 0.30 - 0.55 * (e - 0.5) ** 2)
    p1, p2 = max(0.02, e - px / 2), max(0.02, 1 - e - px / 2)
    s = p1 + px + p2
    return {"1": p1 / s, "X": px / s, "2": p2 / s}


class FreeData:
    def __init__(self, enabled=True, cache_dir=CACHE_DIR):
        self.enabled = enabled
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update(FREE_UA)
        self.lock = threading.RLock()
        self.mem: Dict[str, Optional[str]] = {}
        self.used = defaultdict(int)        # source -> number of matches it helped
        self.saved_calls = 0                # API-Football calls avoided thanks to free data (estimate)
        self.fails: List[str] = []
        self._rows: Dict[Any, list] = {}
        self._fix_main: Optional[list] = None
        self._espn: Dict[Any, list] = {}
        self._elo: Optional[list] = None
        self._intl: Optional[list] = None
        self._intl_names_c: Optional[list] = None
        self._intl_elo_c: Optional[tuple] = None
        self.target_date: Optional[str] = None      # day being analysed (set by the pipeline)
        self.intl_extra = 0                         # recent results added from ESPN on top of the dataset
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass

    # ── transport (never raises) ──
    def _get(self, url, ttl=FREE_TTL_S, timeout=25):
        if not self.enabled:
            return None
        with self.lock:
            if url in self.mem:
                return self.mem[url]
        path = os.path.join(self.cache_dir, "free_" + hashlib.md5(url.encode()).hexdigest() + ".txt")
        txt = None
        try:
            if os.path.exists(path) and time.time() - os.path.getmtime(path) < ttl:
                with open(path, "r", encoding="utf-8") as fh:
                    txt = fh.read()
        except Exception:
            txt = None
        if txt is None:
            err = ""
            for attempt in range(2):
                try:
                    r = self.session.get(url, timeout=timeout)
                    if r.status_code == 200 and r.content:
                        txt = r.content.decode("utf-8-sig", errors="replace")
                        break
                    err = f"HTTP {r.status_code}"
                    if r.status_code in (403, 404):
                        break
                except Exception as e:
                    err = str(e)[:70]
                time.sleep(0.8)
            if txt:
                try:
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(txt)
                except Exception:
                    pass
            else:
                self.fails.append(f"{url.split('//')[-1][:60]}: {err or 'empty'}")
        with self.lock:
            self.mem[url] = txt
        return txt

    # ── football-data.co.uk ──
    def _main_rows(self, code, season):
        key = ("main", code, season)
        with self.lock:
            if key in self._rows:
                return self._rows[key]
        rows = fd_parse_main(self._get(f"{FD_BASE}/mmz4281/{season}/{code}.csv"))
        rows = [r for r in rows if r["hg"] is not None and r["ag"] is not None]
        with self.lock:
            self._rows[key] = rows
        return rows

    def _new_rows(self, code, since):
        key = ("new", code)
        with self.lock:
            if key in self._rows:
                return self._rows[key]
        rows = fd_parse_new(self._get(f"{FD_BASE}/new/{code}.csv"), since)
        with self.lock:
            self._rows[key] = rows
        return rows

    def rows_for(self, league_id, date_str):
        """Played matches of the league (current + previous season when the season is young)."""
        if not self.enabled:
            return []
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        if league_id in FD_MAIN:
            code = FD_MAIN[league_id]
            rows = list(self._main_rows(code, fd_season_code(d)))
            if len(rows) < 150:
                rows += self._main_rows(code, fd_season_code(d, 1))
            return rows
        if league_id in FD_NEW:
            rows = self._new_rows(FD_NEW[league_id], d - timedelta(days=520))
            return [r for r in rows if r["hg"] is not None and r["ag"] is not None]
        return []

    def _names(self, rows):
        return sorted({r["home"] for r in rows} | {r["away"] for r in rows})

    def find_team(self, name, league_id, date_str):
        rows = self.rows_for(league_id, date_str)
        if not rows:
            return None, rows
        return best_match(name, self._names(rows)), rows

    def team_form(self, name, league_id, date_str, n=10):
        nm, rows = self.find_team(name, league_id, date_str)
        if not nm:
            return self.intl_form(name, date_str, n) if league_id not in FD_ALL else None
        d0 = datetime.strptime(date_str, "%Y-%m-%d").date()
        games = sorted([r for r in rows if r["date"] < d0 and nm in (r["home"], r["away"])], key=lambda r: -r["ts"])[:n]
        if len(games) < 3:
            return None
        res, cf, ca, st_ = [], [], [], []
        for i, r in enumerate(games):
            ih = r["home"] == nm
            gf, ga = (r["hg"], r["ag"]) if ih else (r["ag"], r["hg"])
            res.append({"id": f"fd:{r['date'].isoformat()}:{i}", "ts": r["ts"], "home": ih, "gf": gf, "ga": ga,
                        "res": "W" if gf > ga else "D" if gf == ga else "L", "opp": r["away" if ih else "home"], "league_id": league_id})
            if i < 5:
                c1, c2 = (r["hc"], r["ac"]) if ih else (r["ac"], r["hc"])
                if c1 is not None and c2 is not None:
                    cf.append(c1)
                    ca.append(c2)
                s1 = r["hst"] if ih else r["ast"]
                if s1 is not None:
                    st_.append(s1)
        avg = lambda l: float(np.mean(l)) if l else None
        hs = {"cor_for": avg(cf), "cor_against": avg(ca), "cor_n": len(cf), "sot": avg(st_), "xg_for": None, "xg_against": None, "players": {}}
        return {"results": res, "hs": hs, "name": nm, "stale_days": (d0 - games[0]["date"]).days}

    # ── national teams: martj42/international_results (all international matches, free CSV on GitHub) ──
    def intl_rows(self):
        with self.lock:
            if self._intl is not None:
                return self._intl
        txt = self._get(INTL_URL, ttl=12 * 3600, timeout=45)
        rows = []
        if txt and txt[:30].lower().startswith("date"):
            for r in csv.DictReader(io.StringIO(txt)):
                d, hg, ag = _fd_date(r.get("date")), _int_or_none(r.get("home_score")), _int_or_none(r.get("away_score"))
                if not d or hg is None or ag is None:
                    continue
                rows.append({"date": d, "ts": _ts_of(d), "home": r.get("home_team") or "", "away": r.get("away_team") or "", "hg": hg, "ag": ag,
                             "tour": r.get("tournament") or "", "neutral": (r.get("neutral") or "").upper() == "TRUE"})
            rows.sort(key=lambda r: r["ts"])
        names = sorted({r["home"] for r in rows} | {r["away"] for r in rows})
        try:
            rows = self._intl_gap_fill(rows, names)
        except Exception as e:
            self.fails.append(f"espn gap-fill: {str(e)[:60]}")
        with self.lock:
            self._intl = rows
            self._intl_names_c = names
        return rows

    def _intl_gap_fill(self, rows, names):
        """The results file lags by weeks. Top it up with finished matches from ESPN's public scoreboards (best effort)."""
        if not rows or not self.target_date:
            return rows
        tgt = datetime.strptime(self.target_date, "%Y-%m-%d").date()
        last = rows[-1]["date"]
        if (tgt - last).days <= 3:
            return rows
        days = [last + timedelta(days=i) for i in range(1, min((tgt - last).days, 45) + 1) if last + timedelta(days=i) < tgt]
        jobs = [(slug, d.isoformat()) for d in days for slug in ("uefa.nations", "fifa.friendly")]
        have = {(r["date"], r["home"], r["away"]) for r in rows[-400:]}
        added = []
        with ThreadPoolExecutor(max_workers=6) as ex:
            for evs, job in zip(ex.map(lambda j: self.espn_events(*j), jobs), jobs):
                slug = job[0]
                for e in evs:
                    if e["state"] != "post" or e["hg"] is None or e["ag"] is None:
                        continue
                    h, a = best_match(e["home"], names, 0.95), best_match(e["away"], names, 0.95)
                    d = datetime.fromtimestamp(e["ts"], timezone.utc).date()
                    if h and a and h != a and (d, h, a) not in have and d < tgt:
                        have.add((d, h, a))
                        added.append({"date": d, "ts": e["ts"], "home": h, "away": a, "hg": e["hg"], "ag": e["ag"],
                                      "tour": "UEFA Nations League" if slug == "uefa.nations" else "Friendly", "neutral": False})
        if added:
            self.intl_extra = len(added)
            rows = sorted(rows + added, key=lambda r: r["ts"])
        return rows

    def intl_team(self, name):
        self.intl_rows()
        return best_match(name, self._intl_names_c or [], 0.95)      # near-exact only: a club must never be mistaken for a nation

    def intl_form(self, name, date_str, n=10):
        rows = self.intl_rows()
        nm = self.intl_team(name) if rows else None
        if not nm:
            return None
        d0 = datetime.strptime(date_str, "%Y-%m-%d").date()
        games = []
        for r in reversed(rows):
            if r["date"] < d0 and nm in (r["home"], r["away"]):
                games.append(r)
                if len(games) >= n:
                    break
        if len(games) < 3:
            return None
        res = []
        for i, r in enumerate(games):
            ih = r["home"] == nm
            gf, ga = (r["hg"], r["ag"]) if ih else (r["ag"], r["hg"])
            res.append({"id": f"fd:intl:{r['date'].isoformat()}:{i}", "ts": r["ts"], "home": ih, "gf": gf, "ga": ga,
                        "res": "W" if gf > ga else "D" if gf == ga else "L", "opp": r["away" if ih else "home"], "league_id": 0})
        hs = {"cor_for": None, "cor_against": None, "cor_n": 0, "sot": None, "xg_for": None, "xg_against": None, "players": {}}
        return {"results": res, "hs": hs, "name": nm, "stale_days": (d0 - games[0]["date"]).days, "intl": True}

    def intl_h2h(self, home, away, date_str):
        rows = self.intl_rows()
        nh, na = (self.intl_team(home), self.intl_team(away)) if rows else (None, None)
        if not nh or not na or nh == na:
            return []
        d0 = datetime.strptime(date_str, "%Y-%m-%d").date()
        out = []
        for r in reversed(rows):
            if r["date"] < d0 and {r["home"], r["away"]} == {nh, na}:
                ih = r["home"] == nh
                out.append({"ts": r["ts"], "gf": r["hg"] if ih else r["ag"], "ga": r["ag"] if ih else r["hg"]})
                if len(out) >= 8:
                    break
        return out

    @staticmethod
    def _intl_k(tour):
        t = (tour or "").lower()
        if "friendly" in t:
            return 20
        if t == "fifa world cup":
            return 60
        if "qualification" in t:
            return 40
        if any(x in t for x in ("uefa euro", "copa am", "african cup of nations", "afc asian cup", "gold cup", "confederations")):
            return 50
        if "nations league" in t:
            return 40
        return 30

    def intl_elo_table(self, date_str):
        """World-Football-Elo style ratings computed from the full results file (home edge 100, goal-difference multiplier)."""
        with self.lock:
            if self._intl_elo_c and self._intl_elo_c[0] == date_str:
                return self._intl_elo_c[1]
        d0 = datetime.strptime(date_str, "%Y-%m-%d").date()
        elo = defaultdict(lambda: 1500.0)
        for r in self.intl_rows():
            if r["date"] >= d0:
                break
            if r["date"].year < 1990:
                continue
            h, a = r["home"], r["away"]
            dr = elo[h] - elo[a] + (0 if r["neutral"] else 100.0)
            we = 1.0 / (10 ** (-dr / 400.0) + 1.0)
            w = 1.0 if r["hg"] > r["ag"] else 0.5 if r["hg"] == r["ag"] else 0.0
            gd = abs(r["hg"] - r["ag"])
            g = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8.0
            ch = self._intl_k(r["tour"]) * g * (w - we)
            elo[h] += ch
            elo[a] -= ch
        tab = dict(elo)
        with self.lock:
            self._intl_elo_c = (date_str, tab)
        return tab

    def intl_elo(self, home, away, date_str):
        rows = self.intl_rows()
        nh, na = (self.intl_team(home), self.intl_team(away)) if rows else (None, None)
        if not nh or not na or nh == na:
            return None
        tab = self.intl_elo_table(date_str)
        if nh not in tab or na not in tab:
            return None
        return {"h": tab[nh], "a": tab[na], "p": elo_probs(tab[nh], tab[na], hfa=100.0), "intl": True}

    def covers(self, m, date_str):
        """True when free sources can supply form for BOTH teams (no API call needed)."""
        return bool(self.enabled and self.team_form(m["home"], m["league_id"], date_str) and self.team_form(m["away"], m["league_id"], date_str))

    def h2h(self, home, away, league_id, date_str):
        if league_id not in FD_ALL:
            return self.intl_h2h(home, away, date_str)
        nh, rows = self.find_team(home, league_id, date_str)
        na = best_match(away, self._names(rows)) if rows else None
        if not nh or not na or nh == na:
            return []
        d0 = datetime.strptime(date_str, "%Y-%m-%d").date()
        out = []
        for r in sorted(rows, key=lambda r: -r["ts"]):
            if r["date"] < d0 and {r["home"], r["away"]} == {nh, na}:
                ih = r["home"] == nh
                out.append({"ts": r["ts"], "gf": r["hg"] if ih else r["ag"], "ga": r["ag"] if ih else r["hg"]})
        return out[:8]

    def standing(self, name, league_id, date_str):
        if league_id not in FD_MAIN:
            return None
        nm, rows = self.find_team(name, league_id, date_str)
        if not nm:
            return None
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        start = date_cls(d.year if d.month >= 7 else d.year - 1, 7, 1)
        tab = defaultdict(lambda: [0, 0, 0])      # pts, gd, played
        for r in rows:
            if start <= r["date"] < d:
                for t, gf, ga in ((r["home"], r["hg"], r["ag"]), (r["away"], r["ag"], r["hg"])):
                    e = tab[t]
                    e[0] += 3 if gf > ga else 1 if gf == ga else 0
                    e[1] += gf - ga
                    e[2] += 1
        if nm not in tab or len(tab) < 8 or np.mean([v[2] for v in tab.values()]) < 3:
            return None
        order = sorted(tab, key=lambda t: (-tab[t][0], -tab[t][1]))
        return {"rank": order.index(nm) + 1, "pts": tab[nm][0], "of": len(order), "played": tab[nm][2], "desc": None}

    def _fix_rows(self):
        with self.lock:
            if self._fix_main is not None:
                return self._fix_main
        rows = fd_parse_main(self._get(f"{FD_BASE}/fixtures.csv", ttl=3 * 3600))
        rows = [r for r in rows if r["hg"] is None]
        with self.lock:
            self._fix_main = rows
        return rows

    def fd_fixtures(self, league_id, date_str):
        """Upcoming rows (with bookmaker prices) for the league, from fixtures.csv or the new-league file."""
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        if league_id in FD_MAIN:
            return [r for r in self._fix_rows() if r["div"] == FD_MAIN[league_id] and abs((r["date"] - d).days) <= 1]
        if league_id in FD_NEW:
            rows = self._new_rows(FD_NEW[league_id], d - timedelta(days=520))
            return [r for r in rows if r["hg"] is None and abs((r["date"] - d).days) <= 1]
        return []

    # ── ESPN ──
    def espn_events(self, slug, date_str):
        key = (slug, date_str)
        try:
            past = datetime.strptime(date_str, "%Y-%m-%d").date() < datetime.now(timezone.utc).date() - timedelta(days=1)
        except Exception:
            past = False
        with self.lock:
            if key in self._espn:
                return self._espn[key]
        out = []
        txt = self._get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={date_str.replace('-', '')}", ttl=(7 * 86400 if past else 1800), timeout=15)
        try:
            data = json.loads(txt) if txt else {}
            for ev in data.get("events") or []:
                comp = (ev.get("competitions") or [{}])[0]
                cs = comp.get("competitors") or []
                hm = next((c for c in cs if c.get("homeAway") == "home"), None)
                aw = next((c for c in cs if c.get("homeAway") == "away"), None)
                if not hm or not aw:
                    continue
                state = (((ev.get("status") or {}).get("type") or {}).get("state") or "").lower()
                ds = (ev.get("date") or "").replace("Z", "+0000")
                ts = None
                for fmt in ("%Y-%m-%dT%H:%M%z", "%Y-%m-%dT%H:%M:%S%z"):
                    try:
                        ts = int(datetime.strptime(ds, fmt).timestamp())
                        break
                    except Exception:
                        pass
                if ts is None:
                    continue
                out.append({"home": (hm.get("team") or {}).get("displayName") or "", "away": (aw.get("team") or {}).get("displayName") or "",
                            "ts": ts, "state": state, "odds": espn_parse_odds(comp.get("odds")),
                            "hg": _int_or_none(hm.get("score")), "ag": _int_or_none(aw.get("score"))})
        except Exception as e:
            self.fails.append(f"espn parse {slug}: {str(e)[:50]}")
        with self.lock:
            self._espn[key] = out
        return out

    # ── odds for one match (FD first: several markets, then ESPN) ──
    def odds_for(self, home, away, league_id, date_str):
        if not self.enabled:
            return {}, None
        rows = self.fd_fixtures(league_id, date_str)
        if rows:
            names = self._names(rows)
            nh, na = best_match(home, names), best_match(away, names)
            if nh and na:
                for r in rows:
                    if r["home"] == nh and r["away"] == na and r["odds"].get("1"):
                        return dict(r["odds"]), "football-data"
        slug = ESPN_SLUG.get(league_id)
        if slug:
            ev = self.espn_events(slug, date_str)
            if ev:
                names = sorted({e["home"] for e in ev} | {e["away"] for e in ev})
                nh, na = best_match(home, names), best_match(away, names)
                if nh and na:
                    for e in ev:
                        if e["home"] == nh and e["away"] == na and e["odds"].get("1"):
                            return dict(e["odds"]), "espn"
        return {}, None

    # ── ClubElo ──
    def elo_table(self, date_str):
        with self.lock:
            if self._elo is not None:
                return self._elo
        txt = self._get(f"http://api.clubelo.com/{date_str}", ttl=12 * 3600, timeout=20) or self._get(f"https://api.clubelo.com/{date_str}", ttl=12 * 3600, timeout=20)
        tab = []
        try:
            for r in csv.DictReader(io.StringIO(txt or "")):
                e = _f(r.get("Elo"))
                if r.get("Club") and e:
                    tab.append((r["Club"], e))
        except Exception:
            tab = []
        with self.lock:
            self._elo = tab
        return tab

    def elo_for(self, home, away, league_id, country, date_str):
        """Returns {'h','a','p'} or None. Only used for European competitions (ClubElo covers Europe)."""
        if not self.enabled:
            return None
        if not (league_id in ELO_LEAGUE_IDS or (country or "").lower() in ELO_COUNTRIES or league_id in FD_MAIN):
            return self.intl_elo(home, away, date_str) if league_id not in FD_ALL else None
        tab = self.elo_table(date_str)
        if not tab:
            return None
        names = [c for c, _ in tab]
        nh, na = best_match(home, names, 0.88), best_match(away, names, 0.88)
        if not nh or not na or nh == na:
            return None
        e = dict(tab)
        return {"h": e[nh], "a": e[na], "p": elo_probs(e[nh], e[na])}

    # ── settlement helper ──
    def result_for(self, home, away, league_id, date_str):
        if league_id not in FD_ALL:
            rows = self.intl_rows()
            nh, na = (self.intl_team(home), self.intl_team(away)) if rows else (None, None)
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            for r in rows if (nh and na) else []:
                if r["home"] == nh and r["away"] == na and abs((r["date"] - d).days) <= 1:
                    return {"goals": {"home": r["hg"], "away": r["ag"]}, "score": {"fulltime": {"home": r["hg"], "away": r["ag"]}}}
            return None
        nh, rows = self.find_team(home, league_id, date_str)
        na = best_match(away, self._names(rows)) if rows else None
        if not nh or not na:
            return None
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        for r in rows:
            if r["home"] == nh and r["away"] == na and abs((r["date"] - d).days) <= 1:
                f = {"goals": {"home": r["hg"], "away": r["ag"]}, "score": {"fulltime": {"home": r["hg"], "away": r["ag"]}}}
                if r["hthg"] is not None and r["htag"] is not None:
                    f["score"]["halftime"] = {"home": r["hthg"], "away": r["htag"]}
                if r["hc"] is not None and r["ac"] is not None:
                    f["statistics"] = [{"statistics": [{"type": "Corner Kicks", "value": r["hc"]}]},
                                       {"statistics": [{"type": "Corner Kicks", "value": r["ac"]}]}]
                return f
        return None

    # ── fixtures when API-Football gives none (free-only mode) ──
    def free_fixtures(self, date_str, tz, workers=6):
        if not self.enabled:
            return []
        zi = ZoneInfo(tz)
        tgt = datetime.strptime(date_str, "%Y-%m-%d").date()
        out = []
        prev = (tgt - timedelta(days=1)).isoformat()
        jobs = [(slug, ds) for slug in ESPN_SLUG.values() for ds in (date_str, prev)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.espn_events, s, ds): s for s, ds in jobs}
            for fu in as_completed(futs):
                slug = futs[fu]
                try:
                    evs = fu.result()
                except Exception:
                    continue
                lid = ESPN_SLUG_TO_LID[slug]
                for e in evs:
                    if e["state"] not in ("pre", "") or datetime.fromtimestamp(e["ts"], zi).date() != tgt:
                        continue
                    if not any(o["league_id"] == lid and o["ts"] == e["ts"] and _sim(norm_team(o["home"]), norm_team(e["home"])) >= 0.9 for o in out):
                        out.append({"league_id": lid, "home": e["home"], "away": e["away"], "ts": e["ts"], "odds": e["odds"], "src": "espn"})
        london = ZoneInfo("Europe/London")
        for lid in list(FD_MAIN) + list(FD_NEW):
            for r in self.fd_fixtures(lid, date_str):
                try:
                    hh, mm = [int(x) for x in (r["time"] or "15:00").split(":")[:2]]
                    ts = int(datetime(r["date"].year, r["date"].month, r["date"].day, hh, mm, tzinfo=london).timestamp())
                except Exception:
                    ts = r["ts"]
                if datetime.fromtimestamp(ts, zi).date() != tgt:
                    continue
                dup = next((o for o in out if o["league_id"] == lid and best_match(r["home"], [o["home"]]) and best_match(r["away"], [o["away"]])), None)
                if dup is not None:
                    if not dup["odds"].get("O2.5") and r["odds"]:
                        dup["odds"] = dict(r["odds"])          # football-data prices carry more markets than ESPN's
                    continue
                out.append({"league_id": lid, "home": r["home"], "away": r["away"], "ts": ts, "odds": dict(r["odds"]), "src": "football-data"})
        return out

    def summary(self):
        return {"used": dict(self.used), "saved_calls": self.saved_calls, "fails": self.fails[:6]}


def derive_dc_odds(odds):
    """Double-chance prices derived from the 1X2 prices when a source does not quote them (keeps the book's margin)."""
    if not odds or not all(k in odds for k in ("1", "X", "2")):
        return odds
    out = dict(odds)
    inv = {k: 1.0 / odds[k]["odds"] for k in ("1", "X", "2")}
    for key, parts in (("1X", ("1", "X")), ("X2", ("X", "2")), ("12", ("1", "2"))):
        if key not in out:
            s = sum(inv[p] for p in parts)
            if 0 < s < 0.98:
                v = 1.0 / s
                out[key] = {"odds": v, "median": v, "best": v, "low": v, "n": 0, "derived": True}
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
    prof = {"n": len(r10), "last": [f"{x['gf']}-{x['ga']}{x['res']}" for x in r10[:5]]}
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
def estimate_lambdas(pred, ph, pa, elo=None):
    """Returns (lam_h, lam_a, injury_impacts, basis). `basis` records what real data the numbers rest on;
    'elo' = only ClubElo strength was available (weak but team-specific); 'prior' = NO team data at all (match is unusable)."""
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
    elif elo:
        e = elo["p"]
        share = 0.5 + 0.8 * ((e["1"] + e["X"] / 2) - 0.5)
        tot = prior_h + prior_a
        lh, la, m, basis = tot * share, tot * (1 - share), 4, "elo"
    else:
        lh, la, m, basis = prior_h, prior_a, 0, "prior"
    if elo and elo.get("intl") and basis != "elo" and lh + la > 0:      # national teams: form ignores opposition strength, Elo supplies it
        e = elo["p"]
        es = 0.5 + 0.8 * ((e["1"] + e["X"] / 2) - 0.5)
        sh = 0.55 * (lh / (lh + la)) + 0.45 * es
        lh, la = (lh + la) * sh, (lh + la) * (1 - sh)
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


API_DEFAULT_PATTERNS = {(0, 50, 50), (10, 45, 45), (30, 35, 35), (33, 33, 33)}   # the API's placeholders when it has no real data


def api_dist(pred):
    """API-Football's own 1/X/2 percentages, ignoring its placeholder patterns."""
    pc = (pred or {}).get("pct") or []
    if len(pc) != 3 or None in pc or sum(pc) <= 0:
        return None
    if tuple(sorted(int(round(x)) for x in pc)) in API_DEFAULT_PATTERNS:
        return None
    t = sum(pc)
    return {"1": pc[0] / t, "X": pc[1] / t, "2": pc[2] / t}


def api_fair(key, d, default):
    if not d:
        return default
    if key in d:
        return d[key]
    if key in ("1X", "X2", "12"):
        return sum(d[c] for c in key)
    return default


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
    pre = {"api": "api", "elo": "elo"}.get(m.get("ext_src"), "mkt")
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
        risk += 35; why.append(f"{pre}{_pct(fp)}/mdl{_pct(mp[fav])}")
    elif gap >= 0.08:
        risk += 22; why.append(f"{pre}{_pct(fp)}/mdl{_pct(mp[fav])}")
    elif gap >= 0.05:
        risk += 10; why.append(f"{pre}{_pct(fp)}/mdl{_pct(mp[fav])}")
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
    dq += 0.20 if m.get("pred") else (0.10 if m.get("elo") else 0)
    dq += 0.20 if m["ph"].get("n", 0) >= 5 and m["pa"].get("n", 0) >= 5 else (0.10 if m["ph"].get("n", 0) >= 3 else 0)
    dq += 0.10 if m.get("h2h") else 0
    dq += 0.10 if m.get("corner_n", 0) >= 3 else 0
    dq += 0.05 if m.get("inj_known") else 0
    dq += 0.05 if (m["ph"].get("lineup") and m["pa"].get("lineup")) else 0
    dq += 0.05 if m.get("st_h") else 0
    age = m.get("form_age_days") or 0
    if age > 90:                                # newest result is over 3 months old (squad/coach/tournament effects may have changed everything)
        dq = max(0.0, dq - 0.10)
    elif age > 30:
        dq = max(0.0, dq - 0.05)
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


def build_legs(m, allow_est=False, corners_ok_min=3, wide=True):
    """Leg menu for one match. SAFE legs (tier S) pass the strict two-source screen; BORDERLINE legs (tier B) pass a looser
    screen and are shown to the AI with a '~' flag so it can accept or reject them with reasons instead of Python hiding them."""
    fav, risk = m["trap"]["fav"], m["trap"]["risk"]
    dq = m["dq"]
    ext = m.get("ext_d") or m.get("api_1x2")
    legs = []
    for key, p in m["p"].items():
        if key.startswith("C_") and m.get("corner_n", 0) < corners_ok_min:
            continue
        o = m["odds"].get(key)
        src = "book"
        lo = hi = None
        if o:
            odds = o["odds"]
            if o.get("derived"):
                src = "derived"
            else:
                lo, hi = o.get("low", odds), o.get("best", odds)
            p_mkt = market_fair_prob(key, m["odds"])
        elif allow_est and not m["odds"]:           # only estimate when the match has NO bookmaker prices at all
            odds, src = round(max(1.05, 0.93 / max(p, 0.05)), 2), "est"
            p_mkt = api_fair(key, ext, p)           # for 1X2/double-chance legs the independent model (API or Elo) must agree
        else:
            continue
        if not (MIN_LEG_ODDS <= odds <= MAX_LEG_ODDS):
            continue
        p_model = m["model_p"].get(key, p)
        agree = min(p_model, p_mkt)
        if src == "est":
            if p_model < 0.74 or p < 0.74:      # no bookmaker quote = no market check: only clear-cut legs, always shown as borderline
                continue
            tier = "B"
        elif agree >= MIN_AGREE_P:
            tier = "S"
        elif wide and agree >= BORDER_AGREE_P:
            tier = "B"
        else:
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
        if tier == "B":
            pen += 0.02
        if src == "est":
            pen += 0.04
        p_adj = p - pen
        if p_adj < (MIN_LEG_P if tier == "S" else BORDER_MIN_P):
            continue
        edge = p * odds - 1
        eadj = float(np.clip(p_adj * odds - 1, -0.2, 0.15))
        legs.append({"key": key, "label": leg_label(key, m), "p": float(p), "p_adj": float(p_adj), "p_mkt": float(p_mkt),
                     "p_model": float(p_model), "odds": float(odds), "odds_lo": lo, "odds_hi": hi, "edge": float(edge), "src": src,
                     "tier": tier, "rank": float(p_adj + 0.5 * eadj - (0.03 if tier == "B" else 0.0)), "fam": _family(key)})
    legs.sort(key=lambda x: -x["rank"])
    chosen, fams, nb = [], set(), 0
    for lg in legs:
        if lg["fam"] in fams:
            continue
        if lg["tier"] == "B":
            if nb >= 2:
                continue
            nb += 1
        chosen.append(lg)
        fams.add(lg["fam"])
        if len(chosen) >= MAX_MENU_LEGS:
            break
    return chosen


def gate_reasons(m, cfg):
    """Evidence gate: a match may only feed tickets if it rests on real team data AND real prices."""
    why = []
    if m["basis"] == "prior":
        why.append("no team-specific data (league-average defaults only)")
    if not m["odds"] and not cfg.get("allow_est"):
        why.append("no real bookmaker odds")
    if (m.get("form_age_days") or 0) > 200:
        why.append(f"newest form result is {m['form_age_days']} days old")
    if m["dq"] < cfg.get("min_dq", 0.5):
        why.append(f"data quality {m['dq']} < {cfg.get('min_dq', 0.5)}")
    if not why:
        if not m["legs"]:
            why.append("no leg clears the safety filters (model AND market >= 55-60%)")
        elif not any(l["tier"] == "S" for l in m["legs"]) and not cfg.get("allow_b_only"):
            why.append("only borderline legs (model/market agreement 55-60%)")
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


def shortlist_matches(cands, max_n, force_leagues, quota=None):
    quota = quota or CAT_QUOTA
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
        take = math.ceil(quota[cat] * max_n)
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


def _syn_id(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def make_record(m, c, hs_h, hs_a, inj_h, inj_a, lu_h, lu_a, odds, odds_src, st_h, st_a, inj_known, cfg):
    """Turns collected raw data for one match into the analysed record (profiles, Monte Carlo, blend, trap, dq)."""
    ph = build_profile(m["home_id"], c["rh"], hs_h, inj_h, lu_h)
    pa = build_profile(m["away_id"], c["ra"], hs_a, inj_a, lu_a)
    enrich_profile_from_l5(ph, (c["pred"] or {}).get("home"))
    enrich_profile_from_l5(pa, (c["pred"] or {}).get("away"))
    elo = c.get("elo")
    lam_h, lam_a, _, basis = estimate_lambdas(c["pred"], ph, pa, elo)
    mu_ch, mu_ca, cn = estimate_corners(ph, pa, lam_h, lam_a)
    mc = monte_carlo(lam_h, lam_a, mu_ch, mu_ca, n=cfg["sims"], seed=int(m["id"]) % (2 ** 31))
    odds = derive_dc_odds(odds or {})
    p_bl, mkt = blend_probs(mc["p"], odds, cfg["w_model"])
    api_d = api_dist(c["pred"])
    elo_d = elo["p"] if elo else None
    ext_d = api_d or elo_d
    ext_src = "mkt" if mkt else None
    if not mkt and ext_d:                     # no bookmaker prices: an independent model is the cross-check
        b = {k: 0.65 * p_bl[k] + 0.35 * ext_d[k] for k in ("1", "X", "2")}
        tb = sum(b.values())
        b = {k: v / tb for k, v in b.items()}
        p_bl.update(b)
        p_bl["1X"], p_bl["X2"], p_bl["12"] = b["1"] + b["X"], b["X"] + b["2"], b["1"] + b["2"]
        mkt, ext_src = ext_d, ("api" if api_d else "elo")
    ages = [x["stale_days"] for x in (c.get("fh"), c.get("fa")) if x and x.get("stale_days") is not None]
    m = dict(m, form_age_days=max(ages) if ages else None)
    rec = dict(m, ph=ph, pa=pa, pred=c["pred"], h2h=h2h_summary(c["h2h"]), lam_h=lam_h, lam_a=lam_a, basis=basis,
               model_p=mc["p"], p=p_bl, mkt_1x2=mkt, api_1x2=api_d, elo=elo, ext_d=ext_d, ext_src=ext_src, odds=odds,
               odds_src=odds_src, exp_goals=mc["exp_goals"], exp_corners=mc["exp_corners"], top_scores=mc["top_scores"],
               corner_n=cn, st_h=st_h, st_a=st_a, inj_known=inj_known, free_tags=list(c.get("free_tags", [])))
    rec["trap"] = trap_analysis(rec)
    rec["dq"] = data_quality(rec)
    return rec


def collect_free_only(api, free, cfg, log, progress, out):
    """API-Football unavailable (no key / quota out / plan block): build the whole slate from free sources."""
    date_str, tz = cfg["date"], cfg["tz"]
    free.target_date = date_str
    log("🆓 Free-only mode: collecting fixtures from ESPN + football-data.co.uk (no API-Football calls)...")
    fx = free.free_fixtures(date_str, tz)
    out["raw_count"] = len(fx)
    fixtures = []
    for e in fx:
        lid = e["league_id"]
        name, country = LEAGUE_INFO.get(lid, (f"League {lid}", ""))
        if cfg["exclude_minor"] and (TEAM_EXCLUDE_RE.search(e["home"]) or TEAM_EXCLUDE_RE.search(e["away"])):
            continue
        fixtures.append({"id": -_syn_id(f"F:{date_str}:{e['home']}:{e['away']}"), "ts": e["ts"], "status": "NS", "league_id": lid,
                         "league": name, "country": country, "season": None, "round": None,
                         "home_id": _syn_id("T:" + e["home"]), "home": e["home"], "away_id": _syn_id("T:" + e["away"]), "away": e["away"],
                         "cat": league_category(lid, country), "free_odds": e["odds"], "free_src": e["src"]})
    out["cat_counts"] = {k: int(v) for k, v in pd.Series([f["cat"] for f in fixtures]).value_counts().items()} if fixtures else {}
    log(f"   {len(fx)} fixtures found by free sources, {len(fixtures)} eligible {out['cat_counts']}")
    if not fixtures:
        out["reason"] = "No fixtures were found by API-Football or by the free sources for this date. " + ("; ".join(free.fails[:2]) if free.fails else "")
        return out
    cand = shortlist_matches(fixtures, min(len(fixtures), cfg["max_matches"]), cfg["force_leagues"], BREADTH_QUOTA.get(cfg.get("breadth", "Balanced")))
    out["warnings"].append("API-Football returned nothing usable, so this analysis was built ONLY from free sources (football-data.co.uk, ESPN, ClubElo). "
                           "Injuries, lineups and API predictions are unavailable - those matches show 'inj n/a' and the AI is told to demand extra margin.")
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda lid: free.rows_for(lid, date_str), {c["league_id"] for c in cand}))
    free.elo_table(date_str)
    for i, m in enumerate(cand):
        fh, fa = free.team_form(m["home"], m["league_id"], date_str), free.team_form(m["away"], m["league_id"], date_str)
        if not fh or not fa or len(fh["results"]) < 4 or len(fa["results"]) < 4:
            out["excluded"].append({"league": m["league"], "match": f"{m['home']} v {m['away']}", "reason": "no free-source team history"})
            continue
        odds, osrc = dict(m.get("free_odds") or {}), m.get("free_src")
        if not odds.get("O2.5"):
            o2, s2 = free.odds_for(m["home"], m["away"], m["league_id"], date_str)
            if o2 and (not odds or o2.get("O2.5")):
                odds, osrc = o2, s2
        if not odds:
            out["excluded"].append({"league": m["league"], "match": f"{m['home']} v {m['away']}", "reason": "no bookmaker odds from any source"})
            continue
        c = {"pred": None, "rh": fh["results"], "ra": fa["results"], "h2h": free.h2h(m["home"], m["away"], m["league_id"], date_str),
             "elo": free.elo_for(m["home"], m["away"], m["league_id"], m["country"], date_str), "free_tags": ["form"], "fh": fh, "fa": fa}
        if c["h2h"]:
            c["free_tags"].append("h2h")
        if c["elo"]:
            c["free_tags"].append("elo")
        c["free_tags"].append("odds:" + str(osrc))
        rec = make_record(m, c, fh["hs"], fa["hs"], None, None, None, None, odds, "free:" + str(osrc),
                          free.standing(m["home"], m["league_id"], date_str), free.standing(m["away"], m["league_id"], date_str), False, cfg)
        out["matches"].append(rec)
        for t in c["free_tags"]:
            free.used[t.split(":")[0] if not t.startswith("odds") else "odds"] += 1
        progress(0.2 + 0.7 * (i + 1) / len(cand))
    out["caps"] = {"history": False, "league_odds": False, "fixture_odds": False}
    out["deep"], out["budget"] = False, 0
    out["plan"] = out.get("plan") or {"plan": "free sources only"}
    if not out["matches"]:
        out["reason"] = "Free sources found fixtures but none had both team history and bookmaker odds."
    return out


def collect_day(api, cfg, log, progress, free=None):
    free = free or FreeData(enabled=False)
    out = {"warnings": [], "matches": [], "excluded": [], "raw_count": 0, "cat_counts": {}, "plan": None, "reason": None}
    date_str, tz, force = cfg["date"], cfg["tz"], cfg["force"]
    free.target_date = date_str
    ttl = lambda t: 0 if force else t
    stt = api.status()
    out["plan"] = stt
    if api.offline:
        log("ℹ️ No API-Football key configured - using free sources only.")
        return collect_free_only(api, free, cfg, log, progress, out)
    if stt.get("errors") and not stt.get("plan"):
        out["warnings"].append(f"API status problem: {stt.get('errors')}")
    if api.remaining is not None and api.remaining <= 5:
        out["warnings"].append("API daily quota almost exhausted - only cached data and free sources can be used.")
    budget = min(cfg["max_calls"], api.remaining if api.remaining is not None else cfg["max_calls"])
    if api.remaining is not None and budget < 60:
        out["warnings"].append(f"Only {api.remaining} API calls are left today (plan limit {api.limit}), so this run was limited to fit. "
                               "The quota resets at 00:00 UTC (03:00 in Kampala); cached data and free sources are reused for free.")

    def can_spend(k=1):
        """Call-time guard: never exceed the per-run budget or the remaining daily quota, whatever the plan said."""
        return (not api.quota_out and (api.remaining is None or api.remaining > k) and (budget - api.calls) >= k)

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
    cov = fetch_coverage(api, 7 * 86400)
    if cov:
        for f in fixtures:
            c_ = cov.get(f["league_id"])
            f["cov_pred"], f["cov_odds"] = (c_["pred"], c_["odds"]) if c_ else (None, None)
        n0 = len(fixtures)
        # with free sources a competition without API predictions can still be analysed, so only drop it when free data is off
        if not free.enabled:
            fixtures = [f for f in fixtures if f["cov_pred"] is not False]
            log(f"🗂 Coverage filter: dropped {n0 - len(fixtures)} fixtures in competitions the API has no predictions for; "
                f"{sum(1 for f in fixtures if f['cov_odds'])} of {len(fixtures)} remaining have odds coverage")
    lmem = load_state().get("league_mem3", {})     # v3: old entries (written when national teams had no data source) are ignored
    today_ = datetime.now(timezone.utc).date()

    def mem_bad(lid):
        e = lmem.get(str(lid))
        try:
            return bool(e) and e["t"] >= 2 and e["v"] == 0 and (today_ - datetime.fromisoformat(e["d"]).date()).days <= 5
        except Exception:
            return False
    if not force:
        n0 = len(fixtures)
        fixtures = [f for f in fixtures if not mem_bad(f["league_id"]) or f["league_id"] in cfg["force_leagues"] or (free.enabled and (f["league_id"] in FD_ALL or str(f.get("country") or "").lower() == "world"))]
        if n0 != len(fixtures):
            log(f"🧠 Skipped {n0 - len(fixtures)} fixtures in competitions that returned no usable team data in recent runs (retried after 5 days)")
    out["cat_counts"] = {k: int(v) for k, v in pd.Series([f["cat"] for f in fixtures]).value_counts().items()} if fixtures else {}
    log(f"   {out['raw_count']} fixtures on the day, {len(fixtures)} upcoming & eligible {out['cat_counts']}")
    if not fixtures:
        if free.enabled:
            log("🆓 API-Football gave no usable fixtures - switching to free sources only...")
            out2 = collect_free_only(api, free, cfg, log, progress, out)
            if out2["matches"]:
                return out2
        out["reason"] = "No eligible upcoming fixtures were returned by the API. " + ("; ".join(api.error_summary()[:3]) if api.errors else "")
        return out
    progress(0.06)

    # ── Stage 1: candidate pool ──
    pool_n = min(len(fixtures), int(cfg["max_matches"] * 1.6) + 4)
    cand = shortlist_matches(fixtures, pool_n, cfg["force_leagues"], BREADTH_QUOTA.get(cfg.get("breadth", "Balanced")))
    odds_cap = max(6, int(budget * 0.30))
    while len({c["league_id"] for c in cand}) > odds_cap and pool_n > 8:
        pool_n -= 1
        cand = shortlist_matches(fixtures, pool_n, cfg["force_leagues"], BREADTH_QUOTA.get(cfg.get("breadth", "Balanced")))
    odds_by, odds_src = {}, {}

    def fetch_league_odds(lid, s_):
        for params in ({"league": lid, "season": s_, "date": date_str, "timezone": tz}, {"league": lid, "season": s_, "date": date_str}):
            items, errs = [], False
            for page in (1, 2, 3):
                if not can_spend(1):
                    return items
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
                odds_src[f_] = "api"

    # ── capability probe ──
    state_ = load_state()
    plan_key = str(stt.get("plan") or "Free").lower()
    mem = state_.get("caps_mem") or {}
    fresh = False
    try:
        fresh = mem.get("plan") == plan_key and (datetime.now(timezone.utc).date() - datetime.fromisoformat(mem["date"]).date()).days <= 7
    except Exception:
        pass
    sample = cand[0]
    probed_leagues = set()
    free_hist_sample = bool(free.enabled and free.team_form(sample["home"], sample["league_id"], date_str))
    if fresh and not force:
        caps = dict(mem["caps"])
        api.no_history = not caps["history"]
        log("🧠 Using remembered plan capabilities from an earlier run (saves probe calls; tick 'Ignore cache' to re-probe).")
    else:
        log("🔬 Probing what your API plan allows (a few calls)...")
        if plan_key == "free" and free_hist_sample:
            api.no_history = True           # free plan + free-source history: the API history probe would only waste calls
            log("   🆓 Free plan and free-source history available - skipping the API team-history probe.")
        else:
            team_results(api, sample["home_id"], sample["season"], date_str, tz, ttl(3 * 3600))
        caps = {"history": not api.no_history, "league_odds": None, "fixture_odds": False}
        items = fetch_league_odds(sample["league_id"], sample["season"])
        probed_leagues = {(sample["league_id"], sample["season"])}
        store_odds(items)
        if items:
            caps["league_odds"] = True
        elif api.ep_fail.get("/odds", 0) > 0 and not api.ep_ok.get("/odds"):
            caps["league_odds"] = False
        if caps["league_odds"] is False and can_spend(2):
            for c in cand[:2]:
                its = api.get("/odds", {"fixture": c["id"]}, ttl(1200), True).get("response") or []
                if its:
                    store_odds(its, c["id"])
                    caps["fixture_odds"] = True
                    break
        if caps["league_odds"] is not None:
            state_["caps_mem"] = {"plan": plan_key, "date": datetime.now(timezone.utc).isoformat(), "caps": caps}
            save_state(state_)
    out["caps"] = caps
    log(f"   team history: {'yes' if caps['history'] else 'NO (free sources / predictions last-5 form instead)'} | "
        f"odds by league: {'yes' if caps['league_odds'] else 'NO' if caps['league_odds'] is False else 'unknown'} | "
        f"odds by fixture: {'yes' if caps['fixture_odds'] else 'no'}")

    log(f"💰 Stage 1: bookmaker odds for {len(cand)} candidate matches in {len({c['league_id'] for c in cand})} competitions...")
    if caps["league_odds"] is not False:
        todo = {(c["league_id"], c["season"]) for c in cand} - probed_leagues
        with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
            for fu in as_completed([ex.submit(fetch_league_odds, lid, s_) for lid, s_ in todo]):
                try:
                    store_odds(fu.result())
                except Exception as e:
                    api.errors.append(f"odds: {e}")
    if free.enabled:                        # FREE price fallback for every candidate the API has no prices for
        n_free = 0
        for c in cand:
            if c["id"] not in odds_by:
                fo, src_ = free.odds_for(c["home"], c["away"], c["league_id"], date_str)
                if fo:
                    odds_by[c["id"]], odds_src[c["id"]] = fo, "free:" + str(src_)
                    n_free += 1
        if n_free:
            log(f"   🆓 {n_free} matches got bookmaker prices from free sources (football-data.co.uk / ESPN) instead of API-Football")
    if caps["league_odds"] is not False:
        if not odds_by and caps["league_odds"] is None and can_spend(20):
            for c in cand[1:4]:
                its = api.get("/odds", {"fixture": c["id"]}, ttl(1200), True).get("response") or []
                if its:
                    store_odds(its, c["id"])
                    caps["fixture_odds"] = True
    with_odds = [c for c in cand if c["id"] in odds_by]
    if caps["fixture_odds"] or (cfg["allow_est"] and not (caps["league_odds"] and len(with_odds) >= 10)):
        keep = list(cand)
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
        out["reason"] = ("No bookmaker odds were returned for any candidate match (API-Football or free sources). "
                         + ("API said: " + " | ".join(e[:170] for e in errs) + ". " if errs else "No error was reported - odds may not be published yet for this day. ")
                         + "Enable 'model-estimated odds' in Settings to continue with approximate prices.")
        return out

    # ── budget-aware cap on how many get deep data ──
    free_first = free.enabled
    per = ((0.6 if (free_first and cfg.get("save_calls", True)) else 1.0) + (2.0 if caps["history"] else 0.0)
           + (0.6 if caps["fixture_odds"] else 0.0))
    deep = caps["history"] and (cfg["depth"] == "Full" or (cfg["depth"] == "Auto" and budget >= 150))

    def cost(ms, dp):
        return len(ms) * (per + (0.9 if dp else 0.0)) + (len({m["league_id"] for m in ms}) if dp else 0)

    left = budget - api.calls
    n = min(cfg["max_matches"], len(keep))
    if caps["history"] or not free_first:      # with free history the per-call guard protects the quota, so no up-front trimming
        while n > 4 and cost(keep[:n], deep) > left * 0.95:
            n -= 1
        if deep and n < min(cfg["max_matches"], len(keep)) and n < 12:
            n_l = min(cfg["max_matches"], len(keep))
            while n_l > 4 and cost(keep[:n_l], False) > left * 0.95:
                n_l -= 1
            if n_l >= n + 4:
                deep, n = False, n_l
    free_cov = set()
    if free.enabled:
        free_cov = {c["id"] for c in keep if free.covers(c, date_str)}
        if not caps["history"]:
            keep.sort(key=lambda c: 0 if c["id"] in free_cov else 1)      # matches the free sources can fully cover go first (stable order inside each group)
    sl = keep[:n]
    why_cut = (f"beyond your 'max matches' setting ({cfg['max_matches']})" if n >= min(cfg["max_matches"], len(keep)) else "beyond API call budget")
    for c in keep[n:]:
        out["excluded"].append({"league": c["league"], "match": f"{c['home']} v {c['away']}", "reason": why_cut})
    out["deep"], out["budget"] = deep, budget
    leagues = {(m["league_id"], m["season"]) for m in sl}
    est_calls = cost(sl, deep)
    log(f"🎯 Stage 2: data for {len(sl)} matches | depth={'FULL' if deep else 'LITE'} | est. ≤{est_calls:.0f} more API calls "
        f"(budget left ~{left:.0f}; free sources are used first) | ~{est_calls * api.rate.interval / 60:.1f} min at your plan's rate limit")
    if free.enabled:                         # warm the free caches once, in parallel, before the worker threads read them
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda lid: free.rows_for(lid, date_str), {m["league_id"] for m in sl}))
        free.elo_table(date_str)

    # ── injuries ──
    log("🩹 Fetching injuries & suspensions...")
    inj_by = defaultdict(list)
    inj_raw = api.resp("/injuries", {"date": date_str, "timezone": tz}, ttl=ttl(3600)) if can_spend(1) else []
    for i in inj_raw:
        inj_by[((i.get("fixture") or {}).get("id"), (i.get("team") or {}).get("id"))].append(i)
    inj_known = bool(inj_raw)
    progress(0.18)

    # ── predictions + last-10 (both teams), FREE history first ──
    log("📊 Collecting predictions, team form (last 10) and H2H (free sources first)...")
    core = {}
    api_alive = {"v": True}

    def fetch_core(m):
        fh = free.team_form(m["home"], m["league_id"], date_str) if free.enabled else None
        fa = free.team_form(m["away"], m["league_id"], date_str) if free.enabled else None
        good_h, good_a = bool(fh and len(fh["results"]) >= 6), bool(fa and len(fa["results"]) >= 6)
        skip_pred = bool(free.enabled and cfg.get("save_calls", True) and good_h and good_a)      # free form + H2H + Elo/market already give an independent view
        pred = None
        if skip_pred:
            free.saved_calls += 1
        elif api_alive["v"] and can_spend(1):
            pred = parse_predictions(api.resp("/predictions", {"fixture": m["id"]}, ttl=ttl(6 * 3600)))
        tags = []

        def side(f_, good, team_id, name):
            if good:
                if caps["history"]:
                    free.saved_calls += 1
                tags.append("form")
                return f_["results"], f_
            r = team_results(api, team_id, m["season"], date_str, tz, ttl(3 * 3600)) if (api_alive["v"] and not api.no_history and can_spend(2)) else []
            if len(r) < 4 and f_:
                tags.append("form")
                return f_["results"], f_
            return r, None
        rh, used_h = side(fh, good_h, m["home_id"], m["home"])
        ra, used_a = side(fa, good_a, m["away_id"], m["away"])
        h2h = parse_h2h(pred["h2h"], m["home_id"]) if pred else []
        if len(h2h) < 3 and free.enabled:
            fh2 = free.h2h(m["home"], m["away"], m["league_id"], date_str)
            if len(fh2) > len(h2h):
                h2h = fh2
                tags.append("h2h")
        elo = free.elo_for(m["home"], m["away"], m["league_id"], m["country"], date_str) if free.enabled else None
        if elo:
            tags.append("elo")
        if str(odds_src.get(m["id"], "")).startswith("free"):
            tags.append("odds")
        return m["id"], {"pred": pred, "rh": rh, "ra": ra, "h2h": h2h, "elo": elo, "free_tags": tags, "fh": used_h, "fa": used_a}

    def viable(c):
        pr = c["pred"]
        if pr:
            H, A = pr["home"], pr["away"]
            if None not in (H["gf_h"], H["ga_h"], A["gf_a"], A["ga_a"]) and min(H["played_h"] or 0, A["played_a"] or 0) >= 3:
                return True
            if (H.get("l5_played") or 0) >= 3 and (A.get("l5_played") or 0) >= 3 and H.get("l5_gf") is not None and A.get("l5_gf") is not None:
                return True
        return len(c["rh"]) >= 4 and len(c["ra"]) >= 4

    # Probe the API on a few matches the free sources CANNOT cover, so a plan restriction is detected after a handful of calls
    # instead of burning the quota - but a failed probe only switches API fetching off; free-covered matches carry on.
    seen_lg, probe = set(), []
    for m in sl:
        if m["id"] not in free_cov and m["league_id"] not in seen_lg:
            seen_lg.add(m["league_id"])
            probe.append(m)
        if len(probe) == 4:
            break
    for m in probe:
        fid, c = fetch_core(m)
        core[fid] = c
        if sum(1 for x in core.values() if viable(x)) >= 2:
            break
    probed_viable = sum(1 for m in probe if m["id"] in core and viable(core[m["id"]]))
    if len(probe) >= 3 and probed_viable == 0:
        api_alive["v"] = False
        log(f"   ⚠️ API-Football returned no usable team data for {len(probe)} probed matches (plan restriction) - "
            f"continuing ONLY with the {len(free_cov)} matches the free sources cover; no more API calls will be wasted on the others.")
    if not any(viable(c) for c in core.values()) and not free_cov:
        out["reason"] = ("Neither API-Football nor the free sources (football-data.co.uk, international results) had team history for these matches "
                         "(see API errors below - usually a plan restriction). Stopped early to protect your daily API quota; "
                         "without team data any ticket would be a guess.")
        for m in sl:
            out["excluded"].append({"league": m["league"], "match": f"{m['home']} v {m['away']}", "reason": "no team-specific data retrievable"})
        return out
    rest = [m for m in sl if m["id"] not in core]
    done = len(core)
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
        futs = [ex.submit(fetch_core, m) for m in rest if api_alive["v"] or m["id"] in free_cov]
        for m in rest:
            if not (api_alive["v"] or m["id"] in free_cov):
                out["excluded"].append({"league": m["league"], "match": f"{m['home']} v {m['away']}", "reason": "no team data from the API (plan) or free sources"})
        for fu in as_completed(futs):
            try:
                fid, c = fu.result()
                core[fid] = c
            except Exception as e:
                api.errors.append(f"core: {e}")
            done += 1
            progress(0.18 + 0.37 * done / len(sl))
    st2 = load_state()
    lm = st2.setdefault("league_mem3", {})
    for m in sl:
        c_ = core.get(m["id"])
        if c_ is not None:
            e = lm.setdefault(str(m["league_id"]), {"t": 0, "v": 0, "d": datetime.now(timezone.utc).isoformat()})
            e["t"] += 1
            e["v"] += 1 if viable(c_) else 0
            e["d"] = datetime.now(timezone.utc).isoformat()
    save_state(st2)
    if caps["fixture_odds"]:
        log("💰 Fetching per-fixture bookmaker odds (only for matches that have usable team data)...")
        with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
            futs = {ex.submit(lambda mm: api.get("/odds", {"fixture": mm["id"]}, ttl(1200), True).get("response") or [] if can_spend(1) else [], m): m
                    for m in sl if m["id"] not in odds_by and m.get("cov_odds") is not False
                    and viable(core.get(m["id"]) or {"pred": None, "rh": [], "ra": []})}
            for fu in as_completed(futs):
                try:
                    store_odds(fu.result(), futs[fu]["id"])
                except Exception as e:
                    api.errors.append(f"fixture odds: {e}")
    if deep:
        for m in sl:
            c = core.get(m["id"])
            if c and len(c["h2h"]) < 3 and "last" not in api.blocked and can_spend(41) and not api.quota_out:
                c["h2h"] = parse_h2h(api.resp("/fixtures/headtohead", {"h2h": f"{m['home_id']}-{m['away_id']}", "last": 8}, ttl=ttl(24 * 3600)), m["home_id"])

    # ── batched history stats (corners, shots, xG, player ratings) + upcoming lineups + standings ──
    hist, upc, st_by = {}, {}, {}
    if deep:
        log("🧮 Batch-fetching statistics, player ratings & lineups (fixtures?ids=)...")
        hist_ids = set()
        for c in core.values():
            for r in (c["rh"][:5] + c["ra"][:5]):
                if isinstance(r["id"], int):           # free-source rows carry text ids and must never reach the API
                    hist_ids.add(r["id"])

        def batch(ids, ttl_s):
            ids = sorted(i for i in ids if isinstance(i, int) and i > 0)
            chunks = [ids[i:i + 20] for i in range(0, len(ids), 20)]
            res = {}
            with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
                for fu in as_completed([ex.submit(api.resp, "/fixtures", {"ids": "-".join(map(str, c_))}, ttl_s) for c_ in chunks if can_spend(1)]):
                    for f in fu.result():
                        res[f["fixture"]["id"]] = f
            return res
        hist = batch(hist_ids, ttl(5 * 86400))           # finished matches never change: cache for days
        progress(0.75)
        upc = batch([m["id"] for m in sl], ttl(600))
        with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
            futs = {ex.submit(api.resp, "/standings", {"league": lid, "season": s_}, ttl(12 * 3600)): lid for lid, s_ in leagues if can_spend(1)}
            for fu in as_completed(futs):
                try:
                    st_by[futs[fu]] = parse_standings(fu.result())
                except Exception:
                    st_by[futs[fu]] = {}
    progress(0.85)

    # ── build match objects ──
    log("🧠 Building profiles and running Monte Carlo simulations...")
    for m in sl:
        c = core.get(m["id"]) or {"pred": None, "rh": [], "ra": [], "h2h": [], "elo": None, "free_tags": [], "fh": None, "fa": None}
        hs_h = team_history_stats(m["home_id"], [r["id"] for r in c["rh"][:5] if isinstance(r["id"], int)], hist)
        hs_a = team_history_stats(m["away_id"], [r["id"] for r in c["ra"][:5] if isinstance(r["id"], int)], hist)
        if not hs_h.get("cor_n") and c.get("fh"):
            hs_h.update({k: c["fh"]["hs"][k] for k in ("cor_for", "cor_against", "cor_n", "sot")})
        if not hs_a.get("cor_n") and c.get("fa"):
            hs_a.update({k: c["fa"]["hs"][k] for k in ("cor_for", "cor_against", "cor_n", "sot")})
        lus = {((l.get("team") or {}).get("id")): l for l in (upc.get(m["id"]) or {}).get("lineups") or []}
        st_h = (st_by.get(m["league_id"]) or {}).get(m["home_id"]) or (free.standing(m["home"], m["league_id"], date_str) if free.enabled else None)
        st_a = (st_by.get(m["league_id"]) or {}).get(m["away_id"]) or (free.standing(m["away"], m["league_id"], date_str) if free.enabled else None)
        rec = make_record(m, c, hs_h, hs_a, inj_by.get((m["id"], m["home_id"])), inj_by.get((m["id"], m["away_id"])),
                          lus.get(m["home_id"]), lus.get(m["away_id"]), odds_by.get(m["id"], {}), odds_src.get(m["id"]),
                          st_h, st_a, inj_known, cfg)
        for t in set(rec["free_tags"]):
            free.used[t] += 1
        out["matches"].append(rec)
    thin = sum(1 for r in out["matches"] if r["basis"] == "prior")
    if thin:
        out["warnings"].append(f"{thin}/{len(out['matches'])} matches have NO team-specific data (plan restriction, early season or missing coverage) and are excluded from tickets.")
    if free.intl_extra:
        free.used["espn-recent-results"] = free.intl_extra
    if free.enabled and free.used:
        log(f"🆓 Free sources helped: {dict(free.used)} | ≈{free.saved_calls} API calls avoided")
    progress(0.90)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Evidence packs for the AI (three passes: SCOUT -> DESK -> AUDIT) under a per-minute token governor
# ═════════════════════════════════════════════════════════════════════════════
DESK_PROMPT = """You are a sharp sportsbook trader and quant analyst. Python gathered the data, ran a Monte Carlo per match, blended it with de-vigged bookmaker prices (Elo/API % if there are no prices) and screened every leg. YOU choose @WHAT@. Think briefly: shortlist the ~8 best legs, test them against the evidence, answer. Do not enumerate everything.

KEY: dq=data quality 0-1; R/S/S+R/E=data behind xG; NATIONAL=national team; formNd=newest result is N days old (trust Elo/market more); xG home-away; mdl=model 1X2 %; mkt=bookmaker 1/X/2 (fd=free source) then elo/API %; ppg/gf/ga=recent form, [..]=latest scores; H2H W-D-L (home view); inj n/a=unknown (a risk); TRAP=Python trap score (shown when >=25). LEGS id@odds/p=blended %/m=market %; ~=borderline (avoid unless clearly best); ^=estimated price, no market check (avoid).

RULES: 1) use ONLY leg ids shown. 2) @STRUCT@ 3) product of odds >= 3.50 (aim 3.5-4.6). 4) maximise joint probability: prefer p>=75, 65-74 only with strong evidence. @ROLE@
CHECKS: p vs m gap >8pts needs hard evidence (no evidence = trust the market). TRAP>=45: never the favourite-win leg. For each leg state the scenario that loses it. Over1.5 needs total xG>=2.6, Over2.5>=3.0, Under2.5<=2.3, BTTS-No needs a side xG<=0.9, team Over1.5 own xG>=1.9. Extra margin for unknown injuries, dq<0.6, friendlies, stale or Elo-only data. Never 4+ legs of one market type.@RELAXED@

OUTPUT: ONE JSON object only, nothing else:
{"tickets":[{"legs":[{"id":"m3.1X","why":"<=10 words","fail":"<=6 words","vs":"<=8 words, only if p and m differ >8pts"}, ...5 legs],"logic":"<=25 words","risk":"LOW|MED"}],"traps":[{"id":"m5","note":"<=10 words"}],"summary":"<=20 words"}"""

BRIEF_ADDENDUM = "\nIMPORTANT: a previous attempt ran out of space. Keep your thinking VERY short (a few lines), then output the JSON at once. why/fail <= 6 words each."

DESK_ROLES = ["This is Ticket 1: the SAFEST possible ticket.",
              "This is Ticket 2: BALANCED - strong probability plus some positive edge.",
              "This is Ticket 3: VALUE - the best price-versus-evidence legs that are still solid."]

SCOUT_PROMPT = """You are head of trading at a sharp sportsbook and a quant analyst. Python analysed @N@ matches (Monte Carlo + de-vigged market prices). There is room to study only @K@ of them in depth, and @NT@ accumulator(s) x 5 legs from separate matches are needed (at least @NEED@ usable matches).

Pick the @K@ matches you trust MOST as sources of safe accumulator legs, best first. Reject matches with thin, contradictory or trap-prone evidence: TRAP>=45 where the legs are favourite wins, dq<0.6, big favourite with unknown injuries/lineups, friendlies, model vs market gap >8pts without a reason, only ~ (borderline) legs. Keep a mix of leagues and leg types.

DATA KEY: xG home-away; mdl=model 1X2 %; mk=bookmaker 1/X/2 prices (or elo/api %); T=trap score; inj=missing starters (n/a = unknown); ROT=rotation; legs: key@odds p=blended % m=market %, ~=borderline.

OUTPUT: ONE JSON object only: {"keep":["a3","a7", ...up to @K@ ids, best first],"drop":[{"id":"a2","why":"<=8 words"}],"traps":[{"id":"a5","note":"<=14 words"}]}"""

AUDIT_PROMPT = """You are the risk officer of a sportsbook trading desk. A trader drafted @NT@ accumulator(s). BREAK each leg: use the evidence to find the most likely way it loses, then rule. Be sceptical - a leg survives only if its losing scenario is rare and the evidence supports it.

Rules the trader had: no favourite-win leg in TRAP>=45 matches; Over1.5 needs total xG>=2.6, Over2.5>=3.0, Under2.5<=2.3, BTTS-No needs a side with xG<=0.9, team Over1.5 needs own xG>=1.9; p vs m gaps >8pts need hard evidence; unknown injuries (n/a), thin data and borderline (~) legs need extra margin; distinct matches in a ticket; ticket odds >=3.50. FLAGS are Python's lint findings.

ok=true keeps a leg. ok=false needs a swap: an id from that leg's ALT list (same match) or from POOL (an unused match), safer than the old leg, with the ticket's odds product staying >=3.50. Flag a leg only for a concrete reason. GAPS are slots with no valid leg: fill them from POOL with distinct matches.

OUTPUT: ONE JSON object only: {"legs":[{"id":"m3.1X","ok":true,"risk":"LOW|MED|HIGH","issue":"<=14 words"}, ...one per leg],"swaps":[{"old":"m4.O2.5","new":"m4.O1.5"}],"fills":[{"t":2,"id":"m9.1X"}],"note":"<=30 words"}"""


def desk_prompt(ti, n_tickets, reuse, relaxed=False):
    """Prompt for ONE ticket (ticket index ti, 0-based): one focused request per ticket keeps each call inside Groq's per-request limit."""
    return (DESK_PROMPT.replace("@WHAT@", f"ticket {ti + 1} of {n_tickets}: one 5-leg accumulator")
            .replace("@STRUCT@", "Exactly 1 ticket of 5 legs from 5 different matches." + (" Earlier tickets are listed below; avoid their legs and matches wherever possible." if reuse and ti > 0 else ""))
            .replace("@ROLE@", DESK_ROLES[min(ti, 2)])
            .replace("@RELAXED@", "\n- The slate is thin so some safety thresholds were relaxed: be stricter yourself, prefer the best available legs, mark risk MED where honest." if relaxed else ""))


def scout_prompt(n, k, n_tickets, need):
    return SCOUT_PROMPT.replace("@N@", str(n)).replace("@K@", str(k)).replace("@NT@", str(n_tickets)).replace("@NEED@", str(need))


def audit_prompt(n_tickets):
    return AUDIT_PROMPT.replace("@NT@", str(n_tickets))


def _leg_txt(mid, l):
    s = f"{mid}.{l['key']}@{l['odds']:.2f}/p{_pct(l['p'])}/m{_pct(l['p_mkt'])}"
    if l.get("tier") == "B":
        s += "~"
    if l.get("src") in ("est", "derived"):
        s += "^"
    return s


def _inj_txt(pr, m):
    if not m.get("inj_known"):
        return "n/a"
    items = [f"{_short((i['name'] or '?').split(' ')[-1], 10)}{'*' if i['key'] else ''}({i['pos'] or '?'}){'?' if i['type'] != 'Missing Fixture' else ''}"
             for i in pr.get("inj", [])[:5]]
    return ",".join(items) if items else "-"


def _lu_txt(pr):
    l = pr.get("lineup")
    return "?" if not l else (l["formation"] or "ok") + ("ROT" if l["rotation"] else "")


def _side_txt(pr, lab, last=False):
    if not pr.get("n"):
        return f"{lab} n/a"
    rt = f" rt{pr['team_rating']:.1f}" if pr.get("team_rating") else ""
    lst = f" last[{','.join(pr['last'][:4])}]" if last and pr.get("last") else ""
    return f"{lab} {pr['form5']} ppg{pr['ppg5']:.1f} gf{pr['gf_w']:.1f} ga{pr['ga_w']:.1f}{rt}{lst}"


def _mkt_txt(m):
    o = m["odds"]
    if all(k in o for k in ("1", "X", "2")):
        t = "/".join(f"{o[k]['odds']:.2f}" for k in ("1", "X", "2"))
        return t + ("(fd)" if str(m.get("odds_src") or "").startswith("free") else "")
    return "n/a"


def _ext_txt(m):
    d = m.get("ext_d")
    if not d:
        return "n/a"
    return f"{'elo' if m.get('ext_src') == 'elo' or (not m.get('api_1x2') and m.get('elo')) else 'API'} " + "/".join(str(_pct(d[k])) for k in ("1", "X", "2"))


def format_match_block(mid, m, n_legs, tz, detail=3):
    """Compact evidence block. Lines carrying no information (no injuries, unknown lineups, low trap score) are omitted."""
    p = m["p"]
    ko = datetime.fromtimestamp(m["ts"], ZoneInfo(tz)).strftime("%H:%M")
    ph, pa = m["ph"], m["pa"]
    age = (m.get("form_age_days") or 0)
    tags = (f" form{age}d" if age > 30 else "") + (" NATIONAL" if (m.get("elo") or {}).get("intl") else "")
    t = m["trap"]
    legs = " ; ".join(_leg_txt(mid, l) for l in m["legs"][:n_legs])
    goals = f" | BTTS{_pct(p['BTTS_Y'])} O1.5:{_pct(p['O1.5'])} O2.5:{_pct(p['O2.5'])} U2.5:{_pct(p['U2.5'])}" if detail >= 2 else ""
    corn = f" crn{m['exp_corners']:.1f}" if detail >= 2 and m.get("corner_n", 0) >= 3 else ""
    out = [f"#{mid} {m['cat']}|{_short(m['league'], 14)} {_short(m['home'], 14)} v {_short(m['away'], 14)} {ko} dq{m['dq']} {BASIS_TAG.get(m['basis'], '?')}{tags}",
           f"xG{m['lam_h']:.2f}-{m['lam_a']:.2f} mdl{_pct(m['model_p']['1'])}/{_pct(m['model_p']['X'])}/{_pct(m['model_p']['2'])} mkt{_mkt_txt(m)} {_ext_txt(m)}{goals}{corn}"]
    if detail >= 2:
        rk = f" rk{m['st_h']['rank']}/{m['st_a']['rank']}" if m.get("st_h") and m.get("st_a") else ""
        h2 = f" | H2H{m['h2h']['n']} {m['h2h']['w']}-{m['h2h']['d']}-{m['h2h']['l']}" if m.get("h2h") else ""
        out.append(f"{_side_txt(ph, 'H', detail >= 3)} | {_side_txt(pa, 'A', detail >= 3)}{rk}{h2}")
    extra = []
    if not m.get("inj_known"):
        extra.append("inj n/a")
    else:
        hi, ai_ = _inj_txt(ph, m), _inj_txt(pa, m)
        if hi != "-" or ai_ != "-":
            extra.append(f"inj H:{hi} A:{ai_}")
    if ph.get("lineup") or pa.get("lineup"):
        extra.append(f"lu H:{_lu_txt(ph)} A:{_lu_txt(pa)}")
    if t["risk"] >= 25:
        extra.append(f"TRAP{t['risk']}: {', '.join(t['reasons'])}")
    if extra:
        out.append(" | ".join(extra))
    out.append("LEGS " + legs)
    return "\n".join(out)


def format_scout_block(aid, m, tz, n_legs=3):
    ko = datetime.fromtimestamp(m["ts"], ZoneInfo(tz)).strftime("%H:%M")
    t = m["trap"]
    if m.get("inj_known"):
        ko_h = sum(1 for i in m["ph"].get("inj", []) if i["key"] and i["type"] == "Missing Fixture")
        ko_a = sum(1 for i in m["pa"].get("inj", []) if i["key"] and i["type"] == "Missing Fixture")
        inj = f"{ko_h}/{ko_a}"
    else:
        inj = "n/a"
    rot = "ROT" if any((pr.get("lineup") or {}).get("rotation") for pr in (m["ph"], m["pa"])) else ""
    legs = " ".join(f"{l['key']}@{l['odds']:.2f}p{_pct(l['p'])}m{_pct(l['p_mkt'])}{'~' if l.get('tier') == 'B' else ''}" for l in m["legs"][:n_legs])
    trap = f"T{t['risk']}" + (f":{t['reasons'][0]}" if t["risk"] >= 30 and t["reasons"] else "")
    return (f"{aid} {m['cat']}|{_short(m['league'], 14)} {_short(m['home'], 13)} v {_short(m['away'], 13)} {ko} dq{m['dq']} {BASIS_TAG.get(m['basis'], '?')} "
            f"xG{m['lam_h']:.1f}-{m['lam_a']:.1f} mdl{_pct(m['model_p']['1'])}/{_pct(m['model_p']['X'])}/{_pct(m['model_p']['2'])} mk{_mkt_txt(m)} "
            f"{trap} inj{inj}{rot} | {legs}")


def est_tokens(text, cpt):
    return int(len(text) / cpt) + 1


def rank_matches(ms):
    usable = [m for m in ms if m["legs"]]
    usable.sort(key=lambda m: -(sum(l["rank"] for l in m["legs"][:2]) / min(2, len(m["legs"]))))
    return usable


PACK_LEVELS = [(3, 5), (3, 4), (2, 4), (2, 3), (1, 3), (1, 2)]      # (detail, legs shown per match), richest first


def build_ai_pack(matches, tz, cpt, sys_prompt, floor, reserve, ceiling, labels, extra="", start_level=0, max_n=9):
    """DESK pack for ONE ticket: up to `max_n` matches (the caller's order is kept), at the richest level of PACK_LEVELS that leaves
    `reserve` tokens of reply room (hidden reasoning counts against it). Returns (..., level_used)."""
    top = [m for m in matches if m["legs"]][:max_n]
    prompt_cap = ceiling - reserve - TOKEN_SAFETY
    sys_tok = est_tokens(sys_prompt + extra, cpt)
    floor = max(1, min(floor, len(top)))
    for lvl in range(start_level, len(PACK_LEVELS)):
        detail, n_legs = PACK_LEVELS[lvl]
        for n in range(len(top), floor - 1, -1):
            sel = top[:n]
            user = (f"CANDIDATES: {len(sel)} matches. Build the ticket from these LEGS only.\n{extra}\n"
                    + "\n\n".join(format_match_block(labels[id(m)], m, n_legs, tz, detail) for m in sel))
            if sys_tok + est_tokens(user, cpt) <= prompt_cap:
                return sel, {labels[id(m)]: m for m in sel}, user, n_legs, detail, lvl
    sel = top[:floor]
    user = "CANDIDATES:\n" + "\n\n".join(format_match_block(labels[id(m)], m, 2, tz, 1) for m in sel)
    return sel, {labels[id(m)]: m for m in sel}, user, 2, 1, len(PACK_LEVELS) - 1


def build_scout_pack(pool, tz, cpt, sys_prompt, reserve, ceiling):
    prompt_cap = ceiling - reserve - TOKEN_SAFETY
    sys_tok = est_tokens(sys_prompt, cpt)
    for n_legs in (3, 2):
        for n in range(len(pool), 5, -1):
            sel = pool[:n]
            ids = {f"a{i + 1}": m for i, m in enumerate(sel)}
            user = f"SLATE ({len(sel)} matches):\n" + "\n".join(format_scout_block(a, m, tz, n_legs) for a, m in ids.items())
            if sys_tok + est_tokens(user, cpt) <= prompt_cap:
                return sel, ids, user
    return None, {}, ""


def _audit_leg_line(ti, li, l, m, match_legs, detail, flags):
    mid = l["mid"]
    alts = [x for x in match_legs.get(mid, []) if x["key"] != l["key"]][:3]
    ev = (f"xG{m['lam_h']:.1f}-{m['lam_a']:.1f} mdl{_pct(m['model_p']['1'])}/{_pct(m['model_p']['X'])}/{_pct(m['model_p']['2'])} dq{m['dq']} "
          f"| {_side_txt(m['ph'], 'H', detail >= 2)} | {_side_txt(m['pa'], 'A', detail >= 2)} | inj H:{_inj_txt(m['ph'], m)} A:{_inj_txt(m['pa'], m)}")
    if detail >= 2:
        h2 = f" | H2H{m['h2h']['n']} {m['h2h']['w']}-{m['h2h']['d']}-{m['h2h']['l']}" if m.get("h2h") else ""
        ev += f" | lu H:{_lu_txt(m['ph'])} A:{_lu_txt(m['pa'])}{h2}"
    t = m["trap"]
    return (f" {li + 1}) {_leg_txt(mid, l)} | {_short(m['home'], 14)} v {_short(m['away'], 14)} [{_short(m['league'], 14)}] {ev} | TRAP{t['risk']}"
            f"{(':' + ','.join(t['reasons'][:2])) if t['risk'] >= 30 else ''} | ALT {' ; '.join(_leg_txt(mid, a) for a in alts) or '-'}"
            f"{' | FLAGS ' + ','.join(flags) if flags else ''}")


def build_audit_pack(tickets, match_legs, tz, cpt, sys_prompt, reserve, ceiling, per, distinct):
    prompt_cap = ceiling - reserve - TOKEN_SAFETY
    sys_tok = est_tokens(sys_prompt, cpt)
    used = {l["mid"] for t in tickets for l in t["legs"]}
    pool_mids = [mid for mid, lst in match_legs.items() if lst and (mid not in used or not distinct)]
    pool_mids.sort(key=lambda mid: -match_legs[mid][0]["rank"])
    for detail in (2, 1):
        for n_pool in (6, 4, 2):
            lines = []
            for ti, t in enumerate(tickets):
                lines.append(f"T{ti + 1} '{t['name']}' odds {_tprod(t['legs']):.2f} joint p~{_pct(_tprod(t['legs'], 'p'))}% ({len(t['legs'])}/{per} legs)")
                for li, l in enumerate(t["legs"]):
                    lines.append(_audit_leg_line(ti, li, l, l["match"], match_legs, detail, lint_leg(l)))
                if len(t["legs"]) < per:
                    lines.append(f" GAP: ticket {ti + 1} needs {per - len(t['legs'])} more leg(s)")
            pool_lines = [f" {mid} {_short(match_legs[mid][0]['match']['home'], 13)} v {_short(match_legs[mid][0]['match']['away'], 13)}: "
                          + " ; ".join(_leg_txt(mid, x) for x in match_legs[mid][:2]) for mid in pool_mids[:n_pool]]
            user = "\n".join(lines) + "\nPOOL (unused matches):\n" + ("\n".join(pool_lines) if pool_lines else " none")
            if sys_tok + est_tokens(user, cpt) <= prompt_cap:
                return user
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Groq: per-minute token governor, budgeted calls, model fallback
# ═════════════════════════════════════════════════════════════════════════════
class TpmGovernor:
    """Keeps the tokens sent to Groq inside its rolling 60-second window by sleeping until enough old usage has expired."""
    def __init__(self, window_limit, clock=time.monotonic, sleep=time.sleep):
        self.limit = int(window_limit)
        self.clock, self.sleep = clock, sleep
        self.events: List[list] = []

    def _prune(self, now):
        self.events = [e for e in self.events if now - e[0] < 60.0]

    def used(self):
        self._prune(self.clock())
        return sum(e[1] for e in self.events)

    def wait_for(self, tokens, log_fn=None):
        waited = 0.0
        tokens = min(int(tokens), self.limit)
        while True:
            now = self.clock()
            self._prune(now)
            used = sum(e[1] for e in self.events)
            if used + tokens <= self.limit:
                return waited
            need_free, acc, release = used + tokens - self.limit, 0, None
            for t, k in self.events:
                acc += k
                if acc >= need_free:
                    release = t + 60.0
                    break
            wait = min(65.0, max(1.0, (release if release is not None else now + 60.0) - now + 0.5))
            if log_fn:
                log_fn(f"   ⏳ Groq allows ~{self.limit + WINDOW_MARGIN} tokens/minute: waiting {wait:.0f}s so the next AI pass fits the window...")
            self.sleep(wait)
            waited += wait

    def charge(self, tokens):
        e = [self.clock(), int(tokens)]
        self.events.append(e)
        return e

    def settle(self, handle, actual):
        handle[1] = max(0, int(actual))


class TokenBudget:
    def __init__(self, total):
        self.total, self.used = total, 0

    @property
    def remaining(self):
        return self.total - self.used


def _salvage_json(text):
    """Recover a truncated JSON object: cut at the last complete element and close the open brackets."""
    t = text[text.find("{"):] if "{" in text else ""
    if not t:
        return None
    stack, in_str, esc, cuts = [], False, False, []
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            cuts.append((i, "".join(reversed(stack))))
    for i, closers in reversed(cuts[-400:]):
        cand = re.sub(r",\s*$", "", t[: i + 1]) + closers
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", cand))
        except Exception:
            continue
    return None


def _parse_json_ex(content):
    """Returns (object, salvaged_flag)."""
    c = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.I)
    c = re.sub(r"\s*```$", "", c).strip()
    for cand in (c, c[c.find("{"): c.rfind("}") + 1] if "{" in c else ""):
        if cand:
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", cand)), False
            except Exception:
                pass
    return _salvage_json(c), True


def _parse_json(content):
    return _parse_json_ex(content)[0]


def _retry_after(text):
    m = re.search(r"try again in ([0-9hms.]+)", text or "")
    if not m:
        return None
    sec = 0.0
    for val, unit in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1)):
        v = float(val)
        sec += v / 1000 if unit == "ms" else v * 3600 if unit == "h" else v * 60 if unit == "m" else v
    return sec


def _dead_models():
    try:
        d = load_state().get("dead_models") or {}
        return {m for m, t in d.items() if time.time() - float(t) < 7 * 86400}
    except Exception:
        return set()


def _mark_dead(model):
    try:
        st_ = load_state()
        st_.setdefault("dead_models", {})[model] = time.time()
        save_state(st_)
    except Exception:
        pass


def call_groq_budgeted(system_prompt, user_prompt, budget, cpt, effort=None, gov=None, ceiling=None, expect="tickets", log_fn=None, label="desk"):
    """One Groq request inside the per-request ceiling and the run-level token cap.
    - the TpmGovernor spaces requests over the minute;
    - 413 'too large' replies are used to MEASURE the real prompt size and shrink the reply room (or report PROMPT_TOO_LARGE so the
      caller can rebuild a smaller prompt);
    - 400 'Failed to validate JSON' replies carry the model's partial output: it is salvaged, or the request is retried without JSON mode;
    - a model that answers 404 is remembered as unavailable and skipped on later runs."""
    api_key = get_secret("GROQ_API_KEY", "").strip()
    log = []
    info = {"attempts": log, "model": None, "prompt_tokens": 0, "completion_tokens": 0, "label": label, "waited": 0.0, "fallback": False}
    if not api_key:
        log.append({"model": "-", "status": "GROQ_API_KEY not set"})
        info["status"] = "MISSING_KEY"
        return None, info
    ceiling = ceiling or (GROQ_TPM_DEFAULT - TPM_MARGIN)
    chars = len(system_prompt + user_prompt)
    est_prompt = est_tokens(system_prompt + user_prompt, cpt)
    too_large = False
    p_real_max = 0
    valid = (lambda p: isinstance(p, dict) and isinstance(p.get("legs"), list)) if expect == "legs" else (lambda p: isinstance(p, dict) and bool(p.get(expect)))
    dead = _dead_models()
    models = [m for m in GROQ_MODELS if m not in dead] or GROQ_MODELS[:2]
    for model in models:
        cfg = GROQ_MODEL_CONFIG.get(model, GROQ_DEFAULT_CFG)
        eff = effort or cfg.get("reasoning_effort")
        upper = cfg["max_completion_tokens"]      # give the reply ALL the room the request limit allows (unused room costs nothing)
        limit = ceiling                              # per-request token limit; refined from error messages
        p_est = int(est_prompt * 1.03) + 40          # a 413 (rejected before any processing, costs no tokens) corrects an under-estimate; over-padding would steal reply room
        cap_max, json_mode = None, True
        for attempt in range(4):
            cap = min(upper, budget.remaining - p_est - 60, limit - p_est - 60)
            if cap_max is not None:
                cap = min(cap, cap_max)
            if cap < MIN_COMPLETION:
                budget_short = budget.remaining - p_est - 60 < MIN_COMPLETION
                log.append({"model": model, "status": ("SKIPPED_BUDGET(run token cap used up)" if budget_short else f"PROMPT_TOO_LARGE(prompt~{p_est}, reply room={cap})")})
                if budget_short:
                    info["status"] = "TOKEN_CAP"
                    return None, info
                if p_real_max:                              # the prompt itself is the problem, so another model would fail the same way
                    info["actual_cpt"] = chars / p_real_max
                    info["status"] = "PROMPT_TOO_LARGE"
                    return None, info
                too_large = True
                break
            handle = None
            if gov:
                info["waited"] += gov.wait_for(p_est + cap, log_fn)
                handle = gov.charge(p_est + cap)
            payload = {"model": model, "temperature": 0.2, "max_completion_tokens": int(cap),
                       "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            if cfg.get("supports_reasoning_effort") and eff:
                payload["reasoning_effort"] = eff if model.startswith("openai/") or eff == "none" else "none"
                if model.startswith("openai/"):
                    payload["include_reasoning"] = False
            try:
                r = requests.post(GROQ_API_URL, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=GROQ_TIMEOUT)
            except Exception as e:
                if handle:
                    gov.settle(handle, 0)
                log.append({"model": model, "status": f"NETWORK {str(e)[:80]}"})
                time.sleep(2)
                continue
            if r.status_code == 200:
                d = r.json()
                u = d.get("usage") or {}
                pt, ct = u.get("prompt_tokens", est_prompt), u.get("completion_tokens", 0)
                total = u.get("total_tokens", pt + ct)
                budget.used += total
                if handle:
                    gov.settle(handle, total)
                info["prompt_tokens"] += pt
                info["completion_tokens"] += ct
                info["actual_cpt"] = chars / max(1, pt)
                ch = (d.get("choices") or [{}])[0]
                content = (ch.get("message") or {}).get("content") or ""
                parsed, salvaged = _parse_json_ex(content) if content else (None, False)
                if valid(parsed):
                    log.append({"model": model, "status": "SUCCESS" + (" (reply hit the token cap - recovered the complete part)" if salvaged else "")})
                    info.update({"salvaged": salvaged, "model": model, "status": "SUCCESS", "fallback": model != GROQ_MODELS[0]})
                    return parsed, info
                log.append({"model": model, "status": f"BAD_OUTPUT finish={ch.get('finish_reason')} (completion {ct} of room {cap})"})
                if ch.get("finish_reason") == "length":
                    # the model spent its whole reply room (hidden reasoning included) before finishing. A different model or a repeat
                    # of the same request would burn the same tokens again, so stop here and let the caller shrink the request.
                    info["status"] = "TRUNCATED"
                    return None, info
                break
            body = (r.text or "")[:300]
            low = body.lower()
            if r.status_code == 413 or "too large" in low:
                if handle:
                    gov.settle(handle, 0)
                lim, req = re.search(r"Limit (\d+)", body), re.search(r"Requested (\d+)", body)
                if lim and req:
                    limit = min(limit, int(lim.group(1)) - 150)
                    info["learned_limit"] = int(lim.group(1))
                    excess = int(req.group(1)) - int(lim.group(1))
                    p_real = max(1, int(req.group(1)) - int(cap))
                    p_real_max = max(p_real_max, p_real)
                    info["actual_cpt"] = chars / p_real_max
                    cap_max = int(cap) - max(0, excess) - 150          # shrink the reply room by exactly what was too much
                    p_est = max(p_est, p_real + 20)
                    log.append({"model": model, "status": f"HTTP_413 limit {lim.group(1)}, requested {req.group(1)} -> real prompt ~{p_real} tokens, reply room now {max(cap_max, 0)}"})
                else:
                    p_est = int(p_est * 1.2)
                    log.append({"model": model, "status": f"HTTP_413 {body[:120]}"})
                continue
            if r.status_code == 429:
                wait = _retry_after(body)
                lim = re.search(r"Limit (\d+)", body)
                if lim and "per minute" in low:
                    info["learned_limit"] = int(lim.group(1))
                if handle:
                    gov.settle(handle, p_est if "per minute" in low else 0)   # a daily-limit refusal consumed no window tokens
                if "per minute" in low and wait is not None and wait <= 40 and attempt < 3:
                    log.append({"model": model, "status": f"RATE_LIMIT_429 per-minute, waiting {wait:.0f}s"})
                    time.sleep(wait + 0.5)
                    continue
                log.append({"model": model, "status": f"RATE_LIMIT_429 {body[:140]}"})
                break
            if handle:
                gov.settle(handle, 0)
            if r.status_code == 400 and "failed_generation" in (r.text or ""):
                fg = ""
                try:
                    fg = (r.json().get("error") or {}).get("failed_generation") or ""
                except Exception:
                    pass
                parsed, salvaged = _parse_json_ex(fg) if fg else (None, True)
                if valid(parsed):                                  # the model's own partial JSON still contains complete tickets/verdicts
                    budget.used += p_est + int(len(fg) / 3)
                    log.append({"model": model, "status": "SUCCESS (recovered from the model's partially generated JSON)"})
                    info.update({"salvaged": True, "model": model, "status": "SUCCESS", "fallback": model != GROQ_MODELS[0]})
                    return parsed, info
                log.append({"model": model, "status": "HTTP_400 JSON validation failed (reply cut off before the JSON was complete)"})
                info["status"] = "TRUNCATED"                      # same cause as finish=length: do NOT repeat the identical request
                return None, info
            if r.status_code == 404:
                _mark_dead(model)
                log.append({"model": model, "status": "HTTP_404 model not available on this account - skipped from now on"})
                break
            if r.status_code in NON_RETRYABLE:
                log.append({"model": model, "status": f"HTTP_{r.status_code} {body[:140]}"})
                break
            log.append({"model": model, "status": f"HTTP_{r.status_code} {body[:100]} (retrying)"})
            time.sleep(2.5 * (attempt + 1))
    if p_real_max:
        info["actual_cpt"] = chars / p_real_max
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
        div = -0.06 * sum(max(0, c - 2) for c in fam.values()) - 0.05 * sum(1 for l in legs if l.get("tier") == "B")
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
        legs[best[0]] = dict(best[1], why="Python swap: AI ticket was below the 3.5 minimum odds", swapped=True, by="PY")
        fixed = True
    return legs, fixed





def python_only_tickets(match_legs, n_tickets=N_TICKETS):
    tks = optimize_tickets(match_legs, n_tickets=n_tickets)
    out = []
    for i, legs in enumerate(tks):
        for l in legs:
            l["why"] = f"p{_pct(l['p'])}% odds {l['odds']:.2f}"
            l["by"] = "PY"
        out.append({"name": TICKET_LABELS[i], "legs": legs, "risk": "", "repaired": False, "odds": _tprod(legs),
                    "p_joint": _tprod(legs, "p"), "valid": _tprod(legs) >= MIN_TICKET_ODDS, "ai_kept": 0,
                    "logic": "Deterministic Python optimiser (AI unavailable): maximises joint probability with trap/data-quality penalties and odds >= 3.5."})
    return out


HARD_FLAGS = {"trap-fav", "xg-rule"}


def lint_leg(l):
    """Checks a chosen leg against the SAME rules the AI was given. Python never invents a view; it only reports breaches."""
    m, key = l["match"], l["key"]
    flags = []
    tot, lh, la = m["lam_h"] + m["lam_a"], m["lam_h"], m["lam_a"]
    fav, risk = m["trap"]["fav"], m["trap"]["risk"]
    if fav and key == fav and risk >= 45:
        flags.append("trap-fav")
    if ((key == "O1.5" and tot < 2.6) or (key == "O2.5" and tot < 3.0) or (key == "U2.5" and tot > 2.3)
            or (key == "BTTS_N" and min(lh, la) > 0.9) or (key == "H_O1.5" and lh < 1.9) or (key == "A_O1.5" and la < 1.9)):
        flags.append("xg-rule")
    if key.startswith("C_") and m["dq"] < 0.6:
        flags.append("corners-lowdq")
    if str(l.get("by", "")).startswith("AI") and abs(l["p"] - l["p_mkt"]) > 0.08 and not l.get("vs"):
        flags.append("gap-unexplained")
    if l.get("tier") == "B":
        flags.append("borderline")
    if m["dq"] < 0.55:
        flags.append("thin-data")
    if not m.get("inj_known") and fav and key == fav:
        flags.append("inj-unknown")
    return flags


def finish_ticket(t, per=LEGS_PER_TICKET):
    legs = t["legs"]
    t["odds"], t["p_joint"] = _tprod(legs), _tprod(legs, "p")
    t["valid"] = len(legs) == per and t["odds"] >= MIN_TICKET_ODDS
    t["ai_kept"] = sum(1 for l in legs if str(l.get("by", "")).startswith("AI"))
    t["py_legs"] = sum(1 for l in legs if l.get("by") == "PY")
    fam = defaultdict(int)
    for l in legs:
        l["flags"] = lint_leg(l)
        fam[l["fam"]] += 1
    t["stacked"] = any(c >= 4 for c in fam.values())
    return t


def parse_ai_tickets(ai, leg_index, n_tickets=N_TICKETS, per=LEGS_PER_TICKET, distinct=True):
    """Keeps ONLY what the AI chose and what is valid (real id, distinct matches). No Python fill here."""
    raw = (ai or {}).get("tickets") or []
    used, tickets = set(), []
    for ti in range(n_tickets):
        t = raw[ti] if ti < len(raw) and isinstance(raw[ti], dict) else {}
        legs, repaired = [], False
        for it in t.get("legs") or []:
            d = it if isinstance(it, dict) else {"id": str(it)}
            lg = leg_index.get(str(d.get("id") or "").strip().lower())
            if not lg or lg["mid"] in used or lg["mid"] in {l["mid"] for l in legs}:
                repaired = True
                continue
            legs.append(dict(lg, why=d.get("why", ""), fail=d.get("fail", ""), vs=d.get("vs", ""), by="AI"))
        if len(legs) > per:
            legs, repaired = legs[:per], True
        if distinct:
            used |= {l["mid"] for l in legs}
        tickets.append({"name": t.get("name") or TICKET_LABELS[min(ti, 2)], "legs": legs, "logic": t.get("logic", ""),
                        "risk": t.get("risk", ""), "repaired": repaired, "audit_swaps": 0})
    return tickets


def apply_audit(tickets, aud, leg_index, match_legs, per=LEGS_PER_TICKET, distinct=True):
    """Applies the risk officer's verdicts: annotate legs, execute valid swaps, fill gaps. Returns a list of human-readable changes."""
    changes = []
    verdict = {}
    for v in (aud or {}).get("legs") or []:
        if isinstance(v, dict) and v.get("id"):
            verdict[str(v["id"]).strip().lower()] = v
    for t in tickets:
        for l in t["legs"]:
            v = verdict.get(l["id"].lower())
            if v:
                l["audit"] = {"ok": bool(v.get("ok", True)), "risk": str(v.get("risk", "") or "")[:6], "issue": str(v.get("issue", "") or "")[:140]}

    def others(t, leg=None):
        return {x["mid"] for x in t["legs"] if x is not leg}

    def taken_elsewhere(t):
        return {x["mid"] for tt in tickets if tt is not t for x in tt["legs"]} if distinct else set()

    for sw in (aud or {}).get("swaps") or []:
        if not isinstance(sw, dict):
            continue
        old, new = str(sw.get("old") or "").strip().lower(), leg_index.get(str(sw.get("new") or "").strip().lower())
        if not new:
            continue
        for t in tickets:
            for i, l in enumerate(t["legs"]):
                if l["id"].lower() != old or l["id"].lower() == new["id"].lower():
                    continue
                if new["mid"] != l["mid"] and (new["mid"] in others(t, l) or new["mid"] in taken_elsewhere(t)):
                    continue
                cand = dict(new, why=(l.get("why") or "") + " | audit swap", fail=l.get("fail", ""), vs="", by="AI2",
                            audit={"ok": True, "risk": "", "issue": (l.get("audit") or {}).get("issue", "")})
                trial = t["legs"][:i] + [cand] + t["legs"][i + 1:]
                if _tprod(trial) < MIN_TICKET_ODDS <= _tprod(t["legs"]):
                    continue
                t["legs"] = trial
                t["audit_swaps"] = t.get("audit_swaps", 0) + 1
                changes.append(f"{l['id']} -> {new['id']}")
                break
    for fl in (aud or {}).get("fills") or []:
        if not isinstance(fl, dict):
            continue
        try:
            ti = int(fl.get("t")) - 1
        except Exception:
            continue
        new = leg_index.get(str(fl.get("id") or "").strip().lower())
        if not new or not (0 <= ti < len(tickets)):
            continue
        t = tickets[ti]
        if len(t["legs"]) >= per or new["mid"] in others(t) or new["mid"] in taken_elsewhere(t):
            continue
        t["legs"].append(dict(new, why="audit gap-fill", fail="", vs="", by="AI2", audit={"ok": True, "risk": "", "issue": ""}))
        changes.append(f"gap-fill T{ti + 1}: {new['id']}")
    return changes


def py_fix_breaches(tickets, match_legs):
    """Safety net: a leg that breaks one of the AI's OWN hard rules is replaced by the best clean alternative from the same match."""
    fixed = []
    for t in tickets:
        for i, l in enumerate(t["legs"]):
            hard = [f for f in lint_leg(l) if f in HARD_FLAGS]
            if not hard:
                continue
            alts = [x for x in match_legs.get(l["mid"], []) if x["key"] != l["key"] and not [f for f in lint_leg(x) if f in HARD_FLAGS]
                    and x["tier"] == "S"]
            if not alts:
                continue
            best = max(alts, key=lambda x: x["p_adj"])
            trial = t["legs"][:i] + [dict(best, why=f"Python rule-fix: '{l['label']}' broke rule {hard[0]}", fail="", vs="", by="PY")] + t["legs"][i + 1:]
            if _tprod(trial) < MIN_TICKET_ODDS <= _tprod(t["legs"]):
                continue
            t["legs"] = trial
            t["repaired"] = True
            fixed.append(f"{l['id']} -> {best['id']} ({hard[0]})")
    return fixed


def complete_tickets(tickets, match_legs, per=LEGS_PER_TICKET, distinct=True):
    """Python completes only what is still missing, and labels every leg it supplies (by='PY')."""
    fallback = None
    n = len(tickets)
    for ti, t in enumerate(tickets):
        legs = t["legs"]
        taken_other = {l["mid"] for tj, tk in enumerate(tickets) if tj != ti for l in tk["legs"]} if distinct else set()
        if len(legs) < per:
            t["repaired"] = True
            if fallback is None:
                fallback = optimize_tickets(match_legs, n_tickets=max(1, n))
            pool_ = [l for tk in fallback for l in tk] + [dict(x, mid=k) for k, v in match_legs.items() for x in v[:3]]
            while len(legs) < per:
                cur_m = {x["mid"] for x in legs}
                avail = [c for c in pool_ if c["mid"] not in taken_other and c["mid"] not in cur_m]
                if not avail:
                    break
                cur, need = max(_tprod(legs), 1.0), per - len(legs)
                target = (MIN_TICKET_ODDS * 1.03 / cur) ** (1 / need) if cur < MIN_TICKET_ODDS * 1.03 else 1.15
                c = max(avail, key=lambda x: x["rank"] - 0.4 * max(0.0, math.log(x["odds"] / (target * 1.15))))
                legs.append(dict(c, why="Python fill-in (AI leg missing/invalid/duplicate)", fail="", vs="", by="PY"))
        legs, up = enforce_min_odds(legs, match_legs, taken_other)
        t["legs"] = legs
        t["repaired"] = bool(t.get("repaired") or up)
        finish_ticket(t, per)
    return tickets


def finalize_tickets(ai, leg_index, match_legs, n_tickets=N_TICKETS, per=LEGS_PER_TICKET, distinct=True):
    return complete_tickets(parse_ai_tickets(ai, leg_index, n_tickets, per, distinct), match_legs, per, distinct)


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





def build_aifail_message(res, tz):
    return (f"⚽ <b>Der-AI Football Quant Desk</b>\n📅 {_esc(res['date'])}\n\n🤖 <b>AI analysis did not complete - no tickets sent.</b>\n"
            f"{_esc(res.get('ai_note') or 'See the app for details.')[:400]}\n\n"
            "The data was collected successfully. Open the app and press 'Re-run AI analysis' - it reuses the data and costs no API-Football calls.")


def build_nobet_message(res, tz):
    reasons = defaultdict(int)
    for e in res.get("excluded", []):
        reasons[e["reason"]] += 1
    lines = "\n".join(f"• {_esc(k)}: {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:6])
    return (f"⚽ <b>Der-AI Football Quant Desk</b>\n📅 {_esc(res['date'])}\n\n🛑 <b>NO BET TODAY</b>\n{_esc(res.get('reason'))}"
            + (f"\n\n<b>Why matches were excluded</b>\n{lines}" if lines else "")
            + f"\n\nAPI calls used: {res.get('api_calls', 0)}. No tickets are better than tickets built on guesses.")


def _flag_txt(l):
    show = [f for f in (l.get("flags") or []) if f in HARD_FLAGS or f in ("gap-unexplained", "thin-data", "inj-unknown", "corners-lowdq")]
    return ("⚠ " + ",".join(show)) if show else ""


def _by_txt(l):
    b = str(l.get("by") or "")
    return "🤖✓ AI+audit" if b == "AI2" else "🤖 AI" if b == "AI" else "🐍 Python" if b == "PY" else ""


def build_telegram_message(res, tz):
    nums = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    L = [f"⚽ <b>Der-AI Football Quant Desk</b>\n📅 {_esc(res['date'])} | {res['n_analysed']} matches analysed | {res['n_ai']} studied by the AI desk"]
    if res.get("fallback_model"):
        L.append("⚠️ <b>A smaller fallback AI model took part</b> (the top model was rate-limited). Treat this slate with extra caution or re-run later.")
    if res.get("relaxed"):
        L.append("⚠️ <b>Thin slate - safety thresholds relaxed:</b> " + _esc("; ".join(res["relaxed"])) + ". Stakes should be smaller than usual.")
    if res.get("est_mode"):
        L.append("⚠️ <b>Approximate odds:</b> bookmaker prices were unavailable for some legs, so odds are model-estimated. Confirm the price on your bookmaker.")
    for t in res["tickets"]:
        ok = "✅" if t["valid"] else "⚠️"
        s = (f"🎟 <b>{_esc(t['name'])}</b> {ok}\nOdds <b>{t['odds']:.2f}</b> | joint prob ~{_pct(t['p_joint'])}% | risk {_esc(t.get('risk') or '-')} | evidence {t.get('grade', '?')}"
             f" | 🤖 AI legs {t.get('ai_kept', 0)}/{len(t['legs'])}" + (f" | 🐍 Python legs {t.get('py_legs', 0)}" if t.get("py_legs") else ""))
        for i, l in enumerate(t["legs"]):
            m = l["match"]
            ko = datetime.fromtimestamp(m["ts"], ZoneInfo(tz)).strftime("%H:%M")
            px = ""
            if l.get("src") == "est":
                px = " est.odds"
            elif l.get("src") == "derived":
                px = " derived odds"
            s += (f"\n{nums[i]} <i>{_esc(_short(m['league'], 22))}</i> {ko}\n   {_esc(m['home'])} v {_esc(m['away'])}\n"
                  f"   ➜ <b>{_esc(l['label'])}</b> @{l['odds']:.2f}{' (books ' + format(l['odds_lo'], '.2f') + '-' + format(l['odds_hi'], '.2f') + ')' if l.get('odds_lo') else ''}"
                  f" (p {_pct(l['p'])}%{px}) {_by_txt(l)}")
            if l.get("why"):
                s += f"\n   💡 {_esc(l['why'])}"
            if l.get("fail"):
                s += f"\n   🧯 loses if: {_esc(l['fail'])}"
            au = l.get("audit") or {}
            if au.get("issue") and (not au.get("ok", True) or au.get("risk") in ("MED", "HIGH")):
                s += f"\n   🔎 audit: {_esc(au['issue'])}"
            ft = _flag_txt(l)
            if ft:
                s += f"\n   {_esc(ft)}"
        if t.get("logic"):
            s += f"\n\n🧠 {_esc(t['logic'])}"
        if t.get("repaired"):
            s += "\n🔧 Python completed/adjusted part of this ticket (🐍 legs): AI leg missing/invalid, rule breach or odds below 3.5."
        L.append(s)
    ai = res.get("ai") or {}
    traps = [f"• {_esc(res['id_map'][x['id']]['home'])} v {_esc(res['id_map'][x['id']]['away'])}: {_esc(x.get('note'))}"
             for x in (ai.get("traps") or [])[:4] if x.get("id") in res["id_map"]]
    traps += [f"• {_esc(x['match'])}: {_esc(x.get('note'))}" for x in ((res.get("scout") or {}).get("traps") or [])[:3]]
    if traps:
        L.append("🪤 <b>Trap watch</b>\n" + "\n".join(traps[:6]))
    if ai.get("avoid"):
        L.append("🚫 <b>Avoid</b>\n" + "\n".join(f"• {_esc(res['id_map'][x['id']]['home'])} v {_esc(res['id_map'][x['id']]['away'])}: {_esc(x.get('why'))}"
                                                 for x in ai["avoid"][:4] if x.get("id") in res["id_map"]))
    if ai.get("audit_note"):
        L.append(f"🕵️ <b>Risk officer</b>\n{_esc(ai['audit_note'])}")
    if ai.get("summary"):
        L.append(f"📝 <b>Slate read</b>\n{_esc(ai['summary'])}")
    if not res.get("model"):
        L.append("ℹ️ <b>AI reasoning was not applied</b> - these tickets come from the Python Monte Carlo optimiser only (🐍)."
                 + (f" ({_esc(res.get('ai_note'))[:180]})" if res.get("ai_note") else ""))
    ps = "/".join(p["pass"].split(" ")[0] for p in res.get("passes", []) if p.get("status") == "SUCCESS") or "-"
    foot = (f"🤖 {_esc(res.get('model') or 'Python optimiser')} | passes {ps} | AI tokens {res.get('tokens', 0)} | "
            f"API calls {res.get('api_calls', 0)}" + (f" | 🆓 free sources used" if (res.get("free") or {}).get("used") else "")
            + " | ⚠️ Odds are bookmaker snapshots; betting carries risk.")
    L.append(foot)
    return "\n\n".join(L)




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


def settle_history(api, state, free=None):
    """Settles the ledger. Goal/HT/corner results come from football-data.co.uk first (free); API-Football only for what is left."""
    free = free or FreeData(enabled=True)
    free.target_date = datetime.now(timezone.utc).date().isoformat()
    pend = [(e, l) for e in state["history"] if e.get("status") != "SETTLED" for t in e["tickets"] for l in t["legs"] if l.get("result") is None]
    for e, l in pend:
        home, away = l.get("home"), l.get("away")
        if not home and " v " in (l.get("match") or ""):
            home, away = l["match"].split(" v ", 1)
        if home and away and l.get("league_id") in FD_ALL:
            f = free.result_for(home, away, l["league_id"], e["date"])
            if f:
                l["result"] = evaluate_leg(l["key"], f)
    ids = sorted({l["fid"] for e in state["history"] if e.get("status") != "SETTLED" for t in e["tickets"] for l in t["legs"]
                  if l.get("result") is None and isinstance(l.get("fid"), int) and l["fid"] > 0})
    fx = {}
    if not api.offline:
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
    free = FreeData(enabled=bool(cfg.get("use_free", True)))
    if not api_key and not free.enabled:
        return {"error": "API_FOOTBALL_KEY is not set (get one at https://dashboard.api-football.com) and free sources are switched off."}
    api = ApiFootball(api_key or "", per_minute=10)
    stt = api.status()
    plan = (stt.get("plan") or "Free")
    per_min = 10 if str(plan).lower() == "free" else 280
    api.rate = RateLimiter(cfg["rate_override"] or per_min)
    cfg = dict(cfg, workers=3 if per_min <= 10 else 8)
    log(f"🔑 API-Football plan: {plan if api_key else 'no key (free sources only)'} | used {stt.get('used')}/{stt.get('limit')} today")

    data = collect_day(api, cfg, log, progress, free)
    matches = data["matches"]

    def result(**kw):
        base = {"date": cfg["date"], "tickets": [], "ai": None, "model": None, "info": {"attempts": []}, "tokens": 0,
                "n_analysed": len(matches), "n_ai": 0, "matches": matches, "id_map": {}, "user_prompt": "",
                "api_calls": api.calls, "cache_hits": api.cache_hits, "warnings": data["warnings"], "est_mode": False,
                "cat_counts": data["cat_counts"], "plan": data["plan"], "deep": data.get("deep"),
                "api_errors": api.error_summary(), "seconds": round(time.time() - t0, 1),
                "excluded": data["excluded"], "no_bet": False, "reason": "", "caps": data.get("caps"), "ai_failed": False,
                "python_draft": None, "ai_note": "", "cfg": cfg, "quota": (api.remaining, api.limit),
                "free": free.summary(), "relaxed": [], "passes": [], "wasted_tokens": 0, "fallback_model": False, "scout": None, "audit_changes": [], "rule_fixes": []}
        base.update(kw)
        return base

    if not matches:
        return result(no_bet=True, reason=data.get("reason") or "No matches could be analysed.")

    # ── legs + evidence gate, with an automatic relaxation ladder so the slate always yields tickets if at all possible ──
    def run_gate(c):
        el = []
        for m in matches:
            m["legs"] = build_legs(m, allow_est=c["allow_est"], wide=c.get("wide_menu", True))
            why = gate_reasons(m, c)
            m["eligible"], m["excl"] = (not why), "; ".join(why)
            if not why:
                el.append(m)
        return el

    cur = dict(cfg)
    elig = run_gate(cur)
    need_all, relaxed = N_TICKETS * LEGS_PER_TICKET, []
    if cfg.get("auto_relax", True) and len(elig) < need_all:
        for label, upd in (("minimum data quality lowered to 0.40", {"min_dq": min(cfg["min_dq"], 0.40)}),
                           ("matches with only borderline legs allowed", {"allow_b_only": True}),
                           ("model-estimated odds allowed for matches without prices", {"allow_est": True})):
            if len(elig) >= need_all:
                break
            trial = dict(cur, **upd)
            new = run_gate(trial)
            if len(new) > len(elig):
                cur, elig = trial, new
                relaxed.append(label)
            else:
                elig = run_gate(cur)
        if len(elig) < need_all and len(elig) >= 8 and not cur.get("allow_reuse"):      # with <8 matches reused tickets would be near-copies
            cur["allow_reuse"] = True
            relaxed.append("the same match may appear in more than one ticket")
    cur["relaxed_notes"] = bool(relaxed)
    cfg = dict(cur, relaxed=relaxed)
    for m in matches:
        if not m["eligible"]:
            data["excluded"].append({"league": m["league"], "match": f"{m['home']} v {m['away']}", "reason": m["excl"]})
    if relaxed:
        data["warnings"].append("The slate was thin, so safety thresholds were relaxed step by step: " + "; ".join(relaxed) + ". Tickets are graded lower and the AI was told to be stricter.")
    log(f"🛡 Evidence gate: {len(elig)}/{len(matches)} matches qualify" + (f" (after relaxing: {'; '.join(relaxed)})" if relaxed else ""))
    n_t = plan_ticket_count(len(elig), cfg)
    if n_t == 0:
        tip = (f" Only ~{data.get('budget')} API calls were available this run - retry after the daily quota resets (00:00 UTC) or raise the limit."
               if (data.get("budget") or 999) < 60 else "")
        return result(no_bet=True, cfg=cfg, relaxed=relaxed,
                      reason=f"Only {len(elig)} match(es) passed the evidence gate even after relaxing; a 5-leg ticket needs at least 5. "
                             "Betting on thin data would just be guessing, so nothing is recommended." + tip)
    if n_t < N_TICKETS:
        data["warnings"].append(f"Only {len(elig)} matches qualified, so {n_t} ticket(s) were built instead of {N_TICKETS} (no match is reused).")

    stage = run_ai_stage(elig, n_t, cfg, log, progress)
    data["warnings"] += stage["warnings"]
    res = result(tickets=stage["tickets"], ai=stage["ai"], model=stage["model"], info=stage["info"], tokens=stage["tokens"], n_ai=len(stage["sel"]),
                 id_map=stage["id_map"], user_prompt=stage["user_prompt"], est_mode=stage["est_mode"], ai_note=stage["ai_note"],
                 ai_failed=stage["ai_failed"], python_draft=stage["python_draft"], cfg=cfg, relaxed=relaxed, passes=stage["passes"],
                 fallback_model=stage["fallback_model"], scout=stage["scout"], audit_changes=stage["audit_changes"], rule_fixes=stage["rule_fixes"],
                 wasted_tokens=stage["wasted_tokens"])
    if stage["tickets"]:
        save_tickets_to_ledger(cfg["date"], stage["tickets"])
    return res


def plan_ticket_count(n_elig, cfg):
    n_t = min(N_TICKETS, n_elig // LEGS_PER_TICKET)
    if cfg.get("allow_reuse") and n_elig >= LEGS_PER_TICKET:
        n_t = N_TICKETS
    return n_t


def save_tickets_to_ledger(date_str, tickets):
    state = load_state()
    state.setdefault("history", []).append({
        "created": datetime.now(timezone.utc).isoformat(), "date": date_str, "status": "PENDING",
        "tickets": [{"name": t["name"], "odds": round(t["odds"], 2), "outcome": "PENDING", "grade": t.get("grade"),
                     "legs": [{"fid": l["fid"], "key": l["key"], "label": l["label"], "odds": round(l["odds"], 2),
                               "p": round(l["p"], 3), "by": l.get("by"), "match": f"{l['match']['home']} v {l['match']['away']}",
                               "home": l["match"]["home"], "away": l["match"]["away"], "league_id": l["match"].get("league_id"), "result": None}
                              for l in t["legs"]]} for t in tickets]})
    state["history"] = state["history"][-60:]
    save_state(state)


def run_ai_stage(elig, n_t, cfg, log, progress=lambda x: None):
    """AI-first pipeline: SCOUT (orders the matches) -> one DESK call PER TICKET -> AUDIT, all paced by the per-minute token governor.
    One call per ticket keeps every request inside Groq's per-request limit while the AI still sees full evidence for ~8 matches at a time.
    Python verifies, lints and labels; it supplies a leg only when the AI could not (always marked 🐍)."""
    warnings = []
    state = load_state()
    tpm = int(min(int(cfg.get("tpm", GROQ_TPM_DEFAULT)), int(state.get("tpm_limit", 10 ** 9))))
    ceiling = max(3500, tpm - TPM_MARGIN)
    gov = TpmGovernor(max(ceiling + 50, tpm - WINDOW_MARGIN))
    budget = TokenBudget(int(cfg.get("run_tokens", RUN_TOKEN_CAP_DEFAULT)))
    cpt = min(max(float(state.get("cpt", DEFAULT_CHARS_PER_TOKEN)) * 0.95, 1.9), 3.2)
    need, reuse = n_t * LEGS_PER_TICKET, bool(cfg["allow_reuse"])
    distinct = not reuse
    eff = cfg.get("effort") if cfg.get("effort") in EFFORT_RESERVE else "medium"
    use_ai = cfg["use_ai"]
    n_ai = max(int(cfg.get("n_ai", 28)), need)
    passes, attempts, learned, cpt_obs = [], [], [], []

    def record(label, info):
        passes.append({"pass": label, "model": info.get("model"), "prompt": info.get("prompt_tokens", 0),
                       "completion": info.get("completion_tokens", 0), "status": info.get("status"), "waited": round(info.get("waited", 0.0)),
                       "fallback": bool(info.get("fallback"))})
        for a in info.get("attempts", []):
            attempts.append(dict(a, **{"pass": label}))
        if info.get("learned_limit"):
            learned.append(info["learned_limit"])
        if info.get("actual_cpt"):
            cpt_obs.append(info["actual_cpt"])

    # ── PASS 1: SCOUT - the AI orders the qualified matches best-first (it also re-measures the real token density) ──
    ordered = rank_matches(elig)
    scout_info = None
    k_keep = min(len(ordered), need + 6)
    if use_ai and cfg.get("two_pass", True) and len(ordered) > k_keep:      # a scout that would keep every match filters nothing: skip it, save the tokens
        pool = ordered[:max(n_ai, k_keep + 1)]
        sys_sc = scout_prompt(len(pool), k_keep, n_t, need)
        s_sel, s_ids, s_user = build_scout_pack(pool, cfg["tz"], cpt, sys_sc, SCOUT_RESERVE, ceiling)
        if s_sel:
            log(f"🔭 AI scout: reading a compact view of {len(s_sel)} qualified matches and choosing the {k_keep} best, ranked...")
            sc, sinfo = call_groq_budgeted(sys_sc, s_user, budget, cpt, "low", gov, ceiling, "keep", log, "scout")
            record("scout", sinfo)
            if sinfo.get("actual_cpt"):
                cpt = min(cpt, max(1.8, sinfo["actual_cpt"] * 0.97))
            if sc:
                chosen = []
                for x in sc.get("keep") or []:
                    mm = s_ids.get(str(x).strip().lower())
                    if mm is not None and mm not in chosen:
                        chosen.append(mm)
                chosen = chosen[:k_keep]
                topped = 0
                for mm in rank_matches(s_sel):
                    if len(chosen) >= need:
                        break
                    if mm not in chosen:
                        chosen.append(mm)
                        topped += 1
                ordered = chosen
                nm = lambda aid: (lambda mm: f"{mm['home']} v {mm['away']}" if mm else None)(s_ids.get(str(aid).strip().lower()))
                scout_info = {"model": sinfo.get("model"), "kept": len(chosen), "of": len(s_sel), "topped_up": topped,
                              "drops": [{"match": nm(x.get("id")), "why": x.get("why")} for x in (sc.get("drop") or []) if isinstance(x, dict) and nm(x.get("id"))][:8],
                              "traps": [{"match": nm(x.get("id")), "note": x.get("note")} for x in (sc.get("traps") or []) if isinstance(x, dict) and nm(x.get("id"))][:6]}
                log(f"   🔭 scout kept {len(chosen)} of {len(s_sel)} matches" + (f" (+{topped} added by Python rank to reach the {need} needed)" if topped else ""))
            else:
                ordered = ordered[:n_ai]
                warnings.append("The AI scout pass failed, so the desk used Python's ranking to order the matches.")
        else:
            ordered = ordered[:n_ai]
    else:
        ordered = ordered[:n_ai]
    progress(0.92)

    # global labels/legs: ids stay unique across the per-ticket calls
    labels = {id(m): f"m{i + 1}" for i, m in enumerate(ordered)}
    match_legs, leg_index, id_map = {}, {}, {}
    for m in ordered:
        mid = labels[id(m)]
        id_map[mid] = m
        lst = [dict(l, mid=mid, fid=m["id"], match=m, id=f"{mid}.{l['key']}") for l in m["legs"]]
        match_legs[mid] = lst
        for l2 in lst:
            leg_index[l2["id"].lower()] = l2

    def pack_for_call(avail, sys_i, cpt_, extra, start_level=0, force_low=False):
        floor = min(6, len(avail))
        efforts = ["low"] if force_low else [e for e in ("high", "medium", "low") if EFFORT_RESERVE[e] <= EFFORT_RESERVE[eff]]
        for eff_try in efforts:
            reserve = EFFORT_RESERVE[eff_try]
            sel_, idm_, user_, n_legs_, detail_, lvl_ = build_ai_pack(avail, cfg["tz"], cpt_, sys_i, floor, reserve, ceiling, labels, extra, start_level)
            if est_tokens(sys_i + user_, cpt_) <= ceiling - reserve - TOKEN_SAFETY:
                break
        return sel_, idm_, user_, n_legs_, detail_, eff_try, lvl_

    # ── PASS 2: DESK - one focused call per ticket ──
    ai_tickets, ai_traps, ai_avoid, ai_summ, shown, prompts = [], [], [], [], [], []
    used_labels, model_used, last_info = set(), None, {"attempts": []}
    level, trunc_fails, stop_ai, force_low = 0, 0, False, False      # force_low sticks once any call ran out of room
    if use_ai:
        for ti in range(n_t):
            if stop_ai:
                ai_tickets.append({})
                continue
            avail = ([m for m in ordered if labels[id(m)] not in used_labels] if distinct
                     else sorted(ordered, key=lambda m: labels[id(m)] in used_labels))
            if len(avail) < LEGS_PER_TICKET:
                warnings.append(f"Ticket {ti + 1}: fewer than {LEGS_PER_TICKET} unused matches were left, so the AI could not build it; it was completed/verified by Python.")
                ai_tickets.append({})
                continue
            extra = ""
            if reuse and ti > 0:
                extra = "EARLIER TICKETS (do not copy): " + "; ".join(f"T{k + 1}: " + ",".join(x["id"] for x in tk.get("_legs", [])) for k, tk in enumerate(ai_tickets) if tk) + "\n"
            sys_base = desk_prompt(ti, n_t, reuse, bool(cfg.get("relaxed_notes")))
            log(f"🤖 AI desk - ticket {ti + 1} of {n_t}: studying the best candidate matches in detail (Python only verifies afterwards)...")
            ai_i, info_i, sel_i, idm_i, user_i = None, {"attempts": []}, [], {}, ""
            for attempt in range(2):        # at most ONE retry, and only with a smaller request - never an identical one
                sys_i = sys_base + (BRIEF_ADDENDUM if attempt else "")
                sel_i, idm_i, user_i, n_legs, detail, eff_try, lvl = pack_for_call(avail, sys_i, cpt, extra, level, force_low)
                log(f"   📦 {'retry ' if attempt else ''}pack: {len(sel_i)} matches x {n_legs} legs (detail {detail}/3), est. prompt {est_tokens(sys_i + user_i, cpt)} tokens, effort '{eff_try}'")
                ai_i, info_i = call_groq_budgeted(sys_i, user_i, budget, cpt, eff_try, gov, ceiling, "tickets", log, f"desk{ti + 1}")
                record(f"desk{ti + 1}" + (" (retry)" if attempt else ""), info_i)
                if info_i.get("actual_cpt"):
                    cpt = min(cpt, max(1.8, info_i["actual_cpt"] * 0.97))
                st_ = info_i.get("status")
                if ai_i:
                    level = max(level, lvl)
                    break
                if st_ == "PROMPT_TOO_LARGE":
                    log("   ℹ️ The real prompt was bigger than estimated - rebuilding a smaller pack.")
                    continue
                if st_ == "TRUNCATED":
                    level, force_low = min(lvl + 2, len(PACK_LEVELS) - 1), True
                    log("   ℹ️ The model used its whole reply room before finishing - retrying ONCE with a smaller, briefer request.")
                    continue
                break
            last_info = info_i
            if not ai_i and info_i.get("attempts") and all(("per day" in a["status"].lower() or "(tpd)" in a["status"].lower() or a["status"].startswith("HTTP_404"))
                                                         for a in info_i["attempts"]):
                warnings.append("Groq's DAILY token allowance is used up for every available model (it resets daily; the error message says when). "
                                "The remaining AI calls were skipped instead of retrying - press 'Re-run AI analysis' later, it needs no API-Football calls.")
                ai_tickets += [{}] * (n_t - len(ai_tickets))
                break
            shown += [m for m in sel_i if m not in shown]
            prompts.append(f"===== desk call {ti + 1} (ticket {ti + 1}) =====\n{user_i}")
            if ai_i:
                model_used = model_used or info_i.get("model")
                t0 = next((t for t in (ai_i.get("tickets") or []) if isinstance(t, dict)), {})
                legs_ok = []
                for it in t0.get("legs") or []:
                    d = it if isinstance(it, dict) else {"id": str(it)}
                    lid = str(d.get("id") or "").strip().lower()
                    if lid in leg_index and lid.split(".")[0] in idm_i and leg_index[lid]["mid"] not in {leg_index[x["id"].lower()]["mid"] for x in legs_ok}:
                        legs_ok.append(d)
                t0 = dict(t0, legs=legs_ok, name=TICKET_LABELS[min(ti, 2)])
                t0["_legs"] = legs_ok
                ai_tickets.append(t0)
                if distinct:
                    used_labels |= {str(x["id"]).strip().lower().split(".")[0] for x in legs_ok}
                else:
                    used_labels |= {str(x["id"]).strip().lower().split(".")[0] for x in legs_ok}
                ai_traps += [x for x in (ai_i.get("traps") or []) if isinstance(x, dict) and x.get("id") in id_map]
                ai_avoid += [x for x in (ai_i.get("avoid") or []) if isinstance(x, dict) and x.get("id") in id_map]
                if ai_i.get("summary"):
                    ai_summ.append(str(ai_i["summary"]))
                if info_i.get("salvaged"):
                    warnings.append(f"Ticket {ti + 1}: the AI reply was cut short; the complete part was recovered and the rest verified/completed.")
            else:
                ai_tickets.append({})
                warnings.append(f"Ticket {ti + 1}: the AI desk call failed ({info_i.get('status')}); it will be completed by the audit / Python.")
                if info_i.get("status") in ("TRUNCATED", "PROMPT_TOO_LARGE"):
                    trunc_fails += 1
                    if trunc_fails >= 1:        # the retry was already smaller and briefer: another ticket would fail the same way, so stop spending
                        stop_ai = True
                        warnings.append("An AI desk call ran out of reply room even after being retried smaller, so the remaining AI calls were SKIPPED to avoid wasting more tokens. "
                                        "Set 'AI reasoning effort' to low in Settings, or press Re-run AI analysis after a minute.")
        if cpt_obs:
            state["cpt"] = round(0.6 * min(cpt_obs) + 0.4 * (sum(cpt_obs) / len(cpt_obs)), 3)
        if learned:
            state["tpm_limit"] = int(min(learned))
        save_state(state)
    ai = None
    if any(t.get("legs") for t in ai_tickets):
        seen, traps_u = set(), []
        for x in ai_traps:
            if x["id"] not in seen:
                seen.add(x["id"])
                traps_u.append(x)
        ai = {"tickets": ai_tickets, "traps": traps_u[:6], "avoid": ai_avoid[:6], "summary": " ".join(ai_summ)[:320]}
    sel = shown or ordered
    user_prompt = "\n\n".join(prompts)
    info = dict(last_info, attempts=attempts)

    python_draft, ai_note, audit_changes, rule_fixes = None, "", [], []
    if ai:
        tickets = parse_ai_tickets(ai, leg_index, n_t, LEGS_PER_TICKET, distinct)
        # ── PASS 3: AUDIT - a risk officer tries to break every leg, swaps weak legs, fills gaps ──
        if cfg.get("audit", True) and not stop_ai:
            if budget.remaining >= AUDIT_RESERVE + 1800:
                sys_au = audit_prompt(n_t)
                a_user = build_audit_pack(tickets, match_legs, cfg["tz"], cpt, sys_au, AUDIT_RESERVE, ceiling, LEGS_PER_TICKET, distinct)
                if a_user:
                    log("🕵️ AI risk-officer audit: trying to break every leg and filling any gap...")
                    aud, ainfo = call_groq_budgeted(sys_au, a_user, budget, cpt, "low", gov, ceiling, "legs", log, "audit")
                    record("audit", ainfo)
                    if aud:
                        audit_changes = apply_audit(tickets, aud, leg_index, match_legs, LEGS_PER_TICKET, distinct)
                        if aud.get("note"):
                            ai["audit_note"] = str(aud["note"])[:240]
                    else:
                        warnings.append("The AI audit pass failed; the desk's tickets were verified by Python rules only.")
                else:
                    warnings.append("The audit pass was skipped because its evidence did not fit one request.")
            else:
                warnings.append("The audit pass was skipped: the per-analysis AI token cap was nearly used up (raise it in Settings).")
        tickets = complete_tickets(tickets, match_legs, LEGS_PER_TICKET, distinct)
        if cfg.get("rule_fix", True):
            rule_fixes = py_fix_breaches(tickets, match_legs)
            for t in tickets:
                finish_ticket(t, LEGS_PER_TICKET)
        if not cfg.get("allow_py_fallback"):
            keep = [t for t in tickets if t["ai_kept"] >= int(cfg.get("min_ai_legs", 3))]
            if len(keep) < len(tickets):
                warnings.append(f"{len(tickets) - len(keep)} ticket(s) were mostly Python-built because the AI did not supply enough valid legs, so they were withheld "
                                "(Settings -> 'allow Python-built tickets' lets them through, clearly labelled).")
                python_draft = [t for t in tickets if t not in keep]
            tickets = keep
    else:
        draft = python_only_tickets(match_legs, n_t)
        for t in draft:
            finish_ticket(t, LEGS_PER_TICKET)
        if use_ai:
            ai_note = " | ".join(f"{a['model'].split('/')[-1]}: {a['status'][:110]}" for a in attempts[-3:])
            if cfg.get("allow_py_fallback"):
                tickets = draft
                warnings.append(f"AI analysis failed ({info.get('status')}): {ai_note}. Python-built tickets (🐍, NOT AI-approved) were used because you allowed that fallback.")
            else:
                tickets, python_draft = [], draft
                warnings.append(f"AI analysis failed ({info.get('status')}): {ai_note}. No tickets were produced because AI analysis is required. "
                                "Use 'Re-run AI analysis' - it reuses the collected data and costs no API-Football calls.")
        else:
            tickets = draft
    for t in tickets + (python_draft or []):
        dqm = min([l["match"]["dq"] for l in t["legs"]] or [0])
        t["grade"] = "A" if dqm >= 0.75 and not t["repaired"] and not cfg.get("relaxed_notes") else "B" if dqm >= 0.6 else "C"
    est_mode = any(l.get("src") == "est" for t in tickets for l in t["legs"])
    if est_mode:
        warnings.append("Some legs use MODEL-ESTIMATED odds (fair odds x 0.93, no market check). Bookmakers differ slightly - confirm the price on yours.")
    wasted = sum(p["prompt"] + p["completion"] for p in passes if p.get("status") != "SUCCESS")
    if wasted > 1500:
        warnings.append(f"🔥 {wasted} AI tokens went to calls that produced no usable answer (listed in the 'AI passes' table).")
    fallback_model = any(p["fallback"] for p in passes)
    if fallback_model:
        warnings.append("⚠️ At least one AI pass ran on a smaller FALLBACK model (the top model was rate-limited or unavailable). Analysis quality may be lower - "
                        "consider pressing 'Re-run AI analysis' after a minute.")
    return {"tickets": tickets, "ai": ai, "model": model_used, "info": info, "tokens": budget.used, "sel": sel, "id_map": id_map,
            "user_prompt": user_prompt, "est_mode": est_mode, "ai_failed": bool(use_ai and not ai), "python_draft": python_draft,
            "ai_note": ai_note, "warnings": warnings, "passes": passes, "fallback_model": fallback_model, "scout": scout_info, "wasted_tokens": wasted,
            "audit_changes": audit_changes, "rule_fixes": rule_fixes}


def rerun_ai(res, cfg, log):
    """Re-run ONLY the AI stage on already-collected data (no API-Football calls)."""
    elig = [m for m in res["matches"] if m.get("eligible") and m.get("legs")]
    n_t = plan_ticket_count(len(elig), cfg)
    if n_t == 0:
        return res
    stage = run_ai_stage(elig, n_t, cfg, log)
    old = [w for w in res.get("warnings", []) if not any(k in w for k in ("AI analysis failed", "AI unavailable", "MODEL-ESTIMATED", "token cap", "mostly Python-built",
                                                                         "FALLBACK model", "scout pass", "audit pass"))]
    new = dict(res, tickets=stage["tickets"], ai=stage["ai"], model=stage["model"], info=stage["info"], tokens=stage["tokens"], n_ai=len(stage["sel"]),
               id_map=stage["id_map"], user_prompt=stage["user_prompt"], est_mode=stage["est_mode"], ai_note=stage["ai_note"],
               ai_failed=stage["ai_failed"], python_draft=stage["python_draft"], warnings=old + stage["warnings"], cfg=cfg,
               passes=stage["passes"], fallback_model=stage["fallback_model"], scout=stage["scout"], audit_changes=stage["audit_changes"], rule_fixes=stage["rule_fixes"],
               wasted_tokens=stage["wasted_tokens"])
    if new["tickets"]:
        save_tickets_to_ledger(cfg["date"], new["tickets"])
    return new


def test_groq_connection():
    """Tiny request (~100 tokens) to each model: verifies the key, model access and limits without a full analysis."""
    key = get_secret("GROQ_API_KEY", "").strip()
    if not key:
        return [("-", "GROQ_API_KEY is not set")]
    rows = []
    for model in GROQ_MODELS:
        cfg = GROQ_MODEL_CONFIG.get(model, GROQ_DEFAULT_CFG)
        payload = {"model": model, "max_completion_tokens": 120, "temperature": 0, "response_format": {"type": "json_object"},
                   "messages": [{"role": "user", "content": 'Return exactly this JSON and nothing else: {"ok": true}'}]}
        if cfg.get("supports_reasoning_effort"):
            payload["reasoning_effort"] = "low" if model.startswith("openai/") else "none"
            if model.startswith("openai/"):
                payload["include_reasoning"] = False
        try:
            r = requests.post(GROQ_API_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=30)
            if r.status_code == 200:
                rows.append((model, f"✅ working ({(r.json().get('usage') or {}).get('total_tokens', '?')} tokens used)"))
            else:
                rows.append((model, f"❌ HTTP {r.status_code}: {r.text[:170]}"))
        except Exception as e:
            rows.append((model, f"❌ {str(e)[:120]}"))
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═════════════════════════════════════════════════════════════════════════════
def render_results(res, tz):
    for w in res.get("warnings", []):
        st.warning(w)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Matches analysed", res["n_analysed"])
    c2.metric("Studied by AI desk", res["n_ai"])
    c3.metric("AI tokens (all passes)", res["tokens"])
    c4.metric("API calls", res["api_calls"], f"cache {res['cache_hits']}")
    c5.metric("Model", (res.get("model") or "Python optimiser").split("/")[-1] if res["tickets"] else "-")
    if res.get("tokens"):
        w = res.get("wasted_tokens", 0)
        st.caption(f"🧮 AI tokens: **{res['tokens'] - w}** went into answers that were used" + (f" · 🔥 **{w}** went to calls that produced nothing (see 'AI passes')" if w else " · none wasted ✅"))
    if res.get("quota") and res["quota"][0] is not None:
        st.caption(f"📶 API quota left today: **{res['quota'][0]}** of {res['quota'][1]} (resets 00:00 UTC / 03:00 Kampala)")
    if res.get("caps"):
        cp = res["caps"]
        st.caption(f"API plan capabilities detected - team history: {'✅' if cp['history'] else '❌ (free sources / predictions last-5 form used)'} · "
                   f"odds by league: {'✅' if cp['league_odds'] else '❌' if cp['league_odds'] is False else '?'} · odds by fixture: {'✅' if cp['fixture_odds'] else '❌'}")
    fr = res.get("free") or {}
    if fr.get("used") or fr.get("saved_calls"):
        st.caption("🆓 Free sources (no API quota): " + " · ".join(f"{k} ×{v}" for k, v in fr.get("used", {}).items())
                   + (f" · ≈{fr['saved_calls']} API calls avoided" if fr.get("saved_calls") else ""))
    if res.get("relaxed"):
        st.info("🪜 Thin slate - thresholds relaxed: " + "; ".join(res["relaxed"]))
    if res.get("no_bet"):
        st.error(f"🛑 **NO BET** - {res.get('reason')}")
        if res.get("api_errors"):
            st.markdown("**API messages (send these if you need help):**")
            st.code("\n".join(res["api_errors"]), language="text")
    if res.get("ai_failed"):
        st.error("🤖 **AI analysis did not complete.** " + (res.get("ai_note") or ""))
        st.info("Nothing needs to be re-collected: press **Re-run AI analysis** below (no API-Football calls are used).")
    if res.get("python_draft"):
        with st.expander("🐍 Python reference draft - NOT AI-approved (withheld)"):
            for t in res["python_draft"]:
                st.write(f"**{t['name']}** - odds {t['odds']:.2f}: " + " | ".join(f"{l['match']['home']} v {l['match']['away']}: {l['label']} @{l['odds']:.2f}" for l in t["legs"]))
    for t in res["tickets"]:
        st.markdown(f"### 🎟 {t['name']} — odds **{t['odds']:.2f}** {'✅' if t['valid'] else '⚠️ below 3.5'} · joint prob ~{_pct(t['p_joint'])}% · evidence grade **{t.get('grade', '?')}**")
        rows = []
        for l in t["legs"]:
            m = l["match"]
            au = l.get("audit") or {}
            rows.append({"Kick-off": datetime.fromtimestamp(m["ts"], ZoneInfo(tz)).strftime("%H:%M"), "League": m["league"],
                         "Match": f"{m['home']} v {m['away']}", "Pick": l["label"], "Odds": round(l["odds"], 2),
                         "Price": (f"{l['odds_lo']:.2f}-{l['odds_hi']:.2f}" if l.get("odds_lo") else "derived" if l.get("src") == "derived" else "est."),
                         "By": _by_txt(l), "Model+Mkt %": _pct(l["p"]), "Market fair %": _pct(l["p_mkt"]), "Trap": m["trap"]["risk"],
                         "Data": BASIS_TAG.get(m["basis"]), "Why": l.get("why", ""), "Loses if": l.get("fail", ""),
                         "Audit": (("✅ " if au.get("ok", True) else "⚠️ ") + (au.get("issue") or au.get("risk") or "")) if au else "",
                         "Flags": ",".join(l.get("flags") or [])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if res.get("model"):
            st.caption(f"🤖 AI chose {t.get('ai_kept', 0)} of {len(t['legs'])} legs"
                       + (f" ({t.get('audit_swaps', 0)} improved by the audit)" if t.get("audit_swaps") else "")
                       + (f"; 🐍 Python supplied {t.get('py_legs', 0)}" if t.get("py_legs") else "; Python only verified prices, uniqueness, rules and the 3.5 minimum") + ".")
        if t.get("logic"):
            st.info(t["logic"])
        if t.get("repaired"):
            st.caption("🔧 Python completed/adjusted part of this ticket (🐍 legs): AI leg missing/invalid, rule breach or odds below 3.5.")
        if t.get("stacked"):
            st.caption("⚠️ 4+ legs of the same market type - correlated failure risk.")
    if res.get("audit_changes") or res.get("rule_fixes"):
        with st.expander("🕵️ Audit & rule-check changes"):
            for c in res.get("audit_changes", []):
                st.write(f"🤖 audit: {c}")
            for c in res.get("rule_fixes", []):
                st.write(f"🐍 rule-fix: {c}")
    ai = res.get("ai") or {}
    for title, key, fld in (("🪤 Trap watch", "traps", "note"), ("🚫 Avoid", "avoid", "why")):
        items = [(res["id_map"][x["id"]], x.get(fld)) for x in (ai.get(key) or []) if x.get("id") in res["id_map"]]
        if key == "traps":
            items += [({"home": x["match"], "away": ""}, x.get("note")) for x in ((res.get("scout") or {}).get("traps") or [])]
        if items:
            st.markdown(f"#### {title}")
            for m, note in items:
                st.write(f"**{m['home']}{' v ' + m['away'] if m.get('away') else ''}** — {note}")
    if ai.get("audit_note"):
        st.markdown(f"#### 🕵️ Risk officer\n{ai['audit_note']}")
    if ai.get("summary"):
        st.markdown(f"#### 📝 Slate read\n{ai['summary']}")
    sc = res.get("scout")
    if sc:
        with st.expander(f"🔭 AI scout - kept {sc['kept']} of {sc['of']} qualified matches for the full analysis"):
            for x in sc.get("drops", []):
                st.write(f"❌ {x['match']} — {x.get('why')}")
    if res.get("excluded"):
        with st.expander(f"🛡 Excluded by the evidence gate ({len(res['excluded'])})"):
            st.dataframe(pd.DataFrame(res["excluded"]), hide_index=True, width="stretch")
    with st.expander("📋 All analysed matches (Monte Carlo + market blend)"):
        rows = []
        for m in res["matches"]:
            p = m["p"]
            rows.append({"Cat": m["cat"], "League": m["league"], "Match": f"{m['home']} v {m['away']}",
                         "KO": datetime.fromtimestamp(m["ts"], ZoneInfo(tz)).strftime("%H:%M"),
                         "Data": BASIS_TAG.get(m["basis"]), "Sources": ",".join(sorted(set(m.get("free_tags") or []))) or "api",
                         "xG": f"{m['lam_h']:.2f}-{m['lam_a']:.2f}",
                         "1": _pct(p["1"]), "X": _pct(p["X"]), "2": _pct(p["2"]), "1X": _pct(p["1X"]), "X2": _pct(p["X2"]),
                         "BTTS": _pct(p["BTTS_Y"]), "O1.5": _pct(p["O1.5"]), "O2.5": _pct(p["O2.5"]),
                         "Corners": round(m["exp_corners"], 1) if m["corner_n"] >= 3 else None,
                         "Trap": m["trap"]["risk"], "DQ": m["dq"], "Qualified": "yes" if m.get("eligible") else "no",
                         "Likely": ",".join(m["top_scores"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if res.get("user_prompt"):
        with st.expander("🔎 Exact evidence pack sent to the AI desk"):
            st.code(res["user_prompt"], language="text")
    with st.expander("🛠 AI passes / model attempts / API errors"):
        if res.get("passes"):
            st.dataframe(pd.DataFrame(res["passes"]), hide_index=True, width="stretch")
        for a in (res.get("info") or {}).get("attempts", []):
            st.caption(f"[{a.get('pass', '-')}] {a['model']} → {a['status']}")
        for e in res.get("api_errors", []):
            st.caption(f"API: {e}")
        for e in (res.get("free") or {}).get("fails", []):
            st.caption(f"free source: {e}")


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
        n_ai = s1.slider("Max matches the AI scout reads (the desk auto-fits as many as one Groq request allows)", 15, 40, 28)
        max_calls = s1.number_input("Max API-Football calls per run", 30, 5000, 95, step=5,
                                    help="Free plan = 100/day. Pro = 7,500/day. The app auto-shrinks the shortlist to fit.")
        breadth = s1.selectbox("Search breadth", ["Focused", "Balanced", "Wide"], index=1,
                               help="Focused leans on major leagues; Wide gives lower and mid competitions and international games more room to compete for the tickets.")
        depth = s1.selectbox("Data depth", ["Auto", "Full", "Lite"], help="Full adds corners/shots/xG history, player ratings, lineups and standings via batched calls.")
        sims = s2.select_slider("Monte Carlo simulations per match", [10000, 20000, 40000, 80000], 40000)
        w_model = s2.slider("Weight of Monte Carlo model vs bookmaker price", 0.2, 0.8, 0.5, 0.05)
        bookmaker = s2.number_input("Preferred bookmaker id (8 = Bet365, 11 = 1xBet, 4 = Pinnacle)", 1, 500, 8)
        effort = s2.selectbox("AI reasoning effort (gpt-oss)", ["low", "medium", "high"], index=0, help="gpt-oss counts its hidden thinking against the reply room. LOW is recommended on the free Groq tier (8,000 tokens per request): medium/high shrink the evidence pack to leave thinking room, and a call that runs out of room is wasted.")
        exclude_minor = s2.checkbox("Exclude youth / women / reserve teams", True)
        exclude_lower = s2.checkbox("Exclude amateur / regional lower divisions & club friendlies", True)
        force_leagues = {int(x) for x in re.findall(r"\d+", s2.text_input("Always include league ids (comma separated)", ""))}
        rate_override = int(s2.number_input("Requests/minute override (0 = auto by plan)", 0, 900, 0))
        send_tg = st.checkbox("Send tickets to Telegram automatically", True)
        use_ai = st.checkbox("Use AI (Groq)", True)
        allow_py = st.checkbox("Always deliver tickets: if the AI cannot supply a leg or fails, allow Python-built legs/tickets (clearly marked 🐍, never presented as AI-approved)", True)
        if st.button("🧪 Test AI connection (~100 tokens per model)"):
            for mdl, status in test_groq_connection():
                st.write(f"`{mdl}` → {status}")
        min_dq = st.slider("Minimum data-quality score for a match to qualify", 0.3, 0.9, 0.5, 0.05,
                           help="Matches below this (or with no team-specific data / no real odds) are excluded from tickets.")
        allow_reuse = st.checkbox("Allow the same match in more than one ticket (only if too few matches qualify)", False)
        allow_est = st.checkbox("If the API has no odds for a match, use model-estimated approximate odds (clearly labelled)", True,
                                help="Real bookmaker prices are always preferred (median of all bookmakers). Estimates only fill in when a match has none; trap and market-agreement checks are weaker for those matches.")
        st.markdown("##### 🤖 AI pacing & quality")
        a1, a2 = st.columns(2)
        tpm = a1.number_input("Groq tokens-per-minute limit", 4000, 30000, GROQ_TPM_DEFAULT, step=500,
                              help="Every Groq request is kept below this, and the passes (scout -> desk -> audit) are spaced so the rolling 60-second window never overflows. The app also learns the real limit from Groq's error messages.")
        run_tokens = a2.number_input("Max AI tokens per analysis (all passes)", 10000, 120000, RUN_TOKEN_CAP_DEFAULT, step=2000,
                                     help="Protects your daily Groq token allowance. One full analysis (scout + 3 desk calls + audit) normally uses 25-35k.")
        two_pass = a1.checkbox("AI scout first: rank the qualified matches best-first before the per-ticket desk calls", True)
        audit = a2.checkbox("Final AI risk-officer audit (tries to break every leg, swaps weak legs, fills gaps)", True)
        rule_fix = a1.checkbox("Python rule-fix: replace a leg that breaks the AI's own hard rules with the best clean alternative", True)
        min_ai_legs = a2.slider("Minimum AI-chosen legs per ticket (matters only if Python-built tickets are not allowed)", 3, 5, 3)
        wide_menu = a1.checkbox("Widen the leg menu with clearly-flagged borderline legs (the AI decides)", True)
        auto_relax = a2.checkbox("Auto-relax safety thresholds step by step when too few matches qualify (always labelled)", True)
        tg_top_only = a1.checkbox("Send to Telegram only if the TOP model did the analysis", False)
        st.markdown("##### 🆓 Free data sources & quota saving")
        f1, f2 = st.columns(2)
        use_free = f1.checkbox("Use free sources (football-data.co.uk, ClubElo, ESPN) first and as fallback - costs 0 API-Football calls", True)
        save_calls = f2.checkbox("Skip API-Football predictions when free sources already cover the match (saves ~1 call per match)", True)
        st.markdown(f"**AI order:** {' → '.join(GROQ_MODELS)} · Calls: scout → desk (one per ticket) → audit")
        if st.button("📶 Check API quota now (free - does not use quota)"):
            key_ = get_secret("API_FOOTBALL_KEY")
            if key_:
                s_ = ApiFootball(key_, 10).status()
                st.info(f"Plan: {s_.get('plan')} · used {s_.get('used')} of {s_.get('limit')} calls today (resets 00:00 UTC / 03:00 Kampala)")
            else:
                st.warning("API_FOOTBALL_KEY not set.")
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
        st.caption("Free API plan: one run now needs fewer calls because free sources are used first. The AI works in short focused calls (scout, one per ticket, audit) spaced to respect Groq's per-minute token limit, so expect several extra minutes. Re-running the same day reuses cached data; the API quota resets 00:00 UTC (03:00 Kampala).")
        if st.button("🧠 Analyse Football Matches Now", type="primary"):
            cfg = {"date": d.strftime("%Y-%m-%d"), "tz": tz, "max_matches": max_matches, "n_ai": n_ai, "max_calls": int(max_calls),
                   "depth": depth, "sims": int(sims), "w_model": w_model, "bookmaker": int(bookmaker), "effort": effort,
                   "exclude_minor": exclude_minor, "breadth": breadth, "exclude_lower": exclude_lower, "force_leagues": force_leagues, "rate_override": rate_override,
                   "force": force, "use_ai": use_ai, "allow_py_fallback": allow_py, "min_dq": min_dq, "allow_reuse": allow_reuse, "allow_est": allow_est, "tpm": int(tpm), "run_tokens": int(run_tokens), "two_pass": two_pass,
                   "audit": audit, "use_free": use_free, "save_calls": save_calls, "auto_relax": auto_relax, "min_ai_legs": int(min_ai_legs),
                   "wide_menu": wide_menu, "rule_fix": rule_fix}
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
                if send_tg and tg_top_only and res.get("fallback_model") and res["tickets"]:
                    st.info("Telegram skipped: a fallback model took part and you asked for top-model-only delivery. Press Re-run AI analysis later.")
                elif send_tg:
                    msg = (build_nobet_message(res, tz) if res.get("no_bet") else build_aifail_message(res, tz) if res.get("ai_failed") and not res["tickets"]
                           else build_telegram_message(res, tz))
                    ok, why = send_telegram(msg)
                    (st.success if ok else st.warning)("✅ Sent to Telegram" if ok else f"Telegram failed: {why}")
                    st.session_state.notifications.append({"time": datetime.now().strftime("%H:%M"), "ok": ok,
                        "msg": f"{cfg['date']}: " + ("NO-BET notice " if res.get("no_bet") else f"{len(res['tickets'])} ticket(s) ") + ("sent" if ok else f"not sent ({why})")})
        if st.session_state.last_result:
            render_results(st.session_state.last_result, st.session_state.get("last_tz", tz))
            lr = st.session_state.last_result
            if lr.get("cfg") and any(m.get("eligible") for m in lr.get("matches", [])):
                st.divider()
                if st.button("🔁 Re-run AI analysis on the data already collected (uses NO API-Football calls)"):
                    box2 = st.status("Re-running the AI stage...", expanded=True)
                    try:
                        new = rerun_ai(lr, dict(lr["cfg"], effort=effort, use_ai=True, allow_py_fallback=allow_py, tpm=int(tpm), run_tokens=int(run_tokens), two_pass=two_pass, audit=audit, rule_fix=rule_fix, min_ai_legs=int(min_ai_legs)), box2.write)
                        st.session_state.last_result = new
                        if send_tg and (new["tickets"] or new.get("ai_failed")):
                            ok, why = send_telegram(build_telegram_message(new, tz) if new["tickets"] else build_aifail_message(new, tz))
                            st.session_state.notifications.append({"time": datetime.now().strftime("%H:%M"), "ok": ok,
                                                                   "msg": "AI re-run: " + (f"{len(new['tickets'])} ticket(s) " if new["tickets"] else "no tickets ") + ("sent" if ok else f"not sent ({why})")})
                        box2.update(label="AI re-run finished", state="complete")
                        st.rerun()
                    except Exception:
                        box2.update(label="AI re-run failed", state="error")
                        st.error(traceback.format_exc())

    with tab2:
        st.header("📜 Ticket Ledger & Calibration")
        state = load_state()
        if st.button("🔄 Settle finished matches"):
            key = get_secret("API_FOOTBALL_KEY")
            save_state(settle_history(ApiFootball(key or "", 10), state))
            st.success("Ledger updated (free results first, API-Football only for what is left).")
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
