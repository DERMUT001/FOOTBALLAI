# =============================================================================
# Der-AI | Football Quant Desk — V3 (AI Match Selector)
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
#   5. AI (Groq) reads ALL qualified matches with full evidence and selects the TOP 10 safest,
#      most valuable individual match predictions. Python NEVER picks matches - only verifies.
#   6. PYTHON only verifies (real data, valid odds). The AI is the sole decision-maker.
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
CACHE_PURGE_DAYS = 8
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT = 90
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]
GROQ_MODEL_CONFIG = {
    "openai/gpt-oss-120b": {"max_completion_tokens": 4000, "reasoning_effort": "medium", "supports_reasoning_effort": True},
    "openai/gpt-oss-20b": {"max_completion_tokens": 4000, "reasoning_effort": "medium", "supports_reasoning_effort": True},
    "llama-3.3-70b-versatile": {"max_completion_tokens": 2500, "reasoning_effort": None, "supports_reasoning_effort": False},
}
GROQ_DEFAULT_CFG = {"max_completion_tokens": 2000, "reasoning_effort": None, "supports_reasoning_effort": False}
NON_RETRYABLE = {400, 401, 403, 404, 422}

# ── Token policy ────────────────────────────────────────────────────────────
GROQ_TPM_DEFAULT = 8000
TPM_MARGIN = 350
WINDOW_MARGIN = 150
RUN_TOKEN_CAP_DEFAULT = 45000
EFFORT_RESERVE = {"low": 2600, "medium": 3800, "high": 5000}
OUTPUT_RESERVE = 3000
TOKEN_SAFETY = 100
MIN_COMPLETION = 900
DEFAULT_CHARS_PER_TOKEN = 2.3

# ── AI Match Selection rules ────────────────────────────────────────────────
TARGET_PICKS = 10
MIN_PICKS = 1
MIN_LEG_P, MIN_LEG_ODDS, MAX_LEG_ODDS = 0.62, 1.15, 2.00
MIN_AGREE_P = 0.60
BORDER_AGREE_P, BORDER_MIN_P = 0.55, 0.57
MAX_MENU_LEGS = 7
BASIS_TAG = {"season+recent": "S+R", "recent": "R", "season": "S", "elo": "E", "prior": "P"}

# ── League intelligence ─────────────────────────────────────────────────────
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
        self.blocked = set()
        self.blocked_ep = set()
        self.err_counts = {}
        self.ep_fail, self.ep_ok = {}, {}
        self.no_history = False
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
            with self.lock:
                self.ep_fail[endpoint] = self.ep_fail.get(endpoint, 0) + 1
                if self.ep_fail[endpoint] >= 3 and not self.ep_ok.get(endpoint):
                    self.blocked_ep.add(endpoint)

    def error_summary(self):
        with self.lock:
            return [f"{k}  (x{v})" if v > 1 else k for k, v in self.err_counts.items()]

    def status(self):
        if self.offline:
            return {"plan": None, "errors": "no API-Football key configured"}
        try:
            r = self.session.get(f"{APIF_BASE}/status", timeout=20)
            d = r.json()
            errs_ = d.get("errors")
            if isinstance(errs_, dict) and errs_.get("access"):
                self.offline = True
                return {"plan": None, "errors": errs_}
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
                if "request limit" in low or ("requests" in low and "day" in low):
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
# FREE data sources (football-data.co.uk, ClubElo, ESPN)
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
               103: ("Eliteserien", "Norway"), 113: ("Allsvenskan", "Sweden"),
               235: ("Premier League", "Russia"), 169: ("Super League", "China"), 357: ("Premier Division", "Ireland"),
               244: ("Veikkausliiga", "Finland"), 106: ("Ekstraklasa", "Poland"), 283: ("Liga I", "Romania")}
ELO_COUNTRIES = {"england", "spain", "italy", "germany", "france", "netherlands", "portugal", "belgium", "turkey", "scotland",
                 "greece", "switzerland", "austria", "denmark", "norway", "sweden", "poland", "czech-republic", "croatia",
                 "serbia", "ukraine", "russia", "romania", "hungary", "cyprus", "israel", "slovakia", "slovenia", "bulgaria",
                 "wales", "ireland", "northern-ireland", "finland", "iceland", "bosnia", "belarus", "azerbaijan", "kazakhstan"}
ELO_LEAGUE_IDS = {2, 3, 848}
INTL_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
FREE_TTL_S = 6 * 3600
FREE_UA = {"User-Agent": "Mozilla/5.0 (compatible; DerAI-FootballDesk/3.0)"}
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
        self.used = defaultdict(int)
        self.saved_calls = 0
        self.fails: List[str] = []
        self._rows: Dict[Any, list] = {}
        self._fix_main: Optional[list] = None
        self._espn: Dict[Any, list] = {}
        self._elo: Optional[list] = None
        self._intl: Optional[list] = None
        self._intl_names_c: Optional[list] = None
        self._intl_elo_c: Optional[tuple] = None
        self.target_date: Optional[str] = None
        self.intl_extra = 0
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass

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
        return best_match(name, self._intl_names_c or [], 0.95)

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
        tab = defaultdict(lambda: [0, 0, 0])
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
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        if league_id in FD_MAIN:
            return [r for r in self._fix_rows() if r["div"] == FD_MAIN[league_id] and abs((r["date"] - d).days) <= 1]
        if league_id in FD_NEW:
            rows = self._new_rows(FD_NEW[league_id], d - timedelta(days=520))
            return [r for r in rows if r["hg"] is None and abs((r["date"] - d).days) <= 1]
        return []

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
        txt = self._get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={date_str.replace('-', '')}",
                        ttl=(7 * 86400 if past else 1800), timeout=15)
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
                        dup["odds"] = dict(r["odds"])
                    continue
                out.append({"league_id": lid, "home": r["home"], "away": r["away"], "ts": ts, "odds": dict(r["odds"]), "src": "football-data"})
        return out

    def summary(self):
        return {"used": dict(self.used), "saved_calls": self.saved_calls, "fails": self.fails[:6]}


def derive_dc_odds(odds):
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
# Team profile, Monte Carlo, Market blending, Trap analysis
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
    prof["lineup"] = None
    if lineup:
        xi = {((x.get("player") or {}).get("id")) for x in lineup.get("startXI") or []}
        kin = len(prof["key_ids"] & xi)
        prof["lineup"] = {"formation": lineup.get("formation"), "key_in_xi": kin, "key_n": len(prof["key_ids"]),
                          "rotation": bool(len(prof["key_ids"]) >= 5 and kin <= len(prof["key_ids"]) - 3)}
    return prof


def enrich_profile_from_l5(prof, side):
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


def estimate_lambdas(pred, ph, pa, elo=None):
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
    if elo and elo.get("intl") and basis != "elo" and lh + la > 0:
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


def monte_carlo(lam_h, lam_a, mu_ch, mu_ca, n=40000, seed=0):
    rng = np.random.default_rng(seed)
    tempo = rng.gamma(28.0, 1 / 28.0, n)
    lh = lam_h * tempo * np.exp(rng.normal(-0.005, 0.10, n))
    la = lam_a * tempo * np.exp(rng.normal(-0.005, 0.10, n))
    h1, h2 = rng.poisson(lh * 0.45), rng.poisson(lh * 0.55)
    a1, a2 = rng.poisson(la * 0.45), rng.poisson(la * 0.55)
    h, a, hh, ah = h1 + h2, a1 + a2, h1, a1
    rho = -0.06
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


def _devig(odds_map, keys):
    if all(k in odds_map for k in keys):
        inv = [1 / odds_map[k]["odds"] for k in keys]
        s = sum(inv)
        return {k: i / s for k, i in zip(keys, inv)}
    return None


API_DEFAULT_PATTERNS = {(0, 50, 50), (10, 45, 45), (30, 35, 35), (33, 33, 33)}


def api_dist(pred):
    pc = (pred or {}).get("pct") or []
    if len(pc) != 3 or None in pc or sum(pc) <= 0:
        return None
    if tuple(sorted(int(round(x)) for x in pc)) in API_DEFAULT_PATTERNS:
        return None
    t = sum(pc)
    return {"1": pc[0] / t, "X": pc[1] / t, "2": pc[2] / t}


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
        risk += 35
        why.append(f"{pre}{_pct(fp)}/mdl{_pct(mp[fav])}")
    elif gap >= 0.08:
        risk += 22
        why.append(f"{pre}{_pct(fp)}/mdl{_pct(mp[fav])}")
    elif gap >= 0.05:
        risk += 10
        why.append(f"{pre}{_pct(fp)}/mdl{_pct(mp[fav])}")
    if pf.get("n", 0) >= 4 and pf["ppg5"] < 1.4 and fp >= 0.6:
        risk += 15
        why.append(f"fav ppg{pf['ppg5']:.1f}")
    if pu.get("n", 0) >= 4 and pu["ppg5"] >= 1.8:
        risk += 10
        why.append(f"dog ppg{pu['ppg5']:.1f}")
    keyout = [i for i in pf.get("inj", []) if i["key"] and i["type"] == "Missing Fixture"]
    if keyout:
        risk += min(20, 8 * len(keyout))
        why.append(f"{len(keyout)} fav key out")
    if (pf.get("lineup") or {}).get("rotation"):
        risk += 15
        why.append("fav rotated XI")
    h = m.get("h2h")
    if h and h["n"] >= 4:
        fav_unbeaten = (h["w"] + h["d"]) if fav_home else (h["l"] + h["d"])
        dog_unbeaten = h["n"] - (h["w"] if fav_home else h["l"])
        if dog_unbeaten / h["n"] >= 0.6:
            risk += 10
            why.append(f"H2H dog unbeaten {dog_unbeaten}/{h['n']}")
    if not fav_home and fp >= 0.6:
        risk += 8
        why.append("away fav")
    if mp["X"] >= 0.27:
        risk += 10
        why.append(f"X{_pct(mp['X'])}%")
    lam_f = m["lam_h"] if fav_home else m["lam_a"]
    if lam_f < 1.5 and fp >= 0.62:
        risk += 10
        why.append("fav low xG")
    dog_st = m.get("st_a") if fav_home else m.get("st_h")
    if fav_home is False and dog_st and dog_st.get("rank") and dog_st.get("of") and dog_st["rank"] > dog_st["of"] - 4:
        risk += 8
        why.append("home dog in drop zone")
    lname = (m.get("league") or "").lower()
    if "friendl" in lname:
        risk += 20
        why.append("friendly")
    elif "cup" in lname or "copa" in lname or "pokal" in lname or "coppa" in lname:
        risk += 8
        why.append("cup rotation")
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
    if age > 90:
        dq = max(0.0, dq - 0.10)
    elif age > 30:
        dq = max(0.0, dq - 0.05)
    if not m.get("odds"):
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
        elif allow_est and not m["odds"]:
            odds, src = round(max(1.05, 0.93 / max(p, 0.05)), 2), "est"
            p_mkt = ext.get(key, p) if ext else p
        else:
            continue
        if not (MIN_LEG_ODDS <= odds <= MAX_LEG_ODDS):
            continue
        p_model = m["model_p"].get(key, p)
        agree = min(p_model, p_mkt)
        if src == "est":
            if p_model < 0.74 or p < 0.74:
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
        pen += 0.5 * max(0.0, abs(p_model - p_mkt) - 0.08)
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
    if not mkt and ext_d:
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
        if not free.enabled:
            fixtures = [f for f in fixtures if f["cov_pred"] is not False]
        log(f"🗂 Coverage filter: dropped {n0 - len(fixtures)} fixtures; {sum(1 for f in fixtures if f['cov_odds'])} of {len(fixtures)} remaining have odds coverage")
    lmem = load_state().get("league_mem3", {})
    today_ = datetime.now(timezone.utc).date()

    def mem_bad(lid):
        e = lmem.get(str(lid))
        try:
            return bool(e) and e["t"] >= 2 and e["v"] == 0 and (today_ - datetime.fromisoformat(e["d"]).date()).days <= 5
        except Exception:
            return False

    if not force:
        n0 = len(fixtures)
        fixtures = [f for f in fixtures if not mem_bad(f["league_id"]) or f["league_id"] in cfg["force_leagues"]
                    or (free.enabled and (f["league_id"] in FD_ALL or str(f.get("country") or "").lower() == "world"))]
        if n0 != len(fixtures):
            log(f"🧠 Skipped {n0 - len(fixtures)} fixtures in competitions that returned no usable team data in recent runs")
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
        log("🧠 Using remembered plan capabilities from an earlier run.")
    else:
        log("🔬 Probing what your API plan allows...")
        if plan_key == "free" and free_hist_sample:
            api.no_history = True
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
    log(f"💰 Stage 1: bookmaker odds for {len(cand)} candidate matches...")
    if caps["league_odds"] is not False:
        todo = {(c["league_id"], c["season"]) for c in cand} - probed_leagues
        with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
            for fu in as_completed([ex.submit(fetch_league_odds, lid, s_) for lid, s_ in todo]):
                try:
                    store_odds(fu.result())
                except Exception as e:
                    api.errors.append(f"odds: {e}")
    if free.enabled:
        n_free = 0
        for c in cand:
            if c["id"] not in odds_by:
                fo, src_ = free.odds_for(c["home"], c["away"], c["league_id"], date_str)
                if fo:
                    odds_by[c["id"]], odds_src[c["id"]] = fo, "free:" + str(src_)
                    n_free += 1
        if n_free:
            log(f"   🆓 {n_free} matches got bookmaker prices from free sources")
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
        out["reason"] = ("No bookmaker odds were returned for any candidate match. "
                         + ("API said: " + " | ".join(e[:170] for e in errs) + ". " if errs else "No error was reported. ")
                         + "Enable 'model-estimated odds' in Settings to continue with approximate prices.")
        return out
    free_first = free.enabled
    per = ((0.6 if (free_first and cfg.get("save_calls", True)) else 1.0) + (2.0 if caps["history"] else 0.0)
           + (0.6 if caps["fixture_odds"] else 0.0))
    deep = caps["history"] and (cfg["depth"] == "Full" or (cfg["depth"] == "Auto" and budget >= 150))

    def cost(ms, dp):
        return len(ms) * (per + (0.9 if dp else 0.0)) + (len({m["league_id"] for m in ms}) if dp else 0)

    left = budget - api.calls
    n = min(cfg["max_matches"], len(keep))
    if caps["history"] or not free_first:
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
            keep.sort(key=lambda c: 0 if c["id"] in free_cov else 1)
    sl = keep[:n]
    why_cut = (f"beyond your 'max matches' setting ({cfg['max_matches']})" if n >= min(cfg["max_matches"], len(keep)) else "beyond API call budget")
    for c in keep[n:]:
        out["excluded"].append({"league": c["league"], "match": f"{c['home']} v {c['away']}", "reason": why_cut})
    out["deep"], out["budget"] = deep, budget
    leagues = {(m["league_id"], m["season"]) for m in sl}
    est_calls = cost(sl, deep)
    log(f"🎯 Stage 2: data for {len(sl)} matches | depth={'FULL' if deep else 'LITE'} | est. ≤{est_calls:.0f} more API calls")
    if free.enabled:
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda lid: free.rows_for(lid, date_str), {m["league_id"] for m in sl}))
        free.elo_table(date_str)
    log("🩹 Fetching injuries & suspensions...")
    inj_by = defaultdict(list)
    inj_raw = api.resp("/injuries", {"date": date_str, "timezone": tz}, ttl=ttl(3600)) if can_spend(1) else []
    for i in inj_raw:
        inj_by[((i.get("fixture") or {}).get("id"), (i.get("team") or {}).get("id"))].append(i)
    inj_known = bool(inj_raw)
    progress(0.18)
    log("📊 Collecting predictions, team form (last 10) and H2H (free sources first)...")
    core = {}
    api_alive = {"v": True}

    def fetch_core(m):
        fh = free.team_form(m["home"], m["league_id"], date_str) if free.enabled else None
        fa = free.team_form(m["away"], m["league_id"], date_str) if free.enabled else None
        good_h, good_a = bool(fh and len(fh["results"]) >= 6), bool(fa and len(fa["results"]) >= 6)
        skip_pred = bool(free.enabled and cfg.get("save_calls", True) and good_h and good_a)
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
        log(f"   ⚠️ API-Football returned no usable team data for {len(probe)} probed matches - continuing ONLY with free-covered matches.")
    if not any(viable(c) for c in core.values()) and not free_cov:
        out["reason"] = ("Neither API-Football nor the free sources had team history for these matches. "
                         "Stopped early to protect your daily API quota.")
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
        log("💰 Fetching per-fixture bookmaker odds...")
        with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
            futs = {ex.submit(lambda mm: (api.get("/odds", {"fixture": mm["id"]}, ttl(1200), True).get("response") or []) if can_spend(1) else [], m): m
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
    hist, upc, st_by = {}, {}, {}
    if deep:
        log("🧮 Batch-fetching statistics, player ratings & lineups...")
        hist_ids = set()
        for c in core.values():
            for r in (c["rh"][:5] + c["ra"][:5]):
                if isinstance(r["id"], int):
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

        hist = batch(hist_ids, ttl(5 * 86400))
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
        out["warnings"].append(f"{thin}/{len(out['matches'])} matches have NO team-specific data and are excluded.")
    if free.intl_extra:
        free.used["espn-recent-results"] = free.intl_extra
    if free.enabled and free.used:
        log(f"🆓 Free sources helped: {dict(free.used)} | ≈{free.saved_calls} API calls avoided")
    progress(0.90)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# AI MATCH SELECTION — THE AI IS THE SOLE DECISION-MAKER
# ═════════════════════════════════════════════════════════════════════════════
AI_MATCH_PROMPT = """You are an expert football quant analyst at a sharp sportsbook. Python has collected data, run Monte Carlo simulations, and blended model probabilities with de-vigged bookmaker prices for every match below.

YOUR TASK: Select up to {target} safest and most valuable INDIVIDUAL match predictions from the supplied qualified matches, ranked safest first. Prefer returning {target} picks when enough suitable options exist, but never invent a match, market, price, or evidence to fill the target. Return an empty list only if no supplied option is suitable. Select at most one option per match.

DATA KEY:
- xG: expected goals home-away
- mdl: Monte Carlo model 1X2 probabilities (%)
- mkt: bookmaker 1/X/2 odds (fd = free source prices)
- ppg/gf/ga: recent form (points per game, goals for, goals against)
- TRAP: Python trap score (>=40 = dangerous, >=60 = very dangerous)
- dq: data quality 0-1
- H2H: head-to-head W-D-L
- Options: available bets with odds and blended probability

SELECTION RULES:
1. Prefer a mix of match winners, double chance (1X/X2), Over/Under goals, and BTTS when similarly safe options exist; safety takes priority over diversity.
2. STRICTLY AVOID "Either team to win (12)" unless probability is above 80% AND the data overwhelmingly supports it — it offers poor value at typical odds.
3. PREFER: Double Chance (1X or X2), Over 1.5 Goals, BTTS Yes, clear match winners with strong form evidence.
4. Prefer matches with TRAP score below 45; a higher-trap match may be chosen only with a clear reason.
5. Prioritize matches where the MODEL and the MARKET agree (small gap between mdl% and implied market%).
6. For each pick, state the most likely way it LOSES (the failure scenario).
7. 'inj:n/a' (unknown injuries) appears on every match when injury data is unavailable. Do NOT reject a match just for that; simply prefer higher-probability picks.
8. Use only a match and selection shown in the supplied data. Copy its option label and odds exactly; use the option's listed probability as "probability".

OUTPUT FORMAT: A compact JSON object with one key "picks" holding an array. Each array item has:
- "match": "Home v Away"
- "league": "League name"
- "pick": "The specific selection (e.g. 'Home Win', 'Home or Draw (1X)', 'Over 1.5 Goals', 'BTTS - Yes')"
- "odds": decimal odds (number)
- "probability": your estimated probability in percent (number)
- "confidence": "HIGH" or "MED"
- "reason": max 12 words explaining why this is safe
- "risk_if_fails": max 8 words describing how it loses

Output ONLY the JSON object, like {"picks": [ {...}, {...} ]}. No markdown. No explanation."""


def format_matches_for_ai(matches, tz, max_n=15):
    """Format match data into a comprehensive evidence pack for the AI."""
    lines = []
    for i, m in enumerate(matches[:max_n]):
        ko = datetime.fromtimestamp(m["ts"], ZoneInfo(tz)).strftime("%H:%M")
        ph, pa = m["ph"], m["pa"]
        p = m["p"]
        t = m["trap"]
        opts = []
        for l in m.get("legs", [])[:4]:
            opts.append(f"{l['label']}@{l['odds']:.2f}(p:{_pct(l['p'])}%)")
        opts_str = " | ".join(opts) if opts else "no clear options"
        o = m["odds"]
        if all(k in o for k in ("1", "X", "2")):
            mkt_str = f"mkt {o['1']['odds']:.2f}/{o['X']['odds']:.2f}/{o['2']['odds']:.2f}"
        elif m.get("ext_d"):
            mkt_str = f"elo/api 1:{_pct(m['ext_d']['1'])}% X:{_pct(m['ext_d']['X'])}% 2:{_pct(m['ext_d']['2'])}%"
        else:
            mkt_str = "no market prices"
        trap_str = f" | TRAP:{t['risk']}" if t["risk"] >= 25 else ""
        h2h_str = f" | H2H:{m['h2h']['w']}-{m['h2h']['d']}-{m['h2h']['l']}" if m.get("h2h") else ""
        inj_str = "inj:n/a" if not m.get("inj_known") else ""
        block = (f"{i + 1}. {m['league']} | {m['home']} v {m['away']} ({ko}) | dq:{m['dq']}{trap_str}\n"
                 f"   xG:{m['lam_h']:.2f}-{m['lam_a']:.2f} | mdl 1:{_pct(m['model_p']['1'])}% X:{_pct(m['model_p']['X'])}% 2:{_pct(m['model_p']['2'])}% | {mkt_str}\n"
                 f"   Home: {ph.get('form5', '?')} ppg{ph.get('ppg5', 0):.1f} gf{ph.get('gf_w', 0):.1f} ga{ph.get('ga_w', 0):.1f} | "
                 f"Away: {pa.get('form5', '?')} ppg{pa.get('ppg5', 0):.1f} gf{pa.get('gf_w', 0):.1f} ga{pa.get('ga_w', 0):.1f}\n"
                 f"   BTTS:{_pct(p['BTTS_Y'])}% O1.5:{_pct(p['O1.5'])}% O2.5:{_pct(p['O2.5'])}% U2.5:{_pct(p['U2.5'])}%{h2h_str}{' | ' + inj_str if inj_str else ''}\n"
                 f"   Options: {opts_str}")
        lines.append(block)
    return "\n\n".join(lines)


def est_tokens(text, cpt):
    return int(len(text) / cpt) + 1


class TpmGovernor:
    def __init__(self, window_limit, clock=time.monotonic, sleep=time.sleep):
        self.limit = int(window_limit)
        self.clock, self.sleep = clock, sleep
        self.events: List[list] = []

    def _prune(self, now):
        self.events = [e for e in self.events if now - e[0] < 60.0]

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
                log_fn(f"   ⏳ Groq allows ~{self.limit + WINDOW_MARGIN} tokens/minute: waiting {wait:.0f}s...")
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
    t = text[text.find("["):] if "[" in text else ""
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
    c = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.I)
    c = re.sub(r"\s*```$", "", c).strip()
    for cand in (c, c[c.find("["): c.rfind("]") + 1] if "[" in c else ""):
        if cand:
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", cand)), False
            except Exception:
                pass
    return _salvage_json(c), True


def _extract_picks(parsed):
    """Find the list of pick dicts in whatever shape the model returned."""
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        for k in ("picks", "selections", "top_picks", "predictions", "recommendations", "results", "bets"):
            v = parsed.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        for v in parsed.values():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                return v
        if ("match" in parsed or "fixture" in parsed) and ("pick" in parsed or "selection" in parsed or "bet" in parsed):
            return [parsed]
    return []


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


def call_groq_for_picks(system_prompt, user_prompt, budget, cpt, effort=None, gov=None, ceiling=None, log_fn=None):
    """Single Groq call to get AI match picks."""
    api_key = get_secret("GROQ_API_KEY", "").strip()
    log = []
    info = {"attempts": log, "model": None, "prompt_tokens": 0, "completion_tokens": 0}
    if not api_key:
        log.append({"model": "-", "status": "GROQ_API_KEY not set"})
        info["status"] = "MISSING_KEY"
        return None, info
    ceiling = ceiling or (GROQ_TPM_DEFAULT - TPM_MARGIN)
    dead = _dead_models()
    models = [m for m in GROQ_MODELS if m not in dead] or GROQ_MODELS[:2]
    for model in models:
        cfg = GROQ_MODEL_CONFIG.get(model, GROQ_DEFAULT_CFG)
        eff = effort or cfg.get("reasoning_effort")
        upper = cfg["max_completion_tokens"]
        limit = ceiling
        p_est = est_tokens(system_prompt + user_prompt, cpt) + 40
        cap = min(upper, budget.remaining - p_est - 60, limit - p_est - 60)
        if cap < MIN_COMPLETION:
            log.append({"model": model, "status": f"SKIPPED(prompt~{p_est}, room={cap})"})
            continue
        handle = None
        if gov:
            info["waited"] = info.get("waited", 0) + gov.wait_for(p_est + cap, log_fn)
            handle = gov.charge(p_est + cap)
        payload = {"model": model, "temperature": 1.0 if model.startswith("openai/gpt-oss") else 0.2,
                 "max_completion_tokens": int(cap),
                   "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}
        payload["response_format"] = {"type": "json_object"}
        if cfg.get("supports_reasoning_effort") and eff:
            payload["reasoning_effort"] = eff if model.startswith("openai/") or eff == "none" else "none"
            if model.startswith("openai/"):
                payload["include_reasoning"] = False
        for attempt in range(2):
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
                pt, ct = u.get("prompt_tokens", p_est), u.get("completion_tokens", 0)
                total = u.get("total_tokens", pt + ct)
                budget.used += total
                if handle:
                    gov.settle(handle, total)
                info["prompt_tokens"] += pt
                info["completion_tokens"] += ct
                ch = (d.get("choices") or [{}])[0]
                content = (ch.get("message") or {}).get("content") or ""
                parsed, salvaged = _parse_json_ex(content) if content else (None, False)
                found = _extract_picks(parsed) if parsed is not None else []
                if found:
                    log.append({"model": model, "status": f"SUCCESS ({len(found)} picks)"})
                    info.update({"model": model, "status": "SUCCESS"})
                    return found, info
                why_ = ("EMPTY_CONTENT" if not content.strip() else
                        "EMPTY_PICKS_LIST" if parsed is not None else "UNPARSEABLE_JSON")
                log.append({"model": model, "status": f"{why_} finish={ch.get('finish_reason')} ({ct} tokens) raw={content.strip()[:220]!r}"})
                if ch.get("finish_reason") == "length":
                    info["status"] = "TRUNCATED"
                    break
                break
            elif r.status_code == 404:
                _mark_dead(model)
                log.append({"model": model, "status": "HTTP_404 model unavailable"})
                break
            elif r.status_code == 429:
                log.append({"model": model, "status": f"RATE_LIMIT: {r.text[:100]}"})
                time.sleep(15)
            elif r.status_code == 400 and "response_format" in payload:
                log.append({"model": model, "status": f"HTTP_400 retrying plain: {r.text[:100]}"})
                for k_ in ("response_format", "include_reasoning", "reasoning_effort"):
                    payload.pop(k_, None)
                continue
            elif r.status_code in NON_RETRYABLE or r.status_code == 413:
                log.append({"model": model, "status": f"HTTP_{r.status_code}: {r.text[:140]}"})
                break
            else:
                log.append({"model": model, "status": f"HTTP_{r.status_code}"})
                time.sleep(3)
    info["status"] = info.get("status", "FAILED")
    return None, info


# ═════════════════════════════════════════════════════════════════════════════
# AI ANALYSIS STAGE
# ═════════════════════════════════════════════════════════════════════════════
def _num(x, d=0.0):
    try:
        return float(x) if x not in (None, "") else d
    except Exception:
        return d


def run_ai_stage(elig, cfg, log, progress=lambda x: None):
    """AI reads ALL qualified matches and picks the top N safest. Python NEVER picks."""
    warnings = []
    state = load_state()
    tpm = int(min(int(cfg.get("tpm", GROQ_TPM_DEFAULT)), int(state.get("tpm_limit", 10 ** 9))))
    ceiling = max(3500, tpm - TPM_MARGIN)
    gov = TpmGovernor(max(ceiling + 50, tpm - WINDOW_MARGIN))
    budget = TokenBudget(int(cfg.get("run_tokens", RUN_TOKEN_CAP_DEFAULT)))
    cpt = float(state.get("cpt", DEFAULT_CHARS_PER_TOKEN))
    eff = cfg.get("effort") if cfg.get("effort") in EFFORT_RESERVE else "low"
    use_ai = cfg.get("use_ai", True)
    target = max(1, min(int(cfg.get("target_picks", TARGET_PICKS)), len(elig))) if elig else 0
    if not use_ai:
        return {"picks": [], "ai_note": "AI disabled in settings.", "warnings": ["AI is disabled."], "model": None, "passes": [], "tokens": 0}
    if not elig:
        return {"picks": [], "ai_note": "No matches passed the evidence gate.", "warnings": [], "model": None, "passes": [], "tokens": 0}
    ranked = sorted(elig, key=lambda m: -(sum(l["rank"] for l in m.get("legs", [])[:2]) / max(1, len(m.get("legs", [])[:2]))))
    prompt = AI_MATCH_PROMPT.replace("{target}", str(target))
    n_pool = min(len(ranked), 14)
    while True:
        ai_pool = ranked[:n_pool]
        data_str = format_matches_for_ai(ai_pool, cfg["tz"], max_n=len(ai_pool))
        user_msg = f"{prompt}\n\nMATCH DATA:\n{data_str}"
        if n_pool <= 3 or est_tokens(user_msg, cpt) + 1700 <= ceiling:
            break
        n_pool -= 1
    log(f"🤖 AI Analyst: sending {len(ai_pool)} qualified matches for selection (target: {target} picks)...")
    sys_msg = "You are an expert football quant analyst. Output ONLY a valid JSON object with a 'picks' array. No markdown. No explanation."
    picks, info = call_groq_for_picks(sys_msg, user_msg, budget, cpt, eff, gov, ceiling, log)
    progress(1.0)
    if picks and isinstance(picks, list):
        valid_picks = []
        for item in picks:
            mt_ = item.get("match") or item.get("fixture") or item.get("game")
            pk_ = item.get("pick") or item.get("selection") or item.get("bet") or item.get("market")
            if isinstance(item, dict) and mt_ and pk_:
                valid_picks.append({
                    "match": str(mt_),
                    "league": str(item.get("league", "")),
                    "pick": str(pk_),
                    "odds": _num(item.get("odds")),
                    "probability": _num(item.get("probability")),
                    "confidence": str(item.get("confidence", "MED")),
                    "reason": str(item.get("reason", "")),
                    "risk_if_fails": str(item.get("risk_if_fails", "")),
                })
        if valid_picks:
            log(f"   ✅ AI returned {len(valid_picks)} picks using {info.get('model', 'unknown')}")
            return {"picks": valid_picks, "ai_note": f"AI selected {len(valid_picks)} picks ({budget.used} tokens used)",
                    "warnings": warnings, "model": info.get("model"), "passes": info.get("attempts", []), "tokens": budget.used}
        warnings.append("AI returned data but no valid picks could be extracted.")
    else:
        warnings.append(f"AI analysis failed: {info.get('status', 'unknown error')}. Check AI passes for details.")
    last_ = ((info.get("attempts") or [{}])[-1]).get("status", "")
    return {"picks": [], "ai_note": f"AI failed: {info.get('status', 'unknown')} - last attempt: {last_}",
            "warnings": warnings, "model": info.get("model"), "passes": info.get("attempts", []), "tokens": budget.used}


# ═════════════════════════════════════════════════════════════════════════════
# Telegram
# ═════════════════════════════════════════════════════════════════════════════
def send_telegram(text):
    tok, chat = get_secret("TELEGRAM_BOT_TOKEN"), get_secret("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return False, "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing"
    chunks, cur = [], ""
    for para in text.split("\n"):
        if len(cur) + len(para) + 2 > 3800:
            chunks.append(cur)
            cur = ""
        cur += (("\n" if cur else "") + para)
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
    L = [f"⚽ <b>Der-AI Football Quant Desk</b>\n📅 {_esc(res['date'])} | {res['n_analysed']} matches analysed | {res['n_ai']} sent to AI"]
    if res.get("relaxed"):
        L.append("⚠️ <b>Thin slate - thresholds relaxed:</b> " + _esc("; ".join(res["relaxed"])))
    if not res.get("picks"):
        L.append("🛑 <b>NO PICKS TODAY</b> - AI found no matches meeting the safety standard.")
    else:
        L.append(f"🤖 <b>AI selected {len(res['picks'])} safest matches:</b>\n")
        nums = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
        for i, p in enumerate(res["picks"][:10]):
            n = nums[i] if i < len(nums) else f"{i + 1}."
            conf_icon = "🟢" if p.get("confidence") == "HIGH" else "🟡"
            L.append(f"{n} <b>{_esc(p.get('match', ''))}</b> [{_esc(p.get('league', ''))}]\n"
                     f"   ➜ <b>{_esc(p.get('pick', ''))}</b> @{_num(p.get('odds')):.2f} ({p.get('probability', 0)}%)\n"
                     f"   {conf_icon} {_esc(p.get('reason', ''))}\n"
                     f"   ⚠️ Loses if: {_esc(p.get('risk_if_fails', ''))}")
    foot = (f"\n🤖 {_esc(res.get('model') or 'AI')} | AI tokens {res.get('tokens', 0)} | "
            f"API calls {res.get('api_calls', 0)}"
            + (" | 🆓 free sources" if (res.get("free") or {}).get("used") else "")
            + "\n⚠️ Odds are snapshots; betting carries risk.")
    L.append(foot)
    return "\n".join(L)


def build_nobet_message(res, tz):
    return (f"⚽ <b>Der-AI Football Quant Desk</b>\n📅 {_esc(res['date'])}\n🛑 <b>NO PICKS TODAY</b>\n"
            f"{_esc(res.get('reason', 'AI found no matches meeting the safety standard.'))}\n"
            f"API calls used: {res.get('api_calls', 0)}.")


# ═════════════════════════════════════════════════════════════════════════════
# State / Ledger
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


def save_picks_to_ledger(date_str, picks):
    state = load_state()
    state.setdefault("history", []).append({
        "created": datetime.now(timezone.utc).isoformat(), "date": date_str, "status": "PENDING",
        "picks": picks})
    state["history"] = state["history"][-60:]
    save_state(state)


# ═════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════════
def run_full_analysis(cfg, log, progress):
    t0 = time.time()
    api_key = get_secret("API_FOOTBALL_KEY") or get_secret("APISPORTS_KEY")
    free = FreeData(enabled=bool(cfg.get("use_free", True)))
    if not api_key and not free.enabled:
        return {"error": "API_FOOTBALL_KEY is not set and free sources are switched off."}
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
        base = {"date": cfg["date"], "picks": [], "ai_note": "", "model": None, "tokens": 0,
                "n_analysed": len(matches), "n_ai": 0, "matches": matches, "id_map": {},
                "api_calls": api.calls, "cache_hits": api.cache_hits, "warnings": data["warnings"],
                "cat_counts": data["cat_counts"], "plan": data["plan"], "deep": data.get("deep"),
                "api_errors": api.error_summary(), "seconds": round(time.time() - t0, 1),
                "excluded": data["excluded"], "no_bet": False, "reason": "", "caps": data.get("caps"),
                "cfg": cfg, "quota": (api.remaining, api.limit), "free": free.summary(), "relaxed": [],
                "passes": [], "wasted_tokens": 0, "fallback_model": False}
        base.update(kw)
        return base

    if not matches:
        return result(no_bet=True, reason=data.get("reason") or "No matches could be analysed.")

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
    relaxed = []
    if cfg.get("auto_relax", True) and len(elig) < MIN_PICKS:
        for label, upd in (("minimum data quality lowered to 0.40", {"min_dq": min(cfg["min_dq"], 0.40)}),
                           ("matches with only borderline legs allowed", {"allow_b_only": True}),
                           ("model-estimated odds allowed", {"allow_est": True})):
            if len(elig) >= MIN_PICKS:
                break
            trial = dict(cur, **upd)
            new = run_gate(trial)
            if len(new) > len(elig):
                cur, elig = trial, new
                relaxed.append(label)
        elig = run_gate(cur)
    cur["relaxed_notes"] = bool(relaxed)
    cfg = dict(cur, relaxed=relaxed)
    for m in matches:
        if not m["eligible"]:
            data["excluded"].append({"league": m["league"], "match": f"{m['home']} v {m['away']}", "reason": m["excl"]})
    if relaxed:
        data["warnings"].append("Thin slate - thresholds relaxed: " + "; ".join(relaxed))
    log(f"🛡 Evidence gate: {len(elig)}/{len(matches)} matches qualify" + (f" (after relaxing: {'; '.join(relaxed)})" if relaxed else ""))
    if not elig:
        return result(no_bet=True, cfg=cfg, relaxed=relaxed,
                      reason="No matches passed the evidence gate even after relaxing. Betting on thin data is guessing.")

    stage = run_ai_stage(elig, cfg, log, progress)
    data["warnings"] += stage["warnings"]
    res = result(picks=stage["picks"], ai_note=stage["ai_note"], model=stage["model"],
                 tokens=stage.get("tokens", 0), n_ai=len(elig), cfg=cfg, relaxed=relaxed,
                 passes=stage.get("passes", []))
    if stage["picks"]:
        save_picks_to_ledger(cfg["date"], stage["picks"])
    else:
        res["no_bet"] = True
        res["reason"] = stage["ai_note"] or "AI found no matches meeting the safety standard."
    return res


def rerun_ai(res, cfg, log):
    elig = [m for m in res["matches"] if m.get("eligible") and m.get("legs")]
    if not elig:
        return res
    stage = run_ai_stage(elig, cfg, log)
    old = [w for w in res.get("warnings", []) if "AI" not in w]
    new = dict(res, picks=stage["picks"], ai_note=stage["ai_note"], model=stage["model"], tokens=stage.get("tokens", 0),
               n_ai=len(elig), warnings=old + stage["warnings"], cfg=cfg, passes=stage.get("passes", []))
    if new["picks"]:
        save_picks_to_ledger(cfg["date"], new["picks"])
        new["no_bet"] = False
    else:
        new["no_bet"] = True
        new["reason"] = stage["ai_note"] or "AI found no safe picks."
    return new


def test_groq_connection():
    key = get_secret("GROQ_API_KEY", "").strip()
    if not key:
        return [("-", "GROQ_API_KEY is not set")]
    rows = []
    for model in GROQ_MODELS:
        cfg = GROQ_MODEL_CONFIG.get(model, GROQ_DEFAULT_CFG)
        payload = {"model": model, "max_completion_tokens": 120,
                 "temperature": 1.0 if model.startswith("openai/gpt-oss") else 0,
                 "response_format": {"type": "json_object"},
                   "messages": [{"role": "user", "content": 'Return exactly: {"ok": true}'}]}
        if cfg.get("supports_reasoning_effort"):
            payload["reasoning_effort"] = "low" if model.startswith("openai/") else "none"
            if model.startswith("openai/"):
                payload["include_reasoning"] = False
        try:
            r = requests.post(GROQ_API_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=30)
            if r.status_code == 200:
                rows.append((model, f"✅ working ({(r.json().get('usage') or {}).get('total_tokens', '?')} tokens)"))
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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches Analysed", res["n_analysed"])
    c2.metric("Sent to AI", res["n_ai"])
    c3.metric("AI Picks", len(res.get("picks", [])))
    c4.metric("AI Tokens", res.get("tokens", 0))
    if res.get("quota") and res["quota"][0] is not None:
        st.caption(f"📶 API quota left: **{res['quota'][0]}** of {res['quota'][1]} (resets 00:00 UTC)")
    fr = res.get("free") or {}
    if fr.get("used") or fr.get("saved_calls"):
        st.caption("🆓 Free sources: " + " · ".join(f"{k} ×{v}" for k, v in fr.get("used", {}).items())
                   + (f" · ≈{fr['saved_calls']} API calls avoided" if fr.get("saved_calls") else ""))
    if res.get("relaxed"):
        st.info("🪜 Thin slate - thresholds relaxed: " + "; ".join(res["relaxed"]))
    if res.get("no_bet"):
        st.error(f"🛑 **NO PICKS** - {res.get('reason', 'AI found no safe matches.')}")
        if res.get("api_errors"):
            st.markdown("**API messages:**")
            st.code("\n".join(res["api_errors"]), language="text")
    if res.get("picks"):
        note = res.get("ai_note") or f"{len(res['picks'])} AI picks"
        st.success(f"✅ {note}")
        st.markdown("### 🏆 AI Top Picks")
        rows = []
        for i, p in enumerate(res["picks"]):
            rows.append({
                "#": i + 1,
                "Match": p.get("match", ""),
                "League": p.get("league", ""),
                "Pick": p.get("pick", ""),
                "Odds": p.get("odds", ""),
                "Prob%": p.get("probability", ""),
                "Conf": p.get("confidence", ""),
                "Reason": p.get("reason", ""),
                "Loses if": p.get("risk_if_fails", ""),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        if not res.get("no_bet"):
            st.warning("🤖 AI returned no picks. Press 'Re-run AI analysis' to try again.")
    with st.expander("📋 All analysed matches (Monte Carlo + market blend)"):
        rows = []
        for m in res.get("matches", []):
            p = m["p"]
            rows.append({"Cat": m["cat"], "League": m["league"], "Match": f"{m['home']} v {m['away']}",
                         "KO": datetime.fromtimestamp(m["ts"], ZoneInfo(tz)).strftime("%H:%M"),
                         "Data": BASIS_TAG.get(m["basis"]), "xG": f"{m['lam_h']:.2f}-{m['lam_a']:.2f}",
                         "1": _pct(p["1"]), "X": _pct(p["X"]), "2": _pct(p["2"]),
                         "1X": _pct(p["1X"]), "X2": _pct(p["X2"]),
                         "BTTS": _pct(p["BTTS_Y"]), "O1.5": _pct(p["O1.5"]), "O2.5": _pct(p["O2.5"]),
                         "Trap": m["trap"]["risk"], "DQ": m["dq"],
                         "Qualified": "yes" if m.get("eligible") else "no"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if res.get("excluded"):
        with st.expander(f"🛡 Excluded by evidence gate ({len(res['excluded'])})"):
            st.dataframe(pd.DataFrame(res["excluded"]), hide_index=True, use_container_width=True)
    with st.expander("🛠 AI passes / errors"):
        if res.get("passes"):
            st.dataframe(pd.DataFrame(res["passes"]), hide_index=True, use_container_width=True)
        else:
            st.caption("No AI attempts were recorded.")
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
    tab1, tab2, tab3, tab4 = st.tabs(["⚽ Match Analysis", "📜 Pick History", "🔔 Notifications", "⚙️ Settings"])

    with tab4:
        st.header("⚙️ Settings")
        st.info("Secrets needed: `API_FOOTBALL_KEY`, `GROQ_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.")
        s1, s2 = st.columns(2)
        tz = s1.text_input("Time zone", DEFAULT_TZ)
        max_matches = s1.slider("Max matches to collect deep data for", 10, 60, 30)
        target_picks = s1.slider("Target number of AI picks", 3, 15, 10)
        max_calls = s1.number_input("Max API-Football calls per run", 30, 5000, 95, step=5)
        breadth = s1.selectbox("Search breadth", ["Focused", "Balanced", "Wide"], index=1)
        depth = s1.selectbox("Data depth", ["Auto", "Full", "Lite"])
        sims = s2.select_slider("Monte Carlo simulations per match", [10000, 20000, 40000, 80000], 40000)
        w_model = s2.slider("Weight of model vs bookmaker price", 0.2, 0.8, 0.5, 0.05)
        bookmaker = s2.number_input("Preferred bookmaker id", 1, 500, 8)
        effort = s2.selectbox("AI reasoning effort", ["low", "medium", "high"], index=0)
        exclude_minor = s2.checkbox("Exclude youth / women / reserve teams", True)
        exclude_lower = s2.checkbox("Exclude amateur / lower divisions", True)
        force_leagues = {int(x) for x in re.findall(r"\d+", s2.text_input("Always include league ids", ""))}
        rate_override = int(s2.number_input("Requests/minute override (0=auto)", 0, 900, 0))
        send_tg = st.checkbox("Send picks to Telegram automatically", True)
        use_ai = st.checkbox("Use AI (Groq)", True)
        if st.button("🧪 Test AI connection"):
            for mdl, status in test_groq_connection():
                st.write(f"`{mdl}` → {status}")
        min_dq = st.slider("Minimum data-quality score", 0.3, 0.9, 0.5, 0.05)
        allow_est = st.checkbox("Allow model-estimated odds when no bookmaker prices", True)
        st.markdown("##### 🤖 AI settings")
        a1, a2 = st.columns(2)
        tpm = a1.number_input("Groq tokens/minute limit", 4000, 30000, GROQ_TPM_DEFAULT, step=500)
        run_tokens = a2.number_input("Max AI tokens per analysis", 10000, 120000, RUN_TOKEN_CAP_DEFAULT, step=2000)
        auto_relax = st.checkbox("Auto-relax thresholds when few matches qualify", True)
        wide_menu = st.checkbox("Include borderline legs for AI to consider", True)
        st.markdown("##### 🆓 Free data sources")
        use_free = st.checkbox("Use free sources first (0 API calls)", True)
        save_calls = st.checkbox("Skip API predictions when free sources cover the match", True)
        if st.button("📶 Check API quota"):
            key_ = get_secret("API_FOOTBALL_KEY")
            if key_:
                s_ = ApiFootball(key_, 10).status()
                st.info(f"Plan: {s_.get('plan')} · used {s_.get('used')} of {s_.get('limit')} today")
            else:
                st.warning("API_FOOTBALL_KEY not set.")
        if st.button("🧹 Clear cache"):
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
        force = st.checkbox("Ignore cache (fresh data)", False)
        st.caption("The AI reads all qualified matches and selects the safest picks. Python NEVER picks matches — it only collects data and presents it to the AI.")
        if st.button("🧠 Analyse Football Matches Now", type="primary"):
            cfg = {"date": d.strftime("%Y-%m-%d"), "tz": tz, "max_matches": max_matches, "target_picks": target_picks,
                   "max_calls": int(max_calls), "depth": depth, "sims": int(sims), "w_model": w_model,
                   "bookmaker": int(bookmaker), "effort": effort, "exclude_minor": exclude_minor,
                   "breadth": breadth, "exclude_lower": exclude_lower, "force_leagues": force_leagues,
                   "rate_override": rate_override, "force": force, "use_ai": use_ai, "min_dq": min_dq,
                   "allow_est": allow_est, "tpm": int(tpm), "run_tokens": int(run_tokens),
                   "use_free": use_free, "save_calls": save_calls, "auto_relax": auto_relax,
                   "wide_menu": wide_menu}
            bar = st.progress(0.0)
            box = st.status("Running full analysis...", expanded=True)

            def log(m):
                box.write(m)

            def prog(x):
                bar.progress(min(1.0, float(x)))

            try:
                res = run_full_analysis(cfg, log, prog)
            except Exception:
                res = {"error": "Unexpected error:\n" + traceback.format_exc()}
            bar.progress(1.0)
            if "error" in res:
                box.update(label="Analysis stopped", state="error")
                st.error(res["error"])
            else:
                box.update(label=f"Done in {res['seconds']}s" + (" - NO PICKS" if res.get("no_bet") else ""), state="complete")
                st.session_state.last_result = res
                st.session_state.last_tz = tz
                if send_tg:
                    msg = (build_nobet_message(res, tz) if res.get("no_bet")
                           else build_telegram_message(res, tz))
                    ok, why = send_telegram(msg)
                    (st.success if ok else st.warning)("✅ Sent to Telegram" if ok else f"Telegram failed: {why}")
                    st.session_state.notifications.append({"time": datetime.now().strftime("%H:%M"), "ok": ok,
                                                           "msg": f"{cfg['date']}: " + ("NO-PICKS " if res.get("no_bet") else f"{len(res.get('picks', []))} picks ") + ("sent" if ok else f"failed ({why})")})
        if st.session_state.last_result:
            render_results(st.session_state.last_result, st.session_state.get("last_tz", tz))
            lr = st.session_state.last_result
            if lr and lr.get("matches") and any(m.get("eligible") for m in lr.get("matches", [])):
                st.divider()
                if st.button("🔁 Re-run AI analysis (reuses collected data, 0 API calls)"):
                    box2 = st.status("Re-running AI...", expanded=True)
                    try:
                        new = rerun_ai(lr, dict(lr["cfg"], effort=effort, use_ai=True, tpm=int(tpm), run_tokens=int(run_tokens), target_picks=target_picks), box2.write)
                        st.session_state.last_result = new
                        if send_tg and new.get("picks"):
                            ok, why = send_telegram(build_telegram_message(new, tz))
                            st.session_state.notifications.append({"time": datetime.now().strftime("%H:%M"), "ok": ok,
                                                                   "msg": "AI re-run: " + f"{len(new['picks'])} picks " + ("sent" if ok else f"failed ({why})")})
                        box2.update(label="AI re-run finished", state="complete")
                        st.rerun()
                    except Exception:
                        box2.update(label="AI re-run failed", state="error")
                        st.error(traceback.format_exc())

    with tab2:
        st.header("📜 Pick History")
        state = load_state()
        hist = state.get("history", [])
        if hist:
            for entry in reversed(hist[-20:]):
                with st.expander(f"{entry.get('date', '?')} — {len(entry.get('picks', []))} picks"):
                    if entry.get("picks"):
                        st.dataframe(pd.DataFrame(entry["picks"]), hide_index=True, use_container_width=True)
                    else:
                        st.info("No picks recorded.")
        else:
            st.info("No history yet. Run an analysis to start building history.")

    with tab3:
        st.header("🔔 Notifications")
        for n in reversed(st.session_state.notifications):
            (st.success if n["ok"] else st.warning)(f"[{n['time']}] {n['msg']}")
        if not st.session_state.notifications:
            st.info("No notifications yet.")


if __name__ == "__main__":
    main()
