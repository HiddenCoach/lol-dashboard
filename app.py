import os
import time
import io
import base64
from typing import Any, Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

import plotly.graph_objects as go

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# Riot routing
# =========================
EUW_PLATFORM = "euw1"     # summoner-v4 / league-v4 (platform routing)
EUW_REGIONAL = "europe"   # match-v5 / account-v1 (regional routing)

QUEUE_MAP = {
    "Toutes": None,
    "SoloQ (420)": 420,
    "Flex (440)": 440,
}

# =========================
# Map bounds (Summoner's Rift map11)
# =========================
MAP_MIN_X, MAP_MIN_Y = -120, -120
MAP_MAX_X, MAP_MAX_Y = 14870, 14980


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def norm_xy(x: float, y: float) -> Tuple[float, float]:
    nx = (x - MAP_MIN_X) / (MAP_MAX_X - MAP_MIN_X)
    ny = (y - MAP_MIN_Y) / (MAP_MAX_Y - MAP_MIN_Y)
    return clamp01(nx), clamp01(ny)


def norm_to_px(nx: float, ny: float, w: int, h: int) -> Tuple[float, float]:
    # Image origin: top-left. Timeline coords increase y upward -> invert for display.
    px = nx * w
    py = (1.0 - ny) * h
    return px, py


# =========================
# Riot Client
# =========================
class RiotClient:
    def __init__(self, api_key: Optional[str] = None, timeout_s: int = 20, sleep_on_429_s: int = 2):
        self.api_key = api_key or os.getenv("RIOT_API_KEY")
        if not self.api_key:
            raise RuntimeError("RIOT_API_KEY manquant. Ajoute-le dans Streamlit Cloud > Settings > Secrets.")
        self.timeout_s = timeout_s
        self.sleep_on_429_s = sleep_on_429_s

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        for attempt in range(5):
            r = requests.get(url, headers={"X-Riot-Token": self.api_key}, params=params, timeout=self.timeout_s)
            if r.status_code == 429:
                time.sleep(self.sleep_on_429_s * (attempt + 1))
                continue
            if not r.ok:
                raise RuntimeError(f"Erreur Riot API {r.status_code}: {r.text[:800]}")
            return r.json()
        raise RuntimeError("Rate limit (429) persistant. Baisse le nombre de matchs/joueurs.")

    # account-v1 (regional): Riot ID -> puuid
    def get_puuid_by_riot_id(self, game_name: str, tag_line: str) -> str:
        url = f"https://{EUW_REGIONAL}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        return self._get(url)["puuid"]

    # match-v5 (regional)
    def get_match_ids_by_puuid(self, puuid: str, count: int = 20, queue: Optional[int] = None) -> List[str]:
        url = f"https://{EUW_REGIONAL}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params: Dict[str, Any] = {"count": count}
        if queue is not None:
            params["queue"] = queue
        return self._get(url, params=params)

    def get_match(self, match_id: str) -> Dict[str, Any]:
        url = f"https://{EUW_REGIONAL}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return self._get(url)

    def get_timeline(self, match_id: str) -> Dict[str, Any]:
        url = f"https://{EUW_REGIONAL}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
        return self._get(url)


# =========================
# Data Dragon minimap map11.png
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_latest_ddragon_version() -> str:
    r = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=15)
    r.raise_for_status()
    return r.json()[0]


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_minimap_image() -> Image.Image:
    ver = get_latest_ddragon_version()
    url = f"https://ddragon.leagueoflegends.com/cdn/{ver}/img/map/map11.png"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")


def minimap_to_base64_png(minimap: Image.Image) -> str:
    buf = io.BytesIO()
    minimap.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =========================
# Helpers
# =========================
def parse_riot_ids(text: str) -> List[Tuple[str, str, str]]:
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or "#" not in line:
            continue
        game, tag = line.split("#", 1)
        out.append((line, game.strip(), tag.strip()))
    return out


def participant_id_for_puuid(match: Dict[str, Any], puuid: str) -> int:
    for i, p in enumerate(match["info"]["participants"], start=1):
        if p.get("puuid") == puuid:
            return i
    raise ValueError("PUUID non trouvé dans ce match.")


def participant_meta(match: Dict[str, Any], puuid: str) -> Dict[str, Any]:
    for p in match["info"]["participants"]:
        if p.get("puuid") == puuid:
            team_id = p.get("teamId")
            return {
                "championName": p.get("championName") or "UNKNOWN",
                "teamPosition": p.get("teamPosition") or "UNKNOWN",
                "teamId": team_id,
                "side": "BLUE" if team_id == 100 else "RED",
                "win": bool(p.get("win", False)),
            }
    return {"championName": "UNKNOWN", "teamPosition": "UNKNOWN", "teamId": 0, "side": "UNKNOWN", "win": False}


def iter_kill_events(timeline: Dict[str, Any]):
    for fr in timeline["info"]["frames"]:
        ts = fr.get("timestamp", 0)
        for ev in fr.get("events", []):
            if ev.get("type") == "CHAMPION_KILL":
                yield ts, ev


# =========================
# Role-aware zone classification
# - TOP_SIDE vs BOT_SIDE (map-side), NOT "toplane".
# - If role is BOTTOM/UTILITY and death on TOP_SIDE -> OFFSIDE_TOP (coach-friendly)
# - If role is TOP and death on BOT_SIDE -> OFFSIDE_BOT
# =========================
def role_aware_zone(x: float, y: float, role: str) -> str:
    nx, ny = norm_xy(x, y)
    d_diag = abs(nx - ny)
    center_dist = ((nx - 0.5) ** 2 + (ny - 0.5) ** 2) ** 0.5

    # Mid / River
    if d_diag < 0.05 and center_dist < 0.30:
        base = "MID"
    elif d_diag < 0.07:
        base = "RIVER"
    else:
        # IMPORTANT: top side = ny > nx (baron side), bot side = nx > ny (dragon side)
        base = "TOP_SIDE" if ny > nx else "BOT_SIDE"

    # Lane corridor hints (only if far from mid/river)
    top_lane_corridor = (nx < 0.38 and ny > 0.62)
    bot_lane_corridor = (nx > 0.62 and ny < 0.38)
    if base == "TOP_SIDE" and top_lane_corridor:
        base = "TOP_LANE_AREA"
    if base == "BOT_SIDE" and bot_lane_corridor:
        base = "BOT_LANE_AREA"

    role = (role or "UNKNOWN").upper()

    # OFFSIDE tagging by role (fix your complaint)
    if role in {"BOTTOM", "UTILITY"}:
        if base in {"TOP_SIDE", "TOP_LANE_AREA"}:
            return "OFFSIDE_TOP"
        return base
    if role == "TOP":
        if base in {"BOT_SIDE", "BOT_LANE_AREA"}:
            return "OFFSIDE_BOT"
        return base

    return base


def extract_deaths_window(
    timeline: Dict[str, Any],
    victim_pid: int,
    start_min: float,
    end_min: float,
    match_id: str,
    side: str,
    role: str,
) -> pd.DataFrame:
    out = []
    for ts, ev in iter_kill_events(timeline):
        if ev.get("victimId") != victim_pid:
            continue
        pos = ev.get("position")
        if not pos:
            continue
        t_min = ts / 60000.0
        if t_min < start_min or t_min > end_min:
            continue
        x = float(pos["x"])
        y = float(pos["y"])
        out.append({
            "matchId": match_id,
            "minute": float(t_min),
            "x": x,
            "y": y,
            "side": side,     # per match => enables correct blue/red per point when mixed
            "role": role,
            "zone": role_aware_zone(x, y, role),
        })
    return pd.DataFrame(out)


def first15_timeseries(timeline: Dict[str, Any], pid: int) -> pd.DataFrame:
    rows = []
    frames = timeline["info"]["frames"]
    for idx, fr in enumerate(frames):
        t_ms = fr.get("timestamp", idx * 60000)
        t_min = t_ms / 60000.0
        if t_min > 15.0:
            break
        pf = fr.get("participantFrames", {}).get(str(pid))
        if not pf:
            continue

        gold = pf.get("totalGold")
        xp = pf.get("xp")

        dmg = None
        dmg_stats = pf.get("damageStats") or {}
        for k in ["totalDamageDoneToChampions", "damageDealtToChampions", "totalDamageDone"]:
            if k in dmg_stats:
                dmg = dmg_stats[k]
                break

        rows.append({"minute": float(t_min), "gold": gold, "xp": xp, "damage": dmg})
    return pd.DataFrame(rows).sort_values("minute")


def get_snapshots_from_timeline(timeline: Dict[str, Any], pid: int, minutes=(5, 10, 15)) -> Dict[int, Dict[str, Optional[float]]]:
    frames = timeline["info"]["frames"]
    out = {m: {"gold": None, "xp": None} for m in minutes}
    for idx, fr in enumerate(frames):
        t_ms = fr.get("timestamp", idx * 60000)
        t_min = t_ms / 60000.0
        if t_min > max(minutes):
            break
        pf = fr.get("participantFrames", {}).get(str(pid))
        if not pf:
            continue
        gold = pf.get("totalGold")
        xp = pf.get("xp")
        for m in minutes:
            if t_min <= m:
                out[m] = {"gold": float(gold) if gold is not None else None,
                          "xp": float(xp) if xp is not None else None}
    return out


def snapshot_at_15(early_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    if early_df is None or early_df.empty:
        return {"gold15": None, "xp15": None, "damage15": None}
    e = early_df.sort_values("minute")
    last = e.tail(1).iloc[0]
    return {
        "gold15": float(last["gold"]) if pd.notna(last["gold"]) else None,
        "xp15": float(last["xp"]) if pd.notna(last["xp"]) else None,
        "damage15": float(last["damage"]) if pd.notna(last["damage"]) else None,
    }


def mean_safe(vals: List[Optional[float]]) -> Optional[float]:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else None


def find_lane_opponent_pid(match: Dict[str, Any], my_puuid: str, my_role: str) -> Optional[int]:
    my_role = (my_role or "UNKNOWN")
    me_team = None
    for p in match["info"]["participants"]:
        if p.get("puuid") == my_puuid:
            me_team = p.get("teamId")
            break
    if me_team is None or my_role == "UNKNOWN":
        return None
    for i, p in enumerate(match["info"]["participants"], start=1):
        if p.get("teamId") != me_team and (p.get("teamPosition") or "UNKNOWN") == my_role:
            return i
    return None


def find_pid_by_puuid(match: Dict[str, Any], puuid: str) -> Optional[int]:
    for i, p in enumerate(match["info"]["participants"], start=1):
        if p.get("puuid") == puuid:
            return i
    return None


def objectives_0_15(timeline: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for fr in timeline["info"]["frames"]:
        ts = fr.get("timestamp", 0)
        tmin = ts / 60000.0
        if tmin > 15.0:
            break
        for ev in fr.get("events", []):
            t = ev.get("type")
            if t in {"ELITE_MONSTER_KILL", "DRAGON_KILL", "RIFT_HERALD_KILL", "TURRET_PLATE_DESTROYED", "BUILDING_KILL"}:
                rows.append({
                    "minute": round(tmin, 2),
                    "type": t,
                    "teamId": ev.get("teamId"),
                    "monsterType": ev.get("monsterType") or ev.get("monsterSubType"),
                    "laneType": ev.get("laneType"),
                    "towerType": ev.get("towerType"),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["side"] = df["teamId"].apply(lambda tid: "BLUE" if tid == 100 else ("RED" if tid == 200 else "—"))
    return df


# =========================
# Plotting
# =========================
SIDE_COLOR = {"BLUE": "#1E88E5", "RED": "#E53935", "UNKNOWN": "#B0BEC5"}

def minimap_points_plotly(
    minimap: Image.Image,
    deaths_df: pd.DataFrame,
    title: str,
    point_size: int,
    point_alpha: float,
) -> Optional[go.Figure]:
    if deaths_df is None or deaths_df.empty:
        return None

    w, h = minimap.size
    b64 = minimap_to_base64_png(minimap)

    nxy = np.array([norm_xy(x, y) for x, y in zip(deaths_df["x"].values, deaths_df["y"].values)])
    px_py = np.array([norm_to_px(nx, ny, w, h) for nx, ny in nxy])

    d = deaths_df.copy()
    d["px"] = px_py[:, 0]
    d["py"] = px_py[:, 1]

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{b64}",
            xref="x", yref="y",
            x=0, y=0,
            sizex=w, sizey=h,
            sizing="stretch",
            layer="below",
        )
    )

    for side in ["BLUE", "RED", "UNKNOWN"]:
        ds = d[d["side"] == side]
        if ds.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=ds["px"],
                y=ds["py"],
                mode="markers",
                name=side,
                marker=dict(
                    size=point_size,
                    color=SIDE_COLOR.get(side, "#B0BEC5"),
                    opacity=point_alpha,
                    line=dict(width=1.2, color="white"),
                ),
                hovertemplate=(
                    "minute=%{customdata[0]:.1f}<br>"
                    "zone=%{customdata[1]}<br>"
                    "role=%{customdata[2]}<br>"
                    "side=%{customdata[3]}<br>"
                    "match=%{customdata[4]}<extra></extra>"
                ),
                customdata=np.stack([ds["minute"].values, ds["zone"].values, ds["role"].values, ds["side"].values, ds["matchId"].values], axis=1),
            )
        )

    fig.update_xaxes(range=[0, w], showgrid=False, visible=False)
    fig.update_yaxes(range=[h, 0], showgrid=False, visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=55, b=0),
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
    )
    return fig


def minimap_heatmap_matplotlib(
    minimap: Image.Image,
    deaths_df: pd.DataFrame,
    title: str,
    gridsize: int = 48,
) -> Optional[plt.Figure]:
    if deaths_df is None or deaths_df.empty:
        return None

    w, h = minimap.size
    nxy = np.array([norm_xy(x, y) for x, y in zip(deaths_df["x"].values, deaths_df["y"].values)])
    px_py = np.array([norm_to_px(nx, ny, w, h) for nx, ny in nxy])
    px, py = px_py[:, 0], px_py[:, 1]

    fig = plt.figure(figsize=(6.2, 6.2))
    ax = plt.gca()
    ax.imshow(minimap)
    ax.hexbin(px, py, gridsize=gridsize, mincnt=1, alpha=0.72)
    ax.set_title(title)
    ax.set_xlabel("X (pixels minimap)")
    ax.set_ylabel("Y (pixels minimap)")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    return fig


def plot_line(df: pd.DataFrame, x_col: str, y_col: str, title: str, x_label: str, y_label: str):
    if df is None or df.empty or df[y_col].dropna().empty:
        return None
    fig = plt.figure(figsize=(6.4, 3.2))
    plt.plot(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# =========================
# Team comparison (optional)
# - Team A: 5 Riot IDs
# - Compare Team A averages vs their OPPONENTS in detected matches
#   (matches where >=3 members of Team A are present)
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 20)
def fetch_team_match_candidates(team_ids: List[Tuple[str, str, str]], count_per_player: int, queue_val: Optional[int]) -> List[str]:
    """
    Returns candidate match IDs from the first 2 players (reduce API).
    """
    client = RiotClient()
    puuids = []
    for _label, g, t in team_ids[:2]:
        puuids.append(client.get_puuid_by_riot_id(g, t))

    # union of matches from first players to keep cost low
    match_ids = set()
    for p in puuids:
        mids = client.get_match_ids_by_puuid(p, count=count_per_player, queue=queue_val)
        for m in mids:
            match_ids.add(m)
    return list(match_ids)


def compute_team_comparison(team_text: str, count_matches: int, queue_val: Optional[int]) -> Optional[Dict[str, Any]]:
    team_list = parse_riot_ids(team_text)
    if len(team_list) < 5:
        return None

    client = RiotClient()
    # Resolve puuids
    team_puuids = {}
    for rid, g, t in team_list[:5]:
        try:
            team_puuids[rid] = client.get_puuid_by_riot_id(g, t)
        except Exception:
            continue

    if len(team_puuids) < 4:
        return {"error": "Impossible de résoudre assez de PUUID pour la team (min 4). Vérifie les Riot IDs."}

    candidate_match_ids = fetch_team_match_candidates(team_list[:5], count_per_player=count_matches, queue_val=queue_val)
    if not candidate_match_ids:
        return {"error": "Aucun match candidat trouvé pour la team."}

    # For each match, check how many of team A are present
    teamA_gold15, teamA_xp15, teamA_deaths0, teamA_deathsMid = [], [], [], []
    opp_gold15, opp_xp15, opp_deaths0, opp_deathsMid = [], [], [], []

    used_matches = 0
    for mid in candidate_match_ids[:count_matches]:
        try:
            match = client.get_match(mid)
            tl = client.get_timeline(mid)
        except Exception:
            continue

        participants = match["info"]["participants"]
        present = []
        for p in participants:
            if p.get("puuid") in set(team_puuids.values()):
                present.append(p)

        # require at least 3 members together to be meaningful
        if len(present) < 3:
            continue

        used_matches += 1

        # determine Team A teamId (majority)
        team_ids = [p.get("teamId") for p in present if p.get("teamId")]
        if not team_ids:
            continue
        teamA_teamid = max(set(team_ids), key=team_ids.count)

        # compute 0-15 + 15-30 deaths + gold/xp @15 for each participant, split by teamId
        for i, p in enumerate(participants, start=1):
            pid = i
            role = p.get("teamPosition") or "UNKNOWN"
            side = "BLUE" if p.get("teamId") == 100 else "RED"
            snaps = get_snapshots_from_timeline(tl, pid, minutes=(15,))
            gold15 = snaps[15]["gold"]
            xp15 = snaps[15]["xp"]

            d0 = extract_deaths_window(tl, pid, 0.0, 15.0, mid, side, role)
            dmid = extract_deaths_window(tl, pid, 15.0, 30.0, mid, side, role)

            if p.get("teamId") == teamA_teamid:
                if gold15 is not None: teamA_gold15.append(gold15)
                if xp15 is not None: teamA_xp15.append(xp15)
                teamA_deaths0.append(len(d0) if d0 is not None else 0)
                teamA_deathsMid.append(len(dmid) if dmid is not None else 0)
            else:
                if gold15 is not None: opp_gold15.append(gold15)
                if xp15 is not None: opp_xp15.append(xp15)
                opp_deaths0.append(len(d0) if d0 is not None else 0)
                opp_deathsMid.append(len(dmid) if dmid is not None else 0)

    if used_matches == 0:
        return {"error": "Aucun match trouvé avec au moins 3 joueurs de la team ensemble (dans l’échantillon)."}

    return {
        "used_matches": used_matches,
        "teamA": {
            "gold15_mean": float(np.mean(teamA_gold15)) if teamA_gold15 else None,
            "xp15_mean": float(np.mean(teamA_xp15)) if teamA_xp15 else None,
            "deaths0_mean": float(np.mean(teamA_deaths0)) if teamA_deaths0 else None,
            "deathsMid_mean": float(np.mean(teamA_deathsMid)) if teamA_deathsMid else None,
        },
        "opp": {
            "gold15_mean": float(np.mean(opp_gold15)) if opp_gold15 else None,
            "xp15_mean": float(np.mean(opp_xp15)) if opp_xp15 else None,
            "deaths0_mean": float(np.mean(opp_deaths0)) if opp_deaths0 else None,
            "deathsMid_mean": float(np.mean(opp_deathsMid)) if opp_deathsMid else None,
        }
    }


# =========================
# PDF export
# =========================
def build_player_pdf(
    out_path: str,
    player_label: str,
    meta: Dict[str, str],
    minimap: Image.Image,
    deaths_0_15: pd.DataFrame,
    deaths_15_30: pd.DataFrame,
    early_mean: pd.DataFrame,
    laning_summary: str,
    macro_summary: str,
):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    story = []

    story.append(Paragraph(f"<b>Rapport Coach</b> — {player_label}", styles["Title"]))
    story.append(Paragraph(
        f"Champion: {meta.get('champion','—')} | Rôle: {meta.get('role','—')} | Queue: {meta.get('queue','—')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 10))

    # Early heatmap
    fig = minimap_heatmap_matplotlib(minimap, deaths_0_15, "Heatmap morts — Early (0–15)")
    if fig:
        story.append(RLImage(io.BytesIO(fig_to_png_bytes(fig)), width=420, height=420))
        story.append(Spacer(1, 10))

    # Mid heatmap
    fig2 = minimap_heatmap_matplotlib(minimap, deaths_15_30, "Heatmap morts — Mid (15–30)")
    if fig2:
        story.append(RLImage(io.BytesIO(fig_to_png_bytes(fig2)), width=420, height=420))
        story.append(Spacer(1, 10))

    story.append(PageBreak())

    story.append(Paragraph("<b>Early game</b> (0–15)", styles["Heading2"]))
    for y, ylab, ttl in [("gold", "Gold", "Gold (0–15)"), ("xp", "XP", "XP (0–15)"), ("damage", "Dégâts", "Dégâts (metric) (0–15)")]:
        f = plot_line(early_mean, "minute", y, ttl, "Temps (minutes)", ylab)
        if f:
            story.append(RLImage(io.BytesIO(fig_to_png_bytes(f)), width=460, height=230))
            story.append(Spacer(1, 8))

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Review Laning</b>", styles["Heading2"]))
    story.append(Paragraph(laning_summary, styles["Normal"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Macro & Rotations</b>", styles["Heading2"]))
    story.append(Paragraph(macro_summary, styles["Normal"]))

    doc.build(story)


# =========================
# UI
# =========================
st.set_page_config(page_title="EUW Coach Dashboard", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small { opacity:0.85; font-size: 0.92rem; }
.card {
  padding: 14px 16px; border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
}
</style>
""", unsafe_allow_html=True)

st.title("EUW — Coach Dashboard (Coach-friendly)")
st.caption("Minimap interactive (points BLUE/RED par match) + heatmaps early/mid + laning review + macro. EUW uniquement.")


with st.sidebar:
    st.header("Analyse joueur")
    riot_ids_text = st.text_area("Joueurs (1 par ligne) : GameName#TAG", height=140)
    match_count = st.slider("Matchs récents / joueur", 1, 40, 20)
    queue_label = st.selectbox("Queue", list(QUEUE_MAP.keys()), index=1)
    queue_val = QUEUE_MAP[queue_label]

    st.header("Filtres")
    filter_role = st.multiselect("Rôle", ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"], default=[])
    filter_side = st.multiselect("Side (optionnel)", ["BLUE", "RED"], default=[])
    filter_champ = st.text_input("Champion (exact, optionnel)")

    st.header("Minimap")
    point_size = st.slider("Taille points", 6, 44, 18)
    point_alpha = st.slider("Opacité points", 0.25, 1.00, 0.80)
    heat_gridsize = st.slider("Densité heatmap (grille)", 24, 80, 48)

    st.header("Options laning (optionnel)")
    exact_opponent_text = st.text_input("Adversaire exact Riot ID (GameName#TAG)")

    st.header("Comparaison Team (optionnel)")
    enable_team = st.checkbox("Activer comparaison team (5 Riot IDs)", value=False)
    team_text = st.text_area("Team A (5 lignes, GameName#TAG)", height=120, disabled=(not enable_team))
    team_match_count = st.slider("Team: matchs à scanner", 5, 25, 10, disabled=(not enable_team))

    st.header("Perf")
    max_samples = st.slider("Max matchs traités par joueur (CPU)", 5, 40, 20)

    run = st.button("Analyser (EUW)")


def pass_filters(bundle) -> bool:
    if filter_role and bundle["role"] not in filter_role:
        return False
    if filter_side and bundle["side"] not in filter_side:
        return False
    if filter_champ.strip() and bundle["champion"] != filter_champ.strip():
        return False
    return True


@st.cache_data(show_spinner=False, ttl=60 * 20)
def fetch_player_bundle(riot_id_full: str, game: str, tag: str, count: int, queue_val: Optional[int]):
    client = RiotClient()
    puuid = client.get_puuid_by_riot_id(game, tag)
    match_ids = client.get_match_ids_by_puuid(puuid, count=count, queue=queue_val)

    bundles = []
    for mid in match_ids:
        match = client.get_match(mid)
        tl = client.get_timeline(mid)

        pid = participant_id_for_puuid(match, puuid)
        meta = participant_meta(match, puuid)

        champ = meta["championName"]
        role = meta["teamPosition"]
        side = meta["side"]

        # Early segments
        d_0_5 = extract_deaths_window(tl, pid, 0.0, 5.0, mid, side, role)
        d_5_10 = extract_deaths_window(tl, pid, 5.0, 10.0, mid, side, role)
        d_10_15 = extract_deaths_window(tl, pid, 10.0, 15.0, mid, side, role)

        d_0_15 = extract_deaths_window(tl, pid, 0.0, 15.0, mid, side, role)
        d_15_30 = extract_deaths_window(tl, pid, 15.0, 30.0, mid, side, role)

        early = first15_timeseries(tl, pid)
        s15 = snapshot_at_15(early)

        bundles.append({
            "matchId": mid,
            "puuid": puuid,
            "champion": champ,
            "role": role,
            "side": side,
            "win": meta["win"],
            "d_0_5": d_0_5,
            "d_5_10": d_5_10,
            "d_10_15": d_10_15,
            "d_0_15": d_0_15,
            "d_15_30": d_15_30,
            "early": early,
            "s15": s15,
            "match": match,
            "timeline": tl,
        })
    return bundles


def fmt(v: Optional[float], digits: int = 0) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


if run:
    # minimap
    try:
        minimap = fetch_minimap_image()
    except Exception as e:
        st.error(f"Impossible de télécharger la minimap (Data Dragon). Détail: {e}")
        st.stop()

    client = RiotClient()

    # optional opponent exact
    opponent_puuid = None
    if exact_opponent_text and "#" in exact_opponent_text:
        try:
            g, t = exact_opponent_text.split("#", 1)
            opponent_puuid = client.get_puuid_by_riot_id(g.strip(), t.strip())
        except Exception:
            opponent_puuid = None

    # optional team compare
    team_comp = None
    if enable_team:
        with st.spinner("Analyse team (optionnelle)..."):
            team_comp = compute_team_comparison(team_text, count_matches=team_match_count, queue_val=queue_val)

    players = parse_riot_ids(riot_ids_text)
    if not players:
        st.error("Ajoute au moins un Riot ID valide (GameName#TAG).")
        st.stop()

    tabs_players = st.tabs([p[0] for p in players])

    for tab, (rid_full, game, tag) in zip(tabs_players, players):
        with tab:
            st.subheader(f"Joueur : {rid_full}")

            with st.spinner("Récupération matchs + timelines..."):
                bundles = fetch_player_bundle(rid_full, game, tag, match_count, queue_val)

            bundles_f = [b for b in bundles if pass_filters(b)]
            bundles_f = bundles_f[:max_samples]

            st.caption(f"Matchs chargés: {len(bundles)} | Après filtres: {len(bundles_f)}")

            if not bundles_f:
                st.info("Aucun match ne passe les filtres.")
                continue

            # Aggregate
            early_all = []
            gold15_list, xp15_list, dmg15_list = [], [], []
            deaths0_count, deathsMid_count = [], []

            d0_5_all, d5_10_all, d10_15_all = [], [], []
            d0_15_all, d15_30_all = [], []

            # Laning comparisons
            matchup_gold_d = {5: [], 10: [], 15: []}
            matchup_xp_d = {5: [], 10: [], 15: []}
            lobby_gold_d15, lobby_xp_d15 = [], []
            exact_gold_d = {5: [], 10: [], 15: []}
            exact_xp_d = {5: [], 10: [], 15: []}

            # Macro
            zone0 = []
            zoneMid = []
            obj_rows = []

            for b in bundles_f:
                early_all.append(b["early"])
                gold15_list.append(b["s15"]["gold15"])
                xp15_list.append(b["s15"]["xp15"])
                dmg15_list.append(b["s15"]["damage15"])

                # deaths
                for src, acc in [(b["d_0_5"], d0_5_all), (b["d_5_10"], d5_10_all), (b["d_10_15"], d10_15_all),
                                 (b["d_0_15"], d0_15_all), (b["d_15_30"], d15_30_all)]:
                    if src is not None and not src.empty:
                        acc.append(src)

                deaths0_count.append(len(b["d_0_15"]) if b["d_0_15"] is not None else 0)
                deathsMid_count.append(len(b["d_15_30"]) if b["d_15_30"] is not None else 0)

                if b["d_0_15"] is not None and not b["d_0_15"].empty:
                    zone0.append(b["d_0_15"]["zone"])
                if b["d_15_30"] is not None and not b["d_15_30"].empty:
                    zoneMid.append(b["d_15_30"]["zone"])

                # objectives 0-15
                obj_rows.append(objectives_0_15(b["timeline"]))

                # Laning comparisons per match
                match = b["match"]
                tl = b["timeline"]
                my_pid = participant_id_for_puuid(match, b["puuid"])
                my_role = b["role"]

                my_snaps = get_snapshots_from_timeline(tl, my_pid, minutes=(5, 10, 15))

                # role-based opponent
                opp_pid = find_lane_opponent_pid(match, b["puuid"], my_role)
                if opp_pid:
                    opp_snaps = get_snapshots_from_timeline(tl, opp_pid, minutes=(5, 10, 15))
                    for m in (5, 10, 15):
                        if my_snaps[m]["gold"] is not None and opp_snaps[m]["gold"] is not None:
                            matchup_gold_d[m].append(my_snaps[m]["gold"] - opp_snaps[m]["gold"])
                        if my_snaps[m]["xp"] is not None and opp_snaps[m]["xp"] is not None:
                            matchup_xp_d[m].append(my_snaps[m]["xp"] - opp_snaps[m]["xp"])

                # exact opponent (optional)
                if opponent_puuid:
                    pid_exact = find_pid_by_puuid(match, opponent_puuid)
                    if pid_exact:
                        ex_snaps = get_snapshots_from_timeline(tl, pid_exact, minutes=(5, 10, 15))
                        for m in (5, 10, 15):
                            if my_snaps[m]["gold"] is not None and ex_snaps[m]["gold"] is not None:
                                exact_gold_d[m].append(my_snaps[m]["gold"] - ex_snaps[m]["gold"])
                            if my_snaps[m]["xp"] is not None and ex_snaps[m]["xp"] is not None:
                                exact_xp_d[m].append(my_snaps[m]["xp"] - ex_snaps[m]["xp"])

                # lobby mean diff @15 (9 autres)
                participants = match["info"]["participants"]
                others_gold, others_xp = [], []
                for i, _p in enumerate(participants, start=1):
                    snaps = get_snapshots_from_timeline(tl, i, minutes=(15,))
                    if i != my_pid:
                        if snaps[15]["gold"] is not None:
                            others_gold.append(snaps[15]["gold"])
                        if snaps[15]["xp"] is not None:
                            others_xp.append(snaps[15]["xp"])

                if my_snaps[15]["gold"] is not None and others_gold:
                    lobby_gold_d15.append(my_snaps[15]["gold"] - float(np.mean(others_gold)))
                if my_snaps[15]["xp"] is not None and others_xp:
                    lobby_xp_d15.append(my_snaps[15]["xp"] - float(np.mean(others_xp)))

            # Build dfs
            df_0_5 = pd.concat(d0_5_all, ignore_index=True) if d0_5_all else pd.DataFrame(columns=["matchId","minute","x","y","side","role","zone"])
            df_5_10 = pd.concat(d5_10_all, ignore_index=True) if d5_10_all else pd.DataFrame(columns=["matchId","minute","x","y","side","role","zone"])
            df_10_15 = pd.concat(d10_15_all, ignore_index=True) if d10_15_all else pd.DataFrame(columns=["matchId","minute","x","y","side","role","zone"])
            df_0_15 = pd.concat(d0_15_all, ignore_index=True) if d0_15_all else pd.DataFrame(columns=["matchId","minute","x","y","side","role","zone"])
            df_15_30 = pd.concat(d15_30_all, ignore_index=True) if d15_30_all else pd.DataFrame(columns=["matchId","minute","x","y","side","role","zone"])

            early_df = pd.concat(early_all, ignore_index=True) if early_all else pd.DataFrame()
            early_mean = early_df.groupby("minute", as_index=False).mean(numeric_only=True) if not early_df.empty else early_df

            # Headline
            champ = bundles_f[0]["champion"]
            role = bundles_f[0]["role"]

            me_gold15 = mean_safe(gold15_list)
            me_xp15 = mean_safe(xp15_list)
            me_dmg15 = mean_safe(dmg15_list)
            me_deaths0 = float(np.mean(deaths0_count)) if deaths0_count else None
            me_deathsMid = float(np.mean(deathsMid_count)) if deathsMid_count else None

            lane_gold_diff_5 = mean_safe(matchup_gold_d[5])
            lane_gold_diff_10 = mean_safe(matchup_gold_d[10])
            lane_gold_diff_15 = mean_safe(matchup_gold_d[15])
            lane_xp_diff_5 = mean_safe(matchup_xp_d[5])
            lane_xp_diff_10 = mean_safe(matchup_xp_d[10])
            lane_xp_diff_15 = mean_safe(matchup_xp_d[15])

            lobby_gold_diff_15 = mean_safe(lobby_gold_d15)
            lobby_xp_diff_15 = mean_safe(lobby_xp_d15)

            exact_gold_diff_5 = mean_safe(exact_gold_d[5])
            exact_gold_diff_10 = mean_safe(exact_gold_d[10])
            exact_gold_diff_15 = mean_safe(exact_gold_d[15])
            exact_xp_diff_5 = mean_safe(exact_xp_d[5])
            exact_xp_diff_10 = mean_safe(exact_xp_d[10])
            exact_xp_diff_15 = mean_safe(exact_xp_d[15])

            zone0_counts = pd.concat(zone0, ignore_index=True).value_counts() if zone0 else pd.Series(dtype=int)
            zoneMid_counts = pd.concat(zoneMid, ignore_index=True).value_counts() if zoneMid else pd.Series(dtype=int)
            obj_df = pd.concat(obj_rows, ignore_index=True) if obj_rows else pd.DataFrame()

            # Tabs
            t_overview, t_laning, t_macro, t_team, t_exports = st.tabs(
                ["📍 Overview", "⚔️ Laning", "🧭 Macro", "👥 Team (option)", "📄 Export"]
            )

            with t_overview:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Gold@15 (moy.)", f"{me_gold15:.0f}" if me_gold15 is not None else "—")
                c2.metric("XP@15 (moy.)", f"{me_xp15:.0f}" if me_xp15 is not None else "—")
                c3.metric("Dmg@15 (metric)", f"{me_dmg15:.0f}" if me_dmg15 is not None else "—")
                c4.metric("Morts 0–15 (moy.)", f"{me_deaths0:.2f}" if me_deaths0 is not None else "—")
                c5.metric("Morts 15–30 (moy.)", f"{me_deathsMid:.2f}" if me_deathsMid is not None else "—")

                st.markdown(f"<div class='card'><b>Profil</b> : {champ} • {role} • EUW • {queue_label}</div>", unsafe_allow_html=True)
                st.divider()

                left, right = st.columns([1.35, 1])

                with left:
                    st.markdown("#### Points de morts — Early (0–15) (interactif)")
                    figP = minimap_points_plotly(minimap, df_0_15, "Morts (0–15) — points BLUE/RED (selon side du match)", point_size, point_alpha)
                    if figP:
                        st.plotly_chart(figP, use_container_width=True)
                    else:
                        st.info("Aucune mort (0–15) sur l’échantillon filtré.")

                    st.markdown("#### Heatmap (densité) — Early (0–15)")
                    figH = minimap_heatmap_matplotlib(minimap, df_0_15, "Heatmap morts (0–15)", gridsize=heat_gridsize)
                    if figH:
                        st.pyplot(figH, clear_figure=True)

                    st.markdown("#### Heatmaps Early par tranches")
                    h1, h2, h3 = st.columns(3)
                    with h1:
                        f = minimap_heatmap_matplotlib(minimap, df_0_5, "0–5", gridsize=heat_gridsize)
                        if f: st.pyplot(f, clear_figure=True)
                        else: st.caption("—")
                    with h2:
                        f = minimap_heatmap_matplotlib(minimap, df_5_10, "5–10", gridsize=heat_gridsize)
                        if f: st.pyplot(f, clear_figure=True)
                        else: st.caption("—")
                    with h3:
                        f = minimap_heatmap_matplotlib(minimap, df_10_15, "10–15", gridsize=heat_gridsize)
                        if f: st.pyplot(f, clear_figure=True)
                        else: st.caption("—")

                with right:
                    st.markdown("#### Early curves (0–15)")
                    if not early_mean.empty:
                        st.line_chart(early_mean.set_index("minute")[["gold"]], height=170)
                        st.caption("x = Temps (minutes) • y = Gold")
                        st.line_chart(early_mean.set_index("minute")[["xp"]], height=170)
                        st.caption("x = Temps (minutes) • y = XP")
                        if "damage" in early_mean.columns and early_mean["damage"].notna().any():
                            st.line_chart(early_mean.set_index("minute")[["damage"]], height=170)
                            st.caption("x = Temps (minutes) • y = Dégâts (metric)")
                    else:
                        st.info("Pas assez de données early.")

                    st.markdown("#### Comparaison (vs moyenne lobby)")
                    a1, a2 = st.columns(2)
                    a1.metric("Gold@15 vs lobby", f"{lobby_gold_diff_15:.0f}" if lobby_gold_diff_15 is not None else "—")
                    a2.metric("XP@15 vs lobby", f"{lobby_xp_diff_15:.0f}" if lobby_xp_diff_15 is not None else "—")
                    st.caption("Lobby = moyenne des 9 autres joueurs des mêmes matchs.")

                st.divider()
                st.markdown("### Mid game (15–30)")

                m1, m2 = st.columns([1.35, 1])
                with m1:
                    st.markdown("#### Points de morts — Mid (15–30) (interactif)")
                    figPm = minimap_points_plotly(minimap, df_15_30, "Morts (15–30) — points BLUE/RED (selon side du match)", point_size, point_alpha)
                    if figPm:
                        st.plotly_chart(figPm, use_container_width=True)
                    else:
                        st.info("Aucune mort (15–30) sur l’échantillon filtré.")

                with m2:
                    st.markdown("#### Heatmap (densité) — Mid (15–30)")
                    figHm = minimap_heatmap_matplotlib(minimap, df_15_30, "Heatmap morts (15–30)", gridsize=heat_gridsize)
                    if figHm:
                        st.pyplot(figHm, clear_figure=True)

            with t_laning:
                st.markdown("### Review Laning (vs matchup direct = même rôle)")
                l1, l2, l3 = st.columns(3)
                l1.metric("Gold diff @5", f"{lane_gold_diff_5:.0f}" if lane_gold_diff_5 is not None else "—")
                l2.metric("Gold diff @10", f"{lane_gold_diff_10:.0f}" if lane_gold_diff_10 is not None else "—")
                l3.metric("Gold diff @15", f"{lane_gold_diff_15:.0f}" if lane_gold_diff_15 is not None else "—")

                x1, x2, x3 = st.columns(3)
                x1.metric("XP diff @5", f"{lane_xp_diff_5:.0f}" if lane_xp_diff_5 is not None else "—")
                x2.metric("XP diff @10", f"{lane_xp_diff_10:.0f}" if lane_xp_diff_10 is not None else "—")
                x3.metric("XP diff @15", f"{lane_xp_diff_15:.0f}" if lane_xp_diff_15 is not None else "—")

                st.caption("Diff = (toi − adversaire direct même rôle). Si swaps / rôle UNKNOWN, la comparaison peut être partielle.")

                if exact_opponent_text:
                    st.divider()
                    st.markdown("### Option : adversaire exact (Riot ID)")
                    e1, e2, e3 = st.columns(3)
                    e1.metric("Gold diff @5", f"{exact_gold_diff_5:.0f}" if exact_gold_diff_5 is not None else "—")
                    e2.metric("Gold diff @10", f"{exact_gold_diff_10:.0f}" if exact_gold_diff_10 is not None else "—")
                    e3.metric("Gold diff @15", f"{exact_gold_diff_15:.0f}" if exact_gold_diff_15 is not None else "—")
                    ex1, ex2, ex3 = st.columns(3)
                    ex1.metric("XP diff @5", f"{exact_xp_diff_5:.0f}" if exact_xp_diff_5 is not None else "—")
                    ex2.metric("XP diff @10", f"{exact_xp_diff_10:.0f}" if exact_xp_diff_10 is not None else "—")
                    ex3.metric("XP diff @15", f"{exact_xp_diff_15:.0f}" if exact_xp_diff_15 is not None else "—")
                    st.caption("Affiché seulement si cet adversaire est présent dans les matchs analysés.")

                st.divider()
                st.markdown("### Table match-by-match (VOD review)")
                rows = []
                for b in bundles_f:
                    rows.append({
                        "matchId": b["matchId"],
                        "champion": b["champion"],
                        "role": b["role"],
                        "side": b["side"],
                        "win": b["win"],
                        "gold@15": b["s15"]["gold15"],
                        "xp@15": b["s15"]["xp15"],
                        "deaths_0_15": int(len(b["d_0_15"])) if b["d_0_15"] is not None else 0,
                        "deaths_15_30": int(len(b["d_15_30"])) if b["d_15_30"] is not None else 0,
                    })
                st.dataframe(pd.DataFrame(rows).sort_values("matchId"))

            with t_macro:
                st.markdown("### Macro & Rotations — zones role-aware (pas de confusion toplane/topside)")
                z1, z2 = st.columns(2)
                with z1:
                    st.markdown("#### Zones Early (0–15)")
                    if not zone0_counts.empty:
                        st.dataframe(zone0_counts.rename("deaths").to_frame())
                    else:
                        st.info("Aucune mort early.")
                with z2:
                    st.markdown("#### Zones Mid (15–30)")
                    if not zoneMid_counts.empty:
                        st.dataframe(zoneMid_counts.rename("deaths").to_frame())
                    else:
                        st.info("Aucune mort mid.")

                st.divider()
                st.markdown("### Objectifs / Events (0–15)")
                if obj_df is not None and not obj_df.empty:
                    st.dataframe(obj_df.sort_values("minute").head(250))
                else:
                    st.caption("Aucun event objectif détecté 0–15 (selon les matchs).")

            with t_team:
                st.markdown("### Comparaison Team (option)")
                if not enable_team:
                    st.info("Active l’option dans la sidebar pour entrer 5 Riot IDs.")
                else:
                    if team_comp is None:
                        st.info("Entre 5 Riot IDs dans 'Team A' puis relance.")
                    elif isinstance(team_comp, dict) and team_comp.get("error"):
                        st.error(team_comp["error"])
                    else:
                        st.success(f"Matches utilisés (>=3 joueurs ensemble) : {team_comp['used_matches']}")
                        cA1, cA2, cA3, cA4 = st.columns(4)
                        cA1.metric("Team A Gold@15", f"{team_comp['teamA']['gold15_mean']:.0f}" if team_comp["teamA"]["gold15_mean"] is not None else "—")
                        cA2.metric("Team A XP@15", f"{team_comp['teamA']['xp15_mean']:.0f}" if team_comp["teamA"]["xp15_mean"] is not None else "—")
                        cA3.metric("Team A morts 0–15", f"{team_comp['teamA']['deaths0_mean']:.2f}" if team_comp["teamA"]["deaths0_mean"] is not None else "—")
                        cA4.metric("Team A morts 15–30", f"{team_comp['teamA']['deathsMid_mean']:.2f}" if team_comp["teamA"]["deathsMid_mean"] is not None else "—")

                        cB1, cB2, cB3, cB4 = st.columns(4)
                        cB1.metric("Opp Gold@15", f"{team_comp['opp']['gold15_mean']:.0f}" if team_comp["opp"]["gold15_mean"] is not None else "—")
                        cB2.metric("Opp XP@15", f"{team_comp['opp']['xp15_mean']:.0f}" if team_comp["opp"]["xp15_mean"] is not None else "—")
                        cB3.metric("Opp morts 0–15", f"{team_comp['opp']['deaths0_mean']:.2f}" if team_comp["opp"]["deaths0_mean"] is not None else "—")
                        cB4.metric("Opp morts 15–30", f"{team_comp['opp']['deathsMid_mean']:.2f}" if team_comp["opp"]["deathsMid_mean"] is not None else "—")

                        st.caption("Méthode : on cherche des matchs où ≥3 joueurs de la Team A sont ensemble, puis on compare vs l’équipe adverse de ces matchs.")

            with t_exports:
                st.markdown("### Export PDF")
                st.caption("Le PDF inclut heatmap early + heatmap mid + courbes early + résumé laning/macro.")

                laning_summary = (
                    f"Matchup (moyenne): Gold diff @5={fmt(lane_gold_diff_5)}, @10={fmt(lane_gold_diff_10)}, @15={fmt(lane_gold_diff_15)}. "
                    f"XP diff @5={fmt(lane_xp_diff_5)}, @10={fmt(lane_xp_diff_10)}, @15={fmt(lane_xp_diff_15)}. "
                    f"Lobby: Gold@15 vs moyenne={fmt(lobby_gold_diff_15)}, XP@15 vs moyenne={fmt(lobby_xp_diff_15)}."
                )
                if exact_opponent_text:
                    laning_summary += (
                        f" Adversaire exact: Gold diff @5={fmt(exact_gold_diff_5)}, @10={fmt(exact_gold_diff_10)}, @15={fmt(exact_gold_diff_15)}. "
                        f"XP diff @5={fmt(exact_xp_diff_5)}, @10={fmt(exact_xp_diff_10)}, @15={fmt(exact_xp_diff_15)}."
                    )

                macro_summary = (
                    f"Morts 0–15 (moy.)={me_deaths0:.2f} • Morts 15–30 (moy.)={me_deathsMid:.2f}. "
                    f"Zones early top: {', '.join([f'{k}({int(v)})' for k, v in zone0_counts.head(4).items()]) if not zone0_counts.empty else '—'}. "
                    f"Zones mid top: {', '.join([f'{k}({int(v)})' for k, v in zoneMid_counts.head(4).items()]) if not zoneMid_counts.empty else '—'}. "
                    f"Events objectifs 0–15: {len(obj_df) if obj_df is not None and not obj_df.empty else 0}."
                )

                if st.button(f"Générer PDF — {rid_full}", key=f"pdf_{rid_full}"):
                    os.makedirs("exports", exist_ok=True)
                    pdf_path = os.path.join("exports", f"{rid_full.replace('#', '_')}_report.pdf")

                    meta = {"champion": champ, "role": role, "queue": queue_label}

                    build_player_pdf(
                        out_path=pdf_path,
                        player_label=rid_full,
                        meta=meta,
                        minimap=minimap,
                        deaths_0_15=df_0_15,
                        deaths_15_30=df_15_30,
                        early_mean=early_mean,
                        laning_summary=laning_summary,
                        macro_summary=macro_summary,
                    )

                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "Télécharger le PDF",
                            f,
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf",
                            key=f"dl_{rid_full}",
                        )
