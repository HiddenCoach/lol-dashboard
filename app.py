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
                raise RuntimeError(f"Erreur Riot API {r.status_code}: {r.text[:900]}")
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
# Parsing inputs
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


def parse_match_ids(text: str) -> List[str]:
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # match-v5 IDs look like: EUW1_123... (still on europe regional)
        out.append(line)
    return out


# =========================
# Match helpers
# =========================
def participant_id_for_puuid(match: Dict[str, Any], puuid: str) -> int:
    for i, p in enumerate(match["info"]["participants"], start=1):
        if p.get("puuid") == puuid:
            return i
    raise ValueError("PUUID non trouvé dans ce match.")


def find_pid_by_puuid(match: Dict[str, Any], puuid: str) -> Optional[int]:
    for i, p in enumerate(match["info"]["participants"], start=1):
        if p.get("puuid") == puuid:
            return i
    return None


def participant_meta_by_pid(match: Dict[str, Any], pid: int) -> Dict[str, Any]:
    p = match["info"]["participants"][pid - 1]
    team_id = p.get("teamId")
    return {
        "puuid": p.get("puuid"),
        "championName": p.get("championName") or "UNKNOWN",
        "teamPosition": p.get("teamPosition") or "UNKNOWN",
        "teamId": team_id,
        "side": "BLUE" if team_id == 100 else "RED",
        "win": bool(p.get("win", False)),
    }


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
# - OFFSIDE tagging by role.
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
        # baron side (upper-left) corresponds to ny > nx
        base = "TOP_SIDE" if ny > nx else "BOT_SIDE"

    # Lane corridor hints (only if far from mid/river)
    top_lane_corridor = (nx < 0.38 and ny > 0.62)
    bot_lane_corridor = (nx > 0.62 and ny < 0.38)
    if base == "TOP_SIDE" and top_lane_corridor:
        base = "TOP_LANE_AREA"
    if base == "BOT_SIDE" and bot_lane_corridor:
        base = "BOT_LANE_AREA"

    role = (role or "UNKNOWN").upper()

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
            "side": side,      # per match => blue/red points correct even when mixed
            "role": role,
            "zone": role_aware_zone(x, y, role),
        })
    return pd.DataFrame(out)


def extract_kills_window(
    timeline: Dict[str, Any],
    killer_pid: int,
    start_min: float,
    end_min: float,
    match_id: str,
    side: str,
    role: str,
) -> pd.DataFrame:
    out = []
    for ts, ev in iter_kill_events(timeline):
        if ev.get("killerId") != killer_pid:
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
            "side": side,
            "role": role,
            "zone": role_aware_zone(x, y, role),
        })
    return pd.DataFrame(out)


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


def timeline_series_full_minutes(timeline: Dict[str, Any], pid: int, max_min: int) -> pd.DataFrame:
    """
    FULL match series, minute by minute INTEGERS only (0..max_min)
    values forward-filled (cumulative style)
    """
    frames = timeline["info"]["frames"]
    minute_rows = {}  # minute -> {gold,xp,damage}
    for idx, fr in enumerate(frames):
        t_ms = fr.get("timestamp", idx * 60000)
        m = int(t_ms // 60000)  # integer minute
        if m > max_min:
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

        minute_rows[m] = {
            "minute": m,
            "gold": float(gold) if gold is not None else np.nan,
            "xp": float(xp) if xp is not None else np.nan,
            "damage": float(dmg) if dmg is not None else np.nan,
        }

    # build full minutes + forward fill
    df = pd.DataFrame({"minute": list(range(0, max_min + 1))})
    if minute_rows:
        df2 = pd.DataFrame(list(minute_rows.values()))
        df = df.merge(df2, on="minute", how="left")
        df[["gold", "xp", "damage"]] = df[["gold", "xp", "damage"]].ffill()
    else:
        df["gold"] = np.nan
        df["xp"] = np.nan
        df["damage"] = np.nan
    return df


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
                    "minute": int(ts // 60000),
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
    df: pd.DataFrame,
    title: str,
    point_size: int,
    point_alpha: float,
) -> Optional[go.Figure]:
    if df is None or df.empty:
        return None

    w, h = minimap.size
    b64 = minimap_to_base64_png(minimap)

    nxy = np.array([norm_xy(x, y) for x, y in zip(df["x"].values, df["y"].values)])
    px_py = np.array([norm_to_px(nx, ny, w, h) for nx, ny in nxy])

    d = df.copy()
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
                    line=dict(width=1.1, color="white"),
                ),
                hovertemplate=(
                    "min=%{customdata[0]}<br>"
                    "zone=%{customdata[1]}<br>"
                    "role=%{customdata[2]}<br>"
                    "side=%{customdata[3]}<br>"
                    "match=%{customdata[4]}<extra></extra>"
                ),
                customdata=np.stack([
                    np.floor(ds["minute"].values).astype(int),
                    ds["zone"].values,
                    ds["role"].values,
                    ds["side"].values,
                    ds["matchId"].values
                ], axis=1),
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
    df: pd.DataFrame,
    title: str,
    gridsize: int = 48,
) -> Optional[plt.Figure]:
    if df is None or df.empty:
        return None

    w, h = minimap.size
    nxy = np.array([norm_xy(x, y) for x, y in zip(df["x"].values, df["y"].values)])
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


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# =========================
# Team A vs Team B
# =========================
def resolve_team_puuids(client: RiotClient, team_text: str) -> Dict[str, str]:
    team_list = parse_riot_ids(team_text)
    puuids = {}
    for rid, g, t in team_list[:5]:
        try:
            puuids[rid] = client.get_puuid_by_riot_id(g, t)
        except Exception:
            continue
    return puuids


@st.cache_data(show_spinner=False, ttl=60 * 20)
def candidate_match_ids_from_team(client: RiotClient, team_puuids: List[str], count_per_player: int, queue_val: Optional[int]) -> List[str]:
    # union matches from first 2 players to limit API cost
    match_ids = set()
    for puuid in team_puuids[:2]:
        mids = client.get_match_ids_by_puuid(puuid, count=count_per_player, queue=queue_val)
        for m in mids:
            match_ids.add(m)
    return list(match_ids)


def compute_teamA_vs_teamB(
    client: RiotClient,
    teamA_text: str,
    teamB_text: str,
    count_matches: int,
    queue_val: Optional[int],
    scrim_match_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    teamA = resolve_team_puuids(client, teamA_text)
    teamB = resolve_team_puuids(client, teamB_text)

    if len(teamA) < 4 or len(teamB) < 4:
        return {"error": "Il faut au moins 4 Riot IDs valides par team (idéalement 5/5)."}

    teamA_set = set(teamA.values())
    teamB_set = set(teamB.values())

    # candidates
    if scrim_match_ids:
        candidates = scrim_match_ids[:count_matches]
    else:
        candidates = candidate_match_ids_from_team(client, list(teamA_set), count_per_player=count_matches, queue_val=queue_val)
        if not candidates:
            return {"error": "Aucun match candidat trouvé (essaie d’augmenter Team matchs à scanner)."}
        candidates = candidates[:count_matches]

    used = 0
    A_gold15, A_xp15, A_de0, A_deMid = [], [], [], []
    B_gold15, B_xp15, B_de0, B_deMid = [], [], [], []

    for mid in candidates:
        try:
            match = client.get_match(mid)
            tl = client.get_timeline(mid)
        except Exception:
            continue

        participants = match["info"]["participants"]
        presentA = [p for p in participants if p.get("puuid") in teamA_set]
        presentB = [p for p in participants if p.get("puuid") in teamB_set]

        # require meaningful presence (scrim): at least 3 each
        if len(presentA) < 3 or len(presentB) < 3:
            continue

        used += 1

        # compute per participant quick metrics (gold/xp@15 + deaths early/mid)
        for pid in range(1, 11):
            meta = participant_meta_by_pid(match, pid)
            p_puuid = meta["puuid"]
            role = meta["teamPosition"]
            side = meta["side"]

            snaps = get_snapshots_from_timeline(tl, pid, minutes=(15,))
            gold15 = snaps[15]["gold"]
            xp15 = snaps[15]["xp"]

            d0 = extract_deaths_window(tl, pid, 0.0, 15.0, mid, side, role)
            dmid = extract_deaths_window(tl, pid, 15.0, 30.0, mid, side, role)

            if p_puuid in teamA_set:
                if gold15 is not None: A_gold15.append(gold15)
                if xp15 is not None: A_xp15.append(xp15)
                A_de0.append(len(d0) if d0 is not None else 0)
                A_deMid.append(len(dmid) if dmid is not None else 0)
            elif p_puuid in teamB_set:
                if gold15 is not None: B_gold15.append(gold15)
                if xp15 is not None: B_xp15.append(xp15)
                B_de0.append(len(d0) if d0 is not None else 0)
                B_deMid.append(len(dmid) if dmid is not None else 0)

    if used == 0:
        return {"error": "Aucun match trouvé avec ≥3 joueurs de Team A ET ≥3 joueurs de Team B dans l’échantillon."}

    return {
        "used_matches": used,
        "A": {
            "gold15_mean": float(np.mean(A_gold15)) if A_gold15 else None,
            "xp15_mean": float(np.mean(A_xp15)) if A_xp15 else None,
            "deaths0_mean": float(np.mean(A_de0)) if A_de0 else None,
            "deathsMid_mean": float(np.mean(A_deMid)) if A_deMid else None,
        },
        "B": {
            "gold15_mean": float(np.mean(B_gold15)) if B_gold15 else None,
            "xp15_mean": float(np.mean(B_xp15)) if B_xp15 else None,
            "deaths0_mean": float(np.mean(B_de0)) if B_de0 else None,
            "deathsMid_mean": float(np.mean(B_deMid)) if B_deMid else None,
        },
    }


# =========================
# PDF export (basic)
# =========================
def build_player_pdf(
    out_path: str,
    player_label: str,
    meta: Dict[str, str],
    minimap: Image.Image,
    deaths_0_15: pd.DataFrame,
    deaths_15_30: pd.DataFrame,
    kills_all: pd.DataFrame,
    series_mean: pd.DataFrame,
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

    # Death heatmaps
    f1 = minimap_heatmap_matplotlib(minimap, deaths_0_15, "Heatmap morts — Early (0–15)")
    if f1:
        story.append(RLImage(io.BytesIO(fig_to_png_bytes(f1)), width=420, height=420))
        story.append(Spacer(1, 8))

    f2 = minimap_heatmap_matplotlib(minimap, deaths_15_30, "Heatmap morts — Mid (15–30)")
    if f2:
        story.append(RLImage(io.BytesIO(fig_to_png_bytes(f2)), width=420, height=420))
        story.append(Spacer(1, 8))

    # Kills heatmap
    f3 = minimap_heatmap_matplotlib(minimap, kills_all, "Heatmap kills — positions (toutes fenêtres)")
    if f3:
        story.append(RLImage(io.BytesIO(fig_to_png_bytes(f3)), width=420, height=420))

    story.append(PageBreak())

    # Series (minute integers)
    story.append(Paragraph("<b>Courbes</b> (minute par minute)", styles["Heading2"]))
    for col, title in [("gold", "Gold"), ("xp", "XP"), ("damage", "Dégâts (metric)")]:
        fig = plt.figure(figsize=(6.4, 3.0))
        plt.plot(series_mean["minute"], series_mean[col])
        plt.title(f"{title} — moyenne")
        plt.xlabel("Temps (minutes)")
        plt.ylabel(title)
        story.append(RLImage(io.BytesIO(fig_to_png_bytes(fig)), width=460, height=220))
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
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1550px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.card {
  padding: 14px 16px; border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
}
</style>
""", unsafe_allow_html=True)

st.title("Analyse Heatmap et Data, par Hidden Analyst Coach")
st.caption("Minimap interactive, heatmaps deaths/kills, laning review, macro, Team A vs Team B, mode scrim (custom), option analyse matchup vs joueur en particulier)")

with st.sidebar:
    st.header("Mode")
    scrim_mode = st.checkbox("Accès Scrim (parties personnalisées)", value=False)
    scrim_match_ids_text = st.text_area(
        "Scrim: Match IDs (1 par ligne, ex: EUW1_123...)",
        height=110,
        help="Si activé, on analyse UNIQUEMENT ces matchs (utile pour custom/scrim).",
        disabled=(not scrim_mode)
    )

    st.divider()
    st.header("Analyse joueur")
    riot_ids_text = st.text_area("Joueurs (1 par ligne) : GameName#TAG", height=130)

    match_count = st.slider("Matchs récents / joueur", 1, 40, 20, disabled=scrim_mode)
    queue_label = st.selectbox("Queue", list(QUEUE_MAP.keys()), index=1, disabled=scrim_mode)
    queue_val = QUEUE_MAP[queue_label] if not scrim_mode else None

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

    st.header("Team A vs Team B (optionnel)")
    enable_team_vs = st.checkbox("Activer Team A vs Team B", value=False)
    teamA_text = st.text_area("Team A (5 lignes, GameName#TAG)", height=100, disabled=(not enable_team_vs))
    teamB_text = st.text_area("Team B (5 lignes, GameName#TAG)", height=100, disabled=(not enable_team_vs))
    team_match_count = st.slider("Team: matchs à scanner", 5, 30, 12, disabled=(not enable_team_vs))

    st.header("Comparaison joueur (optionnel)")
    compare_player_text = st.text_input("Comparer avec Riot ID (GameName#TAG)")
    compare_mode = st.radio(
    "Mode comparaison",
    ["Même matchs (si présent)", "Ses matchs récents"],
    index=0
)
    compare_match_count = st.slider("Nb matchs pour joueur comparé", 1, 30, 10)

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
def fetch_player_bundle_by_match_list(riot_id_full: str, game: str, tag: str, match_ids: List[str]):
    """
    Scrim mode: analyse ONLY given match IDs
    """
    client = RiotClient()
    puuid = client.get_puuid_by_riot_id(game, tag)

    bundles = []
    for mid in match_ids:
        match = client.get_match(mid)
        tl = client.get_timeline(mid)

        pid = find_pid_by_puuid(match, puuid)
        if not pid:
            continue  # player not in this custom match

        meta = participant_meta(match, puuid)
        champ = meta["championName"]
        role = meta["teamPosition"]
        side = meta["side"]

        # windows deaths
        d_0_5 = extract_deaths_window(tl, pid, 0.0, 5.0, mid, side, role)
        d_5_10 = extract_deaths_window(tl, pid, 5.0, 10.0, mid, side, role)
        d_10_15 = extract_deaths_window(tl, pid, 10.0, 15.0, mid, side, role)
        d_0_15 = extract_deaths_window(tl, pid, 0.0, 15.0, mid, side, role)
        d_15_30 = extract_deaths_window(tl, pid, 15.0, 30.0, mid, side, role)

        # kills (all match window for heatmap)
        k_all = extract_kills_window(tl, pid, 0.0, 90.0, mid, side, role)

        # duration minutes
        dur_s = match["info"].get("gameDuration", 0) or 0
        max_min = int(dur_s // 60)

        series_full = timeline_series_full_minutes(tl, pid, max_min=max_min)

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
            "kills_all": k_all,
            "series_full": series_full,
            "match": match,
            "timeline": tl,
            "max_min": max_min,
        })
    return bundles


@st.cache_data(show_spinner=False, ttl=60 * 20)
def fetch_player_bundle_normal(riot_id_full: str, game: str, tag: str, count: int, queue_val: Optional[int]):
    """
    Normal mode: fetch match ids by puuid
    """
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

        d_0_5 = extract_deaths_window(tl, pid, 0.0, 5.0, mid, side, role)
        d_5_10 = extract_deaths_window(tl, pid, 5.0, 10.0, mid, side, role)
        d_10_15 = extract_deaths_window(tl, pid, 10.0, 15.0, mid, side, role)
        d_0_15 = extract_deaths_window(tl, pid, 0.0, 15.0, mid, side, role)
        d_15_30 = extract_deaths_window(tl, pid, 15.0, 30.0, mid, side, role)

        k_all = extract_kills_window(tl, pid, 0.0, 90.0, mid, side, role)

        dur_s = match["info"].get("gameDuration", 0) or 0
        max_min = int(dur_s // 60)

        series_full = timeline_series_full_minutes(tl, pid, max_min=max_min)

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
            "kills_all": k_all,
            "series_full": series_full,
            "match": match,
            "timeline": tl,
            "max_min": max_min,
        })
    return bundles


def fmt(v: Optional[float], digits: int = 0) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


def mean_safe(values):
    clean = [
        v for v in values
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    return float(np.mean(clean)) if clean else None


def mean_safe(values):
    """
    Moyenne robuste :
    - ignore None
    - ignore NaN
    - retourne None si vide
    """
    clean = [
        v for v in values
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    if not clean:
        return None
    return float(np.mean(clean))


if run:
    # minimap
    try:
        minimap = fetch_minimap_image()
    except Exception as e:
        st.error(f"Impossible de télécharger la minimap (Data Dragon). Détail: {e}")
        st.stop()

    client = RiotClient()

compare_puuid = None
if compare_player_text and "#" in compare_player_text:
    try:
        cg, ct = compare_player_text.split("#", 1)
        compare_puuid = client.get_puuid_by_riot_id(cg.strip(), ct.strip())
    except Exception:
        compare_puuid = None

    # scrim match ids
    scrim_match_ids = parse_match_ids(scrim_match_ids_text) if scrim_mode else None
    if scrim_mode and not scrim_match_ids:
        st.error("Scrim mode activé: ajoute au moins un Match ID dans la sidebar.")
        st.stop()

    # optional opponent exact
    opponent_puuid = None
    if exact_opponent_text and "#" in exact_opponent_text:
        try:
            g, t = exact_opponent_text.split("#", 1)
            opponent_puuid = client.get_puuid_by_riot_id(g.strip(), t.strip())
        except Exception:
            opponent_puuid = None

    # optional Team A vs Team B
    team_vs = None
    if enable_team_vs:
        with st.spinner("Analyse Team A vs Team B..."):
            team_vs = compute_teamA_vs_teamB(
                client,
                teamA_text=teamA_text,
                teamB_text=teamB_text,
                count_matches=team_match_count,
                queue_val=queue_val,
                scrim_match_ids=scrim_match_ids if scrim_mode else None,
            )

    players = parse_riot_ids(riot_ids_text)
    if not players:
        st.error("Ajoute au moins un Riot ID valide (GameName#TAG).")
        st.stop()

    # Global team section (top)
    if enable_team_vs:
        st.markdown("## 👥 Team A vs Team B (option)")
        if team_vs and team_vs.get("error"):
            st.error(team_vs["error"])
        elif team_vs:
            st.success(f"Matches utilisés (≥3v≥3): {team_vs['used_matches']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Team A Gold@15", f"{team_vs['A']['gold15_mean']:.0f}" if team_vs["A"]["gold15_mean"] is not None else "—")
            c2.metric("Team B Gold@15", f"{team_vs['B']['gold15_mean']:.0f}" if team_vs["B"]["gold15_mean"] is not None else "—")
            c3.metric("Team A morts 0–15", f"{team_vs['A']['deaths0_mean']:.2f}" if team_vs["A"]["deaths0_mean"] is not None else "—")
            c4.metric("Team B morts 0–15", f"{team_vs['B']['deaths0_mean']:.2f}" if team_vs["B"]["deaths0_mean"] is not None else "—")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Team A XP@15", f"{team_vs['A']['xp15_mean']:.0f}" if team_vs["A"]["xp15_mean"] is not None else "—")
            d2.metric("Team B XP@15", f"{team_vs['B']['xp15_mean']:.0f}" if team_vs["B"]["xp15_mean"] is not None else "—")
            d3.metric("Team A morts 15–30", f"{team_vs['A']['deathsMid_mean']:.2f}" if team_vs["A"]["deathsMid_mean"] is not None else "—")
            d4.metric("Team B morts 15–30", f"{team_vs['B']['deathsMid_mean']:.2f}" if team_vs["B"]["deathsMid_mean"] is not None else "—")
        st.divider()

    tabs_players = st.tabs([p[0] for p in players])

    for tab, (rid_full, game, tag) in zip(tabs_players, players):
        with tab:
            st.subheader(f"Joueur : {rid_full}")

            with st.spinner("Récupération matchs + timelines..."):
                if scrim_mode:
                    bundles = fetch_player_bundle_by_match_list(rid_full, game, tag, scrim_match_ids)
                else:
                    bundles = fetch_player_bundle_normal(rid_full, game, tag, match_count, queue_val)

            bundles_f = [b for b in bundles if pass_filters(b)]
            bundles_f = bundles_f[:max_samples]
            st.caption(f"Matchs inclus: {len(bundles_f)}")

            if not bundles_f:
                st.info("Aucun match ne passe les filtres / joueur absent des scrims.")
                continue

            # aggregate dfs (deaths segments + mid + kills)
            d0_5_all, d5_10_all, d10_15_all, d0_15_all, d15_30_all = [], [], [], [], []
            kills_all_list = []

            zone0_list, zoneMid_list = [], []
            obj_rows = []

            # series aggregation up to max duration among matches analyzed
            max_duration_min = max([b["max_min"] for b in bundles_f])
            series_stack = []

            # laning compare
            matchup_gold_d = {5: [], 10: [], 15: []}
            matchup_xp_d = {5: [], 10: [], 15: []}
            exact_gold_d = {5: [], 10: [], 15: []}
            exact_xp_d = {5: [], 10: [], 15: []}
            lobby_gold_d15, lobby_xp_d15 = [], []

            # basic stats
            deaths0_count, deathsMid_count = [], []

            for b in bundles_f:
                # dfs
                for src, acc in [
                    (b["d_0_5"], d0_5_all),
                    (b["d_5_10"], d5_10_all),
                    (b["d_10_15"], d10_15_all),
                    (b["d_0_15"], d0_15_all),
                    (b["d_15_30"], d15_30_all),
                ]:
                    if src is not None and not src.empty:
                        acc.append(src)

                if b["kills_all"] is not None and not b["kills_all"].empty:
                    kills_all_list.append(b["kills_all"])

                deaths0_count.append(len(b["d_0_15"]) if b["d_0_15"] is not None else 0)
                deathsMid_count.append(len(b["d_15_30"]) if b["d_15_30"] is not None else 0)

                if b["d_0_15"] is not None and not b["d_0_15"].empty:
                    zone0_list.append(b["d_0_15"]["zone"])
                if b["d_15_30"] is not None and not b["d_15_30"].empty:
                    zoneMid_list.append(b["d_15_30"]["zone"])

                obj_rows.append(objectives_0_15(b["timeline"]))

                # series aligned to max_duration_min
                s = b["series_full"]
                if s["minute"].max() < max_duration_min:
                    # extend & ffill
                    last = s.iloc[-1]
                    extra = pd.DataFrame({"minute": range(int(s["minute"].max()) + 1, max_duration_min + 1)})
                    extra["gold"] = last["gold"]
                    extra["xp"] = last["xp"]
                    extra["damage"] = last["damage"]
                    s = pd.concat([s, extra], ignore_index=True)
                series_stack.append(s[["minute", "gold", "xp", "damage"]].copy())

                # laning comparisons (only if not scrim? still works)
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

                # lobby mean diff @15
                others_gold, others_xp = [], []
                for pid in range(1, 11):
                    snaps = get_snapshots_from_timeline(tl, pid, minutes=(15,))
                    if pid != my_pid:
                        if snaps[15]["gold"] is not None:
                            others_gold.append(snaps[15]["gold"])
                        if snaps[15]["xp"] is not None:
                            others_xp.append(snaps[15]["xp"])

                if my_snaps[15]["gold"] is not None and others_gold:
                    lobby_gold_d15.append(my_snaps[15]["gold"] - float(np.mean(others_gold)))
                if my_snaps[15]["xp"] is not None and others_xp:
                    lobby_xp_d15.append(my_snaps[15]["xp"] - float(np.mean(others_xp)))

            # concat
            df_0_5 = pd.concat(d0_5_all, ignore_index=True) if d0_5_all else pd.DataFrame(columns=["matchId","minute","x","y","side","role","zone"])
            df_5_10 = pd.concat(d5_10_all, ignore_index=True) if d5_10_all else pd.DataFrame(columns=["matchId","minute","x","y","side","role","zone"])
            df_10_15 = pd.concat(d10_15_all, ignore_index=True) if d10_15_all else pd.DataFrame(columns=["matchId","minute","x","y","side","role","zone"])
            df_0_15 = pd.concat(d0_15_all, ignore_index=True) if d0_15_all else pd.DataFrame(columns=["matchId","minute","x","y","side","role","zone"])
            df_15_30 = pd.concat(d15_30_all, ignore_index=True) if d15_30_all else pd.DataFrame(columns=["matchId","minute","x","y","side","role","zone"])
            df_kills = pd.concat(kills_all_list, ignore_index=True) if kills_all_list else pd.DataFrame(columns=["matchId","minute","x","y","side","role","zone"])

            # series mean minute-by-minute (integers)
            series_all = pd.concat(series_stack, ignore_index=True)
            series_mean = series_all.groupby("minute", as_index=False).mean(numeric_only=True)
            # ensure integer minutes in plot
            series_mean["minute"] = series_mean["minute"].astype(int)

            # summaries
            champ = bundles_f[0]["champion"]
            role = bundles_f[0]["role"]

            me_deaths0 = float(np.mean(deaths0_count)) if deaths0_count else None
            me_deathsMid = float(np.mean(deathsMid_count)) if deathsMid_count else None

            lane_gold_diff_5 = mean_safe(matchup_gold_d[5])
            lane_gold_diff_10 = mean_safe(matchup_gold_d[10])
            lane_gold_diff_15 = mean_safe(matchup_gold_d[15])
            lane_xp_diff_5 = mean_safe(matchup_xp_d[5])
            lane_xp_diff_10 = mean_safe(matchup_xp_d[10])
            lane_xp_diff_15 = mean_safe(matchup_xp_d[15])

            exact_gold_diff_5 = mean_safe(exact_gold_d[5])
            exact_gold_diff_10 = mean_safe(exact_gold_d[10])
            exact_gold_diff_15 = mean_safe(exact_gold_d[15])
            exact_xp_diff_5 = mean_safe(exact_xp_d[5])
            exact_xp_diff_10 = mean_safe(exact_xp_d[10])
            exact_xp_diff_15 = mean_safe(exact_xp_d[15])

            lobby_gold_diff_15 = mean_safe(lobby_gold_d15)
            lobby_xp_diff_15 = mean_safe(lobby_xp_d15)

            zone0_counts = pd.concat(zone0_list, ignore_index=True).value_counts() if zone0_list else pd.Series(dtype=int)
            zoneMid_counts = pd.concat(zoneMid_list, ignore_index=True).value_counts() if zoneMid_list else pd.Series(dtype=int)
            obj_df = pd.concat(obj_rows, ignore_index=True) if obj_rows else pd.DataFrame()

            # tabs
            t_overview, t_laning, t_macro, t_kills, t_export = st.tabs(
                ["📍 Overview", "⚔️ Laning", "🧭 Macro", "☠️ Kills Heatmap", "📄 Export"]
            )

            with t_overview:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Morts 0–15 (moy.)", f"{me_deaths0:.2f}" if me_deaths0 is not None else "—")
                c2.metric("Morts 15–30 (moy.)", f"{me_deathsMid:.2f}" if me_deathsMid is not None else "—")
                c3.metric("Gold@15 vs lobby", f"{lobby_gold_diff_15:.0f}" if lobby_gold_diff_15 is not None else "—")
                c4.metric("XP@15 vs lobby", f"{lobby_xp_diff_15:.0f}" if lobby_xp_diff_15 is not None else "—")
                st.markdown(f"<div class='card'><b>Profil</b> : {champ} • {role} • Mode: {'SCRIM' if scrim_mode else 'NORMAL'} • {('—' if scrim_mode else queue_label)}</div>", unsafe_allow_html=True)
                st.divider()

                left, right = st.columns([1.35, 1])

                with left:
                    st.markdown("#### Points de morts — Early (0–15) (interactif)")
                    figP = minimap_points_plotly(minimap, df_0_15, "Deaths (0–15) — points BLUE/RED (selon side du match)", point_size, point_alpha)
                    if figP: st.plotly_chart(figP, use_container_width=True)
                    else: st.info("Aucune mort (0–15).")

                    st.markdown("#### Heatmap morts — Early (0–15)")
                    figH = minimap_heatmap_matplotlib(minimap, df_0_15, "Heatmap deaths (0–15)", gridsize=heat_gridsize)
                    if figH: st.pyplot(figH, clear_figure=True)

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

                    st.markdown("#### Mid game (15–30)")
                    figPm = minimap_points_plotly(minimap, df_15_30, "Deaths (15–30) — points BLUE/RED", point_size, point_alpha)
                    if figPm: st.plotly_chart(figPm, use_container_width=True)
                    else: st.info("Aucune mort (15–30).")
                    figHm = minimap_heatmap_matplotlib(minimap, df_15_30, "Heatmap deaths (15–30)", gridsize=heat_gridsize)
                    if figHm: st.pyplot(figHm, clear_figure=True)

                with right:
                    st.markdown("#### Courbes minute-par-minute (entiers)")
                    # Ensure minute is index and integer
                    plot_df = series_mean.set_index("minute")[["gold", "xp", "damage"]]
                    st.line_chart(plot_df[["gold"]], height=170)
                    st.caption("x = minute entière (0,1,2,...) • y = Gold")
                    st.line_chart(plot_df[["xp"]], height=170)
                    st.caption("x = minute entière (0,1,2,...) • y = XP")
                    st.line_chart(plot_df[["damage"]], height=170)
                    st.caption("x = minute entière (0,1,2,...) • y = Dégâts (metric)")

                    st.markdown("#### Durée max analysée")
                    st.metric("Max minute (match le plus long)", f"{int(series_mean['minute'].max())} min")

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
                st.caption("Diff = (toi − adversaire direct même rôle).")

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

            with t_macro:
                st.markdown("### Macro & Rotations — zones role-aware (corrige toplane vs topside)")
                z1, z2 = st.columns(2)
                with z1:
                    st.markdown("#### Zones deaths Early (0–15)")
                    if not zone0_counts.empty:
                        st.dataframe(zone0_counts.rename("deaths").to_frame())
                    else:
                        st.info("Aucune mort early.")
                with z2:
                    st.markdown("#### Zones deaths Mid (15–30)")
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

            with t_kills:
                st.markdown("### Heatmap des zones où le joueur fait le plus de kills")
                k1, k2 = st.columns([1.35, 1])
                with k1:
                    st.markdown("#### Points de kills (interactif)")
                    figKp = minimap_points_plotly(minimap, df_kills, "Kills — points BLUE/RED (side du match)", point_size, point_alpha)
                    if figKp:
                        st.plotly_chart(figKp, use_container_width=True)
                    else:
                        st.info("Aucun kill détecté (positions) sur l’échantillon.")
                with k2:
                    st.markdown("#### Heatmap kills (densité)")
                    figKh = minimap_heatmap_matplotlib(minimap, df_kills, "Heatmap kills (toutes minutes)", gridsize=heat_gridsize)
                    if figKh:
                        st.pyplot(figKh, clear_figure=True)
                    # zones counts kills
                    if not df_kills.empty:
                        st.markdown("#### Zones kills (role-aware)")
                        st.dataframe(df_kills["zone"].value_counts().rename("kills").to_frame())

            with t_export:
                st.markdown("### Export PDF")
                laning_summary = (
                    f"Matchup (moyenne): Gold diff @5={fmt(lane_gold_diff_5)}, @10={fmt(lane_gold_diff_10)}, @15={fmt(lane_gold_diff_15)}. "
                    f"XP diff @5={fmt(lane_xp_diff_5)}, @10={fmt(lane_xp_diff_10)}, @15={fmt(lane_xp_diff_15)}."
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

                    meta = {"champion": champ, "role": role, "queue": ("SCRIM" if scrim_mode else queue_label)}

                    build_player_pdf(
                        out_path=pdf_path,
                        player_label=rid_full,
                        meta=meta,
                        minimap=minimap,
                        deaths_0_15=df_0_15,
                        deaths_15_30=df_15_30,
                        kills_all=df_kills,
                        series_mean=series_mean,
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
