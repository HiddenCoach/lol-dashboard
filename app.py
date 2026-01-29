import os
import time
import io
from typing import Any, Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# Riot routing
# =========================
EUW_PLATFORM = "euw1"     # summoner-v4 + league-v4 (platform routing)
EUW_REGIONAL = "europe"   # match-v5 + account-v1 (regional routing)

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
    # Image origin: top-left. Timeline coords have increasing y upward -> invert for display.
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
                raise RuntimeError(f"Erreur Riot API {r.status_code}: {r.text[:600]}")
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
# Data Dragon: minimap (map11.png)
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

# =========================
# Match helpers
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

def puuid_by_participant_id(match: Dict[str, Any]) -> Dict[int, str]:
    out = {}
    for i, p in enumerate(match["info"]["participants"], start=1):
        out[i] = p.get("puuid")
    return out

def participant_meta(match: Dict[str, Any], puuid: str) -> Dict[str, Any]:
    for p in match["info"]["participants"]:
        if p.get("puuid") == puuid:
            return {
                "championName": p.get("championName"),
                "teamPosition": p.get("teamPosition") or "UNKNOWN",
                "teamId": p.get("teamId"),
                "win": bool(p.get("win", False)),
            }
    return {"championName": "UNKNOWN", "teamPosition": "UNKNOWN", "teamId": 0, "win": False}

def side_from_teamid(team_id: int) -> str:
    return "BLUE" if team_id == 100 else "RED"

def iter_kill_events(timeline: Dict[str, Any]):
    for fr in timeline["info"]["frames"]:
        ts = fr.get("timestamp", 0)
        for ev in fr.get("events", []):
            if ev.get("type") == "CHAMPION_KILL":
                yield ts, ev

def phase_bucket(minute: float) -> Optional[str]:
    if minute <= 5:
        return "0-5"
    if minute <= 10:
        return "5-10"
    if minute <= 15:
        return "10-15"
    return None

def death_positions_by_phase(timeline: Dict[str, Any], victim_pid: int) -> pd.DataFrame:
    out = []
    for ts, ev in iter_kill_events(timeline):
        if ev.get("victimId") != victim_pid:
            continue
        pos = ev.get("position")
        if not pos:
            continue
        t_min = ts / 60000.0
        ph = phase_bucket(t_min)
        if not ph:
            continue
        out.append({"minute": float(t_min), "phase": ph, "x": float(pos["x"]), "y": float(pos["y"])})
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

        total_gold = pf.get("totalGold")
        xp = pf.get("xp")

        dmg = None
        dmg_stats = pf.get("damageStats") or {}
        for k in ["totalDamageDoneToChampions", "damageDealtToChampions", "totalDamageDone"]:
            if k in dmg_stats:
                dmg = dmg_stats[k]
                break

        rows.append({"minute": float(t_min), "gold": total_gold, "xp": xp, "damage": dmg})
    return pd.DataFrame(rows).sort_values("minute")

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

def get_snapshots_from_timeline(timeline: Dict[str, Any], pid: int, minutes=(5, 10, 15)) -> Dict[int, Dict[str, Optional[float]]]:
    """
    Snapshots (gold/xp/dmg) à 5/10/15 pour un participant, sans construire de DF.
    """
    frames = timeline["info"]["frames"]
    out = {}
    for m in minutes:
        out[m] = {"gold": None, "xp": None, "damage": None}

    # On parcourt les frames 0..15 et on met à jour "dernier connu <= m"
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

        dmg = None
        dmg_stats = pf.get("damageStats") or {}
        for k in ["totalDamageDoneToChampions", "damageDealtToChampions", "totalDamageDone"]:
            if k in dmg_stats:
                dmg = dmg_stats[k]
                break

        for m in minutes:
            if t_min <= m:
                out[m] = {
                    "gold": float(gold) if gold is not None else None,
                    "xp": float(xp) if xp is not None else None,
                    "damage": float(dmg) if dmg is not None else None,
                }
    return out

def mean_safe(vals: List[Optional[float]]) -> Optional[float]:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else None

# =========================
# Matchup + Lobby comparisons
# =========================
def find_lane_opponent_pid(match: Dict[str, Any], my_puuid: str, my_role: str) -> Optional[int]:
    """
    Adversaire direct: même teamPosition, team opposée.
    """
    me_team = None
    for p in match["info"]["participants"]:
        if p.get("puuid") == my_puuid:
            me_team = p.get("teamId")
            break
    if me_team is None or not my_role or my_role == "UNKNOWN":
        return None

    for i, p in enumerate(match["info"]["participants"], start=1):
        if p.get("teamId") != me_team and (p.get("teamPosition") or "UNKNOWN") == my_role:
            return i
    return None

# =========================
# Macro & Rotations
# =========================
def death_zone(x: float, y: float) -> str:
    """
    Classification simple et coach-friendly
    """
    nx, ny = norm_xy(x, y)
    d_diag = abs(nx - ny)
    dist_center = ((nx - 0.5) ** 2 + (ny - 0.5) ** 2) ** 0.5

    if d_diag < 0.06 and dist_center < 0.28:
        return "MID"
    if d_diag < 0.08:
        return "RIVER"
    if ny < nx - 0.12:
        return "TOPSIDE"
    if ny > nx + 0.12:
        return "BOTSIDE"
    return "JUNGLE"

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
def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def plot_minimap_deaths(
    minimap: Image.Image,
    deaths_df: pd.DataFrame,
    title: str,
    side: str,
    style: str = "scatter",
    point_size: int = 95,
    alpha: float = 0.78,
):
    """
    Points gros + couleur selon side (BLUE/RED)
    """
    if deaths_df is None or deaths_df.empty:
        return None

    w, h = minimap.size
    nxy = np.array([norm_xy(x, y) for x, y in zip(deaths_df["x"].values, deaths_df["y"].values)])
    px_py = np.array([norm_to_px(nx, ny, w, h) for nx, ny in nxy])
    px, py = px_py[:, 0], px_py[:, 1]

    color = "#1E88E5" if side == "BLUE" else "#E53935"

    fig = plt.figure(figsize=(6.2, 6.2))
    ax = plt.gca()
    ax.imshow(minimap)
    ax.set_title(title)
    ax.set_xlabel("X (pixels minimap)")
    ax.set_ylabel("Y (pixels minimap)")

    if style == "scatter" or len(px) < 20:
        ax.scatter(
            px, py,
            s=point_size,
            alpha=alpha,
            c=color,
            edgecolors="white",
            linewidths=1.0
        )
    else:
        ax.hexbin(px, py, gridsize=42, mincnt=1, alpha=0.65)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    return fig

def plot_line(df: pd.DataFrame, x_col: str, y_col: str, title: str, x_label: str, y_label: str):
    if df is None or df.empty or df[y_col].dropna().empty:
        return None
    fig = plt.figure(figsize=(6.4, 3.2))
    plt.plot(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel(x_label)  # ex: Temps (minutes)
    plt.ylabel(y_label)  # ex: Gold
    return fig

# =========================
# PDF export
# =========================
def build_player_pdf(
    out_path: str,
    player_label: str,
    meta: Dict[str, str],
    minimap: Image.Image,
    deaths_all: pd.DataFrame,
    deaths_by_phase: Dict[str, pd.DataFrame],
    early_mean: pd.DataFrame,
    laning_summary: str,
    macro_summary: str,
):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    story = []

    story.append(Paragraph(f"<b>Rapport Coach</b> — {player_label}", styles["Title"]))
    story.append(Paragraph(
        f"Champion: {meta.get('champion','—')} | Rôle: {meta.get('role','—')} | Side: {meta.get('side','—')} | Queue: {meta.get('queue','—')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 10))

    # Heatmap globale
    fig = plot_minimap_deaths(minimap, deaths_all, "Morts (0–15) — minimap", side=meta.get("side", "BLUE"), style="scatter")
    if fig:
        story.append(RLImage(io.BytesIO(fig_to_png_bytes(fig)), width=420, height=420))
        story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("Aucune mort 0–15 détectée dans l’échantillon.", styles["Normal"]))

    story.append(Paragraph("<b>Heatmaps par phase</b>", styles["Heading2"]))
    for ph in ["0-5", "5-10", "10-15"]:
        figp = plot_minimap_deaths(minimap, deaths_by_phase.get(ph), f"Morts — {ph}", side=meta.get("side", "BLUE"), style="scatter", point_size=80)
        if figp:
            story.append(RLImage(io.BytesIO(fig_to_png_bytes(figp)), width=360, height=360))
            story.append(Spacer(1, 8))

    story.append(PageBreak())

    story.append(Paragraph("<b>Early game</b> (0–15)", styles["Heading2"]))
    fg = plot_line(early_mean, "minute", "gold", "Gold (0–15)", "Temps (minutes)", "Gold")
    fx = plot_line(early_mean, "minute", "xp", "XP (0–15)", "Temps (minutes)", "XP")
    fd = plot_line(early_mean, "minute", "damage", "Dégâts (metric) (0–15)", "Temps (minutes)", "Dégâts")
    for f in [fg, fx, fd]:
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
# Streamlit UI
# =========================
st.set_page_config(page_title="EUW Coach Dashboard", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small { opacity:0.85; font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

st.title("EUW — Coach Dashboard")
st.caption("Heatmaps de morts sur minimap + early game 0–15 + review laning (matchup) + macro & rotations.")

with st.sidebar:
    st.header("Entrée")
    riot_ids_text = st.text_area("Joueurs (1 par ligne) : GameName#TAG", height=160)
    match_count = st.slider("Matchs récents / joueur", 1, 40, 20)
    queue_label = st.selectbox("Queue", list(QUEUE_MAP.keys()), index=1)
    queue_val = QUEUE_MAP[queue_label]

    st.header("Filtres coach-friendly")
    filter_role = st.multiselect("Rôle", ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"], default=[])
    filter_side = st.multiselect("Side", ["BLUE", "RED"], default=[])
    filter_champ = st.text_input("Champion (exact, optionnel)")

    st.header("Visuel minimap")
    point_size = st.slider("Taille des points (morts)", 30, 180, 95)
    point_alpha = st.slider("Opacité des points", 0.30, 1.00, 0.78)

    st.header("Perf / charge API")
    max_lobby_samples = st.slider("Max matchs pour comparaison lobby/matchup (limite CPU)", 1, 40, 20)

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

        role = meta.get("teamPosition") or "UNKNOWN"
        side = side_from_teamid(meta.get("teamId", 0))
        champ = meta.get("championName") or "UNKNOWN"

        df_deaths = death_positions_by_phase(tl, pid)
        early = first15_timeseries(tl, pid)
        s15 = snapshot_at_15(early)

        bundles.append({
            "matchId": mid,
            "puuid": puuid,
            "champion": champ,
            "role": role,
            "side": side,
            "win": meta.get("win", False),
            "deaths": df_deaths,
            "early": early,
            "s15": s15,
            "match": match,
            "timeline": tl,
        })
    return bundles

def fmt(v: Optional[float], digits: int = 0) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if digits == 0:
        return f"{v:.0f}"
    return f"{v:.{digits}f}"

if run:
    # minimap
    try:
        minimap = fetch_minimap_image()
    except Exception as e:
        st.error(f"Impossible de télécharger la minimap (Data Dragon). Détail: {e}")
        st.stop()

    players = parse_riot_ids(riot_ids_text)
    if not players:
        st.error("Ajoute au moins un Riot ID valide (GameName#TAG).")
        st.stop()

    tabs = st.tabs([p[0] for p in players])

    for tab, (rid_full, game, tag) in zip(tabs, players):
        with tab:
            st.subheader(f"Joueur : {rid_full}")

            with st.spinner("Récupération matchs + timelines..."):
                bundles = fetch_player_bundle(rid_full, game, tag, match_count, queue_val)

            bundles_f = [b for b in bundles if pass_filters(b)]
            st.caption(f"Matchs chargés: {len(bundles)} | Après filtres: {len(bundles_f)}")

            if not bundles_f:
                st.info("Aucun match ne passe les filtres.")
                continue

            # Aggregate deaths & early
            deaths_all = []
            deaths_by_phase = {"0-5": [], "5-10": [], "10-15": []}
            early_all = []

            gold15_list, xp15_list, dmg15_list, deaths15_list = [], [], [], []

            for b in bundles_f:
                if b["deaths"] is not None and not b["deaths"].empty:
                    deaths_all.append(b["deaths"])
                    for ph in deaths_by_phase.keys():
                        deaths_by_phase[ph].append(b["deaths"][b["deaths"]["phase"] == ph])
                    deaths15_list.append(int(len(b["deaths"])))
                else:
                    deaths15_list.append(0)

                early_all.append(b["early"])
                gold15_list.append(b["s15"]["gold15"])
                xp15_list.append(b["s15"]["xp15"])
                dmg15_list.append(b["s15"]["damage15"])

            deaths_all_df = pd.concat(deaths_all, ignore_index=True) if deaths_all else pd.DataFrame(columns=["minute","phase","x","y"])
            deaths_by_phase_df = {
                ph: (pd.concat(lst, ignore_index=True) if lst else pd.DataFrame(columns=["minute","phase","x","y"]))
                for ph, lst in deaths_by_phase.items()
            }
            early_df = pd.concat(early_all, ignore_index=True) if early_all else pd.DataFrame()
            early_mean = early_df.groupby("minute", as_index=False).mean(numeric_only=True) if not early_df.empty else early_df

            me_gold15 = mean_safe(gold15_list)
            me_xp15 = mean_safe(xp15_list)
            me_dmg15 = mean_safe(dmg15_list)
            me_deaths15 = float(np.mean(deaths15_list)) if deaths15_list else None

            # Lane matchup & lobby compare (no external data)
            matchup_gold_d = {5: [], 10: [], 15: []}
            matchup_xp_d = {5: [], 10: [], 15: []}
            lobby_gold_d15, lobby_xp_d15 = [], []

            # Limit CPU if needed
            for b in bundles_f[:max_lobby_samples]:
                my_role = b["role"]
                my_puuid = b["puuid"]
                match = b["match"]
                tl = b["timeline"]

                my_pid = participant_id_for_puuid(match, my_puuid)

                my_snaps = get_snapshots_from_timeline(tl, my_pid, minutes=(5, 10, 15))

                # matchup
                opp_pid = find_lane_opponent_pid(match, my_puuid, my_role)
                if opp_pid:
                    opp_snaps = get_snapshots_from_timeline(tl, opp_pid, minutes=(5, 10, 15))
                    for m in (5, 10, 15):
                        if my_snaps[m]["gold"] is not None and opp_snaps[m]["gold"] is not None:
                            matchup_gold_d[m].append(my_snaps[m]["gold"] - opp_snaps[m]["gold"])
                        if my_snaps[m]["xp"] is not None and opp_snaps[m]["xp"] is not None:
                            matchup_xp_d[m].append(my_snaps[m]["xp"] - opp_snaps[m]["xp"])

                # lobby mean of other 9 @15
                pid_map = puuid_by_participant_id(match)
                others_gold, others_xp = [], []
                for pid in pid_map.keys():
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

            lane_gold_diff_5 = mean_safe(matchup_gold_d[5])
            lane_gold_diff_10 = mean_safe(matchup_gold_d[10])
            lane_gold_diff_15 = mean_safe(matchup_gold_d[15])

            lane_xp_diff_5 = mean_safe(matchup_xp_d[5])
            lane_xp_diff_10 = mean_safe(matchup_xp_d[10])
            lane_xp_diff_15 = mean_safe(matchup_xp_d[15])

            lobby_gold_diff_15 = mean_safe(lobby_gold_d15)
            lobby_xp_diff_15 = mean_safe(lobby_xp_d15)

            # Macro & rotations: zones + timeline deaths + objectives
            if not deaths_all_df.empty:
                deaths_all_df["zone"] = deaths_all_df.apply(lambda r: death_zone(r["x"], r["y"]), axis=1)
                deaths_zone_counts = deaths_all_df["zone"].value_counts()
                death_timeline = deaths_all_df.sort_values("minute")[["minute", "phase", "zone", "x", "y"]]
            else:
                deaths_zone_counts = pd.Series(dtype=int)
                death_timeline = pd.DataFrame(columns=["minute", "phase", "zone", "x", "y"])

            obj_rows = []
            for b in bundles_f[:max_lobby_samples]:
                obj_rows.append(objectives_0_15(b["timeline"]))
            obj_df = pd.concat(obj_rows, ignore_index=True) if obj_rows else pd.DataFrame()

            # Meta for plotting
            player_side = bundles_f[0]["side"]
            player_role = bundles_f[0]["role"]
            player_champ = bundles_f[0]["champion"]

            # Top metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Gold @15 (moy.)", fmt(me_gold15))
            c2.metric("XP @15 (moy.)", fmt(me_xp15))
            c3.metric("Morts 0–15 (moy.)", fmt(me_deaths15, 2))
            c4.metric("Profil", f"{player_champ} • {player_role} • {player_side}")

            st.divider()

            colA, colB = st.columns([1.25, 1])

            # ===== LEFT: heatmaps on minimap (big points + colored side) =====
            with colA:
                st.markdown("### Heatmap des morts (0–15) — sur minimap")
                fig = plot_minimap_deaths(
                    minimap=minimap,
                    deaths_df=deaths_all_df,
                    title=f"{rid_full} — morts (0–15)",
                    side=player_side,
                    style="scatter",
                    point_size=point_size,
                    alpha=point_alpha,
                )
                if fig:
                    st.pyplot(fig, clear_figure=True)
                else:
                    st.info("Aucune mort détectée (0–15) dans l’échantillon filtré.")

                st.markdown("### Heatmaps par phase")
                p1, p2, p3 = st.columns(3)
                for cc, ph in zip([p1, p2, p3], ["0-5", "5-10", "10-15"]):
                    with cc:
                        figp = plot_minimap_deaths(
                            minimap=minimap,
                            deaths_df=deaths_by_phase_df[ph],
                            title=f"{ph}",
                            side=player_side,
                            style="scatter",
                            point_size=max(60, int(point_size * 0.85)),
                            alpha=point_alpha,
                        )
                        if figp:
                            st.pyplot(figp, clear_figure=True)
                        else:
                            st.caption(f"{ph}: —")

                st.markdown("### Macro & Rotations (0–15)")
                if not deaths_zone_counts.empty:
                    st.write("**Répartition des morts par zone :**")
                    st.dataframe(deaths_zone_counts.rename("deaths").to_frame())

                if death_timeline is not None and not death_timeline.empty:
                    st.write("**Timeline des morts :**")
                    st.dataframe(death_timeline)

                if obj_df is not None and not obj_df.empty:
                    st.write("**Objectifs / events 0–15 :**")
                    st.dataframe(obj_df.sort_values("minute").head(200))
                else:
                    st.caption("Aucun event objectif détecté dans 0–15 sur l’échantillon (selon les matchs).")

            # ===== RIGHT: early charts + laning review =====
            with colB:
                st.markdown("### Early game (0–15)")
                fg = plot_line(early_mean, "minute", "gold", "Gold (0–15)", "Temps (minutes)", "Gold")
                fx = plot_line(early_mean, "minute", "xp", "XP (0–15)", "Temps (minutes)", "XP")
                fd = plot_line(early_mean, "minute", "damage", "Dégâts (metric) (0–15)", "Temps (minutes)", "Dégâts")

                for f in [fg, fx, fd]:
                    if f:
                        st.pyplot(f, clear_figure=True)

                st.markdown("### Review Laning (vs matchup direct)")
                l1, l2, l3 = st.columns(3)
                l1.metric("Gold diff @5", fmt(lane_gold_diff_5))
                l2.metric("Gold diff @10", fmt(lane_gold_diff_10))
                l3.metric("Gold diff @15", fmt(lane_gold_diff_15))

                x1, x2, x3 = st.columns(3)
                x1.metric("XP diff @5", fmt(lane_xp_diff_5))
                x2.metric("XP diff @10", fmt(lane_xp_diff_10))
                x3.metric("XP diff @15", fmt(lane_xp_diff_15))

                st.caption("Diff = (toi − adversaire direct même rôle). Si le rôle est UNKNOWN, la comparaison peut être vide.")

                st.markdown("### Comparaison (vs moyenne lobby)")
                a1, a2 = st.columns(2)
                a1.metric("Gold@15 vs lobby", fmt(lobby_gold_diff_15))
                a2.metric("XP@15 vs lobby", fmt(lobby_xp_diff_15))

            # ===== Table match-by-match =====
            st.divider()
            st.markdown("### Table (match par match) — utile VOD review")
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
                    "dmg@15": b["s15"]["damage15"],
                    "deaths_0_15": int(len(b["deaths"])) if b["deaths"] is not None else 0,
                })
            st.dataframe(pd.DataFrame(rows))

            # ===== Export PDF =====
            st.divider()
            st.markdown("### Export PDF")
            if st.button(f"Générer PDF — {rid_full}", key=f"pdf_{rid_full}"):
                os.makedirs("exports", exist_ok=True)
                pdf_path = os.path.join("exports", f"{rid_full.replace('#','_')}_report.pdf")

                laning_summary = (
                    f"Matchup (moyenne sur l’échantillon): "
                    f"Gold diff @5={fmt(lane_gold_diff_5)}, @10={fmt(lane_gold_diff_10)}, @15={fmt(lane_gold_diff_15)}. "
                    f"XP diff @5={fmt(lane_xp_diff_5)}, @10={fmt(lane_xp_diff_10)}, @15={fmt(lane_xp_diff_15)}. "
                    f"Lobby: Gold@15 vs moyenne={fmt(lobby_gold_diff_15)}, XP@15 vs moyenne={fmt(lobby_xp_diff_15)}."
                )
                macro_summary = (
                    f"Morts 0–15 (moy.)={fmt(me_deaths15,2)}. "
                    f"Zones les plus fréquentes: "
                    f"{', '.join([f'{k}({int(v)})' for k, v in deaths_zone_counts.head(4).items()]) if not deaths_zone_counts.empty else '—'}. "
                    f"Objectifs/events détectés 0–15: {len(obj_df) if obj_df is not None and not obj_df.empty else 0}."
                )

                meta = {
                    "champion": player_champ,
                    "role": player_role,
                    "side": player_side,
                    "queue": queue_label,
                }

                build_player_pdf(
                    out_path=pdf_path,
                    player_label=rid_full,
                    meta=meta,
                    minimap=minimap,
                    deaths_all=deaths_all_df,
                    deaths_by_phase=deaths_by_phase_df,
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
