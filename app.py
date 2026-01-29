import os, time, io
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

# ---------------- Routing ----------------
EUW_PLATFORM = "euw1"     # summoner-v4 + league-v4 (platform routing)
EUW_REGIONAL = "europe"   # match-v5 + account-v1 (regional routing)

QUEUE_MAP = {
    "Toutes": None,
    "SoloQ (420)": 420,
    "Flex (440)": 440,
}
QUEUE_TYPE_MAP = {
    None: "RANKED_SOLO_5x5",      # fallback
    420: "RANKED_SOLO_5x5",
    440: "RANKED_FLEX_SR",
}

# ---------------- Map bounds (Summoner's Rift map11) ----------------
# d'après bornes publiques de positions timeline
MAP_MIN_X, MAP_MIN_Y = -120, -120
MAP_MAX_X, MAP_MAX_Y = 14870, 14980

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def norm_xy(x: float, y: float) -> Tuple[float, float]:
    nx = (x - MAP_MIN_X) / (MAP_MAX_X - MAP_MIN_X)
    ny = (y - MAP_MIN_Y) / (MAP_MAX_Y - MAP_MIN_Y)
    return clamp01(nx), clamp01(ny)

def norm_to_px(nx: float, ny: float, w: int, h: int) -> Tuple[float, float]:
    # image origin top-left; timeline y grows upward -> invert for display
    px = nx * w
    py = (1.0 - ny) * h
    return px, py

# ---------------- Riot Client ----------------
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
        raise RuntimeError("Rate limit (429) persistant. Baisse matchs/joueurs, ou réessaie plus tard.")

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

    # summoner-v4 (platform): by-puuid -> encryptedSummonerId
    def get_summoner_by_puuid(self, puuid: str) -> Dict[str, Any]:
        url = f"https://{EUW_PLATFORM}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}"
        return self._get(url)

    # league-v4 (platform): entries by encryptedSummonerId
    def get_league_entries_by_summoner(self, encrypted_summoner_id: str) -> List[Dict[str, Any]]:
        url = f"https://{EUW_PLATFORM}.api.riotgames.com/lol/league/v4/entries/by-summoner/{encrypted_summoner_id}"
        return self._get(url)

# ---------------- DDragon minimap (map11.png) ----------------
@st.cache_data(show_spinner=False, ttl=60*60*24)
def get_latest_ddragon_version() -> str:
    # versions.json renvoie une liste (latest en [0])
    r = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=15)
    r.raise_for_status()
    return r.json()[0]

@st.cache_data(show_spinner=False, ttl=60*60*24)
def fetch_minimap_image() -> Image.Image:
    ver = get_latest_ddragon_version()
    url = f"https://ddragon.leagueoflegends.com/cdn/{ver}/img/map/map11.png"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")

# ---------------- Match parsing helpers ----------------
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
                "teamPosition": p.get("teamPosition"),
                "teamId": p.get("teamId"),
                "win": bool(p.get("win", False)),
            }
    return {}

def side_from_teamid(team_id: int) -> str:
    return "BLUE" if team_id == 100 else "RED"

def iter_kill_events(timeline: Dict[str, Any]):
    for fr in timeline["info"]["frames"]:
        ts = fr.get("timestamp", 0)
        for ev in fr.get("events", []):
            if ev.get("type") == "CHAMPION_KILL":
                yield ts, ev

def phase_bucket(minute: float) -> Optional[str]:
    if minute <= 5: return "0-5"
    if minute <= 10: return "5-10"
    if minute <= 15: return "10-15"
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
    for idx, fr in enumerate(timeline["info"]["frames"]):
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

# ---------------- Rank (same tier/division comparison) ----------------
@st.cache_data(show_spinner=False, ttl=60*30)
def get_rank_for_puuid(puuid: str, queue_type: str) -> Optional[Tuple[str, str, int]]:
    """
    Retourne (tier, rank/division, leaguePoints) ex: ("GOLD","II", 63)
    None si non classé sur cette queue.
    """
    client = RiotClient()
    summ = client.get_summoner_by_puuid(puuid)
    enc_id = summ.get("id")
    entries = client.get_league_entries_by_summoner(enc_id)
    for e in entries:
        if e.get("queueType") == queue_type:
            tier = e.get("tier")
            div = e.get("rank")
            lp = e.get("leaguePoints")
            if tier and div is not None:
                return (str(tier), str(div), int(lp))
    return None

# ---------------- Pretty plotting ----------------
def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def plot_minimap_heatmap(
    minimap: Image.Image,
    deaths_df: pd.DataFrame,
    title: str,
    mode: str = "hex",
):
    """
    Heatmap sur minimap. deaths_df contient colonnes x,y (coords timeline).
    mode: "hex" ou "scatter"
    """
    if deaths_df is None or deaths_df.empty:
        return None

    w, h = minimap.size
    # Convert coords -> pixels
    nxy = np.array([norm_xy(x, y) for x, y in zip(deaths_df["x"].values, deaths_df["y"].values)])
    px_py = np.array([norm_to_px(nx, ny, w, h) for nx, ny in nxy])
    px, py = px_py[:, 0], px_py[:, 1]

    fig = plt.figure(figsize=(6.2, 6.2))
    ax = plt.gca()
    ax.imshow(minimap)
    ax.set_title(title)
    ax.set_xlabel("X (minimap pixels)")
    ax.set_ylabel("Y (minimap pixels)")

    if mode == "scatter" or len(px) < 20:
        ax.scatter(px, py, s=22, alpha=0.6)
    else:
        ax.hexbin(px, py, gridsize=42, mincnt=1, alpha=0.7)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # keep origin top-left
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

# ---------------- PDF export ----------------
def build_player_pdf(
    out_path: str,
    player_label: str,
    meta: Dict[str, str],
    minimap: Image.Image,
    deaths_all: pd.DataFrame,
    deaths_by_phase: Dict[str, pd.DataFrame],
    early_mean: pd.DataFrame,
    peer_summary: Optional[Dict[str, Any]] = None,
):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    story = []

    story.append(Paragraph(f"<b>Rapport Coach</b> — {player_label}", styles["Title"]))
    story.append(Paragraph(
        f"Champion: {meta.get('champion','—')} | Rôle: {meta.get('role','—')} | Side: {meta.get('side','—')} | Queue: {meta.get('queue','—')}",
        styles["Normal"]
    ))
    if peer_summary and peer_summary.get("rank_label"):
        story.append(Paragraph(f"Rang: {peer_summary['rank_label']}", styles["Normal"]))
    story.append(Spacer(1, 10))

    # Heatmap globale
    fig = plot_minimap_heatmap(minimap, deaths_all, "Morts (0–15) — Heatmap sur minimap", mode="hex")
    if fig:
        story.append(RLImage(io.BytesIO(fig_to_png_bytes(fig)), width=420, height=420))
        story.append(Spacer(1, 10))

    # Phases
    story.append(Paragraph("<b>Phases</b> (0–5 / 5–10 / 10–15)", styles["Heading2"]))
    for ph in ["0-5", "5-10", "10-15"]:
        figp = plot_minimap_heatmap(minimap, deaths_by_phase.get(ph), f"Morts — {ph}", mode="hex")
        if figp:
            story.append(RLImage(io.BytesIO(fig_to_png_bytes(figp)), width=360, height=360))
            story.append(Spacer(1, 8))

    story.append(PageBreak())

    # Courbes
    story.append(Paragraph("<b>Early game</b> (0–15)", styles["Heading2"]))
    fg = plot_line(early_mean, "minute", "gold", "Gold (0–15)", "Temps (minutes)", "Gold")
    fx = plot_line(early_mean, "minute", "xp", "XP (0–15)", "Temps (minutes)", "XP")
    fd = plot_line(early_mean, "minute", "damage", "Dégâts (metric) (0–15)", "Temps (minutes)", "Dégâts")
    for f in [fg, fx, fd]:
        if f:
            story.append(RLImage(io.BytesIO(fig_to_png_bytes(f)), width=460, height=230))
            story.append(Spacer(1, 8))

    if peer_summary:
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>Comparaison rang</b>", styles["Heading2"]))
        story.append(Paragraph(peer_summary.get("text", "—"), styles["Normal"]))

    doc.build(story)

# ---------------- UI ----------------
st.set_page_config(page_title="EUW Coach Dashboard", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
.metric-row { display:flex; gap:12px; flex-wrap:wrap; }
.small { opacity:0.85; font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

st.title("EUW — Coach Dashboard")
st.caption("Heatmaps de morts sur minimap + early game 0–15 + comparaison même rang (tier/division).")

with st.sidebar:
    st.header("Entrée")
    riot_ids_text = st.text_area("Joueurs (1 par ligne) : GameName#TAG", height=160)
    match_count = st.slider("Matchs récents / joueur", 1, 40, 20)
    queue_label = st.selectbox("Queue", list(QUEUE_MAP.keys()), index=1)
    queue_val = QUEUE_MAP[queue_label]
    queue_type = QUEUE_TYPE_MAP[queue_val]

    st.header("Filtres coach-friendly")
    filter_role = st.multiselect("Rôle", ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"], default=[])
    filter_side = st.multiselect("Side", ["BLUE", "RED"], default=[])
    filter_champ = st.text_input("Champion (exact, optionnel)")

    st.header("Comparaison rang")
    enable_rank_compare = st.checkbox("Comparer aux joueurs du même rang (tier/division)", value=True)
    max_peer_samples = st.slider("Max joueurs peer à échantillonner (limite API)", 5, 60, 25)

    run = st.button("Analyser (EUW)")

def parse_riot_ids(text: str):
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or "#" not in line:
            continue
        game, tag = line.split("#", 1)
        out.append((line, game.strip(), tag.strip()))
    return out

def pass_filters(bundle) -> bool:
    if filter_role and bundle["role"] not in filter_role:
        return False
    if filter_side and bundle["side"] not in filter_side:
        return False
    if filter_champ.strip() and bundle["champion"] != filter_champ.strip():
        return False
    return True

@st.cache_data(show_spinner=False, ttl=60*20)
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

def mean_or_none(vals: List[Optional[float]]) -> Optional[float]:
    vv = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vv)) if vv else None

def format_delta(me: Optional[float], peer: Optional[float], unit: str = "") -> str:
    if me is None or peer is None:
        return "—"
    d = me - peer
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.0f}{unit}"

if run:
    client = RiotClient()

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

            # --- Aggregate deaths & early ---
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

            me_gold15 = mean_or_none(gold15_list)
            me_xp15 = mean_or_none(xp15_list)
            me_dmg15 = mean_or_none(dmg15_list)
            me_deaths15 = float(np.mean(deaths15_list)) if deaths15_list else None

            # --- Rank compare ---
            peer_summary = None
            if enable_rank_compare:
                with st.spinner("Récupération rang + échantillon peers (limité)..."):
                    try:
                        my_rank = get_rank_for_puuid(bundles_f[0]["puuid"], queue_type)
                    except Exception:
                        my_rank = None

                    if my_rank is None:
                        peer_summary = {"rank_label": "Non classé / placements", "text": "Impossible de comparer: joueur non classé sur cette queue."}
                    else:
                        my_tier, my_div, my_lp = my_rank
                        rank_label = f"{my_tier} {my_div} ({my_lp} LP)"

                        # On va chercher des peers DANS les mêmes matchs (participants), et garder seulement ceux du même tier/division
                        peer_gold15, peer_xp15, peer_dmg15, peer_deaths15 = [], [], [], []
                        sampled = 0

                        for b in bundles_f:
                            if sampled >= max_peer_samples:
                                break

                            match = b["match"]
                            tl = b["timeline"]
                            pid_to_puuid = puuid_by_participant_id(match)

                            # On extrait snapshot@15 + deaths@15 pour chaque participant, SANS appels supplémentaires
                            # Les appels supplémentaires servent uniquement à connaître le rang des participants (tier/div).
                            # On échantillonne pour limiter le volume API.
                            for pid, p_puuid in pid_to_puuid.items():
                                if sampled >= max_peer_samples:
                                    break
                                if p_puuid == b["puuid"]:
                                    continue

                                # rank lookup (cache)
                                try:
                                    r = get_rank_for_puuid(p_puuid, queue_type)
                                except Exception:
                                    continue
                                if not r:
                                    continue
                                tier, div, _lp = r
                                if tier != my_tier or div != my_div:
                                    continue

                                # early snapshot for this participant
                                e = first15_timeseries(tl, pid)
                                s15 = snapshot_at_15(e)
                                peer_gold15.append(s15["gold15"])
                                peer_xp15.append(s15["xp15"])
                                peer_dmg15.append(s15["damage15"])

                                ddf = death_positions_by_phase(tl, pid)
                                peer_deaths15.append(int(len(ddf)) if ddf is not None else 0)

                                sampled += 1

                        p_gold15 = mean_or_none(peer_gold15)
                        p_xp15 = mean_or_none(peer_xp15)
                        p_dmg15 = mean_or_none(peer_dmg15)
                        p_deaths15 = float(np.mean(peer_deaths15)) if peer_deaths15 else None

                        peer_summary = {
                            "rank_label": rank_label,
                            "me": {"gold15": me_gold15, "xp15": me_xp15, "dmg15": me_dmg15, "deaths15": me_deaths15},
                            "peer": {"gold15": p_gold15, "xp15": p_xp15, "dmg15": p_dmg15, "deaths15": p_deaths15},
                            "sampled": sampled,
                            "text": (
                                f"Pe ers (même rang) échantillonnés: {sampled}. "
                                f"Gold@15 peer={p_gold15:.0f} / toi={me_gold15:.0f} ({format_delta(me_gold15, p_gold15)}). "
                                f"XP@15 peer={p_xp15:.0f} / toi={me_xp15:.0f} ({format_delta(me_xp15, p_xp15)}). "
                                f"Morts(0–15) peer={p_deaths15:.2f} / toi={me_deaths15:.2f} ({format_delta(me_deaths15, p_deaths15)})."
                            ) if sampled and me_gold15 and me_xp15 else
                            f"Rang: {rank_label}. Peers échantillonnés: {sampled} (insuffisant pour une moyenne stable)."
                        }

            # --- Top summary row ---
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Gold @15 (moy.)", f"{me_gold15:.0f}" if me_gold15 is not None else "—",
                          format_delta(me_gold15, peer_summary["peer"]["gold15"]) if (peer_summary and peer_summary.get("peer")) else None)
            with c2:
                st.metric("XP @15 (moy.)", f"{me_xp15:.0f}" if me_xp15 is not None else "—",
                          format_delta(me_xp15, peer_summary["peer"]["xp15"]) if (peer_summary and peer_summary.get("peer")) else None)
            wit

