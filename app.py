import os, time, io
from typing import Any, Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

EUW_REGIONAL = "europe"

# ---------- Riot Client ----------
class RiotClient:
    def __init__(self, api_key: Optional[str] = None, timeout_s: int = 20, sleep_on_429_s: int = 2):
        self.api_key = api_key or os.getenv("RIOT_API_KEY")
        if not self.api_key:
            raise RuntimeError("RIOT_API_KEY manquant. Ajoute-le dans Streamlit Cloud > Settings > Secrets.")
        self.timeout_s = timeout_s
        self.sleep_on_429_s = sleep_on_429_s

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        for attempt in range(4):
            r = requests.get(url, headers={"X-Riot-Token": self.api_key}, params=params, timeout=self.timeout_s)
            if r.status_code == 429:
                time.sleep(self.sleep_on_429_s * (attempt + 1))
                continue
            if not r.ok:
                raise RuntimeError(f"Erreur Riot API {r.status_code}: {r.text[:500]}")
            return r.json()
        raise RuntimeError("Rate limit (429) persistant. Baisse matchs/joueurs.")

    def get_puuid_by_riot_id(self, game_name: str, tag_line: str) -> str:
        url = f"https://{EUW_REGIONAL}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        return self._get(url)["puuid"]

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

# ---------- Analysis ----------
MAP_MIN_X, MAP_MIN_Y = -120, -120
MAP_MAX_X, MAP_MAX_Y = 14870, 14980

def norm_xy(x: float, y: float) -> Tuple[float, float]:
    nx = (x - MAP_MIN_X) / (MAP_MAX_X - MAP_MIN_X)
    ny = (y - MAP_MIN_Y) / (MAP_MAX_Y - MAP_MIN_Y)
    return nx, ny

def participant_id_for_puuid(match: Dict[str, Any], puuid: str) -> int:
    for i, p in enumerate(match["info"]["participants"], start=1):
        if p.get("puuid") == puuid:
            return i
    raise ValueError("PUUID non trouvé dans ce match.")

def participant_meta(match: Dict[str, Any], puuid: str) -> Dict[str, Any]:
    for p in match["info"]["participants"]:
        if p.get("puuid") == puuid:
            return {
                "championName": p.get("championName"),
                "teamPosition": p.get("teamPosition"),
                "teamId": p.get("teamId"),
                "win": p.get("win"),
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
        out.append({"minute": t_min, "phase": ph, "x": float(pos["x"]), "y": float(pos["y"])})
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

        rows.append({"minute": float(t_min), "totalGold": total_gold, "xp": xp, "damageMetric": dmg})
    return pd.DataFrame(rows).sort_values("minute")

# ---------- Render ----------
def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def plot_death_heatmap(df_deaths: pd.DataFrame, title: str):
    if df_deaths is None or df_deaths.empty:
        return None
    nxy = np.array([norm_xy(x, y) for x, y in zip(df_deaths["x"].values, df_deaths["y"].values)])
    nx, ny = nxy[:, 0], nxy[:, 1]
    fig = plt.figure()
    plt.hexbin(nx, ny, gridsize=48, mincnt=1)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("X (normalisé)")
    plt.ylabel("Y (normalisé)")
    return fig

def build_player_pdf(out_path: str, player_label: str, meta: Dict[str, str],
                     deaths_all: pd.DataFrame, deaths_by_phase: Dict[str, pd.DataFrame], early_df: pd.DataFrame):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    story = []

    story.append(Paragraph(f"<b>Rapport joueur</b> — {player_label}", styles["Title"]))
    story.append(Paragraph(f"Champion: {meta.get('champion','—')} | Role: {meta.get('role','—')} | Side: {meta.get('side','—')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    fig = plot_death_heatmap(deaths_all, "Heatmap morts — global (0–15)")
    if fig:
        story.append(Paragraph("<b>Morts (0–15)</b>", styles["Heading2"]))
        story.append(RLImage(io.BytesIO(fig_to_png_bytes(fig)), width=480, height=320))
        story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("Aucune mort trouvée sur 0–15 min dans l’échantillon.", styles["Normal"]))
        story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Heatmaps par phase</b>", styles["Heading2"]))
    for ph in ["0-5", "5-10", "10-15"]:
        figp = plot_death_heatmap(deaths_by_phase.get(ph), f"Heatmap morts — {ph}")
        if figp:
            story.append(Paragraph(f"Phase {ph}", styles["Heading3"]))
            story.append(RLImage(io.BytesIO(fig_to_png_bytes(figp)), width=480, height=320))
            story.append(Spacer(1, 10))

    story.append(PageBreak())
    story.append(Paragraph("<b>Early game — 0 à 15</b>", styles["Heading2"]))

    if early_df is not None and not early_df.empty:
        fig = plt.figure()
        plt.plot(early_df["minute"], early_df["totalGold"])
        plt.title(f"{player_label} — Gold (0–15)")
        plt.xlabel("Minute"); plt.ylabel("Total Gold")
        story.append(RLImage(io.BytesIO(fig_to_png_bytes(fig)), width=480, height=280))
        story.append(Spacer(1, 10))

        fig = plt.figure()
        plt.plot(early_df["minute"], early_df["xp"])
        plt.title(f"{player_label} — XP (0–15)")
        plt.xlabel("Minute"); plt.ylabel("XP")
        story.append(RLImage(io.BytesIO(fig_to_png_bytes(fig)), width=480, height=280))
        story.append(Spacer(1, 10))

        if early_df["damageMetric"].notna().any():
            fig = plt.figure()
            plt.plot(early_df["minute"], early_df["damageMetric"])
            plt.title(f"{player_label} — Damage metric (0–15)")
            plt.xlabel("Minute"); plt.ylabel("Damage")
            story.append(RLImage(io.BytesIO(fig_to_png_bytes(fig)), width=480, height=280))
            story.append(Spacer(1, 10))

    doc.build(story)

def parse_riot_ids(text: str):
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or "#" not in line:
            continue
        game, tag = line.split("#", 1)
        out.append((line, game.strip(), tag.strip()))
    return out

# ---------- Streamlit UI ----------
st.set_page_config(page_title="EUW Coach Dashboard", layout="wide")
st.title("EUW — Coach Dashboard (Heatmaps + Early 0–15)")

st.sidebar.header("Entrée")
riot_ids_text = st.sidebar.text_area("Joueurs (1 par ligne) : GameName#TAG", height=180)
match_count = st.sidebar.slider("Matchs récents / joueur", 1, 40, 20)
queue = st.sidebar.selectbox("Queue", ["Toutes", "SoloQ (420)", "Flex (440)"], index=0)
queue_val = None if queue == "Toutes" else int(queue.split("(")[1].split(")")[0])

st.sidebar.header("Filtres coach-friendly")
filter_role = st.sidebar.multiselect("Rôle", ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"], default=[])
filter_side = st.sidebar.multiselect("Side", ["BLUE", "RED"], default=[])
filter_champ = st.sidebar.text_input("Champion (exact, optionnel)")

run = st.sidebar.button("Analyser (EUW)")

@st.cache_data(show_spinner=False, ttl=60*20)
def fetch_player_bundle(riot_id_full: str, game: str, tag: str, count: int, queue_val):
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
        early["matchId"] = mid

        bundles.append({"matchId": mid, "champion": champ, "role": role, "side": side, "win": bool(meta.get("win", False)),
                        "deaths": df_deaths, "early": early})
    return bundles

def pass_filters(bundle) -> bool:
    if filter_role and bundle["role"] not in filter_role: return False
    if filter_side and bundle["side"] not in filter_side: return False
    if filter_champ.strip() and bundle["champion"] != filter_champ.strip(): return False
    return True

if run:
    players = parse_riot_ids(riot_ids_text)
    if not players:
        st.error("Ajoute au moins un Riot ID valide (GameName#TAG).")
        st.stop()

    tabs = st.tabs([p[0] for p in players])

    for tab, (rid_full, game, tag) in zip(tabs, players):
        with tab:
            st.subheader(f"Joueur : {rid_full}")

            with st.spinner("Récupération EUW (match + timeline)..."):
                bundles = fetch_player_bundle(rid_full, game, tag, match_count, queue_val)

            bundles_f = [b for b in bundles if pass_filters(b)]
            st.caption(f"Matchs chargés: {len(bundles)} | Après filtres: {len(bundles_f)}")

            if not bundles_f:
                st.info("Aucun match ne passe les filtres.")
                continue

            deaths_all = []
            deaths_by_phase = {"0-5": [], "5-10": [], "10-15": []}
            early_all = []

            for b in bundles_f:
                if b["deaths"] is not None and not b["deaths"].empty:
                    deaths_all.append(b["deaths"])
                    for ph in deaths_by_phase.keys():
                        deaths_by_phase[ph].append(b["deaths"][b["deaths"]["phase"] == ph])
                early_all.append(b["early"])

            deaths_all_df = pd.concat(deaths_all, ignore_index=True) if deaths_all else pd.DataFrame(columns=["minute","phase","x","y"])
            deaths_by_phase_df = {
                ph: (pd.concat(lst, ignore_index=True) if lst else pd.DataFrame(columns=["minute","phase","x","y"]))
                for ph, lst in deaths_by_phase.items()
            }
            early_df = pd.concat(early_all, ignore_index=True) if early_all else pd.DataFrame()

            colA, colB = st.columns([1.2, 1])

            with colA:
                st.markdown("### Heatmap des morts (0–15) — global")
                fig = plot_death_heatmap(deaths_all_df, f"{rid_full} — morts (0–15)")
                if fig: st.pyplot(fig, clear_figure=True)
                else: st.write("Aucune mort trouvée (0–15).")

                st.markdown("### Heatmaps par phase")
                c1, c2, c3 = st.columns(3)
                for c, ph in zip([c1, c2, c3], ["0-5", "5-10", "10-15"]):
                    with c:
                        figp = plot_death_heatmap(deaths_by_phase_df[ph], f"{ph}")
                        if figp: st.pyplot(figp, clear_figure=True)
                        else: st.write(f"— {ph}: aucune donnée")

            with colB:
                st.markdown("### Early game (0–15)")
                if not early_df.empty:
                    agg = early_df.groupby("minute", as_index=False).agg({"totalGold":"mean","xp":"mean","damageMetric":"mean"})
                    st.line_chart(agg.set_index("minute")[["totalGold"]])
                    st.line_chart(agg.set_index("minute")[["xp"]])
                    if agg["damageMetric"].notna().any():
                        st.line_chart(agg.set_index("minute")[["damageMetric"]])
                    else:
                        st.caption("Damage metric non dispo via frames sur cet échantillon (recalcul via events possible).")

                st.markdown("### Export PDF")
                if st.button(f"Générer PDF pour {rid_full}"):
                    os.makedirs("exports", exist_ok=True)
                    pdf_path = os.path.join("exports", f"{rid_full.replace('#','_')}_report.pdf")
                    m0 = bundles_f[0]
                    meta = {"champion": m0["champion"], "role": m0["role"], "side": m0["side"]}
                    early_mean = early_df.groupby("minute", as_index=False).mean(numeric_only=True) if not early_df.empty else early_df
                    build_player_pdf(pdf_path, rid_full, meta, deaths_all_df, deaths_by_phase_df, early_mean)
                    with open(pdf_path, "rb") as f:
                        st.download_button("Télécharger le PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf")
