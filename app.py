import io
import os
import time
from datetime import datetime, timezone
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

st.set_page_config(page_title="Hidden | LoL Coach Analytics", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
:root{--accent:#7c8cff}
.block-container{max-width:1580px;padding-top:1.6rem}
[data-testid="stSidebar"]{border-right:1px solid #303545}
h1,h2,h3{letter-spacing:-.035em}
h1{font-size:2.2rem!important}
[data-testid="stMetric"]{background:rgba(124,140,255,.065);border:1px solid rgba(124,140,255,.16);border-radius:14px;padding:15px}
[data-testid="stMetricLabel"]{font-size:.88rem}
[data-testid="stMetricValue"]{font-weight:750}
.stTabs [data-baseweb="tab-list"]{gap:6px;flex-wrap:wrap}
.stTabs [data-baseweb="tab"]{border-radius:9px;padding:10px 15px;background:rgba(124,140,255,.06)}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
.hero{padding:23px 25px;border-radius:18px;background:linear-gradient(110deg,rgba(95,108,240,.22),rgba(28,35,62,.07));border:1px solid rgba(130,145,255,.22);margin-bottom:20px}
.muted{opacity:.75;font-size:.92rem}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
ROLE_NAMES = {"TOP":"Top", "JUNGLE":"Jungle", "MIDDLE":"Mid", "BOTTOM":"ADC", "UTILITY":"Support"}
SIDE_COLORS = {"BLUE":"#4a9df5", "RED":"#f26a72"}
WARD_COLORS = {"Control / Pink":"#f14e5b", "Stealth / Support":"#3ddc97", "Farsight / Blue":"#45a9ff", "Non classée":"#aab2c7"}
WINDOWS = [(0,5),(5,10),(10,15),(15,30)]
MAP_MAX = 15000


def api_key():
    try:
        value = st.secrets.get("RIOT_API_KEY", "")
    except Exception:
        value = ""
    return value or os.environ.get("RIOT_API_KEY", "")


def riot_get(path, params=None, platform="europe"):
    key = api_key()
    if not key:
        raise ValueError("Clé Riot absente. Ajoute RIOT_API_KEY dans Streamlit Cloud > Settings > Secrets.")
    url = f"https://{platform}.api.riotgames.com{path}"
    for attempt in range(5):
        try:
            response = requests.get(url, headers={"X-Riot-Token":key}, params=params, timeout=22)
        except requests.RequestException as exc:
            if attempt == 4:
                raise RuntimeError(f"Réseau Riot : {exc}") from exc
            time.sleep(1 + attempt)
            continue
        if response.status_code == 429:
            wait = min(int(response.headers.get("Retry-After", "2")), 12)
            time.sleep(max(1, wait))
            continue
        if response.status_code in (401,403):
            raise RuntimeError("Clé Riot expirée ou accès refusé (401/403). Renouvelle la clé.")
        if response.status_code == 404:
            raise LookupError(f"Ressource introuvable : {path}")
        response.raise_for_status()
        return response.json()
    raise RuntimeError("Rate limit Riot atteint : réduis le nombre de matchs et réessaie.")


@st.cache_data(ttl=10800, show_spinner=False)
def resolve_id(riot_id):
    if "#" not in riot_id:
        raise ValueError("Format requis : GameName#TAG")
    game, tag = [s.strip() for s in riot_id.rsplit("#",1)]
    if not game or not tag:
        raise ValueError("Format requis : GameName#TAG")
    from urllib.parse import quote
    return riot_get(f"/riot/account/v1/accounts/by-riot-id/{quote(game,safe='')}/{quote(tag,safe='')}")["puuid"]


@st.cache_data(ttl=900, show_spinner=False)
def match_ids(puuid, count, queue):
    params = {"count":int(count)}
    if queue:
        params["queue"] = int(queue)
    return riot_get(f"/lol/match/v5/matches/by-puuid/{puuid}/ids", params)


@st.cache_data(ttl=1800, show_spinner=False)
def match_data(mid):
    return riot_get(f"/lol/match/v5/matches/{mid}")


@st.cache_data(ttl=1800, show_spinner=False)
def timeline_data(mid):
    return riot_get(f"/lol/match/v5/matches/{mid}/timeline")


@st.cache_data(ttl=86400, show_spinner=False)
def ddragon():
    try:
        return requests.get("https://ddragon.leagueoflegends.com/api/versions.json",timeout=12).json()[0]
    except Exception:
        return "16.22.1"


@st.cache_data(ttl=86400, show_spinner=False)
def minimap():
    # CommunityDragon map asset: DDragon map11 image is not guaranteed available.
    candidates = [
        "https://raw.communitydragon.org/latest/game/assets/maps/kitpieces/summoners_rift/minimap.png",
        "https://ddragon.leagueoflegends.com/cdn/15.1.1/img/map/map11.png",
    ]
    for url in candidates:
        try:
            r = requests.get(url, timeout=14)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            pass
    return None


def participant(match, puuid):
    return next((p for p in match.get("info",{}).get("participants",[]) if p.get("puuid")==puuid),None)


def pid_of(match, puuid):
    p = participant(match,puuid)
    return int(p.get("participantId",0)) if p else 0


def side_of(p):
    return "BLUE" if p.get("teamId")==100 else "RED"


def full_name(p):
    n = p.get("riotIdGameName") or p.get("summonerName") or "Inconnu"
    t = p.get("riotIdTagline")
    return f"{n}#{t}" if t else n


def collect_events(tl, pid, mid, side, role):
    out = []
    for fr in tl.get("info",{}).get("frames",[]):
        for ev in fr.get("events",[]):
            kind = ev.get("type")
            if kind == "CHAMPION_KILL" and ev.get("victimId")==pid:
                category = "Death"
            elif kind == "CHAMPION_KILL" and ev.get("killerId")==pid:
                category = "Kill"
            elif kind == "WARD_PLACED" and ev.get("creatorId")==pid:
                category = "Ward"
            else:
                continue
            pos = ev.get("position") or {}
            if "x" not in pos or "y" not in pos:
                # Riot timeline often omits WARD_PLACED position. Do not invent coordinates.
                if category == "Ward":
                    out.append({"matchId":mid,"minute":ev.get("timestamp",fr.get("timestamp",0))/60000,
                                "x":np.nan,"y":np.nan,"side":side,"role":role,"category":category,
                                "wardType":ev.get("wardType","UNKNOWN")})
                continue
            out.append({"matchId":mid,"minute":ev.get("timestamp",fr.get("timestamp",0))/60000,
                        "x":float(pos["x"]),"y":float(pos["y"]),"side":side,"role":role,
                        "category":category,"wardType":ev.get("wardType","UNKNOWN")})
    return out


def ward_kind(raw):
    wt = str(raw or "").upper()
    if "CONTROL" in wt or "PINK" in wt:
        return "Control / Pink"
    if "BLUE" in wt or "FARSIGHT" in wt:
        return "Farsight / Blue"
    if any(t in wt for t in ("YELLOW", "TRINKET", "SIGHT", "STEALTH", "TOTEM")):
        return "Stealth / Support"
    return "Non classée"


def collect_series(tl,pid,mid):
    rows=[]
    for fr in tl.get("info",{}).get("frames",[]):
        pf=fr.get("participantFrames",{}).get(str(pid),{})
        if not pf:
            continue
        m=int(fr.get("timestamp",0)//60000)
        stats=pf.get("damageStats") or {}
        rows.append({"matchId":mid,"minute":m,"gold":pf.get("totalGold"),"xp":pf.get("xp"),
                     "cs":(pf.get("minionsKilled") or 0)+(pf.get("jungleMinionsKilled") or 0),
                     "damage":stats.get("totalDamageDoneToChampions",np.nan)})
    return rows


def load_player(rid, count, queue, scrims):
    puuid=resolve_id(rid)
    mids=scrims if scrims else match_ids(puuid,count,queue)
    bundles=[]
    errors=[]
    for mid in dict.fromkeys(mids):
        try:
            m=match_data(mid)
            p=participant(m,puuid)
            if p is None:
                continue
            tl=timeline_data(mid)
            pid=pid_of(m,puuid)
            bundles.append({"matchId":mid,"match":m,"timeline":tl,"player":p,"puuid":puuid,
                            "side":side_of(p),"role":p.get("teamPosition") or p.get("individualPosition") or "UNKNOWN",
                            "events":collect_events(tl,pid,mid,side_of(p),p.get("teamPosition","UNKNOWN")),
                            "series":collect_series(tl,pid,mid)})
        except Exception as exc:
            errors.append(f"{mid}: {exc}")
    return bundles,errors


def event_frame(bundles):
    rows=[e for b in bundles for e in b["events"]]
    return pd.DataFrame(rows,columns=["matchId","minute","x","y","side","role","category","wardType"])


def series_frame(bundles):
    rows=[r for b in bundles for r in b["series"]]
    return pd.DataFrame(rows,columns=["matchId","minute","gold","xp","cs","damage"])


def mean_series(bundles):
    df=series_frame(bundles)
    if df.empty:
        return df
    # Only matches that actually reached each minute contribute to the mean.
    df=df.drop_duplicates(["matchId","minute"],keep="last")
    return df.groupby("minute",as_index=False)[["gold","xp","cs","damage"]].mean()


def event_map(df,title,group="side",size=13,opacity=.8):
    fig=go.Figure()
    bg=minimap()
    if bg is not None:
        fig.add_layout_image(dict(source=bg,xref="x",yref="y",x=0,y=MAP_MAX,
                                  sizex=MAP_MAX,sizey=MAP_MAX,sizing="stretch",layer="below",opacity=.96))
    else:
        fig.add_annotation(text="Fond de carte indisponible",xref="paper",yref="paper",x=.5,y=.5,showarrow=False)
    palette=WARD_COLORS if group=="wardGroup" else SIDE_COLORS
    if df is not None and not df.empty:
        data=df.dropna(subset=["x","y"])
        for name,part in data.groupby(group):
            hover=[f"{r.matchId}<br>{r.minute:.1f} min<br>{r.side}<br>{r.role}" for r in part.itertuples()]
            fig.add_trace(go.Scatter(x=part["x"],y=part["y"],mode="markers",name=str(name),
                                     marker=dict(size=size,color=palette.get(name,"#aab2c7"),opacity=opacity,
                                                 line=dict(width=.7,color="#ffffff")),
                                     text=hover,hovertemplate="%{text}<extra></extra>"))
    fig.update_xaxes(range=[0,MAP_MAX],visible=False,showgrid=False)
    fig.update_yaxes(range=[0,MAP_MAX],visible=False,showgrid=False,scaleanchor="x",scaleratio=1)
    fig.update_layout(title=title,height=490,margin=dict(l=5,r=5,t=48,b=5),
                      legend=dict(orientation="h",y=1.03,x=0),paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#14212b")
    return fig


def line_chart(df,metric,title,other=None,label="Comparaison"):
    fig=go.Figure()
    if not df.empty and metric in df:
        fig.add_trace(go.Scatter(x=df.minute,y=df[metric],name="Joueur",mode="lines",
                                 line=dict(color="#8797ff",width=3)))
    if other is not None and not other.empty and metric in other:
        fig.add_trace(go.Scatter(x=other.minute,y=other[metric],name=label,mode="lines",
                                 line=dict(color="#f6b765",width=2)))
    fig.update_layout(title=title,height=300,margin=dict(l=10,r=10,t=44,b=25),
                      xaxis_title="Minute",legend=dict(orientation="h",y=1.12),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    return fig


def snapshots(tl,pid):
    df=pd.DataFrame(collect_series(tl,pid,"x"))
    if df.empty:
        return {}
    return {minute:df[df.minute<=minute].iloc[-1] for minute in (5,10,15) if not df[df.minute<=minute].empty}


def laning_table(bundles,exact_puuid=None):
    rows=[]
    for b in bundles:
        m=b["match"]
        p=b["player"]
        pid=pid_of(m,b["puuid"])
        enemies=[x for x in m["info"]["participants"] if x.get("teamId")!=p.get("teamId")]
        if exact_puuid:
            opponent=next((x for x in enemies if x.get("puuid")==exact_puuid),None)
        else:
            opponent=next((x for x in enemies if x.get("teamPosition")==p.get("teamPosition")),None)
        if not opponent:
            continue
        mine=snapshots(b["timeline"],pid)
        theirs=snapshots(b["timeline"],opponent.get("participantId",0))
        for minute in (5,10,15):
            if minute not in mine or minute not in theirs:
                continue
            rows.append({"Match":b["matchId"],"Minute":minute,"Adversaire":full_name(opponent),
                         "Champion adverse":opponent.get("championName"),
                         "Gold diff":mine[minute]["gold"]-theirs[minute]["gold"],
                         "XP diff":mine[minute]["xp"]-theirs[minute]["xp"],
                         "CS diff":mine[minute]["cs"]-theirs[minute]["cs"]})
    return pd.DataFrame(rows)


def match_details(b):
    info=b["match"]["info"]
    p=b["player"]
    st.subheader("Draft & fiche du match")
    cols=st.columns(5)
    fields=[("Résultat","Victoire" if p.get("win") else "Défaite"),
            ("Side",side_of(p)),("Champion",p.get("championName","—")),
            ("K / D / A",f"{p.get('kills',0)} / {p.get('deaths',0)} / {p.get('assists',0)}"),
            ("Durée",f"{int(info.get('gameDuration',0)//60)} min")]
    for col,(k,v) in zip(cols,fields):
        col.metric(k,v)
    st.caption(f"Match {b['matchId']} · Patch {info.get('gameVersion','—')} · Queue {info.get('queueId','—')} · "
               f"{datetime.fromtimestamp(info.get('gameCreation',0)/1000,tz=timezone.utc).strftime('%d/%m/%Y %H:%M UTC')}")
    teams={t.get("teamId"):t for t in info.get("teams",[])}
    c1,c2=st.columns(2)
    for col,team_id in ((c1,100),(c2,200)):
        with col:
            st.markdown(f"#### {'🔵 BLUE' if team_id==100 else '🔴 RED'} — {'WIN' if teams.get(team_id,{}).get('win') else 'LOSE'}")
            members=[x for x in info.get("participants",[]) if x.get("teamId")==team_id]
            order={r:i for i,r in enumerate(ROLES)}
            members.sort(key=lambda x:order.get(x.get("teamPosition"),99))
            rows=[]
            for x in members:
                rows.append({"Rôle":ROLE_NAMES.get(x.get("teamPosition"),"—"),"Pseudo":full_name(x),
                             "Champion":x.get("championName"),"K/D/A":f"{x.get('kills',0)}/{x.get('deaths',0)}/{x.get('assists',0)}",
                             "CS":(x.get("totalMinionsKilled") or 0)+(x.get("neutralMinionsKilled") or 0),
                             "Gold":x.get("goldEarned",0),"Dégâts":x.get("totalDamageDealtToChampions",0),
                             "Vision":x.get("visionScore",0)})
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            bans=[str(x.get("championId")) for x in teams.get(team_id,{}).get("bans",[]) if x.get("championId",-1)>0]
            st.caption("Bans (IDs champions) : "+(", ".join(bans) if bans else "non disponibles"))
    st.caption("L'API Match-v5 donne les champions joués et les bans, mais pas l'ordre exact des picks ni toutes les phases de draft.")


def export_pdf(rid,bundles):
    buff=io.BytesIO()
    doc=SimpleDocTemplate(buff,pagesize=A4,rightMargin=35,leftMargin=35,topMargin=35,bottomMargin=35)
    styles=getSampleStyleSheet()
    story=[Paragraph("Hidden | Rapport Coach",styles["Title"]),Spacer(1,12),
           Paragraph(escape(rid),styles["Heading2"]),
           Paragraph(f"Matchs : {len(bundles)}",styles["Normal"]),Spacer(1,12)]
    data=[["Match","Champion","Side","Résultat","KDA"]]
    for b in bundles:
        p=b["player"]
        data.append([b["matchId"][-17:],p.get("championName","—"),b["side"],
                     "WIN" if p.get("win") else "LOSE",f"{p.get('kills',0)}/{p.get('deaths',0)}/{p.get('assists',0)}"])
    table=Table(data,repeatRows=1,colWidths=[140,90,55,65,100])
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#303d67")),
                               ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                               ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke,colors.white]),
                               ("FONTSIZE",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story.append(table)
    doc.build(story)
    return buff.getvalue()


def parse_lines(text):
    return list(dict.fromkeys(s.strip() for s in text.splitlines() if s.strip()))


st.markdown("""<div class='hero'><h1>🎯 Hidden | Coach Analytics</h1>
<div class='muted'>EUW · Analyse individuelle · Laning · Macro · Vision · Draft · Scrims</div></div>""",unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuration")
    ids_text=st.text_area("Riot IDs à analyser (un par ligne)",height=125,placeholder="Joueur#EUW")
    scrim_mode=st.toggle("Mode scrim / match IDs",value=False)
    scrim_text=st.text_area("Match IDs EUW1_…",height=90,disabled=not scrim_mode)
    count=st.slider("Matchs récents",1,30,10,disabled=scrim_mode)
    queue_label=st.selectbox("File",["Toutes","SoloQ (420)","Flex (440)"],index=1,disabled=scrim_mode)
    queue={"Toutes":None,"SoloQ (420)":420,"Flex (440)":440}[queue_label]
    if st.button("Lancer l'analyse",type="primary",use_container_width=True):
        ids=parse_lines(ids_text)
        mids=parse_lines(scrim_text) if scrim_mode else []
        if not ids:
            st.error("Ajoute au moins un Riot ID.")
        elif scrim_mode and not mids:
            st.error("Ajoute au moins un Match ID.")
        elif not api_key():
            st.error("Configure RIOT_API_KEY dans les Secrets.")
        else:
            results={}
            with st.spinner("Récupération des matchs et timelines Riot..."):
                progress=st.progress(0)
                for i,rid in enumerate(ids):
                    try:
                        results[rid]=load_player(rid,count,queue,mids)
                    except Exception as exc:
                        results[rid]=([], [str(exc)])
                    progress.progress((i+1)/len(ids))
            st.session_state["results_v2"]=results
            st.session_state["queue_v2"]=queue
            st.session_state["scrim_v2"]=scrim_mode
    if st.button("Vider les données chargées",use_container_width=True):
        st.session_state.pop("results_v2",None)
        st.rerun()
    st.divider()
    st.caption("La sélection d'un match et les filtres n'effacent jamais les résultats déjà chargés.")

results=st.session_state.get("results_v2")
if not results:
    st.info("Renseigne un Riot ID dans le panneau de gauche puis lance l'analyse.")
    st.stop()

players=st.tabs(list(results))
for tab,(rid,(bundles,errors)) in zip(players,results.items()):
    with tab:
        st.subheader(rid)
        if errors:
            with st.expander(f"{len(errors)} avertissement(s) API / matchs ignorés"):
                for err in errors:
                    st.warning(err)
        if not bundles:
            st.warning("Aucun match exploitable pour ce joueur.")
            continue
        f1,f2,f3,f4=st.columns([1.8,1,1,1])
        with f1:
            options=["Tous les matchs"]+[b["matchId"] for b in bundles]
            chosen=st.selectbox("Périmètre",options,key=f"scope_{rid}")
        with f2:
            sides=st.multiselect("Side",["BLUE","RED"],key=f"sides_{rid}")
        with f3:
            roles=st.multiselect("Rôle",ROLES,key=f"roles_{rid}")
        with f4:
            champs=sorted(set(b["player"].get("championName","—") for b in bundles))
            champion=st.selectbox("Champion",["Tous"]+champs,key=f"champ_{rid}")
        filtered=[b for b in bundles if (chosen=="Tous les matchs" or b["matchId"]==chosen)
                  and (not sides or b["side"] in sides)
                  and (not roles or b["role"] in roles)
                  and (champion=="Tous" or b["player"].get("championName")==champion)]
        if not filtered:
            st.warning("Aucun match ne correspond aux filtres. Les données restent chargées : ajuste les filtres.")
            continue
        wins=sum(bool(b["player"].get("win")) for b in filtered)
        k=sum(b["player"].get("kills",0) for b in filtered)
        d=sum(b["player"].get("deaths",0) for b in filtered)
        a=sum(b["player"].get("assists",0) for b in filtered)
        duration=sum(max(1,b["match"]["info"].get("gameDuration",0))/60 for b in filtered)
        cs=sum((b["player"].get("totalMinionsKilled") or 0)+(b["player"].get("neutralMinionsKilled") or 0) for b in filtered)
        metrics=st.columns(5)
        for col,label,value in zip(metrics,["Matchs","Winrate","KDA","CS/min","Vision / match"],
                                   [str(len(filtered)),f"{wins/len(filtered)*100:.1f}%",f"{(k+a)/max(1,d):.2f}",
                                    f"{cs/max(1,duration):.2f}",
                                    f"{np.mean([b['player'].get('visionScore',0) for b in filtered]):.1f}"]):
            col.metric(label,value)
        if chosen!="Tous les matchs":
            match_details(filtered[0])
        events=event_frame(filtered)
        series=mean_series(filtered)
        overview,laning,macro,warding,kills,compare,teams,export=st.tabs(
            ["📊 Overview","⚔️ Laning","🧭 Macro","👁️ Warding","☠️ Kills","🔄 Comparer","👥 Teams","📥 Export"])
        with overview:
            st.markdown("#### Évolution des ressources")
            left,right=st.columns(2)
            with left:
                st.plotly_chart(line_chart(series,"gold","Gold cumulés"),use_container_width=True)
                st.plotly_chart(line_chart(series,"cs","CS cumulés"),use_container_width=True)
            with right:
                st.plotly_chart(line_chart(series,"xp","XP cumulée"),use_container_width=True)
                st.plotly_chart(line_chart(series,"damage","Dégâts cumulés aux champions (si disponibles)"),use_container_width=True)
            deaths=events[events.category=="Death"]
            st.plotly_chart(event_map(deaths,"Positions des morts · toutes les minutes"),use_container_width=True)
        with laning:
            st.markdown("#### Différences de ressources contre l'adversaire de rôle")
            exact=st.text_input("Riot ID de l'adversaire exact (optionnel)",key=f"opp_{rid}")
            opponent_puuid=None
            if exact.strip():
                try:
                    opponent_puuid=resolve_id(exact.strip())
                except Exception as exc:
                    st.warning(str(exc))
            lane=laning_table(filtered,opponent_puuid)
            if lane.empty:
                st.info("Aucun matchup comparable trouvé sur ce périmètre.")
            else:
                st.dataframe(lane,hide_index=True,use_container_width=True)
                st.markdown("#### Moyenne par minute")
                st.dataframe(lane.groupby("Minute")[["Gold diff","XP diff","CS diff"]].mean().round(1),use_container_width=True)
        with macro:
            st.markdown("#### Morts par période")
            deaths=events[events.category=="Death"]
            cols=st.columns(2)
            for idx,(start,end) in enumerate(WINDOWS):
                part=deaths[(deaths.minute>=start)&(deaths.minute<end)]
                with cols[idx%2]:
                    st.plotly_chart(event_map(part,f"Morts {start}–{end} min"),use_container_width=True)
            objective_rows=[]
            for b in filtered:
                for fr in b["timeline"].get("info",{}).get("frames",[]):
                    for ev in fr.get("events",[]):
                        if ev.get("type") in ("ELITE_MONSTER_KILL","BUILDING_KILL","TURRET_PLATE_DESTROYED"):
                            objective_rows.append({"Match":b["matchId"],"Minute":round(ev.get("timestamp",0)/60000,1),
                                                   "Événement":ev.get("type"),"Monstre":ev.get("monsterType","—"),
                                                   "Équipe / acteur":ev.get("killerId",ev.get("teamId","—"))})
            st.markdown("#### Chronologie des objectifs")
            if objective_rows:
                st.dataframe(pd.DataFrame(objective_rows).sort_values("Minute"),hide_index=True,use_container_width=True)
            else:
                st.info("Aucun événement objectif disponible.")
        with warding:
            st.markdown("#### Vision · 4 périodes")
            st.caption("Rouge : control wards · Vert : trinkets jaunes et wards de support lorsqu'identifiables · Bleu : farsight.")
            st.warning("Limite Riot API : les événements WARD_PLACED n'incluent généralement pas de coordonnées. Les emplacements absents ne peuvent pas être reconstitués de manière fiable. Les cartes ne montrent que les positions réellement fournies.")
            ward=events[events.category=="Ward"].copy()
            ward["wardGroup"]=ward["wardType"].map(ward_kind)
            ward_side=st.selectbox("Side du joueur",["Tous","BLUE","RED"],key=f"ward_side_{rid}")
            if ward_side!="Tous":
                ward=ward[ward.side==ward_side]
            st.metric("Wards placées (événements détectés)",len(ward))
            if not ward.empty:
                st.dataframe(ward.groupby(["wardGroup","side"]).size().rename("Nombre").reset_index(),hide_index=True,use_container_width=True)
            cols=st.columns(2)
            for idx,(start,end) in enumerate(WINDOWS):
                part=ward[(ward.minute>=start)&(ward.minute<end)]
                with cols[idx%2]:
                    st.plotly_chart(event_map(part,f"Vision {start}–{end} min",group="wardGroup"),use_container_width=True)
                    st.caption(f"{len(part)} poses détectées · {len(part.dropna(subset=['x','y']))} positions disponibles")
            st.divider()
            st.markdown("#### Comparer la vision d'un joueur pro")
            pro_id=st.text_input("Riot ID du joueur pro (EUW)",key=f"pro_{rid}")
            pro_mode=st.radio("Échantillon pro",["Ses matchs récents","Même match(s)"],horizontal=True,key=f"pro_mode_{rid}")
            pro_count=st.slider("Matchs pro",1,20,5,key=f"pro_count_{rid}")
            pro_side=st.selectbox("Side du pro",["Tous","BLUE","RED"],key=f"pro_side_{rid}")
            if st.button("Charger la comparaison vision",key=f"load_pro_{rid}"):
                if not pro_id.strip():
                    st.warning("Renseigne un Riot ID pro.")
                else:
                    try:
                        pro_puuid=resolve_id(pro_id.strip())
                        mids=[b["matchId"] for b in filtered] if pro_mode=="Même match(s)" else match_ids(pro_puuid,pro_count,st.session_state.get("queue_v2"))
                        pro_bundles=[]
                        for mid in mids:
                            try:
                                m=match_data(mid)
                                p=participant(m,pro_puuid)
                                if p:
                                    tl=timeline_data(mid)
                                    pro_bundles.append({"events":collect_events(tl,pid_of(m,pro_puuid),mid,side_of(p),p.get("teamPosition","UNKNOWN"))})
                            except Exception:
                                continue
                        st.session_state[f"pro_data_{rid}"]=(pro_id,pro_bundles)
                    except Exception as exc:
                        st.error(f"Impossible de charger le pro : {exc}")
            pro_cached=st.session_state.get(f"pro_data_{rid}")
            if pro_cached:
                pro_label,pro_bundles=pro_cached
                pro_ev=event_frame(pro_bundles)
                pro_ward=pro_ev[pro_ev.category=="Ward"].copy()
                pro_ward["wardGroup"]=pro_ward["wardType"].map(ward_kind)
                if pro_side!="Tous":
                    pro_ward=pro_ward[pro_ward.side==pro_side]
                st.caption(f"{pro_label} · {len(pro_bundles)} matchs trouvés · {len(pro_ward)} poses détectées")
                cols=st.columns(2)
                for idx,(start,end) in enumerate(WINDOWS):
                    part=pro_ward[(pro_ward.minute>=start)&(pro_ward.minute<end)]
                    with cols[idx%2]:
                        st.plotly_chart(event_map(part,f"{pro_label} · {start}–{end} min",group="wardGroup"),use_container_width=True)
        with kills:
            k_events=events[events.category=="Kill"]
            cols=st.columns(2)
            for idx,(start,end) in enumerate(WINDOWS):
                with cols[idx%2]:
                    part=k_events[(k_events.minute>=start)&(k_events.minute<end)]
                    st.plotly_chart(event_map(part,f"Kills {start}–{end} min"),use_container_width=True)
        with compare:
            st.markdown("#### Comparaison de courbes contre un autre joueur EUW")
            other_id=st.text_input("Riot ID comparé",key=f"other_{rid}")
            other_count=st.slider("Nombre de matchs comparés",1,20,5,key=f"other_count_{rid}")
            if st.button("Charger le joueur comparé",key=f"load_other_{rid}"):
                try:
                    comp_bundles,comp_errors=load_player(other_id.strip(),other_count,st.session_state.get("queue_v2"),[])
                    st.session_state[f"other_data_{rid}"]=(other_id,comp_bundles,comp_errors)
                except Exception as exc:
                    st.error(str(exc))
            comp_cached=st.session_state.get(f"other_data_{rid}")
            if comp_cached:
                other_label,comp_bundles,comp_errors=comp_cached
                if comp_errors:
                    st.caption(f"{len(comp_errors)} matchs non chargés")
                comp_series=mean_series(comp_bundles)
                if comp_series.empty:
                    st.warning("Pas de données comparables.")
                else:
                    cols=st.columns(2)
                    for idx,(metric,title) in enumerate([("gold","Gold"),("xp","XP"),("cs","CS"),("damage","Dégâts")]):
                        with cols[idx%2]:
                            st.plotly_chart(line_chart(series,metric,title,comp_series,other_label),use_container_width=True)
        with teams:
            st.markdown("#### Team A vs Team B · analyse des matchs communs")
            team_a=st.text_area("Team A (5 Riot IDs)",key=f"team_a_{rid}",height=100)
            team_b=st.text_area("Team B (5 Riot IDs)",key=f"team_b_{rid}",height=100)
            st.caption("Compare les deux équipes sur les matchs déjà chargés pour ce joueur. Pour des scrims, charge les Match IDs dans le mode scrim.")
            if st.button("Comparer les équipes",key=f"team_btn_{rid}"):
                try:
                    set_a={resolve_id(x) for x in parse_lines(team_a)}
                    set_b={resolve_id(x) for x in parse_lines(team_b)}
                    if not set_a or not set_b:
                        st.warning("Renseigne les deux équipes.")
                    else:
                        rows=[]
                        for b in filtered:
                            ps=b["match"]["info"]["participants"]
                            aa=[p for p in ps if p.get("puuid") in set_a]
                            bb=[p for p in ps if p.get("puuid") in set_b]
                            if len(aa)<3 or len(bb)<3:
                                continue
                            for name,pp in (("Team A",aa),("Team B",bb)):
                                rows.append({"Match":b["matchId"],"Équipe":name,"Joueurs présents":len(pp),
                                             "Gold total fin":sum(p.get("goldEarned",0) for p in pp),
                                             "Kills":sum(p.get("kills",0) for p in pp),
                                             "Morts":sum(p.get("deaths",0) for p in pp),
                                             "Win":bool(pp[0].get("win"))})
                        if rows:
                            df_team=pd.DataFrame(rows)
                            st.dataframe(df_team,hide_index=True,use_container_width=True)
                            st.dataframe(df_team.groupby("Équipe")[["Gold total fin","Kills","Morts","Win"]].mean().round(2),use_container_width=True)
                        else:
                            st.info("Aucun match commun avec au moins 3 joueurs de chaque équipe dans l'échantillon.")
                except Exception as exc:
                    st.error(str(exc))
        with export:
            st.markdown("#### Télécharger les données du périmètre actif")
            st.download_button("Rapport PDF",data=export_pdf(rid,filtered),file_name="hidden_coach_report.pdf",mime="application/pdf",key=f"pdf_{rid}")
            st.download_button("Courbes CSV",data=series.to_csv(index=False).encode("utf-8-sig"),file_name="courbes.csv",mime="text/csv",key=f"csv_series_{rid}")
            st.download_button("Événements CSV",data=events.to_csv(index=False).encode("utf-8-sig"),file_name="events.csv",mime="text/csv",key=f"csv_events_{rid}")
