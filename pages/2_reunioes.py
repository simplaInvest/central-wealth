# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
import plotly.express as px
import plotly.graph_objects as go
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, date, timedelta
import calendar

# ------------------------------------------------------------
# CONFIGURAÇÕES DA PÁGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="Reuniões - Expert Comercial Wealth",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# AUTENTICAÇÃO (mantém seu comportamento original)
# ------------------------------------------------------------
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.error("Você precisa estar logado para acessar esta página.")
    st.stop()

# ------------------------------------------------------------
# CABEÇALHO
# ------------------------------------------------------------
st.title("Reuniões")
st.caption("Dashboard dinâmico integrando as abas **CONSULTOR** e **SDR** (Google Sheets).")

# ------------------------------------------------------------
# CONFIGURAÇÕES DE PLANILHA
# ------------------------------------------------------------
SHEET_URL_DEFAULT = (
    "https://docs.google.com/spreadsheets/d/1aRjUMdsk9kasJgsQYBMi4LeawC9iQbXe7L_BANqjPjA/edit?usp=sharing"
)
TAB_CONSULTOR = "CONSULTOR"
TAB_SDR = "SDR"

# ------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------------------
def _get_gspread_client():
    sa_info = dict(st.secrets["gcp_service_account"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)


def _slugify(s: str) -> str:
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def find_column_by_patterns(df: pd.DataFrame, patterns: list[str]) -> str | None:
    """Retorna o primeiro nome de coluna cujo slug contém qualquer um dos padrões."""
    if df is None or df.empty:
        return None
    pats = [_slugify(p) for p in patterns]
    for col in df.columns:
        cslug = _slugify(col)
        if any(p in cslug for p in pats):
            return col
    return None


def get_role_column(df: pd.DataFrame, canonical: str) -> str | None:
    """Detecta automaticamente o nome da coluna que representa SDR ou Consultor."""
    if canonical in df.columns:
        return canonical
    patterns = (
        ["sdr", "responsavel", "responsável", "criador", "owner", "agendador", "pré-vendas", "pre_vendas"]
        if canonical == "sdr"
        else ["consultor", "closer", "responsavel_consultor", "responsável consultor"]
    )
    return find_column_by_patterns(df, patterns)


@st.cache_data(ttl=300, show_spinner=False)
def load_worksheet(url: str, tab_name: str) -> pd.DataFrame:
    gc = _get_gspread_client()
    sh = gc.open_by_url(url)
    ws = sh.worksheet(tab_name)
    df = pd.DataFrame(ws.get_all_records())
    df.columns = [_slugify(c) for c in df.columns]
    for col in df.columns:
        if "data" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def ensure_session_data():
    """Carrega as planilhas CONSULTOR e SDR automaticamente e guarda no SessionState."""
    if "reunioes_consultor" in st.session_state and "reunioes_sdr" in st.session_state:
        return
    try:
        with st.spinner("🔄 Carregando planilhas do Google Sheets..."):
            df_consultor = load_worksheet(SHEET_URL_DEFAULT, TAB_CONSULTOR)
            df_sdr = load_worksheet(SHEET_URL_DEFAULT, TAB_SDR)
        st.session_state["reunioes_consultor"] = df_consultor
        st.session_state["reunioes_sdr"] = df_sdr
        st.success("✅ Planilhas carregadas com sucesso.")
    except Exception as e:
        st.error(f"Erro ao carregar planilhas: {e}")
        st.stop()


def label_status(status_raw: str) -> str:
    if not isinstance(status_raw, str):
        return "indefinido"
    s = _slugify(status_raw)
    if "executad" in s or "realizad" in s:
        return "executada"
    if "no_show" in s or "nao_compareceu" in s or "não_compareceu" in s:
        return "no show"
    if "qualificad" in s:
        return "qualificada"
    if "nao_qualificad" in s or "não_qualificad" in s:
        return "não qualificada"
    if "remarcad" in s or "reagendad" in s:
        return "reagendada"
    return status_raw


def label_contrato(status_raw: str) -> str:
    if not isinstance(status_raw, str):
        return "sem status"
    s = _slugify(status_raw)
    if "assinado" in s or "fechado" in s or "ganho" in s:
        return "assinado"
    if "enviado" in s or "proposta" in s:
        return "enviado"
    return "sem status"

# ------------------------------------------------------------
# CARREGAMENTO DE DADOS
# ------------------------------------------------------------
ensure_session_data()
df_consultor = st.session_state["reunioes_consultor"].copy()
df_sdr = st.session_state["reunioes_sdr"].copy()

# Normalizações básicas
if "status_reuniao" in df_consultor.columns:
    df_consultor["status_reuniao_norm"] = df_consultor["status_reuniao"].apply(label_status)
elif any("status" in c for c in df_consultor.columns):
    col = [c for c in df_consultor.columns if "status" in c][0]
    df_consultor["status_reuniao_norm"] = df_consultor[col].apply(label_status)
else:
    df_consultor["status_reuniao_norm"] = "indefinido"

if "contrato_status" in df_consultor.columns:
    df_consultor["contrato_status_norm"] = df_consultor["contrato_status"].apply(label_contrato)
else:
    df_consultor["contrato_status_norm"] = "sem status"

# ------------------------------------------------------------
# FILTROS (TOPO DA PÁGINA)
# ------------------------------------------------------------
with st.expander("🔎 Filtros", expanded=False):
    box = st.container()
    with box:
        col1, col2, col3, col4, col5 = st.columns([2, 1.2, 2, 2, 2])

        with col1:
            filtro_por = st.radio("Filtrar por:", ["Data da reunião", "Data do agendamento"], horizontal=True)

        # Seleciona coluna de data por planilha
        if "agendamento" in filtro_por.lower():
            sdr_date_candidate = "data_agendamento"
        else:
            sdr_date_candidate = "data_reuniao"
        sdr_date_col = sdr_date_candidate if sdr_date_candidate in df_sdr.columns else next((c for c in df_sdr.columns if "data" in c), None)
        cons_date_col = "data_reuniao" if "data_reuniao" in df_consultor.columns else next((c for c in df_consultor.columns if "data" in c), None)

        # Define o intervalo padrão com base nas colunas encontradas
        datas = []
        if sdr_date_col:
            datas.extend(df_sdr[sdr_date_col].dropna().tolist())
        if cons_date_col:
            datas.extend(df_consultor[cons_date_col].dropna().tolist())
        if datas:
            dmin, dmax = pd.to_datetime(min(datas)).date(), pd.to_datetime(max(datas)).date()
        else:
            dmin, dmax = date.today() - timedelta(days=30), date.today()

        with col2:
            presets = {
                "Hoje": (date.today(), date.today()),
                "Ontem": (date.today() - timedelta(days=1), date.today() - timedelta(days=1)),
                "Últimos 7 dias": (date.today() - timedelta(days=6), date.today()),
                "Mês atual": (date.today().replace(day=1), date.today()),
                "Personalizado": (dmin, dmax),
            }
            preset = st.selectbox("Período", list(presets.keys()), index=2)

        # Intervalo: mostra o seletor quando "Personalizado", senão exibe o intervalo atual
        drange = presets[preset] if preset != "Personalizado" else None
        with col3:
            if preset == "Personalizado":
                drange = st.date_input("Intervalo", value=(dmin, dmax))
            else:
                st.write(f"{presets[preset][0].strftime('%d/%m/%Y')} – {presets[preset][1].strftime('%d/%m/%Y')}")
        data_ini, data_fim = drange if drange else presets[preset]

        # Seletores de pessoas
        sdr_col_sidebar = get_role_column(df_sdr, "sdr")
        consultor_col_sidebar = get_role_column(df_consultor, "consultor")
        sdrs = sorted(df_sdr[sdr_col_sidebar].dropna().unique().tolist()) if sdr_col_sidebar else []
        consultores = sorted(df_consultor[consultor_col_sidebar].dropna().unique().tolist()) if consultor_col_sidebar else []

        with col4:
            f_sdr = st.multiselect("SDR", sdrs)
        with col5:
            f_consultor = st.multiselect("Consultor", consultores)

# Período anterior para deltas
def _periodo_anterior(ini: date, fim: date, preset_nome: str) -> tuple[date, date]:
    dias = (fim - ini).days + 1
    if preset_nome == "Mês atual":
        # mês anterior completo
        prev_ref = ini.replace(day=1) - timedelta(days=1)
        prev_start = prev_ref.replace(day=1)
        last_day = calendar.monthrange(prev_ref.year, prev_ref.month)[1]
        prev_end = prev_ref.replace(day=last_day)
        return prev_start, prev_end
    # período imediatamente anterior com o mesmo tamanho
    prev_end = ini - timedelta(days=1)
    prev_start = prev_end - timedelta(days=dias - 1)
    return prev_start, prev_end

prev_ini, prev_fim = _periodo_anterior(data_ini, data_fim, preset)

# ------------------------------------------------------------
# FILTROS APLICADOS
# ------------------------------------------------------------
def apply_filters(df: pd.DataFrame, date_col: str | None, start: date | None = None, end: date | None = None):
    rng_ini = start or data_ini
    rng_fim = end or data_fim
    if date_col and date_col in df.columns:
        df = df[df[date_col].dt.date.between(rng_ini, rng_fim)]
    if f_sdr and sdr_col_sidebar in df.columns:
        df = df[df[sdr_col_sidebar].isin(f_sdr)]
    if f_consultor and consultor_col_sidebar in df.columns:
        df = df[df[consultor_col_sidebar].isin(f_consultor)]
    return df

sdr_f = apply_filters(df_sdr.copy(), sdr_date_col)
cons_f = apply_filters(df_consultor.copy(), cons_date_col)

# Período anterior
sdr_prev = apply_filters(df_sdr.copy(), sdr_date_col, prev_ini, prev_fim)
cons_prev = apply_filters(df_consultor.copy(), cons_date_col, prev_ini, prev_fim)

# ------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------
total_agendadas = len(sdr_f)
executadas_n_qual = (cons_f["status_da_reuniao"] == "Executada não Qualificada").sum()
executadas_qual = (cons_f["status_da_reuniao"] == "Executada Qualificada").sum()
no_show = (cons_f["status_da_reuniao"] == "No-Show").sum()
remarcadas = (cons_f["status_da_reuniao"] == "Remarcada").sum()
enviados = (cons_f["contrato"] == "Contrato Enviado (Em negociação)").sum()
assinados = (cons_f["contrato"] == "Contrato Assinado").sum()
futuro = (cons_f["contrato"] == "Futuro").sum()
perdeu = (cons_f["status_reuniao_norm"] == "Perdeu").sum()
executadas = executadas_qual + executadas_n_qual

# Valores do período anterior para cálculo de deltas nos cards
prev_total_agendadas = len(sdr_prev)
prev_executadas_n_qual = (cons_prev["status_da_reuniao"] == "Executada não Qualificada").sum() if "status_da_reuniao" in cons_prev.columns else 0
prev_executadas_qual = (cons_prev["status_da_reuniao"] == "Executada Qualificada").sum() if "status_da_reuniao" in cons_prev.columns else 0
prev_no_show = (cons_prev["status_da_reuniao"] == "No-Show").sum() if "status_da_reuniao" in cons_prev.columns else 0
prev_remarcadas = (cons_prev["status_da_reuniao"] == "Remarcada").sum() if "status_da_reuniao" in cons_prev.columns else 0
prev_enviados = (cons_prev["contrato"] == "Contrato Enviado (Em negociação)").sum() if "contrato" in cons_prev.columns else 0
prev_assinados = (cons_prev["contrato"] == "Contrato Assinado").sum() if "contrato" in cons_prev.columns else 0
prev_futuro = (cons_prev["contrato"] == "Futuro").sum() if "contrato" in cons_prev.columns else 0
prev_perdeu = (cons_prev["status_reuniao_norm"] == "Perdeu").sum() if "status_reuniao_norm" in cons_prev.columns else 0

# Deltas (atual - anterior)
delta_agendadas = int(total_agendadas) - int(prev_total_agendadas)
delta_executadas_qual = int(executadas_qual) - int(prev_executadas_qual)
delta_executadas_n_qual = int(executadas_n_qual) - int(prev_executadas_n_qual)
delta_remarcadas = int(remarcadas) - int(prev_remarcadas)
delta_no_show = int(no_show) - int(prev_no_show)
delta_enviados = int(enviados) - int(prev_enviados)
delta_assinados = int(assinados) - int(prev_assinados)
delta_futuro = int(futuro) - int(prev_futuro)
delta_perdeu = int(perdeu) - int(prev_perdeu)

# ------------------------------------------------------------
# MÉTRICAS (DOIS GRUPOS NA MESMA LINHA, SEM DELTAS)
# ------------------------------------------------------------
row = st.columns([5, 3])
with row[0]:
    st.caption("Reuniões")
    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    rc1.metric("🗓️ Agendadas", int(total_agendadas), delta=int(delta_agendadas))
    rc2.metric("Executadas Qualificadas", int(executadas_qual), delta=int(delta_executadas_qual))
    rc3.metric("Executadas Não Qualificadas", int(executadas_n_qual), delta=int(delta_executadas_n_qual))
    rc4.metric("Remarcadas", int(remarcadas), delta=int(delta_remarcadas))
    rc5.metric("No-Show", int(no_show), delta=int(delta_no_show))
with row[1]:
    st.caption("Contratos")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Perdeu", int(perdeu), delta=int(delta_perdeu))
    cc2.metric("🔜 Futuras", int(futuro), delta=int(delta_futuro))
    cc3.metric("📄 Contratos (Env/Ass)", f"{int(enviados)}/{int(assinados)}", delta=f"{delta_enviados:+d}/{delta_assinados:+d}")
st.markdown("---")

# ------------------------------------------------------------
# FUNIL DE CONVERSÃO
# ------------------------------------------------------------
st.subheader("Funil de Conversão")
col_funil, col_conv = st.columns([3, 2])

# Gráfico de funil
with col_funil:
    funil = pd.DataFrame({
        "Etapa": ["Contratos Assinados", "Contratos Enviados", "Qualificadas", "Executadas", "Agendadas"],
        "Qtd": [
            assinados,
            enviados,
            executadas_qual,
            executadas,
            total_agendadas,
        ],
    })
    fig_funil = px.funnel(funil, x="Qtd", y="Etapa", color="Etapa")
    st.plotly_chart(fig_funil, use_container_width=True)

# Coluna de conversões com deltas
with col_conv:
    def _pct(a: int | float, b: int | float) -> float:
        return round((a / b * 100), 1) if b else 0.0

    # Período atual
    conv_exec = _pct(executadas, total_agendadas)
    conv_qual = _pct(executadas_qual, executadas)
    conv_env = _pct(enviados, executadas_qual)
    conv_ass = _pct(assinados, enviados)

    # Período anterior
    prev_total_agendadas = len(sdr_prev)
    prev_executadas_n_qual = (cons_prev["status_da_reuniao"] == "Executada não Qualificada").sum() if "status_da_reuniao" in cons_prev.columns else 0
    prev_executadas_qual = (cons_prev["status_da_reuniao"] == "Executada Qualificada").sum() if "status_da_reuniao" in cons_prev.columns else 0
    prev_executadas = int(prev_executadas_qual) + int(prev_executadas_n_qual)
    prev_enviados = (cons_prev["contrato"] == "Contrato Enviado (Em negociação)").sum() if "contrato" in cons_prev.columns else 0
    prev_assinados = (cons_prev["contrato"] == "Contrato Assinado").sum() if "contrato" in cons_prev.columns else 0

    conv_exec_prev = _pct(prev_executadas, prev_total_agendadas)
    conv_qual_prev = _pct(prev_executadas_qual, prev_executadas)
    conv_env_prev = _pct(prev_enviados, prev_executadas_qual)
    conv_ass_prev = _pct(prev_assinados, prev_enviados)

    st.caption("Conversões do período (Δ vs anterior)")
    m1, m2 = st.columns(2)
    m1.metric("Execução", f"{conv_exec:.1f}%", delta=f"{(conv_exec - conv_exec_prev):+.1f} pp")
    m2.metric("Qualificação", f"{conv_qual:.1f}%", delta=f"{(conv_qual - conv_qual_prev):+.1f} pp")
    m3, m4 = st.columns(2)
    m3.metric("Envio de Contrato", f"{conv_env:.1f}%", delta=f"{(conv_env - conv_env_prev):+.1f} pp")
    m4.metric("Fechamento", f"{conv_ass:.1f}%", delta=f"{(conv_ass - conv_ass_prev):+.1f} pp")

# ------------------------------------------------------------
# EVOLUÇÃO DIÁRIA
# ------------------------------------------------------------
st.subheader("Evolução Diária")

# Seletor de variável (mesmas opções dos cards)
opcoes_evolucao = [
    "🗓️ Agendadas",
    "Executadas Qualificadas",
    "Executadas Não Qualificadas",
    "Remarcadas",
    "No-Show",
    "Perdeu",
    "🔜 Futuras",
    "📄 Contratos (Env/Ass)",
]
variavel = st.selectbox("Variável", opcoes_evolucao, index=0)

def _by_day(df: pd.DataFrame, date_col: str | None, mask: pd.Series | None, name: str) -> pd.Series:
    if df is None or df.empty or not date_col or date_col not in df.columns:
        return pd.Series(dtype=int, name=name)
    if mask is not None:
        df = df[mask]
    d = df[date_col].dropna().dt.floor("D")
    return d.groupby(d).size().rename(name)

# Índice diário baseado no filtro de período
idx = pd.date_range(data_ini, data_fim, freq="D")
ts = pd.DataFrame({"Data": idx})

# Construção da série de acordo com a variável escolhida
if variavel == "🗓️ Agendadas":
    serie = _by_day(sdr_f, sdr_date_col, None, "Agendadas")
    ts = ts.merge(serie.rename_axis("Data").reset_index(), on="Data", how="left")
elif variavel == "Executadas Qualificadas":
    mask = (cons_f["status_da_reuniao"] == "Executada Qualificada") if "status_da_reuniao" in cons_f.columns else None
    serie = _by_day(cons_f, cons_date_col, mask, "Executadas Qualificadas")
    ts = ts.merge(serie.rename_axis("Data").reset_index(), on="Data", how="left")
elif variavel == "Executadas Não Qualificadas":
    mask = (cons_f["status_da_reuniao"] == "Executada não Qualificada") if "status_da_reuniao" in cons_f.columns else None
    serie = _by_day(cons_f, cons_date_col, mask, "Executadas Não Qualificadas")
    ts = ts.merge(serie.rename_axis("Data").reset_index(), on="Data", how="left")
elif variavel == "Remarcadas":
    mask = (cons_f["status_da_reuniao"] == "Remarcada") if "status_da_reuniao" in cons_f.columns else None
    serie = _by_day(cons_f, cons_date_col, mask, "Remarcadas")
    ts = ts.merge(serie.rename_axis("Data").reset_index(), on="Data", how="left")
elif variavel == "No-Show":
    mask = (cons_f["status_da_reuniao"] == "No-Show") if "status_da_reuniao" in cons_f.columns else None
    serie = _by_day(cons_f, cons_date_col, mask, "No-Show")
    ts = ts.merge(serie.rename_axis("Data").reset_index(), on="Data", how="left")
elif variavel == "Perdeu":
    mask = (cons_f["status_reuniao_norm"] == "Perdeu") if "status_reuniao_norm" in cons_f.columns else None
    serie = _by_day(cons_f, cons_date_col, mask, "Perdeu")
    ts = ts.merge(serie.rename_axis("Data").reset_index(), on="Data", how="left")
elif variavel == "🔜 Futuras":
    mask = (cons_f["contrato"] == "Futuro") if "contrato" in cons_f.columns else None
    serie = _by_day(cons_f, cons_date_col, mask, "Futuras")
    ts = ts.merge(serie.rename_axis("Data").reset_index(), on="Data", how="left")
elif variavel == "📄 Contratos (Env/Ass)":
    mask_env = (cons_f["contrato"] == "Contrato Enviado (Em negociação)") if "contrato" in cons_f.columns else None
    mask_ass = (cons_f["contrato"] == "Contrato Assinado") if "contrato" in cons_f.columns else None
    serie_env = _by_day(cons_f, cons_date_col, mask_env, "Contratos Enviados")
    serie_ass = _by_day(cons_f, cons_date_col, mask_ass, "Contratos Assinados")
    ts = ts.merge(serie_env.rename_axis("Data").reset_index(), on="Data", how="left")
    ts = ts.merge(serie_ass.rename_axis("Data").reset_index(), on="Data", how="left")

# Ajustes finais: preencher zeros e plotar
for c in ts.columns:
    if c != "Data":
        ts[c] = ts[c].fillna(0).astype(int)

ycols = [c for c in ts.columns if c != "Data"]
if ycols:
    fig = px.line(ts, x="Data", y=ycols, markers=True)
    # Anotar valores sobre os pontos
    for i, yname in enumerate(ycols):
        fig.data[i].text = ts[yname]
        fig.data[i].textposition = "top center"
    fig.update_traces(mode="lines+markers+text")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem séries para exibir no período selecionado.")
st.markdown("---")

# ------------------------------------------------------------
# STATUS DA REUNIÃO (Pizza)
# ------------------------------------------------------------
st.subheader("Distribuição por Status da Reunião")
if "status_da_reuniao" in cons_f:
    dist = cons_f["status_da_reuniao"].value_counts().reset_index()
    dist.columns = ["Status", "Qtd"]
    fig_pie = px.pie(dist, names="Status", values="Qtd", hole=0.4)
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

# ------------------------------------------------------------
# PERFORMANCE POR SDR / CONSULTOR
# ------------------------------------------------------------
st.subheader("Performance por SDR e Consultor")
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Por SDR**")
    sdr_col = get_role_column(sdr_f, "sdr")
    if sdr_col:
        ag = sdr_f.groupby(sdr_col).size().rename("Agendadas")
        ex = cons_f.groupby(get_role_column(cons_f, "sdr")).size().rename("Executadas") if get_role_column(cons_f, "sdr") else pd.Series(dtype=int)
        perf = pd.concat([ag, ex], axis=1).fillna(0).reset_index().rename(columns={sdr_col: "SDR"})
        # ordenar por Agendadas em ordem decrescente para ranking
        perf = perf.sort_values(by="Agendadas", ascending=False)
        perf["Taxa Comparecimento (%)"] = np.where(perf["Agendadas"]>0, (perf["Executadas"]/perf["Agendadas"]*100), 0).round(1)
        fig_sdr = px.bar(
            perf,
            x="Agendadas",
            y="SDR",
            orientation="h",
            hover_data=["Executadas","Taxa Comparecimento (%)"],
            text="Agendadas",
        )
        fig_sdr.update_traces(textposition="outside")
        fig_sdr.update_layout(yaxis={
            "categoryorder": "array",
            "categoryarray": perf["SDR"].tolist(),
            "autorange": "reversed",  # maior no topo
        })
        st.plotly_chart(fig_sdr, use_container_width=True)
        st.dataframe(perf, use_container_width=True, hide_index=True)
    else:
        st.info("Coluna SDR não encontrada na planilha.")

with c2:
    st.markdown("**Por Consultor**")
    cons_col = get_role_column(cons_f, "consultor")
    if cons_col:
        grp = cons_f.groupby(cons_col).agg(
            Reunioes=("status_da_reuniao", "count"),
            Executadas=("status_da_reuniao", lambda x: x.isin(["Executada Qualificada", "Executada não Qualificada"]).sum()),
            Qualificadas=("status_da_reuniao", lambda x: (x=="Executada Qualificada").sum()),
            Contratos_Assinados=("contrato", lambda x: (x=="Contrato Assinado").sum()),
        ).reset_index()
        # ordenar por Contratos Assinados em ordem decrescente para ranking
        grp = grp.sort_values(by="Contratos_Assinados", ascending=False)
        grp["Tx Exec (%)"] = np.where(grp["Reunioes"]>0, grp["Executadas"]/grp["Reunioes"]*100, 0).round(1)
        grp["Tx Qualif (%)"] = np.where(grp["Executadas"]>0, grp["Qualificadas"]/grp["Executadas"]*100, 0).round(1)
        grp["Tx Fech (%)"] = np.where(grp["Qualificadas"]>0, grp["Contratos_Assinados"]/grp["Qualificadas"]*100, 0).round(1)
        fig_cons = px.bar(
            grp,
            x="Contratos_Assinados",
            y=cons_col,
            orientation="h",
            hover_data=["Reunioes","Tx Exec (%)","Tx Fech (%)"],
            text="Contratos_Assinados",
        )
        fig_cons.update_traces(textposition="outside")
        fig_cons.update_layout(yaxis={
            "categoryorder": "array",
            "categoryarray": grp[cons_col].tolist(),
            "autorange": "reversed",  # maior no topo
        })
        st.plotly_chart(fig_cons, use_container_width=True)
        st.dataframe(grp, use_container_width=True, hide_index=True)
    else:
        st.info("Coluna Consultor não encontrada na planilha.")

st.markdown("---")

# ------------------------------------------------------------
# PIPELINE DE CONTRATOS
# ------------------------------------------------------------
st.subheader("Pipeline de Contratos")
pipe = cons_f.copy()
if not pipe.empty:
    ct = pipe["contrato"].value_counts().reset_index()
    ct.columns = ["Status do Contrato", "Qtd"]
    fig_ct = px.bar(ct, x="Status do Contrato", y="Qtd", color="Status do Contrato", text="Qtd")
    fig_ct.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_ct, use_container_width=True)
else:
    st.info("Sem dados de contratos.")

# ------------------------------------------------------------
# RODAPÉ
# ------------------------------------------------------------
st.caption("🕒 Todos os números respeitam o **Filtro de Data** (reunião ou agendamento). Use o preset **Hoje** para ver apenas os dados do dia.")
