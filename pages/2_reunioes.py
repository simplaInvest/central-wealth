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
import textwrap

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
# ESTILO GLOBAL (CSS)
# ------------------------------------------------------------
try:
    with open("assets/styles.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ------------------------------------------------------------
# AUTENTICAÇÃO (mantém seu comportamento original)
# ------------------------------------------------------------
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.error("Você precisa estar logado para acessar esta página.")
    st.stop()

try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
except Exception:
    _st_autorefresh = None
if _st_autorefresh:
    _st_autorefresh(interval=900000, key="reunioes_refresh")

# ------------------------------------------------------------
# CABEÇALHO
# ------------------------------------------------------------
st.title("🗓️ Reuniões")
st.caption("Dashboard dinâmico integrando as abas **CONSULTOR** e **SDR** (Google Sheets).")

# ------------------------------------------------------------
# SIDEBAR: Navegação
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("**Navegação**")
    st.page_link("pages/1_central.py", label="📊 Central")
    st.page_link("pages/2_reunioes.py", label="🗓️ Reuniões")
    st.markdown("---")
    last_up = st.session_state.get("sheets_last_update_at")
    if last_up:
        st.caption(f"Última atualização das planilhas: {last_up.strftime('%d/%m/%Y %H:%M:%S')}")
    

# ------------------------------------------------------------
# CONFIGURAÇÕES DE PLANILHA
# ------------------------------------------------------------
SHEET_URL_DEFAULT = (
    "https://docs.google.com/spreadsheets/d/1aRjUMdsk9kasJgsQYBMi4LeawC9iQbXe7L_BANqjPjA/edit?usp=sharing"
)
TAB_CONSULTOR = "CONSULTOR"
TAB_SDR = "SDR"
TAB_EQUIPE = "EQUIPE"

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
    if df is None or df.empty:
        return None
    can = _slugify(canonical)
    cols = list(df.columns)
    # mapeamento direto por nomes de colunas fornecidos
    def _resolve(colname: str) -> str | None:
        if colname in cols:
            return colname
        slug = _slugify(colname)
        if slug in cols:
            return slug
        return None
    if can == "sdr":
        col = _resolve("nome_do_sdr")
        if col:
            return col
    else:  # consultor
        # df_consultor
        col = _resolve("nome_do_consultor")
        if col:
            return col
        # df_sdr
        col = _resolve("nome_do_consultor_closer")
        if col:
            return col
    if can in cols:
        return can
    def score(cslug: str, role: str) -> int:
        s = 0
        if role == "sdr":
            primary = ["sdr","agendador","pre_vendas","pre-vendas","pré-vendas","responsavel_sdr","sdr_responsavel"]
            fallback = ["responsavel","owner","criador","agendamento","agendamentos","pre_venda"]
        else:
            primary = ["consultor","closer","responsavel_consultor","consultor_responsavel"]
            fallback = ["responsavel","owner","dono_conta"]
        for p in primary:
            if p in cslug:
                s += 3
        for p in fallback:
            if p in cslug:
                s += 1
        return s
    best_col = None
    best_score = 0
    for col in cols:
        cslug = _slugify(col)
        sc = score(cslug, can)
        if sc > best_score:
            best_score = sc
            best_col = col
    return best_col

@st.cache_data(ttl=900, show_spinner=False)
def load_worksheet(url: str, tab_name: str) -> pd.DataFrame:
    gc = _get_gspread_client()
    sh = gc.open_by_url(url)
    ws = sh.worksheet(tab_name)
    df = pd.DataFrame(ws.get_all_records())
    df.columns = [_slugify(c) for c in df.columns]
    for col in df.columns:
        if "data" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df.attrs["loaded_at"] = datetime.now()
    return df


def ensure_session_data():
    try:
        with st.spinner("🔄 Carregando planilhas do Google Sheets..."):
            df_consultor = load_worksheet(SHEET_URL_DEFAULT, TAB_CONSULTOR)
            df_sdr = load_worksheet(SHEET_URL_DEFAULT, TAB_SDR)
            df_equipe = load_worksheet(SHEET_URL_DEFAULT, TAB_EQUIPE)
            st.session_state["reunioes_consultor"] = df_consultor
            st.session_state["reunioes_sdr"] = df_sdr
            st.session_state["equipe"] = df_equipe
            ts_cons = df_consultor.attrs.get("loaded_at")
            ts_sdr = df_sdr.attrs.get("loaded_at")
            ts_equipe = df_equipe.attrs.get("loaded_at")
            st.session_state["sheets_last_update_at"] = max(ts_cons, ts_sdr, ts_equipe) if ts_cons and ts_sdr and ts_equipe else (ts_cons or ts_sdr or ts_equipe or datetime.now())

        if "reunioes_consultor_raw" not in st.session_state or "reunioes_sdr_raw" not in st.session_state or "equipe_raw" not in st.session_state:
            gc = _get_gspread_client()
            sh = gc.open_by_url(SHEET_URL_DEFAULT)
            ws_cons = sh.worksheet(TAB_CONSULTOR)
            ws_sdr = sh.worksheet(TAB_SDR)
            ws_equipe = sh.worksheet(TAB_EQUIPE)
            df_cons_raw = pd.DataFrame(ws_cons.get_all_records())
            df_sdr_raw = pd.DataFrame(ws_sdr.get_all_records())
            df_equipe_raw = pd.DataFrame(ws_equipe.get_all_records())
            st.session_state["reunioes_consultor_raw"] = df_cons_raw
            st.session_state["reunioes_sdr_raw"] = df_sdr_raw
            st.session_state["equipe_raw"] = df_equipe_raw
        st.success("✅ Planilhas carregadas com sucesso.")
    except Exception as e:
        st.error(f"Erro ao carregar planilhas: {e}")
        st.stop()

# ------------------------------------------------------------
# Sidebar utilidades: limpar cache e recarregar planilhas
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("**Utilidades**")
    if st.button("🧹 Limpar cache e recarregar", use_container_width=True, help="Apaga o cache e recarrega as planilhas do Google Sheets"):
        # Limpa caches (dados/recursos) e remove dataframes do Session State
        try:
            st.cache_data.clear()
        except Exception as e:
            st.warning(f"Falha ao limpar cache de dados: {e}")
        try:
            # Pode não existir em algumas versões; proteger com try
            st.cache_resource.clear()
        except Exception:
            pass
        for k in ["reunioes_consultor", "reunioes_sdr", "reunioes_consultor_raw", "reunioes_sdr_raw"]:
            if k in st.session_state:
                st.session_state.pop(k, None)
        st.success("Cache limpo. Recarregando dados…")
        # Reinicia a execução para que o carregamento ocorra no início do script
        try:
            # Streamlit >= 1.27
            st.rerun()
        except Exception:
            # Compatibilidade com versões que ainda possuam experimental_rerun
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
            else:
                st.warning("Não foi possível forçar o reload automático. Recarregue a página manualmente (Ctrl+R).")


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


def style_fig(fig: go.Figure) -> go.Figure:
    """Aplica tema escuro e identidade visual (dourado/branco/preto) aos gráficos."""
    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#FAFAFA"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#333333", zeroline=False, color="#FAFAFA")
    fig.update_yaxes(showgrid=True, gridcolor="#333333", zeroline=False, color="#FAFAFA")
    return fig

def _by_day(df: pd.DataFrame, date_col: str | None, mask: pd.Series | None, name: str) -> pd.Series:
    """Agrupa por dia a partir de `date_col`, aplicando `mask` opcional e retorna a série nomeada."""
    if df is None or df.empty or not date_col or date_col not in df.columns:
        return pd.Series(dtype=int, name=name)
    if mask is not None:
        df = df[mask]
    d = df[date_col].dropna().dt.floor("D")
    return d.groupby(d).size().rename(name)

# ------------------------------------------------------------
# CARREGAMENTO DE DADOS
# ------------------------------------------------------------
ensure_session_data()
df_consultor = st.session_state["reunioes_consultor"].copy()
df_sdr = st.session_state["reunioes_sdr"].copy()
df_equipe = st.session_state["equipe"].copy()

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
        sdrs = df_equipe[df_equipe["cargo"] == "SDR"]["nome"].tolist()
        consultores = df_equipe[df_equipe["cargo"] == "Consultor"]["nome"].tolist()

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
# REGRA DE CONTAGEM (SDR): "Tipo de Agendamento" vazio ou "Primeira Reunião"
# ------------------------------------------------------------
tipo_agendamento_col = find_column_by_patterns(df_sdr, ["Tipo de Agendamento", "tipo de agendamento"]) or (
    "tipo_de_agendamento" if "tipo_de_agendamento" in df_sdr.columns else None
)

def _filter_primeira_reuniao_ou_vazio(df: pd.DataFrame, col: str | None) -> pd.DataFrame:
    if not col or col not in df.columns:
        return df
    serie = df[col]
    # vazio: NaN ou string vazia
    vazio = serie.isna() | (serie.astype(str).str.strip() == "")
    # normaliza texto para comparar com/sem acento e case-insensitive
    def _norm(s: str) -> str:
        s = str(s).strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s
    primeira = serie.astype(str).map(_norm) == "primeira reuniao"
    return df[vazio | primeira]

# aplica a regra ao SDR atual e período anterior
sdr_f = _filter_primeira_reuniao_ou_vazio(sdr_f, tipo_agendamento_col)
sdr_prev = _filter_primeira_reuniao_ou_vazio(sdr_prev, tipo_agendamento_col)

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
# ABAS PRINCIPAIS
# ------------------------------------------------------------
tabs = st.tabs(["Funil", "Evolução Diária", "Performance e Rankings", "Pipeline de Contratos"])

# ------------------------------------------------------------
# ABA: FUNIL
# ------------------------------------------------------------
with tabs[0]:
    # KPIs removidos desta aba para evitar duplicação; mantidos apenas acima das abas.

    # Funil de conversão + métricas de conversão
    st.subheader("Funil de Conversão")
    col_funil, col_conv = st.columns([3, 2])
    with col_funil:
        funil = pd.DataFrame({
            "Etapa": ["Contratos Assinados", "Contratos Enviados", "Qualificadas", "Executadas", "Agendadas"],
            "Qtd": [assinados, enviados, executadas_qual, executadas, total_agendadas],
        })
        fig_funil = px.funnel(funil, x="Qtd", y="Etapa", color="Etapa")
        style_fig(fig_funil)
        st.plotly_chart(fig_funil, use_container_width=True)

    with col_conv:
        def _pct(a: int | float, b: int | float) -> float:
            return round((a / b * 100), 1) if b else 0.0

        conv_exec = _pct(executadas, total_agendadas)
        conv_qual = _pct(executadas_qual, executadas)
        conv_env = _pct(enviados, executadas_qual)
        conv_ass = _pct(assinados, enviados)

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

    # Distribuição por Status
    st.subheader("Distribuição por Status da Reunião")
    if "status_da_reuniao" in cons_f:
        dist = cons_f["status_da_reuniao"].value_counts().reset_index()
        dist.columns = ["Status", "Qtd"]
        fig_bar_status = px.bar(dist, x="Qtd", y="Status", orientation="h", text="Qtd", color="Status")
        fig_bar_status.update_traces(textposition="outside", cliponaxis=False)
        style_fig(fig_bar_status)
        st.plotly_chart(fig_bar_status, use_container_width=True)

# ------------------------------------------------------------
# ABA: EVOLUÇÃO DIÁRIA
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("Evolução Diária")
    idx = pd.bdate_range(data_ini, data_fim)
    # Alturas ampliadas para aproveitar mais espaço vertical disponível
    gauge_h = 270
    line_h = 340
    bar_h = 340

    sdr_only_agendamentos = sdr_f[(sdr_f["nome_do_sdr"].notna()) & (sdr_f["nome_do_sdr"] != "Consultor (eu mesmo)")]
    agendadas_sdrs = len(sdr_only_agendamentos)
    sdr_only_feedbacks = cons_f[(cons_f["nome_do_sdr"].notna()) & (cons_f["nome_do_sdr"] != "Consultor (eu mesmo)")]

    cons_only_agendamentos = sdr_f[(sdr_f["nome_do_sdr"].notna()) & (sdr_f["nome_do_sdr"] == "Consultor (eu mesmo)")]
    agendadas_cons = len(cons_only_agendamentos)
    cons_only_feedbacks = cons_f[(cons_f["nome_do_sdr"].notna()) & (cons_f["nome_do_sdr"] == "Consultor (eu mesmo)")]

    filtro_operador = st.radio("Filtrar por:", ["SDRs", "Consultores"], horizontal=True)

    if filtro_operador == "SDRs":
        agendadas_f = agendadas_sdrs
        only_f = sdr_only_agendamentos
        reunioes_f = sdr_only_feedbacks
        executadas_qual_f = (reunioes_f["status_da_reuniao"] == "Executada Qualificada").sum() if "status_da_reuniao" in reunioes_f.columns else 0
    else:
        agendadas_f = agendadas_cons
        only_f = cons_only_agendamentos
        reunioes_f = cons_only_feedbacks
        executadas_qual_f = (reunioes_f["status_da_reuniao"] == "Executada Qualificada").sum() if "status_da_reuniao" in reunioes_f.columns else 0
    
    # Métricas globais para metas
    n_sdrs = df_equipe[df_equipe['cargo'] == 'SDR'].shape[0]
    dias_uteis = len(pd.bdate_range(data_ini, data_fim))
    meta_agendadas_sdrs = 7 * n_sdrs * dias_uteis
    meta_realizadas_sdrs = 4 * n_sdrs * dias_uteis
    meta_assinados_sdrs = 1 * n_sdrs * dias_uteis

    # Linha única: velocímetros com anotações de conversão nos gaps
    cols_top = st.columns([1, 0.28, 1, 0.28, 1])
    with cols_top[0]:
        fig_g1 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=int(agendadas_f),
                title={"text": "Agendadas", "font": {"size": 12}},
                number={"font": {"size": 40}},
                gauge={"axis": {"range": [0, meta_agendadas_sdrs]}, "bar": {"color": "#bfa94c"}},
            )
        )
        fig_g1.update_layout(height=gauge_h, margin=dict(t=10, b=0, l=0, r=0))
        style_fig(fig_g1)
        st.plotly_chart(fig_g1, use_container_width=True)

    with cols_top[2]:
        fig_g2 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=int(executadas_qual_f),
                title={"text": "Executadas Qualificadas", "font": {"size": 12}},
                number={"font": {"size": 40}},
                gauge={"axis": {"range": [0, meta_realizadas_sdrs]}, "bar": {"color": "#bfa94c"}},
            )
        )
        fig_g2.update_layout(height=gauge_h, margin=dict(t=10, b=0, l=0, r=0))
        style_fig(fig_g2)
        st.plotly_chart(fig_g2, use_container_width=True)

    with cols_top[4]:
        assinados_f = (reunioes_f["contrato"] == "Contrato Assinado").sum()
        fig_g3 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=int(assinados_f),
                title={"text": "Contratos Assinados", "font": {"size": 12}},
                number={"font": {"size": 40}},
                gauge={"axis": {"range": [0, meta_assinados_sdrs]}, "bar": {"color": "#bfa94c"}},
            )
        )
        fig_g3.update_layout(height=gauge_h, margin=dict(t=10, b=0, l=0, r=0))
        style_fig(fig_g3)
        st.plotly_chart(fig_g3, use_container_width=True)

    conv_exec = _pct(executadas_qual_f, agendadas_f)
    conv_env = _pct(assinados_f, executadas_qual_f)
    with cols_top[1]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Execução</div>
                <div class='metric-value'>{conv_exec:.1f}%</div>
                <div class='metric-sub'>Ag → Qualif</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols_top[3]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Assinado</div>
                <div class='metric-value'>{conv_env:.1f}%</div>
                <div class='metric-sub'>Qualif → Assinados</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Linha 3: gráficos (3 colunas)
    c1, c2, c3 = st.columns(3)
    with c1:
        # Gráfico de Linha Acumulado
        serie_ag = _by_day(only_f, sdr_date_col, None, "Agendadas")
        ts_ag = pd.DataFrame({"Data": idx}).merge(serie_ag.rename_axis("Data").reset_index(), on="Data", how="left")
        if "Agendadas" in ts_ag.columns:
            ts_ag["Agendadas"] = ts_ag["Agendadas"].fillna(0).astype(int)
            ts_ag["Agendadas_Acumulado"] = ts_ag["Agendadas"].cumsum()
            fig_l1 = px.line(ts_ag, x="Data", y="Agendadas_Acumulado", markers=True)
            fig_l1.update_traces(mode="lines+markers+text", text=ts_ag["Agendadas_Acumulado"], textposition="top center", cliponaxis=False)
            # Meta diária acumulada (SDRs): 7 * n_sdrs por dia
            meta_ag_acc = None
            if filtro_operador == "SDRs":
                meta_ag_acc = (np.arange(1, len(idx) + 1) * (7 * n_sdrs)).astype(float)
                fig_l1.add_scatter(
                    x=idx,
                    y=meta_ag_acc,
                    name="Meta diária (acumulada)",
                    line=dict(color="#bfa94c", width=2, dash="dash"),
                    mode="lines",
                )
            y_max_acc_1 = float(ts_ag["Agendadas_Acumulado"].max()) if len(ts_ag) else 0.0
            y_meta_1 = float(meta_ag_acc.max()) if meta_ag_acc is not None else 0.0
            fig_l1.update_yaxes(range=[0, max(1.0, y_max_acc_1, y_meta_1) * 1.12], automargin=True)
            fig_l1.update_layout(height=line_h, margin=dict(t=20, b=10, l=0, r=0))
            style_fig(fig_l1)
            st.plotly_chart(fig_l1, use_container_width=True)

            # Gráfico de Barras
            fig_b1 = px.bar(ts_ag, x="Data", y="Agendadas", text="Agendadas")
            fig_b1.update_traces(textposition="outside", cliponaxis=False)
            y_max_1 = max(int(ts_ag["Agendadas"].max()), 0)
            fig_b1.update_yaxes(range=[0, max(y_max_1, (7 * n_sdrs)) * 1.2])
            fig_b1.add_hline(y=(7 * n_sdrs), line_dash="dash", line_color="#bfa94c")
            fig_b1.update_layout(height=bar_h)
            style_fig(fig_b1)
            st.plotly_chart(fig_b1, use_container_width=True)

    with c2:
        # Gráfico de Linha Acumulado
        mask_eq = (reunioes_f["status_da_reuniao"] == "Executada Qualificada") if "status_da_reuniao" in reunioes_f.columns else None
        serie_eq = _by_day(reunioes_f, cons_date_col, mask_eq, "Executadas Qualificadas")
        ts_eq = pd.DataFrame({"Data": idx}).merge(serie_eq.rename_axis("Data").reset_index(), on="Data", how="left")
        if "Executadas Qualificadas" in ts_eq.columns:
            ts_eq["Executadas Qualificadas"] = ts_eq["Executadas Qualificadas"].fillna(0).astype(int)
            ts_eq["Executadas_Qualificadas_Acumulado"] = ts_eq["Executadas Qualificadas"].cumsum()
            fig_l2 = px.line(ts_eq, x="Data", y="Executadas_Qualificadas_Acumulado", markers=True)
            fig_l2.update_traces(mode="lines+markers+text", text=ts_eq["Executadas_Qualificadas_Acumulado"], textposition="top center", cliponaxis=False)
            # Meta diária acumulada (SDRs): 4 * n_sdrs por dia
            meta_eq_acc = None
            if filtro_operador == "SDRs":
                meta_eq_acc = (np.arange(1, len(idx) + 1) * (4 * n_sdrs)).astype(float)
                fig_l2.add_scatter(
                    x=idx,
                    y=meta_eq_acc,
                    name="Meta diária (acumulada)",
                    line=dict(color="#bfa94c", width=2, dash="dash"),
                    mode="lines",
                )
            y_max_acc_2 = float(ts_eq["Executadas_Qualificadas_Acumulado"].max()) if len(ts_eq) else 0.0
            y_meta_2 = float(meta_eq_acc.max()) if meta_eq_acc is not None else 0.0
            fig_l2.update_yaxes(range=[0, max(1.0, y_max_acc_2, y_meta_2) * 1.12], automargin=True)
            fig_l2.update_layout(height=line_h, margin=dict(t=20, b=10, l=0, r=0))
            style_fig(fig_l2)
            st.plotly_chart(fig_l2, use_container_width=True)

            # Gráfico de Barras
            fig_b2 = px.bar(ts_eq, x="Data", y="Executadas Qualificadas", text="Executadas Qualificadas")
            fig_b2.update_traces(textposition="outside", cliponaxis=False)
            y_max_2 = max(int(ts_eq["Executadas Qualificadas"].max()), 0)
            fig_b2.update_yaxes(range=[0, max(y_max_2, (4 * n_sdrs)) * 1.2])
            fig_b2.add_hline(y=(4 * n_sdrs), line_dash="dash", line_color="#bfa94c")
            fig_b2.update_layout(height=bar_h)
            style_fig(fig_b2)
            st.plotly_chart(fig_b2, use_container_width=True)

    with c3:
        # Gráfico de Linha Acumulado
        mask_ass = (reunioes_f["contrato"] == "Contrato Assinado") if "contrato" in reunioes_f.columns else None
        serie_ass = _by_day(reunioes_f, cons_date_col, mask_ass, "Contratos Assinados")
        ts_ass = pd.DataFrame({"Data": idx}).merge(serie_ass.rename_axis("Data").reset_index(), on="Data", how="left")
        if "Contratos Assinados" in ts_ass.columns:
            ts_ass["Contratos Assinados"] = ts_ass["Contratos Assinados"].fillna(0).astype(int)
            ts_ass["Contratos_Assinados_Acumulado"] = ts_ass["Contratos Assinados"].cumsum()
            fig_l3 = px.line(ts_ass, x="Data", y="Contratos_Assinados_Acumulado", markers=True)
            fig_l3.update_traces(mode="lines+markers+text", text=ts_ass["Contratos_Assinados_Acumulado"], textposition="top center", cliponaxis=False)
            # Meta diária acumulada (SDRs): 1 * n_sdrs por dia
            meta_ass_acc = None
            if filtro_operador == "SDRs":
                meta_ass_acc = (np.arange(1, len(idx) + 1) * (1 * n_sdrs)).astype(float)
                fig_l3.add_scatter(
                    x=idx,
                    y=meta_ass_acc,
                    name="Meta diária (acumulada)",
                    line=dict(color="#bfa94c", width=2, dash="dash"),
                    mode="lines",
                )
            y_max_acc_3 = float(ts_ass["Contratos_Assinados_Acumulado"].max()) if len(ts_ass) else 0.0
            y_meta_3 = float(meta_ass_acc.max()) if meta_ass_acc is not None else 0.0
            fig_l3.update_yaxes(range=[0, max(1.0, y_max_acc_3, y_meta_3) * 1.12], automargin=True)
            fig_l3.update_layout(height=line_h, margin=dict(t=20, b=10, l=0, r=0))
            style_fig(fig_l3)
            st.plotly_chart(fig_l3, use_container_width=True)

            # Gráfico de Barras
            fig_b3 = px.bar(ts_ass, x="Data", y="Contratos Assinados", text="Contratos Assinados")
            fig_b3.update_traces(textposition="outside", cliponaxis=False)
            y_max_3 = max(int(ts_ass["Contratos Assinados"].max()), 0)
            fig_b3.update_yaxes(range=[0, max(y_max_3, (1 * n_sdrs)) * 1.2])
            fig_b3.add_hline(y=(1 * n_sdrs), line_dash="dash", line_color="#bfa94c")
            fig_b3.update_layout(height=bar_h)
            style_fig(fig_b3)
            st.plotly_chart(fig_b3, use_container_width=True)
    st.markdown("---")

# ------------------------------------------------------------
# PERFORMANCE POR SDR / CONSULTOR
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("Performance e Rankings")
    colA, colB = st.columns(2)

    # Coluna A: Performance por SDR
    with colA:
        st.markdown("**Por SDR**")
        sdr_col = get_role_column(sdr_f, "sdr")
        if sdr_col:
            ag = sdr_f.groupby(sdr_col).size().rename("Agendadas")
            ex = cons_f.groupby(get_role_column(cons_f, "sdr")).size().rename("Executadas") if get_role_column(cons_f, "sdr") else pd.Series(dtype=int)
            perf = pd.concat([ag, ex], axis=1).fillna(0).reset_index().rename(columns={sdr_col: "SDR"})
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
                "autorange": "reversed",
            })
            dias_uteis = len(pd.bdate_range(data_ini, data_fim))
            meta = int(dias_uteis * 7)
            x_max = max(int(perf["Agendadas"].max()) if not perf.empty else 0, meta)
            fig_sdr.update_xaxes(range=[0, x_max * 1.1])
            fig_sdr.add_vline(x=meta, line_dash="dash", line_color="#bfa94c")
            fig_sdr.add_scatter(x=[None], y=[None], mode="lines", name="Meta", line=dict(color="#bfa94c", dash="dash"))
            fig_sdr.add_annotation(x=meta, y=1, xref="x", yref="paper", text=f"Meta ({meta})", showarrow=False, font=dict(color="#bfa94c", size=14))
            total_barras = int(perf["Agendadas"].sum()) if not perf.empty else 0
            fig_sdr.add_annotation(
                x=1.0, y=0.067, xref="paper", yref="paper",
                text=f"Total: {total_barras}", showarrow=False,
                xanchor="right", yanchor="middle", align="center",
                font=dict(color="#bfa94c", size=18),
                bordercolor="#bfa94c", borderwidth=1.8, borderpad=4,
                bgcolor="rgba(0,0,0,0)",
            )
            style_fig(fig_sdr)
            st.plotly_chart(fig_sdr, use_container_width=True)
            st.dataframe(perf, use_container_width=True, hide_index=True)
        else:
            st.info("Coluna SDR não encontrada na planilha.")

    # Coluna B: Agendamentos (Consultores) acima de Contratos por Consultor
    with colB:
        st.markdown("**Agendamentos por Consultor (eu mesmo)**")
        try:
            sdr_col = get_role_column(sdr_f, "sdr")
            cons_col_in_sdr = get_role_column(sdr_f, "consultor")
            cons_col_in_cons = get_role_column(cons_f, "consultor")
            if sdr_col and cons_col_in_sdr:
                df_cons_ag = sdr_f.copy()
                df_cons_ag = df_cons_ag[df_cons_ag[sdr_col].fillna("") == "Consultor (eu mesmo)"]
                ag = df_cons_ag.groupby(cons_col_in_sdr).size().rename("Agendadas")
                if cons_col_in_cons and "status_da_reuniao" in cons_f.columns:
                    ex = cons_f.groupby(cons_col_in_cons)["status_da_reuniao"].apply(lambda x: x.isin(["Executada Qualificada", "Executada não Qualificada"]).sum()).rename("Executadas")
                else:
                    ex = pd.Series(dtype=int)
                perf_c = pd.concat([ag, ex], axis=1).fillna(0).reset_index()
                if "Consultor" not in perf_c.columns:
                    if "index" in perf_c.columns:
                        perf_c = perf_c.rename(columns={"index": "Consultor"})
                    else:
                        first_col = perf_c.columns[0]
                        perf_c = perf_c.rename(columns={first_col: "Consultor"})
                perf_c = perf_c.sort_values(by="Agendadas", ascending=False)
                perf_c["Taxa Comparecimento (%)"] = np.where(perf_c["Agendadas"]>0, (perf_c["Executadas"]/perf_c["Agendadas"]*100), 0).round(1)
                fig_cag = px.bar(
                    perf_c,
                    x="Agendadas",
                    y="Consultor",
                    orientation="h",
                    hover_data=["Executadas","Taxa Comparecimento (%)"],
                    text="Agendadas",
                )
                fig_cag.update_traces(textposition="outside")
                fig_cag.update_layout(yaxis={
                    "categoryorder": "array",
                    "categoryarray": perf_c["Consultor"].tolist(),
                    "autorange": "reversed",
                })
                dias_uteis = len(pd.bdate_range(data_ini, data_fim))
                meta_c = int(dias_uteis * 2)
                x_max_c = max(int(perf_c["Agendadas"].max()) if not perf_c.empty else 0, meta_c)
                fig_cag.update_xaxes(range=[0, x_max_c * 1.1])
                fig_cag.add_vline(x=meta_c, line_dash="dash", line_color="#bfa94c")
                fig_cag.add_scatter(x=[None], y=[None], mode="lines", name="Meta", line=dict(color="#bfa94c", dash="dash"))
                fig_cag.add_annotation(x=meta_c, y=1, xref="x", yref="paper", text=f"Meta ({meta_c})", showarrow=False, font=dict(color="#bfa94c", size=14))
                # Total de agendamentos (caixa, semelhante ao gráfico de SDR)
                total_barras_c = int(perf_c["Agendadas"].sum()) if not perf_c.empty else 0
                fig_cag.add_annotation(
                    x=1.0, y=0.067, xref="paper", yref="paper",
                    text=f"Total: {total_barras_c}", showarrow=False,
                    xanchor="right", yanchor="middle", align="center",
                    font=dict(color="#bfa94c", size=18),
                    bordercolor="#bfa94c", borderwidth=1.8, borderpad=4,
                    bgcolor="#000000"
                )
                style_fig(fig_cag)
                st.plotly_chart(fig_cag, use_container_width=True)
                st.dataframe(perf_c, use_container_width=True, hide_index=True)
            else:
                st.info("Não foi possível identificar as colunas de SDR/Consultor na planilha de agendamentos.")
        except Exception as e:
            st.warning(f"Falha ao gerar ranking de agendamentos por Consultor: {e}")

        st.markdown("**Contratos por Consultor**")
        cons_col = get_role_column(cons_f, "consultor")
        if cons_col:
            grp = cons_f.groupby(cons_col).agg(
                Reunioes=("status_da_reuniao", "count"),
                Executadas=("status_da_reuniao", lambda x: x.isin(["Executada Qualificada", "Executada não Qualificada"]).sum()),
                Qualificadas=("status_da_reuniao", lambda x: (x=="Executada Qualificada").sum()),
                Contratos_Assinados=("contrato", lambda x: (x=="Contrato Assinado").sum()),
            ).reset_index()
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
                "autorange": "reversed",
            })
            style_fig(fig_cons)
            st.plotly_chart(fig_cons, use_container_width=True)
            st.dataframe(grp, use_container_width=True, hide_index=True)
        else:
            st.info("Coluna Consultor não encontrada na planilha.")

    st.divider()
    st.subheader("☎️ **Ranking de SDRs**")
    
    rows = []
    for sdr in sdrs:
        sdr_reunioes = sdr_f.loc[sdr_f["nome_do_sdr"].eq(sdr)].shape[0]
        sdr_n_qual = cons_f.loc[cons_f["nome_do_sdr"].eq(sdr) & cons_f["status_da_reuniao"].eq("Executada Qualificada")].shape[0]
        sdr_n_cont = cons_f.loc[cons_f["nome_do_sdr"].eq(sdr) & cons_f["contrato"].eq("Contrato Assinado")].shape[0]

        rows.append({
            "SDR": sdr,
            "Reuniões Marcadas": sdr_reunioes,
            "Reuniões Qualificadas": sdr_n_qual,
            "Contratos Assinados": sdr_n_cont,
            "Conv Ag→Qualif (%)": (round((sdr_n_qual / sdr_reunioes * 100), 1) if sdr_reunioes > 0 else 0.0),
            "Conv Qualif→Contrato (%)": (round((sdr_n_cont / sdr_n_qual * 100), 1) if sdr_n_qual > 0 else 0.0),
        })

    sdr_ranking = pd.DataFrame(rows)
    # adicionar conversão total Ag→Contrato e ordenar desc
    sdr_ranking["Conv Ag→Contrato (%)"] = np.where(
        sdr_ranking["Reuniões Marcadas"] > 0,
        (sdr_ranking["Contratos Assinados"] / sdr_ranking["Reuniões Marcadas"]) * 100,
        0.0,
    ).round(1)
    sdr_ranking = sdr_ranking.sort_values(by="Conv Ag→Contrato (%)", ascending=False).reset_index(drop=True)

    # renderização em cards
    html = ["<div class='ranking-grid'>"]
    for i, row in sdr_ranking.iterrows():
        title = row["SDR"]
        ag = int(row["Reuniões Marcadas"]) if not pd.isna(row["Reuniões Marcadas"]) else 0
        qual = int(row["Reuniões Qualificadas"]) if not pd.isna(row["Reuniões Qualificadas"]) else 0
        cont = int(row["Contratos Assinados"]) if not pd.isna(row["Contratos Assinados"]) else 0
        conv_ag_qual = float(row["Conv Ag→Qualif (%)"]) if not pd.isna(row["Conv Ag→Qualif (%)"]) else 0.0
        conv_qual_cont = float(row["Conv Qualif→Contrato (%)"]) if not pd.isna(row["Conv Qualif→Contrato (%)"]) else 0.0
        conv_total = float(row["Conv Ag→Contrato (%)"]) if not pd.isna(row["Conv Ag→Contrato (%)"]) else 0.0
        card = textwrap.dedent(f"""
        <div class='ranking-card'>
          <div class='rank'>#{i+1}</div>
          <div class='title'>{title}</div>
          <div class='progress'><div class='fill' style='width:{conv_total}%;'></div></div>
          <div class='metrics'>
            <div class='item'><span class='label'>Agendadas</span><span class='value'>{ag}</span></div>
            <div class='item'><span class='label'>Conv Ag→Qualif</span><span class='value'>{conv_ag_qual:.1f}%</span></div>
            <div class='item'><span class='label'>Qualificadas</span><span class='value'>{qual}</span></div>
            <div class='item'><span class='label'>Conv Qualif→Contrato</span><span class='value'>{conv_qual_cont:.1f}%</span></div>
            <div class='item'><span class='label'>Contratos</span><span class='value'>{cont}</span></div>
            <div class='item'><span class='label'>Conv Total</span><span class='value'>{conv_total:.1f}%</span></div>
          </div>
        </div>
        """).strip()
        html.append(card)
    html.append("</div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)


with tabs[3]:
    st.subheader("Pipeline de Contratos")
    pipe = cons_f.copy()
    if not pipe.empty:
        ct = pipe["contrato"].value_counts().reset_index()
        ct.columns = ["Status do Contrato", "Qtd"]
        fig_ct = px.bar(ct, x="Status do Contrato", y="Qtd", color="Status do Contrato", text="Qtd")
        fig_ct.update_traces(textposition="outside", cliponaxis=False)
        style_fig(fig_ct)
        st.plotly_chart(fig_ct, use_container_width=True)
    else:
        st.info("Sem dados de contratos.")

st.markdown("---")

# ------------------------------------------------------------
# RODAPÉ
# ------------------------------------------------------------
st.caption("🕒 Todos os números respeitam o **Filtro de Data** (reunião ou agendamento). Use o preset **Hoje** para ver apenas os dados do dia.")
