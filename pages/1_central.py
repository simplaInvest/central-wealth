import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from data.transformer import formata_reais, formata_delta_milhoes_brl, formata_milhoes_brl
from data.api_calls import request_generic
from viz.charts import bar_chart, pie_chart

# =========================
# ------ PAGE CONFIG ------
# =========================
st.set_page_config(page_title="Central", layout="wide")
st.logo(image='assets/z_logo_light.png', size='large')
st.write(""); st.write(""); st.write("")

st.markdown("""
    <style>
        /* Seletor mais genérico */
        .block-container { 
            padding-top: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# ------ PAGE LAYOUT ------
# =========================

metrics_cols = st.columns(6)

dados_custodia = request_generic("https://api.simplainvest.com.br/wealth/metricas-custodia")

# pegue os valores numéricos crus
v_total        = float(dados_custodia['custodia_total'])
v_corretoras   = float(dados_custodia['custodia_corretoras'])
v_fee_fixo     = float(dados_custodia['custodia_fee_fixo'])
v_b2c          = float(dados_custodia['custodia_b2c'])
v_b2b          = float(dados_custodia['custodia_b2b'])
v_inativa      = float(dados_custodia['custodia_inativa'])
proporcao_custodia = dados_custodia['proporcao_custodia']

dados_historico = request_generic('https://api.simplainvest.com.br/wealth/historico-custodia') 
hist_custodia = pd.DataFrame(dados_historico["histórico custódia"]) 
hist_clientes = pd.DataFrame(dados_historico["histórico clientes"]) 
hist_captacao = pd.DataFrame(dados_historico["histórico captação"])

# exemplo de delta (pega penúltimo valor de CAPTACAO)
delta_capt_total = float(hist_captacao['CAPTACAO'].iloc[-2])

with metrics_cols[0]:
    st.metric(
        label="Custódia Total",
        value=formata_milhoes_brl(v_total),
        help="Custódia todas as corretoras + externo",
        delta=formata_delta_milhoes_brl(delta_capt_total),
        border=True
    )

with metrics_cols[1]:
    st.metric("Custódia Corretoras", formata_milhoes_brl(v_corretoras), help="Custódia todas as corretoras", border=True)

with metrics_cols[2]:
    st.metric("Custódia Fee Fixo", formata_milhoes_brl(v_fee_fixo), help="Custódia dos consultores somada", border=True)

with metrics_cols[3]:
    st.metric("Custódia B2C", formata_milhoes_brl(v_b2c), help="Custódia B2C", border=True)

with metrics_cols[4]:
    st.metric("Custódia B2B", formata_milhoes_brl(v_b2b), help="Custódia B2B", border=True)

with metrics_cols[5]:
    st.metric("Custódia Inativa", formata_milhoes_brl(v_inativa), help="Custódia no Código do Mudinho e do Youssef", border=True)

grafs_cols = st.columns(2)

with grafs_cols[0]:
    fig, df_plot = bar_chart(
        hist_custodia,
        date_col="DATA",
        value_col="TOTAL",
        title="Evolução da Custódia em Meses",
        bar_color="#C9A227",       # dourado
        secondary_color="#FFFFFF", # branco
        decimals=0,
        prefix="R$ ",
        suffix="",
        showgrid=True,
        text_inside=False,
        compact_thousands=True,
    )
    fig.update_layout(
        height=320,  # ↓ experimente 260–360
        margin=dict(t=40, b=30, l=10, r=10),  # margens mais enxutas
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2))  # legenda horizontal fora do gráfico (opcional)

    st.plotly_chart(fig, use_container_width=True)

    fig, df_plot = bar_chart(
        hist_captacao,
        date_col="DATA",
        value_col="CAPTACAO",
        title="Captação Mês a Mês",
        bar_color="#C9A227",       # dourado
        secondary_color="#FFFFFF", # branco
        decimals=0,
        prefix="R$ ",
        suffix="",
        showgrid=True,
        text_inside=False,
        compact_thousands=True,
    )
    fig.update_layout(
        height=320,  # ↓ experimente 260–360
        margin=dict(t=40, b=30, l=10, r=10),  # margens mais enxutas
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2))  # legenda horizontal fora do gráfico (opcional)

    st.plotly_chart(fig, use_container_width=True)

with grafs_cols[1]:
    fig, df_plot = bar_chart(
        hist_clientes,
        date_col="DATA",
        value_col="TOTAL",
        title="Evolução número de clientes",
        bar_color="#C9A227",       # dourado
        secondary_color="#FFFFFF", # branco
        decimals=0,
        prefix="",
        suffix="",
        showgrid=True,
        text_inside=False,
        compact_thousands=False,
    )
    fig.update_layout(
        height=320,  # ↓ experimente 260–360
        margin=dict(t=40, b=30, l=10, r=10),  # margens mais enxutas
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2))  # legenda horizontal fora do gráfico (opcional)
    st.plotly_chart(fig, use_container_width=True)

    fig, df_plot = pie_chart(
        proporcao_custodia,
        title="Distribuição das Corretoras",
        night_mode=True,              # otimizado para o tema escuro do Streamlit
        hole=0.45,                    # donut
        sort_slices=True,
        show_values=True,
        min_percent_for_label=3.0,    # esconde rótulos muito pequenos
        group_small_into_others=True, # agrupa fatias < 3% em 'Outros'
        decimals_pct=1,
        value_prefix="",              # ex.: "R$ "
        value_suffix="%",             # se seus 'valores' já forem percentuais, pode usar "%"
    )
    fig.update_layout(
        height=300,
        margin=dict(t=40, b=30, l=10, r=10),
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2))  # opcional
    st.plotly_chart(fig, use_container_width=True)

