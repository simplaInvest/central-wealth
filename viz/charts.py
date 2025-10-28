# Função solicitada + exemplo de uso com o arquivo enviado
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Optional

def bar_chart(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str = "Evolução da Métrica",
    bar_color: str = "#C9A227",       # dourado principal
    secondary_color: str = "#FFFFFF", # branco secundário (texto/contraste)
    height: int = 450,
    width: int = 900,
    decimals: int = 0,
    prefix: str = "",
    suffix: str = "",
    showgrid: bool = True,
    font_family: str = "Arial",
    text_inside: bool = False,        # False -> valores acima da barra
    compact_thousands: bool = True,   # True -> 1.2M, 340k etc.
    night_mode: bool = True,          # Ajuste para tema noturno do Streamlit
):
    """
    Cria um gráfico de barras (Plotly) em dourado/branco com anotações dos valores.
    Retorna: (fig, df_plot)
    """
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError("As colunas informadas não existem no DataFrame.")
    
    # Copia para não alterar DF original
    d = df[[date_col, value_col]].copy()
    
    # Parse de data (tenta D/M/Y e fallback para parsing genérico)
    try:
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="raise")
    except Exception:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    
    # Remove linhas com data nula e ordena
    d = d.dropna(subset=[date_col]).sort_values(by=date_col)
    
    # Trata valores
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col])
    
    # Formatação compacta (k/M/B) para rótulos
    def _compact(v):
        abs_v = abs(v)
        if abs_v >= 1_000_000_000:
            return f"{v/1_000_000_000:.{decimals}f}B"
        if abs_v >= 1_000_000:
            return f"{v/1_000_000:.{decimals}f}M"
        if abs_v >= 1_000:
            return f"{v/1_000:.{decimals}f}k"
        return f"{v:.{decimals}f}"
    
    if compact_thousands:
        d["_label"] = d[value_col].apply(_compact)
    else:
        d["_label"] = d[value_col].apply(
            lambda v: f"{v:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
    
    d["_label"] = prefix + d["_label"] + suffix
    
    # Textos sobre as barras
    textposition = "outside" if not text_inside else "inside"
    # cor dos rótulos: no tema escuro, fora = branco; dentro = quase preto para contraste no dourado
    text_color = secondary_color if textposition == "outside" else "#111111"
    
    # HOVER (evitar f-string com %{y})
    hovertemplate = "%{x|%d/%m/%Y}<br>" + value_col + ": " + prefix + "%{y:,.2f}" + suffix + "<extra></extra>"
    
    # Cria figura
    fig = go.Figure(
        data=[
            go.Bar(
                x=d[date_col],
                y=d[value_col],
                text=d["_label"],
                textposition=textposition,
                textfont=dict(color=text_color),
                marker=dict(
                    color=bar_color,
                    line=dict(color=bar_color, width=0),
                ),
                hovertemplate=hovertemplate,
            )
        ]
    )
    
    # Estilo (tema noturno do Streamlit)
    if night_mode:
        plot_bg = "rgba(0,0,0,0)"   # transparente para casar com o tema escuro do Streamlit
        paper_bg = "rgba(0,0,0,0)"
        font_color = "#FFFFFF"
        grid_color = "rgba(255,255,255,0.15)" if showgrid else "rgba(0,0,0,0)"
        axis_line_color = "rgba(255,255,255,0.25)"
        hover_bg = "#222222"
        hover_font = "#FFFFFF"
        template = "plotly_dark"
    else:
        plot_bg = secondary_color
        paper_bg = secondary_color
        font_color = "#111111"
        grid_color = "#E8E2C8" if showgrid else secondary_color
        axis_line_color = "#BBBBBB"
        hover_bg = "#F9F9F9"
        hover_font = "#111111"
        template = "plotly_white"
    
    fig.update_layout(
        template=template,
        title=dict(text=title, x=0.01, xanchor="left", font=dict(color=font_color)),
        height=height,
        width=width,
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font=dict(family=font_family, color=font_color),
        margin=dict(l=50, r=30, t=60, b=50),
        hoverlabel=dict(bgcolor=hover_bg, bordercolor=axis_line_color, font=dict(color=hover_font)),
        bargap=0.2,
    )

    fig.update_xaxes(
        title=None,
        showgrid=False,
        showline=True,
        linecolor=axis_line_color,
        tickformat="%b\n%Y",
        tickfont=dict(color=font_color),
    )
    fig.update_yaxes(
        title=None,
        showgrid=showgrid,
        gridcolor=grid_color,
        zeroline=False,
        tickfont=dict(color=font_color),
    )
    
    # Evita cortar rótulos fora da área do gráfico
    fig.update_traces(cliponaxis=False)
    
    return fig, d

def pie_chart(
    data: Dict[str, float],
    title: str = "Distribuição por Categoria",
    night_mode: bool = True,
    hole: float = 0.45,                  # 0 para pizza, >0 para donut
    sort_slices: bool = True,            # ordena fatias por valor (desc)
    show_values: bool = True,            # mostra % nas fatias
    min_percent_for_label: float = 3.0,  # oculta rótulos abaixo desse %
    group_small_into_others: bool = True,
    others_label: str = "Outros",
    decimals_pct: int = 1,               # casas decimais nos % do rótulo
    value_prefix: str = "",              # prefixo nos valores do hover (ex.: "R$ ")
    value_suffix: str = "",              # sufixo nos valores do hover
    color_sequence: Optional[List[str]] = None,  # sequência de cores personalizada
    font_family: str = "Arial",
    height: int = 420,
    width: int = 700,
):
    """
    Cria um gráfico de pizza (ou donut) em Plotly, adequado ao modo noturno do Streamlit.
    - data: dict {"Categoria": valor, ...}
    - night_mode: troca o tema para fundo escuro e textos claros
    - hole: 0 para pizza; 0.3–0.6 para donut
    - sort_slices: ordena fatias desc
    - show_values: exibe % como rótulo
    - min_percent_for_label: esconde rótulos muito pequenos
    - group_small_into_others: agrupa fatias com % < min_percent_for_label em "Outros"
    - color_sequence: lista de cores hex; se None, usa uma paleta dourado + neutros
    Retorna: fig (plotly.graph_objects.Figure) e um DataFrame com labels/values/percent.
    """
    # --- Preparação do dataframe ---
    s = pd.Series(data, dtype="float64")
    s = s.dropna()
    if s.empty:
        raise ValueError("O dicionário 'data' está vazio ou inválido.")

    # Ordena se for o caso
    if sort_slices:
        s = s.sort_values(ascending=False)

    total = s.sum()
    pct = (s / total) * 100
    df = pd.DataFrame({"label": s.index, "value": s.values, "percent": pct.values})

    # Agrupar fatias pequenas em "Outros"
    if group_small_into_others and min_percent_for_label > 0:
        small = df["percent"] < min_percent_for_label
        if small.any():
            others_value = df.loc[small, "value"].sum()
            others_pct = df.loc[small, "percent"].sum()
            df = df.loc[~small].copy()
            if others_value > 0:
                df = pd.concat(
                    [df, pd.DataFrame([{"label": others_label, "value": others_value, "percent": others_pct}])],
                    ignore_index=True
                )

    # Recalcula ordem (garantir maiores primeiro após agrupamento)
    if sort_slices:
        df = df.sort_values(by="value", ascending=False).reset_index(drop=True)

    # --- Paleta de cores ---
    # Padrão: destaque dourado + neutros legíveis em dark
    if color_sequence is None:
        color_sequence = [
            "#C9A227",  # gold (principal)
            "#7A7A7A",  # cinza médio
            "#3E3E3E",  # cinza escuro
            "#B3B3B3",  # cinza claro
            "#8C7853",  # bronze
            "#4F4F4F",  # grafite
            "#9A8F68",  # dourado esmaecido
            "#2E2E2E",  # quase preto
        ]
    # Se houver mais categorias que cores, repete paleta
    if len(color_sequence) < len(df):
        reps = (len(df) // len(color_sequence)) + 1
        color_sequence = (color_sequence * reps)[:len(df)]

    # --- Texto e hover ---
    # rótulo curto com %; valores ficam no hover
    textinfo = "percent" if show_values else "none"

    # Exibe rótulos apenas se percent >= min_percent_for_label
    texts = []
    for p in df["percent"]:
        if show_values and p >= min_percent_for_label:
            texts.append(f"{p:.{decimals_pct}f}%")
        else:
            texts.append("")  # sem rótulo na fatia

    hovertemplate = (
        "<b>%{label}</b><br>"
        f"Valor: {value_prefix}" + "%{value:,.2f}" + f"{value_suffix}<br>"
        "Participação: %{percent:.2%}<extra></extra>"
    )

    # --- Layout dark ---
    if night_mode:
        template = "plotly_dark"
        plot_bg = "rgba(0,0,0,0)"   # transparente para se integrar ao tema do Streamlit
        paper_bg = "rgba(0,0,0,0)"
        font_color = "#FFFFFF"
        legend_bg = "rgba(0,0,0,0)"
        legend_border = "rgba(255,255,255,0.25)"
        line_color = "rgba(255,255,255,0.25)"  # borda sutil nas fatias
    else:
        template = "plotly_white"
        plot_bg = "#FFFFFF"
        paper_bg = "#FFFFFF"
        font_color = "#111111"
        legend_bg = "#FFFFFF"
        legend_border = "rgba(0,0,0,0.15)"
        line_color = "rgba(0,0,0,0.15)"

    # --- Figura ---
    fig = go.Figure(
        data=[
            go.Pie(
                labels=df["label"],
                values=df["value"],
                hole=hole,
                text=texts,            # rótulos custom (controle por min_percent_for_label)
                textinfo="text",       # usamos 'text' (já controlado acima)
                hovertemplate=hovertemplate,
                marker=dict(
                    colors=color_sequence,
                    line=dict(color=line_color, width=1),
                ),
                sort=False,            # já ordenamos manualmente
                direction="clockwise",
                pull=[0.04] + [0]*(len(df)-1),  # dá um leve destaque à maior fatia
            )
        ]
    )

    fig.update_layout(
        template=template,
        title=dict(text=title, x=0.01, xanchor="left", font=dict(color=font_color)),
        height=height,
        width=width,
        showlegend=True,
        legend=dict(
            bgcolor=legend_bg,
            bordercolor=legend_border,
            borderwidth=1,
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,   # legenda à direita
            font=dict(family=font_family, color=font_color, size=12),
            itemwidth=30,
        ),
        margin=dict(l=20, r=160, t=60, b=20),  # espaço para legenda à direita
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=font_family, color=font_color),
    )

    return fig, df