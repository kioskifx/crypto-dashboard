import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── CONFIG ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Market Monitor", layout="wide")
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; max-width: 98%; }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

COLOR_POS = '#A5D6F7'
COLOR_NEG = '#F6B995'

_today = datetime.now()
np.random.seed(_today.day)   # same daily seed as original

# ── DATA GENERATORS ────────────────────────────────────────────────────────────
def gen_returns(is_weekly=False):
    bins = ['-20%', '-15%', '-10%', '-5%', '5%', '10%', '15%', '20%']
    # Base distributions tuned to match the screenshot (bullish day/week)
    base = [0, 11, 8, 18, 55, 118, 75, 25] if is_weekly else [0, 1, 19, 53, 238, 5, 1, 1]
    noise  = np.random.randint(-3, 4, len(base))
    counts = [max(0, int(b + n)) for b, n in zip(base, noise)]
    return pd.DataFrame({'Bin': bins, 'Count': counts})

def gen_zscores():
    sectors = ['MEME','L1','DINO','AI','LST','NEW','CEXDEX','L2',
               'NFT','NAR','DEFI','GAMING','DEPIN','PRIVACY','AGENTS','INFRA','DWF','INFOFI']
    scores  = np.random.uniform(-1.2, 1.2, len(sectors))
    return (pd.DataFrame({'Sector': sectors, 'Z-Score': scores})
              .sort_values('Z-Score', ascending=True)
              .reset_index(drop=True))

def gen_setups():
    # Fixed counts — matches screenshot
    return pd.DataFrame({
        'Setup': ['Backside Short','Breakdown','Contd. Short',
                  'Episodic Pivot','Contd. Breakout','Bottom Bounce'],
        'Count': [-9, -7, -1, 6, 29, 30]
    })

def gen_timeseries(days=120):
    dates = [_today - timedelta(days=x) for x in range(days - 1, -1, -1)]
    v2d   = np.random.uniform(0.5, 2.5, days) + np.sin(np.linspace(0, 12, days)) * 0.5
    v5d   = np.random.uniform(0.3, 1.5, days) + np.sin(np.linspace(0, 12, days)) * 0.3
    v2d[-1], v5d[-1] = 10.0, 5.0          # today's spike visible in screenshot

    osc_p = np.random.uniform(-8, 8, days)
    osc_s = np.random.uniform(-15, 15, days)
    osc_s[15:25] = np.random.uniform(-90, -50, 10)   # Feb crash (arrow in screenshot)

    return pd.DataFrame({'Date': dates, '2_day': v2d, '5_day': v5d,
                         'Osc_Primary': osc_p, 'Osc_Secondary': osc_s})

def gen_calendar():
    """Full-year calendar grid: 7 rows (Mon–Sun) × ~54 week columns."""
    year       = _today.year
    today_date = _today.date()
    all_dates  = pd.date_range(f'{year}-01-01', f'{year}-12-31')

    # Anchor week 0 to the Monday on-or-before Jan 1
    jan1      = all_dates[0]
    first_mon = jan1 - timedelta(days=jan1.dayofweek)

    rows = []
    for d in all_dates:
        is_past = d.date() <= today_date
        val     = float(np.random.uniform(-1, 1)) if is_past else np.nan
        rows.append({
            'date':     d,
            'day':      d.day,
            'dow':      d.dayofweek,          # 0 = Mon … 6 = Sun
            'month':    d.month,
            'week_col': (d.date() - first_mon.date()).days // 7,
            'val':      val,
        })
    return pd.DataFrame(rows)


# ── PRE-COMPUTE ────────────────────────────────────────────────────────────────
df_p1d   = gen_returns(is_weekly=False)
df_p1w   = gen_returns(is_weekly=True)
df_zp1d  = gen_zscores()          # consumes some random state
df_zp1w  = gen_zscores()          # advances state → different from p1d
df_setups = gen_setups()
df_ts    = gen_timeseries()
df_cal   = gen_calendar()


# ── LAYOUT HELPERS ─────────────────────────────────────────────────────────────
def clean(fig, height=350):
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10),
    )
    return fig

def bar_v(df, x_col, y_col, title):
    """Vertical bar chart for return distributions."""
    colors = [COLOR_NEG if '-' in str(x) else COLOR_POS for x in df[x_col]]
    fig = go.Figure(go.Bar(
        x=df[x_col], y=df[y_col],
        marker_color=colors,
        text=df[y_col], textposition='outside',
    ))
    fig.update_yaxes(range=[0, df[y_col].max() * 1.3])
    fig.update_xaxes(fixedrange=True)
    st.markdown(f"<p style='text-align:center;font-weight:bold;'>{title}</p>",
                unsafe_allow_html=True)
    st.plotly_chart(clean(fig), use_container_width=True, config={'displayModeBar': False})

def bar_h(df, x_col, y_col, title, abs_text=False):
    """Horizontal bar chart for z-scores / setups."""
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in df[x_col]]
    txt    = df[x_col].abs().round(1) if abs_text else df[x_col].round(1)
    fig = go.Figure(go.Bar(
        x=df[x_col], y=df[y_col], orientation='h',
        marker_color=colors,
        text=txt, textposition='outside',
    ))
    # Give text room on both sides
    xpad = df[x_col].abs().max() * 0.25
    fig.update_xaxes(range=[df[x_col].min() - xpad, df[x_col].max() + xpad])
    st.markdown(f"<p style='text-align:center;font-weight:bold;'>{title}</p>",
                unsafe_allow_html=True)
    st.plotly_chart(clean(fig), use_container_width=True, config={'displayModeBar': False})


# ── HEADER ─────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([9, 1])
c1.markdown("## Market Monitor")
c2.markdown(f"#### {_today.strftime('%Y-%m-%d')}")
st.divider()


# ── ROW 1 ──────────────────────────────────────────────────────────────────────
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
with r1c1: bar_v(df_p1d,   'Bin',     'Count',   'P1D Returns')
with r1c2: bar_v(df_p1w,   'Bin',     'Count',   'P1W Returns')
with r1c3: bar_h(df_zp1d,  'Z-Score', 'Sector',  'Sector Z-Scores P1D')
with r1c4: bar_h(df_zp1w,  'Z-Score', 'Sector',  'Sector Z-Scores P1W')
with r1c5: bar_h(df_setups,'Count',   'Setup',   'Setups P1W', abs_text=True)

st.write("")


# ── ROW 2 ──────────────────────────────────────────────────────────────────────
r2c1, r2c2, r2c3 = st.columns(3)

with r2c1:
    st.markdown("<p style='text-align:center;font-weight:bold;'>Primary Breadth Ratio (5% daily moves)</p>",
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['2_day'],
                             mode='lines', name='2 day ratio', line=dict(color=COLOR_POS)))
    fig.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['5_day'],
                             mode='lines', name='5 day ratio', line=dict(color=COLOR_NEG)))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(clean(fig, 300), use_container_width=True, config={'displayModeBar': False})

with r2c2:
    st.markdown("<p style='text-align:center;font-weight:bold;'>Primary Breadth Oscillator (5% daily moves)</p>",
                unsafe_allow_html=True)
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in df_ts['Osc_Primary']]
    fig = go.Figure(go.Bar(x=df_ts['Date'], y=df_ts['Osc_Primary'], marker_color=colors))
    st.plotly_chart(clean(fig, 300), use_container_width=True, config={'displayModeBar': False})

with r2c3:
    st.markdown("<p style='text-align:center;font-weight:bold;'>Secondary Breadth Oscillator (1σ weekly moves)</p>",
                unsafe_allow_html=True)
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in df_ts['Osc_Secondary']]
    fig = go.Figure(go.Bar(x=df_ts['Date'], y=df_ts['Osc_Secondary'], marker_color=colors))
    st.plotly_chart(clean(fig, 300), use_container_width=True, config={'displayModeBar': False})

st.write("")


# ── ROW 3: YEAR CALENDAR HEATMAP ───────────────────────────────────────────────
st.markdown("<p style='text-align:center;font-weight:bold;'>Daily Performance Heatmap</p>",
            unsafe_allow_html=True)

# Build 7 × n_weeks matrices
n_weeks = int(df_cal['week_col'].max()) + 1
z    = np.full((7, n_weeks), np.nan)
text = np.full((7, n_weeks), '', dtype=object)

for _, row in df_cal.iterrows():
    r, c       = int(row['dow']), int(row['week_col'])
    z[r, c]    = row['val']
    text[r, c] = str(int(row['day']))

# Month label annotations (centred over each month's week columns)
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
month_spans = df_cal.groupby('month')['week_col'].agg(['min', 'max'])

annotations = []
for m, span in month_spans.iterrows():
    x_mid = (span['min'] + span['max']) / 2
    annotations.append(dict(
        x=x_mid, y=1.10,
        xref='x', yref='paper',
        text=f"<b>{month_names[m - 1]}</b>",
        showarrow=False,
        font=dict(size=10),
        xanchor='center',
    ))

fig_cal = go.Figure(go.Heatmap(
    z=z,
    x=list(range(n_weeks)),
    y=list(range(7)),
    text=text,
    texttemplate="%{text}",
    textfont={"size": 9, "color": "black"},
    colorscale=[[0, COLOR_NEG], [0.5, '#f0f0f0'], [1, COLOR_POS]],
    showscale=False,
    xgap=3, ygap=3,
    zmin=-1, zmax=1,
    hoverinfo='none',
))

fig_cal.update_layout(
    height=240,
    margin=dict(l=35, r=10, t=45, b=10),
    yaxis=dict(
        ticktext=['M', 'T', 'W', 'T', 'F', 'S', 'S'],
        tickvals=list(range(7)),
        autorange='reversed',
        showgrid=False,
        zeroline=False,
        tickfont=dict(size=10),
    ),
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    annotations=annotations,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)

st.plotly_chart(fig_cal, use_container_width=True, config={'displayModeBar': False})
