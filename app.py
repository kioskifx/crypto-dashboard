import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════════════════════
# 1 — CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Market Monitor", layout="wide")
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; max-width: 98%; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

COLOR_POS = '#A5D6F7'
COLOR_NEG = '#F6B995'
_today    = datetime.now()
np.random.seed(_today.day)


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — COIN UNIVERSE
# ═══════════════════════════════════════════════════════════════════════════════
# (sector_name, n_coins, market_beta)
SECTOR_DEF = [
    ('MEME',    30, 1.80),
    ('L1',      20, 1.00),
    ('DINO',    18, 0.70),
    ('AI',      25, 1.40),
    ('LST',     12, 0.90),
    ('NEW',     15, 1.20),
    ('CEXDEX',  15, 1.10),
    ('L2',      18, 1.20),
    ('NFT',     15, 1.50),
    ('NAR',     10, 0.80),
    ('DEFI',    20, 1.30),
    ('GAMING',  18, 1.60),
    ('DEPIN',   12, 1.30),
    ('PRIVACY', 10, 1.10),
    ('AGENTS',  18, 1.50),
    ('INFRA',   15, 1.10),
    ('DWF',      8, 1.00),
    ('INFOFI',  10, 1.20),
]
SECTORS      = [d[0] for d in SECTOR_DEF]
coin_sectors = np.array([s for name, n, _ in SECTOR_DEF for s in [name] * n])
coin_betas   = np.array([b for name, n, b in SECTOR_DEF for _ in range(n)])
N_COINS      = len(coin_sectors)  # 289

# Per-coin idiosyncratic daily volatility (sampled once, stable across charts)
coin_idio_vol = np.random.uniform(0.020, 0.055, N_COINS)

# Primary breadth universe: 25 highest-beta coins
primary_idx = np.argsort(coin_betas)[-25:]


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — SINGLE RETURN MATRIX  (all charts derive from this)
# ═══════════════════════════════════════════════════════════════════════════════
LOOKBACK = 90
dates_all = np.array([_today - timedelta(days=x) for x in range(LOOKBACK - 1, -1, -1)])

def build_market_factor(n):
    """
    Regime-structured daily market factor (index 0 = oldest, n-1 = today).
    With LOOKBACK=90 and today=Mar 18 2026:
      0–36  : mild bear / chop
      37–51 : CRASH (Jan 25 – Feb 8)
      52–74 : recovery
      75–83 : chop
      84–88 : pre-surge
      89    : today — forced +8.5% (strong up-day, matches P1D histogram)
    """
    regimes = [
        (37,  -0.004, 0.016),
        (52,  -0.025, 0.030),
        (75,  +0.009, 0.018),
        (84,  +0.002, 0.021),
        (n-1, +0.030, 0.020),
    ]
    f, i = np.zeros(n), 0
    for end, mu, sigma in regimes:
        while i < end:
            f[i] = np.random.normal(mu, sigma)
            i += 1
    f[-1] = 0.085   # today: hard +8.5%
    return f

mkt_factor   = build_market_factor(LOOKBACK)
sector_drift = {s: np.random.normal(0, 0.003, LOOKBACK) for s in SECTORS}

# ret_all[day, coin] = beta * market_factor[day] + sector_drift[day] + idio_noise
ret_all = np.zeros((LOOKBACK, N_COINS))
for c in range(N_COINS):
    sec = coin_sectors[c]
    ret_all[:, c] = (coin_betas[c] * mkt_factor
                     + sector_drift[sec]
                     + np.random.normal(0, coin_idio_vol[c], LOOKBACK))

# Compounded price index per coin, starts at 1.0
price_all = np.cumprod(1 + ret_all, axis=0)  # (LOOKBACK, N_COINS)


# ═══════════════════════════════════════════════════════════════════════════════
# 4 — DERIVED STATS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 4a · Return Histograms (P1D and P1W) ─────────────────────────────────────
# Bins: 8 buckets excluding the -5% to +5% flat zone.
# today_ret   = ret_all[-1]          (N_COINS,)  single-day returns
# week_ret    = price[-1]/price[-6]  (N_COINS,)  5-day compounded returns
_EDGES  = np.array([-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20])
_LABELS = ['-20%', '-15%', '-10%', '-5%', '5%', '10%', '15%', '20%']

def return_hist(ret_1d):
    cats   = np.digitize(ret_1d, _EDGES)
    counts = [(cats == bi).sum() for bi in [0, 1, 2, 3, 5, 6, 7, 8]]
    return pd.DataFrame({'Bin': _LABELS, 'Count': counts})

today_ret = ret_all[-1]
week_ret  = (price_all[-1] / price_all[-6]) - 1

df_p1d = return_hist(today_ret)
df_p1w = return_hist(week_ret)


# ── 4b · Sector Z-Scores (P1D and P1W) ───────────────────────────────────────
# For each sector: compute mean return for the period (today or this week).
# Z-score that mean against the sector's own 60-day daily mean distribution.
# → Positive Z means sector is outperforming its historical norm.
ZSCORE_WIN = 60

def sector_zscore(coin_ret_now, coin_ret_hist):
    rows = []
    for sec in SECTORS:
        mask       = coin_sectors == sec
        now_mean   = coin_ret_now[mask].mean()
        hist_means = coin_ret_hist[:, mask].mean(axis=1)
        mu, sigma  = hist_means.mean(), hist_means.std()
        z = (now_mean - mu) / max(float(sigma), 1e-9)
        rows.append({'Sector': sec, 'Z-Score': round(z, 2)})
    return (pd.DataFrame(rows)
              .sort_values('Z-Score', ascending=True)
              .reset_index(drop=True))

hist_d = ret_all[-ZSCORE_WIN-1:-1]  # 60 daily returns excluding today
hist_w = np.array([                 # 60 rolling 5-day returns
    (price_all[-ZSCORE_WIN-1+i] / price_all[-ZSCORE_WIN-6+i]) - 1
    for i in range(ZSCORE_WIN)
])

df_zp1d = sector_zscore(today_ret, hist_d)
df_zp1w = sector_zscore(week_ret,  hist_w)


# ── 4c · Setups P1W ───────────────────────────────────────────────────────────
# Pattern detection on the price matrix.
# Each setup is a binary flag per coin; counts are aggregated across the universe.
# Bullish setups → positive count, bearish setups → negative count.
def compute_setups(pm, rm, window=20):
    p0      = pm[-1]
    p1      = pm[-2]
    high20  = pm[-window-1:-1].max(axis=0)
    low20   = pm[-window-1:-1].min(axis=0)
    range20 = (high20 / np.maximum(low20, 1e-9)) - 1
    trend5  = rm[-5:].sum(axis=0)
    ret_d   = rm[-1]
    in_bo5  = pm[-6:-1].max(axis=0) > high20
    in_bd5  = pm[-6:-1].min(axis=0) < low20
    pfl     = (p0 - low20) / np.maximum(low20, 1e-9)

    breakout   = (p0 > high20) & (p1 <= high20)
    breakdown  = (p0 < low20)  & (p1 >= low20)
    contd_bo   = in_bo5 & (trend5 > 0) & ~breakout
    contd_sh   = in_bd5 & (trend5 < 0) & ~breakdown
    bot_bounce = (pfl < 0.05) & (p0 > p1)
    backside   = (pfl > 0.10) & (p0 < p1) & (trend5 < 0)
    epivot     = (range20 < 0.15) & (np.abs(ret_d) > 0.08)

    return pd.DataFrame({
        'Setup': ['Bottom Bounce','Contd. Breakout','Episodic Pivot',
                  'Contd. Short', 'Breakdown',      'Backside Short'],
        'Count': [
             int(bot_bounce.sum()),
             int(contd_bo.sum()),
             int(epivot.sum()),
            -int(contd_sh.sum()),
            -int(breakdown.sum()),
            -int(backside.sum()),
        ]
    })

df_setups = compute_setups(price_all, ret_all)


# ── 4d · Breadth Time Series ──────────────────────────────────────────────────
DISPLAY  = 60
d_dates  = dates_all[-DISPLAY:]
d_ret    = ret_all[-DISPLAY:]   # (60, N_COINS)

# PRIMARY: 25 highest-beta coins, ±5% daily threshold
# up5_p[d] = # of primary coins with daily return > +5% on day d
# dn5_p[d] = # of primary coins with daily return < -5% on day d
# ratio_Nd  = rolling N-day sum(up5) / sum(dn5)  → conviction measure
# osc_p     = up5 - dn5                          → net movers per day
up5_p = (d_ret[:, primary_idx] >  0.05).sum(axis=1).astype(float)
dn5_p = (d_ret[:, primary_idx] < -0.05).sum(axis=1).astype(float)

def rolling_ratio(up, dn, w):
    r = np.zeros(len(up))
    for i in range(len(up)):
        u = up[max(0, i-w+1):i+1].sum()
        d = dn[max(0, i-w+1):i+1].sum()
        r[i] = u / max(d, 0.5)
    return r

ratio_2d    = rolling_ratio(up5_p, dn5_p, 2)
ratio_5d    = rolling_ratio(up5_p, dn5_p, 5)
osc_primary = up5_p - dn5_p

# SECONDARY: full 289-coin universe, per-coin 1σ weekly threshold
# coin_weekly_sigma[c] = std of 5-day returns across the PRE-CRASH window (days 0-36)
# Using pre-crash vol as baseline so thresholds reflect "normal" conditions.
# Each day d: compute each coin's rolling 5-day return.
# up_1s = # coins above +1σ, dn_1s = # coins below -1σ
# osc_secondary = (up_1s - dn_1s) / N_COINS × 100  → scaled ±100
pre_crash_wk = np.array([
    (price_all[i] / price_all[i-5]) - 1
    for i in range(5, 37)
])  # (32, N_COINS)
coin_weekly_sigma = pre_crash_wk.std(axis=0)  # (N_COINS,)

osc_secondary = np.zeros(DISPLAY)
for d in range(DISPLAY):
    pm_idx = (LOOKBACK - DISPLAY) + d
    if pm_idx >= 5:
        w_ret = (price_all[pm_idx] / price_all[pm_idx-5]) - 1
        up_1s = (w_ret >  coin_weekly_sigma).sum()
        dn_1s = (w_ret < -coin_weekly_sigma).sum()
        osc_secondary[d] = (up_1s - dn_1s) / N_COINS * 100

df_ts = pd.DataFrame({
    'Date':          d_dates,
    '2_day':         ratio_2d,
    '5_day':         ratio_5d,
    'Osc_Primary':   osc_primary,
    'Osc_Secondary': osc_secondary,
})


# ── 4e · Calendar Heatmap ─────────────────────────────────────────────────────
# Cell value = median coin return for that day (from ret_all within LOOKBACK,
# plausible noise for dates older than LOOKBACK, NaN for future dates).
daily_median = np.median(ret_all, axis=1)  # (LOOKBACK,)

year       = _today.year
today_date = _today.date()
year_dates = pd.date_range(f'{year}-01-01', f'{year}-12-31')
first_mon  = year_dates[0] - timedelta(days=year_dates[0].dayofweek)

cal_rows = []
for yr_d in year_dates:
    days_ago = (today_date - yr_d.date()).days
    if 0 <= days_ago < LOOKBACK:
        val = float(daily_median[LOOKBACK - 1 - days_ago])
    elif days_ago >= LOOKBACK:
        val = float(np.random.normal(0, 0.008))
    else:
        val = np.nan
    cal_rows.append({
        'day':      yr_d.day,
        'dow':      yr_d.dayofweek,
        'month':    yr_d.month,
        'week_col': (yr_d.date() - first_mon.date()).days // 7,
        'val':      val,
    })
df_cal = pd.DataFrame(cal_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — LAYOUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def clean(fig, height=350):
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0), height=height,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10),
    )
    return fig

def bar_v(df, x_col, y_col, title):
    colors = [COLOR_NEG if '-' in str(x) else COLOR_POS for x in df[x_col]]
    fig = go.Figure(go.Bar(x=df[x_col], y=df[y_col],
                           marker_color=colors, text=df[y_col], textposition='outside'))
    fig.update_yaxes(range=[0, max(df[y_col].max() * 1.3, 1)])
    st.markdown(f"<p style='text-align:center;font-weight:bold;'>{title}</p>",
                unsafe_allow_html=True)
    st.plotly_chart(clean(fig), use_container_width=True, config={'displayModeBar': False})

def bar_h(df, x_col, y_col, title, abs_text=False):
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in df[x_col]]
    txt    = df[x_col].abs().round(1) if abs_text else df[x_col].round(1)
    xpad   = max(df[x_col].abs().max() * 0.28, 0.1)
    fig = go.Figure(go.Bar(x=df[x_col], y=df[y_col], orientation='h',
                           marker_color=colors, text=txt, textposition='outside'))
    fig.update_xaxes(range=[df[x_col].min() - xpad, df[x_col].max() + xpad])
    st.markdown(f"<p style='text-align:center;font-weight:bold;'>{title}</p>",
                unsafe_allow_html=True)
    st.plotly_chart(clean(fig), use_container_width=True, config={'displayModeBar': False})


# ═══════════════════════════════════════════════════════════════════════════════
# 6 — UI
# ═══════════════════════════════════════════════════════════════════════════════
c1, c2 = st.columns([9, 1])
c1.markdown("## Market Monitor")
c2.markdown(f"#### {_today.strftime('%Y-%m-%d')}")
st.divider()

# ── Row 1 ─────────────────────────────────────────────────────────────────────
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
with r1c1: bar_v(df_p1d,    'Bin',     'Count',  'P1D Returns')
with r1c2: bar_v(df_p1w,    'Bin',     'Count',  'P1W Returns')
with r1c3: bar_h(df_zp1d,   'Z-Score', 'Sector', 'Sector Z-Scores P1D')
with r1c4: bar_h(df_zp1w,   'Z-Score', 'Sector', 'Sector Z-Scores P1W')
with r1c5: bar_h(df_setups, 'Count',   'Setup',  'Setups P1W', abs_text=True)
st.write("")

# ── Row 2 ─────────────────────────────────────────────────────────────────────
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

# ── Row 3 · Calendar Heatmap ──────────────────────────────────────────────────
st.markdown("<p style='text-align:center;font-weight:bold;'>Daily Performance Heatmap</p>",
            unsafe_allow_html=True)

n_weeks  = int(df_cal['week_col'].max()) + 1
z_cal    = np.full((7, n_weeks), np.nan)
text_cal = np.full((7, n_weeks), '', dtype=object)

for _, row in df_cal.iterrows():
    r, c = int(row['dow']), int(row['week_col'])
    z_cal[r, c]    = row['val']
    text_cal[r, c] = str(int(row['day']))

MNAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_spans = df_cal.groupby('month')['week_col'].agg(['min','max'])
annotations = [
    dict(x=(sp['min']+sp['max'])/2, y=1.10, xref='x', yref='paper',
         text=f"<b>{MNAMES[m-1]}</b>", showarrow=False,
         font=dict(size=10), xanchor='center')
    for m, sp in month_spans.iterrows()
]

fig_cal = go.Figure(go.Heatmap(
    z=z_cal, x=list(range(n_weeks)), y=list(range(7)),
    text=text_cal, texttemplate="%{text}",
    textfont={"size": 9, "color": "black"},
    colorscale=[[0, COLOR_NEG], [0.5, '#f0f0f0'], [1, COLOR_POS]],
    showscale=False, xgap=3, ygap=3,
    zmin=-0.05, zmax=0.05, hoverinfo='none',
))
fig_cal.update_layout(
    height=240, margin=dict(l=35, r=10, t=45, b=10),
    yaxis=dict(ticktext=['M','T','W','T','F','S','S'], tickvals=list(range(7)),
               autorange='reversed', showgrid=False, zeroline=False,
               tickfont=dict(size=10)),
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    annotations=annotations,
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
)
st.plotly_chart(fig_cal, use_container_width=True, config={'displayModeBar': False})
