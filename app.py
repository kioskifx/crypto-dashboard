import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
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
SECTOR_DEF = [
    ('MEME',    30, 1.80), ('L1',      20, 1.00), ('DINO',    18, 0.70),
    ('AI',      25, 1.40), ('LST',     12, 0.90), ('NEW',     15, 1.20),
    ('CEXDEX',  15, 1.10), ('L2',      18, 1.20), ('NFT',     15, 1.50),
    ('NAR',     10, 0.80), ('DEFI',    20, 1.30), ('GAMING',  18, 1.60),
    ('DEPIN',   12, 1.30), ('PRIVACY', 10, 1.10), ('AGENTS',  18, 1.50),
    ('INFRA',   15, 1.10), ('DWF',      8, 1.00), ('INFOFI',  10, 1.20),
]
SECTORS      = [d[0] for d in SECTOR_DEF]
coin_sectors = np.array([s for name, n, _ in SECTOR_DEF for s in [name] * n])
coin_betas   = np.array([b for name, n, b in SECTOR_DEF for _ in range(n)])
N_COINS      = len(coin_sectors)  # 289

coin_idio_vol = np.random.uniform(0.020, 0.055, N_COINS)
primary_idx   = np.argsort(coin_betas)[-25:]


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — SHORT RETURN MATRIX  (90 days — basis for Row 1 charts)
# ═══════════════════════════════════════════════════════════════════════════════
LOOKBACK  = 90
DISPLAY   = 60
dates_all = np.array([_today - timedelta(days=x) for x in range(LOOKBACK - 1, -1, -1)])

def build_market_factor(n, crash_offset=52):
    """
    Regime-structured daily market factor.
    crash_offset = how many days before today the crash STARTS.
    Crash duration = 15 days.
    """
    cs = n - crash_offset          # crash start index
    ce = cs + 15                   # crash end index
    regimes = [
        (cs,   -0.004, 0.016),     # mild bear leading into crash
        (ce,   -0.025, 0.030),     # CRASH
        (ce+23,+0.009, 0.018),     # recovery
        (ce+32,+0.002, 0.021),     # chop
        (n-1,  +0.030, 0.020),     # surge
    ]
    f, i = np.zeros(n), 0
    for end, mu, sigma in regimes:
        end = min(end, n - 1)
        while i < end:
            f[i] = np.random.normal(mu, sigma)
            i += 1
    f[-1] = 0.085   # today: forced +8.5%
    return f

mkt_factor   = build_market_factor(LOOKBACK)
sector_drift = {s: np.random.normal(0, 0.003, LOOKBACK) for s in SECTORS}

ret_all = np.zeros((LOOKBACK, N_COINS))
for c in range(N_COINS):
    sec = coin_sectors[c]
    ret_all[:, c] = (coin_betas[c] * mkt_factor
                     + sector_drift[sec]
                     + np.random.normal(0, coin_idio_vol[c], LOOKBACK))

price_all = np.cumprod(1 + ret_all, axis=0)   # (90, 289)


# ═══════════════════════════════════════════════════════════════════════════════
# 4a — RETURN HISTOGRAMS
# ═══════════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════════
# 4b — SECTOR Z-SCORES
# ═══════════════════════════════════════════════════════════════════════════════
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
    return pd.DataFrame(rows).sort_values('Z-Score', ascending=True).reset_index(drop=True)

hist_d  = ret_all[-ZSCORE_WIN-1:-1]
hist_w  = np.array([(price_all[-ZSCORE_WIN-1+i] / price_all[-ZSCORE_WIN-6+i]) - 1
                     for i in range(ZSCORE_WIN)])
df_zp1d = sector_zscore(today_ret, hist_d)
df_zp1w = sector_zscore(week_ret,  hist_w)


# ═══════════════════════════════════════════════════════════════════════════════
# 4c — SETUPS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_setups(pm, rm, window=20):
    p0, p1  = pm[-1], pm[-2]
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
        'Count': [int(bot_bounce.sum()), int(contd_bo.sum()), int(epivot.sum()),
                  -int(contd_sh.sum()), -int(breakdown.sum()), -int(backside.sum())]
    })

df_setups = compute_setups(price_all, ret_all)


# ═══════════════════════════════════════════════════════════════════════════════
# 4d — BREADTH TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════
d_dates = dates_all[-DISPLAY:]
d_ret   = ret_all[-DISPLAY:]

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

pre_crash_wk      = np.array([(price_all[i] / price_all[i-5]) - 1 for i in range(5, 37)])
coin_weekly_sigma = pre_crash_wk.std(axis=0)

osc_secondary = np.zeros(DISPLAY)
for d in range(DISPLAY):
    pm_idx = (LOOKBACK - DISPLAY) + d
    if pm_idx >= 5:
        w_ret = (price_all[pm_idx] / price_all[pm_idx-5]) - 1
        osc_secondary[d] = ((w_ret > coin_weekly_sigma).sum()
                           - (w_ret < -coin_weekly_sigma).sum()) / N_COINS * 100

df_ts = pd.DataFrame({'Date': d_dates, '2_day': ratio_2d, '5_day': ratio_5d,
                      'Osc_Primary': osc_primary, 'Osc_Secondary': osc_secondary})


# ═══════════════════════════════════════════════════════════════════════════════
# 4e — CALENDAR HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
daily_median = np.median(ret_all, axis=1)
year         = _today.year
today_date   = _today.date()
year_dates   = pd.date_range(f'{year}-01-01', f'{year}-12-31')
first_mon    = year_dates[0] - timedelta(days=year_dates[0].dayofweek)

cal_rows = []
for yr_d in year_dates:
    days_ago  = (today_date - yr_d.date()).days
    is_future = days_ago < 0
    if 0 <= days_ago < LOOKBACK:
        val = float(daily_median[LOOKBACK - 1 - days_ago])
    elif days_ago >= LOOKBACK:
        val = float(np.random.normal(0, 0.008))
    else:
        val = np.nan
    cal_rows.append({'day': yr_d.day, 'dow': yr_d.dayofweek, 'month': yr_d.month,
                     'week_col': (yr_d.date() - first_mon.date()).days // 7,
                     'val': val, 'future': is_future})
df_cal = pd.DataFrame(cal_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 4f — TREND BLOCK  (BTC / ETH / SOL vs 20d / 50d / 200d SMA — live Binance)
# ═══════════════════════════════════════════════════════════════════════════════
# Binance public klines endpoint — no API key required.
# 260 daily closes give enough runway for a clean 200d SMA.
# TOTAL3 is a TradingView index with no direct exchange pair; replaced with SOL
# as the most liquid large-cap alt proxy.
# ttl=300 → refreshes at most every 5 minutes to avoid rate limits.

LONG_LB = 260

@st.cache_data(ttl=300)
def fetch_closes(symbol: str, limit: int = 260) -> np.ndarray | None:
    """
    Fetch daily close prices from Binance spot klines.
    Returns a float64 numpy array of length `limit` (oldest first),
    or None if the request fails.
    """
    url = "https://api.binance.com/api/v3/klines"
    try:
        r = requests.get(url, params={"symbol": symbol, "interval": "1d", "limit": limit},
                         timeout=10)
        r.raise_for_status()
        closes = np.array([float(row[4]) for row in r.json()], dtype=np.float64)
        return closes if len(closes) >= 200 else None
    except Exception:
        return None

btc_px = fetch_closes("BTCUSDT", LONG_LB)
eth_px = fetch_closes("ETHUSDT", LONG_LB)
sol_px = fetch_closes("SOLUSDT", LONG_LB)

# Fallback: if any fetch fails, mark as unavailable (handled in render)
_fetch_ok = {
    "BTC":  btc_px is not None,
    "ETH":  eth_px is not None,
    "SOL":  sol_px is not None,
}

def sma(px: np.ndarray, w: int) -> float:
    return float(px[-w:].mean())

def ma_vs(px: np.ndarray, w: int):
    """(display_string, colour) of current price vs SMA(w)."""
    ma = sma(px, w)
    r  = px[-1] / ma - 1
    s  = f'+{r*100:.1f}%' if r >= 0 else f'{r*100:.1f}%'
    if   r >  0.015: c = COLOR_POS
    elif r < -0.015: c = COLOR_NEG
    else:            c = '#E8E060'
    return s, c

def regime(px: np.ndarray):
    """
    Regime from MA alignment:
      Bull       — price > 50d AND 50d > 200d
      Recovery   — price > 50d AND 50d < 200d  (bounced but below 200d)
      Correction — price < 50d AND price > 200d
      Bear       — price < 50d AND price < 200d
    """
    p    = px[-1]
    m50  = sma(px, 50)
    m200 = sma(px, 200)
    if   p > m50 and m50 > m200: return 'Bull',       '#4CAF50'
    elif p > m50:                 return 'Recovery',   '#E8E060'
    elif p > m200:                return 'Correction', COLOR_NEG
    else:                         return 'Bear',       '#cc3333'

trend_data = []
for label, px in [('BTC', btc_px), ('ETH', eth_px), ('SOL', sol_px)]:
    if px is None:
        trend_data.append({'name': label,
                           'vs20': ('N/A', '#888'), 'vs50': ('N/A', '#888'),
                           'vs200': ('N/A', '#888'), 'regime': ('Unavailable', '#888')})
        continue
    v20, c20   = ma_vs(px, 20)
    v50, c50   = ma_vs(px, 50)
    v200, c200 = ma_vs(px, 200)
    reg, rc    = regime(px)
    trend_data.append({'name': label,
                       'vs20':  (v20, c20),  'vs50':  (v50, c50),
                       'vs200': (v200, c200), 'regime': (reg, rc)})


# ═══════════════════════════════════════════════════════════════════════════════
# 4g — MA BREADTH  (% of 289 coins above 20d / 50d / 200d SMA, 60-day history)
# ═══════════════════════════════════════════════════════════════════════════════
# Build a full (260, 289) coin price matrix using the same regime factor logic
# as the short matrix, extended backwards by (LONG_LB - LOOKBACK) days.
# f_long_coins is independent of the live BTC/ETH fetch — this is the simulated
# alt-coin universe and does not need real price data.
np.random.seed(_today.day + 77)

def _build_coin_long_factor(n):
    cs = n - 52
    ce = cs + 15
    boundaries = [
        (int(n * 0.20), +0.0012, 0.013),
        (int(n * 0.38), +0.0001, 0.017),
        (int(n * 0.52), -0.0012, 0.016),
        (cs,            -0.0004, 0.014),
        (ce,            -0.018,  0.029),
        (ce + 23,       +0.002,  0.018),
        (ce + 32,       +0.0008, 0.021),
        (n - 1,         +0.004,  0.019),
    ]
    f, i = np.zeros(n), 0
    for end, mu, sigma in boundaries:
        end = min(int(end), n - 1)
        while i < end:
            f[i] = np.random.normal(mu, sigma)
            i += 1
    f[-1] = 0.085
    return f

f_long_coins = _build_coin_long_factor(LONG_LB)

sd_long = {}
for s in SECTORS:
    prefix     = np.random.normal(0, 0.003, LONG_LB - LOOKBACK)
    sd_long[s] = np.concatenate([prefix, sector_drift[s]])

_tmp_ret = np.zeros((LONG_LB, N_COINS))
for c in range(N_COINS):
    sec            = coin_sectors[c]
    idio           = np.random.normal(0, coin_idio_vol[c], LONG_LB)
    _tmp_ret[:, c] = coin_betas[c] * f_long_coins + sd_long[sec] + idio
px_long = np.cumprod(1 + _tmp_ret, axis=0)   # (260, 289)

def pct_above_sma(px_matrix, n_display, window):
    """
    For each of the last n_display days, compute:
        % of coins whose current price > their own trailing SMA(window)
    Returns array of length n_display (values 0–100).
    """
    n    = px_matrix.shape[0]
    result = np.full(n_display, np.nan)
    for d in range(n_display):
        day_idx = n - n_display + d
        if day_idx < window:
            continue
        curr_px = px_matrix[day_idx]                         # (N_COINS,)
        ma_px   = px_matrix[day_idx - window + 1: day_idx + 1].mean(axis=0)
        result[d] = (curr_px > ma_px).mean() * 100
    return result

pct_20  = pct_above_sma(px_long, DISPLAY, 20)
pct_50  = pct_above_sma(px_long, DISPLAY, 50)
pct_200 = pct_above_sma(px_long, DISPLAY, 200)


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — LAYOUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def clean(fig, height=350):
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=height,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(size=10))
    return fig

def bar_v(df, x_col, y_col, title):
    colors = [COLOR_NEG if '-' in str(x) else COLOR_POS for x in df[x_col]]
    fig = go.Figure(go.Bar(x=df[x_col], y=df[y_col], marker_color=colors,
                           text=df[y_col], textposition='outside'))
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

def trend_table_html(data):
    """Compact HTML table for the Structure block."""
    def td(v, c, bold=False):
        fw = 'bold' if bold else 'normal'
        return f'<td style="padding:5px 14px;color:{c};font-weight:{fw};">{v}</td>'

    header = '''
    <tr style="border-bottom:1px solid #dddddd;">
        <th style="padding:5px 14px;text-align:left;color:#888;font-weight:normal;">Asset</th>
        <th style="padding:5px 14px;text-align:left;color:#888;font-weight:normal;">vs 20d SMA</th>
        <th style="padding:5px 14px;text-align:left;color:#888;font-weight:normal;">vs 50d SMA</th>
        <th style="padding:5px 14px;text-align:left;color:#888;font-weight:normal;">vs 200d SMA</th>
        <th style="padding:5px 14px;text-align:left;color:#888;font-weight:normal;">Regime</th>
    </tr>'''

    rows = ''
    for d in data:
        rows += f'''<tr>
            {td(d["name"], "inherit", bold=True)}
            {td(d["vs20"][0],   d["vs20"][1])}
            {td(d["vs50"][0],   d["vs50"][1])}
            {td(d["vs200"][0],  d["vs200"][1])}
            {td(d["regime"][0], d["regime"][1], bold=True)}
        </tr>'''

    return f'''
    <table style="width:auto;border-collapse:collapse;font-size:13px;margin-top:4px;">
        <thead>{header}</thead>
        <tbody>{rows}</tbody>
    </table>'''


# ═══════════════════════════════════════════════════════════════════════════════
# 6 — UI
# ═══════════════════════════════════════════════════════════════════════════════
c1, c2 = st.columns([9, 1])
c1.markdown("## Market Monitor")
c2.markdown(f"#### {_today.strftime('%Y-%m-%d')}")
st.divider()

# ── Row 1 · Returns / Z-Scores / Setups ──────────────────────────────────────
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
with r1c1: bar_v(df_p1d,    'Bin',     'Count',  'P1D Returns')
with r1c2: bar_v(df_p1w,    'Bin',     'Count',  'P1W Returns')
with r1c3: bar_h(df_zp1d,   'Z-Score', 'Sector', 'Sector Z-Scores P1D')
with r1c4: bar_h(df_zp1w,   'Z-Score', 'Sector', 'Sector Z-Scores P1W')
with r1c5: bar_h(df_setups, 'Count',   'Setup',  'Setups P1W', abs_text=True)

st.write("")

# ── Row 1.5 · Structure — BTC / ETH / TOTAL3 vs MAs ─────────────────────────
st.markdown("<p style='font-weight:bold;font-size:14px;'>Structure</p>",
            unsafe_allow_html=True)
st.markdown(trend_table_html(trend_data), unsafe_allow_html=True)
st.write("")

# ── Row 2 · Breadth Charts ───────────────────────────────────────────────────
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

# ── Row 2.5 · MA Breadth ─────────────────────────────────────────────────────
# % of 289 coins trading above each MA — structural health indicator.
# The three timeframes tell different stories:
#   20d = short-term momentum (fast, responsive)
#   50d = intermediate trend participation
#   200d = long-term structural health (slow, sticky)
# 50% is the key threshold: above = majority in uptrend, below = majority not.

COLOR_50  = '#F5C842'   # 50d line
COLOR_200 = '#B57EF5'   # 200d line

r25c1, r25c2, r25c3 = st.columns(3)

for col, series, color, title, ma_label in [
    (r25c1, pct_20,  COLOR_POS, '% Coins Above 20d SMA', '20d'),
    (r25c2, pct_50,  COLOR_50,  '% Coins Above 50d SMA', '50d'),
    (r25c3, pct_200, COLOR_200, '% Coins Above 200d SMA', '200d'),
]:
    with col:
        st.markdown(f"<p style='text-align:center;font-weight:bold;'>{title}</p>",
                    unsafe_allow_html=True)
        fig = go.Figure()
        # Colour the area under the line: positive (>50) = blue, negative (<50) = orange
        above = np.where(series >= 50, series, 50.0)
        below = np.where(series <= 50, series, 50.0)
        fig.add_trace(go.Scatter(
            x=d_dates, y=above, fill='tozeroy', fillcolor='rgba(165,214,247,0.25)',
            line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=d_dates, y=below, fill='tozeroy', fillcolor='rgba(246,185,149,0.25)',
            line=dict(width=0), showlegend=False, hoverinfo='skip'))
        # Main line
        fig.add_trace(go.Scatter(
            x=d_dates, y=series, mode='lines', name=ma_label,
            line=dict(color=color, width=1.8), showlegend=False))
        # 50% reference line
        fig.add_hline(y=50, line=dict(color='#aaaaaa', width=1, dash='dot'))
        fig.update_yaxes(range=[0, 100], ticksuffix='%')
        st.plotly_chart(clean(fig, 280), use_container_width=True,
                        config={'displayModeBar': False})

st.write("")

# ── Row 3 · Calendar Heatmap ──────────────────────────────────────────────────
st.markdown("<p style='text-align:center;font-weight:bold;'>Daily Performance Heatmap</p>",
            unsafe_allow_html=True)

n_weeks  = int(df_cal['week_col'].max()) + 1
z_past   = np.full((7, n_weeks), np.nan)
z_future = np.full((7, n_weeks), np.nan)
text_cal = np.full((7, n_weeks), '', dtype=object)

for _, row in df_cal.iterrows():
    r, c = int(row['dow']), int(row['week_col'])
    text_cal[r, c] = str(int(row['day']))
    if row['future']:
        z_future[r, c] = 0.0
    else:
        z_past[r, c] = row['val']

MNAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_spans = df_cal.groupby('month')['week_col'].agg(['min','max'])

annotations = [
    dict(x=(sp['min']+sp['max'])/2, y=1.08, xref='x', yref='paper',
         text=f"<b>{MNAMES[m-1]}</b>", showarrow=False,
         font=dict(size=11, color='#cccccc'), xanchor='center')
    for m, sp in month_spans.iterrows()
]

month_separators = []
for m, sp in month_spans.iterrows():
    if m == df_cal['month'].min():
        continue
    x_sep = sp['min'] - 0.5
    month_separators.append(dict(
        type='line', x0=x_sep, x1=x_sep, y0=-0.5, y1=6.5,
        xref='x', yref='y', line=dict(color='#555555', width=1.5)))

fig_cal = go.Figure()
fig_cal.add_trace(go.Heatmap(
    z=z_future, x=list(range(n_weeks)), y=list(range(7)),
    text=text_cal, texttemplate="%{text}", textfont={"size": 9, "color": "#666666"},
    colorscale=[[0, '#2a2a2a'], [1, '#2a2a2a']],
    showscale=False, xgap=3, ygap=3, zmin=0, zmax=1, hoverinfo='none'))
fig_cal.add_trace(go.Heatmap(
    z=z_past, x=list(range(n_weeks)), y=list(range(7)),
    text=text_cal, texttemplate="%{text}", textfont={"size": 9, "color": "#111111"},
    colorscale=[[0, COLOR_NEG], [0.5, '#e8e8e8'], [1, COLOR_POS]],
    showscale=False, xgap=3, ygap=3, zmin=-0.05, zmax=0.05, hoverinfo='none'))

fig_cal.update_layout(
    height=250, margin=dict(l=35, r=10, t=50, b=10),
    paper_bgcolor='#111111', plot_bgcolor='#111111',
    yaxis=dict(ticktext=['M','T','W','T','F','S','S'], tickvals=list(range(7)),
               autorange='reversed', showgrid=False, zeroline=False,
               tickfont=dict(size=10, color='#aaaaaa')),
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
               range=[-0.5, n_weeks - 0.5]),
    annotations=annotations, shapes=month_separators)
st.plotly_chart(fig_cal, use_container_width=True, config={'displayModeBar': False})
