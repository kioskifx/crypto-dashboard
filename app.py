import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — REAL COIN UNIVERSE  (named symbols → sector)
# ═══════════════════════════════════════════════════════════════════════════════
# All Binance USDT perpetual / spot symbols.
# Sectors kept aligned with the original 18-sector taxonomy.
# Coins not listed on Binance are excluded; missing fetches are handled gracefully.
COIN_MAP = {
    # MEME
    'DOGEUSDT':'MEME', 'SHIBUSDT':'MEME', 'PEPEUSDT':'MEME',
    'FLOKIUSDT':'MEME', 'BONKUSDT':'MEME', 'WIFUSDT':'MEME', 'MEMEUSDT':'MEME',
    # L1
    'SOLUSDT':'L1',  'ADAUSDT':'L1',  'AVAXUSDT':'L1', 'SUIUSDT':'L1',
    'NEARUSDT':'L1', 'APTUSDT':'L1',  'ATOMUSDT':'L1', 'ICPUSDT':'L1',
    'ALGOUSDT':'L1', 'DOTUSDT':'L1',
    # DINO  (legacy coins)
    'XRPUSDT':'DINO', 'LTCUSDT':'DINO', 'BCHUSDT':'DINO', 'XLMUSDT':'DINO',
    'ETCUSDT':'DINO', 'DASHUSDT':'DINO', 'EOSUSDT':'DINO', 'TRXUSDT':'DINO',
    # AI
    'FETUSDT':'AI', 'RENDERUSDT':'AI', 'WLDUSDT':'AI', 'OCEANUSDT':'AI',
    'AGIXUSDT':'AI', 'NMRUSDT':'AI',  'ALTUSDT':'AI', 'TAOUSDT':'AI',
    # LST  (liquid staking)
    'LDOUSDT':'LST', 'ANKRUSDT':'LST', 'SSVUSDT':'LST',
    # NEW  (recent launches / airdrop tokens)
    'JUPUSDT':'NEW', 'STRKUSDT':'NEW', 'EIGENUSDT':'NEW',
    'ZKUSDT':'NEW',  'WUSDT':'NEW',
    # CEXDEX
    'BNBUSDT':'CEXDEX', 'GMXUSDT':'CEXDEX', 'DYDXUSDT':'CEXDEX',
    # L2
    'ARBUSDT':'L2', 'OPUSDT':'L2', 'MATICUSDT':'L2', 'MNTUSDT':'L2',
    'IMXUSDT':'L2', 'METISUSDT':'L2',
    # NFT
    'APEUSDT':'NFT', 'BLURUSDT':'NFT', 'SANDUSDT':'NFT',
    'MANAUSDT':'NFT', 'ENJUSDT':'NFT',
    # NAR  (narrative / large-cap misc)
    'LINKUSDT':'NAR', 'UNIUSDT':'NAR', 'AAVEUSDT':'NAR', 'MKRUSDT':'NAR',
    # DEFI
    'COMPUSDT':'DEFI', 'CRVUSDT':'DEFI', 'SNXUSDT':'DEFI',
    'SUSHIUSDT':'DEFI', '1INCHUSDT':'DEFI', 'YFIUSDT':'DEFI',
    # GAMING
    'AXSUSDT':'GAMING', 'ILVUSDT':'GAMING', 'MAGICUSDT':'GAMING',
    'RONUSDT':'GAMING', 'GALAUSDT':'GAMING',
    # DEPIN
    'FILUSDT':'DEPIN', 'STORJUSDT':'DEPIN', 'ARUSDT':'DEPIN',
    'GRTUSDT':'DEPIN', 'HNTUSDT':'DEPIN',
    # PRIVACY
    'ZECUSDT':'PRIVACY', 'ROSEUSDT':'PRIVACY', 'SCRTUSDT':'PRIVACY',
    # AGENTS
    'VIRTUALUSDT':'AGENTS', 'AIUSDT':'AGENTS',
    # INFRA
    'API3USDT':'INFRA', 'TIAUSDT':'INFRA', 'BANDUSDT':'INFRA', 'CELOUSDT':'INFRA',
    # DWF  (DWF Labs portfolio coins)
    'SYNUSDT':'DWF', 'ORDIUSDT':'DWF', 'SATSUSDT':'DWF',
    # INFOFI
    'PYTHUSDT':'INFOFI', 'STXUSDT':'INFOFI', 'BIGTIMEUSDT':'INFOFI',
}

SECTORS    = sorted(set(COIN_MAP.values()))
ALL_SYMS   = list(COIN_MAP.keys())
LONG_LB    = 260   # days of history — enough for 200d SMA
DISPLAY    = 60    # days shown in time-series charts


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — LIVE DATA FETCH  (Binance public klines, no API key)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def fetch_all_ohlcv(symbols: tuple, limit: int = 260) -> dict:
    """
    Concurrent fetch of daily close prices from Binance spot klines.
    Returns {symbol: np.ndarray of closes (oldest→newest)} for every symbol
    that responded successfully with ≥ 21 bars.
    Uses ThreadPoolExecutor so all requests fire in parallel; typical wall-clock
    time ~1-2 s for 75 coins on a normal connection.
    """
    url = "https://api.binance.com/api/v3/klines"

    def _fetch_one(sym):
        try:
            r = requests.get(url,
                             params={"symbol": sym, "interval": "1d", "limit": limit},
                             timeout=10)
            r.raise_for_status()
            closes = np.array([float(row[4]) for row in r.json()], dtype=np.float64)
            return sym, closes if len(closes) >= 21 else None
        except Exception:
            return sym, None

    result = {}
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(_fetch_one, s): s for s in symbols}
        for f in as_completed(futures):
            sym, closes = f.result()
            if closes is not None:
                result[sym] = closes
    return result

with st.spinner("Fetching live market data…"):
    raw = fetch_all_ohlcv(tuple(ALL_SYMS), limit=LONG_LB)

# Map symbol → sector, only for coins that fetched successfully
fetched_syms    = [s for s in ALL_SYMS if s in raw]
coin_sector_arr = np.array([COIN_MAP[s] for s in fetched_syms])
N_COINS         = len(fetched_syms)

# Align all price arrays to the same length (use the minimum available)
min_len  = min(len(raw[s]) for s in fetched_syms)
LONG_LB  = min_len                          # actual history available

# price_all: (LONG_LB, N_COINS), each column = one coin's daily closes
price_all = np.column_stack([raw[s][-LONG_LB:] for s in fetched_syms])

# ret_all: daily log-return (ln(p_t / p_{t-1})), (LONG_LB-1, N_COINS)
# Using log returns for numerical stability; all downstream math works the same.
ret_all  = np.diff(np.log(price_all), axis=0)   # (LONG_LB-1, N_COINS)
price_all = price_all[1:]                        # align: (LONG_LB-1, N_COINS)
LONG_LB  -= 1

dates_all = np.array([_today - timedelta(days=x)
                       for x in range(LONG_LB - 1, -1, -1)])


# ═══════════════════════════════════════════════════════════════════════════════
# 4a — RETURN HISTOGRAMS  (P1D and P1W)
# ═══════════════════════════════════════════════════════════════════════════════
_EDGES  = np.array([-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20])
_LABELS = ['-20%', '-15%', '-10%', '-5%', '5%', '10%', '15%', '20%']

def return_hist(ret_1d):
    cats   = np.digitize(ret_1d, _EDGES)
    counts = [(cats == bi).sum() for bi in [0, 1, 2, 3, 5, 6, 7, 8]]
    return pd.DataFrame({'Bin': _LABELS, 'Count': counts})

today_ret = ret_all[-1]                                      # (N_COINS,)
week_ret  = np.log(price_all[-1] / price_all[-6])           # 5-day log return

df_p1d = return_hist(today_ret)
df_p1w = return_hist(week_ret)


# ═══════════════════════════════════════════════════════════════════════════════
# 4b — SECTOR Z-SCORES  (P1D and P1W)
# ═══════════════════════════════════════════════════════════════════════════════
# For each sector: compute mean log-return for the period.
# Z-score against that sector's own 60-day rolling daily mean distribution.
ZSCORE_WIN = 60

def sector_zscore(coin_ret_now, coin_ret_hist):
    rows = []
    for sec in SECTORS:
        mask = coin_sector_arr == sec
        if mask.sum() == 0:
            continue
        now_mean   = coin_ret_now[mask].mean()
        hist_means = coin_ret_hist[:, mask].mean(axis=1)
        mu, sigma  = hist_means.mean(), hist_means.std()
        z = (now_mean - mu) / max(float(sigma), 1e-9)
        rows.append({'Sector': sec, 'Z-Score': round(z, 2)})
    return (pd.DataFrame(rows)
              .sort_values('Z-Score', ascending=True)
              .reset_index(drop=True))

hist_d  = ret_all[-ZSCORE_WIN-1:-1]          # 60 daily returns exc. today
# 60 rolling 5-day returns (log)
hist_w  = np.array([
    np.log(price_all[-ZSCORE_WIN-1+i] / price_all[-ZSCORE_WIN-6+i])
    for i in range(ZSCORE_WIN)
])

df_zp1d = sector_zscore(today_ret, hist_d)
df_zp1w = sector_zscore(week_ret,  hist_w)


# ═══════════════════════════════════════════════════════════════════════════════
# 4c — SETUPS P1W
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
# Primary universe: coins with the highest average absolute daily return over
# the last 60 days (i.e. the most volatile / high-beta subset of whatever fetched).
# We pick the top 25 as the "primary" universe.
LOOKBACK = min(90, LONG_LB)
d_dates  = dates_all[-DISPLAY:]
d_ret    = ret_all[-DISPLAY:]       # (DISPLAY, N_COINS)
d_px     = price_all[-DISPLAY:]

avg_abs_ret    = np.abs(ret_all[-LOOKBACK:]).mean(axis=0)
primary_idx    = np.argsort(avg_abs_ret)[-25:]

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

# Secondary breadth: full universe, per-coin 1σ weekly threshold.
# Baseline volatility = std of 5-day returns in the first 60 days of our window
# (oldest available, before any recent crash/surge distorts the threshold).
baseline_wk = np.array([
    np.log(price_all[i] / price_all[i-5])
    for i in range(5, min(65, LONG_LB))
])   # (≤60, N_COINS)
coin_weekly_sigma = baseline_wk.std(axis=0)
coin_weekly_sigma = np.maximum(coin_weekly_sigma, 0.01)   # floor at 1%

osc_secondary = np.zeros(DISPLAY)
for d in range(DISPLAY):
    pm_idx = (LONG_LB - DISPLAY) + d
    if pm_idx >= 5:
        w_ret = np.log(price_all[pm_idx] / price_all[pm_idx-5])
        osc_secondary[d] = ((w_ret > coin_weekly_sigma).sum()
                           - (w_ret < -coin_weekly_sigma).sum()) / N_COINS * 100

df_ts = pd.DataFrame({
    'Date': d_dates, '2_day': ratio_2d, '5_day': ratio_5d,
    'Osc_Primary': osc_primary, 'Osc_Secondary': osc_secondary,
})


# ═══════════════════════════════════════════════════════════════════════════════
# 4e — CALENDAR HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
# Daily median log-return across all fetched coins.
daily_median = np.median(ret_all, axis=1)   # (LONG_LB,)

year       = _today.year
today_date = _today.date()
year_dates = pd.date_range(f'{year}-01-01', f'{year}-12-31')
first_mon  = year_dates[0] - timedelta(days=year_dates[0].dayofweek)

cal_rows = []
for yr_d in year_dates:
    days_ago  = (today_date - yr_d.date()).days
    is_future = days_ago < 0
    if 0 <= days_ago < LONG_LB:
        val = float(daily_median[LONG_LB - 1 - days_ago])
    else:
        val = np.nan   # beyond our history window or future → grey cell
    cal_rows.append({
        'day':      yr_d.day,
        'dow':      yr_d.dayofweek,
        'month':    yr_d.month,
        'week_col': (yr_d.date() - first_mon.date()).days // 7,
        'val':      val,
        'future':   is_future,
    })
df_cal = pd.DataFrame(cal_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 4f — STRUCTURE  (BTC / ETH / SOL vs 20d / 50d / 200d SMA — live Binance)
# ═══════════════════════════════════════════════════════════════════════════════
# BTC, ETH, SOL are already in raw (fetched above). Use their price arrays directly.

def sma(px, w):
    return float(px[-w:].mean())

def ma_vs(px, w):
    ma = sma(px, w)
    r  = px[-1] / ma - 1
    s  = f'+{r*100:.1f}%' if r >= 0 else f'{r*100:.1f}%'
    if   r >  0.015: c = COLOR_POS
    elif r < -0.015: c = COLOR_NEG
    else:            c = '#E8E060'
    return s, c

def regime(px):
    p    = px[-1]
    m50  = sma(px, 50)
    m200 = sma(px, 200)
    if   p > m50 and m50 > m200: return 'Bull',       '#4CAF50'
    elif p > m50:                 return 'Recovery',   '#E8E060'
    elif p > m200:                return 'Correction', COLOR_NEG
    else:                         return 'Bear',       '#cc3333'

trend_data = []
for label, sym in [('BTC', 'BTCUSDT'), ('ETH', 'ETHUSDT'), ('SOL', 'SOLUSDT')]:
    px = raw.get(sym)
    if px is None or len(px) < 200:
        trend_data.append({'name': label,
                           'vs20': ('N/A','#888'), 'vs50': ('N/A','#888'),
                           'vs200': ('N/A','#888'), 'regime': ('Unavailable','#888')})
        continue
    v20, c20   = ma_vs(px, 20)
    v50, c50   = ma_vs(px, 50)
    v200, c200 = ma_vs(px, 200)
    reg, rc    = regime(px)
    trend_data.append({'name': label,
                       'vs20': (v20,c20), 'vs50': (v50,c50),
                       'vs200': (v200,c200), 'regime': (reg,rc)})


# ═══════════════════════════════════════════════════════════════════════════════
# 4g — MA BREADTH  (% coins above 20d / 50d / 200d SMA over last 60 days)
# ═══════════════════════════════════════════════════════════════════════════════
def pct_above_sma(px_matrix, n_display, window):
    n = px_matrix.shape[0]
    result = np.full(n_display, np.nan)
    for d in range(n_display):
        day_idx = n - n_display + d
        if day_idx < window:
            continue
        curr_px = px_matrix[day_idx]
        ma_px   = px_matrix[day_idx - window + 1 : day_idx + 1].mean(axis=0)
        result[d] = (curr_px > ma_px).mean() * 100
    return result

pct_20  = pct_above_sma(price_all, DISPLAY, 20)
pct_50  = pct_above_sma(price_all, DISPLAY, 50)
pct_200 = pct_above_sma(price_all, DISPLAY, 200)


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
    def td(v, c, bold=False):
        fw = 'bold' if bold else 'normal'
        return f'<td style="padding:5px 14px;color:{c};font-weight:{fw};">{v}</td>'
    header = '''<tr style="border-bottom:1px solid #dddddd;">
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
            {td(d["vs20"][0], d["vs20"][1])}
            {td(d["vs50"][0], d["vs50"][1])}
            {td(d["vs200"][0], d["vs200"][1])}
            {td(d["regime"][0], d["regime"][1], bold=True)}
        </tr>'''
    return f'<table style="width:auto;border-collapse:collapse;font-size:13px;margin-top:4px;"><thead>{header}</thead><tbody>{rows}</tbody></table>'


# ═══════════════════════════════════════════════════════════════════════════════
# 6 — UI
# ═══════════════════════════════════════════════════════════════════════════════
c1, c2 = st.columns([9, 1])
c1.markdown("## Market Monitor")
c2.markdown(f"#### {_today.strftime('%Y-%m-%d')}")
st.caption(f"Universe: {N_COINS} coins across {len(SECTORS)} sectors · data via Binance · refreshes every 5 min")
st.divider()

# ── Row 1 · Returns / Z-Scores / Setups ──────────────────────────────────────
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
with r1c1: bar_v(df_p1d,    'Bin',     'Count',  'P1D Returns')
with r1c2: bar_v(df_p1w,    'Bin',     'Count',  'P1W Returns')
with r1c3: bar_h(df_zp1d,   'Z-Score', 'Sector', 'Sector Z-Scores P1D')
with r1c4: bar_h(df_zp1w,   'Z-Score', 'Sector', 'Sector Z-Scores P1W')
with r1c5: bar_h(df_setups, 'Count',   'Setup',  'Setups P1W', abs_text=True)
st.write("")

# ── Row 1.5 · Structure ───────────────────────────────────────────────────────
st.markdown("<p style='font-weight:bold;font-size:14px;'>Structure</p>",
            unsafe_allow_html=True)
st.markdown(trend_table_html(trend_data), unsafe_allow_html=True)
st.write("")

# ── Row 2 · Breadth ───────────────────────────────────────────────────────────
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
COLOR_50  = '#F5C842'
COLOR_200 = '#B57EF5'

r25c1, r25c2, r25c3 = st.columns(3)
for col, series, color, title in [
    (r25c1, pct_20,  COLOR_POS, '% Coins Above 20d SMA'),
    (r25c2, pct_50,  COLOR_50,  '% Coins Above 50d SMA'),
    (r25c3, pct_200, COLOR_200, '% Coins Above 200d SMA'),
]:
    with col:
        st.markdown(f"<p style='text-align:center;font-weight:bold;'>{title}</p>",
                    unsafe_allow_html=True)
        fig = go.Figure()
        above = np.where(series >= 50, series, 50.0)
        below = np.where(series <= 50, series, 50.0)
        fig.add_trace(go.Scatter(x=d_dates, y=above, fill='tozeroy',
                                 fillcolor='rgba(165,214,247,0.25)',
                                 line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=d_dates, y=below, fill='tozeroy',
                                 fillcolor='rgba(246,185,149,0.25)',
                                 line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=d_dates, y=series, mode='lines',
                                 line=dict(color=color, width=1.8), showlegend=False))
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
    elif not np.isnan(row['val']):
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
    month_separators.append(dict(
        type='line', x0=sp['min']-0.5, x1=sp['min']-0.5, y0=-0.5, y1=6.5,
        xref='x', yref='y', line=dict(color='#555555', width=1.5)))

fig_cal = go.Figure()
fig_cal.add_trace(go.Heatmap(
    z=z_future, x=list(range(n_weeks)), y=list(range(7)),
    text=text_cal, texttemplate="%{text}", textfont={"size": 9, "color": "#666666"},
    colorscale=[[0,'#2a2a2a'],[1,'#2a2a2a']],
    showscale=False, xgap=3, ygap=3, zmin=0, zmax=1, hoverinfo='none'))
fig_cal.add_trace(go.Heatmap(
    z=z_past, x=list(range(n_weeks)), y=list(range(7)),
    text=text_cal, texttemplate="%{text}", textfont={"size": 9, "color": "#111111"},
    colorscale=[[0,COLOR_NEG],[0.5,'#e8e8e8'],[1,COLOR_POS]],
    showscale=False, xgap=3, ygap=3, zmin=-0.05, zmax=0.05, hoverinfo='none'))

fig_cal.update_layout(
    height=250, margin=dict(l=35, r=10, t=50, b=10),
    paper_bgcolor='#111111', plot_bgcolor='#111111',
    yaxis=dict(ticktext=['M','T','W','T','F','S','S'], tickvals=list(range(7)),
               autorange='reversed', showgrid=False, zeroline=False,
               tickfont=dict(size=10, color='#aaaaaa')),
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
               range=[-0.5, n_weeks-0.5]),
    annotations=annotations, shapes=month_separators)
st.plotly_chart(fig_cal, use_container_width=True, config={'displayModeBar': False})
