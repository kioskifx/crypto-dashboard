import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="Market Monitor", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; max-width: 98%; }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- COLORS ---
COLOR_POS = '#A5D6F7' # Pastel Blue
COLOR_NEG = '#F6B995' # Pastel Orange

# --- DATA GENERATORS ---
np.random.seed(datetime.now().day) # Changes daily

def gen_returns(is_weekly=False):
    bins = ['-20%', '-15%', '-10%', '-5%', '5%', '10%', '15%', '20%']
    mult = 1.5 if is_weekly else 1.0
    counts = [np.random.randint(0, 5), np.random.randint(5, 15), np.random.randint(15, 30), 
              np.random.randint(40, 80), np.random.randint(100, 250), np.random.randint(40, 80), 
              np.random.randint(5, 20), np.random.randint(0, 10)]
    return pd.DataFrame({'Bin': bins, 'Count': [int(c * mult) for c in counts]})

def gen_zscores():
    sectors = ['MEME', 'L1', 'DINO', 'AI', 'LST', 'NEW', 'CEXDEX', 'L2', 'NFT', 'NAR', 'DEFI', 'GAMING', 'DEPIN', 'PRIVACY', 'AGENTS', 'INFRA', 'DWF', 'INFOFI']
    scores = np.random.uniform(-1.2, 1.2, len(sectors))
    df = pd.DataFrame({'Sector': sectors, 'Z-Score': scores}).sort_values('Z-Score', ascending=True)
    return df

def gen_setups():
    setups = ['Backside Short', 'Breakdown', 'Contd. Short', 'Episodic Pivot', 'Contd. Breakout', 'Bottom Bounce']
    counts = [-9, -7, -1, 6, 29, 30]
    return pd.DataFrame({'Setup': setups, 'Count': counts})

def gen_timeseries(days=120):
    dates = [datetime.today() - timedelta(days=x) for x in range(days)]
    dates.reverse()
    val_2d = np.random.uniform(0, 3, days) + np.sin(np.linspace(0, 15, days))
    val_5d = np.random.uniform(0, 1.5, days) + np.sin(np.linspace(0, 15, days)) * 0.5
    
    # Force a spike at the end to match your arrow
    val_2d[-1], val_5d[-1] = 10, 5 
    
    osc_prim = np.random.uniform(-15, 10, days)
    osc_sec = np.random.uniform(-20, 20, days)
    # Simulate historical crash
    osc_sec[20:30] = np.random.uniform(-90, -40, 10) 
    
    return pd.DataFrame({'Date': dates, '2_day': val_2d, '5_day': val_5d, 'Osc_Primary': osc_prim, 'Osc_Secondary': osc_sec})

df_p1d = gen_returns()
df_p1w = gen_returns(is_weekly=True)
df_zp1d = gen_zscores()
df_zp1w = gen_zscores()
df_setups = gen_setups()
df_ts = gen_timeseries()

# --- HELPER FUNCTION FOR CHARTS ---
def apply_clean_layout(fig, height=350):
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10)
    )
    return fig

# --- HEADER ---
col_t, col_d = st.columns([9, 1])
col_t.markdown("## Market Monitor")
col_d.markdown(f"#### {datetime.now().strftime('%Y-%m-%d')}")
st.divider()

# --- ROW 1: 5 COLUMNS ---
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)

with r1c1:
    st.markdown("<p style='text-align: center; font-weight: bold;'>P1D Returns</p>", unsafe_allow_html=True)
    colors = [COLOR_NEG if '-' in b else COLOR_POS for b in df_p1d['Bin']]
    fig = go.Figure(go.Bar(x=df_p1d['Bin'], y=df_p1d['Count'], marker_color=colors, text=df_p1d['Count'], textposition='outside'))
    st.plotly_chart(apply_clean_layout(fig), use_container_width=True, config={'displayModeBar': False})

with r1c2:
    st.markdown("<p style='text-align: center; font-weight: bold;'>P1W Returns</p>", unsafe_allow_html=True)
    colors = [COLOR_NEG if '-' in b else COLOR_POS for b in df_p1w['Bin']]
    fig = go.Figure(go.Bar(x=df_p1w['Bin'], y=df_p1w['Count'], marker_color=colors, text=df_p1w['Count'], textposition='outside'))
    st.plotly_chart(apply_clean_layout(fig), use_container_width=True, config={'displayModeBar': False})

with r1c3:
    st.markdown("<p style='text-align: center; font-weight: bold;'>Sector Z-Scores P1D</p>", unsafe_allow_html=True)
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in df_zp1d['Z-Score']]
    fig = go.Figure(go.Bar(x=df_zp1d['Z-Score'], y=df_zp1d['Sector'], orientation='h', marker_color=colors, text=df_zp1d['Z-Score'].round(1), textposition='outside'))
    st.plotly_chart(apply_clean_layout(fig), use_container_width=True, config={'displayModeBar': False})

with r1c4:
    st.markdown("<p style='text-align: center; font-weight: bold;'>Sector Z-Scores P1W</p>", unsafe_allow_html=True)
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in df_zp1w['Z-Score']]
    fig = go.Figure(go.Bar(x=df_zp1w['Z-Score'], y=df_zp1w['Sector'], orientation='h', marker_color=colors, text=df_zp1w['Z-Score'].round(1), textposition='outside'))
    st.plotly_chart(apply_clean_layout(fig), use_container_width=True, config={'displayModeBar': False})

with r1c5:
    st.markdown("<p style='text-align: center; font-weight: bold;'>Setups P1W</p>", unsafe_allow_html=True)
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in df_setups['Count']]
    fig = go.Figure(go.Bar(x=df_setups['Count'], y=df_setups['Setup'], orientation='h', marker_color=colors, text=abs(df_setups['Count']), textposition='outside'))
    st.plotly_chart(apply_clean_layout(fig), use_container_width=True, config={'displayModeBar': False})

st.write("")

# --- ROW 2: 3 COLUMNS ---
r2c1, r2c2, r2c3 = st.columns(3)

with r2c1:
    st.markdown("<p style='text-align: center; font-weight: bold;'>Primary Breadth Ratio (5% daily moves)</p>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['2_day'], mode='lines', name='2 day ratio', line=dict(color=COLOR_POS)))
    fig.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['5_day'], mode='lines', name='5 day ratio', line=dict(color=COLOR_NEG)))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(apply_clean_layout(fig, 300), use_container_width=True, config={'displayModeBar': False})

with r2c2:
    st.markdown("<p style='text-align: center; font-weight: bold;'>Primary Breadth Oscillator (5% daily moves)</p>", unsafe_allow_html=True)
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in df_ts['Osc_Primary']]
    fig = go.Figure(go.Bar(x=df_ts['Date'], y=df_ts['Osc_Primary'], marker_color=colors))
    st.plotly_chart(apply_clean_layout(fig, 300), use_container_width=True, config={'displayModeBar': False})

with r2c3:
    st.markdown("<p style='text-align: center; font-weight: bold;'>Secondary Breadth Oscillator (1σ weekly moves)</p>", unsafe_allow_html=True)
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in df_ts['Osc_Secondary']]
    fig = go.Figure(go.Bar(x=df_ts['Date'], y=df_ts['Osc_Secondary'], marker_color=colors))
    st.plotly_chart(apply_clean_layout(fig, 300), use_container_width=True, config={'displayModeBar': False})

# --- ROW 3: CALENDAR HEATMAP ---
st.write("")
st.markdown("<p style='text-align: center; font-weight: bold;'>Daily Performance Heatmap</p>", unsafe_allow_html=True)

# Generate simple calendar data
dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now())
df_cal = pd.DataFrame({'Date': dates, 'Value': np.random.uniform(-1, 1, len(dates))})
df_cal['Week'] = df_cal['Date'].dt.isocalendar().week
df_cal['Day'] = df_cal['Date'].dt.day_name().str[:3]
df_cal['DayNum'] = df_cal['Date'].dt.day

# Map to matrix for plotting
days_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
heatmap_data = df_cal.pivot(index='Day', columns='Week', values='Value').reindex(days_order)
text_data = df_cal.pivot(index='Day', columns='Week', values='DayNum').reindex(days_order).fillna('')

fig_cal = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    text=text_data.values,
    texttemplate="%{text}",
    textfont={"size": 10, "color": "black"},
    colorscale=[[0, COLOR_NEG], [0.5, "white"], [1, COLOR_POS]],
    showscale=False,
    xgap=3, ygap=3,
    hoverinfo='none'
))
fig_cal.update_layout(
    height=250, 
    margin=dict(l=20, r=20, t=10, b=10),
    yaxis=dict(autorange="reversed", showgrid=False, zeroline=False),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig_cal, use_container_width=True, config={'displayModeBar': False})
