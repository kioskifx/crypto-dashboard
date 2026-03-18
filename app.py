import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Market Monitor", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Simulated data generation for instant rendering
np.random.seed(42)

def generate_returns_data(multiplier=1):
    bins = ['-20%', '-15%', '-10%', '-5%', '5%', '10%', '15%', '20%']
    counts = [np.random.randint(0, 5), np.random.randint(1, 10), np.random.randint(10, 30), 
              np.random.randint(30, 80), np.random.randint(100, 250), np.random.randint(30, 80), 
              np.random.randint(5, 20), np.random.randint(0, 10)]
    return pd.DataFrame({'Bin': bins, 'Count': [c * multiplier for c in counts]})

def generate_zscores():
    sectors = ['MEME', 'L1', 'DINO', 'AI', 'LRT', 'NEW', 'CEXDEX', 'L2', 'NFT', 'NAR', 'DEFI', 'GAMING']
    scores = np.random.uniform(0.1, 1.2, len(sectors))
    scores = np.sort(scores)[::-1] 
    return pd.DataFrame({'Sector': sectors, 'Z-Score': scores}).sort_values('Z-Score', ascending=True)

def generate_timeseries(days=90):
    dates = [datetime.today() - timedelta(days=x) for x in range(days)]
    val_2d = np.random.uniform(0, 4, days) + np.sin(np.linspace(0, 10, days))
    val_5d = np.random.uniform(0, 2, days) + np.sin(np.linspace(0, 10, days)) * 0.5
    osc_primary = np.random.uniform(-20, 15, days)
    
    val_2d[0], val_5d[0] = 12, 5 
    osc_primary[0] = 13
    
    return pd.DataFrame({
        'Date': dates[::-1], '2_day': val_2d[::-1], '5_day': val_5d[::-1],
        'Osc_Primary': osc_primary[::-1]
    })

df_p1d = generate_returns_data()
df_z_p1d = generate_zscores()
df_ts = generate_timeseries()

COLOR_POS = '#88CEEB' 
COLOR_NEG = '#F4A460' 
COLOR_HL = '#FFFF00'  

col_title, col_date = st.columns([8, 1])
with col_title:
    st.markdown("### Market Monitor")
with col_date:
    st.markdown(f"**{datetime.now().strftime('%Y-%m-%d')}**")
st.divider()

# Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("<p style='text-align: center; font-weight: bold;'>P1D Returns</p>", unsafe_allow_html=True)
    colors = [COLOR_NEG if '-' in b else COLOR_POS for b in df_p1d['Bin']]
    fig1 = go.Figure(data=[go.Bar(x=df_p1d['Bin'], y=df_p1d['Count'], marker_color=colors)])
    fig1.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("<p style='text-align: center; font-weight: bold;'>Primary Breadth Ratio</p>", unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['2_day'], mode='lines', name='2 day ratio', line=dict(color=COLOR_POS)))
    fig2.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['5_day'], mode='lines', name='5 day ratio', line=dict(color=COLOR_NEG)))
    fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.markdown("<p style='text-align: center; font-weight: bold;'>Sector Z-Scores P1D</p>", unsafe_allow_html=True)
    colors = [COLOR_POS] * len(df_z_p1d)
    colors[-1], colors[-2] = COLOR_HL, COLOR_HL 
    fig3 = go.Figure(data=[go.Bar(x=df_z_p1d['Z-Score'], y=df_z_p1d['Sector'], orientation='h', marker_color=colors)])
    fig3.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=650, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)
