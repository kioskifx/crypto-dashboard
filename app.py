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
df_zp1d = gen
