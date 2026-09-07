import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
import hashlib
import secrets as pysecrets
import requests
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from github import Github, GithubException

# ==============================================================================
# APP CONFIG
# ==============================================================================
st.set_page_config(
    page_title="CDU Hybrid Digital Twin Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

DB_PATH = "audit_telemetry.db"
GUEST_MODEL_FILE = "models/guest_model.pkl"
PBKDF2_ITERATIONS = 200_000

# Physical Engineering Bounds matched to actual CDU operating cuts
YIELD_BOUNDS = {
    'flow_offgas': (0.100, 0.250),
    'flow_naphtha': (0.020, 0.120),
    'flow_kero': (0.040, 0.100),
    'flow_lago': (0.150, 0.280),
    'flow_residue': (0.400, 0.650)
}

PHYSICS_SLOPES = {
    'flow_offgas': 0.00030,
    'flow_naphtha': 0.00045,
    'flow_kero': 0.00025,
    'flow_lago': 0.00100,
    'flow_residue': -0.00200
}

# ==============================================================================
# VISUAL THEME — light/dark, switchable at runtime
# ==============================================================================
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False


def build_theme_css(dark: bool) -> str:
    if dark:
        t = dict(
            app_bg="#0B1220", app_text="#E7ECF3", muted_text="#9AA7BD",
            panel_bg="#131B2E", panel_border="#243049",
            metric_bg="#131B2E", metric_border="#243049", metric_label="#9AA7BD",
            sidebar_bg="#060B15", sidebar_text="#E7ECF3",
            input_bg="#1B263B", input_text="#E7ECF3", input_border="#2E3B55",
            hero_grad="linear-gradient(120deg, #050A14 0%, #10233F 55%, #0E4C63 100%)",
            table_filter="invert(0.9) hue-rotate(180deg)",
        )
    else:
        t = dict(
            app_bg="#FFFFFF", app_text="#1D2939", muted_text="#667085",
            panel_bg="#F7F9FC", panel_border="#E3E8EF",
            metric_bg="#FFFFFF", metric_border="#E3E8EF", metric_label="#475467",
            sidebar_bg="#0B2540", sidebar_text="#E7ECF3",
            input_bg="#FFFFFF", input_text="#0B2540", input_border="#D0D5DD",
            hero_grad="linear-gradient(120deg, #0B2540 0%, #1565C0 55%, #00B4D8 70%)",
            table_filter="none",
        )

    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    .stApp {{ background: {t['app_bg']}; color: {t['app_text']}; }}
    .stApp p, .stApp li, .stApp label, .stApp .stMarkdown {{ color: {t['app_text']}; }}
    .stApp .stCaption, [data-testid="stCaptionContainer"] {{ color: {t['muted_text']} !important; }}

    /* Hero header */
    .hero-banner {{
        background: {t['hero_grad']};
        padding: 26px 32px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(11, 37, 64, 0.25);
    }}
    .hero-title {{
        color: #FFFFFF;
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.01em;
    }}
    .hero-subtitle {{
        color: rgba(255,255,255,0.85);
        font-size: 0.92rem;
        margin-top: 6px;
    }}
    .hero-badge {{
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: #fff;
        border: 1px solid rgba(255,255,255,0.3);
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.72rem;
        margin-right: 8px;
        margin-top: 12px;
    }}

    /* Section cards */
    .section-card {{
        background: {t['panel_bg']};
        border: 1px solid {t['panel_border']};
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 18px;
    }}

    /* KPI tiles */
    div[data-testid="stMetric"] {{
        background: {t['metric_bg']};
        border: 1px solid {t['metric_border']};
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
    }}
    div[data-testid="stMetricLabel"] {{ font-weight: 600; color: {t['metric_label']}; }}
    div[data-testid="stMetricValue"] {{ color: {t['app_text']}; }}

    /* Sidebar branding — labels/headings only, NOT the text you type */
    section[data-testid="stSidebar"] {{ background: {t['sidebar_bg']}; }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stMarkdown {{
        color: {t['sidebar_text']} !important;
    }}

    /* Text you actually type into any input, anywhere in the app — always readable */
    input, textarea, select {{
        color: {t['input_text']} !important;
        background-color: {t['input_bg']} !important;
        border: 1px solid {t['input_border']} !important;
    }}
    [data-baseweb="select"] > div {{
        background-color: {t['input_bg']} !important;
        border-color: {t['input_border']} !important;
    }}

    /* Buttons */
    div.stButton > button, div.stFormSubmitButton > button {{
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid {t['panel_border']};
    }}
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {{
        background: linear-gradient(90deg, #1565C0, #00B4D8);
        color: #FFFFFF;
        border: none;
    }}

    /* Top utility bar (theme toggle) */
    .topbar-label {{
        color: {t['muted_text']};
        font-size: 0.8rem;
        text-align: right;
        margin-top: 6px;
    }}

    .footer-note {{
        text-align: center;
        color: {t['muted_text']};
        font-size: 0.78rem;
        padding-top: 24px;
    }}
</style>
"""


st.markdown(build_theme_css(st.session_state["dark_mode"]), unsafe_allow_html=True)

_top_l, _top_r = st.columns([6, 1])
with _top_r:
    st.toggle("Dark mode", key="dark_mode")


def hero_header(title, subtitle, badges=None):
    badges_html = "".join(f"<span class='hero-badge'>{b}</span>"for b in (badges or []))
    st.markdown(
        f"""
        <div class="hero-banner">
            <p class="hero-title">{title}</p>
            <p class="hero-subtitle">{subtitle}</p>
            {badges_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# EMBEDDED EXCEL DATASET (cdu data.xlsx)
# ==============================================================================
EXCEL_RECORDS = [
    {"crude_flow": 1939.92, "crude_density": 873.8, "crude_api": 30.420176, "sulphur_wt_pct": 2.13, "cot_degC": 350.280, "flash_zone_p_kgcm2": 1.51, "stripping_steam_flow": 18.20, "top_pa": 436.93, "kero_pa_flow_tph": 406.19, "lago_pa_flow_tph": 1041.17, "top_temp_degC": 117.99, "lago_d86_95_degC": 251.44, "flow_offgas": 295.632125, "flow_naphtha": 86.17, "flow_kero": 113.28, "flow_lago": 391.03, "flow_residue": 1072.25},
    {"crude_flow": 1933.65, "crude_density": 875.2, "crude_api": 30.161163, "sulphur_wt_pct": 2.08, "cot_degC": 349.760, "flash_zone_p_kgcm2": 1.51, "stripping_steam_flow": 18.63, "top_pa": 424.83, "kero_pa_flow_tph": 394.05, "lago_pa_flow_tph": 1029.96, "top_temp_degC": 117.47, "lago_d86_95_degC": 251.60, "flow_offgas": 290.379661, "flow_naphtha": 84.27, "flow_kero": 113.09, "flow_lago": 390.26, "flow_residue": 1070.22},
    {"crude_flow": 1935.35, "crude_density": 875.8, "crude_api": 30.050411, "sulphur_wt_pct": 2.23, "cot_degC": 350.550, "flash_zone_p_kgcm2": 1.53, "stripping_steam_flow": 18.81, "top_pa": 444.88, "kero_pa_flow_tph": 412.42, "lago_pa_flow_tph": 1039.50, "top_temp_degC": 118.33, "lago_d86_95_degC": 252.17, "flow_offgas": 301.855768, "flow_naphtha": 95.36, "flow_kero": 113.84, "flow_lago": 394.83, "flow_residue": 1053.10},
    {"crude_flow": 1837.70, "crude_density": 872.0, "crude_api": 30.754415, "sulphur_wt_pct": 2.29, "cot_degC": 349.275, "flash_zone_p_kgcm2": 1.55, "stripping_steam_flow": 18.49, "top_pa": 443.77, "kero_pa_flow_tph": 403.30, "lago_pa_flow_tph": 1039.09, "top_temp_degC": 118.24, "lago_d86_95_degC": 250.06, "flow_offgas": 307.525607, "flow_naphtha": 96.48, "flow_kero": 111.62, "flow_lago": 375.93, "flow_residue": 974.26},
    {"crude_flow": 1918.65, "crude_density": 871.0, "crude_api": 30.940700, "sulphur_wt_pct": 2.16, "cot_degC": 349.340, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 16.64, "top_pa": 471.26, "kero_pa_flow_tph": 424.19, "lago_pa_flow_tph": 1043.68, "top_temp_degC": 120.68, "lago_d86_95_degC": 250.92, "flow_offgas": 328.723607, "flow_naphtha": 85.84, "flow_kero": 115.53, "flow_lago": 376.69, "flow_residue": 1015.18},
    {"crude_flow": 1940.04, "crude_density": 870.0, "crude_api": 31.127414, "sulphur_wt_pct": 2.18, "cot_degC": 350.100, "flash_zone_p_kgcm2": 1.59, "stripping_steam_flow": 17.27, "top_pa": 469.60, "kero_pa_flow_tph": 423.48, "lago_pa_flow_tph": 1044.65, "top_temp_degC": 119.95, "lago_d86_95_degC": 251.21, "flow_offgas": 323.873455, "flow_naphtha": 90.53, "flow_kero": 115.98, "flow_lago": 379.93, "flow_residue": 1013.62},
    {"crude_flow": 1936.03, "crude_density": 873.8, "crude_api": 30.420176, "sulphur_wt_pct": 2.28, "cot_degC": 350.660, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 18.22, "top_pa": 470.28, "kero_pa_flow_tph": 425.54, "lago_pa_flow_tph": 1044.58, "top_temp_degC": 120.09, "lago_d86_95_degC": 252.01, "flow_offgas": 324.193152, "flow_naphtha": 88.82, "flow_kero": 115.99, "flow_lago": 384.94, "flow_residue": 1018.69},
    {"crude_flow": 1940.45, "crude_density": 871.8, "crude_api": 30.791638, "sulphur_wt_pct": 2.22, "cot_degC": 349.280, "flash_zone_p_kgcm2": 1.61, "stripping_steam_flow": 17.60, "top_pa": 490.76, "kero_pa_flow_tph": 448.00, "lago_pa_flow_tph": 1044.44, "top_temp_degC": 122.43, "lago_d86_95_degC": 251.69, "flow_offgas": 341.000527, "flow_naphtha": 106.06, "flow_kero": 115.95, "flow_lago": 391.39, "flow_residue": 1000.19},
    {"crude_flow": 1927.76, "crude_density": 872.8, "crude_api": 30.605694, "sulphur_wt_pct": 2.35, "cot_degC": 346.635, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 15.52, "top_pa": 480.80, "kero_pa_flow_tph": 432.63, "lago_pa_flow_tph": 1044.65, "top_temp_degC": 123.96, "lago_d86_95_degC": 249.74, "flow_offgas": 347.207295, "flow_naphtha": 115.16, "flow_kero": 116.07, "flow_lago": 374.13, "flow_residue": 991.19},
    {"crude_flow": 1938.39, "crude_density": 866.0, "crude_api": 31.878580, "sulphur_wt_pct": 2.35, "cot_degC": 346.470, "flash_zone_p_kgcm2": 1.57, "stripping_steam_flow": 16.19, "top_pa": 468.26, "kero_pa_flow_tph": 419.14, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 122.32, "lago_d86_95_degC": 249.55, "flow_offgas": 338.476313, "flow_naphtha": 106.01, "flow_kero": 115.96, "flow_lago": 377.64, "flow_residue": 1007.09},
    {"crude_flow": 1937.52, "crude_density": 870.4, "crude_api": 31.052677, "sulphur_wt_pct": 2.35, "cot_degC": 345.925, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 17.06, "top_pa": 472.93, "kero_pa_flow_tph": 427.50, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 122.28, "lago_d86_95_degC": 249.51, "flow_offgas": 340.505672, "flow_naphtha": 105.00, "flow_kero": 116.03, "flow_lago": 378.89, "flow_residue": 1009.61},
    {"crude_flow": 1941.00, "crude_density": 869.6, "crude_api": 31.202334, "sulphur_wt_pct": 2.29, "cot_degC": 347.165, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 17.58, "top_pa": 482.02, "kero_pa_flow_tph": 435.53, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 123.08, "lago_d86_95_degC": 250.05, "flow_offgas": 346.529244, "flow_naphtha": 110.15, "flow_kero": 116.00, "flow_lago": 379.74, "flow_residue": 1004.89},
    {"crude_flow": 1937.93, "crude_density": 869.2, "crude_api": 31.277266, "sulphur_wt_pct": 2.19, "cot_degC": 347.935, "flash_zone_p_kgcm2": 1.54, "stripping_steam_flow": 16.74, "top_pa": 483.99, "kero_pa_flow_tph": 438.30, "lago_pa_flow_tph": 1044.97, "top_temp_degC": 122.95, "lago_d86_95_degC": 250.78, "flow_offgas": 348.064560, "flow_naphtha": 110.16, "flow_kero": 116.00, "flow_lago": 382.49, "flow_residue": 996.65},
    {"crude_flow": 1939.84, "crude_density": 866.6, "crude_api": 31.765636, "sulphur_wt_pct": 2.28, "cot_degC": 347.880, "flash_zone_p_kgcm2": 1.57, "stripping_steam_flow": 17.20, "top_pa": 492.20, "kero_pa_flow_tph": 444.60, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 123.49, "lago_d86_95_degC": 251.68, "flow_offgas": 352.333830, "flow_naphtha": 115.89, "flow_kero": 115.93, "flow_lago": 384.77, "flow_residue": 989.47},
    {"crude_flow": 1936.56, "crude_density": 870.4, "crude_api": 31.052677, "sulphur_wt_pct": 2.29, "cot_degC": 347.330, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 16.71, "top_pa": 484.54, "kero_pa_flow_tph": 435.59, "lago_pa_flow_tph": 1044.89, "top_temp_degC": 123.51, "lago_d86_95_degC": 250.91, "flow_offgas": 347.854086, "flow_naphtha": 112.56, "flow_kero": 115.98, "flow_lago": 380.20, "flow_residue": 996.86},
    {"crude_flow": 1941.52, "crude_density": 874.0, "crude_api": 30.383181, "sulphur_wt_pct": 2.27, "cot_degC": 347.455, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.21, "top_pa": 496.07, "kero_pa_flow_tph": 443.85, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 124.77, "lago_d86_95_degC": 251.05, "flow_offgas": 356.126569, "flow_naphtha": 123.00, "flow_kero": 115.98, "flow_lago": 381.71, "flow_residue": 980.99},
    {"crude_flow": 1939.92, "crude_density": 875.0, "crude_api": 30.198057, "sulphur_wt_pct": 2.30, "cot_degC": 347.600, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 17.37, "top_pa": 490.15, "kero_pa_flow_tph": 435.53, "lago_pa_flow_tph": 1044.97, "top_temp_degC": 124.74, "lago_d86_95_degC": 251.49, "flow_offgas": 353.473595, "flow_naphtha": 120.31, "flow_kero": 115.98, "flow_lago": 383.69, "flow_residue": 984.77},
    {"crude_flow": 1940.40, "crude_density": 873.8, "crude_api": 30.420176, "sulphur_wt_pct": 2.21, "cot_degC": 347.635, "flash_zone_p_kgcm2": 1.57, "stripping_steam_flow": 16.96, "top_pa": 497.02, "kero_pa_flow_tph": 445.69, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 125.10, "lago_d86_95_degC": 251.52, "flow_offgas": 356.162386, "flow_naphtha": 122.37, "flow_kero": 115.99, "flow_lago": 383.95, "flow_residue": 980.37},
    {"crude_flow": 1935.53, "crude_density": 874.0, "crude_api": 30.383181, "sulphur_wt_pct": 2.26, "cot_degC": 347.465, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.47, "top_pa": 497.23, "kero_pa_flow_tph": 443.08, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 125.79, "lago_d86_95_degC": 251.27, "flow_offgas": 357.513488, "flow_naphtha": 128.47, "flow_kero": 115.99, "flow_lago": 382.72, "flow_residue": 971.86},
    {"crude_flow": 1940.21, "crude_density": 876.0, "crude_api": 30.013584, "sulphur_wt_pct": 2.21, "cot_degC": 347.385, "flash_zone_p_kgcm2": 1.56, "stripping_steam_flow": 17.10, "top_pa": 500.41, "kero_pa_flow_tph": 449.19, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 125.79, "lago_d86_95_degC": 250.77, "flow_offgas": 357.771963, "flow_naphtha": 127.35, "flow_kero": 116.03, "flow_lago": 380.60, "flow_residue": 972.19},
    {"crude_flow": 1941.13, "crude_density": 875.0, "crude_api": 30.198057, "sulphur_wt_pct": 2.23, "cot_degC": 347.795, "flash_zone_p_kgcm2": 1.57, "stripping_steam_flow": 16.97, "top_pa": 503.20, "kero_pa_flow_tph": 449.98, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 125.99, "lago_d86_95_degC": 250.99, "flow_offgas": 359.851080, "flow_naphtha": 128.43, "flow_kero": 115.98, "flow_lago": 382.68, "flow_residue": 970.61},
    {"crude_flow": 1937.66, "crude_density": 873.4, "crude_api": 30.494275, "sulphur_wt_pct": 2.22, "cot_degC": 347.880, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.43, "top_pa": 501.99, "kero_pa_flow_tph": 451.98, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.11, "lago_d86_95_degC": 251.10, "flow_offgas": 359.971253, "flow_naphtha": 128.91, "flow_kero": 116.00, "flow_lago": 383.04, "flow_residue": 970.21},
    {"crude_flow": 1935.21, "crude_density": 875.4, "crude_api": 30.124286, "sulphur_wt_pct": 2.20, "cot_degC": 347.500, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.20, "top_pa": 500.41, "kero_pa_flow_tph": 449.61, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.10, "lago_d86_95_degC": 250.77, "flow_offgas": 360.596009, "flow_naphtha": 130.00, "flow_kero": 116.01, "flow_lago": 381.16, "flow_residue": 967.65},
    {"crude_flow": 1941.13, "crude_density": 876.4, "crude_api": 29.939970, "sulphur_wt_pct": 2.14, "cot_degC": 347.460, "flash_zone_p_kgcm2": 1.57, "stripping_steam_flow": 16.92, "top_pa": 500.56, "kero_pa_flow_tph": 448.91, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.10, "lago_d86_95_degC": 250.41, "flow_offgas": 361.353386, "flow_naphtha": 131.00, "flow_kero": 116.02, "flow_lago": 380.05, "flow_residue": 968.74},
    {"crude_flow": 1940.36, "crude_density": 878.0, "crude_api": 29.646355, "sulphur_wt_pct": 2.11, "cot_degC": 347.785, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.06, "top_pa": 503.73, "kero_pa_flow_tph": 450.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.23, "lago_d86_95_degC": 250.70, "flow_offgas": 362.464687, "flow_naphtha": 132.00, "flow_kero": 115.98, "flow_lago": 381.47, "flow_residue": 966.50},
    {"crude_flow": 1938.86, "crude_density": 876.0, "crude_api": 30.013584, "sulphur_wt_pct": 2.21, "cot_degC": 347.750, "flash_zone_p_kgcm2": 1.57, "stripping_steam_flow": 17.20, "top_pa": 503.11, "kero_pa_flow_tph": 450.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.30, "lago_d86_95_degC": 250.84, "flow_offgas": 363.090623, "flow_naphtha": 133.00, "flow_kero": 116.00, "flow_lago": 382.49, "flow_residue": 964.81},
    {"crude_flow": 1939.96, "crude_density": 874.0, "crude_api": 30.383181, "sulphur_wt_pct": 2.24, "cot_degC": 347.735, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.20, "top_pa": 503.00, "kero_pa_flow_tph": 449.62, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.31, "lago_d86_95_degC": 250.93, "flow_offgas": 364.088656, "flow_naphtha": 134.00, "flow_kero": 116.00, "flow_lago": 383.00, "flow_residue": 963.88},
    {"crude_flow": 1940.35, "crude_density": 874.4, "crude_api": 30.309355, "sulphur_wt_pct": 2.21, "cot_degC": 347.925, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.20, "top_pa": 505.28, "kero_pa_flow_tph": 450.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.47, "lago_d86_95_degC": 251.24, "flow_offgas": 366.195610, "flow_naphtha": 136.00, "flow_kero": 116.00, "flow_lago": 384.60, "flow_residue": 961.43},
    {"crude_flow": 1939.06, "crude_density": 877.0, "crude_api": 29.829989, "sulphur_wt_pct": 2.23, "cot_degC": 347.905, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.20, "top_pa": 507.03, "kero_pa_flow_tph": 450.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 126.68, "lago_d86_95_degC": 251.46, "flow_offgas": 368.513728, "flow_naphtha": 139.00, "flow_kero": 116.00, "flow_lago": 385.66, "flow_residue": 957.51},
    {"crude_flow": 1937.89, "crude_density": 877.0, "crude_api": 29.829989, "sulphur_wt_pct": 2.22, "cot_degC": 348.065, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.20, "top_pa": 512.44, "kero_pa_flow_tph": 450.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 127.34, "lago_d86_95_degC": 251.81, "flow_offgas": 372.457894, "flow_naphtha": 147.00, "flow_kero": 116.00, "flow_lago": 386.43, "flow_residue": 948.55},
    {"crude_flow": 1939.46, "crude_density": 874.0, "crude_api": 30.383181, "sulphur_wt_pct": 2.25, "cot_degC": 348.160, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.20, "top_pa": 515.65, "kero_pa_flow_tph": 450.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 127.81, "lago_d86_95_degC": 252.02, "flow_offgas": 375.986326, "flow_naphtha": 154.00, "flow_kero": 116.00, "flow_lago": 387.03, "flow_residue": 940.54},
    {"crude_flow": 1938.83, "crude_density": 874.0, "crude_api": 30.383181, "sulphur_wt_pct": 2.22, "cot_degC": 348.245, "flash_zone_p_kgcm2": 1.58, "stripping_steam_flow": 17.20, "top_pa": 518.73, "kero_pa_flow_tph": 450.00, "lago_pa_flow_tph": 1045.00, "top_temp_degC": 128.26, "lago_d86_95_degC": 252.12, "flow_offgas": 381.925482, "flow_naphtha": 159.69, "flow_kero": 116.00, "flow_lago": 387.71, "flow_residue": 932.92}
]


@st.cache_data(show_spinner=False)
def get_demo_dataframe():
    return pd.DataFrame(EXCEL_RECORDS)


# ==============================================================================
# GITHUB SYNCHRONIZATION ENGINE (push + pull)
# ==============================================================================
def _github_ready():
    return "GITHUB_TOKEN" in st.secrets and "GITHUB_REPO" in st.secrets


def sync_file_to_github(local_file_path, repo_file_path, commit_message="Auto-sync from Streamlit App"):
    """Pushes or updates a file directly in the configured GitHub repository."""
    if not _github_ready():
        return False
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo_name = st.secrets["GITHUB_REPO"]
        g = Github(token)
        repo = g.get_repo(repo_name)
        branch = st.secrets.get("GITHUB_BRANCH", repo.default_branch)

        with open(local_file_path, "rb") as f:
            content = f.read()

        try:
            existing_file = repo.get_contents(repo_file_path, ref=branch)
            repo.update_file(
                path=repo_file_path, message=commit_message,
                content=content, sha=existing_file.sha, branch=branch,
            )
        except GithubException as e:
            if e.status == 404:
                repo.create_file(path=repo_file_path, message=commit_message, content=content, branch=branch)
            else:
                raise e
        return True
    except Exception as err:
        st.sidebar.warning(f"GitHub Sync Alert: {err}")
        return False


def pull_db_from_github_if_missing():
    """
    Restores the SQLite DB from GitHub on a fresh/ephemeral instance.
    Without this, every redeploy silently wipes all provisioned client
    accounts, saved models, and simulation history.
    """
    if os.path.exists(DB_PATH) or not _github_ready():
        return
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo_name = st.secrets["GITHUB_REPO"]
        g = Github(token)
        repo = g.get_repo(repo_name)
        branch = st.secrets.get("GITHUB_BRANCH", repo.default_branch)
        contents = repo.get_contents(DB_PATH, ref=branch)
        with open(DB_PATH, "wb") as f:
            f.write(contents.decoded_content)
    except Exception:
        pass  # no remote copy yet — a fresh DB will be created


# ==============================================================================
# PASSWORD HASHING (salted PBKDF2 — stdlib only, no new dependency)
# ==============================================================================
def hash_password(plain_password, salt_hex=None):
    if salt_hex is None:
        salt_hex = pysecrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", plain_password.encode("utf-8"), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS
    ).hex()
    return f"{salt_hex}${digest}"


def verify_password(plain_password, stored_value):
    try:
        salt_hex, _ = stored_value.split("$", 1)
    except (ValueError, AttributeError):
        return False
    return hash_password(plain_password, salt_hex) == stored_value


# ==============================================================================
# DATABASE & ACCESS CONTROL LAYER
# ==============================================================================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, password_hash TEXT, role TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, login_time TIMESTAMP,
            ip_address TEXT, city TEXT, region TEXT, country TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS protected_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, model_tag TEXT,
            model_path TEXT, training_rows INTEGER, created_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS simulation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, timestamp TIMESTAMP,
            inputs_json TEXT, outputs_json TEXT)''')

    # IMPORTANT: "INSERT OR IGNORE"(not REPLACE) — this seeds default accounts
    # only if they don't already exist, so an admin password change survives
    # the next script rerun instead of being silently reset every time.
    c.execute("INSERT OR IGNORE INTO users VALUES (?, ?, ?)", ("admin", hash_password("Admin@123"), "admin"))
    c.execute("INSERT OR IGNORE INTO users VALUES (?, ?, ?)", ("engineer1", hash_password("User@123"), "user"))
    conn.commit()
    conn.close()


pull_db_from_github_if_missing()
init_db()


def verify_login(username, password):
    clean_u = username.strip().lower()
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT password_hash, role FROM users WHERE username = ?", (clean_u,))
    row = c.fetchone()
    conn.close()
    if row and verify_password(password.strip(), row["password_hash"]):
        return row["role"]
    return None


def create_client_user(new_username, plain_password):
    clean_u = new_username.strip().lower()
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'user')",
                   (clean_u, hash_password(plain_password.strip())))
        conn.commit()
        success, msg = True, f"Client account `{clean_u}` created successfully."
    except sqlite3.IntegrityError:
        success, msg = False, f"Username `{clean_u}` already exists."
    conn.close()
    if success:
        sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Created client user {clean_u}")
    return success, msg


def reset_client_password(target_username, new_plain_password):
    conn = get_db_connection()
    conn.execute("UPDATE users SET password_hash = ? WHERE username = ?",
                 (hash_password(new_plain_password.strip()), target_username))
    conn.commit()
    conn.close()
    sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Reset password for {target_username}")


def delete_client_user(target_username):
    conn = get_db_connection()
    conn.execute("DELETE FROM users WHERE username = ?", (target_username,))
    conn.execute("DELETE FROM protected_models WHERE username = ?", (target_username,))
    conn.execute("DELETE FROM simulation_history WHERE username = ?", (target_username,))
    conn.commit()
    conn.close()
    sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Deleted user {target_username}")


def get_client_ip():
    """
    Streamlit's Python backend receives the request that hits the server,
    not the visitor's browser directly. Behind Streamlit Community Cloud
    or any reverse proxy, the real visitor IP (if forwarded at all) shows
    up in the X-Forwarded-For header, not as the socket's peer address.
    Without header forwarding configured on the host, this cannot be
    recovered server-side — that's why it previously always resolved to
    the server's own outbound IP.
    """
    try:
        headers = st.context.headers  # available on Streamlit 1.37+
        if headers:
            xff = headers.get("X-Forwarded-For")
            if xff:
                return xff.split(",")[0].strip()
            real_ip = headers.get("X-Real-IP")
            if real_ip:
                return real_ip.strip()
    except Exception:
        pass
    return None


def get_visitor_geo():
    client_ip = get_client_ip()
    url = f"https://ipapi.co/{client_ip}/json/" if client_ip else "https://ipapi.co/json/"
    try:
        res = requests.get(url, timeout=3).json()
        if res.get("error"):
            raise ValueError(res.get("reason", "geo lookup failed"))
        return {
            "ip": res.get("ip", client_ip or "Unknown"),
            "city": res.get("city", "Unknown"),
            "region": res.get("region", "Unknown"),
            "country": res.get("country_name", "Unknown"),
        }
    except Exception:
        return {
            "ip": client_ip or "Unavailable",
            "city": "Unavailable", "region": "Unavailable", "country": "Unavailable",
        }


def log_login_event(username):
    geo = get_visitor_geo()
    conn = get_db_connection()
    conn.execute('''INSERT INTO access_logs (username, login_time, ip_address, city, region, country)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                 (username, datetime.now(), geo["ip"], geo["city"], geo["region"], geo["country"]))
    conn.commit()
    conn.close()
    sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Login log for {username}")


# ==============================================================================
# LOGIN THROTTLING (basic brute-force mitigation)
# ==============================================================================
MAX_LOGIN_ATTEMPTS = 5


def login_is_locked():
    return st.session_state.get("failed_logins", 0) >= MAX_LOGIN_ATTEMPTS


def register_failed_login():
    st.session_state["failed_logins"] = st.session_state.get("failed_logins", 0) + 1


def reset_failed_logins():
    st.session_state["failed_logins"] = 0


# ==============================================================================
# PHYSICS-INFORMED CONTINUOUS ENGINE
# ==============================================================================
def density_to_api(density_val):
    sg = density_val / 1000.0 if density_val > 10.0 else density_val
    if sg <= 0:
        return 30.0
    return (141.5 / sg) - 131.5


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class PhysicsInformedYieldModel:
    def __init__(self):
        self.ml_model = MultiOutputRegressor(LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42))
        self.linear_reg = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.flow_targets = []
        self.baseline_stats = {}

    def fit(self, X, y_yields, flow_targets, baseline_stats):
        self.flow_targets = flow_targets
        self.baseline_stats = baseline_stats
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.fit(X_scaled, y_yields)
        logits = np.log(np.clip(y_yields.values, 1e-4, 1.0))
        self.linear_reg.fit(X_scaled, logits)

    def predict(self, X_df):
        X_scaled = self.scaler.transform(X_df)
        ml_preds = self.ml_model.predict(X_scaled)
        linear_preds = softmax(self.linear_reg.predict(X_scaled))
        base_yields = 0.70 * ml_preds + 0.30 * linear_preds

        cot = X_df['cot_degC'].values if 'cot_degC' in X_df else self.baseline_stats['mean_cot']
        fzp = X_df['flash_zone_p_kgcm2'].values if 'flash_zone_p_kgcm2' in X_df else self.baseline_stats['mean_p']
        steam = X_df['stripping_steam_flow'].values if 'stripping_steam_flow' in X_df else self.baseline_stats['mean_steam']
        api = X_df['crude_api'].values if 'crude_api' in X_df else self.baseline_stats.get('mean_api', 30.5)

        delta_severity = (
            (cot - self.baseline_stats['mean_cot'])
            - 16.5 * (fzp - self.baseline_stats['mean_p'])
            + 0.55 * (steam - self.baseline_stats['mean_steam'])
            + 0.85 * (api - self.baseline_stats.get('mean_api', 30.5))
        )

        final_yields = np.zeros_like(base_yields)
        for i, col in enumerate(self.flow_targets):
            slope = PHYSICS_SLOPES.get(col, 0.0)
            adj = base_yields[:, i] + (slope * delta_severity)
            min_b, max_b = YIELD_BOUNDS.get(col, (0.01, 0.90))
            final_yields[:, i] = np.clip(adj, min_b, max_b)

        return final_yields / np.sum(final_yields, axis=1, keepdims=True)


@st.cache_resource(show_spinner=False)
def load_pipeline_cached(path, mtime):
    """Cached by (path, mtime) so retraining invalidates the cache but
    normal reruns (every widget click) don't re-read the pickle from disk."""
    return joblib.load(path)


def load_pipeline(path):
    if not os.path.exists(path):
        return None
    return load_pipeline_cached(path, os.path.getmtime(path))


# ==============================================================================
# DEFAULT MODEL INITIALIZATION
# ==============================================================================
def train_default_guest_model_if_missing():
    if os.path.exists(GUEST_MODEL_FILE):
        return
    df = get_demo_dataframe()
    input_cols = ['crude_flow', 'crude_api', 'sulphur_wt_pct', 'cot_degC', 'flash_zone_p_kgcm2', 'stripping_steam_flow', 'lago_d86_95_degC']
    flow_cols = ['flow_offgas', 'flow_naphtha', 'flow_kero', 'flow_lago', 'flow_residue']
    state_cols = ['top_pa', 'kero_pa_flow_tph', 'lago_pa_flow_tph', 'top_temp_degC']
    crude_col = 'crude_flow'

    yield_targets = df[flow_cols].div(df[crude_col], axis=0)
    baseline_stats = {
        'mean_cot': float(df['cot_degC'].mean()), 'mean_p': float(df['flash_zone_p_kgcm2'].mean()),
        'mean_steam': float(df['stripping_steam_flow'].mean()), 'mean_api': float(df['crude_api'].mean()),
    }

    yield_model = PhysicsInformedYieldModel()
    yield_model.fit(df[input_cols], yield_targets, flow_cols, baseline_stats)

    scaler_states = StandardScaler()
    model_states = MultiOutputRegressor(LGBMRegressor(n_estimators=250, learning_rate=0.03, random_state=42))
    model_states.fit(scaler_states.fit_transform(df[input_cols]), df[state_cols])

    pipeline = {
        "yield_model": yield_model, "model_states": model_states, "scaler_states": scaler_states,
        "input_cols": input_cols, "flow_targets": flow_cols, "state_targets": state_cols,
        "crude_col": crude_col, "baseline_stats": baseline_stats, "training_rows": len(df),
        "last_known_inputs": df[input_cols].iloc[0].to_dict(),
    }
    joblib.dump(pipeline, GUEST_MODEL_FILE)


train_default_guest_model_if_missing()

# ==============================================================================
# SIDEBAR LOGIN & MULTI-TENANT GATEWAY
# ==============================================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = "Guest"
    st.session_state["role"] = "guest"

st.sidebar.markdown("## CDU Digital Twin")
st.sidebar.caption("Hybrid Physics + ML Refinery Platform")
st.sidebar.divider()

if not st.session_state["authenticated"]:
    with st.sidebar.expander("Member / Client Login", expanded=True):
        if login_is_locked():
            st.error("Too many failed attempts. Please refresh the page to try again.")
        else:
            with st.form("login_form", clear_on_submit=False):
                login_user = st.text_input("Username")
                login_pass = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
            if submitted:
                role = verify_login(login_user, login_pass)
                if role:
                    reset_failed_logins()
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = login_user.strip().lower()
                    st.session_state["role"] = role
                    log_login_event(login_user)
                    st.rerun()
                else:
                    register_failed_login()
                    st.error("Invalid username or password.")
else:
    st.sidebar.success(f"Signed in as **{st.session_state['username']}**  \n_{st.session_state['role'].upper()}_")
    if st.sidebar.button("Log Out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["username"] = "Guest"
        st.session_state["role"] = "guest"
        st.rerun()

st.sidebar.divider()

nav_options = ["1. Model Training & DCS Upload", "2. Yield Prediction"]
if st.session_state["authenticated"]:
    nav_options.append("3. Protected Workspace & History")
if st.session_state["role"] == "admin":
    nav_options.append("Admin Audit & Telemetry")

page = st.sidebar.radio("Navigation", nav_options)
st.sidebar.divider()
st.sidebar.caption("v2.0 · Hybrid Digital Twin Engine")

# ==============================================================================
# PAGE 1: MODEL TRAINING & DCS UPLOAD
# ==============================================================================
if page == "1. Model Training & DCS Upload":
    hero_header(
        "Column Data Ingestion & Model Training",
        "Upload historical plant logs or load the calibrated refinery dataset to train a fresh digital twin.",
        badges=["Physics-Informed", "LightGBM + Ridge Hybrid"],
    )

    uploaded_file = st.file_uploader("Upload DCS Historical Data (CSV or Excel)", type=["csv", "xlsx"])
    col1, col2 = st.columns([1, 4])
    use_synthetic = col1.button("Load Demo DCS Dataset")

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        if any("unnamed" in str(col).lower() for col in raw_df.columns):
            raw_df.columns = raw_df.iloc[0].astype(str)
            raw_df = raw_df[1:].reset_index(drop=True)
        st.session_state['active_df'] = raw_df
        st.session_state['active_src_name'] = f"Uploaded File: `{uploaded_file.name}`"
    elif use_synthetic:
        st.session_state['active_df'] = get_demo_dataframe()
        st.session_state['active_src_name'] = "Loaded CDU Refinery Historical Dataset (32 DCS runs)"

    if 'active_df' not in st.session_state:
        st.session_state['active_df'] = get_demo_dataframe()
        st.session_state['active_src_name'] = "Default Calibrated CDU Operating Dataset"

    df = st.session_state['active_df']

    m1, m2, m3 = st.columns(3)
    m1.metric("Active Dataset", st.session_state.get('active_src_name', '').split(":")[0])
    m2.metric("Rows", len(df))
    m3.metric("Columns", len(df.columns))

    st.subheader("Data Inspector")
    st.dataframe(df.head(5), use_container_width=True)
    with st.expander("View Complete Raw Dataset"):
        st.dataframe(df, use_container_width=True)

    has_density = any("dens" in str(c).lower() or "sg" in str(c).lower() for c in df.columns)
    if st.checkbox("Calculate crude_api automatically from Density/SG", value=has_density):
        dens_cols = list(df.columns)
        selected_dens = st.selectbox("Select Density Column", options=dens_cols,
                                      index=dens_cols.index('crude_density') if 'crude_density' in dens_cols else 0)
        df['crude_api'] = df[selected_dens].apply(density_to_api)
        st.success(f"Calculated `crude_api` from `{selected_dens}`")

    default_inputs = ['crude_flow', 'crude_api', 'sulphur_wt_pct', 'cot_degC', 'flash_zone_p_kgcm2', 'stripping_steam_flow', 'lago_d86_95_degC']
    default_flows = ['flow_offgas', 'flow_naphtha', 'flow_kero', 'flow_lago', 'flow_residue']
    default_states = ['top_pa', 'kero_pa_flow_tph', 'lago_pa_flow_tph', 'top_temp_degC']

    st.markdown("### Configure Model Schema")
    with st.container():
        c1, c2, c3 = st.columns(3)
        input_cols = c1.multiselect("Inputs (X)", list(df.columns), default=[c for c in default_inputs if c in df.columns])
        flow_cols = c2.multiselect("Product Flows (Y1)", list(df.columns), default=[c for c in default_flows if c in df.columns])
        state_cols = c3.multiselect("Internal States (Y2)", list(df.columns), default=[c for c in default_states if c in df.columns])
        crude_col = st.selectbox("Crude Inlet Flow Tag", list(df.columns),
                                  index=list(df.columns).index('crude_flow') if 'crude_flow' in df.columns else 0)

    save_as_protected = False
    model_tag = "guest_model"
    if st.session_state["authenticated"]:
        st.divider()
        c_save1, c_save2 = st.columns([1, 2])
        save_as_protected = c_save1.checkbox("Save model into my protected private vault", value=True)
        if save_as_protected:
            model_tag = c_save2.text_input("Protected Model Tag", value=f"{st.session_state['username']}_v1")

    if st.button("Train Digital Twin", type="primary"):
        if not input_cols or not flow_cols:
            st.error("Please select at least one input column and one product-flow column.")
            st.stop()
        with st.spinner("Training model with continuous thermodynamic gradients..."):
            all_needed = list(set(input_cols + flow_cols + state_cols + [crude_col]))
            clean_df = df[all_needed].apply(pd.to_numeric, errors='coerce').dropna()

            # Guard against zero/negative crude flow rows before dividing by them
            clean_df = clean_df[clean_df[crude_col] > 0]
            if clean_df.empty:
                st.error("No valid rows remain after removing zero/negative crude-flow entries.")
                st.stop()

            total_out = clean_df[flow_cols].sum(axis=1)
            valid_df = clean_df[np.abs(total_out - clean_df[crude_col]) / clean_df[crude_col] < 0.05].copy()

            if valid_df.empty:
                st.error("Mass balance error: Data does not close within 5%.")
                st.stop()
            if len(valid_df) < 5:
                st.error(f"Only {len(valid_df)} valid rows after cleaning — need at least 5 to train/test split.")
                st.stop()

            yield_targets = valid_df[flow_cols].div(valid_df[crude_col], axis=0)
            X_tr, X_te, yf_tr, yf_te, ys_tr, ys_te, c_tr, c_te = train_test_split(
                valid_df[input_cols], yield_targets, valid_df[state_cols], valid_df[crude_col], test_size=0.2, random_state=42
            )

            baseline_stats = {
                'mean_cot': float(valid_df['cot_degC'].mean()) if 'cot_degC' in valid_df.columns else 348.0,
                'mean_p': float(valid_df['flash_zone_p_kgcm2'].mean()) if 'flash_zone_p_kgcm2' in valid_df.columns else 1.56,
                'mean_steam': float(valid_df['stripping_steam_flow'].mean()) if 'stripping_steam_flow' in valid_df.columns else 17.2,
                'mean_api': float(valid_df['crude_api'].mean()) if 'crude_api' in valid_df.columns else 30.5,
            }

            yield_model = PhysicsInformedYieldModel()
            yield_model.fit(X_tr, yf_tr, flow_cols, baseline_stats)

            scaler_states = StandardScaler()
            model_states = MultiOutputRegressor(LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42))
            model_states.fit(scaler_states.fit_transform(X_tr), ys_tr)

            pipeline = {
                "yield_model": yield_model, "model_states": model_states, "scaler_states": scaler_states,
                "input_cols": input_cols, "flow_targets": flow_cols, "state_targets": state_cols,
                "crude_col": crude_col, "baseline_stats": baseline_stats, "training_rows": len(valid_df),
                "last_known_inputs": clean_df[input_cols].iloc[-1].to_dict(),
            }

            if save_as_protected and st.session_state["authenticated"]:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"models/protected_{st.session_state['username']}_{timestamp_str}.pkl"
                joblib.dump(pipeline, save_path)

                conn = get_db_connection()
                conn.execute('''INSERT INTO protected_models (username, model_tag, model_path, training_rows, created_at)
                                 VALUES (?, ?, ?, ?, ?)''',
                             (st.session_state["username"], model_tag, save_path, len(valid_df), datetime.now()))
                conn.commit()
                conn.close()

                sync_file_to_github(save_path, save_path, f"Auto-sync: New protected model {model_tag}")
                sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Registered protected model {model_tag}")
                st.success(f"Model saved to your vault and pushed to GitHub as `{model_tag}`!")
            else:
                joblib.dump(pipeline, GUEST_MODEL_FILE)
                sync_file_to_github(GUEST_MODEL_FILE, GUEST_MODEL_FILE, "Auto-sync: Updated guest model")
                st.success("Model trained and saved into public guest sandbox.")

# ==============================================================================
# PAGE 2: REAL-TIME PREDICTION
# ==============================================================================
elif page == "2. Yield Prediction":
    hero_header(
        "Autonomous CDU Prediction & Dynamic Sensitivity",
        "Simulate product recovery and internal column state from live operating boundary conditions.",
        badges=["Real-time Inference", "Furnace Sensitivity Sweep"],
    )

    active_pipeline = None
    if st.session_state["authenticated"]:
        conn = get_db_connection()
        user_models_df = pd.read_sql_query(
            "SELECT id, model_tag, model_path FROM protected_models WHERE username = ?",
            conn, params=(st.session_state["username"],)
        )
        conn.close()

        if user_models_df.empty:
            st.warning("You do not have any models in your private vault yet. Please train and save one on Page 1 first.")
            st.stop()
        else:
            tag_to_path = dict(zip(user_models_df["model_tag"], user_models_df["model_path"]))
            selected_tag = st.selectbox("Select Your Private Model:", options=list(tag_to_path.keys()))
            chosen_path = tag_to_path[selected_tag]
            active_pipeline = load_pipeline(chosen_path)
            if active_pipeline is None:
                st.error("Selected model artifact is missing from disk.")
                st.stop()
    else:
        active_pipeline = load_pipeline(GUEST_MODEL_FILE)
        if active_pipeline is None:
            st.warning("No default model available. Please train one on Page 1 or log in.")
            st.stop()

    input_cols = active_pipeline["input_cols"]
    flow_targets = active_pipeline["flow_targets"]
    state_targets = active_pipeline["state_targets"]
    last_in = active_pipeline.get("last_known_inputs", {})
    stats = active_pipeline["baseline_stats"]

    st.markdown("#### 1. Crude Assay Properties")
    with st.container():
        c_dens1, c_dens2 = st.columns(2)
        last_api = float(last_in.get('crude_api', 30.5))
        default_density = float(141.5 / (last_api + 131.5) * 1000.0) if last_api else 874.0

        input_density = c_dens1.number_input("Crude Density (kg/m³ or SG @ 15°C)", value=default_density, format="%.2f")
        calculated_api = density_to_api(input_density)
        c_dens2.metric("Calculated Crude API", f"{calculated_api:.2f} °API")

    st.markdown("#### 2. Operating Boundary Inputs")
    st.caption("Pre-filled with the most recent DCS reading in the active model.")
    input_data = {}
    cols = st.columns(3)
    for i, feat in enumerate(input_cols):
        if feat == 'crude_api':
            input_data[feat] = calculated_api
        else:
            fallback = last_in.get(feat, 348.0 if "cot" in feat.lower() else (1939.0 if "crude" in feat.lower() else (1.56 if "p_kgcm2" in feat.lower() else 17.2)))
            input_data[feat] = cols[i % 3].number_input(feat, value=float(fallback), format="%.2f")

    if st.button("Run Simulation & Predict", type="primary"):
        crude_col = active_pipeline["crude_col"]
        if input_data.get(crude_col, 0) <= 0:
            st.error(f"`{crude_col}` must be greater than zero to run a simulation.")
            st.stop()

        try:
            input_df = pd.DataFrame([input_data])[input_cols]
            norm_yields = active_pipeline["yield_model"].predict(input_df)[0]
            crude_in = input_data[crude_col]
            pred_flows = norm_yields * crude_in

            scaled_state = active_pipeline["scaler_states"].transform(input_df)
            pred_states = active_pipeline["model_states"].predict(scaled_state)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        cot_delta = input_data.get('cot_degC', 348.0) - stats['mean_cot']
        fzp_delta = input_data.get('flash_zone_p_kgcm2', 1.56) - stats['mean_p']

        for k, s in enumerate(state_targets):
            if "top_temp" in s.lower():
                pred_states[k] += 0.15 * cot_delta - 2.5 * fzp_delta
            elif "pa" in s.lower():
                pred_states[k] += 0.80 * cot_delta

        if st.session_state["authenticated"]:
            conn = get_db_connection()
            conn.execute('''INSERT INTO simulation_history (username, timestamp, inputs_json, outputs_json)
                             VALUES (?, ?, ?, ?)''',
                         (st.session_state["username"], datetime.now(), json.dumps(input_data),
                          json.dumps(dict(zip(flow_targets, pred_flows.tolist())))))
            conn.commit()
            conn.close()
            sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Logged run for {st.session_state['username']}")

        st.divider()

        kpi_cols = st.columns(len(flow_targets) + 1)
        label_map = {
            'flow_offgas': 'Off-Gas & LPG', 'flow_naphtha': 'Naphtha', 'flow_kero': 'Kerosene',
            'flow_lago': 'LAGO', 'flow_residue': 'Atm. Residue',
        }
        for i, col in enumerate(flow_targets):
            kpi_cols[i].metric(label_map.get(col, col), f"{pred_flows[i]:.1f} t/h", f"{norm_yields[i]*100:.1f}%")
        kpi_cols[-1].metric("Total Mass Out", f"{np.sum(pred_flows):.1f} t/h", "0.00% closure error")

        c_left, c_right = st.columns(2)
        with c_left:
            st.subheader("Product Recovery Yields")
            st.table(pd.DataFrame({
                "Cut Stream": flow_targets,
                "Yield (wt%)": [f"{y*100:.2f}%"for y in norm_yields],
                "Rate (t/h)": [f"{f:.2f}"for f in pred_flows],
            }))
        with c_right:
            st.subheader("Predicted Column Profile")
            st.table(pd.DataFrame({
                "Parameter": state_targets,
                "Predicted Value": [f"{v:.2f} {'°C' if 'temp' in n.lower() else 't/h'}"for n, v in zip(state_targets, pred_states)],
            }))

        st.divider()
        st.subheader("Feed & Recovery Analytics")
        display_labels = [label_map.get(col, col) for col in flow_targets]
        plot_df = pd.DataFrame({
            "Product Cut": display_labels, "Mass Flow (t/h)": pred_flows, "Yield Share (%)": norm_yields * 100.0,
        })

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown("**Mass Recovery by Product Cut (t/h)**")
            st.bar_chart(plot_df.set_index("Product Cut")[["Mass Flow (t/h)"]], color="#1565C0")
        with g_col2:
            st.markdown("**Yield Fraction Breakdown (% Recovery)**")
            st.bar_chart(plot_df.set_index("Product Cut")[["Yield Share (%)"]], color="#FF8F00")

        with st.expander("Dynamic Furnace Sensitivity Curve (Yield % vs COT)", expanded=True):
            base_cot = float(input_data.get('cot_degC', stats['mean_cot']))
            cot_sweep = np.linspace(base_cot - 15.0, base_cot + 15.0, 31)
            sweep_yields = []
            for temp in cot_sweep:
                temp_input = input_data.copy()
                temp_input['cot_degC'] = temp
                temp_df = pd.DataFrame([temp_input])[input_cols]
                y_pred = active_pipeline["yield_model"].predict(temp_df)[0]
                sweep_yields.append(y_pred * 100.0)

            sensitivity_df = pd.DataFrame(np.array(sweep_yields), columns=display_labels, index=np.round(cot_sweep, 1))
            sensitivity_df.index.name = "Furnace COT (°C)"
            st.line_chart(sensitivity_df)
            st.caption("Shows physical yield shifts across a ±15°C COT range holding feed assay and pressure steady.")

# ==============================================================================
# PAGE 3: PROTECTED WORKSPACE
# ==============================================================================
elif page == "3. Protected Workspace & History":
    hero_header(f"Protected Workspace", f"Signed in as {st.session_state['username']}", badges=["Private", "Isolated per user"])
    conn = get_db_connection()

    tab_my_models, tab_my_sims = st.tabs(["My Saved Models", "My Simulation History"])

    with tab_my_models:
        my_models = pd.read_sql_query(
            "SELECT id, model_tag, training_rows, created_at, model_path FROM protected_models WHERE username = ? ORDER BY created_at DESC",
            conn, params=(st.session_state['username'],)
        )
        if my_models.empty:
            st.info("No protected models saved yet. Train one on Page 1 while signed in.")
        else:
            st.dataframe(my_models, use_container_width=True)

    with tab_my_sims:
        my_sims = pd.read_sql_query(
            "SELECT timestamp, inputs_json, outputs_json FROM simulation_history WHERE username = ? ORDER BY timestamp DESC",
            conn, params=(st.session_state['username'],)
        )
        if my_sims.empty:
            st.info("No logged simulations on record.")
        else:
            st.dataframe(my_sims, use_container_width=True)

    conn.close()

# ==============================================================================
# PAGE 4: ADMIN GOVERNANCE & TELEMETRY
# ==============================================================================
elif page == "Admin Audit & Telemetry":
    hero_header("Enterprise Client Governance & Audit Portal", "Provision accounts, inspect client workspaces, review access telemetry.", badges=["Admin Only"])
    conn = get_db_connection()

    tab_manage, tab_client_inspect, tab_logs = st.tabs([
        "Client Account Provisioning", "Inspect & Modify Client Spaces", "Access & Location Audit"
    ])

    with tab_manage:
        st.subheader("Create New Client Account")
        with st.form("create_client_form", clear_on_submit=True):
            c_u1, c_u2, c_u3 = st.columns([2, 2, 1])
            new_client_user = c_u1.text_input("New Client Username", placeholder="e.g. refinery_client_a")
            new_client_pass = c_u2.text_input("Initial Password", type="password", placeholder="Enter a strong password")
            create_submitted = c_u3.form_submit_button("Create Account", type="primary", use_container_width=True)
        if create_submitted:
            if new_client_user and new_client_pass:
                if len(new_client_pass) < 8:
                    st.warning("Choose a password with at least 8 characters.")
                else:
                    ok, msg = create_client_user(new_client_user, new_client_pass)
                    st.success(msg) if ok else st.error(msg)
                    if ok:
                        st.rerun()
            else:
                st.warning("Please provide both username and password.")

        st.divider()
        st.subheader("Existing Accounts")
        users_df = pd.read_sql_query("SELECT username, role FROM users ORDER BY role ASC, username ASC", conn)
        st.dataframe(users_df, use_container_width=True)

        st.markdown("#### Password Reset / Account Management")
        client_list = [u for u in users_df["username"].tolist() if u != "admin"]
        if client_list:
            with st.form("reset_pw_form", clear_on_submit=True):
                c_sel, c_np, c_btn1, c_btn2 = st.columns([2, 2, 1, 1])
                selected_client = c_sel.selectbox("Select Client", options=client_list)
                reset_pw = c_np.text_input("New Password", type="password", placeholder="Enter new password")
                reset_submitted = c_btn1.form_submit_button("Reset Password", use_container_width=True)
                delete_submitted = c_btn2.form_submit_button("Delete Client", use_container_width=True)

            if reset_submitted:
                if reset_pw and len(reset_pw) >= 8:
                    reset_client_password(selected_client, reset_pw)
                    st.success(f"Password updated for `{selected_client}`.")
                else:
                    st.error("Enter a password with at least 8 characters.")

            if delete_submitted:
                delete_client_user(selected_client)
                st.warning(f"Client `{selected_client}` deleted.")
                st.rerun()

    with tab_client_inspect:
        st.subheader("Client Data Inspector & Editor")
        all_clients = [u for u in users_df["username"].tolist() if u != "admin"]

        if not all_clients:
            st.info("No registered clients available to inspect.")
        else:
            chosen_user = st.selectbox("Select Client Profile to Inspect:", options=all_clients)

            col_m, col_s = st.columns(2)
            with col_m:
                st.markdown(f"**Models Saved by `{chosen_user}`:**")
                client_models = pd.read_sql_query(
                    "SELECT id, model_tag, training_rows, created_at, model_path FROM protected_models WHERE username = ?",
                    conn, params=(chosen_user,)
                )
                if client_models.empty:
                    st.caption("No models saved by this client.")
                else:
                    st.dataframe(client_models, use_container_width=True)
                    del_m_id = st.selectbox("Delete Model ID", options=client_models["id"].tolist(), key="del_m_key")
                    if st.button("Delete Selected Model", key="del_m_btn"):
                        m_row = client_models[client_models["id"] == del_m_id].iloc[0]
                        if os.path.exists(m_row["model_path"]):
                            os.remove(m_row["model_path"])
                        conn.execute("DELETE FROM protected_models WHERE id = ?", (del_m_id,))
                        conn.commit()
                        sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Admin removed model {del_m_id}")
                        st.success(f"Removed model `{m_row['model_tag']}`.")
                        st.rerun()

            with col_s:
                st.markdown(f"**Simulations Run by `{chosen_user}`:**")
                client_sims = pd.read_sql_query(
                    "SELECT timestamp, inputs_json, outputs_json FROM simulation_history WHERE username = ? ORDER BY timestamp DESC",
                    conn, params=(chosen_user,)
                )
                if client_sims.empty:
                    st.caption("No simulations logged for this client.")
                else:
                    st.dataframe(client_sims, use_container_width=True)

    with tab_logs:
        st.subheader("Global Sign-in Geolocation & Telemetry")
        st.caption("Contains user IP/location data — treat as sensitive and avoid pushing this table to a public repo.")
        access_df = pd.read_sql_query(
            "SELECT username, login_time, ip_address, city, region, country FROM access_logs ORDER BY login_time DESC", conn
        )
        st.dataframe(access_df, use_container_width=True)

    conn.close()

st.markdown("<div class='footer-note'>CDU Hybrid Digital Twin Platform · Internal Engineering Tool</div>", unsafe_allow_html=True)
