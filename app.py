import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import hashlib
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="CropVision - Yield Analytics",
    page_icon="\U0001F33E",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CROP CATEGORIES
# =====================================================
CROP_CATEGORIES = {
    "Vegetables": {
        "icon": "\U0001F96C",
        "crops": [
            "Onion", "Potato", "Sweet Potato", "Tapioca", "Garlic",
            "Ginger", "Dry Chillies"
        ],
        "badge_cls": "veg",
    },
    "Spices & Dhanya": {
        "icon": "\U0001F33F",
        "crops": [
            "Coriander", "Black Pepper", "Cardamom", "Turmeric", "Ginger"
        ],
        "badge_cls": "spice",
    },
    "Cereals & Grains": {
        "icon": "\U0001F33E",
        "crops": [
            "Rice", "Wheat", "Maize", "Bajra", "Jowar", "Ragi", "Barley",
            "Small Millets", "Other Cereals"
        ],
        "badge_cls": "",
    },
    "Pulses": {
        "icon": "\U0001FAD8",
        "crops": [
            "Arhar/Tur", "Gram", "Masoor", "Moong(Green Gram)", "Urad",
            "Cowpea(Lobia)", "Horse-Gram", "Khesari", "Moth",
            "Peas & Beans (Pulses)", "Other  Rabi Pulses",
            "Other Kharif Pulses", "Other Summer Pulses"
        ],
        "badge_cls": "pulse",
    },
    "Oilseeds": {
        "icon": "\U0001F95C",
        "crops": [
            "Groundnut", "Sesamum", "Linseed", "Castor Seed",
            "Rapeseed &Mustard", "Soyabean", "Safflower", "Sunflower",
            "Niger Seed", "Guar Seed", "Other Oilseeds", "Oilseeds Total"
        ],
        "badge_cls": "oil",
    },
    "Cash Crops": {
        "icon": "\U0001F3ED",
        "crops": [
            "Sugarcane", "Cotton(Lint)", "Jute", "Mesta", "Tobacco", "Sannhamp"
        ],
        "badge_cls": "cash",
    },
    "Fruits & Plantation": {
        "icon": "\U0001F34C",
        "crops": [
            "Banana", "Coconut", "Arecanut", "Cashewnut"
        ],
        "badge_cls": "fruit",
    },
}

# =====================================================
# RAINFALL-BASED CROP RECOMMENDATIONS
# =====================================================
RAINFALL_CROP_MAP = {
    "very_low": {
        "range": "Below 500 mm",
        "label": "Arid / Very Low Rainfall",
        "label_icon": "\U0001F3DC\uFE0F",
        "crops": ["Bajra", "Moth", "Guar Seed", "Castor Seed", "Barley", "Gram"],
        "advice": "Focus on drought-resistant millets and pulses. Drip irrigation strongly recommended. Mulching helps retain soil moisture.",
    },
    "low": {
        "range": "500 to 750 mm",
        "label": "Low Rainfall",
        "label_icon": "\U0001F324\uFE0F",
        "crops": ["Jowar", "Groundnut", "Sesamum", "Safflower", "Horse-Gram", "Rapeseed &Mustard", "Sunflower", "Linseed"],
        "advice": "Semi-arid crops perform well. Consider intercropping with pulses. Rainwater harvesting is beneficial.",
    },
    "moderate": {
        "range": "750 to 1150 mm",
        "label": "Moderate Rainfall",
        "label_icon": "\U0001F326\uFE0F",
        "crops": ["Wheat", "Maize", "Soyabean", "Onion", "Potato", "Coriander", "Turmeric", "Cotton(Lint)", "Arhar/Tur"],
        "advice": "Ideal zone for most Rabi and Kharif crops. Crop rotation with pulses improves soil health. Monitor for fungal diseases.",
    },
    "high": {
        "range": "1150 to 2000 mm",
        "label": "High Rainfall",
        "label_icon": "\U0001F327\uFE0F",
        "crops": ["Rice", "Sugarcane", "Jute", "Banana", "Ginger", "Black Pepper", "Cardamom", "Tapioca"],
        "advice": "Paddy and plantation crops thrive. Ensure proper drainage. Watch for waterlogging and pest outbreaks during monsoon.",
    },
    "very_high": {
        "range": "Above 2000 mm",
        "label": "Very High Rainfall",
        "label_icon": "\u26C8\uFE0F",
        "crops": ["Rice", "Arecanut", "Coconut", "Cashewnut", "Black Pepper", "Cardamom", "Tapioca"],
        "advice": "Plantation and paddy crops suited best. Terrace farming recommended on slopes. Flood-resistant varieties preferred.",
    },
}

# =====================================================
# PRECAUTIONS DATA
# =====================================================
PRECAUTIONS = {
    "Excessive Rainfall": {
        "icon": "\U0001F30A",
        "tips": [
            "Build proper drainage channels to prevent waterlogging",
            "Use raised-bed farming techniques for vegetables",
            "Apply fungicides preventively as humidity promotes fungal growth",
            "Harvest mature crops immediately to prevent rotting",
            "Store seeds and grains in moisture-proof containers",
        ],
    },
    "Drought Conditions": {
        "icon": "\u2600\uFE0F",
        "tips": [
            "Switch to drought-resistant crop varieties (Bajra, Jowar)",
            "Implement drip irrigation to maximize water efficiency",
            "Use mulching to reduce soil moisture evaporation",
            "Avoid deep ploughing during dry spells",
            "Schedule watering in early morning or late evening",
        ],
    },
    "Pest Management": {
        "icon": "\U0001F41B",
        "tips": [
            "Regularly inspect crops for early signs of infestation",
            "Use integrated pest management (IPM) before heavy pesticides",
            "Rotate crops each season to break pest life cycles",
            "Introduce natural predators like ladybugs for aphid control",
            "Maintain field hygiene - remove crop residues after harvest",
        ],
    },
    "Fertilizer Usage": {
        "icon": "\U0001F9EA",
        "tips": [
            "Conduct soil testing before deciding fertilizer type and quantity",
            "Prefer organic manure and compost where possible",
            "Do not over-apply nitrogen - it damages long-term soil health",
            "Split fertilizer application across crop growth stages",
            "Combine chemical fertilizers with bio-fertilizers for balance",
        ],
    },
    "Soil Health": {
        "icon": "\U0001F331",
        "tips": [
            "Rotate between cereals and legumes to replenish nitrogen",
            "Avoid mono-cropping for more than 2 consecutive seasons",
            "Use green manuring crops (Dhaincha, Sunhemp) in off-season",
            "Maintain soil pH between 6.0 and 7.5 for most crops",
            "Reduce tillage to preserve beneficial soil micro-organisms",
        ],
    },
    "Seasonal Planning": {
        "icon": "\U0001F321\uFE0F",
        "tips": [
            "Kharif (Jun-Oct): Sow rice, maize, cotton with monsoon onset",
            "Rabi (Oct-Mar): Sow wheat, gram, mustard with winter cooling",
            "Zaid/Summer (Mar-Jun): Grow watermelon, cucumber, moong",
            "Consult local agri-extension services for sowing dates",
            "Keep buffer stock of seeds for re-sowing if crop fails",
        ],
    },
}


# =====================================================
# MEGA CSS - PREMIUM DASHBOARD THEME
# =====================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding: 1.2rem 2rem 2rem 2rem;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1724 0%, #141d2f 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e8ecf1;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #b0bec5 !important;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #4fc3f7 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #90a4ae;
    }

    /* KPI CARD */
    .kpi-card {
        background: linear-gradient(135deg, #1a2332 0%, #1e293b 100%);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 22px 24px;
        min-height: 130px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 16px 16px 0 0;
    }
    .kpi-card.blue::before  { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    .kpi-card.green::before { background: linear-gradient(90deg, #22c55e, #4ade80); }
    .kpi-card.amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .kpi-card.rose::before  { background: linear-gradient(90deg, #f43f5e, #fb7185); }
    .kpi-card.purple::before{ background: linear-gradient(90deg, #a855f7, #c084fc); }
    .kpi-card.cyan::before  { background: linear-gradient(90deg, #06b6d4, #22d3ee); }

    .kpi-label {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .kpi-value {
        color: #f1f5f9;
        font-size: 30px;
        font-weight: 800;
        line-height: 1.1;
    }
    .kpi-sub {
        color: #64748b;
        font-size: 12px;
        margin-top: 6px;
        font-weight: 400;
    }
    .kpi-icon {
        position: absolute;
        top: 18px; right: 20px;
        font-size: 28px;
        opacity: 0.6;
    }

    /* SECTION CARD */
    .section-card {
        background: linear-gradient(135deg, #1a2332 0%, #1e293b 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 24px;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    .section-card h3 {
        color: #e2e8f0;
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 16px;
    }

    /* LOGIN PAGE */
    .login-header {
        text-align: center;
        padding: 10px 0 6px 0;
    }
    .login-icon-ring {
        width: 80px; height: 80px;
        margin: 0 auto 16px auto;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(168,85,247,0.18));
        border: 2px solid rgba(59,130,246,0.3);
        display: flex; align-items: center; justify-content: center;
        font-size: 38px;
        box-shadow: 0 0 40px rgba(59,130,246,0.12);
    }
    .login-header h1 {
        color: #f1f5f9;
        font-size: 32px;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.3px;
    }
    .login-header p {
        color: #64748b;
        font-size: 14px;
        margin: 0 0 8px 0;
    }
    .login-shimmer-bar {
        height: 3px;
        border-radius: 4px;
        background: linear-gradient(90deg, #3b82f6, #a855f7, #3b82f6);
        background-size: 200% 100%;
        animation: login-shimmer 3s linear infinite;
        margin: 18px 0 24px 0;
    }
    @keyframes login-shimmer {
        0%   { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    .login-features {
        display: flex;
        justify-content: center;
        gap: 32px;
        margin-top: 16px;
        padding-top: 18px;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
    .login-feat-item { text-align: center; }
    .login-feat-item .feat-icon { font-size: 22px; margin-bottom: 4px; }
    .login-feat-item .feat-label { color: #64748b; font-size: 11px; font-weight: 500; }
    .login-footer {
        text-align: center;
        margin-top: 20px;
    }
    .login-footer p {
        color: #475569;
        font-size: 11px;
        margin: 0;
    }

    /* HEADER BAR */
    .header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .header-bar h2 {
        color: #f1f5f9;
        font-size: 26px;
        font-weight: 800;
        margin: 0;
    }
    .header-date {
        color: #64748b;
        font-size: 13px;
        background: rgba(255,255,255,0.04);
        padding: 6px 14px;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* CROP CATEGORY BADGE */
    .cat-badge {
        display: inline-block;
        background: rgba(59,130,246,0.12);
        color: #93c5fd;
        border: 1px solid rgba(59,130,246,0.2);
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        margin: 3px 4px;
        transition: all 0.2s;
    }
    .cat-badge:hover {
        background: rgba(59,130,246,0.22);
        transform: scale(1.03);
    }
    .cat-badge.veg   { background: rgba(34,197,94,0.12); color: #86efac; border-color: rgba(34,197,94,0.2); }
    .cat-badge.spice { background: rgba(245,158,11,0.12); color: #fcd34d; border-color: rgba(245,158,11,0.2); }
    .cat-badge.pulse { background: rgba(168,85,247,0.12); color: #d8b4fe; border-color: rgba(168,85,247,0.2); }
    .cat-badge.oil   { background: rgba(6,182,212,0.12); color: #67e8f9; border-color: rgba(6,182,212,0.2); }
    .cat-badge.cash  { background: rgba(244,63,94,0.12); color: #fda4af; border-color: rgba(244,63,94,0.2); }
    .cat-badge.fruit { background: rgba(251,191,36,0.12); color: #fde68a; border-color: rgba(251,191,36,0.2); }

    /* PRECAUTION CARD */
    .precaution-card {
        background: linear-gradient(135deg, #1a2332 0%, #1e293b 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 14px;
    }
    .precaution-card h4 {
        color: #e2e8f0;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .precaution-card li {
        color: #94a3b8;
        font-size: 13.5px;
        margin-bottom: 6px;
        line-height: 1.5;
    }

    /* SUGGESTION BOX */
    .suggestion-box {
        background: linear-gradient(135deg, #0f3325 0%, #14412e 100%);
        border: 1px solid rgba(34,197,94,0.2);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
    }
    .suggestion-box h3 {
        color: #86efac;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .suggestion-box .advice {
        color: #a7f3d0;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 14px;
        font-style: italic;
    }

    /* TABLE */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* BUTTON */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.3px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 20px rgba(59,130,246,0.4);
        transform: translateY(-1px);
    }

    /* INPUTS */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        border-radius: 10px !important;
    }

    /* DIVIDER */
    .soft-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 24px 0;
    }

    /* WELCOME BANNER */
    .welcome-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a2744 100%);
        border: 1px solid rgba(59,130,246,0.15);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .welcome-banner .wb-emoji { font-size: 36px; }
    .welcome-banner .wb-text h3 {
        color: #e2e8f0;
        font-size: 18px;
        font-weight: 700;
        margin: 0 0 4px 0;
    }
    .welcome-banner .wb-text p {
        color: #94a3b8;
        font-size: 13px;
        margin: 0;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def make_kpi(label, value, sub, icon, color):
    return (
        f'<div class="kpi-card {color}">'
        f'<div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


def get_rainfall_zone(rainfall):
    if rainfall < 500:
        return "very_low"
    elif rainfall < 750:
        return "low"
    elif rainfall < 1150:
        return "moderate"
    elif rainfall < 2000:
        return "high"
    else:
        return "very_high"


# =====================================================
# AUTHENTICATION
# =====================================================
USERS_DB = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "farmer": hashlib.sha256("crop2026".encode()).hexdigest(),
    "demo": hashlib.sha256("demo".encode()).hexdigest(),
}


def check_login(username, password):
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    return USERS_DB.get(username.lower()) == pwd_hash


def login_page():
    inject_css()

    # Hide sidebar & style the full page background
    st.markdown(
        '<style>'
        'section[data-testid="stSidebar"]{display:none;}'
        '[data-testid="stHeader"]{background:transparent;}'
        '[data-testid="stAppViewContainer"]{'
        '  background: radial-gradient(ellipse at 30% 20%, rgba(59,130,246,0.12) 0%, transparent 50%),'
        '              radial-gradient(ellipse at 70% 80%, rgba(168,85,247,0.10) 0%, transparent 50%),'
        '              linear-gradient(160deg, #0b1120 0%, #101829 40%, #0f1724 100%);'
        '}'
        '.block-container{padding-top:6vh !important; max-width:100% !important;}'
        '</style>',
        unsafe_allow_html=True,
    )

    # Push content to center using columns
    _, col_center, _ = st.columns([1.1, 1, 1.1])

    with col_center:
        # Brand header
        st.markdown(
            '<div class="login-header">'
            '<div class="login-icon-ring">&#127806;</div>'
            '<h1>CropVision</h1>'
            '<p>Yield Analytics &amp; Smart Farming Dashboard</p>'
            '</div>'
            '<div class="login-shimmer-bar"></div>',
            unsafe_allow_html=True,
        )

        # Form inputs
        username = st.text_input("Username", placeholder="e.g. admin")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

        col_btn, col_info = st.columns([1.5, 1])
        with col_btn:
            login_clicked = st.button("Sign In", width="stretch")
        with col_info:
            with st.popover("Demo Accounts"):
                st.markdown(
                    "| Username | Password |\n"
                    "|----------|----------|\n"
                    "| `admin` | `admin123` |\n"
                    "| `farmer` | `crop2026` |\n"
                    "| `demo` | `demo` |"
                )

        if login_clicked:
            if check_login(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")

        # Feature icons row
        st.markdown(
            '<div class="login-features">'
            '<div class="login-feat-item"><div class="feat-icon">&#128202;</div><div class="feat-label">Analytics</div></div>'
            '<div class="login-feat-item"><div class="feat-icon">&#129302;</div><div class="feat-label">AI Predictions</div></div>'
            '<div class="login-feat-item"><div class="feat-icon">&#127783;&#65039;</div><div class="feat-label">Rainfall</div></div>'
            '<div class="login-feat-item"><div class="feat-icon">&#9888;&#65039;</div><div class="feat-label">Precautions</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Footer
        st.markdown(
            '<div class="login-footer"><p>CropVision v2.0 &bull; 2026 AgriTech</p></div>',
            unsafe_allow_html=True,
        )


# =====================================================
# LOAD MODELS & DATA
# =====================================================
@st.cache_resource
def load_all():
    reg_model = joblib.load("best_reg_model.pkl")
    clf_model = joblib.load("best_classification_model.pkl")
    scaler = joblib.load("scaler_reg.pkl")
    le_state = joblib.load("le_state.pkl")
    le_crop = joblib.load("le_item.pkl")

    df = pd.read_csv("./crop-yield-in-indian-states-dataset/crop_yield.csv")
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True, errors="ignore")

    for col in ["Crop", "Season", "State"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    return reg_model, clf_model, scaler, le_state, le_crop, df


# =====================================================
# MAIN APP ENTRY
# =====================================================
def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login_page()
        return

    reg_model, clf_model, scaler, le_state, le_crop, df = load_all()

    inject_css()

    # Sidebar
    with st.sidebar:
        st.markdown("# &#127806; CropVision")
        uname = st.session_state.get("username", "user")
        st.markdown(
            f'<p style="color:#64748b;font-size:12px;">Logged in as '
            f'<b style="color:#93c5fd;">{uname}</b></p>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Predict Yield",
                "Rainfall Advisor",
                "Crop Categories",
                "Precautions",
            ],
            label_visibility="collapsed",
        )

        st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

        if st.button("Sign Out", width="stretch"):
            st.session_state["authenticated"] = False
            st.session_state["username"] = ""
            st.rerun()

        st.markdown("")
        st.markdown(
            '<div style="position:fixed;bottom:20px;left:20px;">'
            '<p style="color:#475569;font-size:11px;">CropVision v2.0<br>2026 AgriTech</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    if page == "Dashboard":
        render_dashboard(df)
    elif page == "Predict Yield":
        render_predict(df, reg_model, scaler, le_crop, le_state)
    elif page == "Rainfall Advisor":
        render_rainfall_advisor(df)
    elif page == "Crop Categories":
        render_crop_categories(df)
    elif page == "Precautions":
        render_precautions()


# =====================================================
# DASHBOARD
# =====================================================
def render_dashboard(df):
    today = datetime.now().strftime("%d %B %Y")
    st.markdown(
        f'<div class="header-bar">'
        f'<h2>&#128202; Agricultural Analytics</h2>'
        f'<div class="header-date">&#128197; {today}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    username = st.session_state.get("username", "User")
    st.markdown(
        f'<div class="welcome-banner">'
        f'<div class="wb-emoji">&#128075;</div>'
        f'<div class="wb-text">'
        f'<h3>Welcome back, {username.title()}!</h3>'
        f'<p>Here\'s your crop analytics overview with insights across Indian states.</p>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # KPI ROW 1
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            make_kpi("Average Yield", f"{df['Yield'].mean():.2f}", "hg/ha across all crops", "&#128200;", "blue"),
            unsafe_allow_html=True,
        )
    with c2:
        best_crop = df.groupby("Crop")["Yield"].mean().idxmax()
        st.markdown(make_kpi("Best Crop", best_crop, "Highest avg yield", "&#127942;", "green"), unsafe_allow_html=True)
    with c3:
        best_state = df.groupby("State")["Yield"].mean().idxmax()
        st.markdown(make_kpi("Top State", best_state, "By average yield", "&#128506;", "amber"), unsafe_allow_html=True)
    with c4:
        st.markdown(
            make_kpi("Avg Rainfall", f"{int(df['Annual_Rainfall'].mean()):,} mm", "Annual average", "&#127783;&#65039;", "cyan"),
            unsafe_allow_html=True,
        )

    st.markdown("")

    # KPI ROW 2
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(
            make_kpi("Total Records", f"{len(df):,}", "Data entries", "&#128203;", "purple"), unsafe_allow_html=True
        )
    with c6:
        st.markdown(
            make_kpi("Crop Types", str(df["Crop"].nunique()), "Unique crops tracked", "&#127807;", "green"),
            unsafe_allow_html=True,
        )
    with c7:
        st.markdown(
            make_kpi("States", str(df["State"].nunique()), "Covered regions", "&#127963;&#65039;", "rose"),
            unsafe_allow_html=True,
        )
    with c8:
        st.markdown(
            make_kpi("Avg Fertilizer", f"{df['Fertilizer'].mean():,.0f}", "tonnes used", "&#129514;", "amber"),
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # YIELD TREND
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Yield Trend Over Years")

    yearly = df.groupby("Crop_Year")["Yield"].mean().reset_index()
    area_fig = px.area(
        yearly, x="Crop_Year", y="Yield",
        template="plotly_dark",
        labels={"Yield": "Yield (hg/ha)", "Crop_Year": "Year"},
        color_discrete_sequence=["#3b82f6"],
    )
    area_fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    area_fig.update_traces(line=dict(width=2.5), fillcolor="rgba(59,130,246,0.15)")
    st.plotly_chart(area_fig, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    # DONUT + BAR
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Yield by Season")
        season_avg = df.groupby("Season")["Yield"].mean().reset_index()
        donut_fig = px.pie(
            season_avg, names="Season", values="Yield",
            hole=0.65, template="plotly_dark",
            color_discrete_sequence=["#3b82f6", "#22c55e", "#f59e0b", "#f43f5e", "#a855f7", "#06b6d4"],
        )
        donut_fig.update_layout(
            height=370, showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            legend=dict(font=dict(color="#94a3b8", size=12)),
        )
        st.plotly_chart(donut_fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Top 10 Crops by Yield")
        crop_avg = df.groupby("Crop")["Yield"].mean().sort_values(ascending=False).head(10).reset_index()
        bar_fig = px.bar(
            crop_avg, x="Yield", y="Crop", orientation="h",
            template="plotly_dark",
            labels={"Yield": "Yield (hg/ha)", "Crop": ""},
            color="Yield",
            color_continuous_scale=["#1e3a5f", "#3b82f6", "#60a5fa"],
        )
        bar_fig.update_layout(
            height=370,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            showlegend=False, coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(bar_fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # SCATTER + STATE BAR
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Rainfall vs Yield Correlation")
        sample = df.sample(min(2000, len(df)), random_state=42)
        scatter_fig = px.scatter(
            sample, x="Annual_Rainfall", y="Yield",
            template="plotly_dark", opacity=0.5,
            color_discrete_sequence=["#60a5fa"],
            labels={"Annual_Rainfall": "Rainfall (mm)", "Yield": "Yield (hg/ha)"},
        )
        scatter_fig.update_layout(
            height=370,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(scatter_fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Top 10 States by Yield")
        state_avg = df.groupby("State")["Yield"].mean().sort_values(ascending=False).head(10).reset_index()
        state_fig = px.bar(
            state_avg, x="State", y="Yield",
            template="plotly_dark",
            labels={"Yield": "Yield (hg/ha)", "State": ""},
            color="Yield",
            color_continuous_scale=["#14532d", "#22c55e", "#86efac"],
        )
        state_fig.update_layout(
            height=370,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            showlegend=False, coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickangle=-45),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(state_fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # DATA TABLE
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Top Producing Crop-State Combinations")
    top_combos = (
        df.groupby(["Crop", "State", "Season"])["Yield"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    top_combos.columns = ["Crop", "State", "Season", "Avg Yield (hg/ha)"]
    top_combos["Avg Yield (hg/ha)"] = top_combos["Avg Yield (hg/ha)"].round(2)
    st.dataframe(top_combos, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# PREDICT YIELD
# =====================================================
def render_predict(df, reg_model, scaler, le_crop, le_state):
    st.markdown(
        '<div class="header-bar">'
        '<h2>&#127806; Crop Yield Prediction</h2>'
        '<div class="header-date">ML-Powered Forecast</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="welcome-banner">'
        '<div class="wb-emoji">&#129302;</div>'
        '<div class="wb-text">'
        '<h3>Smart Yield Forecast</h3>'
        '<p>Enter agricultural parameters below to get an AI-powered yield prediction along with intelligent crop suggestions.</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Crop Details")
        crop = st.selectbox("Crop", sorted(le_crop.classes_))
        season = st.selectbox("Season", ["Kharif", "Rabi", "Summer", "Winter", "Whole Year", "Autumn"])
        year = st.slider("Crop Year", 1990, 2030, 2025)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Location & Weather")
        state = st.selectbox("State", sorted(le_state.classes_))
        rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=1000.0, step=50.0)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Inputs Applied")
        fertilizer = st.number_input("Fertilizer (tonnes)", min_value=0.0, value=100.0, step=10.0)
        pesticide = st.number_input("Pesticide (litres)", min_value=0.0, value=50.0, step=5.0)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    season_map = {"Kharif": 0, "Rabi": 1, "Summer": 2, "Winter": 3, "Whole Year": 4, "Autumn": 5}

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("Predict Yield", width="stretch")

    if predict_clicked:
        input_df = pd.DataFrame([{
            "Crop": le_crop.transform([crop])[0],
            "Crop_Year": year,
            "Season": season_map.get(season, 0),
            "State": le_state.transform([state])[0],
            "Annual_Rainfall": rainfall,
            "Fertilizer": fertilizer,
            "Pesticide": pesticide,
        }])
        input_scaled = scaler.transform(input_df)
        pred = reg_model.predict(input_scaled)

        st.markdown("")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                make_kpi("Predicted Yield", f"{pred[0]:.2f}", "hg/ha (hectogram per hectare)", "&#127919;", "green"),
                unsafe_allow_html=True,
            )
        with r2:
            zone = get_rainfall_zone(rainfall)
            info = RAINFALL_CROP_MAP[zone]
            st.markdown(
                make_kpi("Rainfall Zone", info["label"], info["range"], "&#127783;&#65039;", "cyan"),
                unsafe_allow_html=True,
            )
        with r3:
            q67 = df["Yield"].quantile(0.67)
            q33 = df["Yield"].quantile(0.33)
            if pred[0] > q67:
                quality = "High"
                quality_icon = "&#128994;"
            elif pred[0] > q33:
                quality = "Medium"
                quality_icon = "&#128993;"
            else:
                quality = "Low"
                quality_icon = "&#128308;"
            st.markdown(
                make_kpi("Yield Rating", f"{quality_icon} {quality}", "Based on historical data", "&#11088;", "amber"),
                unsafe_allow_html=True,
            )

        st.markdown("")
        zone = get_rainfall_zone(rainfall)
        info = RAINFALL_CROP_MAP[zone]
        badges = "".join(f'<span class="cat-badge">{c}</span>' for c in info["crops"])
        st.markdown(
            f'<div class="suggestion-box">'
            f'<h3>&#127793; Smart Crop Suggestions for {info["range"]} Rainfall</h3>'
            f'<div class="advice">&#128161; {info["advice"]}</div>'
            f'<div>{badges}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# =====================================================
# RAINFALL ADVISOR
# =====================================================
def render_rainfall_advisor(df):
    st.markdown(
        '<div class="header-bar">'
        '<h2>&#127783;&#65039; Rainfall-Based Crop Advisor</h2>'
        '<div class="header-date">Smart Recommendations</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="welcome-banner">'
        '<div class="wb-emoji">&#129504;</div>'
        '<div class="wb-text">'
        '<h3>Intelligent Sowing Suggestions</h3>'
        '<p>Select your region\'s expected annual rainfall to get data-driven crop recommendations and farming advice.</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    rainfall = st.slider("Expected Annual Rainfall (mm)", 100, 4000, 1000, step=50)

    zone = get_rainfall_zone(rainfall)
    info = RAINFALL_CROP_MAP[zone]

    badges = "".join(f'<span class="cat-badge">{c}</span>' for c in info["crops"])
    st.markdown(
        f'<div class="suggestion-box">'
        f'<h3>{info["label_icon"]} {info["label"]} &mdash; {info["range"]}</h3>'
        f'<div class="advice">&#128161; {info["advice"]}</div>'
        f'<div style="margin-top:12px;">'
        f'<strong style="color:#86efac;font-size:14px;">Recommended Crops:</strong><br><br>'
        f'{badges}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Crops Performance at Similar Rainfall Levels")

    tolerance = 200
    filtered = df[
        (df["Annual_Rainfall"] >= rainfall - tolerance) & (df["Annual_Rainfall"] <= rainfall + tolerance)
    ]

    if len(filtered) > 10:
        perf = filtered.groupby("Crop")["Yield"].agg(["mean", "count"]).reset_index()
        perf.columns = ["Crop", "Avg Yield (hg/ha)", "Data Points"]
        perf = perf[perf["Data Points"] >= 3].sort_values("Avg Yield (hg/ha)", ascending=False).head(15)
        perf["Avg Yield (hg/ha)"] = perf["Avg Yield (hg/ha)"].round(2)

        bar_fig = px.bar(
            perf, x="Avg Yield (hg/ha)", y="Crop", orientation="h",
            template="plotly_dark",
            color="Avg Yield (hg/ha)",
            color_continuous_scale=["#14532d", "#22c55e", "#86efac"],
        )
        bar_fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            showlegend=False, coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(bar_fig, width="stretch")
        st.dataframe(perf, width="stretch", hide_index=True)
    else:
        st.info("Not enough data points in this rainfall range. Try adjusting the slider.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### All Rainfall Zones Reference")

    for _key, val in RAINFALL_CROP_MAP.items():
        zone_badges = "".join(f'<span class="cat-badge">{c}</span>' for c in val["crops"])
        st.markdown(
            f'<div class="precaution-card">'
            f'<h4>{val["label_icon"]} {val["label"]} &mdash; {val["range"]}</h4>'
            f'<p style="color:#a7f3d0;font-size:13px;font-style:italic;margin-bottom:10px;">&#128161; {val["advice"]}</p>'
            f'<div>{zone_badges}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# CROP CATEGORIES
# =====================================================
def render_crop_categories(df):
    st.markdown(
        '<div class="header-bar">'
        '<h2>&#128194; Crop Categories</h2>'
        '<div class="header-date">Browse by Type</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="welcome-banner">'
        '<div class="wb-emoji">&#128450;</div>'
        '<div class="wb-text">'
        '<h3>Category-wise Crop Explorer</h3>'
        '<p>Browse crops organized by type: Vegetables, Spices, Cereals, Pulses, Oilseeds, Cash Crops &amp; Fruits.</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    tab_labels = [f"{v['icon']} {k}" for k, v in CROP_CATEGORIES.items()]
    tabs = st.tabs(tab_labels)

    for tab, (cat_name, cat_info) in zip(tabs, CROP_CATEGORIES.items()):
        with tab:
            crops = cat_info["crops"]
            badge_cls = cat_info["badge_cls"]

            badges_html = "".join(f'<span class="cat-badge {badge_cls}">{c}</span>' for c in crops)
            st.markdown(
                f'<div class="section-card">'
                f'<h3>{cat_info["icon"]} {cat_name}</h3>'
                f'<div style="margin-bottom:16px;">{badges_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            available_crops = [c for c in crops if c in df["Crop"].values]
            if available_crops:
                cat_data = df[df["Crop"].isin(available_crops)]
                cat_summary = (
                    cat_data.groupby("Crop")
                    .agg(Avg_Yield=("Yield", "mean"), Avg_Rainfall=("Annual_Rainfall", "mean"), Records=("Yield", "count"))
                    .sort_values("Avg_Yield", ascending=False)
                    .reset_index()
                )
                cat_summary["Avg_Yield"] = cat_summary["Avg_Yield"].round(2)
                cat_summary["Avg_Rainfall"] = cat_summary["Avg_Rainfall"].round(0).astype(int)
                cat_summary.columns = ["Crop", "Avg Yield (hg/ha)", "Avg Rainfall (mm)", "Data Points"]

                col_chart, col_table = st.columns([1.2, 1])

                with col_chart:
                    cat_bar = px.bar(
                        cat_summary, x="Avg Yield (hg/ha)", y="Crop",
                        orientation="h", template="plotly_dark",
                        color_discrete_sequence=["#3b82f6"],
                    )
                    cat_bar.update_layout(
                        height=max(200, len(cat_summary) * 40),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94a3b8"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    )
                    st.plotly_chart(cat_bar, width="stretch")

                with col_table:
                    st.dataframe(cat_summary, width="stretch", hide_index=True)
            else:
                st.info(f"No matching data for {cat_name} crops in the dataset.")


# =====================================================
# PRECAUTIONS
# =====================================================
def render_precautions():
    st.markdown(
        '<div class="header-bar">'
        '<h2>&#9888;&#65039; Agricultural Precautions</h2>'
        '<div class="header-date">Best Practices</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="welcome-banner">'
        '<div class="wb-emoji">&#128737;&#65039;</div>'
        '<div class="wb-text">'
        '<h3>Farming Best Practices &amp; Safety Guidelines</h3>'
        '<p>Essential precautions for soil health, pest management, fertilizer use, and seasonal planning.</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    items = list(PRECAUTIONS.items())
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(items):
                title, data = items[idx]
                tips_html = "".join(f"<li>{t}</li>" for t in data["tips"])
                with col:
                    st.markdown(
                        f'<div class="precaution-card">'
                        f'<h4>{data["icon"]} {title}</h4>'
                        f'<ul style="padding-left:18px;margin:0;">{tips_html}</ul>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-card">'
        '<h3>Key Takeaways</h3>'
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:12px;">'
        '<div style="text-align:center;padding:16px;">'
        '<div style="font-size:32px;margin-bottom:8px;">&#129514;</div>'
        '<div style="color:#e2e8f0;font-weight:700;font-size:14px;margin-bottom:4px;">Test Before You Apply</div>'
        '<div style="color:#64748b;font-size:12px;">Always conduct soil testing before applying fertilizers or pesticides.</div>'
        '</div>'
        '<div style="text-align:center;padding:16px;">'
        '<div style="font-size:32px;margin-bottom:8px;">&#128260;</div>'
        '<div style="color:#e2e8f0;font-weight:700;font-size:14px;margin-bottom:4px;">Rotate Your Crops</div>'
        '<div style="color:#64748b;font-size:12px;">Alternate between cereals and legumes to maintain soil fertility.</div>'
        '</div>'
        '<div style="text-align:center;padding:16px;">'
        '<div style="font-size:32px;margin-bottom:8px;">&#128197;</div>'
        '<div style="color:#e2e8f0;font-weight:700;font-size:14px;margin-bottom:4px;">Plan by Season</div>'
        '<div style="color:#64748b;font-size:12px;">Match crop selection to Kharif, Rabi, or Zaid seasons for best results.</div>'
        '</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    main()

