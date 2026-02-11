import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Crop Yield Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS (FINSET STYLE)
# =====================================================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 1.5rem;
}
.card {
    background: #161b22;
    border-radius: 18px;
    padding: 10px;
    height: 120px;
            
}
.card h3 {
    color: #9da5b4;
    font-size: 22px;
    margin-bottom: 6px;
}
.card h1 {
    color: white;
    font-size: 30px;
    margin: 0;
    margin-top : -32px;

}
.section {
    background: #161b22;
    border-radius: 20px;
    padding: 20px;
    margin-top: 20px;
}
.sidebar .sidebar-content {
    background: #0e1117;
}
</style>
""", unsafe_allow_html=True)

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


reg_model, clf_model, scaler, le_state, le_crop, df = load_all()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown("## 🌾 Crop Yield")
page = st.sidebar.radio("", ["📊 Dashboard", "🌾 Predict"])

# =====================================================
# DASHBOARD
# =====================================================
if page == "📊 Dashboard":

    st.markdown("## 📊 Agricultural Analytics")

    # ---------------- KPI CARDS ----------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="card">
            <h3>Average Yield</h3>
            <h1>{df['Yield'].mean():.2f} hg/ha</h1>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        best_crop = df.groupby("Crop")["Yield"].mean().idxmax()
        st.markdown(f"""
        <div class="card">
            <h3>Best Crop</h3>
            <h1>{best_crop}</h1>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        best_state = df.groupby("State")["Yield"].mean().idxmax()
        st.markdown(f"""
        <div class="card">
            <h3>Top State</h3>
            <h1>{best_state}</h1>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="card">
            <h3>Avg Rainfall</h3>
            <h1>{int(df['Annual_Rainfall'].mean())} mm</h1>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- AREA CHART ----------------
    yearly = df.groupby("Crop_Year")["Yield"].mean().reset_index()

    area_fig = px.area(
        yearly,
        x="Crop_Year",
        y="Yield",
        template="plotly_dark",
        labels={"Yield": "Yield (hg/ha)", "Crop_Year": "Year"}
    )
    area_fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📈 Yield Trend Over Years")
    st.plotly_chart(area_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- DONUT + BAR ----------------
    col1, col2 = st.columns(2)

    with col1:
        season_avg = df.groupby("Season")["Yield"].mean().reset_index()
        donut_fig = px.pie(
            season_avg,
            names="Season",
            values="Yield",
            hole=0.65,
            template="plotly_dark"
        )
        donut_fig.update_layout(height=360, showlegend=True)

        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### 🍩 Yield Contribution by Season")
        st.plotly_chart(donut_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        crop_avg = (
            df.groupby("Crop")["Yield"]
            .mean()
            .sort_values(ascending=False)
            .head(8)
            .reset_index()
        )

        bar_fig = px.bar(
            crop_avg,
            x="Yield",
            y="Crop",
            orientation="h",
            template="plotly_dark",
            labels={"Yield": "Yield (hg/ha)", "Crop": ""}
        )
        bar_fig.update_layout(height=360)

        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### 🌾 Top Crops by Yield")
        st.plotly_chart(bar_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- STATE PERFORMANCE ----------------
    state_avg = (
        df.groupby("State")["Yield"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    state_fig = px.bar(
        state_avg,
        x="State",
        y="Yield",
        template="plotly_dark",
        labels={"Yield": "Yield (hg/ha)", "State": ""}
    )
    state_fig.update_layout(height=360)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 🗺 State-wise Performance (Top 10)")
    st.plotly_chart(state_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# PREDICTION TAB
# =====================================================
else:

    st.markdown("## 🌾 Crop Yield Prediction")

    col1, col2 = st.columns(2)

    with col1:
        crop = st.selectbox("Crop", le_crop.classes_)
        season = st.selectbox("Season", ["Kharif", "Rabi", "Summer", "Winter", "Whole Year"])
        rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0)

    with col2:
        state = st.selectbox("State", le_state.classes_)
        fertilizer = st.number_input("Fertilizer Used (tonnes)", min_value=0.0)
        pesticide = st.number_input("Pesticide Used (ltr)", min_value=0.0)

    year = st.slider("Crop Year", 1990, 2030, 2025)

    season_map = {
        "Kharif": 0,
        "Rabi": 1,
        "Summer": 2,
        "Winter": 3,
        "Whole Year": 4
    }

    input_df = pd.DataFrame([{
        "Crop": le_crop.transform([crop])[0],
        "Crop_Year": year,
        "Season": season_map[season],
        "State": le_state.transform([state])[0],
        "Annual_Rainfall": rainfall,
        "Fertilizer": fertilizer,
        "Pesticide": pesticide
    }])

    input_scaled = scaler.transform(input_df)

    if st.button("Predict"):
        pred = reg_model.predict(input_scaled)
        st.success(f"🌾 Predicted Crop Yield: {pred[0]:.2f} hg/ha")
