import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import KNNImputer


st.set_page_config(
    page_title="Cardia · Heart Intelligence",
    page_icon="♥",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    :root {
        --ink: #f7f8fb;
        --muted: #99a5b8;
        --line: rgba(255,255,255,.09);
        --paper: #121a27;
        --paper-raised: #172131;
        --canvas: #090e16;
        --navy: #0d1522;
        --coral: #ff6f6a;
        --coral-soft: rgba(255,111,106,.12);
        --teal: #39c5ad;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: radial-gradient(circle at 80% 0%, #152237 0, var(--canvas) 34rem); color: var(--ink); }
    .block-container { max-width: 1440px; padding: 2.2rem 3rem 4rem; }
    h1, h2, h3, h4 { font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; }
    h1 { color: var(--ink); font-size: 2.6rem !important; line-height: 1.08 !important; }
    h2 { color: var(--ink); }
    p, .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111c2c 0%, #0b121d 100%);
        border-right: 1px solid var(--line);
        min-width: 270px;
    }
    [data-testid="stSidebar"] > div:first-child { padding: 1.8rem 1.15rem; }
    [data-testid="stSidebar"] * { color: #e9eef5; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: .72rem .8rem;
        border-radius: 12px;
        margin: .16rem 0;
        transition: .2s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.08); }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: rgba(235,106,101,.18);
        box-shadow: inset 3px 0 0 var(--coral);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
    [data-testid="stSidebar"] .stRadio > div { gap: .2rem; }
    [data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child { display: none; }

    .brand { display: flex; align-items: center; gap: .75rem; margin: .1rem .35rem 2.1rem; }
    .brand-mark {
        width: 38px; height: 38px; border-radius: 12px; display: grid; place-items: center;
        background: linear-gradient(145deg, #f17b76, #d94e56); color: white; font-size: 1.1rem;
        box-shadow: 0 8px 20px rgba(235,106,101,.25);
    }
    .brand-name { font-family: 'Manrope'; font-size: 1.14rem; font-weight: 800; color: white; }
    .brand-sub { color: #9daabe; font-size: .69rem; text-transform: uppercase; letter-spacing: .12em; }
    .side-label { margin: 0 .55rem .7rem; color: #7f8da3; font-size: .68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .14em; }
    .side-status {
        position: relative; margin-top: 2.25rem; padding: 1rem;
        background: rgba(255,255,255,.055); border: 1px solid rgba(255,255,255,.08); border-radius: 14px;
    }
    .status-dot { width: 7px; height: 7px; border-radius: 99px; background: #51d5b7; display: inline-block; margin-right: .45rem; box-shadow: 0 0 0 4px rgba(81,213,183,.1); }
    .side-status strong { font-size: .78rem; color: #f3f6f9; }
    .side-status p { margin: .4rem 0 0; font-size: .7rem; line-height: 1.45; color: #8e9cb0; }

    .topbar { display:flex; justify-content:space-between; align-items:center; margin-bottom:1.7rem; }
    .eyebrow { color: var(--coral); font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.13em; margin-bottom:.45rem; }
    .top-title { margin:0; font-family:'Manrope'; font-size:1.45rem; font-weight:800; color:var(--ink); letter-spacing:-.035em; }
    .top-date { background:rgba(255,255,255,.045); border:1px solid var(--line); border-radius:12px; padding:.58rem .85rem; font-size:.78rem; color:#aab6c8; box-shadow:0 10px 28px rgba(0,0,0,.16); }

    .hero {
        position: relative; overflow: hidden; border-radius: 24px; padding: 2.35rem 2.5rem;
        background: linear-gradient(115deg, #151f30 0%, #1b2e43 58%, #17403f 130%);
        border:1px solid rgba(255,255,255,.08); box-shadow: 0 24px 60px rgba(0,0,0,.28); margin-bottom: 1.35rem;
    }
    .hero:after { content:""; position:absolute; width:330px; height:330px; right:-70px; top:-120px; border-radius:50%; border:60px solid rgba(255,255,255,.035); }
    .hero-kicker { color:#f49691; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; font-weight:800; }
    .hero h1 { color:white !important; max-width:690px; margin:.6rem 0 .75rem; }
    .hero p { color:#b9c5d3; max-width:610px; margin:0; line-height:1.65; font-size:.95rem; }
    .hero-chip { display:inline-flex; margin-top:1.35rem; gap:.5rem; align-items:center; padding:.48rem .72rem; border:1px solid rgba(255,255,255,.13); border-radius:99px; color:#e7edf3; font-size:.73rem; background:rgba(255,255,255,.06); }

    .metric-card { background:linear-gradient(145deg,var(--paper-raised),var(--paper)); border:1px solid var(--line); border-radius:18px; padding:1.25rem 1.35rem; min-height:128px; box-shadow:0 16px 40px rgba(0,0,0,.2); }
    .metric-icon { width:34px; height:34px; border-radius:10px; display:grid; place-items:center; background:var(--coral-soft); color:var(--coral); font-weight:800; margin-bottom:.75rem; }
    .metric-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.085em; color:#8f9caf; font-weight:700; }
    .metric-value { font-family:'Manrope'; font-size:1.55rem; font-weight:800; color:var(--ink); letter-spacing:-.04em; margin-top:.22rem; }
    .metric-note { font-size:.72rem; color:#758397; margin-top:.18rem; }

    .section-head { margin: 2rem 0 1rem; }
    .section-head h2 { margin:0; font-size:1.45rem; }
    .section-head p { margin:.35rem 0 0; font-size:.86rem; }
    div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"] { background:linear-gradient(145deg,var(--paper-raised),var(--paper)); border:1px solid var(--line); border-radius:18px; padding:.65rem; box-shadow:0 18px 44px rgba(0,0,0,.22); overflow:hidden; }

    /* Streamlit controls: explicit dark surfaces and readable values */
    [data-testid="stWidgetLabel"] p, label, .stSlider p { color:#c3ccda !important; font-weight:600 !important; }
    div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] > div {
        background:#0d1521 !important; border:1px solid #293548 !important; border-radius:11px !important;
        color:#f4f6fa !important; box-shadow:none !important;
    }
    div[data-baseweb="select"] *, div[data-testid="stNumberInput"] input {
        color:#f4f6fa !important; -webkit-text-fill-color:#f4f6fa !important;
    }
    div[data-testid="stNumberInput"] input { background:#0d1521 !important; border:0 !important; }
    div[data-testid="stNumberInput"] button { background:#182334 !important; color:#dbe2ec !important; border-color:#293548 !important; }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background:#111a27 !important; border:1px solid #2b3749 !important; color:#f4f6fa !important;
    }
    li[role="option"] { color:#dce3ed !important; background:#111a27 !important; }
    li[role="option"]:hover, li[aria-selected="true"] { background:#202b3b !important; color:#fff !important; }
    [data-testid="stSelectbox"] svg { fill:#96a4b8 !important; }
    div[data-testid="stSlider"] { padding-top:.2rem; }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background:#263245; }
    div[data-testid="stSlider"] [role="slider"] { background:var(--coral) !important; border-color:var(--coral) !important; }
    .stButton > button {
        width:100%; border:0; border-radius:12px; padding:.78rem 1rem; color:white;
        background:linear-gradient(120deg, #eb6a65, #dc555e); font-weight:700;
        box-shadow:0 10px 22px rgba(235,106,101,.22); transition:.2s ease;
    }
    .stButton > button:hover { transform:translateY(-1px); color:white; border:0; box-shadow:0 13px 26px rgba(235,106,101,.28); }
    [data-testid="stForm"] { background:linear-gradient(145deg,var(--paper-raised),#101824); border:1px solid var(--line); border-radius:20px; padding:1.25rem 1.35rem 1.45rem; box-shadow:0 20px 50px rgba(0,0,0,.24); }
    [data-testid="stForm"] h3 { color:#f7f8fb !important; }

    .insight-card { background:linear-gradient(145deg,var(--paper-raised),var(--paper)); border:1px solid var(--line); border-radius:18px; padding:1.2rem 1.3rem; min-height:112px; box-shadow:0 14px 34px rgba(0,0,0,.18); }
    .insight-card strong { color:var(--ink); font-size:.9rem; display:block; margin-bottom:.4rem; }
    .insight-card p { margin:0; font-size:.78rem; line-height:1.55; }
    .result-card { border-radius:20px; padding:1.5rem 1.6rem; color:white; margin-top:1.1rem; box-shadow:0 15px 34px rgba(23,38,61,.12); }
    .result-low { background:linear-gradient(120deg,#177f73,#29a58f); }
    .result-high { background:linear-gradient(120deg,#c94652,#ed6e66); }
    .result-label { opacity:.75; text-transform:uppercase; letter-spacing:.12em; font-size:.68rem; font-weight:700; }
    .result-score { font-family:'Manrope'; font-size:2.1rem; font-weight:800; margin:.25rem 0; }
    .result-card p { color:rgba(255,255,255,.82); margin:0; font-size:.82rem; }
    .disclaimer { margin-top:1rem; padding:.8rem 1rem; background:#211d13; border:1px solid #403821; border-radius:12px; color:#d4bd7d; font-size:.72rem; line-height:1.5; }

    /* Keep Streamlit's viewport scrollable and preserve sidebar controls. */
    html, body, #root, .stApp { min-height:100%; }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        overflow-y:auto !important;
        overscroll-behavior-y:contain;
    }
    [data-testid="stHeader"] {
        visibility:visible !important;
        background:linear-gradient(180deg,rgba(9,14,22,.96),rgba(9,14,22,.72),transparent) !important;
        pointer-events:auto !important;
    }
    [data-testid="stHeader"] button,
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stExpandSidebarButton"] {
        color:#dce4ee !important;
        background:#172131 !important;
        border:1px solid rgba(255,255,255,.12) !important;
        border-radius:10px !important;
        box-shadow:0 8px 24px rgba(0,0,0,.24) !important;
    }
    [data-testid="stHeader"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="stExpandSidebarButton"]:hover {
        color:white !important;
        background:#233147 !important;
        border-color:rgba(255,111,106,.45) !important;
    }
    [data-testid="stHeader"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stExpandSidebarButton"] svg { fill:currentColor !important; }

    #MainMenu, footer { visibility:hidden; }
    @media (max-width: 900px) { .block-container { padding:1.4rem 1rem 3rem; } .hero { padding:1.7rem 1.4rem; } .top-date { display:none; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_clean_data():
    data = pd.read_csv("heart.csv")
    cat_mapping = {
        "Sex": {"M": 0, "F": 1},
        "ChestPainType": {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3},
        "RestingECG": {"Normal": 0, "ST": 1, "LVH": 2},
        "ExerciseAngina": {"N": 0, "Y": 1},
        "ST_Slope": {"Up": 0, "Flat": 1, "Down": 2},
    }
    for column, mapping in cat_mapping.items():
        data[column] = data[column].map(mapping)
    data["Cholesterol"] = data["Cholesterol"].replace(0, np.nan)
    data["RestingBP"] = data["RestingBP"].replace(0, np.nan)
    imputer = KNNImputer(n_neighbors=3)
    clean = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    int_columns = clean.columns.drop("Oldpeak")
    clean[int_columns] = clean[int_columns].astype("int32")
    return clean


df = load_and_clean_data()

st.sidebar.markdown(
    """
    <div class="brand">
      <div class="brand-mark">♥</div>
      <div><div class="brand-name">Cardia</div><div class="brand-sub">Heart intelligence</div></div>
    </div>
    <div class="side-label">Workspace</div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio("Navigation", ["Overview", "Analytics", "Risk Assessment"])
st.sidebar.markdown(
    """
    <div class="side-status">
      <strong><span class="status-dot"></span> Dataset ready</strong>
      <p>Patient records have been cleaned and securely prepared for analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def page_header(kicker, title):
    st.markdown(
        f"""
        <div class="topbar">
          <div><div class="eyebrow">{kicker}</div><div class="top-title">{title}</div></div>
          <div class="top-date">Clinical analytics workspace&nbsp; · &nbsp;2026</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(icon, label, value, note):
    st.markdown(
        f"""<div class="metric-card"><div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div><div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


def style_chart(fig, height=390):
    fig.update_layout(
        height=height,
        margin=dict(l=28, r=28, t=55, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#9ca9bc", size=12),
        title_font=dict(family="Manrope", color="#f4f6fa", size=17),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        hoverlabel=dict(bgcolor="#101824", font_color="white", bordercolor="#2b3749"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    return fig


if page == "Overview":
    page_header("Patient intelligence", "Clinical overview")
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Preventive care, powered by data</div>
          <h1>Understand heart health.<br>Act with confidence.</h1>
          <p>Explore population-level risk signals and run a guided patient assessment in one focused, clinically inspired workspace.</p>
          <div class="hero-chip">● &nbsp; 918 records analyzed &nbsp;·&nbsp; 12 health indicators</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prevalence = df["HeartDisease"].mean() * 100
    metrics = st.columns(4)
    with metrics[0]: metric_card("◎", "Patient records", f"{len(df):,}", "Complete observations")
    with metrics[1]: metric_card("↗", "Risk prevalence", f"{prevalence:.1f}%", "Positive cohort share")
    with metrics[2]: metric_card("⌁", "Median age", f"{df['Age'].median():.0f} yrs", "Across all patients")
    with metrics[3]: metric_card("♡", "Avg. max heart rate", f"{df['MaxHR'].mean():.0f}", "Beats per minute")

    st.markdown('<div class="section-head"><h2>Population snapshot</h2><p>A clear view of the data behind the assessment.</p></div>', unsafe_allow_html=True)
    left, right = st.columns([1.45, 1])
    with left:
        display_df = df.head(10).rename(columns={
            "RestingBP": "Resting BP", "ChestPainType": "Chest Pain", "HeartDisease": "Risk",
            "Cholesterol": "Cholesterol", "MaxHR": "Max HR"
        })
        st.dataframe(display_df, width="stretch", hide_index=True, height=390)
    with right:
        age_group = pd.cut(df["Age"], bins=[20, 39, 49, 59, 69, 100], labels=["20–39", "40–49", "50–59", "60–69", "70+"])
        age_risk = df.assign(AgeGroup=age_group).groupby("AgeGroup", observed=False)["HeartDisease"].mean().mul(100).reset_index()
        fig = px.bar(age_risk, x="AgeGroup", y="HeartDisease", title="Risk prevalence by age", color_discrete_sequence=["#eb6a65"])
        fig.update_traces(marker_cornerradius=7, hovertemplate="%{x}: %{y:.1f}%<extra></extra>")
        fig.update_yaxes(title="Prevalence (%)")
        fig.update_xaxes(title=None)
        st.plotly_chart(style_chart(fig), width="stretch", config={"displayModeBar": False})


elif page == "Analytics":
    page_header("Cohort exploration", "Visual analytics")
    st.markdown('<div class="section-head"><h2>Explore the signals</h2><p>Choose a lens to reveal patterns across the patient cohort.</p></div>', unsafe_allow_html=True)
    viz_type = st.selectbox("Analysis view", ["Risk distribution", "Age & risk profile", "Feature correlations"])

    if viz_type == "Risk distribution":
        counts = df["HeartDisease"].value_counts().rename(index={0: "Lower risk", 1: "Elevated risk"}).reset_index()
        counts.columns = ["Status", "Patients"]
        fig = px.pie(counts, values="Patients", names="Status", hole=.68, title="Patient risk distribution", color="Status", color_discrete_map={"Lower risk":"#1f9d8b", "Elevated risk":"#eb6a65"})
        fig.update_traces(textinfo="percent+label", textfont_size=13, marker=dict(line=dict(color="#121a27", width=5)), hovertemplate="%{label}: %{value} patients<extra></extra>")
        fig.add_annotation(text=f"<b>{len(df)}</b><br><span style='font-size:11px'>patients</span>", showarrow=False, font=dict(size=20, color="#f4f6fa"))
        st.plotly_chart(style_chart(fig, 480), width="stretch", config={"displayModeBar": False})
    elif viz_type == "Age & risk profile":
        plot_df = df.copy()
        plot_df["Status"] = plot_df["HeartDisease"].map({0:"Lower risk", 1:"Elevated risk"})
        fig = px.histogram(plot_df, x="Age", color="Status", barmode="overlay", nbins=30, title="Age distribution by risk status", color_discrete_map={"Lower risk":"#1f9d8b", "Elevated risk":"#eb6a65"}, opacity=.82)
        fig.update_xaxes(title="Patient age")
        fig.update_yaxes(title="Number of patients")
        st.plotly_chart(style_chart(fig, 480), width="stretch", config={"displayModeBar": False})
    else:
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, color_continuous_scale=[[0,"#174d59"],[.5,"#1a2331"],[1,"#ef6865"]], zmin=-1, zmax=1, aspect="auto", title="Feature correlation matrix")
        fig.update_traces(hovertemplate="%{x} × %{y}<br>Correlation: %{z:.2f}<extra></extra>")
        st.plotly_chart(style_chart(fig, 590), width="stretch", config={"displayModeBar": False})

    insights = st.columns(3)
    with insights[0]: st.markdown('<div class="insight-card"><strong>Age is only one signal</strong><p>Risk emerges from the interaction of several clinical and lifestyle indicators.</p></div>', unsafe_allow_html=True)
    with insights[1]: st.markdown('<div class="insight-card"><strong>Exercise data matters</strong><p>Max heart rate and exercise-induced angina provide valuable cardiovascular context.</p></div>', unsafe_allow_html=True)
    with insights[2]: st.markdown('<div class="insight-card"><strong>Context before conclusion</strong><p>Use cohort patterns to guide questions—not as a substitute for clinical review.</p></div>', unsafe_allow_html=True)


else:
    page_header("Guided screening", "Risk assessment")
    st.markdown(
        """<div class="hero" style="padding:1.8rem 2rem"><div class="hero-kicker">Patient assessment</div>
        <h1 style="font-size:2rem !important">Build a clearer risk picture.</h1>
        <p>Enter the patient’s current measurements. All fields are used to create an indicative cardiovascular risk score.</p></div>""",
        unsafe_allow_html=True,
    )

    with st.form("risk_form"):
        st.markdown("### Patient measurements")
        st.caption("Complete the fields below, then run the assessment.")
        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            age = st.slider("Age", 20, 100, 50)
            sex = st.selectbox("Biological sex", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
            cp = st.selectbox("Chest pain type", [0, 1, 2, 3], format_func=lambda x: ["Atypical angina", "Non-anginal pain", "Asymptomatic", "Typical angina"][x])
        with col2:
            bp = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 120)
            chol = st.number_input("Cholesterol (mg/dL)", 100, 650, 200)
            fbs = st.selectbox("Fasting blood sugar > 120 mg/dL", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col3:
            max_hr = st.slider("Maximum heart rate", 60, 220, 140)
            angina = st.selectbox("Exercise-induced angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            slope = st.selectbox("ST segment slope", [0, 1, 2], format_func=lambda x: ["Up-sloping", "Flat", "Down-sloping"][x])
        submitted = st.form_submit_button("Run heart risk assessment  →")

    if submitted:
        raw_score = (
            0.025 * (age - 40) + 0.018 * (bp - 120) + 0.004 * (chol - 180)
            + 0.85 * angina + 0.5 * (cp == 2) + 0.35 * (slope > 0)
            + 0.25 * fbs - 0.012 * (max_hr - 140) + 0.12 * (sex == 0)
        )
        probability = 1 / (1 + np.exp(-raw_score))
        elevated = probability >= .5
        css_class = "result-high" if elevated else "result-low"
        result = "Elevated risk pattern" if elevated else "Lower risk pattern"
        guidance = "This combination of indicators warrants timely clinical follow-up." if elevated else "Current inputs suggest a comparatively lower risk profile. Continue preventive care."
        st.markdown(
            f"""<div class="result-card {css_class}"><div class="result-label">Assessment complete</div>
            <div class="result-score">{probability:.0%} · {result}</div><p>{guidance}</p></div>
            <div class="disclaimer"><b>Important:</b> This screening result is illustrative and is not a medical diagnosis. Always consult a qualified healthcare professional for clinical interpretation and next steps.</div>""",
            unsafe_allow_html=True,
        )
