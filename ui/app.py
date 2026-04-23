import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from core.graph import build_graph
from core.ml_pipeline import run_ml_pipeline, improve_model, get_last_predictions
from utils.feedback_store import save_feedback, get_feedback_summary
from utils.report_generator import generate_report

from sklearn.metrics import confusion_matrix, roc_curve, auc

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f5f6f8 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #1a202c !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu, footer,
.stDeployButton { display: none !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f0f2f5; }
::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 3px; }

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 40% at 10% 10%, rgba(66,153,225,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 90% 90%, rgba(99,179,237,0.05) 0%, transparent 60%),
        #f5f6f8;
    pointer-events: none;
    z-index: 0;
}

.block-container {
    max-width: 1280px !important;
    padding: 0 2.5rem 3rem !important;
    position: relative;
    z-index: 1;
}

/* ── TOP NAV BAR ── */
.topbar {
    background: #ffffff;
    border-bottom: 1px solid #e8edf2;
    padding: 0 2.5rem;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    margin: 0 -2.5rem 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    animation: slideDown 0.4s ease-out both;
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.topbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-icon {
    width: 30px; height: 30px;
    background: linear-gradient(135deg, #3182ce, #63b3ed);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    box-shadow: 0 2px 8px rgba(49,130,206,0.3);
}

.brand-name {
    font-size: 1rem;
    font-weight: 600;
    color: #1a202c;
    letter-spacing: -0.01em;
}

.brand-version {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #a0aec0;
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    padding: 2px 7px;
    border-radius: 4px;
    margin-left: 4px;
}

.topbar-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #38a169;
    background: #f0fff4;
    border: 1px solid #c6f6d5;
    padding: 4px 12px;
    border-radius: 100px;
}

.status-dot {
    width: 6px; height: 6px;
    background: #38a169;
    border-radius: 50%;
    animation: blink 2.5s ease-in-out infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── STEP PROGRESS ── */
.progress-bar {
    display: flex;
    align-items: center;
    background: #ffffff;
    border: 1px solid #e8edf2;
    border-radius: 12px;
    padding: 0.6rem 1.25rem;
    margin-bottom: 1.75rem;
    overflow-x: auto;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    animation: fadeUp 0.5s ease-out 0.2s both;
}

.step-item {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 0 14px;
    position: relative;
    white-space: nowrap;
}

.step-item:not(:last-child)::after {
    content: '';
    position: absolute;
    right: -1px; top: 50%; transform: translateY(-50%);
    width: 1px; height: 16px;
    background: #e2e8f0;
}

.step-num {
    width: 20px; height: 20px;
    border-radius: 50%;
    background: #edf2f7;
    color: #a0aec0;
    font-size: 0.6rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}

.step-num.active { background: #3182ce; color: #fff; box-shadow: 0 0 0 3px rgba(49,130,206,0.15); }
.step-num.done   { background: #38a169; color: #fff; }
.step-label      { font-size: 0.72rem; font-weight: 500; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.05em; }
.step-label.active { color: #3182ce; }
.step-label.done   { color: #38a169; }

/* ── SECTION HEADER ── */
.sec-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.5rem 0 1rem;
    animation: fadeUp 0.5s ease-out both;
}

.sec-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #a0aec0;
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 500;
}

.sec-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.sec-line { flex: 1; height: 1px; background: #e8edf2; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── WHITE CARD ── */
.w-card {
    background: #ffffff;
    border: 1px solid #e8edf2;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03), 0 4px 16px rgba(0,0,0,0.025);
    animation: fadeUp 0.5s ease-out both;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}

.w-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.04);
    border-color: #d1dce8;
}

.w-card-accent { border-top: 3px solid #3182ce; }

/* ── METRICS ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin-top: 0.75rem;
}

.m-card {
    background: #f7fafc;
    border: 1px solid #e8edf2;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
    transition: all 0.2s ease;
}

.m-card:hover { background: #ebf4ff; border-color: #bee3f8; transform: translateY(-1px); }

.m-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #a0aec0;
    margin-bottom: 5px;
}

.m-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    color: #2b6cb0;
}

/* ── BEST BANNER ── */
.best-banner {
    background: linear-gradient(to right, #ebf8ff, #f0fff4);
    border: 1px solid #bee3f8;
    border-left: 4px solid #3182ce;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 1rem 0;
    animation: fadeUp 0.4s ease-out 0.2s both;
}

.best-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3182ce;
    font-weight: 500;
}

.best-name { font-size: 1rem; font-weight: 600; color: #1a202c; margin-top: 1px; }

/* ── MODEL HEADER ── */
.model-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #f0f4f8;
}

.model-name { font-family: 'DM Mono', monospace; font-size: 0.82rem; font-weight: 500; color: #2d3748; }

.badge-best {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    background: #ebf4ff;
    border: 1px solid #bee3f8;
    color: #2b6cb0;
    padding: 2px 9px;
    border-radius: 100px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── SELECTS ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #ffffff !important;
    border: 1px solid #d1dce8 !important;
    border-radius: 9px !important;
    color: #2d3748 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    transition: border-color 0.2s !important;
}

[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stMultiSelect"] > div > div:hover {
    border-color: #3182ce !important;
    box-shadow: 0 0 0 3px rgba(49,130,206,0.1) !important;
}

[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stRadio"] label {
    color: #718096 !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    font-family: 'DM Mono', monospace !important;
}

[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: #ebf4ff !important;
    color: #2b6cb0 !important;
    border: 1px solid #bee3f8 !important;
    border-radius: 6px !important;
    font-size: 0.72rem !important;
}

/* ── BUTTONS ── */
[data-testid="stButton"] > button {
    background: #3182ce !important;
    border: none !important;
    border-radius: 9px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    padding: 0.6rem 1.75rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(49,130,206,0.3), 0 4px 12px rgba(49,130,206,0.2) !important;
}

[data-testid="stButton"] > button:hover {
    background: #2c5282 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(49,130,206,0.35) !important;
}

[data-testid="stDownloadButton"] > button {
    background: #f0fff4 !important;
    border: 1px solid #9ae6b4 !important;
    border-radius: 9px !important;
    color: #276749 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="stDownloadButton"] > button:hover {
    background: #c6f6d5 !important;
    transform: translateY(-1px) !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #cbd5e0 !important;
    border-radius: 14px !important;
    transition: all 0.25s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #3182ce !important;
    background: #ebf4ff !important;
}

/* ── GRAPH WRAP ── */
.graph-wrap {
    background: #ffffff;
    border: 1px solid #e8edf2;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}

.graph-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #a0aec0;
    margin-bottom: 0.5rem;
    font-weight: 500;
}

/* ── STAT CARD ── */
.stat-card {
    background: #ffffff;
    border: 1px solid #e8edf2;
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 1.5rem;
    height: 100%;
    animation: fadeUp 0.5s ease-out 0.1s both;
}

.stat-block { text-align: center; }

.stat-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #a0aec0;
    margin-bottom: 4px;
}

.stat-val {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #3182ce;
    line-height: 1;
}

.stat-file {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #718096;
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    padding: 3px 10px;
    border-radius: 6px;
    display: inline-block;
}

/* ── EMPTY STATE ── */
.empty-state {
    text-align: center;
    padding: 6rem 2rem;
    animation: fadeUp 0.6s ease-out both;
}

.empty-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.3; }
.empty-title { font-size: 1rem; font-weight: 600; color: #718096; margin-bottom: 0.35rem; }
.empty-sub { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #a0aec0; }

/* ── RADIO ── */
[data-testid="stRadio"] span {
    color: #4a5568 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] > div {
    border-color: #3182ce !important;
    border-top-color: transparent !important;
}

/* ── FOOTER ── */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    border-top: 1px solid #e8edf2;
    margin-top: 3rem;
}

.footer-txt {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #cbd5e0;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ── TOP NAV ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <div class="brand-icon">🧬</div>
        <span class="brand-name">AutoML Studio</span>
        <span class="brand-version">v2.0</span>
    </div>
    <div class="topbar-status">
        <span class="status-dot"></span>
        All Systems Ready
    </div>
</div>
""", unsafe_allow_html=True)

# ── STEP PROGRESS ────────────────────────────────────────────────────────────
has_result = "ml_result" in st.session_state

def step(n, label, state="idle"):
    cls_n = "step-num done" if state=="done" else ("step-num active" if state=="active" else "step-num")
    cls_l = "step-label done" if state=="done" else ("step-label active" if state=="active" else "step-label")
    return f'<div class="step-item"><div class="{cls_n}">{n}</div><span class="{cls_l}">{label}</span></div>'

st.markdown(f"""
<div class="progress-bar">
    {step(1, "Ingest", "active")}
    {step(2, "Preview", "active")}
    {step(3, "Config", "active")}
    {step(4, "Train", "done" if has_result else "active")}
    {step(5, "Results", "done" if has_result else "idle")}
    {step(6, "Visualize", "idle")}
    {step(7, "Export", "idle")}
    {step(8, "Feedback", "idle")}
</div>
""", unsafe_allow_html=True)

# ── UPLOAD ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-header">
    <span class="sec-num">01</span>
    <span class="sec-title">Data Ingestion</span>
    <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your CSV dataset here or click to browse",
    type=["csv"],
    help="Accepts CSV files up to 200 MB"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ── DATASET PREVIEW ──────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-header">
        <span class="sec-num">02</span>
        <span class="sec-title">Dataset Preview</span>
        <div class="sec-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_tbl, col_stat = st.columns([3, 1], gap="medium")
    rows, ncols = df.shape

    with col_tbl:
        st.dataframe(df.head(10), width='stretch', hide_index=True)

    with col_stat:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-block">
                <div class="stat-lbl">Rows</div>
                <div class="stat-val">{rows:,}</div>
            </div>
            <div class="stat-block">
                <div class="stat-lbl">Columns</div>
                <div class="stat-val">{ncols}</div>
            </div>
            <div class="stat-block">
                <div class="stat-lbl">File</div>
                <div class="stat-file">{uploaded_file.name}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── CONFIGURATION ────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-header" style="animation-delay:.1s">
        <span class="sec-num">03</span>
        <span class="sec-title">Configuration</span>
        <div class="sec-line"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        with st.container(border=True):
            task_type = st.selectbox(
                "Problem Type",
                ["Auto Detect", "Classification", "Regression"]
            )
            metrics_required = st.multiselect(
                "Evaluation Metrics",
                ["Accuracy", "F1 Score", "Precision", "Recall"],
                default=["Accuracy", "F1 Score"]
            )

    with c2:
        with st.container(border=True):
            graphs_required = st.multiselect(
                "Visualization Outputs",
                ["Confusion Matrix", "ROC Curve", "Feature Importance"],
                default=["Confusion Matrix"]
            )
            improvement_goal = st.selectbox(
                "Optimization Strategy",
                ["Max Accuracy", "Balanced Performance"]
            )

    st.session_state["user_requirements"] = {
        "task": task_type,
        "metrics": metrics_required,
        "graphs": graphs_required,
        "goal": improvement_goal
    }

    # ── TRAINING ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-header" style="animation-delay:.15s">
        <span class="sec-num">04</span>
        <span class="sec-title">Model Training</span>
        <div class="sec-line"></div>
    </div>
    """, unsafe_allow_html=True)

    train_col, _ = st.columns([1, 4])
    with train_col:
        run_clicked = st.button("⚡  Run AutoML Training", width='stretch')

    if run_clicked:
        with st.spinner("Training models — this may take a moment..."):
            ml_result = run_ml_pipeline(
                df,
                st.session_state.get("user_requirements", {})
            )
        st.session_state["ml_result"] = ml_result

    # ── RESULTS ──────────────────────────────────────────────────────────
    if "ml_result" in st.session_state:
        ml_result = st.session_state["ml_result"]
        user_req  = st.session_state.get("user_requirements", {})

        st.markdown("""
        <div class="sec-header" style="animation-delay:.2s">
            <span class="sec-num">05</span>
            <span class="sec-title">Model Comparison</span>
            <div class="sec-line"></div>
        </div>
        """, unsafe_allow_html=True)

        selected_metrics = user_req.get("metrics", ["Accuracy"])
        if not selected_metrics:
            selected_metrics = ["Accuracy"]

        metric_key_map = {
            "Accuracy":  "accuracy",
            "F1 Score":  "f1_score",
            "Precision": "precision",
            "Recall":    "recall",
        }

        for model, metrics in ml_result["results"].items():
            is_best = model == ml_result["best_model"]
            label = f"🏆 {model}  ·  Best Model" if is_best else model
            with st.container(border=True):
                st.markdown(f"**{label}**")
                cols = st.columns(len(selected_metrics))
                for col, m_label in zip(cols, selected_metrics):
                    m_key = metric_key_map.get(m_label)
                    if m_key and m_key in metrics:
                        col.metric(m_label, f"{metrics[m_key]:.4f}")

        st.success(f"✅  Best model: **{ml_result['best_model']}**")

        # ── VISUALIZATIONS ───────────────────────────────────────────────
        graphs = user_req.get("graphs", [])
        y_test, y_pred = get_last_predictions()

        if graphs:
            st.markdown("""
            <div class="sec-header" style="animation-delay:.25s">
                <span class="sec-num">06</span>
                <span class="sec-title">Visualizations</span>
                <div class="sec-line"></div>
            </div>
            """, unsafe_allow_html=True)

        g1, g2 = st.columns(2, gap="medium")

        if "Confusion Matrix" in graphs:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f7fafc')
            ax.imshow(cm, cmap='Blues', alpha=0.7)
            ax.set_title("Confusion Matrix", color='#4a5568', fontsize=11, pad=12)
            ax.tick_params(colors='#a0aec0')
            for spine in ax.spines.values():
                spine.set_edgecolor('#e2e8f0')
            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='#2d3748', fontsize=12, fontweight='600')
            plt.tight_layout()
            with g1:
                st.pyplot(fig)

        if "ROC Curve" in graphs:
            try:
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                roc_auc = auc(fpr, tpr)
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                fig2.patch.set_facecolor('#ffffff')
                ax2.set_facecolor('#f7fafc')
                ax2.plot(fpr, tpr, color='#3182ce', linewidth=2.5,
                         label=f"AUC = {roc_auc:.3f}")
                ax2.fill_between(fpr, tpr, alpha=0.08, color='#3182ce')
                ax2.plot([0, 1], [0, 1], linestyle='--', color='#e2e8f0', linewidth=1.5)
                ax2.set_title("ROC Curve", color='#4a5568', fontsize=11, pad=12)
                ax2.tick_params(colors='#a0aec0')
                ax2.legend(facecolor='#ffffff', edgecolor='#e2e8f0',
                           labelcolor='#4a5568', fontsize=9)
                for spine in ax2.spines.values():
                    spine.set_edgecolor('#e2e8f0')
                plt.tight_layout()
                with g2:
                    st.pyplot(fig2)
            except Exception:
                st.info("ROC curve not available for this task type.")

    # ── FEEDBACK ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-header" style="animation-delay:.35s">
        <span class="sec-num">08</span>
        <span class="sec-title">Feedback Loop</span>
        <div class="sec-line"></div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        feedback = st.radio(
            "How would you rate the prediction quality?",
            ["👍  Accurate — results are good", "👎  Inaccurate — needs improvement"],
            horizontal=True
        )

        is_negative = "👎" in feedback or "Inaccurate" in feedback
        suggestion = ""
        if is_negative:
            suggestion = st.text_area(
                "What should be improved? (optional)",
                placeholder="e.g. Try deeper trees, improve recall, use gradient boosting...",
                height=80
            )

        fb_col, imp_col, _ = st.columns([1, 1, 4])
        with fb_col:
            if st.button("Submit Feedback", width='stretch'):
                if "ml_result" in st.session_state:
                    save_feedback({
                        "model": st.session_state["ml_result"]["best_model"],
                        "feedback": feedback,
                        "suggestion": suggestion
                    })
                    st.success("✓  Feedback recorded.")
                else:
                    st.info("Please run training first.")

        with imp_col:
            if is_negative and "ml_result" in st.session_state:
                if st.button("🔁  Improve Model", width='stretch'):
                    with st.spinner("Re-training with your feedback..."):
                        improve_result = improve_model(df)
                    st.session_state["improve_result"] = improve_result

        if "improve_result" in st.session_state:
            r = st.session_state["improve_result"]
            st.success(f"🔁 Improved · **{r['model_used']}** · Strategy: `{r['strategy']}`")
            ri1, ri2 = st.columns(2)
            ri1.metric("Accuracy", f"{r['accuracy']:.4f}")
            ri2.metric("F1 Score", f"{r['f1_score']:.4f}")
            if r["skipped_models"]:
                st.caption(f"Skipped (bad feedback): {', '.join(r['skipped_models'])}")

    # ── POST-FEEDBACK VISUALIZATIONS + EXPORT ────────────────────────────
    if "improve_result" in st.session_state:
        user_req = st.session_state.get("user_requirements", {})
        graphs   = user_req.get("graphs", [])
        y_test, y_pred = get_last_predictions()

        if graphs:
            st.markdown("""
            <div class="sec-header">
                <span class="sec-num">06</span>
                <span class="sec-title">Updated Visualizations</span>
                <div class="sec-line"></div>
            </div>
            """, unsafe_allow_html=True)

            g1, g2 = st.columns(2, gap="medium")

            if "Confusion Matrix" in graphs:
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#f7fafc')
                ax.imshow(cm, cmap='Greens', alpha=0.7)
                ax.set_title("Confusion Matrix (Improved)", color='#4a5568', fontsize=11, pad=12)
                ax.tick_params(colors='#a0aec0')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#e2e8f0')
                for i in range(len(cm)):
                    for j in range(len(cm)):
                        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                                color='#2d3748', fontsize=12, fontweight='600')
                plt.tight_layout()
                with g1:
                    st.pyplot(fig)

            if "ROC Curve" in graphs:
                try:
                    fpr, tpr, _ = roc_curve(y_test, y_pred)
                    roc_auc = auc(fpr, tpr)
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    fig2.patch.set_facecolor('#ffffff')
                    ax2.set_facecolor('#f7fafc')
                    ax2.plot(fpr, tpr, color='#38a169', linewidth=2.5, label=f"AUC = {roc_auc:.3f}")
                    ax2.fill_between(fpr, tpr, alpha=0.08, color='#38a169')
                    ax2.plot([0, 1], [0, 1], linestyle='--', color='#e2e8f0', linewidth=1.5)
                    ax2.set_title("ROC Curve (Improved)", color='#4a5568', fontsize=11, pad=12)
                    ax2.tick_params(colors='#a0aec0')
                    ax2.legend(facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#4a5568', fontsize=9)
                    for spine in ax2.spines.values():
                        spine.set_edgecolor('#e2e8f0')
                    plt.tight_layout()
                    with g2:
                        st.pyplot(fig2)
                except Exception:
                    st.info("ROC curve not available for this task type.")

    # ── EXPORT ───────────────────────────────────────────────────────────
    if "ml_result" in st.session_state:
        ml_result = st.session_state["ml_result"]
        st.markdown("""
        <div class="sec-header">
            <span class="sec-num">07</span>
            <span class="sec-title">Export Report</span>
            <div class="sec-line"></div>
        </div>
        """, unsafe_allow_html=True)

        exp_col, _ = st.columns([1, 4])
        with exp_col:
            if st.button("📄  Generate PDF Report", width='stretch'):
                with st.spinner("Compiling report..."):
                    file_path = generate_report(df, ml_result)
                with open(file_path, "rb") as f:
                    st.download_button(
                        label="⬇  Download Report",
                        data=f,
                        file_name="automl_report.pdf",
                        mime="application/pdf",
                        width='stretch'
                    )

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📂</div>
        <div class="empty-title">No dataset loaded</div>
        <div class="empty-sub">Upload a CSV file above to initialize the AutoML pipeline</div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <div class="footer-txt">AutoML Studio · Agentic Pipeline · Multi-Model Optimization</div>
</div>
""", unsafe_allow_html=True)
