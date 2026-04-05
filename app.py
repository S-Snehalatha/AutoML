"""
AutoML Studio — Fully Automatic Pipeline
Upload data once → everything runs automatically.
No button-clicking required for pipeline steps.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.modules.data_ingestion   import DataIngestion
from backend.modules.eda              import AutoEDA
from backend.modules.preprocessing    import AutoPreprocessor
from backend.modules.model_trainer    import AutoModelTrainer
from backend.modules.model_evaluator  import ModelEvaluator
from backend.modules.explainability   import ModelExplainer
from backend.modules.model_selector   import ModelSelector
from backend.modules.prediction       import PredictionEngine
from backend.modules.nlp_output       import NLPOutputGenerator
from backend.modules.documentation    import DocumentationGenerator
from backend.modules.downloader       import (
    download_model, download_pipeline, download_predictions,
    download_leaderboard, download_evaluation_report,
    download_preprocessed_data, download_eda_report,
    download_figure, render_download_center,
)
from backend.auth.auth_ui       import render_auth_gate, render_profile, render_history
from backend.auth.auth_manager  import AuthManager
from backend.auth.history_manager import HistoryManager

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #6C63FF; --secondary: #FF6584; --accent: #43E97B;
    --bg-dark: #0D0D1A; --bg-card: #1A1A2E;
    --text-primary: #F0F0FF; --text-muted: #8888AA;
}
.stApp { background: linear-gradient(135deg,#0D0D1A 0%,#1A1A2E 100%); }
.main-header {
    font-family:'Space Mono',monospace; font-size:2.8rem; font-weight:700;
    background:linear-gradient(90deg,#6C63FF,#43E97B,#FF6584);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    text-align:center; padding:1.5rem 0 0.3rem; letter-spacing:-1px;
}
.sub-header {
    font-family:'DM Sans',sans-serif; color:#8888AA;
    text-align:center; font-size:1.05rem; margin-bottom:1.5rem;
}
.section-title {
    font-family:'Space Mono',monospace; font-size:1.2rem; color:#6C63FF;
    border-bottom:2px solid rgba(108,99,255,0.3);
    padding-bottom:6px; margin:1.5rem 0 1rem;
}
.info-box {
    background:linear-gradient(135deg,rgba(108,99,255,0.08),rgba(67,233,123,0.05));
    border:1px solid rgba(108,99,255,0.25); border-left:4px solid #6C63FF;
    border-radius:10px; padding:1rem 1.2rem; margin:1.2rem 0;
    font-family:'DM Sans',sans-serif; font-size:0.92rem; color:#C8C8E8; line-height:1.6;
}
.info-box h4 { color:#43E97B; margin:0 0 0.4rem; font-size:0.95rem; }
.tip-box {
    background:rgba(255,101,132,0.07); border:1px solid rgba(255,101,132,0.2);
    border-left:4px solid #FF6584; border-radius:10px;
    padding:0.8rem 1.2rem; margin:0.8rem 0;
    font-family:'DM Sans',sans-serif; font-size:0.88rem; color:#C8C8E8;
}
.pipeline-banner {
    background:linear-gradient(90deg,rgba(108,99,255,0.15),rgba(67,233,123,0.1));
    border:1px solid rgba(108,99,255,0.3); border-radius:12px;
    padding:1rem 1.5rem; margin:1rem 0; text-align:center;
    font-family:'DM Sans',sans-serif;
}
.auto-badge {
    display:inline-block; background:linear-gradient(90deg,#6C63FF,#43E97B);
    color:white; padding:2px 10px; border-radius:12px;
    font-size:0.72rem; font-weight:700; font-family:'Space Mono',monospace;
    margin-left:6px; vertical-align:middle;
}
.prediction-box {
    background:linear-gradient(135deg,#1A1A2E,#16213E);
    border:2px solid #43E97B; border-radius:16px;
    padding:1.5rem 2rem; margin:1rem 0;
    font-family:'DM Sans',sans-serif;
}
div[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0D0D1A 0%,#16213E 100%);
    border-right:1px solid rgba(108,99,255,0.2);
}
.stButton > button {
    background:linear-gradient(90deg,#6C63FF,#43E97B) !important;
    color:white !important; border:none !important; border-radius:8px !important;
    font-family:'DM Sans',sans-serif !important; font-weight:600 !important;
    transition:all 0.3s !important;
}
.stButton > button:hover {
    transform:translateY(-2px);
    box-shadow:0 8px 25px rgba(108,99,255,0.4) !important;
}
.stTabs [data-baseweb="tab"] { font-family:'DM Sans',sans-serif; font-weight:500; color:#8888AA; }
.stTabs [data-baseweb="tab-highlight"] { background:linear-gradient(90deg,#6C63FF,#43E97B); }
.step-done  { color:#43E97B; }
.step-active{ color:#6C63FF; font-weight:700; }
.step-wait  { color:#555577; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "data": None, "schema": None,
    "eda_figures": None, "eda_done": False,
    "preprocessed_data": None, "preprocessor": None,
    "trained_models": None, "leaderboard": None,
    "best_model": None, "trainer": None,
    "task_type": None, "target_col": None,
    "evaluation_results": None, "feature_names": None,
    "shap_figs": None, "pipeline_step": 0,
    "auto_preprocessed": False, "auto_trained": False,
    "auto_evaluated": False, "auto_shap": False,
    "max_models": 8, "hp_method": "Random Search", "cv_folds": 5,
    # Auth
    "authenticated": False, "current_user": None, "auth_token": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def info_box(title: str, body: str):
    st.markdown(f'<div class="info-box"><h4>ℹ️ {title}</h4>{body}</div>', unsafe_allow_html=True)

def tip_box(text: str):
    st.markdown(f'<div class="tip-box">💡 <b>Tip:</b> {text}</div>', unsafe_allow_html=True)

def auto_badge():
    return '<span class="auto-badge">AUTO</span>'

def run_auto_pipeline():
    """
    Core auto-runner: called every time the app renders.
    If data is loaded but a step hasn't run yet, it runs it silently.
    """
    ss = st.session_state
    if ss["data"] is None:
        return

    # ── Step 1: Auto EDA ─────────────────────────────────────
    if not ss["eda_done"]:
        with st.spinner("🔍 Running Automated EDA…"):
            eda = AutoEDA()
            ss["eda_figures"] = eda.run(ss["data"], ss["target_col"])
            ss["eda_done"] = True
            ss["pipeline_step"] = max(ss["pipeline_step"], 2)

    # ── Step 2: Auto Preprocessing ───────────────────────────
    if not ss["auto_preprocessed"] and ss["target_col"] is not None or \
       (not ss["auto_preprocessed"] and ss["task_type"] == "Clustering"):
        with st.spinner("⚙️ Running Automated Preprocessing…"):
            preprocessor = AutoPreprocessor(
                num_impute="median", cat_impute="most_frequent",
                cat_encode="auto", scaler="standard",
                outlier_method="iqr", feat_select="auto",
                target_col=ss["target_col"],
                task_type=ss["task_type"] or "Classification",
            )
            X, y, feature_names, report = preprocessor.fit_transform(ss["data"])
            ss["preprocessed_data"] = (X, y)
            ss["preprocessor"] = preprocessor
            ss["feature_names"] = feature_names
            ss["preprocessing_report"] = report
            ss["auto_preprocessed"] = True
            ss["pipeline_step"] = max(ss["pipeline_step"], 3)

    # ── Step 3: Auto Training ────────────────────────────────
    if ss["auto_preprocessed"] and not ss["auto_trained"]:
        X, y = ss["preprocessed_data"]
        with st.spinner("🤖 Training all models automatically…"):
            trainer = AutoModelTrainer(
                task_type=ss["task_type"] or "Classification",
                max_models=ss["max_models"],
                hp_method=ss["hp_method"],
                cv_folds=ss["cv_folds"],
                feature_names=ss["feature_names"],
            )
            trained_models, leaderboard = trainer.train(X, y)
            ss["trained_models"] = trained_models
            ss["leaderboard"]    = leaderboard
            ss["trainer"]        = trainer
            selector = ModelSelector(task_type=ss["task_type"] or "Classification")
            ss["best_model"] = selector.select(trained_models, leaderboard)
            ss["auto_trained"] = True
            ss["pipeline_step"] = max(ss["pipeline_step"], 4)

    # ── Step 4: Auto Evaluation ──────────────────────────────
    if ss["auto_trained"] and not ss["auto_evaluated"]:
        X, y = ss["preprocessed_data"]
        with st.spinner("📊 Evaluating best model…"):
            evaluator = ModelEvaluator(task_type=ss["task_type"] or "Classification")
            eval_results = evaluator.evaluate(ss["best_model"], X, y, ss["feature_names"])
            ss["evaluation_results"] = eval_results
            ss["auto_evaluated"] = True
            ss["pipeline_step"]  = max(ss["pipeline_step"], 5)

    # ── Step 5: Auto SHAP ────────────────────────────────────
    if ss["auto_evaluated"] and not ss["auto_shap"]:
        X, _ = ss["preprocessed_data"]
        with st.spinner("💡 Computing SHAP explanations…"):
            try:
                explainer = ModelExplainer()
                ss["shap_figs"] = explainer.explain(
                    ss["best_model"]["model"], X,
                    ss["feature_names"],
                    task_type=ss["task_type"] or "Classification",
                )
            except Exception:
                ss["shap_figs"] = {}
            ss["auto_shap"] = True
            ss["pipeline_step"] = max(ss["pipeline_step"], 6)

    # ── Step 6: Auto-save run to history ─────────────────────
    if ss["auto_shap"] and not ss.get("history_saved"):
        try:
            user = ss.get("current_user", {})
            if user and ss.get("best_model"):
                lb = ss.get("leaderboard", [])
                HistoryManager.save_run(user["username"], {
                    "dataset_name":  ss.get("uploaded_filename", "Unknown"),
                    "n_rows":        ss.get("schema", {}).get("n_rows", 0),
                    "n_cols":        ss.get("schema", {}).get("n_cols", 0),
                    "task_type":     ss.get("task_type", "Unknown"),
                    "target_col":    ss.get("target_col"),
                    "best_model":    ss["best_model"].get("name"),
                    "metrics":       ss["best_model"].get("metrics", {}),
                    "models_trained": [m["name"] for m in (ss.get("trained_models") or [])],
                    "feature_names": ss.get("feature_names", []),
                    "leaderboard":   lb,
                })
        except Exception:
            pass
        ss["history_saved"] = True


# ─────────────────────────────────────────────────────────────
# AUTH GATE — must pass before anything else renders
# ─────────────────────────────────────────────────────────────
if not render_auth_gate():
    st.stop()   # Stop rendering if not authenticated

# ─────────────────────────────────────────────────────────────
# SIDEBAR  (only shown when authenticated)
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    user = st.session_state.get("current_user", {})
    uname = user.get("full_name") or user.get("username", "User")
    color = user.get("profile", {}).get("avatar_color", "#6C63FF")
    initial = uname[0].upper()

    # User avatar + greeting
    st.markdown(
        f'''<div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem">
            <div style="width:40px;height:40px;border-radius:50%;background:{color};
                        display:flex;align-items:center;justify-content:center;
                        font-family:Space Mono;font-weight:700;color:white;font-size:1rem">{initial}</div>
            <div>
                <div style="font-family:DM Sans;font-weight:600;color:#F0F0FF;font-size:0.92rem">{uname}</div>
                <div style="font-family:DM Sans;color:#8888AA;font-size:0.75rem">@{user.get("username","")}</div>
            </div>
        </div>''',
        unsafe_allow_html=True,
    )

    st.markdown('<p style="font-family:Space Mono;font-size:1.1rem;color:#6C63FF;font-weight:700;margin:0">⚡ AutoML Studio</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    steps = [
        ("📤","Upload Data"),("🔍","EDA"),("⚙️","Preprocessing"),
        ("🤖","Model Training"),("📊","Evaluation"),
        ("💡","Explainability"),("🎯","Prediction"),
    ]
    cur = st.session_state.pipeline_step
    for i,(icon,label) in enumerate(steps):
        cls = "step-done" if i < cur else ("step-active" if i == cur else "step-wait")
        mark = "✅" if i < cur else ("▶" if i == cur else "○")
        st.markdown(f'<p class="{cls}" style="font-family:DM Sans;margin:4px 0">{icon} {mark} {label}</p>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Pipeline Settings**")
    st.session_state.max_models = st.slider("Max models", 3, 14, 8)
    st.session_state.hp_method  = st.selectbox("HPO method",
        ["Random Search","Grid Search","Bayesian Optimization"])
    st.session_state.cv_folds   = st.slider("CV folds", 2, 10, 5)

    st.markdown("---")
    col_reset, col_logout = st.columns(2)
    with col_reset:
        if st.button("🔄 Reset"):
            pipeline_keys = [k for k in DEFAULTS if k not in
                             ("authenticated","current_user","auth_token")]
            for k in pipeline_keys:
                st.session_state[k] = DEFAULTS[k]
            st.rerun()
    with col_logout:
        if st.button("🚪 Logout"):
            token = st.session_state.get("auth_token")
            if token:
                AuthManager.logout(token)
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.markdown('<p style="color:#555577;font-size:0.72rem;font-family:DM Sans;margin-top:1rem">'
                'AutoML Studio v2.0<br>scikit-learn · XGBoost · LightGBM<br>CatBoost · SHAP · Plotly</p>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">⚡ AutoML Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your dataset once — the full ML pipeline runs automatically</p>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📤 Upload & Configure",
    "🔍 EDA",
    "⚙️ Preprocessing",
    "🤖 Model Training",
    "📊 Evaluation",
    "💡 Explainability",
    "🎯 Prediction",
    "⬇️ Downloads",
    "📚 Documentation",
    "🕐 My History",
    "👤 My Profile",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD & CONFIGURE
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="section-title">Step 1 — Upload Your Dataset</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="pipeline-banner">
        🚀 <b>Fully Automatic Pipeline</b> — Upload your CSV/Excel/JSON, choose a target column,
        then switch to any tab. Every step runs automatically — no clicking required.
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_config = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "📂 Drop your dataset here",
            type=["csv","xlsx","xls","json"],
            help="Supports CSV, Excel (.xlsx/.xls), and JSON formats",
        )

        if uploaded_file:
            # Only reload if a new file was uploaded
            if ("uploaded_filename" not in st.session_state or
                    st.session_state.get("uploaded_filename") != uploaded_file.name):

                with st.spinner("🔍 Loading and detecting schema…"):
                    ingestion = DataIngestion()
                    df, schema = ingestion.load(uploaded_file)

                if df is not None:
                    # Reset pipeline for new data
                    for k, v in DEFAULTS.items():
                        st.session_state[k] = v
                    st.session_state["data"] = df
                    st.session_state["schema"] = schema
                    st.session_state["uploaded_filename"] = uploaded_file.name
                    st.session_state["pipeline_step"] = 1
                    st.rerun()

        if st.session_state.data is not None:
            df     = st.session_state.data
            schema = st.session_state.schema

            st.success(f"✅ **{uploaded_file.name if uploaded_file else st.session_state.get('uploaded_filename','')}** — "
                       f"{schema['n_rows']:,} rows × {schema['n_cols']} columns")

            # Schema summary cards
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("📋 Rows",     f"{schema['n_rows']:,}")
            c2.metric("📊 Columns",  schema['n_cols'])
            c3.metric("🔢 Numeric",  schema['n_numeric'])
            c4.metric("🔤 Categorical", schema['n_categorical'])

            # Column details table
            with st.expander("📋 Column Details", expanded=True):
                schema_df = pd.DataFrame({
                    "Column":  list(schema["column_types"].keys()),
                    "Type":    list(schema["column_types"].values()),
                    "Missing %": [f"{schema['missing_pct'].get(c,0):.1f}%" for c in schema["column_types"]],
                    "Unique":  [schema["n_unique"].get(c,0) for c in schema["column_types"]],
                })
                st.dataframe(schema_df, use_container_width=True)

            st.markdown("**📌 Data Preview (first 5 rows)**")
            st.dataframe(df.head(), use_container_width=True)

    with col_config:
        st.markdown("#### ⚙️ Task Configuration")
        if st.session_state.data is not None:
            df = st.session_state.data

            col_options = ["(no target — clustering)"] + list(df.columns)
            prev_target = st.session_state.get("target_col")
            default_idx = col_options.index(prev_target) if prev_target in col_options else 0

            selected_target = st.selectbox("🎯 Target Column", col_options, index=default_idx)
            new_target = None if selected_target == "(no target — clustering)" else selected_target

            # If target changed, reset downstream steps
            if new_target != st.session_state.target_col:
                st.session_state.target_col = new_target
                for key in ["eda_done","auto_preprocessed","auto_trained",
                            "auto_evaluated","auto_shap","eda_figures",
                            "preprocessed_data","preprocessor","trained_models",
                            "leaderboard","best_model","evaluation_results","shap_figs"]:
                    st.session_state[key] = DEFAULTS[key]
                st.session_state.pipeline_step = 1
                st.rerun()

            if new_target:
                n_unique = df[new_target].nunique()
                auto_task = "Classification" if (df[new_target].dtype == object or n_unique <= 20) else "Regression"
                task_choice = st.selectbox("📌 Task Type",
                    ["Auto-detect","Classification","Regression","Clustering"],
                    index=0)
                st.session_state.task_type = auto_task if task_choice == "Auto-detect" else task_choice
                st.info(f"🤖 Task: **{st.session_state.task_type}**")
            else:
                st.session_state.task_type = "Clustering"
                st.info("🔵 Mode: **Clustering** (unsupervised)")

            st.markdown("---")
            tip_box("Once you set a target column, switch to any tab — the full pipeline runs automatically in the background.")

        else:
            st.info("⬆️ Upload a dataset to begin.")

    # ── Explainer ─────────────────────────────────────────────
    info_box("What is AutoML Studio?",
        "AutoML Studio automatically handles the entire machine learning pipeline: "
        "data cleaning → feature engineering → training multiple models → evaluating each one → "
        "selecting the best → explaining predictions. You don't need to write any code. "
        "Just upload your CSV and pick your target column.")
    info_box("Supported File Formats",
        "<b>CSV</b> — comma-separated, most common format.<br>"
        "<b>Excel (.xlsx/.xls)</b> — Microsoft Excel spreadsheets.<br>"
        "<b>JSON</b> — array of records or flat objects.<br>"
        "Files up to ~50MB work well. Larger files may be slow.")
    info_box("What is a Target Column?",
        "The target column is the variable you want to <b>predict</b>. For example:<br>"
        "• Predicting house prices → target = <code>price</code><br>"
        "• Classifying emails as spam → target = <code>is_spam</code><br>"
        "• Diagnosing a disease → target = <code>diagnosis</code><br>"
        "If you have no target, select <i>no target</i> and clustering will run.")


# ══════════════════════════════════════════════════════════════
# AUTO-RUNNER — executes before remaining tabs render
# ══════════════════════════════════════════════════════════════
if st.session_state.data is not None and st.session_state.target_col is not None:
    run_auto_pipeline()
elif st.session_state.data is not None and st.session_state.task_type == "Clustering":
    run_auto_pipeline()


# ══════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(f'<p class="section-title">Exploratory Data Analysis {auto_badge()}</p>',
                unsafe_allow_html=True)

    if st.session_state.data is None:
        st.info("⬆️ Upload a dataset in the **Upload & Configure** tab to begin.")
    elif not st.session_state.eda_done:
        st.info("⏳ EDA is running automatically… please wait a moment.")
    else:
        figs = st.session_state.eda_figures
        df   = st.session_state.data

        # Summary stats
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            with st.expander("📋 Statistical Summary", expanded=False):
                st.dataframe(num_df.describe().round(4), use_container_width=True)

        eda_tabs = st.tabs([
            "📊 Distributions","📦 Box Plots","🔥 Correlation",
            "🔵 Scatter","📊 Categorical","❓ Missing Values","📈 Pair Plot",
        ])

        with eda_tabs[0]:
            st.subheader("Histograms & Distributions")
            if figs and "histograms" in figs:
                for fig in figs["histograms"]:
                    st.plotly_chart(fig, use_container_width=True)
            info_box("What are Histograms?",
                "A histogram shows how values of a numeric feature are distributed. "
                "Tall bars mean many data points fall in that range. "
                "<b>Bell-shaped</b> curves suggest a normal distribution, which many algorithms prefer. "
                "<b>Skewed</b> distributions (leaning left or right) may need log-transformation. "
                "<b>Multiple peaks</b> may indicate sub-groups in your data.")
            tip_box("Features with extreme skew (long tails) can hurt model performance. "
                    "Consider using Robust or MinMax scaling.")

        with eda_tabs[1]:
            st.subheader("Box Plots — Outlier Detection")
            if figs and "boxplots" in figs:
                for fig in figs["boxplots"]:
                    st.plotly_chart(fig, use_container_width=True)
            info_box("How to Read a Box Plot",
                "The <b>box</b> spans from the 25th to 75th percentile (middle 50% of data). "
                "The <b>line inside</b> is the median. "
                "The <b>whiskers</b> extend to 1.5× the box width. "
                "<b>Dots beyond the whiskers</b> are outliers — extreme values that may distort model training. "
                "AutoML Studio automatically removes outliers during preprocessing.")

        with eda_tabs[2]:
            st.subheader("Correlation Heatmap")
            if figs and "correlation" in figs:
                st.plotly_chart(figs["correlation"], use_container_width=True)
            info_box("What is Correlation?",
                "Correlation measures how strongly two features move together. "
                "<b>+1.0</b> = perfectly positively correlated (both increase together). "
                "<b>-1.0</b> = perfectly negatively correlated (one increases, other decreases). "
                "<b>0.0</b> = no linear relationship. "
                "Features with <b>correlation > 0.9</b> are nearly duplicates — one can often be removed. "
                "Features strongly correlated with the <b>target</b> are your most useful predictors.")
            tip_box("Highly correlated feature pairs (dark red/blue squares) are candidates for removal "
                    "to avoid multicollinearity.")

        with eda_tabs[3]:
            st.subheader("Scatter Plots — Feature vs Target")
            if figs and "scatter" in figs:
                for fig in figs["scatter"]:
                    st.plotly_chart(fig, use_container_width=True)
            info_box("What do Scatter Plots tell you?",
                "Each dot is one row in your dataset. The <b>x-axis</b> is a feature, the <b>y-axis</b> "
                "is the target. A clear upward/downward trend means that feature strongly predicts the target. "
                "A <b>random cloud of dots</b> means weak predictive power. "
                "The <b>trend line</b> shows the overall linear relationship.")

        with eda_tabs[4]:
            st.subheader("Categorical Feature Charts")
            if figs and "categorical" in figs:
                for fig in figs["categorical"]:
                    st.plotly_chart(fig, use_container_width=True)
            info_box("What are Count/Bar Charts?",
                "These charts show how often each <b>category</b> appears in your data. "
                "A heavily <b>imbalanced</b> bar chart (one category dominates) can cause models to be biased. "
                "For example, if 95% of your records are 'Not Fraud', a naive model could just predict 'Not Fraud' "
                "every time and still be 95% accurate — but useless.")
            tip_box("If your target column is heavily imbalanced (e.g. 98% class A, 2% class B), "
                    "focus on F1 Score and ROC AUC rather than Accuracy.")

        with eda_tabs[5]:
            st.subheader("Missing Value Heatmap")
            if figs and "missing" in figs:
                st.plotly_chart(figs["missing"], use_container_width=True)
            info_box("Understanding Missing Values",
                "Missing values appear as colored cells in this heatmap. "
                "<b>Columns with many missing values</b> (>50%) are usually dropped. "
                "<b>Randomly missing values</b> are imputed using median/mean/KNN. "
                "<b>Structurally missing</b> values (e.g. salary is blank for unemployed people) "
                "carry information and can be encoded as a separate feature. "
                "AutoML Studio automatically imputes all missing values.")

        with eda_tabs[6]:
            st.subheader("Pair Plot")
            if figs and "pairplot" in figs:
                st.plotly_chart(figs["pairplot"], use_container_width=True)
            info_box("What is a Pair Plot?",
                "A pair plot shows <b>every combination of two numeric features</b> plotted against each other. "
                "The diagonal shows the distribution of each individual feature. "
                "You can quickly spot which feature pairs have clear linear or non-linear relationships. "
                "Colors represent different target classes (for classification tasks).")

        download_eda_report(st.session_state.eda_figures)


# ══════════════════════════════════════════════════════════════
# TAB 3 — PREPROCESSING
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(f'<p class="section-title">Data Preprocessing {auto_badge()}</p>',
                unsafe_allow_html=True)

    if st.session_state.data is None:
        st.info("⬆️ Upload a dataset first.")
    elif not st.session_state.auto_preprocessed:
        st.info("⏳ Preprocessing is running automatically…")
    else:
        report = st.session_state.get("preprocessing_report", {})
        X, y   = st.session_state.preprocessed_data

        # Summary cards
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Input Features",    report.get("input_features", "—"))
        c2.metric("Output Features",   report.get("output_features", "—"))
        c3.metric("Rows After Clean",  f"{report.get('rows_after',0):,}")
        c4.metric("Outliers Removed",  report.get("outliers_removed", 0))

        with st.expander("🔎 Full Preprocessing Report"):
            for k, v in report.items():
                st.write(f"**{k.replace('_',' ').title()}:** {v}")

        # Show preprocessed data sample
        with st.expander("📋 Preprocessed Data Sample (first 5 rows)"):
            fn = st.session_state.feature_names or [f"f{i}" for i in range(X.shape[1])]
            prev_df = pd.DataFrame(X[:5], columns=fn)
            st.dataframe(prev_df, use_container_width=True)

        # Download preprocessed data
        download_preprocessed_data(X, y, st.session_state.feature_names or [],
                                   target_col=st.session_state.target_col or "target")

        # ── Explainers ────────────────────────────────────────
        info_box("What is Data Preprocessing?",
            "Raw data is almost never ready for machine learning. Preprocessing cleans and transforms it:<br>"
            "• <b>Missing value imputation</b> — fills empty cells using median, mean, or statistical models<br>"
            "• <b>Categorical encoding</b> — converts text categories into numbers models can understand<br>"
            "• <b>Feature scaling</b> — normalizes numeric ranges so no feature dominates due to its scale<br>"
            "• <b>Outlier removal</b> — removes extreme values that could distort the model<br>"
            "• <b>Feature selection</b> — keeps only the most informative features")
        info_box("Why Feature Scaling Matters",
            "Imagine two features: <code>age</code> (range 0–100) and <code>salary</code> (range 0–200,000). "
            "Without scaling, salary dominates the model purely because its numbers are bigger. "
            "<b>StandardScaler</b> centers features around 0 with unit variance. "
            "<b>MinMaxScaler</b> compresses values between 0 and 1. "
            "<b>RobustScaler</b> is best when outliers are present.")
        info_box("How AutoML Studio Preprocessed Your Data",
            f"• <b>Numeric imputation:</b> Median (robust to outliers)<br>"
            f"• <b>Categorical imputation:</b> Most frequent value<br>"
            f"• <b>Encoding:</b> One-Hot for low-cardinality, Ordinal for high-cardinality<br>"
            f"• <b>Scaling:</b> StandardScaler on numeric features<br>"
            f"• <b>Outliers:</b> IQR method (removed values beyond 3×IQR)<br>"
            f"• <b>Feature selection:</b> SelectKBest (F-test) — kept top {report.get('output_features','N/A')} features")
        tip_box("The preprocessed dataset has been downloaded above. You can use it directly "
                "to train models in other tools like Jupyter Notebook.")


# ══════════════════════════════════════════════════════════════
# TAB 4 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown(f'<p class="section-title">Automated Model Training {auto_badge()}</p>',
                unsafe_allow_html=True)

    if st.session_state.data is None:
        st.info("⬆️ Upload a dataset first.")
    elif not st.session_state.auto_trained:
        st.info("⏳ Models are being trained automatically… (this may take 1–3 minutes)")
    else:
        models   = st.session_state.trained_models
        lb       = st.session_state.leaderboard
        best     = st.session_state.best_model
        task     = st.session_state.task_type

        st.success(f"✅ **{len(models)} models** trained | 🏆 Best: **{best['name']}** | Task: **{task}**")

        # Leaderboard table
        st.markdown("### 🏆 Model Leaderboard")
        lb_df = pd.DataFrame(lb)
        lb_df = lb_df.sort_values(lb_df.columns[1], ascending=False).reset_index(drop=True)
        lb_df.index += 1
        lb_df.index.name = "Rank"
        try:
            styled = lb_df.style.background_gradient(subset=[lb_df.columns[1]], cmap="Greens")
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.dataframe(lb_df, use_container_width=True)

        # Bar chart
        import plotly.express as px
        metric_col = lb_df.columns[1]
        fig = px.bar(lb_df, x="Model", y=metric_col, color=metric_col,
                     color_continuous_scale=["#6C63FF","#43E97B"],
                     title=f"Model Comparison — {metric_col}",
                     template="plotly_dark")
        fig.update_layout(paper_bgcolor="#1A1A2E", plot_bgcolor="#0D0D1A")
        st.plotly_chart(fig, use_container_width=True)

        # Best model card
        st.markdown("### 🥇 Best Model Details")
        bm = best
        cols = st.columns(len(bm.get("metrics",{})) + 1)
        cols[0].metric("Model Name", bm["name"])
        for col, (k,v) in zip(cols[1:], bm.get("metrics",{}).items()):
            col.metric(k, f"{v:.4f}" if isinstance(v,float) else str(v))

        download_leaderboard(lb)

        # ── Explainers ────────────────────────────────────────
        info_box("What algorithms were trained?",
            "<b>Classification:</b> Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, "
            "SVM, KNN, Neural Network, XGBoost, LightGBM, CatBoost<br><br>"
            "<b>Regression:</b> Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, "
            "Gradient Boosting, SVR, KNN, Neural Network, XGBoost, LightGBM, CatBoost<br><br>"
            "<b>Clustering:</b> K-Means (k=2…5), DBSCAN, Hierarchical Clustering")
        info_box("What is Hyperparameter Optimization (HPO)?",
            "Every model has <b>hyperparameters</b> — settings that control how the model learns "
            "(e.g. how deep a tree can grow, how fast a neural network learns). "
            "AutoML Studio automatically searches for the best combination using:<br>"
            "• <b>Random Search</b> — samples random combinations (fast, good for large spaces)<br>"
            "• <b>Grid Search</b> — tries every combination (thorough but slow)<br>"
            "• <b>Bayesian Optimization</b> — learns from previous trials to find the best settings faster")
        info_box("How is the best model selected?",
            f"For <b>Classification</b>: highest weighted F1 Score (balances precision and recall across all classes)<br>"
            f"For <b>Regression</b>: highest R² Score (proportion of variance explained)<br>"
            f"For <b>Clustering</b>: highest Silhouette Score (how well-separated clusters are)<br><br>"
            f"The winning model is: <b>{best['name']}</b>")
        tip_box("You can retrain with different settings using the sidebar sliders — "
                "increase 'Max models' to train more algorithms.")


# ══════════════════════════════════════════════════════════════
# TAB 5 — EVALUATION
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown(f'<p class="section-title">Model Evaluation & Visual Analysis {auto_badge()}</p>',
                unsafe_allow_html=True)

    if not st.session_state.auto_evaluated:
        if st.session_state.data is None:
            st.info("⬆️ Upload a dataset first.")
        else:
            st.info("⏳ Evaluation is running automatically…")
    else:
        eval_r = st.session_state.evaluation_results
        task   = st.session_state.task_type
        best   = st.session_state.best_model

        # Metrics
        st.markdown(f"### 📈 Performance Metrics — {best['name']}")
        metrics = eval_r.get("metrics", {})
        metric_cols = st.columns(len(metrics))
        for col, (k,v) in zip(metric_cols, metrics.items()):
            col.metric(k, f"{v:.4f}" if isinstance(v,float) else str(v))

        # Plots
        figs = eval_r.get("figures", {})
        if figs:
            valid_figs = {k: v for k,v in figs.items() if v is not None}
            if valid_figs:
                plot_tabs = st.tabs(list(valid_figs.keys()))
                for ptab, (name, fig) in zip(plot_tabs, valid_figs.items()):
                    with ptab:
                        st.plotly_chart(fig, use_container_width=True)
                        download_figure(fig, label=name, button_label=f"⬇️ Download {name}")

        download_evaluation_report(eval_r, best["name"])

        # ── Metric explainers ─────────────────────────────────
        if task == "Classification":
            info_box("Understanding Classification Metrics",
                "<b>Accuracy</b> — % of predictions that are correct. Misleading when classes are imbalanced.<br>"
                "<b>Precision</b> — of all predicted positives, what % were actually positive? (avoids false alarms)<br>"
                "<b>Recall</b> — of all actual positives, what % did we catch? (avoids missed detections)<br>"
                "<b>F1 Score</b> — harmonic mean of Precision and Recall. Best single metric for imbalanced data.<br>"
                "<b>ROC AUC</b> — probability that model ranks a random positive higher than a random negative. "
                "0.5 = random, 1.0 = perfect. Values above 0.85 are generally considered strong.")
            info_box("How to Read the Confusion Matrix",
                "The confusion matrix shows how predictions compare to actual labels:<br>"
                "• <b>True Positives (top-left)</b> — correctly predicted positives<br>"
                "• <b>True Negatives (bottom-right)</b> — correctly predicted negatives<br>"
                "• <b>False Positives (top-right)</b> — predicted positive but actually negative (Type I error)<br>"
                "• <b>False Negatives (bottom-left)</b> — predicted negative but actually positive (Type II error)<br>"
                "A perfect model has all values on the diagonal.")
            info_box("ROC Curve Explained",
                "The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate vs False Positive Rate "
                "at every possible classification threshold. "
                "A curve that hugs the top-left corner is excellent. "
                "The diagonal line represents random guessing. "
                "<b>AUC</b> (Area Under Curve) summarizes the whole curve in one number.")
            info_box("Lift & Cumulative Gain Charts",
                "<b>Lift Chart</b> — shows how much better your model is versus random guessing in each decile. "
                "A lift of 3 means targeting that decile gives 3× more positives than random.<br><br>"
                "<b>Cumulative Gain Chart</b> — shows what percentage of all positives you capture "
                "when targeting the top X% of scored records. Useful in marketing campaigns.")
        elif task == "Regression":
            info_box("Understanding Regression Metrics",
                "<b>MAE (Mean Absolute Error)</b> — average absolute difference between predicted and actual. "
                "Easy to interpret in the same units as your target variable.<br>"
                "<b>RMSE (Root Mean Squared Error)</b> — like MAE but penalizes large errors more heavily. "
                "More sensitive to outliers.<br>"
                "<b>R² Score</b> — proportion of variance in the target explained by the model. "
                "R²=1.0 means perfect prediction. R²=0 means the model is no better than predicting the mean. "
                "Negative R² means the model is worse than the mean.")
            info_box("How to Read the Residual Plot",
                "Residuals = Actual − Predicted. A good model should have residuals:<br>"
                "• <b>Centered around zero</b> — no systematic over/under-prediction<br>"
                "• <b>Randomly scattered</b> — no patterns (patterns indicate the model is missing structure)<br>"
                "• <b>Constant variance</b> — not fanning out (heteroscedasticity)<br>"
                "A funnel shape suggests the model underperforms at extreme values.")
        else:
            info_box("Understanding Clustering Metrics",
                "<b>Silhouette Score</b> measures how similar each point is to its own cluster vs other clusters. "
                "Ranges from -1 to +1:<br>"
                "• Near <b>+1</b> — well-separated, compact clusters<br>"
                "• Near <b>0</b> — overlapping clusters<br>"
                "• Near <b>-1</b> — points may be in wrong clusters<br>"
                "Scores above 0.5 are generally considered good.")

        tip_box("Click any plot tab above to explore visual analysis. Each chart can be downloaded as an interactive HTML file.")


# ══════════════════════════════════════════════════════════════
# TAB 6 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown(f'<p class="section-title">Model Explainability — SHAP {auto_badge()}</p>',
                unsafe_allow_html=True)

    if not st.session_state.auto_shap:
        if st.session_state.data is None:
            st.info("⬆️ Upload a dataset first.")
        else:
            st.info("⏳ SHAP explanations are being computed automatically…")
    else:
        shap_figs = st.session_state.shap_figs or {}
        valid = {k:v for k,v in shap_figs.items() if v is not None}

        if valid:
            shap_tabs = st.tabs(list(valid.keys()))
            for stab, (name, fig) in zip(shap_tabs, valid.items()):
                with stab:
                    st.plotly_chart(fig, use_container_width=True)
                    download_figure(fig, label=name, button_label=f"⬇️ Download {name}")
        else:
            st.warning("SHAP computation was skipped or unavailable for this model type. "
                       "Feature importance charts are shown in the Evaluation tab.")

        st.session_state.pipeline_step = max(st.session_state.pipeline_step, 6)

        # ── Explainers ────────────────────────────────────────
        info_box("What is SHAP?",
            "SHAP (SHapley Additive exPlanations) is a game-theory approach to explaining "
            "individual predictions. It answers: <b>'How much did each feature contribute to this prediction?'</b><br><br>"
            "SHAP assigns every feature a <b>SHAP value</b> — a number that represents "
            "how much that feature pushed the prediction higher or lower compared to the average prediction.")
        info_box("SHAP Plot Types Explained",
            "<b>Feature Importance (bar chart)</b> — the average magnitude of SHAP values across all predictions. "
            "Taller bars = more important features overall.<br><br>"
            "<b>Beeswarm Plot</b> — shows the SHAP value for every single prediction. "
            "Each dot is one row of data. Color shows the feature value (red = high, blue = low). "
            "Dots on the right pushed the prediction UP, dots on the left pushed it DOWN.<br><br>"
            "<b>Dependence Plot</b> — shows how one feature's value affects its SHAP contribution. "
            "Reveals non-linear effects and interactions.")
        tip_box("SHAP is model-agnostic — it works on any trained model. "
                "If SHAP computation is slow, it's because complex models (like Neural Networks) "
                "require sampling to estimate SHAP values.")


# ══════════════════════════════════════════════════════════════
# TAB 7 — PREDICTION
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<p class="section-title">Make Predictions with Trained Model</p>',
                unsafe_allow_html=True)

    if st.session_state.best_model is None:
        st.info("⬆️ Upload a dataset and wait for training to complete.")
    else:
        best = st.session_state.best_model
        st.success(f"🤖 Using: **{best['name']}** | Task: **{st.session_state.task_type}**")

        pred_mode = st.radio("📌 Prediction Mode",
            ["✏️ Manual Input (single prediction)", "📂 Upload CSV (batch prediction)"],
            horizontal=True)

        if "Manual" in pred_mode:
            st.markdown("#### Enter values for each feature below:")
            df_orig    = st.session_state.data
            preprocessor = st.session_state.preprocessor
            target     = st.session_state.target_col
            feat_cols  = [c for c in df_orig.columns if c != target]

            input_values = {}
            grid_cols = st.columns(3)
            for i, col_name in enumerate(feat_cols):
                with grid_cols[i % 3]:
                    sample = df_orig[col_name].dropna()
                    sample_val = sample.iloc[0] if len(sample) > 0 else 0
                    if df_orig[col_name].dtype in [np.float64, np.int64, float, int]:
                        input_values[col_name] = st.number_input(
                            col_name, value=float(sample_val),
                            help=f"Range: {df_orig[col_name].min():.2f} – {df_orig[col_name].max():.2f}")
                    else:
                        opts = df_orig[col_name].dropna().unique().tolist()
                        input_values[col_name] = st.selectbox(col_name, opts,
                            help=f"{len(opts)} unique values")

            # Auto-predict whenever inputs change (no button needed)
            input_df = pd.DataFrame([input_values])
            engine = PredictionEngine(
                model=st.session_state.best_model["model"],
                preprocessor=preprocessor,
                task_type=st.session_state.task_type,
                feature_names=st.session_state.feature_names,
            )
            try:
                prediction, confidence = engine.predict(input_df)
                nlp = NLPOutputGenerator()
                explanation = nlp.generate(
                    prediction=prediction,
                    confidence=confidence,
                    task_type=st.session_state.task_type,
                    target_col=st.session_state.target_col,
                    input_data=input_values,
                )
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### 🎯 Prediction: `{prediction}`")
                if confidence is not None:
                    st.progress(confidence)
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                st.markdown(f"**📝 Natural Language Explanation:**")
                st.info(explanation)
                st.markdown('</div>', unsafe_allow_html=True)

                # Download single prediction
                pred_df = pd.DataFrame([{**input_values,
                    (target or "prediction"): prediction,
                    "confidence": confidence if confidence else "N/A",
                    "explanation": explanation,
                }])
                download_predictions([prediction], input_df, target or "Prediction")
            except Exception as e:
                st.error(f"Prediction error: {e}")

        else:
            st.markdown("#### Upload a CSV file with the same feature columns (no target column needed)")
            batch_file = st.file_uploader("📂 Upload CSV", type=["csv"], key="batch_pred")
            if batch_file:
                batch_df = pd.read_csv(batch_file)
                st.write(f"Loaded **{len(batch_df):,}** rows for prediction.")
                with st.spinner("Running batch predictions…"):
                    engine = PredictionEngine(
                        model=st.session_state.best_model["model"],
                        preprocessor=st.session_state.preprocessor,
                        task_type=st.session_state.task_type,
                        feature_names=st.session_state.feature_names,
                    )
                    preds = engine.batch_predict(batch_df)
                    result_df = batch_df.copy()
                    result_df["Prediction"] = preds
                st.dataframe(result_df, use_container_width=True)
                download_predictions(preds, batch_df, st.session_state.target_col or "Prediction")

        st.session_state.pipeline_step = max(st.session_state.pipeline_step, 7)

        info_box("How Predictions Work",
            "The trained model takes your input feature values and passes them through "
            "the same preprocessing pipeline used during training "
            "(same imputation, encoding, and scaling). "
            "The preprocessed values are fed to the best model which outputs a prediction.<br><br>"
            "<b>Confidence</b> (for classification) is the model's probability score for the predicted class. "
            "A confidence of 95% means the model is very certain. Below 60% means uncertainty.")
        info_box("Manual vs Batch Prediction",
            "<b>Manual Input</b> — enter one row of features and get an immediate prediction with explanation.<br>"
            "<b>Batch Prediction</b> — upload a CSV with many rows. All are predicted at once. "
            "Useful for scoring a large dataset. "
            "The output CSV will have your original features plus a <code>Prediction</code> column appended.")
        tip_box("For batch prediction, your CSV should have the same column names as the training data, "
                "but does NOT need the target column.")


# ══════════════════════════════════════════════════════════════
# TAB 8 — DOWNLOADS
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<p class="section-title">⬇️ Download Center</p>', unsafe_allow_html=True)
    render_download_center(st.session_state)

    if st.session_state.get("evaluation_results"):
        st.markdown("### 📈 Download Individual Evaluation Charts")
        figs = st.session_state["evaluation_results"].get("figures", {})
        if figs:
            dl_cols = st.columns(3)
            for i, (name, fig) in enumerate(figs.items()):
                with dl_cols[i % 3]:
                    if fig is not None:
                        download_figure(fig, label=name, button_label=f"⬇️ {name}")

    info_box("What can you download?",
        "<b>Model (.pkl)</b> — the trained scikit-learn model. Load it in Python to make predictions "
        "without retraining.<br>"
        "<b>Full Pipeline (.pkl)</b> — model + preprocessor bundled. Feed raw data directly.<br>"
        "<b>Leaderboard (.csv)</b> — all model names and their metrics for comparison.<br>"
        "<b>Evaluation Report (.json)</b> — structured metrics file for logging/reporting.<br>"
        "<b>Preprocessed Data (.csv)</b> — the cleaned, encoded, scaled dataset.<br>"
        "<b>EDA Report (.html)</b> — all charts in one interactive webpage.<br>"
        "<b>Individual Charts (.html)</b> — each evaluation plot as a standalone interactive chart.")
    info_box("How to use the downloaded model",
        "After downloading <code>pipeline.pkl</code>, load it in Python:<br><br>"
        "<code>import pickle, pandas as pd<br>"
        "bundle = pickle.load(open('pipeline.pkl', 'rb'))<br>"
        "model = bundle['model']<br>"
        "preprocessor = bundle['preprocessor']<br>"
        "new_data = pd.DataFrame([{'age': 35, 'salary': 55000}])<br>"
        "X = preprocessor.transform(new_data)<br>"
        "prediction = model.predict(X)</code>")
    tip_box("The .pkl files work with any Python environment that has scikit-learn installed. "
            "Share them with your team or deploy via the FastAPI server at localhost:8000.")


# ══════════════════════════════════════════════════════════════
# TAB 9 — DOCUMENTATION
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown('<p class="section-title">📚 User Manual & Documentation</p>', unsafe_allow_html=True)
    doc_gen = DocumentationGenerator()
    doc_gen.render_in_streamlit(st.session_state)
    st.session_state.pipeline_step = max(st.session_state.pipeline_step, 8)


# ══════════════════════════════════════════════════════════════
# TAB 10 — MY HISTORY
# ══════════════════════════════════════════════════════════════
with tabs[9]:
    render_history()


# ══════════════════════════════════════════════════════════════
# TAB 11 — MY PROFILE
# ══════════════════════════════════════════════════════════════
with tabs[10]:
    render_profile()
