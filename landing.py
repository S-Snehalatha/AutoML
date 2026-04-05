"""
landing.py — SmartForge Landing Page + Auth Flow
Handles: Landing → Register → Login → App
"""

import streamlit as st
from backend.auth.auth_manager import AuthManager


# ─────────────────────────────────────────────────────────────
# FULL PAGE CSS
# ─────────────────────────────────────────────────────────────
LANDING_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stApp > header { display: none; }
div[data-testid="stToolbar"] { display: none; }
div[data-testid="stDecoration"] { display: none; }
div[data-testid="stStatusWidget"] { display: none; }
section[data-testid="stSidebar"] { display: none !important; }

.stApp {
    background: #040812;
    font-family: 'Inter', sans-serif;
}

/* ── Landing Hero ── */
.hero-wrap {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2rem;
    position: relative;
    overflow: hidden;
}

.hero-grid {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(108,99,255,0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(108,99,255,0.06) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

.hero-glow {
    position: fixed;
    width: 700px; height: 700px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(108,99,255,0.12) 0%, transparent 70%);
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
    z-index: 0;
}

.hero-content { position: relative; z-index: 1; max-width: 860px; }

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(108,99,255,0.12);
    border: 1px solid rgba(108,99,255,0.35);
    border-radius: 100px;
    padding: 6px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #a89dff;
    letter-spacing: 0.08em;
    margin-bottom: 2rem;
}

.hero-badge::before {
    content: '';
    width: 6px; height: 6px;
    background: #43E97B;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
}

.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(3rem, 8vw, 5.5rem);
    font-weight: 900;
    line-height: 1;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
}

.hero-title .t1 {
    background: linear-gradient(135deg, #ffffff 0%, #c4bdff 50%, #8b82ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: block;
}

.hero-title .t2 {
    background: linear-gradient(135deg, #43E97B 0%, #38f9d7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: block;
    font-size: 0.55em;
    font-weight: 600;
    letter-spacing: 0.15em;
    margin-top: 0.4rem;
    font-family: 'Inter', sans-serif;
}

.hero-subtitle {
    font-size: clamp(1rem, 2.5vw, 1.25rem);
    color: #7a7fa8;
    line-height: 1.7;
    max-width: 620px;
    margin: 1.5rem auto 3rem;
    font-weight: 300;
}

.hero-subtitle strong { color: #a89dff; font-weight: 500; }

.hero-cta {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 4rem;
}

.btn-primary {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(135deg, #6C63FF 0%, #5451d6 100%);
    color: white;
    padding: 14px 32px;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    border: none;
    cursor: pointer;
    transition: all 0.25s;
    text-decoration: none;
    box-shadow: 0 0 30px rgba(108,99,255,0.35);
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 45px rgba(108,99,255,0.55);
}

.btn-secondary {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: transparent;
    color: #a89dff;
    padding: 13px 32px;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    border: 1px solid rgba(108,99,255,0.35);
    cursor: pointer;
    transition: all 0.25s;
    text-decoration: none;
}

.btn-secondary:hover {
    background: rgba(108,99,255,0.1);
    border-color: rgba(108,99,255,0.6);
    transform: translateY(-2px);
}

/* Feature pills */
.feature-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    justify-content: center;
    margin-top: 1rem;
}

.pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: #6b7098;
    font-family: 'JetBrains Mono', monospace;
}

/* Stats bar */
.stats-bar {
    display: flex;
    gap: 3rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 3rem auto;
    padding: 1.5rem 2rem;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    max-width: 700px;
}

.stat-item { text-align: center; }
.stat-num {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #6C63FF;
}
.stat-label {
    font-size: 0.78rem;
    color: #6b7098;
    margin-top: 3px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

/* ── Auth Cards ── */
.auth-outer {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    background: #040812;
}

.auth-card {
    background: rgba(255,255,255,0.032);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 20px;
    padding: 2.8rem 2.5rem;
    width: 100%;
    max-width: 460px;
    backdrop-filter: blur(20px);
    position: relative;
}

.auth-card::before {
    content: '';
    position: absolute;
    top: -1px; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(108,99,255,0.6), transparent);
}

.auth-logo {
    text-align: center;
    margin-bottom: 0.3rem;
}

.auth-logo .brand {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #43E97B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.auth-tagline {
    text-align: center;
    font-size: 0.82rem;
    color: #6b7098;
    margin-bottom: 2rem;
    font-family: 'JetBrains Mono', monospace;
}

.auth-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #e8e8f0;
    margin-bottom: 0.3rem;
}

.auth-subtitle {
    font-size: 0.88rem;
    color: #6b7098;
    margin-bottom: 1.8rem;
}

.auth-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 1.4rem 0;
    color: #3a3d5c;
    font-size: 0.78rem;
}

.auth-divider::before, .auth-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.07);
}

.auth-switch {
    text-align: center;
    font-size: 0.85rem;
    color: #6b7098;
    margin-top: 1.5rem;
}

.auth-switch span {
    color: #8b82ff;
    cursor: pointer;
    font-weight: 500;
}

.auth-switch span:hover { color: #a89dff; text-decoration: underline; }

/* Field labels */
.field-label {
    font-size: 0.82rem;
    color: #9098c0;
    margin-bottom: 5px;
    font-weight: 500;
    display: block;
}

/* Strength bar */
.strength-wrap { margin-top: 6px; }
.strength-track {
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
}
.strength-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s, background 0.3s;
}
.strength-label {
    font-size: 0.72rem;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* Password toggle */
.pw-wrap { position: relative; }

/* Requirements list */
.req-list {
    list-style: none;
    font-size: 0.78rem;
    color: #6b7098;
    margin-top: 8px;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.8;
}
.req-list li::before { content: '○  '; }
.req-list li.ok { color: #43E97B; }
.req-list li.ok::before { content: '✓  '; }

/* Back link */
.back-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #6b7098;
    font-size: 0.82rem;
    cursor: pointer;
    margin-bottom: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    transition: color 0.2s;
}
.back-link:hover { color: #a89dff; }

/* Streamlit overrides for dark inputs */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(108,99,255,0.25) !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.55rem 0.85rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(108,99,255,0.6) !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.12) !important;
}
.stTextInput > label { display: none !important; }

.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #6C63FF, #5451d6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(108,99,255,0.5) !important;
}

.stCheckbox > label { color: #9098c0 !important; font-size: 0.85rem !important; }

/* Error / success messages */
.stAlert { border-radius: 10px !important; font-size: 0.88rem !important; }

/* Remove empty space */
.block-container { padding-top: 0 !important; padding-bottom: 0 !important; }
</style>
"""

# ─────────────────────────────────────────────────────────────
# PAGE: LANDING
# ─────────────────────────────────────────────────────────────
def render_landing():
    st.markdown(LANDING_CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero-grid"></div><div class="hero-glow"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero-content" style="max-width:860px;margin:0 auto;padding:4rem 1rem 2rem;text-align:center">

      <div class="hero-badge">⚡ AI-POWERED MACHINE LEARNING AUTOMATION</div>

      <h1 class="hero-title">
        <span class="t1">SmartForge</span>
        <span class="t2">AN INTELLIGENT AUTOML SYSTEM</span>
      </h1>

      <p class="hero-subtitle">
        Upload your dataset. <strong>SmartForge</strong> automatically runs the entire
        machine learning pipeline — EDA, preprocessing, training 14+ models,
        evaluation, SHAP explanations and deployment-ready predictions.
        <br>No code. No expertise required.
      </p>

      <div class="stats-bar">
        <div class="stat-item">
          <div class="stat-num">14+</div>
          <div class="stat-label">ML ALGORITHMS</div>
        </div>
        <div class="stat-item">
          <div class="stat-num">100%</div>
          <div class="stat-label">AUTOMATED</div>
        </div>
        <div class="stat-item">
          <div class="stat-num">3</div>
          <div class="stat-label">TASK TYPES</div>
        </div>
        <div class="stat-item">
          <div class="stat-num">SHAP</div>
          <div class="stat-label">EXPLAINABILITY</div>
        </div>
      </div>

      <div class="feature-pills">
        <span class="pill">Auto EDA</span>
        <span class="pill">Missing Value Imputation</span>
        <span class="pill">Feature Selection</span>
        <span class="pill">Hyperparameter Tuning</span>
        <span class="pill">ROC · Confusion Matrix</span>
        <span class="pill">SHAP Values</span>
        <span class="pill">Batch Predictions</span>
        <span class="pill">CSV · Excel · JSON</span>
        <span class="pill">Model Export (.pkl)</span>
      </div>

    </div>
    """, unsafe_allow_html=True)

    # CTA buttons using Streamlit columns
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_c1, col_c2, col_r = st.columns([1, 1.2, 1.2, 1])
    with col_c1:
        if st.button("🚀 Get Started Free", key="hero_register"):
            st.session_state["page"] = "register"
            st.rerun()
    with col_c2:
        if st.button("🔐 Sign In", key="hero_login"):
            st.session_state["page"] = "login"
            st.rerun()

    # Features grid
# Features grid — using st.columns to avoid CSS grid rendering issues
    st.markdown("<br>", unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(108,99,255,0.15);
                    border-radius:14px;padding:1.4rem;border-top:2px solid #6C63FF;height:160px">
          <div style="font-size:1.4rem;margin-bottom:0.5rem">📊</div>
          <div style="font-family:Orbitron,sans-serif;font-size:0.82rem;color:#a89dff;
                      font-weight:600;margin-bottom:0.4rem">AUTO EDA</div>
          <div style="font-size:0.8rem;color:#6b7098;line-height:1.6">
            7 visualization types generated automatically — histograms, heatmaps, pair plots and more.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with fc2:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(67,233,123,0.15);
                    border-radius:14px;padding:1.4rem;border-top:2px solid #43E97B;height:160px">
          <div style="font-size:1.4rem;margin-bottom:0.5rem">🤖</div>
          <div style="font-family:Orbitron,sans-serif;font-size:0.82rem;color:#43E97B;
                      font-weight:600;margin-bottom:0.4rem">14+ MODELS</div>
          <div style="font-size:0.8rem;color:#6b7098;line-height:1.6">
            XGBoost, LightGBM, CatBoost, Random Forest, Neural Networks and more trained in parallel.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with fc3:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,101,132,0.15);
                    border-radius:14px;padding:1.4rem;border-top:2px solid #FF6584;height:160px">
          <div style="font-size:1.4rem;margin-bottom:0.5rem">💡</div>
          <div style="font-family:Orbitron,sans-serif;font-size:0.82rem;color:#FF6584;
                      font-weight:600;margin-bottom:0.4rem">EXPLAINABILITY</div>
          <div style="font-size:0.8rem;color:#6b7098;line-height:1.6">
            SHAP values, feature importance, beeswarm plots — understand every prediction.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: REGISTER
# ─────────────────────────────────────────────────────────────
def render_register():
    st.markdown(LANDING_CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero-grid"></div>', unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        # Back link
        if st.button("← Back to Home", key="back_from_reg"):
            st.session_state["page"] = "landing"
            st.rerun()

        st.markdown("""
        <div class="auth-card">
          <div class="auth-logo"><span class="brand">SmartForge</span></div>
          <div class="auth-tagline">INTELLIGENT AUTOML SYSTEM</div>
          <div class="auth-title">Create your account</div>
          <div class="auth-subtitle">Start automating machine learning in seconds.</div>
        </div>
        """, unsafe_allow_html=True)

        # Form fields
        full_name = st.text_input("Full Name",        placeholder="Your full name",                   key="reg_name")
        username  = st.text_input("Username",         placeholder="letters, numbers, underscore only", key="reg_user")
        email     = st.text_input("Email Address",    placeholder="you@example.com",                  key="reg_email")
        org       = st.text_input("Organization",     placeholder="Company / University (optional)",  key="reg_org")
        password  = st.text_input("Password",         placeholder="Min 8 chars, 1 uppercase, 1 number", type="password", key="reg_pw")
        confirm   = st.text_input("Confirm Password", placeholder="Repeat your password",             type="password", key="reg_conf")

        # Password strength meter
        if password:
            score, label, color, reqs = _password_strength(password)
            pct = int((score / 5) * 100)
            st.markdown(f"""
            <div class="strength-wrap">
              <div class="strength-track">
                <div class="strength-fill" style="width:{pct}%;background:{color}"></div>
              </div>
              <div class="strength-label" style="color:{color}">{label}</div>
              <ul class="req-list">
                <li class="{'ok' if reqs[0] else ''}">At least 8 characters</li>
                <li class="{'ok' if reqs[1] else ''}">One uppercase letter</li>
                <li class="{'ok' if reqs[2] else ''}">One number</li>
                <li class="{'ok' if reqs[3] else ''}">One special character (!@#$…)</li>
              </ul>
            </div>
            """, unsafe_allow_html=True)

        # Confirm match indicator
        if confirm and password:
            if confirm == password:
                st.markdown('<p style="color:#43E97B;font-size:0.8rem;font-family:JetBrains Mono,monospace">✓ Passwords match</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#FF6584;font-size:0.8rem;font-family:JetBrains Mono,monospace">✗ Passwords do not match</p>', unsafe_allow_html=True)

        terms = st.checkbox("I agree to the Terms of Service and Privacy Policy", key="reg_terms")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Create Account →", key="reg_submit"):
            error = None
            if not username or not email or not password or not confirm:
                error = "Please fill in all required fields."
            elif password != confirm:
                error = "Passwords do not match."
            elif not terms:
                error = "Please accept the Terms of Service."
            else:
                with st.spinner("Creating your account…"):
                    ok, msg = AuthManager.register(
                        username=username,
                        email=email,
                        password=password,
                        full_name=full_name,
                    )
                    if ok and org:
                        AuthManager.update_profile(username.lower(), {"profile.organization": org})
                if ok:
                    st.success(f"🎉 Account created! Welcome to SmartForge.")
                    st.session_state["reg_success_user"] = username
                    import time; time.sleep(1)
                    st.session_state["page"] = "login"
                    st.rerun()
                else:
                    error = msg

            if error:
                st.error(f"❌ {error}")

        st.markdown("""
        <div class="auth-switch">
          Already have an account?
        </div>
        """, unsafe_allow_html=True)

        col_sw = st.columns([2, 1, 2])
        with col_sw[1]:
            if st.button("Sign In", key="switch_to_login"):
                st.session_state["page"] = "login"
                st.rerun()


# ─────────────────────────────────────────────────────────────
# PAGE: LOGIN
# ─────────────────────────────────────────────────────────────
def render_login():
    st.markdown(LANDING_CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero-grid"></div>', unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        if st.button("← Back to Home", key="back_from_login"):
            st.session_state["page"] = "landing"
            st.rerun()

        # Show success message if coming from registration
        if st.session_state.get("reg_success_user"):
            st.success(f"✅ Account created for **{st.session_state.reg_success_user}**. Please sign in.")

        st.markdown("""
        <div class="auth-card">
          <div class="auth-logo"><span class="brand">SmartForge</span></div>
          <div class="auth-tagline">INTELLIGENT AUTOML SYSTEM</div>
          <div class="auth-title">Welcome back</div>
          <div class="auth-subtitle">Sign in to continue to your workspace.</div>
        </div>
        """, unsafe_allow_html=True)

        identifier = st.text_input("Username or Email", placeholder="Enter username or email", key="login_id")
        password   = st.text_input("Password",          placeholder="Enter your password",     type="password", key="login_pw")

        col_rem, col_forgot = st.columns([1, 1])
        with col_rem:
            st.checkbox("Remember me", key="login_remember")
        with col_forgot:
            if st.button("Forgot password?", key="forgot_pw"):
                st.session_state["page"] = "forgot"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign In →", key="login_submit"):
            if not identifier or not password:
                st.error("❌ Please enter your username/email and password.")
            else:
                with st.spinner("Signing in…"):
                    ok, msg, user = AuthManager.login(identifier, password)
                if ok:
                    token = AuthManager.create_session(user["username"])
                    st.session_state["auth_token"]       = token
                    st.session_state["current_user"]     = user
                    st.session_state["authenticated"]    = True
                    st.session_state["reg_success_user"] = None
                    st.session_state["page"]             = "app"
                    st.success(f"✅ Welcome back, {user.get('full_name') or user['username']}!")
                    import time; time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

        st.markdown('<div class="auth-divider">or</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-switch">New to SmartForge?</div>', unsafe_allow_html=True)
        col_sw = st.columns([2, 1, 2])
        with col_sw[1]:
            if st.button("Register", key="switch_to_reg"):
                st.session_state["page"] = "register"
                st.rerun()


# ─────────────────────────────────────────────────────────────
# PAGE: FORGOT PASSWORD
# ─────────────────────────────────────────────────────────────
def render_forgot():
    st.markdown(LANDING_CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero-grid"></div>', unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        if st.button("← Back to Sign In", key="back_from_forgot"):
            st.session_state["page"] = "login"
            st.rerun()

        st.markdown("""
        <div class="auth-card">
          <div class="auth-logo"><span class="brand">SmartForge</span></div>
          <div class="auth-tagline">INTELLIGENT AUTOML SYSTEM</div>
          <div class="auth-title">Reset Password</div>
          <div class="auth-subtitle">Enter your username and a new password below.</div>
        </div>
        """, unsafe_allow_html=True)

        username   = st.text_input("Username", placeholder="Your username", key="forgot_user")
        new_pw     = st.text_input("New Password",     type="password",
                                    placeholder="Min 8 chars, 1 uppercase, 1 number", key="forgot_pw1")
        confirm_pw = st.text_input("Confirm New Password", type="password",
                                    placeholder="Repeat new password", key="forgot_pw2")

        if new_pw:
            score, label, color, _ = _password_strength(new_pw)
            pct = int((score / 5) * 100)
            st.markdown(f"""
            <div class="strength-wrap">
              <div class="strength-track">
                <div class="strength-fill" style="width:{pct}%;background:{color}"></div>
              </div>
              <div class="strength-label" style="color:{color}">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Reset Password →", key="forgot_submit"):
            if not username or not new_pw or not confirm_pw:
                st.error("❌ Please fill in all fields.")
            elif new_pw != confirm_pw:
                st.error("❌ Passwords do not match.")
            else:
                from backend.auth.database import MongoDB
                users = MongoDB.users()
                if users is None:
                    st.error("❌ Database connection failed.")
                else:
                    user = users.find_one({"username": username.lower()})
                    if not user:
                        st.error("❌ No account found with that username.")
                    else:
                        ok, msg = AuthManager.validate_password(new_pw)
                        if not ok:
                            st.error(f"❌ {msg}")
                        else:
                            new_hash = AuthManager.hash_password(new_pw)
                            from datetime import datetime
                            users.update_one(
                                {"username": username.lower()},
                                {"$set": {"password_hash": new_hash,
                                           "updated_at": datetime.utcnow()}}
                            )
                            st.success("✅ Password reset successfully! Please sign in.")
                            import time; time.sleep(1.2)
                            st.session_state["page"] = "login"
                            st.rerun()


# ─────────────────────────────────────────────────────────────
# MASTER ROUTER — call this from app.py
# ─────────────────────────────────────────────────────────────
def run_auth_flow() -> bool:
    """
    Manages the full auth flow via st.session_state["page"].
    Returns True when the user is authenticated and ready to use the app.
    Returns False and calls st.stop() when auth pages are shown.
    """
    # Initialise page state
    if "page" not in st.session_state:
        st.session_state["page"] = "landing"

    # Restore session token
    if not st.session_state.get("authenticated"):
        token = st.session_state.get("auth_token")
        if token:
            user = AuthManager.validate_session(token)
            if user:
                st.session_state["current_user"]  = user
                st.session_state["authenticated"] = True
                st.session_state["page"]          = "app"

    page = st.session_state.get("page", "landing")

    if page == "app" and st.session_state.get("authenticated"):
        return True          # ← authenticated: render the main app

    # Auth / landing pages
    if page == "landing":
        render_landing()
    elif page == "register":
        render_register()
    elif page == "login":
        render_login()
    elif page == "forgot":
        render_forgot()
    else:
        render_landing()

    st.stop()               # ← stop app.py from rendering tabs
    return False


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _password_strength(password: str):
    reqs = [
        len(password) >= 8,
        any(c.isupper() for c in password),
        any(c.isdigit() for c in password),
        any(c in "!@#$%^&*()_+-=[]{}|;:',.<>?" for c in password),
    ]
    score = sum(reqs) + (1 if len(password) >= 12 else 0)
    score = min(score, 5)
    labels = ["Very Weak", "Weak", "Fair", "Good", "Strong", "Very Strong"]
    colors = ["#FF6584", "#FF8C42", "#FFD700", "#7dd87d", "#43E97B", "#6C63FF"]
    return score, labels[score], colors[score], reqs
