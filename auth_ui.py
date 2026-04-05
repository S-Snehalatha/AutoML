"""
auth_ui.py — Delegates auth gate to the landing page flow.
Keeps backward compatibility with app.py's import.
"""

import streamlit as st
from datetime import datetime
from backend.auth.auth_manager    import AuthManager
from backend.auth.history_manager import HistoryManager


# ─────────────────────────────────────────────────────────────
# MAIN GATE — imported by app.py
# ─────────────────────────────────────────────────────────────
def render_auth_gate() -> bool:
    """
    Called at the very top of app.py (before any tabs).
    Delegates to the landing page / auth flow in landing.py.
    Returns True only when the user is authenticated.
    Calls st.stop() for all unauthenticated states.
    """
    from frontend.landing import run_auth_flow
    return run_auth_flow()


# ─────────────────────────────────────────────────────────────
# PROFILE PAGE  (used in app.py tabs[10])
# ─────────────────────────────────────────────────────────────
AUTH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');
.avatar-circle {
    width:56px;height:56px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    font-size:1.4rem;font-weight:700;color:white;
    font-family:'Space Mono',monospace;margin:0 auto 1rem;
}
.stat-card {
    background:rgba(108,99,255,0.1);border:1px solid rgba(108,99,255,0.2);
    border-radius:12px;padding:1rem;text-align:center;
    font-family:'DM Sans',sans-serif;
}
.stat-number{font-size:1.8rem;font-weight:700;color:#43E97B;font-family:'Space Mono',monospace;}
.stat-label{color:#8888AA;font-size:0.82rem;}
.history-card{
    background:linear-gradient(135deg,rgba(108,99,255,0.08),rgba(67,233,123,0.05));
    border:1px solid rgba(108,99,255,0.2);border-left:4px solid #6C63FF;
    border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.8rem;
    font-family:'DM Sans',sans-serif;
}
.history-title{font-weight:600;color:#F0F0FF;font-size:0.95rem;}
.history-meta{color:#8888AA;font-size:0.8rem;margin-top:0.2rem;}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.72rem;font-weight:700;font-family:'Space Mono',monospace;}
.badge-clf{background:rgba(108,99,255,0.2);color:#6C63FF;border:1px solid #6C63FF;}
.badge-reg{background:rgba(67,233,123,0.15);color:#43E97B;border:1px solid #43E97B;}
.badge-clu{background:rgba(255,101,132,0.15);color:#FF6584;border:1px solid #FF6584;}
</style>
"""


def _badge(task):
    css = {"Classification":"clf","Regression":"reg","Clustering":"clu"}.get(task,"clf")
    return f'<span class="badge badge-{css}">{task}</span>'


def _avatar(user):
    color   = user.get("profile",{}).get("avatar_color","#6C63FF")
    initial = (user.get("full_name") or user.get("username","U"))[0].upper()
    return f'<div class="avatar-circle" style="background:{color}">{initial}</div>'


def render_profile():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    user = st.session_state.get("current_user", {})
    if not user:
        st.warning("Not logged in.")
        return

    st.markdown(_avatar(user), unsafe_allow_html=True)
    name_display = user.get("full_name") or user["username"]
    st.markdown(f'<h2 style="text-align:center;font-family:Space Mono;color:#F0F0FF">{name_display}</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center;color:#8888AA;font-family:DM Sans">@{user["username"]} · {user.get("email","")}</p>', unsafe_allow_html=True)
    org = user.get("profile",{}).get("organization","")
    if org:
        st.markdown(f'<p style="text-align:center;color:#6C63FF;font-family:DM Sans">🏢 {org}</p>', unsafe_allow_html=True)

    st.markdown("---")
    stats      = user.get("stats", {})
    hist_stats = HistoryManager.get_stats(user["username"])
    c1,c2,c3,c4 = st.columns(4)
    for col, num, label in [
        (c1, hist_stats.get("total_runs",0),      "Pipeline Runs"),
        (c2, stats.get("total_models",0),         "Models Trained"),
        (c3, stats.get("datasets_uploaded",0),    "Datasets"),
        (c4, user.get("login_count",0),           "Logins"),
    ]:
        col.markdown(f'<div class="stat-card"><div class="stat-number">{num}</div>'
                     f'<div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("✏️ Edit Profile", expanded=False):
        with st.form("profile_form"):
            new_name  = st.text_input("Full Name",    value=user.get("full_name",""))
            new_org   = st.text_input("Organization", value=user.get("profile",{}).get("organization",""))
            new_bio   = st.text_area("Bio",           value=user.get("profile",{}).get("bio",""), height=80)
            new_color = st.color_picker("Avatar Color", value=user.get("profile",{}).get("avatar_color","#6C63FF"))
            if st.form_submit_button("💾 Save Changes"):
                AuthManager.update_profile(user["username"], {
                    "full_name": new_name, "profile.organization": new_org,
                    "profile.bio": new_bio, "profile.avatar_color": new_color,
                })
                updated = AuthManager.get_user(user["username"])
                if updated:
                    st.session_state["current_user"] = updated
                st.success("✅ Profile updated!")
                st.rerun()

    with st.expander("🔑 Change Password", expanded=False):
        with st.form("password_form"):
            old_pw  = st.text_input("Current Password",     type="password")
            new_pw  = st.text_input("New Password",         type="password")
            conf_pw = st.text_input("Confirm New Password", type="password")
            if st.form_submit_button("🔒 Change Password"):
                if new_pw != conf_pw:
                    st.error("New passwords do not match.")
                else:
                    ok, msg = AuthManager.change_password(user["username"], old_pw, new_pw)
                    st.success(msg) if ok else st.error(msg)

    st.markdown("---")
    if hist_stats.get("total_runs", 0) > 0:
        st.markdown(
            f"**🕐 First run:** {hist_stats.get('first_run','—')} &nbsp;|&nbsp; "
            f"**Last run:** {hist_stats.get('last_run','—')} &nbsp;|&nbsp; "
            f"**Favourite model:** {hist_stats.get('favourite_model','—')}",
            unsafe_allow_html=True,
        )


def render_history():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    st.markdown('<p style="font-family:Space Mono;font-size:1.3rem;color:#6C63FF;'
                'border-bottom:2px solid rgba(108,99,255,0.3);padding-bottom:6px">'
                '🕐 My Pipeline History</p>', unsafe_allow_html=True)

    user = st.session_state.get("current_user", {})
    if not user:
        st.warning("Not logged in.")
        return

    runs = HistoryManager.get_user_history(user["username"])
    if not runs:
        st.info("📭 No pipeline runs yet. Upload a dataset and run the pipeline to see history here.")
        return

    st.markdown(f"**{len(runs)} run(s)** — newest first")
    st.markdown("---")

    for run in runs:
        task    = run.get("task_type","Unknown")
        badge   = _badge(task)
        date    = (run["created_at"].strftime("%d %b %Y, %H:%M")
                   if isinstance(run.get("created_at"), datetime)
                   else str(run.get("created_at",""))[:16])
        best    = run.get("best_model","N/A")
        ds      = run.get("dataset_name","Unknown")
        n_r     = run.get("n_rows", 0)
        n_c     = run.get("n_cols", 0)
        metrics = run.get("metrics", {})
        met_str = " | ".join(
            f"**{k}:** {v:.4f}" if isinstance(v,float) else f"**{k}:** {v}"
            for k,v in list(metrics.items())[:3]
        )
        st.markdown(
            f'<div class="history-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div class="history-title">📁 {ds}</div>{badge}</div>'
            f'<div class="history-meta">🗓 {date} &nbsp;|&nbsp; 🔢 {n_r:,}×{n_c} &nbsp;|&nbsp; 🏆 {best}</div>'
            f'<div style="margin-top:0.4rem;color:#C8C8E8;font-size:0.85rem">{met_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        c_sp, c_view, c_del = st.columns([3,1,1])
        with c_view:
            if st.button("🔍 Details", key=f"view_{run['_id']}"):
                st.session_state["view_run_id"] = run["_id"]
        with c_del:
            if st.button("🗑️ Delete", key=f"del_{run['_id']}"):
                if HistoryManager.delete_run(run["_id"], user["username"]):
                    st.success("Deleted.")
                    st.rerun()

        if st.session_state.get("view_run_id") == run["_id"]:
            with st.expander("📊 Full Details", expanded=True):
                t1,t2,t3 = st.tabs(["📈 Metrics","📋 Models","🔧 Features"])
                with t1:
                    if metrics:
                        mc = st.columns(len(metrics))
                        for col,(k,v) in zip(mc, metrics.items()):
                            col.metric(k, f"{v:.4f}" if isinstance(v,float) else str(v))
                with t2:
                    lb = run.get("leaderboard",[])
                    if lb:
                        import pandas as pd
                        st.dataframe(pd.DataFrame(lb), use_container_width=True)
                with t3:
                    feats = run.get("feature_names",[])
                    if feats:
                        st.write(", ".join(feats[:50]))
