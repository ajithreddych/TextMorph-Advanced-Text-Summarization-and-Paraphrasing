# frontend/admin.py
import streamlit as st
import requests
import math
from datetime import datetime

API_BASE = "http://127.0.0.1:5000"
API_ADMIN_SUMMARIES = f"{API_BASE}/admin/summaries"
API_ADMIN_SUMMARY = f"{API_BASE}/admin/summary"  # /admin/summary/<id>

API_ADMIN_PARAPHS = f"{API_BASE}/admin/paraphrases"
API_ADMIN_PARAPH = f"{API_BASE}/admin/paraphrase"  # /admin/paraphrase/<id>
API_ADMIN_USAGE = f"{API_BASE}/admin/usage"

def admin_dashboard():
    st.title("🔧 Admin Dashboard")
    admin_menu = st.sidebar.radio(
        "Admin Navigation",
        ["📄 View Summaries", "🔄 View Paraphrases", "📊 Usage Stats", "🚪 Logout"]
    )

    if admin_menu == "📄 View Summaries":
        view_summaries_section()
    elif admin_menu == "🔄 View Paraphrases":
        view_paraphrases_section()
    elif admin_menu == "📊 Usage Stats":
        view_usage_section()
    else:
        st.write(f"Selected Admin Menu: {admin_menu}")
        st.info("Other admin functionalities go here.")

    if admin_menu == "🚪 Logout" and st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.session_state.role = "user"
        st.rerun()


# ---------------------- Summaries helpers (unchanged behavior) ----------------------
def fetch_summaries_from_backend(q: str = ""):
    headers = {"Authorization": st.session_state.get("token")} if st.session_state.get("token") else {}
    params = {"q": q} if q else {}
    res = requests.get(API_ADMIN_SUMMARIES, headers=headers, params=params, timeout=15)
    if res.status_code != 200:
        raise RuntimeError(f"{res.status_code} - {res.text}")
    return res.json().get("summaries", [])


def view_summaries_section():
    st.header("📄 View & Edit All Summaries")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        q = st.text_input("Search by user, model, text...", key="admin_search_q")
    with col2:
        per_page = st.selectbox("Per page", [5, 10, 20, 50], index=1, key="admin_per_page")
    with col3:
        if st.button("🔄 Refresh"):
            if "admin_summaries_cache" in st.session_state:
                del st.session_state["admin_summaries_cache"]
            st.experimental_set_query_params(_refresh=str(datetime.utcnow().timestamp()))
            st.rerun()

    if "admin_summaries_cache" not in st.session_state:
        try:
            st.session_state["admin_summaries_cache"] = fetch_summaries_from_backend(q)
        except Exception as e:
            st.error(f"Failed to fetch summaries: {e}")
            return

    summaries = st.session_state["admin_summaries_cache"]
    if q:
        qlower = q.lower()
        summaries = [
            s for s in summaries
            if qlower in (s.get("input_text") or "").lower()
            or qlower in (s.get("summary_text") or "").lower()
            or qlower in (s.get("model_name") or "").lower()
            or qlower in (s.get("user_name") or "").lower()
            or qlower in (s.get("user_email") or "").lower()
        ]

    if not summaries:
        st.info("No summaries found.")
        return

    total = len(summaries)
    pages = math.ceil(total / per_page)
    page_num = st.number_input("Page", min_value=1, max_value=max(1, pages), value=1, step=1, key="admin_page_num")
    start = (page_num - 1) * per_page
    end = start + per_page
    page_items = summaries[start:end]
    st.markdown(f"**Showing {start+1}-{min(end,total)} of {total} summaries**")

    for s in page_items:
        summary_id = s.get("summary_id") or s.get("id")
        user_id = s.get("user_id")
        user_name = s.get("user_name", "")
        model_name = s.get("model_name", "")
        size = s.get("size", "")
        created_at = s.get("created_at", "")
        input_text = s.get("input_text", "")
        summary_text = s.get("summary_text", "")

        header = f"ID {summary_id} — User {user_name or user_id} — {model_name} / {size} — {created_at}"
        with st.expander(header, expanded=False):
            st.write("**Input text**")
            st.text_area(f"input_{summary_id}", input_text, height=120, key=f"input_{summary_id}_display")
            st.write("**Generated summary**")
            edited = st.text_area(f"summary_{summary_id}", summary_text, height=180, key=f"summary_{summary_id}_edit")

            bcol1, bcol2, _ = st.columns([1, 1, 6])

            with bcol1:
                if st.button("💾 Save", key=f"save_{summary_id}"):
                    payload = {"summary_text": edited, "model_name": model_name, "size": size}
                    try:
                        headers = {"Authorization": st.session_state.get("token")} if st.session_state.get("token") else {}
                        upd = requests.put(f"{API_ADMIN_SUMMARY}/{summary_id}", headers=headers, json=payload, timeout=10)
                        if upd.status_code in (200, 204):
                            st.success("Saved successfully.")
                            for idx, item in enumerate(st.session_state["admin_summaries_cache"]):
                                if item.get("summary_id") == summary_id:
                                    st.session_state["admin_summaries_cache"][idx]["summary_text"] = edited
                                    break
                        else:
                            try:
                                err = upd.json()
                            except Exception:
                                err = upd.text
                            st.error(f"Failed to save: {upd.status_code} - {err}")
                    except Exception as e:
                        st.error(f"Error while saving: {e}")

            with bcol2:
                delete_flag_key = f"to_delete_{summary_id}"
                if delete_flag_key not in st.session_state:
                    st.session_state[delete_flag_key] = False

                if not st.session_state.get(delete_flag_key):
                    if st.button("🗑️ Delete", key=f"delete_{summary_id}"):
                        st.session_state[delete_flag_key] = True
                else:
                    st.warning(f"Confirm delete summary {summary_id}? This action cannot be undone.")
                    confirm_col, cancel_col = st.columns([1, 1])
                    with confirm_col:
                        if st.button("Yes, Delete", key=f"confirm_delete_{summary_id}"):
                            try:
                                headers = {"Authorization": st.session_state.get("token")} if st.session_state.get("token") else {}
                                d = requests.delete(f"{API_ADMIN_SUMMARY}/{summary_id}", headers=headers, timeout=10)
                                if d.status_code in (200, 204):
                                    st.success("Deleted successfully.")
                                    st.session_state["admin_summaries_cache"] = [
                                        item for item in st.session_state["admin_summaries_cache"]
                                        if item.get("summary_id") != summary_id
                                    ]
                                    st.session_state[delete_flag_key] = False
                                else:
                                    try:
                                        err = d.json()
                                    except Exception:
                                        err = d.text
                                    st.error(f"Delete failed: {d.status_code} - {err}")
                            except Exception as e:
                                st.error(f"Error deleting: {e}")
                    with cancel_col:
                        if st.button("Cancel", key=f"cancel_delete_{summary_id}"):
                            st.session_state[delete_flag_key] = False

            st.write("---")
            meta_cols = st.columns(4)
            meta_cols[0].write(f"**Summary ID**\n{summary_id}")
            meta_cols[1].write(f"**User ID**\n{user_id}")
            meta_cols[2].write(f"**Model**\n{model_name}")
            meta_cols[3].write(f"**Created**\n{created_at}")

    st.write("---")
    st.write(f"Page {page_num} / {pages}")


# ---------------------- Paraphrases section (new) ----------------------
def fetch_paraphrases_from_backend(q: str = ""):
    headers = {"Authorization": st.session_state.get("token")} if st.session_state.get("token") else {}
    params = {"q": q} if q else {}
    res = requests.get(API_ADMIN_PARAPHS, headers=headers, params=params, timeout=15)
    if res.status_code != 200:
        raise RuntimeError(f"{res.status_code} - {res.text}")
    return res.json().get("paraphrases", [])


def view_paraphrases_section():
    st.header("✍️ View & Edit All Paraphrases")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        q = st.text_input("Search by user, model, text...", key="admin_par_q")
    with col2:
        per_page = st.selectbox("Per page", [5, 10, 20, 50], index=1, key="admin_par_per_page")
    with col3:
        if st.button("🔄 Refresh Paraphrases"):
            if "admin_paraphrases_cache" in st.session_state:
                del st.session_state["admin_paraphrases_cache"]
            st.experimental_set_query_params(_refresh_par=str(datetime.utcnow().timestamp()))
            st.rerun()

    if "admin_paraphrases_cache" not in st.session_state:
        try:
            st.session_state["admin_paraphrases_cache"] = fetch_paraphrases_from_backend(q)
        except Exception as e:
            st.error(f"Failed to fetch paraphrases: {e}")
            return

    paraphrases = st.session_state["admin_paraphrases_cache"]
    if q:
        qlower = q.lower()
        paraphrases = [
            p for p in paraphrases
            if qlower in (p.get("input_text") or "").lower()
            or qlower in (p.get("paraphrase_text") or "").lower()
            or qlower in (p.get("model_name") or "").lower()
            or qlower in (p.get("user_name") or "").lower()
            or qlower in (p.get("user_email") or "").lower()
        ]

    if not paraphrases:
        st.info("No paraphrases found.")
        return

    total = len(paraphrases)
    pages = math.ceil(total / per_page)
    page_num = st.number_input("Page", min_value=1, max_value=max(1, pages), value=1, step=1, key="admin_par_page_num")
    start = (page_num - 1) * per_page
    end = start + per_page
    page_items = paraphrases[start:end]
    st.markdown(f"**Showing {start+1}-{min(end,total)} of {total} paraphrases**")

    for p in page_items:
        par_id = p.get("paraphrase_id") or p.get("id")
        user_id = p.get("user_id")
        user_name = p.get("user_name", "")
        model_name = p.get("model_name", "")
        size = p.get("size", "")
        complexity = p.get("complexity", "")
        created_at = p.get("created_at", "")
        input_text = p.get("input_text", "")
        paraphrase_text = p.get("paraphrase_text", "")

        header = f"ID {par_id} — User {user_name or user_id} — {model_name} / {size} / {complexity} — {created_at}"
        with st.expander(header, expanded=False):
            st.write("**Input text**")
            st.text_area(f"par_input_{par_id}", input_text, height=120, key=f"par_input_{par_id}_display")
            st.write("**Generated paraphrase**")
            edited = st.text_area(f"par_{par_id}", paraphrase_text, height=180, key=f"par_{par_id}_edit")

            bcol1, bcol2, _ = st.columns([1, 1, 6])

            with bcol1:
                if st.button("💾 Save", key=f"save_par_{par_id}"):
                    payload = {
                        "paraphrase_text": edited,
                        "model_name": model_name,
                        "size": size,
                        "complexity": complexity
                    }
                    try:
                        headers = {"Authorization": st.session_state.get("token")} if st.session_state.get("token") else {}
                        upd = requests.put(f"{API_ADMIN_PARAPH}/{par_id}", headers=headers, json=payload, timeout=10)
                        if upd.status_code in (200, 204):
                            st.success("Saved successfully.")
                            for idx, item in enumerate(st.session_state["admin_paraphrases_cache"]):
                                if item.get("paraphrase_id") == par_id:
                                    st.session_state["admin_paraphrases_cache"][idx]["paraphrase_text"] = edited
                                    break
                        else:
                            try:
                                err = upd.json()
                            except Exception:
                                err = upd.text
                            st.error(f"Failed to save: {upd.status_code} - {err}")
                    except Exception as e:
                        st.error(f"Error while saving: {e}")

            with bcol2:
                delete_flag_key = f"to_delete_par_{par_id}"
                if delete_flag_key not in st.session_state:
                    st.session_state[delete_flag_key] = False

                if not st.session_state.get(delete_flag_key):
                    if st.button("🗑️ Delete", key=f"delete_par_{par_id}"):
                        st.session_state[delete_flag_key] = True
                else:
                    st.warning(f"Confirm delete paraphrase {par_id}? This action cannot be undone.")
                    confirm_col, cancel_col = st.columns([1, 1])
                    with confirm_col:
                        if st.button("Yes, Delete", key=f"confirm_delete_par_{par_id}"):
                            try:
                                headers = {"Authorization": st.session_state.get("token")} if st.session_state.get("token") else {}
                                d = requests.delete(f"{API_ADMIN_PARAPH}/{par_id}", headers=headers, timeout=10)
                                if d.status_code in (200, 204):
                                    st.success("Deleted successfully.")
                                    st.session_state["admin_paraphrases_cache"] = [
                                        item for item in st.session_state["admin_paraphrases_cache"]
                                        if item.get("paraphrase_id") != par_id
                                    ]
                                    st.session_state[delete_flag_key] = False
                                else:
                                    try:
                                        err = d.json()
                                    except Exception:
                                        err = d.text
                                    st.error(f"Delete failed: {d.status_code} - {err}")
                            except Exception as e:
                                st.error(f"Error deleting: {e}")
                    with cancel_col:
                        if st.button("Cancel", key=f"cancel_delete_par_{par_id}"):
                            st.session_state[delete_flag_key] = False

            st.write("---")
            meta_cols = st.columns(4)
            meta_cols[0].write(f"**Paraphrase ID**\n{par_id}")
            meta_cols[1].write(f"**User ID**\n{user_id}")
            meta_cols[2].write(f"**Model**\n{model_name}")
            meta_cols[3].write(f"**Created**\n{created_at}")

    st.write("---")
    st.write(f"Page {page_num} / {pages}")


def view_usage_section():
    """Admin: show usage stats (totals, thumbs up/down pie, feedback table, user counts)."""
    st.header("📊 Usage & Feedback Statistics")
    headers = {"Authorization": st.session_state.get("token")} if st.session_state.get("token") else {}

    # Controls
    col1, col2 = st.columns([3,1])
    with col1:
        q = st.text_input("Search feedback (text, user email/name)...", key="usage_search_q")
    with col2:
        if st.button("🔄 Refresh Usage"):
            # clear cache and return to let Streamlit re-render and re-fetch
            st.session_state.pop("admin_usage_cache", None)
            st.session_state["_usage_refresh_ts"] = str(datetime.utcnow().timestamp())
            return

    # Fetch usage data (cached)
    if "admin_usage_cache" not in st.session_state:
        try:
            res = requests.get(API_ADMIN_USAGE, headers=headers, timeout=20)
            if res.status_code != 200:
                st.error(f"Failed to fetch usage: {res.status_code} - {res.text}")
                return
            st.session_state["admin_usage_cache"] = res.json()
        except Exception as e:
            st.error(f"Error calling usage endpoint: {e}")
            return

    data = st.session_state["admin_usage_cache"] or {}

    # Top-level metrics row (safe defaults)
    totals = data.get("totals", {})
    total_users = int(totals.get("total_users", 0))
    total_summaries = int(totals.get("total_summaries", 0))
    total_paraphrases = int(totals.get("total_paraphrases", 0))
    total_feedbacks = int(totals.get("total_feedbacks", 0))

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Users", total_users)
    col_b.metric("Total Summaries", total_summaries)
    col_c.metric("Total Paraphrases", total_paraphrases)
    col_d.metric("Total Feedbacks", total_feedbacks)

    st.markdown("---")

    # Thumbs up / down pie chart (safe defaults)
    fb_counts = data.get("feedback_counts") or {}
    up = int(fb_counts.get("up", 0))
    down = int(fb_counts.get("down", 0))

    st.subheader("👍 / 👎 Feedback Distribution")
    try:
        import matplotlib.pyplot as plt
        labels = [f"👍 Up ({up})", f"👎 Down ({down})"]
        sizes = [up, down]
        fig, ax = plt.subplots()
        if sum(sizes) == 0:
            # draw a single-segment neutral pie
            ax.pie([1], labels=["No feedback yet"], autopct=None)
        else:
            ax.pie(sizes, labels=labels, autopct=lambda pct: f"{pct:.1f}% ({int(round(pct * sum(sizes) / 100))})")
        ax.set_aspect('equal')
        st.pyplot(fig)
    except Exception:
        st.info("Could not render pie chart.")
        st.write(f"Up: {up}, Down: {down}")

    st.markdown("---")

    # Feedback details table (searchable)
    st.subheader("🗣️ User Feedbacks (most recent first)")
    feedback_rows = data.get("feedback_rows") or []

    # Apply simple search filter if user entered q
    if q:
        qlow = q.lower()
        feedback_rows = [
            r for r in feedback_rows
            if qlow in (r.get("comment") or "").lower()
            or qlow in (r.get("user_name") or "").lower()
            or qlow in (r.get("user_email") or "").lower()
            or qlow in (r.get("task_type") or "").lower()
        ]

    if not feedback_rows:
        st.info("No feedback rows to show.")
    else:
        import pandas as pd
        # limit to 200 rows for UI performance
        df_rows = pd.DataFrame(feedback_rows[:200])
        # normalize column order for better look
        cols = ["feedback_id","user_id","user_name","user_email","task_type","task_id","rating","comment","created_at"]
        cols = [c for c in cols if c in df_rows.columns] + [c for c in df_rows.columns if c not in cols]
        st.dataframe(df_rows[cols], use_container_width=True)

    st.markdown("---")

    # User-level stats: number of users who gave feedback and top contributors
    st.subheader("👥 Feedback contributors & user details")
    user_stats = data.get("user_feedback_stats") or []
    if not user_stats:
        st.info("No user-level stats available.")
    else:
        import pandas as pd
        df_users = pd.DataFrame(user_stats)
        # friendly column order
        ucols = ["user_id","user_name","user_email","feedback_count","up_count","down_count","last_feedback"]
        ucols = [c for c in ucols if c in df_users.columns] + [c for c in df_users.columns if c not in ucols]
        st.dataframe(df_users[ucols], use_container_width=True)

    st.markdown("---")

    # ---------- CLEAN "Quick Breakdown" UI ----------
    st.subheader("📌 Quick Breakdown")

    breakdown = data.get("breakdown") or {}
    # tolerate different shapes: prefer breakdown.by_task, else use totals as fallback
    top_models = breakdown.get("top_models") if isinstance(breakdown.get("top_models"), list) else []
    by_task = breakdown.get("by_task") if isinstance(breakdown.get("by_task"), dict) else {}
    top_users = breakdown.get("top_users") if isinstance(breakdown.get("top_users"), list) else []

    # fallbacks if backend didn't provide by_task
    s_count = int(by_task.get("summarization", totals.get("total_summaries", 0)))
    p_count = int(by_task.get("paraphrasing", totals.get("total_paraphrases", 0)))

    # Cards for quick glance
    bcol1, bcol2, bcol3 = st.columns([1.2, 1.2, 1.6])
    with bcol1:
        if top_models:
            t0 = top_models[0]
            st.metric("Top Model (calls)", f"{t0.get('model')} ({t0.get('count')})")
        else:
            st.metric("Top Model (calls)", "—")
    with bcol2:
        st.metric("Summaries / Paraphrases", f"{s_count} / {p_count}")
    with bcol3:
        if top_users:
            tu = top_users[0]
            st.metric("Top User (actions)", f"{tu.get('user_name')} ({tu.get('count')})")
        else:
            st.metric("Top User (actions)", "—")

    st.markdown("---")

    # ---------- Top models table ----------
    st.subheader("🏆 Top Models (by usage)")
    if not top_models:
        st.info("No model breakdown available.")
    else:
        import pandas as pd
        df_models = pd.DataFrame(top_models)
        st.dataframe(df_models, use_container_width=True)

    st.markdown("---")

    # ---------- By-task breakdown (small bar-like display) ----------
    st.subheader("🔎 By Task")
    st.write(f"Summarizations: **{s_count}**  ·  Paraphrases: **{p_count}**")

    # small inline bar-style visualization using simple text progress bars
    def draw_bar(value, maxv= max(s_count, p_count, 1), width=30):
        filled = int((value / maxv) * width) if maxv > 0 else 0
        return "█" * filled + "░" * (width - filled)

    st.text(f"Summaries   {draw_bar(s_count)}  {s_count}")
    st.text(f"Paraphrases {draw_bar(p_count)}  {p_count}")

    st.markdown("---")

    # ---------- Top users ----------
    st.subheader("👥 Top Users by Activity")
    if not top_users:
        st.info("No user breakdown available.")
    else:
        import pandas as pd
        df_users_top = pd.DataFrame(top_users)
        st.dataframe(df_users_top.head(50), use_container_width=True)

    st.markdown("---")

    st.caption("Tip: Use the search box above to filter feedback rows. Download the CSV to get full raw records.")

    # allow downloading a CSV of feedback rows
    if feedback_rows:
        import io, csv
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["feedback_id","user_id","user_name","user_email","task_type","task_id","rating","comment","created_at"])
        for r in feedback_rows:
            writer.writerow([
                r.get("feedback_id"),
                r.get("user_id"),
                r.get("user_name"),
                r.get("user_email"),
                r.get("task_type"),
                r.get("task_id"),
                r.get("rating"),
                (r.get("comment") or "").replace("\n"," "),
                r.get("created_at")
            ])
        st.download_button("⬇️ Download feedback CSV", data=buf.getvalue(), file_name="feedback_export.csv", mime="text/csv")


if __name__ == "__main__":
    admin_dashboard()
