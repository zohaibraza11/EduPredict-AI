"""EduPredict AI - Student Performance Detection System (Streamlit UI)"""
import pandas as pd
import streamlit as st
import model as m

st.set_page_config(page_title="EduPredict AI", page_icon="🎓", layout="wide")
st.markdown("""<style>
html,body,[class*="css"]{font-family:Georgia,'Times New Roman',serif}
.stApp{background:#faf7f0}
[data-testid="stSidebar"]{background:#1f2a44}
[data-testid="stSidebar"] *{color:#f5f1e6!important}
[data-testid="stSidebar"] label{font-size:1.02rem;padding:3px 0}
.brand{font-size:1.4rem;font-weight:700;border-bottom:2px solid #c9a227;padding-bottom:10px;margin-bottom:10px}
.hdr{background:#fff;border:1px solid #e2dccb;border-left:6px solid #c9a227;border-radius:6px;padding:16px 22px;margin-bottom:20px}
.hdr h2{margin:0;color:#1f2a44}.hdr p{margin:2px 0 0;color:#6b6b6b}
.card{border-radius:8px;padding:20px;text-align:center;color:#fff;margin-bottom:12px}
.card h2{margin:0;font-size:2rem;color:#fff}
.stButton>button,.stFormSubmitButton>button{background:#1f2a44;color:#fff;border:1px solid #c9a227;border-radius:6px;padding:.5rem 1.4rem}
</style>""", unsafe_allow_html=True)

COLORS = {"At Risk": "#c0392b", "Average": "#d68910", "Good": "#2471a3", "Excellent": "#1e8449"}
EMOJI = {"At Risk": "⚠️", "Average": "📘", "Good": "👍", "Excellent": "🏆"}


@st.cache_resource
def get_model():
    return m.load_model()


def header(title, sub):
    st.markdown(f'<div class="hdr"><h2>{title}</h2><p>{sub}</p></div>', unsafe_allow_html=True)


def result_card(label, text):
    st.markdown(f'<div class="card" style="background:{COLORS[label]}"><p>{text}</p>'
                f'<h2>{EMOJI[label]} {label}</h2></div>', unsafe_allow_html=True)


def show_detail(row, label):
    """Full performance details of one student."""
    vals = {f: float(row[f]) for f in m.FEATURES}
    result_card(label, f"{row['name']} ({row['id']})")
    cols = st.columns(4)
    for i, f in enumerate(m.FEATURES):
        cols[i % 4].metric(f.replace("_", " ").title(), vals[f])
    _, proba = m.predict(clf, pd.DataFrame([vals]))
    a, b = st.columns(2)
    a.markdown("**AI confidence**")
    a.bar_chart(pd.Series(proba).reindex(m.LABELS))
    b.markdown("**Suggestions**")
    for t in m.suggestions(vals):
        b.info(t)


clf = get_model()
df = m.get_students()

st.sidebar.markdown('<div class="brand">🎓 EduPredict AI</div>', unsafe_allow_html=True)
st.sidebar.caption("Student Performance Detection System")
page = st.sidebar.radio("Menu", ["🏛️ Dashboard", "➕ Predict & Save", "🔍 Student Records",
                                 "⚠️ At-Risk Alerts", "📈 Analytics", "🧠 Model Comparison",
                                 "📘 AI Methodology", "📁 Batch Prediction"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.metric("Students on record", len(df))

if page.endswith("Dashboard"):
    header("Dashboard", "Overview of all students on record")
    k = st.columns(4)
    k[0].metric("Total Students", len(df))
    k[1].metric("Avg Attendance", f"{df.attendance.mean():.1f}%")
    k[2].metric("At Risk", int((df.performance == "At Risk").sum()))
    k[3].metric("Excellent", int((df.performance == "Excellent").sum()))
    a, b = st.columns(2)
    a.markdown("**Performance distribution**")
    a.bar_chart(df.performance.value_counts().reindex(m.LABELS).fillna(0))
    b.markdown("**Recently added students**")
    b.dataframe(df.tail(8)[["id", "name", "performance", "added_on"]], hide_index=True, use_container_width=True)

elif page.endswith("Predict & Save"):
    header("Predict & Save", "Detect a student's performance. The record is saved automatically.")
    with st.form("predict"):
        name = st.text_input("Student name")
        c1, c2 = st.columns(2)
        att = c1.slider("Attendance (%)", 0, 100, 80)
        study = c1.slider("Daily study hours", 0.0, 10.0, 3.0, 0.5)
        assign = c1.slider("Assignment score", 0, 100, 70)
        quiz = c1.slider("Quiz score", 0, 100, 65)
        mid = c2.slider("Midterm score", 0, 100, 65)
        sleep = c2.slider("Sleep hours", 3.0, 10.0, 7.0, 0.5)
        gpa = c2.slider("Previous GPA", 1.0, 4.0, 2.8, 0.1)
        go = st.form_submit_button("🚀 Predict & Save")
    if go:
        if not name.strip():
            st.error("Please enter the student name.")
        else:
            vals = dict(attendance=att, study_hours=study, assignment_score=assign,
                        quiz_score=quiz, midterm_score=mid, sleep_hours=sleep, previous_gpa=gpa)
            label, proba = m.predict(clf, pd.DataFrame([vals]))
            sid = m.add_student(name.strip(), vals, label)
            st.success(f"Saved as {sid}. Total students on record: {len(df) + 1}")
            result_card(label, f"{name.strip()} - predicted performance (Ensemble)")
            pm = m.predict_models(clf, pd.DataFrame([vals]))
            st.caption("Individual models:  " + "   |   ".join(f"{k}: **{v}**" for k, v in pm.items()))
            a, b = st.columns(2)
            a.markdown("**AI confidence**")
            a.bar_chart(pd.Series(proba).reindex(m.LABELS))
            b.markdown("**Suggestions**")
            for t in m.suggestions(vals):
                b.info(t)

elif page.endswith("Student Records"):
    header("Student Records", "Search by name, then click a row to see the full performance details")
    c1, c2 = st.columns([2, 1])
    q = c1.text_input("🔍 Search student name")
    flt = c2.multiselect("Filter by performance", m.LABELS)
    view = df[df.name.str.contains(q, case=False, na=False)] if q else df
    if flt:
        view = view[view.performance.isin(flt)]
    view = view.reset_index(drop=True)
    st.caption(f"{len(view)} student(s) found")
    ev = st.dataframe(view[["id", "name", "performance"]], on_select="rerun",
                      selection_mode="single-row", hide_index=True, use_container_width=True)
    if ev.selection.rows:
        s = view.iloc[ev.selection.rows[0]]
        st.markdown("---")
        show_detail(s, s["performance"])
    st.download_button("⬇️ Download all records (CSV)", df.to_csv(index=False), "student_records.csv")

elif page.endswith("At-Risk Alerts"):
    header("At-Risk Alerts", "Students who need immediate attention")
    risk = df[df.performance == "At Risk"].copy()
    risk["main_concern"] = risk.apply(lambda r: m.suggestions(r[m.FEATURES].to_dict())[0], axis=1)
    st.metric("Students at risk", len(risk))
    st.dataframe(risk[["id", "name", "attendance", "midterm_score", "previous_gpa", "main_concern"]]
                 .sort_values("midterm_score"), hide_index=True, use_container_width=True)

elif page.endswith("Analytics"):
    header("Analytics", "How each factor relates to performance")
    st.markdown("**Average of each factor per performance category**")
    st.dataframe(df.groupby("performance")[m.FEATURES].mean().reindex(m.LABELS).round(1), use_container_width=True)
    f = st.selectbox("Compare a factor", m.FEATURES)
    st.bar_chart(df.groupby("performance")[f].mean().reindex(m.LABELS))

elif page.endswith("Model Comparison"):
    header("Model Comparison", "KNN and Naive Bayes (self-written) and their Ensemble, tested on unseen students")
    ev = clf["eval"]
    st.caption(f"Trained on {clf['n_train']} students, tested on {clf['n_test']} unseen students.")
    tbl = pd.DataFrame({n: {"Accuracy": e["accuracy"], "Precision": e["precision"], "Recall": e["recall"],
                            "F1-score": e["f1"], "Prediction time (s)": e["time"]} for n, e in ev.items()}).T.round(3)
    st.dataframe(tbl, use_container_width=True)
    st.bar_chart(tbl[["Accuracy", "Precision", "Recall", "F1-score"]])
    pick = st.selectbox("Show details for", list(ev))
    a, b = st.columns(2)
    a.markdown("**Confusion matrix**")
    a.dataframe(ev[pick]["cm"], use_container_width=True)
    b.markdown("**Per-class precision / recall / F1**")
    b.dataframe(ev[pick]["per_class"], use_container_width=True)
    st.info("Precision = of the students predicted in a class, how many truly belong. "
            "Recall = of the students truly in a class, how many were found.")
    if st.button("Retrain models"):
        st.success(f"Retrained. Ensemble accuracy: {m.train():.2%}")
        st.cache_resource.clear()

elif page.endswith("AI Methodology"):
    header("AI Methodology & Ethics", "Why these techniques, how they work, and their limits")
    st.markdown("""
**Problem:** Detect early which students are *At Risk* so teachers can help them in time.

**Why KNN?** Students with similar attendance, marks and habits tend to perform similarly. KNN uses exactly this idea.
**Why Naive Bayes?** It is fast, works with small data and gives clear probabilities for each class.

**KNN - step by step**
1. Standardise all 7 features. 2. Measure the distance from the new student to every training student.
3. Take the 7 nearest. 4. The majority class among them is the prediction.

**Naive Bayes - step by step**
1. Learn each class's prior probability, and the mean and variance of every feature in that class.
2. For a new student multiply prior x likelihood of each feature (Gaussian). 3. The class with the highest probability wins.

**Integration (Ensemble):** both models output class probabilities; the system averages them (soft voting). One model's weakness is covered by the other.

**Ethics & limitations**
- **Privacy:** student records are personal data; use only with permission and keep them secure.
- **Bias:** the model must not judge a student by gender, background or similar traits. None are used here.
- **Misuse:** the result is a *support tool*, not a final judgement. A teacher must make the decision, and a label must never be used to punish.
- **Limitations:** the data is synthetic, so real-world accuracy would be lower. Only 7 factors are considered.
""")

else:
    header("Batch Prediction", "Predict performance for many students from a CSV file")
    st.write(f"Required columns: `{', '.join(m.FEATURES)}`")
    st.download_button("⬇️ Sample CSV", m.generate_data(10).drop(columns="performance").to_csv(index=False), "sample.csv")
    f = st.file_uploader("Upload CSV", type="csv")
    if f:
        try:
            d = pd.read_csv(f)
        except Exception:
            st.error("Could not read this file. Please upload a valid CSV.")
            st.stop()
        miss = [c for c in m.FEATURES if c not in d.columns]
        if miss:
            st.error(f"Missing columns: {miss}")
        else:
            try:
                d["predicted_performance"] = m.predict_df(clf, d)
            except ValueError as e:
                st.error(str(e))
                st.stop()
            st.dataframe(d, use_container_width=True)
            st.download_button("⬇️ Download results", d.to_csv(index=False), "results.csv")
