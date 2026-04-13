"""
Email Spam Detection — Streamlit App
Prepared By: Sajid Ali

Usage:
    streamlit run app.py

Expects model.pkl (a joblib/pickle-serialised sklearn Pipeline with
steps named 'tfidf' and 'clf') in the same directory.
"""

import re
import pickle
import joblib
import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = pathlib.Path("model.pkl")

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove URLs / numbers / punctuation, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner="Loading model…")
def load_model() -> Pipeline:
    """Load the trained sklearn Pipeline from disk."""
    if not MODEL_PATH.exists():
        st.error(f"❌ `{MODEL_PATH}` not found. Place your trained pipeline in the same folder as `app.py`.")
        st.stop()
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    return model


def predict(model: Pipeline, text: str) -> tuple[str, float]:
    """Return (label, spam_probability)."""
    cleaned = clean_text(text)
    prob    = model.predict_proba([cleaned])[0][1]
    label   = "SPAM" if prob >= 0.5 else "HAM"
    return label, prob


# ── Styling ───────────────────────────────────────────────────────────────────
SPAM_COLOR = "#e74c3c"
HAM_COLOR  = "#2ecc71"

st.markdown("""
<style>
    .result-spam {
        background: linear-gradient(135deg, #ff6b6b22, #e74c3c11);
        border-left: 5px solid #e74c3c;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-ham {
        background: linear-gradient(135deg, #6bcb7722, #2ecc7111);
        border-left: 5px solid #2ecc71;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .big-label { font-size: 2rem; font-weight: 800; }
    .sub-text  { font-size: 0.95rem; color: #888; margin-top: 0.3rem; }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #4a4a4a;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/spam.png", width=64)
    st.title("📧 Spam Detector")
    st.markdown("**Model:** TF-IDF + Logistic Regression")
    st.markdown("**Author:** Sajid Ali")
    st.divider()

    st.markdown("### ⚙️ Settings")
    threshold = st.slider(
        "Spam Probability Threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Emails with spam probability ≥ this value are flagged as SPAM."
    )
    show_keywords = st.checkbox("Show top spam/ham keywords", value=True)
    top_n = st.slider("Number of keywords to show", 5, 30, 15, disabled=not show_keywords)

    st.divider()
    st.caption("Tip: Paste any email text in the main panel and click **Analyze**.")


# ── Main Layout ───────────────────────────────────────────────────────────────
st.title("📧 Email Spam Detection")
st.markdown("Detect whether an email is **spam** or **ham (legitimate)** using a trained TF-IDF + Logistic Regression model.")

tab1, tab2, tab3 = st.tabs(["🔍 Single Email", "📋 Batch Analysis", "📊 Model Insights"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Email Prediction
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Enter an email to analyse</div>', unsafe_allow_html=True)

    # Quick sample emails
    samples = {
        "Select a sample…": "",
        "🚨 Spam #1 — Prize winner": "Congratulations! You've won a FREE iPhone. Click here to claim your prize now!",
        "🚨 Spam #2 — Bank alert": "URGENT: Your bank account has been suspended. Verify now to avoid charges.",
        "🚨 Spam #3 — Cash prize": "Win $1000 cash prize! Send your details to claim immediately!",
        "✅ Ham #1 — Lunch invite": "Hey, are we still meeting for lunch tomorrow at 12?",
        "✅ Ham #2 — Work email": "Please find the attached quarterly report for your review.",
        "✅ Ham #3 — Code review": "Can you please review the pull request I submitted this morning?",
    }

    chosen = st.selectbox("Quick samples", list(samples.keys()))
    prefill = samples[chosen]

    email_input = st.text_area(
        "Email text",
        value=prefill,
        height=200,
        placeholder="Paste or type the email body here…",
        label_visibility="collapsed",
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze", use_container_width=True, type="primary")
    with col_clear:
        if st.button("🗑️ Clear"):
            email_input = ""

    if analyze_clicked:
        if not email_input.strip():
            st.warning("Please enter some email text first.")
        else:
            cleaned    = clean_text(email_input)
            prob       = model.predict_proba([cleaned])[0][1]
            is_spam    = prob >= threshold
            label      = "SPAM" if is_spam else "HAM"
            color      = SPAM_COLOR if is_spam else HAM_COLOR
            icon       = "🚨" if is_spam else "✅"
            css_class  = "result-spam" if is_spam else "result-ham"
            confidence = prob if is_spam else (1 - prob)

            st.markdown(
                f"""
                <div class="{css_class}">
                    <div class="big-label" style="color:{color}">{icon} {label}</div>
                    <div class="sub-text">
                        Spam probability: <strong>{prob*100:.1f}%</strong> &nbsp;|&nbsp;
                        Confidence: <strong>{confidence*100:.1f}%</strong> &nbsp;|&nbsp;
                        Threshold: <strong>{threshold:.0%}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Probability gauge
            fig, ax = plt.subplots(figsize=(7, 0.6))
            ax.barh([""], [prob],        color=SPAM_COLOR, alpha=0.85, height=0.5)
            ax.barh([""], [1 - prob], left=[prob], color=HAM_COLOR, alpha=0.85, height=0.5)
            ax.axvline(threshold, color="white", linewidth=2, linestyle="--")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Spam probability →")
            ax.set_xticks([0, 0.25, 0.5, threshold, 0.75, 1.0])
            ax.set_xticklabels(["0%", "25%", "50%", f"{threshold:.0%}\n(threshold)", "75%", "100%"], fontsize=8)
            ax.tick_params(left=False, labelleft=False)
            ax.spines[:].set_visible(False)
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Analysis
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Analyse multiple emails at once</div>', unsafe_allow_html=True)

    input_method = st.radio("Input method", ["Paste emails (one per line)", "Upload CSV file"], horizontal=True)

    emails_to_check = []

    if input_method == "Paste emails (one per line)":
        bulk_text = st.text_area(
            "Emails (one per line)",
            height=200,
            placeholder="Paste each email on a new line…",
            label_visibility="collapsed",
        )
        if bulk_text.strip():
            emails_to_check = [e.strip() for e in bulk_text.strip().splitlines() if e.strip()]

    else:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.dataframe(df_up.head(), use_container_width=True)
            col_name = st.selectbox("Which column contains the email text?", df_up.columns.tolist())
            emails_to_check = df_up[col_name].astype(str).tolist()

    if st.button("🔍 Analyze All", type="primary") and emails_to_check:
        with st.spinner(f"Classifying {len(emails_to_check)} emails…"):
            cleaned_list = [clean_text(e) for e in emails_to_check]
            probs        = model.predict_proba(cleaned_list)[:, 1]
            labels       = ["🚨 SPAM" if p >= threshold else "✅ HAM" for p in probs]

        results_df = pd.DataFrame({
            "Email Preview" : [e[:90] + ("…" if len(e) > 90 else "") for e in emails_to_check],
            "Prediction"    : labels,
            "Spam Prob %"   : [f"{p*100:.1f}%" for p in probs],
        })

        spam_count = sum(1 for l in labels if "SPAM" in l)
        ham_count  = len(labels) - spam_count

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Emails", len(emails_to_check))
        m2.metric("🚨 Spam Detected", spam_count)
        m3.metric("✅ Ham (Legitimate)", ham_count)

        # Pie chart
        if spam_count > 0 or ham_count > 0:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(
                [spam_count, ham_count],
                labels=["Spam", "Ham"],
                colors=[SPAM_COLOR, HAM_COLOR],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=2),
            )
            ax.set_title("Batch Result Distribution", fontweight="bold")
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        st.dataframe(
            results_df,
            use_container_width=True,
            column_config={
                "Spam Prob %" : st.column_config.TextColumn("Spam Prob %"),
                "Prediction"  : st.column_config.TextColumn("Prediction"),
            },
        )

        csv_out = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results as CSV", csv_out, "spam_results.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights (top keywords)
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Model Insights</div>', unsafe_allow_html=True)

    try:
        vectorizer    = model.named_steps["tfidf"]
        clf           = model.named_steps["clf"]
        feature_names = np.array(vectorizer.get_feature_names_out())
        coef          = clf.coef_[0]

        if show_keywords:
            top_spam_idx = coef.argsort()[-top_n:][::-1]
            top_ham_idx  = coef.argsort()[:top_n]

            spam_words   = feature_names[top_spam_idx]
            spam_weights = coef[top_spam_idx]
            ham_words    = feature_names[top_ham_idx]
            ham_weights  = coef[top_ham_idx]

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f"#### 🚨 Top {top_n} Spam Keywords")
                fig, ax = plt.subplots(figsize=(6, top_n * 0.38 + 1))
                ax.barh(spam_words[::-1], spam_weights[::-1], color=SPAM_COLOR, alpha=0.85, edgecolor="white")
                ax.set_xlabel("Logistic Regression Weight")
                ax.set_title(f"Top {top_n} SPAM Keywords", fontweight="bold", color=SPAM_COLOR)
                ax.axvline(0, color="black", linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with col_b:
                st.markdown(f"#### ✅ Top {top_n} Ham Keywords")
                fig, ax = plt.subplots(figsize=(6, top_n * 0.38 + 1))
                ax.barh(ham_words, np.abs(ham_weights), color=HAM_COLOR, alpha=0.85, edgecolor="white")
                ax.set_xlabel("|Logistic Regression Weight|")
                ax.set_title(f"Top {top_n} HAM Keywords", fontweight="bold", color="#27ae60")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # Model info table
        st.divider()
        st.markdown("#### ℹ️ Pipeline Configuration")
        tfidf_params = vectorizer.get_params()
        clf_params   = clf.get_params()

        info = {
            "TF-IDF — max_features"  : tfidf_params.get("max_features"),
            "TF-IDF — ngram_range"   : str(tfidf_params.get("ngram_range")),
            "TF-IDF — sublinear_tf"  : tfidf_params.get("sublinear_tf"),
            "TF-IDF — stop_words"    : tfidf_params.get("stop_words"),
            "LogReg — C"             : clf_params.get("C"),
            "LogReg — solver"        : clf_params.get("solver"),
            "LogReg — class_weight"  : clf_params.get("class_weight"),
            "Vocabulary size"        : len(feature_names),
        }
        st.table(pd.DataFrame(info.items(), columns=["Parameter", "Value"]))

    except (KeyError, AttributeError) as exc:
        st.warning(f"Could not extract model internals: {exc}. Make sure the pipeline has steps named `tfidf` and `clf`.")
