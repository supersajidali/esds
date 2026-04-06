# =============================================================================
#  Email Spam Detection — Streamlit App
#  Prepared By: Sajid Ali
#  Run: streamlit run app.py
# =============================================================================

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
TEXT_COL     = "text"
LABEL_COL    = "label"
SPAM_VALUE   = "spam"
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide",
)

st.markdown("""
    <style>
        .main-title {font-size:2.4rem; font-weight:800; color:#e74c3c; text-align:center;}
        .sub-title  {font-size:1.1rem; color:#555; text-align:center; margin-bottom:1.5rem;}
        .metric-card{background:#f8f9fa; border-radius:12px; padding:18px; text-align:center;
                     border:1px solid #dee2e6; box-shadow:0 2px 6px rgba(0,0,0,.06);}
        .spam-badge {background:#e74c3c; color:white; padding:6px 18px; border-radius:20px;
                     font-size:1.1rem; font-weight:700;}
        .ham-badge  {background:#2ecc71; color:white; padding:6px 18px; border-radius:20px;
                     font-size:1.1rem; font-weight:700;}
        .section-header{font-size:1.25rem; font-weight:700; color:#2c3e50;
                         border-left:4px solid #e74c3c; padding-left:10px; margin:1rem 0 .5rem;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📧 Email Spam Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">TF-IDF + Logistic Regression · Prepared by Sajid Ali</p>', unsafe_allow_html=True)

# =============================================================================
# Helpers
# =============================================================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner="🔧 Training model…")
def train_pipeline(df: pd.DataFrame):
    df = df[[TEXT_COL, LABEL_COL]].dropna().copy()
    df["clean_text"] = df[TEXT_COL].apply(clean_text)
    df["spam"] = (df[LABEL_COL].astype(str).str.lower() == SPAM_VALUE).astype(int)

    X, y = df["clean_text"], df["spam"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=50_000,
            sublinear_tf=True, min_df=2, stop_words="english",
        )),
        ("clf", LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])

    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)
    pipeline.fit(X_train, y_train)

    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    return pipeline, df, X_train, X_test, y_train, y_test, y_pred, y_pred_prob, cv_f1


# =============================================================================
# Sidebar — Upload
# =============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload emails.csv", type=["csv"])
    st.markdown("---")
    st.markdown("**Expected columns:**")
    st.code("label , text", language="text")
    st.markdown("Labels should be `spam` / `ham`")
    st.markdown("---")
    st.markdown("**Model:** Logistic Regression  \n**Features:** TF-IDF (1-gram + 2-gram)")

if uploaded is None:
    st.info("👈  Upload your **emails.csv** in the sidebar to get started.")
    st.stop()

# =============================================================================
# Load & Train
# =============================================================================
raw_df = pd.read_csv(uploaded)

if TEXT_COL not in raw_df.columns or LABEL_COL not in raw_df.columns:
    st.error(f"CSV must contain columns: `{LABEL_COL}` and `{TEXT_COL}`")
    st.stop()

pipeline, df, X_train, X_test, y_train, y_test, y_pred, y_pred_prob, cv_f1 = train_pipeline(raw_df)

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_pred_prob)

# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Predict", "📊 EDA", "📈 Model Performance", "🔑 Keywords", "📋 Dataset"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Predict
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">Classify an Email</p>', unsafe_allow_html=True)

    user_email = st.text_area(
        "Paste or type email text below:",
        placeholder="e.g. Congratulations! You've won a FREE iPhone…",
        height=160,
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        classify = st.button("🚀 Classify", use_container_width=True)

    if classify:
        if not user_email.strip():
            st.warning("Please enter some email text.")
        else:
            cleaned = clean_text(user_email)
            pred    = pipeline.predict([cleaned])[0]
            prob    = pipeline.predict_proba([cleaned])[0][1]

            st.markdown("---")
            c1, c2, c3 = st.columns(3)

            with c1:
                label_html = (
                    '<span class="spam-badge">🚨 SPAM</span>'
                    if pred == 1 else
                    '<span class="ham-badge">✅ HAM</span>'
                )
                st.markdown(f"**Verdict:** {label_html}", unsafe_allow_html=True)

            with c2:
                st.metric("Spam Probability", f"{prob*100:.1f}%")

            with c3:
                st.metric("Ham Probability", f"{(1-prob)*100:.1f}%")

            # Probability bar
            st.markdown("**Confidence:**")
            prob_df = pd.DataFrame({
                "Class": ["Ham", "Spam"],
                "Probability": [(1 - prob) * 100, prob * 100],
            })
            fig, ax = plt.subplots(figsize=(7, 1.4))
            colors = ["#2ecc71", "#e74c3c"]
            ax.barh(prob_df["Class"], prob_df["Probability"], color=colors, edgecolor="white", height=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)")
            ax.axvline(50, color="gray", linewidth=0.8, linestyle="--")
            for i, v in enumerate(prob_df["Probability"]):
                ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")
    st.markdown('<p class="section-header">Batch Predict on Sample Emails</p>', unsafe_allow_html=True)

    samples = [
        "Congratulations! You've won a FREE iPhone. Click here to claim your prize now!",
        "Hey, are we still meeting for lunch tomorrow at 12?",
        "URGENT: Your bank account has been suspended. Verify now to avoid charges.",
        "Please find the attached quarterly report for your review.",
        "Win $1000 cash prize! Send your details to claim immediately!",
        "Can you please review the pull request I submitted this morning?",
    ]

    cleaned_s = [clean_text(e) for e in samples]
    preds_s   = pipeline.predict(cleaned_s)
    probs_s   = pipeline.predict_proba(cleaned_s)[:, 1]

    results = pd.DataFrame({
        "Email"       : [e[:80] + ("…" if len(e) > 80 else "") for e in samples],
        "Verdict"     : ["🚨 SPAM" if p == 1 else "✅ HAM" for p in preds_s],
        "Spam Prob %" : [f"{p*100:.1f}%" for p in probs_s],
    })
    st.dataframe(results, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)

    counts = df[LABEL_COL].value_counts()
    spam_n = counts.get(SPAM_VALUE, 0)
    ham_n  = counts.get("ham", 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Emails", f"{len(df):,}")
    m2.metric("Spam", f"{spam_n:,}")
    m3.metric("Ham",  f"{ham_n:,}")
    m4.metric("Spam Rate", f"{spam_n/len(df)*100:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = ["#e74c3c", "#2ecc71"]
        ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title("Email Count by Label", fontweight="bold")
        ax.set_ylabel("Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
               colors=colors, startangle=90,
               wedgeprops=dict(edgecolor="white", linewidth=2))
        ax.set_title("Spam vs Ham Ratio", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<p class="section-header">Text Length Analysis</p>', unsafe_allow_html=True)

    df["text_length"] = df[TEXT_COL].astype(str).apply(len)
    df["word_count"]  = df[TEXT_COL].astype(str).apply(lambda x: len(x.split()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for label, color in zip([SPAM_VALUE, "ham"], ["#e74c3c", "#2ecc71"]):
        subset = df[df[LABEL_COL] == label]
        axes[0].hist(subset["text_length"], bins=25, alpha=0.65, label=label, color=color, edgecolor="white")
        axes[1].hist(subset["word_count"],  bins=25, alpha=0.65, label=label, color=color, edgecolor="white")

    axes[0].set_title("Character Length Distribution", fontweight="bold")
    axes[0].set_xlabel("Characters"); axes[0].set_ylabel("Frequency"); axes[0].legend()
    axes[1].set_title("Word Count Distribution", fontweight="bold")
    axes[1].set_xlabel("Words"); axes[1].set_ylabel("Frequency"); axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">Key Metrics</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy",   f"{accuracy*100:.2f}%")
    m2.metric("ROC-AUC",         f"{roc_auc:.4f}")
    m3.metric("Mean CV F1",      f"{cv_f1.mean():.4f}")
    m4.metric("Training Emails", f"{len(X_train):,}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"],
                    annot_kws={"size": 18, "weight": "bold"})
        ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix", fontweight="bold")
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.75, labels[i][j], ha="center", fontsize=9, color="gray")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<p class="section-header">ROC Curve</p>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="steelblue", lw=2.5, label=f"LR (AUC = {roc_auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.08, color="steelblue")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve", fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<p class="section-header">5-Fold Cross-Validation F1 Scores</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(5)
    bars = ax.bar(x, cv_f1, color="steelblue", alpha=0.85, edgecolor="white", linewidth=1.5)
    ax.axhline(cv_f1.mean(), color="crimson", linestyle="--", lw=2, label=f"Mean F1 = {cv_f1.mean():.3f}")
    ax.set_xticks(x); ax.set_xticklabels([f"Fold {i+1}" for i in range(5)])
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1 Score")
    ax.set_title("Cross-Validation F1 per Fold", fontweight="bold")
    ax.legend()
    for bar, score in zip(bars, cv_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<p class="section-header">Classification Report</p>', unsafe_allow_html=True)
    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Keywords
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="section-header">Top Spam & Ham Keywords</p>', unsafe_allow_html=True)

    top_n = st.slider("Number of keywords to show", 10, 30, 20)

    vectorizer    = pipeline.named_steps["tfidf"]
    clf           = pipeline.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef          = clf.coef_[0]

    top_spam_idx = coef.argsort()[-top_n:][::-1]
    top_ham_idx  = coef.argsort()[:top_n]

    spam_words   = feature_names[top_spam_idx]
    spam_weights = coef[top_spam_idx]
    ham_words    = feature_names[top_ham_idx]
    ham_weights  = coef[top_ham_idx]

    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, top_n * 0.32)))

    axes[0].barh(spam_words[::-1], spam_weights[::-1], color="#e74c3c", alpha=0.85, edgecolor="white")
    axes[0].set_title(f"Top {top_n} SPAM Keywords", fontsize=13, fontweight="bold", color="#e74c3c")
    axes[0].set_xlabel("Logistic Regression Weight")
    axes[0].axvline(0, color="black", linewidth=0.8)

    axes[1].barh(ham_words, np.abs(ham_weights), color="#2ecc71", alpha=0.85, edgecolor="white")
    axes[1].set_title(f"Top {top_n} HAM Keywords", fontsize=13, fontweight="bold", color="#27ae60")
    axes[1].set_xlabel("|Logistic Regression Weight|")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        spam_kw_df = pd.DataFrame({"Keyword": spam_words, "Weight": spam_weights.round(4)})
        st.dataframe(spam_kw_df, use_container_width=True, hide_index=True)
    with col2:
        ham_kw_df = pd.DataFrame({"Keyword": ham_words, "Weight": ham_weights.round(4)})
        st.dataframe(ham_kw_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Dataset
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<p class="section-header">Raw Dataset</p>', unsafe_allow_html=True)

    search = st.text_input("🔎 Filter emails by keyword", "")
    display_df = raw_df.copy()
    if search:
        mask = display_df[TEXT_COL].str.contains(search, case=False, na=False)
        display_df = display_df[mask]

    st.write(f"Showing **{len(display_df):,}** of **{len(raw_df):,}** rows")
    st.dataframe(display_df, use_container_width=True, height=450)

    csv_bytes = raw_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Dataset", csv_bytes, "emails.csv", "text/csv")
