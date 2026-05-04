import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


st.set_page_config(
    page_title="Football Analytics Dashboard",
    page_icon="⚽",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv("epl_week4_model_ready.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("best_rf_model.pkl")
    return model

df = load_data()
model = load_model()


st.title("⚽ Football Analytics & Match Prediction Dashboard")
st.markdown(
    """
    This dashboard presents a machine learning-based football analytics system
    using engineered features such as recent form, expected goals, and shot accuracy.
    It supports team-level analysis, match outcome prediction, and model interpretability.
    """
)


st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Overview", "Dataset Visualisations", "Team Analysis", "Match Prediction", "Model Insights"]
)


if section == "Overview":
    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches", len(df))
    col2.metric("Total Teams", df["HomeTeam"].nunique())
    col3.metric("Available Features", len(df.columns))

    st.subheader("Dataset Sample")
    st.dataframe(df.head(25))

    st.subheader("What this system does")
    st.write("""
    - Analyses football performance using engineered metrics
    - Predicts match outcomes using machine learning
    - Compares team-level attacking and form indicators
    - Interprets model behaviour through explainability outputs
    """)


elif section == "Dataset Visualisations":
    st.header("Dataset Visualisations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Match Result Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        df["FTR"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Full Time Result Distribution")
        ax.set_xlabel("Result")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.subheader("Goals Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["FTHG"], bins=10, alpha=0.7, label="Home Goals")
        ax.hist(df["FTAG"], bins=10, alpha=0.7, label="Away Goals")
        ax.set_title("Goals Distribution")
        ax.legend()
        st.pyplot(fig)

    st.subheader("xG vs Goals (Home)")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["xG_home"], df["FTHG"], alpha=0.6)
    ax.set_xlabel("Home xG")
    ax.set_ylabel("Home Goals")
    ax.set_title("Home xG vs Actual Goals")
    st.pyplot(fig)

    st.subheader("Top Teams by Average Home xG")
    avg_home_xg = df.groupby("HomeTeam")["xG_home"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_home_xg.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Average Home xG")
    st.pyplot(fig)


elif section == "Team Analysis":
    st.header("Team Analysis")

    teams = sorted(df["HomeTeam"].unique())
    team = st.selectbox("Select Team", teams)

    home_matches = df[df["HomeTeam"] == team]
    away_matches = df[df["AwayTeam"] == team]

    avg_home_xg = home_matches["xG_home"].mean()
    avg_away_xg = away_matches["xG_away"].mean()
    home_win_rate = (home_matches["FTR"] == "H").mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Home xG", f"{avg_home_xg:.2f}")
    col2.metric("Avg Away xG", f"{avg_away_xg:.2f}")
    col3.metric("Home Win Rate", f"{home_win_rate:.2%}")

    st.subheader("Recent Home xG Trend")
    if "home_xg_last5" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        home_matches["home_xg_last5"].reset_index(drop=True).plot(ax=ax)
        ax.set_title(f"{team} - Home xG (Last 5 Matches Feature)")
        ax.set_ylabel("xG")
        st.pyplot(fig)

    st.subheader("Goals Scored at Home")
    fig, ax = plt.subplots(figsize=(8, 4))
    home_matches["FTHG"].reset_index(drop=True).plot(kind="bar", ax=ax)
    ax.set_title(f"{team} - Home Goals by Match")
    ax.set_ylabel("Goals")
    st.pyplot(fig)


elif section == "Match Prediction":
    st.header("Match Outcome Prediction")

    teams = sorted(df["HomeTeam"].unique())
    home_team = st.selectbox("Select Home Team", teams)
    away_team = st.selectbox("Select Away Team", teams)

    if home_team == away_team:
        st.warning("Please choose two different teams.")
    else:
        if st.button("Predict Match Outcome"):
            home_stats = df[df["HomeTeam"] == home_team].mean(numeric_only=True)
            away_stats = df[df["AwayTeam"] == away_team].mean(numeric_only=True)

            input_data = pd.DataFrame({
                "home_form_last5": [home_stats.get("home_form_last5", 0)],
                "away_form_last5": [away_stats.get("away_form_last5", 0)],
                "form_diff_last5": [home_stats.get("home_form_last5", 0) - away_stats.get("away_form_last5", 0)],
                "home_goals_for_last5": [home_stats.get("home_goals_for_last5", 0)],
                "away_goals_for_last5": [away_stats.get("away_goals_for_last5", 0)],
                "goals_for_diff_last5": [home_stats.get("home_goals_for_last5", 0) - away_stats.get("away_goals_for_last5", 0)],
                "home_goals_against_last5": [home_stats.get("home_goals_against_last5", 0)],
                "away_goals_against_last5": [away_stats.get("away_goals_against_last5", 0)],
                "goals_against_diff_last5": [home_stats.get("home_goals_against_last5", 0) - away_stats.get("away_goals_against_last5", 0)],
                "home_xg_last5": [home_stats.get("home_xg_last5", 0)],
                "away_xg_last5": [away_stats.get("away_xg_last5", 0)],
                "xg_diff_last5": [home_stats.get("home_xg_last5", 0) - away_stats.get("away_xg_last5", 0)],
                "home_shot_acc_last5": [home_stats.get("home_shot_acc_last5", 0)],
                "away_shot_acc_last5": [away_stats.get("away_shot_acc_last5", 0)],
                "shot_acc_diff_last5": [home_stats.get("home_shot_acc_last5", 0) - away_stats.get("away_shot_acc_last5", 0)]
            })

            prediction = model.predict(input_data)[0]
            probs = model.predict_proba(input_data)[0]

            label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
            st.success(f"Predicted Outcome: {label_map.get(prediction, prediction)}")

            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                "Outcome": ["Away Win", "Draw", "Home Win"],
                "Probability": probs
            })
            st.dataframe(prob_df)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(prob_df["Outcome"], prob_df["Probability"])
            ax.set_ylim(0, 1)
            ax.set_title("Prediction Probability")
            st.pyplot(fig)


elif section == "Model Insights":
    st.header("Model Insights and Explainability")

    st.subheader("Random Forest Feature Importance")
    features_week4 = [
        "home_form_last5",
        "away_form_last5",
        "form_diff_last5",
        "home_goals_for_last5",
        "away_goals_for_last5",
        "goals_for_diff_last5",
        "home_goals_against_last5",
        "away_goals_against_last5",
        "goals_against_diff_last5",
        "home_xg_last5",
        "away_xg_last5",
        "xg_diff_last5",
        "home_shot_acc_last5",
        "away_shot_acc_last5",
        "shot_acc_diff_last5"
    ]

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=features_week4).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    feat_imp.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    st.subheader("Saved SHAP Feature Importance Plot")
    st.image("shap_home_win_bar.png", caption="SHAP Feature Importance - Home Win")

    st.subheader("Saved SHAP Summary Plot")
    st.image("shap_home_win_summary.png", caption="SHAP Summary Plot - Home Win")