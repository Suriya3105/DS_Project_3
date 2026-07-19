import pandas as pd
import streamlit as st

# ===============================
# Load Dataset
# ===============================

df_clean = pd.read_csv("HR_Job_Placement_Clean.csv")

# ===============================
# Feature Engineering
# ===============================

# Experience Category
df_clean["experience_category"] = pd.cut(
    df_clean["years_of_experience"],
    bins=[-1, 0, 2, 5],
    labels=["Fresher", "Junior", "Senior"]
)

# Average Academic Score
df_clean["academic_average"] = (
    df_clean["ssc_percentage"] +
    df_clean["hsc_percentage"] +
    df_clean["degree_percentage"]
) / 3

# Academic Performance Band
df_clean["academic_band"] = pd.cut(
    df_clean["academic_average"],
    bins=[0, 60, 75, 100],
    labels=["Low", "Medium", "High"]
)

# Skills Match Level
df_clean["skills_match_level"] = pd.cut(
    df_clean["skills_match_percentage"],
    bins=[0, 60, 80, 100],
    labels=["Low", "Medium", "High"]
)

# Interview Performance Category
df_clean["interview_performance"] = pd.cut(
    df_clean["technical_score"],
    bins=[0, 60, 80, 100],
    labels=["Poor", "Average", "Excellent"]
)

# Placement Probability Score
df_clean["placement_probability_score"] = (
    df_clean["technical_score"] +
    df_clean["aptitude_score"] +
    df_clean["communication_score"] +
    df_clean["skills_match_percentage"]
) / 4

# ===============================
# Dashboard KPIs
# ===============================

total_candidates = len(df_clean)

placed_candidates = (df_clean["status"] == "Placed").sum()

placement_rate = (placed_candidates / total_candidates) * 100

job_acceptance_rate = placement_rate

average_interview_score = df_clean["technical_score"].mean()

average_skills_match = df_clean["skills_match_percentage"].mean()

offer_dropout_rate = 0

high_risk_candidates = df_clean[
    (df_clean["technical_score"] < 60) |
    (df_clean["skills_match_percentage"] < 60)
]

high_risk_percentage = (
    len(high_risk_candidates) / total_candidates
) * 100

st.title("🎓 Placement Prediction Analytics Dashboard")

st.subheader("Placement KPI Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Total Candidates", f"{total_candidates:,}")
col2.metric("Placement Rate (%)", f"{placement_rate:.2f}")
col3.metric("Job Acceptance Rate (%)", f"{job_acceptance_rate:.2f}")

col4, col5, col6 = st.columns(3)

col4.metric("Average Interview Score", f"{average_interview_score:.2f}")
col5.metric("Average Skills Match (%)", f"{average_skills_match:.2f}")
col6.metric("High Risk Candidate (%)", f"{high_risk_percentage:.2f}")

st.metric("Offer Dropout Rate (%)", f"{offer_dropout_rate:.2f}")

st.header("Feature Engineering Analytics")

st.subheader("Experience Category")

st.bar_chart(
    df_clean["experience_category"].value_counts().sort_index()
)

st.subheader("Academic Performance Band")

st.bar_chart(
    df_clean["academic_band"].value_counts().sort_index()
)

st.subheader("Skills Match Level")

st.bar_chart(
    df_clean["skills_match_level"].value_counts().sort_index()
)

st.subheader("Interview Performance")

st.bar_chart(
    df_clean["interview_performance"].value_counts().sort_index()
)

st.subheader("Placement Probability Score Distribution")

st.line_chart(
    df_clean["placement_probability_score"]
)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,4))

ax.hist(
    df_clean["placement_probability_score"],
    bins=20
)

ax.set_title("Placement Probability Score Distribution")
ax.set_xlabel("Probability Score")
ax.set_ylabel("Candidates")

st.pyplot(fig)

st.subheader("Derived Feature Summary")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Average Academic Score",
    f"{df_clean['academic_average'].mean():.2f}"
)

c2.metric(
    "Average Placement Probability",
    f"{df_clean['placement_probability_score'].mean():.2f}"
)

c3.metric(
    "Experienced Candidates",
    f"{(df_clean['years_of_experience'] > 0).sum():,}"
)

