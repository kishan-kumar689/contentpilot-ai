
import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

# Utility: clean column names

import unicodedata
import re as _re

def clean_col(col):
  
    col = unicodedata.normalize("NFKD", col)
    col = col.encode("ascii", "ignore").decode("ascii")

 
    col = col.replace("\u00a0", " ")

    
    col = col.strip().lower()

    
    col = col.replace("%", " pct")
    col = col.replace("(", "").replace(")", "")
    col = col.replace("/", " per ")

   
    col = _re.sub(r"\s+", "_", col)

    return col



# Preprocess YouTube dataset

def preprocess_youtube_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

   
    df.columns = [clean_col(c) for c in df.columns]

  
    if "date" not in df.columns:
        raise ValueError("Could not find a 'Date' column in the uploaded file.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()

   
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek  # 0=Mon..6=Sun
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    
    percent_cols = [
        "impressions_click_through_rate_pct",
        "likes_vs_dislikes_pct",
        "average_percentage_viewed_pct",
    ]
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("%", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    
    num_cols = [
        "average_views_per_viewer",
        "unique_viewers",
        "impressions",
        "comments_added",
        "shares",
        "dislikes",
        "subscribers_lost",
        "subscribers_gained",
        "likes",
        "videos_published",
        "videos_added",
        "subscribers",
        "views",
        "watch_time_hours",
        "average_view_duration",
        "your_estimated_revenue_usd",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df



# Train daily-views model

def train_views_model(df: pd.DataFrame):
    if "views" not in df.columns:
        raise ValueError("Could not find 'views' column after cleaning.")

    data = df.dropna(subset=["views"]).copy()

    all_feature_candidates = [
        "impressions",
        "impressions_click_through_rate_pct",
        "videos_published",
        "average_views_per_viewer",
        "average_percentage_viewed_pct",
        "watch_time_hours",
        "subscribers",
        "dayofweek",
        "month",
        "is_weekend",
    ]
    feature_cols = [c for c in all_feature_candidates if c in data.columns]

    X = data[feature_cols].fillna(0)
    y = data["views"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

    return model, feature_cols, metrics, data



# Weekly strategy helpers

def suggest_youtube_posting_days(df, n_days=3):
    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    views_by_dow = df.groupby("dayofweek")["views"].mean().sort_values(ascending=False)
    top_days_idx = views_by_dow.head(n_days).index.tolist()
    top_days = [dow_map[i] for i in top_days_idx]
    return views_by_dow, top_days


def build_weekly_content_plan(best_days, videos_per_week=4):
    all_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plan = {day: 0 for day in all_days}
    i = 0
    for _ in range(videos_per_week):
        day = best_days[i % len(best_days)]
        plan[day] += 1
        i += 1
    return plan



# Title scoring + A/B optimizer

power_phrases = [
    "how to", "guide", "tips", "mistakes", "secrets",
    "strategy", "step by step", "tutorial", "for beginners"
]

def score_title(title, main_keyword=None):
    t = str(title).strip()
    t_lower = t.lower()
    score = 0

    # length
    length = len(t)
    if 40 <= length <= 70:
        score += 2
    elif 25 <= length < 40 or 70 < length <= 90:
        score += 1

    # number
    if re.search(r"\d", t):
        score += 1

    # power phrase
    if any(phrase in t_lower for phrase in power_phrases):
        score += 1

    # main keyword
    if main_keyword and main_keyword.lower() in t_lower:
        score += 2

    return score
    
#Step 1 – Add a helper function


def explain_title(title, main_keyword=None):
    """Return human-readable reasons for a title's score."""
    reasons = []
    t = str(title).strip()
    t_lower = t.lower()
    length = len(t)

   
    if 40 <= length <= 70:
        reasons.append("Length is in the ideal range (~40–70 characters).")
    elif 25 <= length < 40 or 70 < length <= 90:
        reasons.append("Length is acceptable but could be closer to 40–70 characters.")
    else:
        reasons.append("Length is outside the ideal range; consider adjusting it.")

   
    if re.search(r"\d", t):
        reasons.append("Contains a number, which usually attracts more clicks.")

    
    used_phrases = [phrase for phrase in power_phrases if phrase in t_lower]
    if used_phrases:
        reasons.append(
            "Uses strong phrases like " + ", ".join(f'"{p}"' for p in used_phrases) + "."
        )
    else:
        reasons.append("Does not use any strong 'how to / tips / guide' type phrase.")

    
    if main_keyword and main_keyword.lower() in t_lower:
        reasons.append(f"Includes the main keyword: '{main_keyword}'.")
    else:
        if main_keyword:
            reasons.append(f"Does not include the main keyword '{main_keyword}'.")
        else:
            reasons.append("No main keyword was specified for scoring.")

    return reasons
    

def compare_titles(title_a, title_b, main_keyword=None):
    score_a = score_title(title_a, main_keyword)
    score_b = score_title(title_b, main_keyword)

    if score_a > score_b:
        recommendation = "A"
    elif score_b > score_a:
        recommendation = "B"
    else:
        recommendation = "Tie"

    return {
        "title_a": {"title": title_a, "score": score_a},
        "title_b": {"title": title_b, "score": score_b},
        "recommendation": recommendation,
    }



# STREAMLIT APP

st.set_page_config(
    page_title="ContentPilot AI – YouTube Creator Agent",
    layout="wide"
)

st.title("🎬 ContentPilot AI – YouTube Creator Agent")
st.write(
    "Upload your daily YouTube analytics export and let the agent learn "
    "what drives your views, suggest posting days, and score your titles."
)


st.sidebar.header("1️⃣ Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload YouTube daily analytics CSV",
    type=["csv"],
    help="Export from YouTube Analytics as CSV (daily metrics)."
)

st.sidebar.header("2️⃣ Agent Settings")
target_videos_per_week = st.sidebar.slider(
    "Target videos per week", min_value=1, max_value=10, value=4
)
main_keyword = st.sidebar.text_input(
    "Main keyword / niche (for title scoring):",
    value="youtube"
)

st.sidebar.header("3️⃣ Candidate Titles (Optional)")
title_a_input = st.sidebar.text_input(
    "Title A", value="5 Mistakes New YouTubers Make (Avoid These!)"
)
title_b_input = st.sidebar.text_input(
    "Title B", value="YouTube Growth Guide for Beginners"
)

if uploaded_file is None:
    st.info("👆 Upload a CSV in the sidebar to get started.")
    st.stop()

# Load and preprocess
try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

try:
    yt = preprocess_youtube_df(df_raw)
except Exception as e:
    st.error(f"Error preprocessing data: {e}")
    st.stop()

st.success("✅ Data loaded and preprocessed successfully.")


tab1, tab2, tab3 = st.tabs(["📊 Model & Drivers", "🗓 Strategy Planner", "🧠 Title Optimizer"])

# ==============
# Tab 1: Model & Drivers
# ==============
with tab1:
    st.subheader("Daily Views Model – Performance & Drivers")

    st.write("Sample of cleaned data:")
    st.dataframe(yt.head())

    try:
        model, feature_cols, metrics, data = train_views_model(yt)
    except Exception as e:
        st.error(f"Error training model: {e}")
        st.stop()

    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("MAE (views)", f"{metrics['MAE']:,.0f}")
    col_metric2.metric("RMSE (views)", f"{metrics['RMSE']:,.0f}")
    col_metric3.metric("R²", f"{metrics['R2']:.3f}")

    st.markdown(
    f"""
    On the test data, the model explains about **{metrics['R2']*100:.1f}%** of the variation 
    in daily views. On average, it is off by roughly **{metrics['MAE']:,.0f} views per day**, 
    which is reasonable for the scale of this channel.
    """
)

    st.markdown("**Features used in model:**")
    st.write(feature_cols)

    # Feature importance
    st.markdown("### Feature Importance – What Drives Views?")
    feat_imp = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    st.write(feat_imp.to_frame("importance"))

    fig, ax = plt.subplots(figsize=(6, 4))
    feat_imp.plot(kind="bar", ax=ax)
    ax.set_title("Feature Importance – Daily Views Model")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "The higher the importance, the more that feature contributes to predicting daily views. "
        "This explains *why* the channel performs the way it does."
    )



# ==============
# Tab 2: Strategy Planner
# ==============
with tab2:
    st.subheader("Weekly Strategy & Content Calendar")

    try:
        views_by_dow, best_days = suggest_youtube_posting_days(data)
    except Exception as e:
        st.error(f"Error computing views by day-of-week: {e}")
        st.stop()

    st.markdown("**Average views by day-of-week:**")
    st.write(views_by_dow)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    views_by_dow.plot(kind="bar", ax=ax2)
    ax2.set_title("Average Views by Day of Week (0=Mon)")
    ax2.set_ylabel("Average Views")
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown(f"**Suggested best posting days:** {', '.join(best_days)}")

   
dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

best_day_idx = views_by_dow.idxmax()
worst_day_idx = views_by_dow.idxmin()

best_day_name = dow_map.get(best_day_idx, str(best_day_idx))
worst_day_name = dow_map.get(worst_day_idx, str(worst_day_idx))

best_avg = views_by_dow.max()
worst_avg = views_by_dow.min()

st.markdown(
    f"The **strongest day** is **{best_day_name}** with around **{best_avg:,.0f}** "
    f"average views, while the **weakest day** is **{worst_day_name}** "
    f"with about **{worst_avg:,.0f}** views."
)


if worst_avg > 0:
    uplift_pct = (best_avg - worst_avg) / worst_avg * 100
    st.markdown(
        f"Focusing more uploads on the strongest days instead of the weakest ones "
        f"can improve average daily views by roughly **{uplift_pct:.1f}%**."
    )

    weekly_plan = build_weekly_content_plan(best_days, videos_per_week=target_videos_per_week)
    st.markdown(f"**Weekly content plan for {target_videos_per_week} videos/week:**")
    st.write(pd.DataFrame.from_dict(weekly_plan, orient="index", columns=["videos_per_day"]))

    st.caption(
        "Strategy: Focus your uploads on the historically strongest days, and distribute your "
        f"{target_videos_per_week} videos across them for maximum impact."
    )


# ==============
# Tab 3: Title Optimizer
# ==============
with tab3:
    st.subheader("Title Scoring & A/B Comparison (Prototype)")

    st.write(
        "This is a simple heuristic title-scoring module. It rewards titles that:\n"
        "- Have a good length (~40–70 characters)\n"
        "- Use numbers\n"
        "- Include power phrases like 'how to', 'guide', 'tips'\n"
        "- Contain your main keyword"
    )

    if title_a_input or title_b_input:
        res = compare_titles(title_a_input, title_b_input, main_keyword=main_keyword)

        st.markdown("**Scores:**")
        st.write(pd.DataFrame({
            "Title": [res["title_a"]["title"], res["title_b"]["title"]],
            "Score": [res["title_a"]["score"], res["title_b"]["score"]],
            "Label": ["A", "B"]
        }))

        if res["recommendation"] == "Tie":
            st.info("Both titles are scored equally. Try tweaking one of them (add a number / power word / keyword).")
        else:
            st.success(f"Recommended: **Title {res['recommendation']}** based on heuristic score.")
            
         
    st.markdown("### Why each title got its score")

    reasons_a = explain_title(title_a_input, main_keyword=main_keyword)
    reasons_b = explain_title(title_b_input, main_keyword=main_keyword)

    st.markdown("**Title A analysis:**")
    for r in reasons_a:
        st.write("- " + r)

    st.markdown("**Title B analysis:**")
    for r in reasons_b:
        st.write("- " + r)


    st.caption(
        "In future, this rule-based title scorer can be replaced with a model trained on real title + CTR data, "
        "or combined with an LLM to actually generate and iterate titles."
    )

st.markdown("---")
st.caption(
    "Future Work: Extend this agent with a podcast completion model and LLM-based post assets "
    "(titles, hooks, descriptions, thumbnail text), making ContentPilot AI a multi-format creator assistant."
)
