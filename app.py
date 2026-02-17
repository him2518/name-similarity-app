import pandas as pd
import streamlit as st

from matcher import match_names


st.set_page_config(page_title="Name Similarity & Confidence Engine", page_icon="🧾", layout="wide")


def confidence_cell_style(value: float) -> str:
    if value >= 85:
        return "background-color: #d1fae5; color: #065f46; font-weight: 600;"
    if value >= 65:
        return "background-color: #fef3c7; color: #92400e; font-weight: 600;"
    return "background-color: #fee2e2; color: #991b1b; font-weight: 600;"


st.title("Name Similarity & Confidence Engine")
st.caption(
    "Compare two names with multiple algorithms to handle spelling variation and transliteration issues."
)

left, right = st.columns(2)
with left:
    name_1 = st.text_input("Name 1", placeholder="e.g. Mohammad Faizan Shaikh")
with right:
    name_2 = st.text_input("Name 2", placeholder="e.g. Mohd Faizan Sheikh")

compare = st.button("Compare Names", type="primary")

if compare:
    if not name_1.strip() or not name_2.strip():
        st.warning("Please enter both names.")
    else:
        result = match_names(name_1, name_2)

        weighted = result["weighted_score_percent"]
        st.subheader("Overall Confidence")
        st.metric("Weighted Match Confidence", f"{weighted}%")
        st.metric("Average Across Algorithms", f"{result['average_score_percent']}%")
        st.progress(min(max(weighted / 100, 0.0), 1.0))
        st.info(f"Decision Hint: {result['decision_hint']}")

        st.subheader("Normalized View")
        n1, n2 = st.columns(2)
        with n1:
            st.write(f"Input 1 normalized: `{result['normalized']['name_1_clean']}`")
            st.write(f"Input 1 phonetic: `{result['normalized']['name_1_phonetic']}`")
        with n2:
            st.write(f"Input 2 normalized: `{result['normalized']['name_2_clean']}`")
            st.write(f"Input 2 phonetic: `{result['normalized']['name_2_phonetic']}`")

        st.subheader("Algorithm-Wise Confidence")
        result_df = pd.DataFrame(
            [
                {
                    "Algorithm": f"{a['name']} ({a['description']})",
                    "Weight (%)": a["weight_percent"],
                    "Confidence (%)": a["score_percent"],
                    "Detailed Matching Explanation (This Case)": a["detailed_explanation"],
                }
                for a in result["algorithms"]
            ]
        )
        styled_df = result_df.style.map(confidence_cell_style, subset=["Confidence (%)"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.caption("Color code: green = high confidence, yellow = moderate confidence, red = low confidence.")
        st.caption("Weighted confidence uses algorithm-specific weightage; average confidence is the simple mean.")

        st.caption(
            "Tip: for financial KYC, combine this score with DOB/ID matching to reduce false approvals."
        )

st.divider()
st.markdown(
    "**Examples to try:** `Rakesh Kumar` vs `Raakesh Kumaar`, `Mohammad Irfan` vs `Mohd Irfan`, `Shreya Nair` vs `Sreya Nayar`"
)
st.markdown("<div style='text-align:center; color:#6b7280;'>Developed by Himanshu</div>", unsafe_allow_html=True)
