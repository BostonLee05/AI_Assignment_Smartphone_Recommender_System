import streamlit as st

st.set_page_config(page_title="Smartphone Recommender System", layout="centered")

st.title("📱 Smartphone Recommendation System")

# Tabs (Hybrid FIRST)
tab1, tab2, tab3 = st.tabs([
    "🔀 Hybrid",
    "🤝 Collaborative Filtering",
    "📊 Content-Based"
])

# ==============================
# TAB 1 (HYBRID)
# ==============================
with tab1:
    st.subheader("Hybrid Recommendation System")

    try:
        import Hybrid.hybrid_recommendation
    except Exception as e:
        st.error("Error loading Hybrid module")
        st.exception(e)

# ==============================
# TAB 2 (COLLABORATIVE)
# ==============================
with tab2:
    st.subheader("Collaborative Filtering Recommendation")

    try:
        from Collaborative.streamlit_collaborative import run_collaborative
        run_collaborative()
    except Exception as e:
        st.error("Error loading Collaborative module.")
        st.exception(e)

# ==============================
# TAB 3 (CONTENT BASED)
# ==============================
with tab3:
    st.subheader("📊 Content-Based Recommendation")

    try:
        import Content_Based.streamlit_content_based
    except Exception as e:
        st.error("Error loading Content-Based module")
        st.exception(e)