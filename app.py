import streamlit as st

st.set_page_config(page_title="Smartphone Recommender System", layout="centered")

st.title("📱 Smartphone Recommendation System")

tab1, tab2, tab3 = st.tabs([
    "🔀 Hybrid",
    "🤝 Collaborative",
    "📊 Content-Based"
])

# HYBRID
with tab1:
    try:
        import Hybrid.streamlit_hybrid
    except Exception as e:
        st.error("Error loading Hybrid module")
        st.exception(e)

# COLLABORATIVE
with tab2:
    try:
        from Collaborative.streamlit_collaborative import run_collaborative
        run_collaborative()
    except Exception as e:
        st.error("Error loading Collaborative module")
        st.exception(e)

# CONTENT
with tab3:
    st.subheader("Content-Based Recommendation")

    try:
        from Content_Based.streamlit_content_based import run_content_based
        run_content_based()
    except Exception as e:
        st.error("Error loading Content-Based module")
        st.exception(e)