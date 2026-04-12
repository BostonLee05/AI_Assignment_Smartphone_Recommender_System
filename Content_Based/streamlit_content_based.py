def run_content_based():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    st.title("📱 Smartphone Recommendation System")

    # ==============================
    # LOAD DATA
    # ==============================
    @st.cache_data
    def load_data():
        df = pd.read_csv('smartphones.csv')
        df = df.fillna('')

        df['model'] = df['model'].str.lower().str.strip()
        df['brand_name'] = df['brand_name'].str.lower().str.strip()
        df['os'] = df['os'].str.lower().str.strip()
        df['processor_brand'] = df['processor_brand'].str.lower().str.strip()
        df['battery_capacity'] = pd.to_numeric(df['battery_capacity'], errors='coerce').fillna(0)
        df['ram_capacity'] = pd.to_numeric(df['ram_capacity'], errors='coerce').fillna(0)
        df['internal_memory'] = pd.to_numeric(df['internal_memory'], errors='coerce').fillna(0)
        df['refresh_rate'] = pd.to_numeric(df['refresh_rate'], errors='coerce').fillna(0)

        df['metadata'] = (
                df['brand_name'] + " " +
                df['os'] + " " +
                df['processor_brand'] + " " +
                df['battery_capacity'].astype(str) + " mAh " +
                df['ram_capacity'].astype(str) + " GB " +
                df['internal_memory'].astype(str) + " GB " +
                df['refresh_rate'].astype(str) + " Hz"
        )

        return df


    df = load_data()


    # ==============================
    # TF-IDF + SIMILARITY
    # ==============================
    @st.cache_resource
    def compute_similarity(data):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data['metadata'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        return cosine_sim


    cosine_sim = compute_similarity(df)
    indices = pd.Series(df.index, index=df['model']).drop_duplicates()


    # ==============================
    # RECOMMENDATION FUNCTION
    # ==============================
    def get_recommendations(model, cosine_sim=cosine_sim):
        model = model.lower().strip()

        if model not in indices:
            return None

        idx = indices[model]

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]

        smart_indices = [i[0] for i in sim_scores]

        return df.iloc[smart_indices][['model', 'brand_name', 'os']]


    # ==============================
    # EVALUATION FUNCTION
    # ==============================
    def evaluate_system(df, recommendations_func, k=10, sample_size=50):
        sample_size = min(sample_size, len(df))
        test_samples = df.sample(sample_size, random_state=42)

        precision_scores = []
        recall_scores = []
        f1_scores = []

        for _, row in test_samples.iterrows():
            target_name = row['model']
            target_brand = row['brand_name']

            recommendations = recommendations_func(target_name)

            if recommendations is None:
                continue

            rec_top_k = recommendations.head(k)

            # ==============================
            # TRUE RELEVANT ITEMS (GROUND TRUTH)
            # ==============================
            total_relevant = df[df['brand_name'] == target_brand].shape[0] - 1  # exclude itself

            if total_relevant == 0:
                continue

            # ==============================
            # RECOMMENDED RELEVANT ITEMS
            # ==============================
            relevant_count = rec_top_k[
                rec_top_k['brand_name'] == target_brand
                ].shape[0]

            # ==============================
            # METRICS
            # ==============================
            precision = relevant_count / k
            recall = relevant_count / total_relevant

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return {
            "precision": np.mean(precision_scores) if precision_scores else 0,
            "recall": np.mean(recall_scores) if recall_scores else 0,
            "f1": np.mean(f1_scores) if f1_scores else 0
        }


    # ==============================
    # STREAMLIT UI
    # ==============================
    st.title("📱 Smartphone Recommendation System")

    user_input = st.text_input("Enter Smartphone Name:")

    if st.button("Get Recommendations"):
        results = get_recommendations(user_input)

        if results is None:
            st.error("Smartphone not found in dataset.")
        else:
            st.success(f"Recommendations for '{user_input}':")
            st.dataframe(results)

    st.subheader("📊 Evaluation Metrics vs K")

    if st.button("Generate Evaluation Chart"):
        k_values = list(range(1, 11))

        precision_scores = []
        recall_scores = []
        f1_scores = []

        for k in k_values:
            scores = evaluate_system(df, get_recommendations, k=k)

            precision_scores.append(scores['precision'])
            recall_scores.append(scores['recall'])
            f1_scores.append(scores['f1'])

        # ==============================
        # PLOT
        # ==============================
        fig, ax = plt.subplots()

        ax.plot(k_values, precision_scores, marker='o', label='Precision')
        ax.plot(k_values, recall_scores, marker='s', label='Recall')
        ax.plot(k_values, f1_scores, marker='^', label='F1-score')

        ax.set_xlabel("K (Top Recommendations)")
        ax.set_ylabel("Score")
        ax.set_title("Evaluation Metrics vs K")
        ax.legend()

        st.pyplot(fig)

        # ==============================
        # TABLE
        # ==============================
        result_df = pd.DataFrame({
            'K': k_values,
            'Precision': precision_scores,
            'Recall': recall_scores,
            'F1-score': f1_scores
        })

        st.dataframe(result_df)

        # ==============================
        # AVERAGE METRICS
        # ==============================
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)

        st.subheader("📌 Average Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Avg Precision", f"{avg_precision:.4f}")
        col2.metric("Avg Recall", f"{avg_recall:.4f}")
        col3.metric("Avg F1-score", f"{avg_f1:.4f}")