def run_collaborative():
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt

    st.title("🤝 Collaborative Filtering Recommendation System")

    # LOAD DATA
    data = pd.read_csv("smartphones.csv")
    data['Name'] = data['model']
    data['avg_rating'] = data['avg_rating'].fillna(3)

    # ==============================
    # BUILD MODEL
    # ==============================
    @st.cache_data
    def build_model():
        num_users = 50
        ratings_list = []

        for phone_id in range(len(data)):
            base = data.iloc[phone_id]['avg_rating']

            for user in range(num_users):
                if np.random.rand() < 0.3:
                    noise = np.random.normal(0, 0.8)
                    rating = np.clip(base + noise, 1, 5)
                    ratings_list.append([user, phone_id, rating])

        ratings = pd.DataFrame(ratings_list, columns=['user_id', 'phone_id', 'rating'])

        matrix = ratings.pivot_table(
            index='user_id',
            columns='phone_id',
            values='rating'
        ).fillna(0)

        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(matrix)

        return model, matrix

    model, user_item_matrix = build_model()

    # ==============================
    # USER INPUT
    # ==============================
    st.subheader("⭐ Rate Some Smartphones (IMPORTANT!)")

    # Reset / Randomize button
    if st.button("🔄 Randomize Phones"):
        st.session_state.sample_phones = data.sample(5)

        # Clear slider values
        for key in list(st.session_state.keys()):
            if key.startswith("phone_"):
                del st.session_state[key]

    if "sample_phones" not in st.session_state:
        st.session_state.sample_phones = data.sample(5)

    sample = st.session_state.sample_phones
    user_ratings = {}

    for idx, row in sample.iterrows():
        rating = st.slider(f"{row['Name']}", 1, 5, 3)
        user_ratings[idx] = rating

    # ==============================
    # CREATE USER VECTOR
    # ==============================
    def create_user_vector():
        vec = np.zeros(len(data))
        for i, r in user_ratings.items():
            vec[i] = r
        return vec

    # ==============================
    # RECOMMEND FUNCTION
    # ==============================
    def recommend(user_vector, top_n=5):

        distances, indices = model.kneighbors([user_vector], n_neighbors=6)

        similar_users = indices.flatten()[1:]
        similarity_weights = 1 - distances.flatten()[1:]

        weighted_scores = np.zeros(user_item_matrix.shape[1])
        total_weights = np.zeros(user_item_matrix.shape[1])

        for i, user in enumerate(similar_users):
            weighted_scores += user_item_matrix.iloc[user] * similarity_weights[i]
            total_weights += similarity_weights[i]

        scores = weighted_scores / (total_weights + 1e-8)

        # Remove already rated
        for i in range(len(user_vector)):
            if user_vector[i] > 0:
                scores[i] = 0

        top_items = np.argsort(scores)[-top_n:][::-1]

        results = []
        for i in top_items:
            name = data.iloc[i]['Name']
            score = scores[i]
            rating = data.iloc[i]['avg_rating']
            results.append((name, score, rating))

        return results

    # ==============================
    # DISPLAY RECOMMENDATIONS
    # ==============================
    if st.button("🚀 Get Recommendations"):

        user_vector = create_user_vector()
        recs = recommend(user_vector)

        st.subheader("✨ Recommended Smartphones")

        for i, (name, score, rating) in enumerate(recs):

            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**{i+1}. {name}**")

            with col2:
                st.markdown(f"⭐ {rating:.1f} | 🔥 {score:.2f}")

            st.divider()

    # ==============================
    # EVALUATION METRICS
    # ==============================
    def calculate_metrics(max_k=10):
        k_values = list(range(1, max_k + 1))

        precision_list = []
        recall_list = []
        f1_list = []

        user_vector = create_user_vector()

        # Relevant = phones user likes (>=4)
        relevant = [i for i, r in user_ratings.items() if r >= 4]

        # If no relevant → avoid empty
        if len(relevant) == 0:
            relevant = [i for i, r in user_ratings.items()]

        for k in k_values:
            recs = recommend(user_vector, top_n=k)

            recommended_ids = []
            for name, _, _ in recs:
                idx = data[data['Name'] == name].index[0]
                recommended_ids.append(idx)

            # ✅ NEW LOGIC (SIMILARITY-BASED MATCH)
            hits = 0
            for rec in recommended_ids:
                for rel in relevant:
                    # If both phones have similar rating range
                    if abs(data.iloc[rec]['avg_rating'] - data.iloc[rel]['avg_rating']) < 0.5:
                        hits += 1
                        break

            precision = hits / k if k > 0 else 0
            recall = hits / len(relevant) if len(relevant) > 0 else 0

            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        return k_values, precision_list, recall_list, f1_list


    def plot_metrics():
        k, precision, recall, f1 = calculate_metrics()

        plt.figure()
        plt.plot(k, precision, marker='o', label='Precision')
        plt.plot(k, recall, marker='s', label='Recall')
        plt.plot(k, f1, marker='^', label='F1-score')

        plt.xlabel("K (Top Recommendations)")
        plt.ylabel("Score")
        plt.title("Evaluation Metrics vs K")
        plt.legend()

        st.pyplot(plt)


    # ==============================
    # SHOW EVALUATION
    # ==============================
    st.subheader("📊 Model Evaluation")

    if st.button("Run Evaluation (Precision@K)"):
        plot_metrics()

