def run_hybrid():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    from sklearn.preprocessing import MinMaxScaler
    import warnings

    warnings.filterwarnings('ignore')

    st.title("🤝📱 Ultimate Hybrid Recommendation System")
    st.write("Combine your explicit preferences (search) with your implicit behavior (ratings) for perfect recommendations.")

    # ==============================
    # 1. LOAD & PREP DATA
    # ==============================
    @st.cache_data
    def load_data():
        df = pd.read_csv('smartphones.csv')
        df['Name'] = df['model']
        
        # Fill missing ratings with mean
        df['avg_rating'] = df['avg_rating'].fillna(df['avg_rating'].mean())
        
        # Normalize ratings for the global score component (0 to 1 scale)
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['normalized_avg_rating'] = scaler.fit_transform(df[['avg_rating']])

        # Content-Based Metadata Prep
        for col in ['brand_name', 'os', 'processor_brand']:
            df[col] = df[col].astype(str).str.lower().str.strip()
            
        for col in ['battery_capacity', 'ram_capacity', 'internal_memory', 'refresh_rate']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df['metadata'] = (
            df['brand_name'] + " " +
            df['os'] + " " +
            df['processor_brand'] + " " +
            df['battery_capacity'].astype(str) + " mAh " +
            df['ram_capacity'].astype(str) + " gb " +
            df['internal_memory'].astype(str) + " gb " +
            df['refresh_rate'].astype(str) + " hz"
        )
        return df

    data = load_data()

    # ==============================
    # 2. BUILD MODELS (CF + CB)
    # ==============================
    @st.cache_resource
    def build_models(_df):
        # --- CONTENT-BASED (TF-IDF) ---
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(_df['metadata'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # --- COLLABORATIVE FILTERING (Mock Data Simulation) ---
        num_users = 50
        ratings_list = []
        
        for phone_id in range(len(_df)):
            base = _df.iloc[phone_id]['avg_rating']
            for user in range(num_users):
                if np.random.rand() < 0.3: # 30% sparsity
                    noise = np.random.normal(0, 0.8)
                    rating = np.clip(base + noise, 1, 5)
                    ratings_list.append([user, phone_id, rating])

        ratings = pd.DataFrame(ratings_list, columns=['user_id', 'phone_id', 'rating'])
        user_item_matrix = ratings.pivot_table(index='user_id', columns='phone_id', values='rating').fillna(0)

        cf_model = NearestNeighbors(metric='cosine', algorithm='brute')
        cf_model.fit(user_item_matrix)

        return cosine_sim, cf_model, user_item_matrix

    cosine_sim, cf_model, user_item_matrix = build_models(data)

    # ==============================
    # 3. TABS UI
    # ==============================
    tab1, tab2 = st.tabs(["🎯 Live Recommendations", "📊 System Evaluation"])

    # --- TAB 1: LIVE RECOMMENDATIONS ---
    with tab1:
        st.subheader("Step 1: Target a Specific Phone (Optional)")
        target_phone = st.text_input("🔍 Enter a Smartphone Name to find similar specs (Content-Based):", placeholder="e.g., Apple iPhone 14")
        
        st.divider()
        
        st.subheader("Step 2: Rate Phones to Build Your Profile")
        st.write("Rate these random phones. We will find users with similar tastes (Collaborative).")

        if st.button("🔄 Randomize Rating Phones"):
            st.session_state.sample_phones = data.sample(5)
            for key in list(st.session_state.keys()):
                if key.startswith("h_phone_"):
                    del st.session_state[key]

        if "sample_phones" not in st.session_state:
            st.session_state.sample_phones = data.sample(5)

        sample = st.session_state.sample_phones
        user_ratings = {}

        for idx, row in sample.iterrows():
            rating = st.slider(f"{row['Name']}", 1, 5, 3, key=f"h_phone_{idx}")
            user_ratings[idx] = rating

        # --- HYBRID RECOMMENDATION ENGINE ---
        if st.button("🚀 Get Ultimate Hybrid Recommendations"):
            with st.spinner("Calculating custom hybrid scores..."):
                
                # 1. Create User Vector for Collaborative Filtering
                user_vector = np.zeros(len(data))
                for i, r in user_ratings.items():
                    user_vector[i] = r

                # 2. Collaborative Filtering Scores (Similar Users)
                distances, indices = cf_model.kneighbors([user_vector], n_neighbors=6)
                similar_users = indices.flatten()[1:]
                similarity_weights = 1 - distances.flatten()[1:]

                cf_scores = np.zeros(len(data))
                total_weights = np.zeros(len(data))

                for i, user in enumerate(similar_users):
                    cf_scores += user_item_matrix.iloc[user].values * similarity_weights[i]
                    total_weights += similarity_weights[i]

                cf_scores = cf_scores / (total_weights + 1e-8)
                if cf_scores.max() > 0:
                    cf_scores = cf_scores / cf_scores.max() # Normalize 0-1

                # 3. Content-Based Scores (Text Input OR Liked Phones)
                cb_scores = np.zeros(len(data))
                
                # Check if user entered a specific target phone
                target_matched = False
                if target_phone:
                    phone_match = data[data['Name'].str.lower().str.strip() == target_phone.lower().strip()]
                    if not phone_match.empty:
                        target_idx = phone_match.index[0]
                        cb_scores = cosine_sim[target_idx]
                        target_matched = True
                        st.success(f"✅ Found '{data.iloc[target_idx]['Name']}'! Heavily weighting recommendations towards similar specs.")
                    else:
                        st.warning(f"⚠️ Could not find '{target_phone}' in the database. Falling back to your slider ratings for Content matching.")

                # If no target phone entered OR target phone not found, fallback to slider ratings
                if not target_matched:
                    liked_items = [i for i, r in user_ratings.items() if r >= 4]
                    if liked_items:
                        for item_idx in liked_items:
                            rating_weight = user_ratings[item_idx] / 5.0
                            cb_scores += cosine_sim[item_idx] * rating_weight
                    else:
                        # Fallback if no ratings >= 4
                        for item_idx, r in user_ratings.items():
                            cb_scores += cosine_sim[item_idx] * (r / 5.0)
                            
                if cb_scores.max() > 0:
                    cb_scores = cb_scores / cb_scores.max() # Normalize 0-1

                # 4. Global Popularity Scores
                global_scores = data['normalized_avg_rating'].values

                # 5. Hybrid Weighted Score Calculation
                # If they explicitly searched for a phone, give Content-Based a slightly higher weight
                if target_matched:
                    hybrid_scores = (0.35 * cf_scores) + (0.55 * cb_scores) + (0.10 * global_scores)
                else:
                    hybrid_scores = (0.45 * cf_scores) + (0.45 * cb_scores) + (0.10 * global_scores)

                # Remove already rated items from recommendations
                for i in user_ratings.keys():
                    hybrid_scores[i] = -1
                    
                # Remove the target search phone itself from recommendations
                if target_matched:
                    hybrid_scores[target_idx] = -1

                top_indices = np.argsort(hybrid_scores)[-5:][::-1]

                st.subheader("✨ Top Hybrid Recommendations")
                
                for i in top_indices:
                    name = data.iloc[i]['Name']
                    h_score = hybrid_scores[i]
                    cf_val = cf_scores[i]
                    cb_val = cb_scores[i]
                    rating = data.iloc[i]['avg_rating']
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{name}**")
                        st.caption(f"Specs: {data.iloc[i]['brand_name'].title()} | {data.iloc[i]['ram_capacity']}GB RAM | {data.iloc[i]['battery_capacity']}mAh")
                    with col2:
                        st.markdown(f"🏆 **Score: {h_score:.2f}**")
                    
                    st.progress(cf_val, text=f"Collaborative Match: {cf_val:.2f}")
                    st.progress(cb_val, text=f"Content Match: {cb_val:.2f}")
                    st.divider()

    # --- TAB 2: SYSTEM EVALUATION ---
    with tab2:
        st.subheader("📊 Model Evaluation")
        st.write("Evaluates the current live user's vector across K recommendations.")

        if st.button("Run Hybrid Evaluation (Precision@K)"):
            with st.spinner("Calculating metrics..."):
                k_values = list(range(1, 11))
                precision_list, recall_list, f1_list = [], [], []

                user_vector = np.zeros(len(data))
                for i, r in user_ratings.items():
                    user_vector[i] = r

                relevant_ids = [i for i, r in user_ratings.items() if r >= 4]
                if not relevant_ids:
                    relevant_ids = list(user_ratings.keys()) 

                distances, indices = cf_model.kneighbors([user_vector], n_neighbors=6)
                sim_weights = 1 - distances.flatten()[1:]
                cf_eval = np.zeros(len(data))
                tot_weights = np.zeros(len(data))
                
                for idx, user in enumerate(indices.flatten()[1:]):
                    cf_eval += user_item_matrix.iloc[user].values * sim_weights[idx]
                    tot_weights += sim_weights[idx]
                
                cf_eval = cf_eval / (tot_weights + 1e-8)
                if cf_eval.max() > 0: cf_eval = cf_eval / cf_eval.max()

                cb_eval = np.zeros(len(data))
                for item_idx in relevant_ids:
                    cb_eval += cosine_sim[item_idx]
                if cb_eval.max() > 0: cb_eval = cb_eval / cb_eval.max()

                hybrid_eval = (0.5 * cf_eval) + (0.5 * cb_eval)
                
                for i in user_ratings.keys():
                    hybrid_eval[i] = -1

                ranked_items = np.argsort(hybrid_eval)[::-1]

                for k in k_values:
                    top_k = ranked_items[:k]
                    hits = sum(1 for rec in top_k for rel in relevant_ids if cosine_sim[rec][rel] > 0.6)
                                
                    precision = hits / k if k > 0 else 0
                    recall = hits / len(relevant_ids) if len(relevant_ids) > 0 else 0
                    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_list.append(f1)

                fig, ax = plt.subplots()
                ax.plot(k_values, precision_list, marker='o', label='Precision')
                ax.plot(k_values, recall_list, marker='s', label='Recall')
                ax.plot(k_values, f1_list, marker='^', label='F1-score')

                ax.set_xlabel("K (Top Recommendations)")
                ax.set_ylabel("Score")
                ax.set_title("Hybrid Evaluation Metrics vs K")
                ax.legend()
                st.pyplot(fig)

                st.subheader("📌 Average Evaluation Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Precision", f"{np.mean(precision_list):.4f}")
                col2.metric("Avg Recall", f"{np.mean(recall_list):.4f}")
                col3.metric("Avg F1-score", f"{np.mean(f1_list):.4f}")