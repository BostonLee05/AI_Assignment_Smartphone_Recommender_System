def run_hybrid():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.feature_extraction.text import TfidfVectorizer
    from collections import defaultdict
    import warnings

    warnings.filterwarnings('ignore')

    # --------------------------------------------------------
    # BACKEND LOGIC: DATA PREP & HYBRID MODELS
    # --------------------------------------------------------

    @st.cache_data
    def load_data():
        df = pd.read_csv('smartphones.csv')
        df = df.drop_duplicates(subset=['model']).reset_index(drop=True)
        
        df['avg_rating'] = df['avg_rating'].fillna(df['avg_rating'].mean())
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('')
            else:
                df[col] = df[col].fillna(0)
                
        scaler = MinMaxScaler(feature_range=(0, 5))
        df['normalized_avg_rating'] = scaler.fit_transform(df[['avg_rating']])
        
        df['content_features'] = (
            df['brand_name'].astype(str) + ' ' + 
            df['os'].astype(str) + ' ' + 
            df['processor_brand'].astype(str) + ' ' +
            df['ram_capacity'].astype(str) + 'gb ' + 
            df['internal_memory'].astype(str) + 'gb'
        ).fillna('').str.lower()
        
        return df

    df_items = load_data()

    # Content-Based Matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_items['content_features'])
    content_sim_matrix = cosine_similarity(tfidf_matrix)
    content_sim_df = pd.DataFrame(content_sim_matrix, index=df_items['model'], columns=df_items['model'])

    # Collaborative Filtering Mock Data
    np.random.seed(42)
    sample_phones = df_items['model'].sample(150, random_state=42).tolist()
    mock_ratings = []

    for u_id in range(1, 51):
        n = np.random.randint(5, 20) 
        for p in np.random.choice(sample_phones, n, replace=False):
            mock_ratings.append({'user_id': u_id, 'model': p, 'rating': float(np.random.randint(1, 6))})
            
    df_ratings = pd.DataFrame(mock_ratings)

    user_item_matrix = df_ratings.pivot_table(index='user_id', columns='model', values='rating').fillna(0)
    user_sim_df = pd.DataFrame(cosine_similarity(user_item_matrix), index=user_item_matrix.index, columns=user_item_matrix.index)

    # --- HISTORICAL PREDICTION FUNCTIONS (Used for Tab 2 Evaluation) ---
    def predict_cb(u_id, item):
        u_ratings = df_ratings[df_ratings['user_id'] == u_id]
        num, den = 0, 0
        for _, row in u_ratings.iterrows():
            sim = content_sim_df.loc[item, row['model']]
            num += sim * row['rating']
            den += sim
        return num / den if den > 0 else 3.0

    def predict_cf(u_id, item):
        if item not in user_item_matrix.columns: 
            return 3.0
        sim_users = user_sim_df[u_id].drop(u_id)
        item_rats = user_item_matrix[item].drop(u_id)
        
        mask = item_rats > 0 
        if not mask.any(): 
            return 3.0
        return np.dot(sim_users[mask], item_rats[mask]) / sim_users[mask].sum()

    # --- LIVE PREDICTION FUNCTIONS (Used for Tab 1 User Input) ---
    def predict_live_cb(live_ratings_dict, target_phone, item):
        # If user explicitly searched a phone, calculate similarity based heavily on that phone
        if target_phone and target_phone in content_sim_df.index:
            return content_sim_df.loc[item, target_phone] * 5.0 # Max out the rating multiplier
            
        # Otherwise, calculate based on slider ratings
        num, den = 0, 0
        for m, r in live_ratings_dict.items():
            if m in content_sim_df.index:
                sim = content_sim_df.loc[item, m]
                num += sim * r
                den += sim
        return num / den if den > 0 else 3.0

    def predict_live_cf(live_ratings_dict, item):
        if item not in user_item_matrix.columns: 
            return 3.0
            
        # Build the live user vector to compare against mock users
        live_vec = np.zeros(len(user_item_matrix.columns))
        cols = list(user_item_matrix.columns)
        for m, r in live_ratings_dict.items():
            if m in cols:
                live_vec[cols.index(m)] = r
                
        # Calculate cosine similarity between live user and all mock users
        sims = cosine_similarity([live_vec], user_item_matrix.values)[0]
        sim_users = pd.Series(sims, index=user_item_matrix.index)
        
        item_rats = user_item_matrix[item]
        mask = item_rats > 0 
        if not mask.any(): 
            return 3.0
            
        den = sim_users[mask].sum()
        return np.dot(sim_users[mask], item_rats[mask]) / den if den > 0 else 3.0

    # --------------------------------------------------------
    # FRONTEND LOGIC: STREAMLIT UI & DASHBOARD
    # --------------------------------------------------------

    st.subheader("Hybrid Recommendations")

    tab1, tab2 = st.tabs(["Live Recommendations", "System Evaluation"])

    # Tab 1: Show recommendations for the LIVE user
    with tab1:
        st.markdown("### Step 1: Target a Specific Phone (Content-Based)")
        # User input for Content-Based
        target_phone_input = st.text_input("Enter a Smartphone Name to find similar specs:", placeholder="e.g., Apple iPhone 14")
        
        st.divider()
        
        st.markdown("### Step 2: Rate Phones (Collaborative Filtering)")
        if st.button("🔄 Randomize Phones"):
            st.session_state.live_sample = df_items.sample(5)
            for key in list(st.session_state.keys()):
                if key.startswith("live_rating_"):
                    del st.session_state[key]

        if "live_sample" not in st.session_state:
            st.session_state.live_sample = df_items.sample(5)

        live_ratings = {}
        for idx, row in st.session_state.live_sample.iterrows():
            model_name = row['model']
            rating = st.slider(f"Rate {model_name}", 1, 5, 3, key=f"live_rating_{idx}")
            live_ratings[model_name] = rating
        
        if st.button("🚀 Get Hybrid Recommendations"):
            with st.spinner("Calculating custom hybrid scores..."):
                all_models = df_items['model'].unique()
                unrated = [m for m in all_models if m not in live_ratings.keys() and m != target_phone_input]
                
                # Check if target phone exists in dataset
                valid_target = None
                if target_phone_input:
                    match = df_items[df_items['model'].str.contains(target_phone_input, case=False, na=False)]
                    if not match.empty:
                        valid_target = match.iloc[0]['model']
                        st.success(f"Targeting specs similar to: **{valid_target}**")
                    else:
                        st.warning("Phone not found in database. Relying on your slider ratings instead.")

                results = []
                # Process a sample of unrated phones to keep computation fast
                for m in unrated[:150]: 
                    cb = predict_live_cb(live_ratings, valid_target, m)
                    cf = predict_live_cf(live_ratings, m)
                    glob = df_items.loc[df_items['model'] == m, 'normalized_avg_rating'].values[0]
                    
                    # If target phone is found, give CB slightly more weight
                    if valid_target:
                        final = (0.3 * cf) + (0.5 * cb) + (0.2 * glob)
                    else:
                        final = (0.5 * cf) + (0.3 * cb) + (0.2 * glob)
                    
                    results.append({
                        'Smartphone': m, 
                        'Hybrid Score': round(final, 2), 
                        'Content': round(cb, 2), 
                        'Collab': round(cf, 2)
                    })
                
                recs = pd.DataFrame(results).sort_values(by='Hybrid Score', ascending=False).head(5)
                st.dataframe(recs, use_container_width=True)

    # Tab 2: Show evaluation metrics (Exactly as you originally wrote it)
    with tab2:
        if st.button("Generate Evaluation Chart"):
            with st.spinner("Running calculations for RMSE and K=1 to 10. This might take a moment..."):
                
                # --- 1. CALCULATE RMSE ---
                actuals = []
                predictions = []
                
                for _, row in df_ratings.sample(200, random_state=42).iterrows():
                    u = row['user_id']
                    m = row['model']
                    actual_rating = row['rating']
                    
                    cb = predict_cb(u, m)
                    cf = predict_cf(u, m)
                    glob = df_items.loc[df_items['model'] == m, 'normalized_avg_rating'].values[0] if m in df_items['model'].values else 0
                    
                    hybrid_prediction = (0.5 * cf) + (0.3 * cb) + (0.2 * glob)
                    
                    actuals.append(actual_rating)
                    predictions.append(hybrid_prediction)
                
                rmse_score = np.sqrt(mean_squared_error(actuals, predictions))
                
                # --- 2. CALCULATE PRECISION, RECALL, F1 vs K ---
                k_values = list(range(1, 11))
                precision_scores = []
                recall_scores = []
                f1_scores = []

                sample_users = df_ratings['user_id'].unique()[:15] 
                user_relevant = defaultdict(list)
                user_recs = defaultdict(list)

                for u in sample_users:
                    u_ratings = df_ratings[df_ratings['user_id'] == u]
                    for _, row in u_ratings.iterrows():
                        m = row['model']
                        if row['rating'] >= 4.0:
                            user_relevant[u].append(m)
                        
                        cb = predict_cb(u, m)
                        cf = predict_cf(u, m)
                        
                        if m in df_items['model'].values:
                            glob = df_items.loc[df_items['model'] == m, 'normalized_avg_rating'].values[0]
                        else:
                            glob = 0
                            
                        hyb = (0.5 * cf) + (0.3 * cb) + (0.2 * glob)
                        user_recs[u].append((m, hyb))

                for k in k_values:
                    k_prec, k_rec = [], []
                    for u in sample_users:
                        sorted_recs = sorted(user_recs[u], key=lambda x: x[1], reverse=True)[:k]
                        top_k_items = [item for item, score in sorted_recs]
                        rel_items = user_relevant[u]
                        
                        hits = len(set(top_k_items).intersection(set(rel_items)))
                        k_prec.append(hits / k)
                        k_rec.append(hits / len(rel_items) if len(rel_items) > 0 else 1)
                    
                    avg_p = np.mean(k_prec)
                    avg_r = np.mean(k_rec)
                    f1 = 2 * (avg_p * avg_r) / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0
                    
                    precision_scores.append(avg_p)
                    recall_scores.append(avg_r)
                    f1_scores.append(f1)

                # --- 3. PLOT THE CHART ---
                fig, ax = plt.subplots()

                ax.plot(k_values, precision_scores, marker='o', label='Precision')
                ax.plot(k_values, recall_scores, marker='s', label='Recall')
                ax.plot(k_values, f1_scores, marker='^', label='F1-score')

                ax.set_xlabel("K (Top Recommendations)")
                ax.set_ylabel("Score")
                ax.set_title("Evaluation Metrics vs K")
                ax.legend()

                st.pyplot(fig)

                # --- 4. SHOW RESULTS TABLE ---
                result_df = pd.DataFrame({
                    'K': k_values,
                    'Precision': precision_scores,
                    'Recall': recall_scores,
                    'F1-score': f1_scores
                })

                st.dataframe(result_df, use_container_width=True)

                # --- 5. SHOW ALL 4 METRICS CARDS ---
                avg_precision = np.mean(precision_scores)
                avg_recall = np.mean(recall_scores)
                avg_f1 = np.mean(f1_scores)

                st.subheader("System Evaluation Scores")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RMSE (Lower is Better)", f"{rmse_score:.4f}")
                col2.metric("Avg Precision@K", f"{avg_precision:.4f}")
                col3.metric("Avg Recall@K", f"{avg_recall:.4f}")
                col4.metric("Avg F1-score", f"{avg_f1:.4f}")