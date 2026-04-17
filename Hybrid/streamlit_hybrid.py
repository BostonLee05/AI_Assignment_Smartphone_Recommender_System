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
    # DATA PREPARATION & MODEL BUILDING
    # --------------------------------------------------------

    @st.cache_data
    def load_data():
        df = pd.read_csv('smartphones.csv')
        df = df.drop_duplicates(subset=['model']).reset_index(drop=True)
        
        # Fill missing ratings with the dataset mean
        df['avg_rating'] = df['avg_rating'].fillna(df['avg_rating'].mean())
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('')
            else:
                df[col] = df[col].fillna(0)
                
        # Normalize the average rating to a 0-1 scale for the global score component
        scaler = MinMaxScaler(feature_range=(0, 5))
        df['normalized_avg_rating'] = scaler.fit_transform(df[['avg_rating']])
        
        # Combine phone specifications into a single text feature for TF-IDF
        df['content_features'] = (
            df['brand_name'].astype(str) + ' ' + 
            df['os'].astype(str) + ' ' + 
            df['processor_brand'].astype(str) + ' ' +
            df['ram_capacity'].astype(str) + 'gb ' + 
            df['internal_memory'].astype(str) + 'gb'
        ).fillna('').str.lower()
        
        return df

    df_items = load_data()

    # Build Content-Based Matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_items['content_features'])
    content_sim_matrix = cosine_similarity(tfidf_matrix)
    content_sim_df = pd.DataFrame(content_sim_matrix, index=df_items['model'], columns=df_items['model'])

    # Generate Mock Data for Collaborative Filtering
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

    # --- HISTORICAL PREDICTION FUNCTIONS (For System Evaluation) ---
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

    # --- LIVE PREDICTION FUNCTIONS (For User Input) ---
    def predict_live_cb(live_ratings_dict, target_phone, item):
        # Weight heavily if the user explicitly searched for a specific phone
        if target_phone and target_phone in content_sim_df.index:
            return content_sim_df.loc[item, target_phone] * 5.0 
            
        # Otherwise, calculate similarity based on their slider ratings
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
            
        # Create a vector for the current user to compare against historical mock users
        live_vec = np.zeros(len(user_item_matrix.columns))
        cols = list(user_item_matrix.columns)
        for m, r in live_ratings_dict.items():
            if m in cols:
                live_vec[cols.index(m)] = r
                
        sims = cosine_similarity([live_vec], user_item_matrix.values)[0]
        sim_users = pd.Series(sims, index=user_item_matrix.index)
        
        item_rats = user_item_matrix[item]
        mask = item_rats > 0 
        if not mask.any(): 
            return 3.0
            
        den = sim_users[mask].sum()
        return np.dot(sim_users[mask], item_rats[mask]) / den if den > 0 else 3.0

    # --------------------------------------------------------
    # STREAMLIT USER INTERFACE
    # --------------------------------------------------------

    st.title("Hybrid Recommendation System")
    st.write("This module combines Content-Based Filtering (searching by phone specs) and Collaborative Filtering (user ratings) to generate recommendations.")

    tab1, tab2 = st.tabs(["Live Recommendations", "System Evaluation"])

    # --- TAB 1: LIVE RECOMMENDATIONS ---
    with tab1:
        st.markdown("### Step 1: Search for a Phone (Optional)")
        target_phone_input = st.text_input("Enter a smartphone name to find similar specifications:", placeholder="e.g., Apple iPhone 14")
        
        st.divider()
        
        st.markdown("### Step 2: Rate Sample Phones")
        
        # Re-sample phones and clear old slider states
        if st.button("Randomize Sample Phones", key="hybrid_randomize_btn"):
            st.session_state.live_sample = df_items.sample(5)
            for key in list(st.session_state.keys()):
                if key.startswith("live_rating_"):
                    del st.session_state[key]
            st.rerun()

        if "live_sample" not in st.session_state:
            st.session_state.live_sample = df_items.sample(5)

        live_ratings = {}
        for idx, row in st.session_state.live_sample.iterrows():
            model_name = row['model']
            rating = st.slider(f"Rate {model_name}", 1, 5, 3, key=f"live_rating_{idx}")
            live_ratings[model_name] = rating
        
        if st.button("Generate Recommendations", key="hybrid_rec_btn"):
            with st.spinner("Calculating hybrid scores..."):
                all_models = df_items['model'].unique()
                unrated = [m for m in all_models if m not in live_ratings.keys() and m != target_phone_input]
                
                valid_target = None
                if target_phone_input:
                    match = df_items[df_items['model'].str.contains(target_phone_input, case=False, na=False)]
                    if not match.empty:
                        valid_target = match.iloc[0]['model']
                        st.success(f"Targeting specifications similar to: **{valid_target}**")
                    else:
                        st.warning("Phone not found in the database. Relying on slider ratings instead.")

                results = []
                for m in unrated[:150]: 
                    cb = predict_live_cb(live_ratings, valid_target, m)
                    cf = predict_live_cf(live_ratings, m)
                    glob = df_items.loc[df_items['model'] == m, 'normalized_avg_rating'].values[0]
                    
                    # Adjust weights based on whether a specific phone was searched
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
                st.subheader("Top Recommendations")
                st.dataframe(recs, use_container_width=True)

    # --- TAB 2: SYSTEM EVALUATION ---
    with tab2:
        if st.button("Run System Evaluation", key="hybrid_eval_btn"):
            with st.spinner("Calculating RMSE and Precision/Recall metrics. This may take a moment..."):
                
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

                # --- 3. PLOT METRICS ---
                fig, ax = plt.subplots()

                ax.plot(k_values, precision_scores, marker='o', label='Precision')
                ax.plot(k_values, recall_scores, marker='s', label='Recall')
                ax.plot(k_values, f1_scores, marker='^', label='F1-score')

                ax.set_xlabel("K (Number of Recommendations)")
                ax.set_ylabel("Score")
                ax.set_title("System Evaluation Metrics vs K")
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

                # --- 5. SHOW METRICS ---
                avg_precision = np.mean(precision_scores)
                avg_recall = np.mean(recall_scores)
                avg_f1 = np.mean(f1_scores)

                st.subheader("System Evaluation Scores")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RMSE (Lower is Better)", f"{rmse_score:.4f}")
                col2.metric("Avg Precision@K", f"{avg_precision:.4f}")
                col3.metric("Avg Recall@K", f"{avg_recall:.4f}")
                col4.metric("Avg F1-score", f"{avg_f1:.4f}")