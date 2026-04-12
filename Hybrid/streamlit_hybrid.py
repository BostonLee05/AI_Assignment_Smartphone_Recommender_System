def run_hybrid():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import MinMaxScaler
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

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_items['content_features'])
    content_sim_matrix = cosine_similarity(tfidf_matrix)
    content_sim_df = pd.DataFrame(content_sim_matrix, index=df_items['model'], columns=df_items['model'])

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


    # --------------------------------------------------------
    # FRONTEND LOGIC: STREAMLIT UI & DASHBOARD
    # --------------------------------------------------------

    st.subheader("Hybrid Recommendations")

    # Added tab3 here for the Gantt Chart
    tab1, tab2, tab3 = st.tabs(["Live Recommendations", "System Evaluation", "Project Timeline"])

    # Tab 1: Show recommendations for a specific user
    with tab1:
        selected_user = st.slider("Select User ID to generate recommendations:", 1, 50, 1)
        
        with st.spinner("Calculating custom hybrid scores..."):
            all_models = df_items['model'].unique()
            rated = df_ratings[df_ratings['user_id'] == selected_user]['model'].tolist()
            unrated = [m for m in all_models if m not in rated]
            
            results = []
            for m in unrated[:100]: 
                cb = predict_cb(selected_user, m)
                cf = predict_cf(selected_user, m)
                glob = df_items.loc[df_items['model'] == m, 'normalized_avg_rating'].values[0]
                
                final = (0.5 * cf) + (0.3 * cb) + (0.2 * glob)
                
                results.append({
                    'Smartphone': m, 
                    'Hybrid Score': round(final, 2), 
                    'Content': round(cb, 2), 
                    'Collab': round(cf, 2)
                })
            
            recs = pd.DataFrame(results).sort_values(by='Hybrid Score', ascending=False).head(5)
            st.dataframe(recs, use_container_width=True)

    # Tab 2: Show evaluation metrics
    with tab2:
        if st.button("Generate Evaluation Chart"):
            with st.spinner("Running calculations for K=1 to 10. This might take a moment..."):
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

                # plot using matplotlib
                fig, ax = plt.subplots()

                ax.plot(k_values, precision_scores, marker='o', label='Precision')
                ax.plot(k_values, recall_scores, marker='s', label='Recall')
                ax.plot(k_values, f1_scores, marker='^', label='F1-score')

                ax.set_xlabel("K (Top Recommendations)")
                ax.set_ylabel("Score")
                ax.set_title("Evaluation Metrics vs K")
                ax.legend()

                st.pyplot(fig)

                # result table
                result_df = pd.DataFrame({
                    'K': k_values,
                    'Precision': precision_scores,
                    'Recall': recall_scores,
                    'F1-score': f1_scores
                })

                st.dataframe(result_df, use_container_width=True)

                # average metric cards
                avg_precision = np.mean(precision_scores)
                avg_recall = np.mean(recall_scores)
                avg_f1 = np.mean(f1_scores)

                st.subheader("Average Evaluation Metrics")

                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Precision", f"{avg_precision:.4f}")
                col2.metric("Avg Recall", f"{avg_recall:.4f}")
                col3.metric("Avg F1-score", f"{avg_f1:.4f}")

    # Tab 3: Project Gantt Chart
    with tab3:
        st.subheader("Project Gantt Chart")
        st.write("This timeline illustrates the chronological phases required to successfully develop the Hybrid Smartphone Recommender System.")
        
        # Data perfectly matching the 4 stages from Figure 3.5
        gantt_data = [
            {"Task": "Data Preprocessing", "Start": 1, "Duration": 2, "Color": "#85C1E9"}, # Light Blue
            {"Task": "Algorithm Development", "Start": 3, "Duration": 4, "Color": "#3498DB"}, # Solid Blue
            {"Task": "Hybrid Integration", "Start": 7, "Duration": 2, "Color": "#F5B7B1"}, # Pink/Light Red
            {"Task": "UI & Deployment", "Start": 9, "Duration": 3, "Color": "#E74C3C"}  # Solid Red
        ]
        
        df_gantt = pd.DataFrame(gantt_data)
        
        # Create a visual Gantt chart using matplotlib
        fig_gantt, ax_gantt = plt.subplots(figsize=(10, 4))
        
        # Draw horizontal bars with the specific colors from the screenshot
        for i, row in df_gantt.iterrows():
            ax_gantt.barh(row['Task'], row['Duration'], left=row['Start'], color=row['Color'], edgecolor='black')
        
        # Formatting the chart to look clean and professional
        ax_gantt.set_xlabel("Timeline (Weeks)")
        ax_gantt.set_ylabel("Task")
        ax_gantt.set_xticks(range(1, 13)) # Sets X-axis from Week 1 to 12
        ax_gantt.invert_yaxis() # Puts the first task at the top instead of the bottom
        
        st.pyplot(fig_gantt)