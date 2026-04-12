import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import warnings

# ignore warnings to keep the terminal output clean
warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv('smartphones.csv')
    df = df.drop_duplicates(subset=['model']).reset_index(drop=True)
    
    # fill missing average ratings with the mean
    df['avg_rating'] = df['avg_rating'].fillna(df['avg_rating'].mean())
    
    # clean up missing values to prevent data errors
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')
        else:
            df[col] = df[col].fillna(0)
            
    # normalize the average rating to a 0-5 scale
    scaler = MinMaxScaler(feature_range=(0, 5))
    df['normalized_avg_rating'] = scaler.fit_transform(df[['avg_rating']])
    
    # combine all phone specs into one string for content-based matching
    df['content_features'] = (
         df['brand_name'].astype(str) + ' ' + 
         df['os'].astype(str) + ' ' + 
         df['processor_brand'].astype(str) + ' ' +
         df['ram_capacity'].astype(str) + 'gb ' + 
         df['internal_memory'].astype(str) + 'gb'
    ).fillna('').str.lower()
    
    return df

# setup data and matrices
df_items = load_data()

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_items['content_features'])
content_sim_matrix = cosine_similarity(tfidf_matrix)
content_sim_df = pd.DataFrame(content_sim_matrix, index=df_items['model'], columns=df_items['model'])

# generate dummy users for collaborative filtering since dataset lacks user history
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

# prediction functions
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

def get_recommendations(user_id, top_n=5):
    all_models = df_items['model'].unique()
    rated = df_ratings[df_ratings['user_id'] == user_id]['model'].tolist()
    unrated = [m for m in all_models if m not in rated]
    
    results = []
    for m in unrated[:100]: # limit to 100 for processing speed
        cb = predict_cb(user_id, m)
        cf = predict_cf(user_id, m)
        glob = df_items.loc[df_items['model'] == m, 'normalized_avg_rating'].values[0]
        final = (0.5 * cf) + (0.3 * cb) + (0.2 * glob)
        
        results.append({'Smartphone': m, 'Hybrid Score': round(final, 2)})
    
    recs = pd.DataFrame(results).sort_values(by='Hybrid Score', ascending=False).head(top_n)
    return recs['Smartphone'].tolist()

# run the terminal script
if __name__ == "__main__":
    print("\n--- Hybrid Recommendation System ---")
    user_test = 1
    print(f"\nGenerating Top 5 Recommendations for User {user_test}...")
    recs = get_recommendations(user_test)
    
    for i, r in enumerate(recs):
        print(f"{i+1}. {r}")
    
    print("\nSystem running successfully! For full evaluation metrics and interactive charts, please run the Streamlit app.")