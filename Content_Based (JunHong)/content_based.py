import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 1. LOAD & CLEAN DATA
df = pd.read_csv('../smartphones.csv')

# standardize column names
df.columns = df.columns.str.lower()

# fill missing values
df = df.fillna('')

# ensure consistent string format
df['model'] = df['model'].str.lower().str.strip()
df['brand_name'] = df['brand_name'].str.lower().str.strip()
df['os'] = df['os'].str.lower().str.strip()
df['processor_brand'] = df['processor_brand'].str.lower().str.strip()
df['battery_capacity'] = pd.to_numeric(df['battery_capacity'], errors='coerce').fillna(0)
df['ram_capacity'] = pd.to_numeric(df['ram_capacity'], errors='coerce').fillna(0)
df['refresh_rate'] = pd.to_numeric(df['refresh_rate'], errors='coerce').fillna(0)

# 2. CREATE METADATA
df['metadata'] = (
    df['brand_name'] + " " +
    df['os'] + " " +
    df['processor_brand'] + " " +
    df['battery_capacity'].astype(str) + " mAh " +
    df['ram_capacity'].astype(str) + " GB " +
    df['refresh_rate'].astype(str) + " Hz"
)

# 3. TF-IDF + SIMILARITY
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['metadata'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# map name → index
indices = pd.Series(df.index, index=df['model']).drop_duplicates()

# 4. RECOMMENDATION FUNCTION
def get_recommendations(model, cosine_sim=cosine_sim):
    model = model.lower().strip()

    if model not in indices:
        return "Smartphone not found in dataset."

    idx = indices[model]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]  # top 10 excluding itself

    smart_indices = [i[0] for i in sim_scores]

    return df.iloc[smart_indices][['model', 'brand_name', 'os']]

# 5. USER INTERACTION
print("--- Smartphone Recommender System ---")
user_input = input("Enter Smartphone Name: ")

results = get_recommendations(user_input)

print(f"\nRecommendations for '{user_input}':")
print(results)

# 6. EVALUATION (Precision@K)
def evaluate_system(df, recommendations_func, k=10, sample_size=50):

    sample_size = min(sample_size, len(df))
    test_samples = df.sample(sample_size, random_state=42)

    precision_scores = []

    for _, row in test_samples.iterrows():
        target_name = row['model']
        target_brand = row['brand_name']
        target_os = row['os']

        recommendations = recommendations_func(target_name)

        if isinstance(recommendations, str):
            continue

        # take top-k recommendations safely
        rec_top_k = recommendations.head(k)

        relevant_count = rec_top_k[
            rec_top_k['brand_name'] == target_brand
        ].shape[0]

        precision_at_k = relevant_count / k
        precision_scores.append(precision_at_k)

    return np.mean(precision_scores) if precision_scores else 0

# RUN EVALUATION
m_precision = evaluate_system(df, get_recommendations, k=10)

print("\n--- Evaluation Results ---")
print(f"Mean Precision@10: {m_precision:.2%}")