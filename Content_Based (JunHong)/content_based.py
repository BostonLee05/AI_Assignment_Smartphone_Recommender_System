import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 1. Load data
df = pd.read_csv('../smartphones_data.csv.csv')
df.columns = df.columns.str.lower()
df = df.fillna("")

# 2. Convert numeric safely
df['ram'] = pd.to_numeric(df['ram'], errors='coerce').fillna(0).astype(int)
df['storage'] = pd.to_numeric(df['storage'], errors='coerce').fillna(0).astype(int)
df['battery_cap'] = pd.to_numeric(df['battery_cap'], errors='coerce').fillna(0).astype(int)

# 3. Create metadata
df['metadata'] = df['brand_name'] + " " + \
                 df['name'] + " " + \
                 df['processor_brand'] + " " + \
                 df['os'] + " " + \
                 df['ram'].astype(str) + "GB RAM " + \
                 df['storage'].astype(str) + "GB Storage " + \
                 df['battery_cap'].astype(str) + "mAh"

# 4. TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['metadata'])

# 5. Similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 6. Index mapping
indices = pd.Series(df.index, index=df['name'].str.lower()).drop_duplicates()

# 7. Recommendation function
def get_recommend(phone_name, top_n=10):
    if phone_name not in indices:
        return "Phone not found"

    idx = indices[phone_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    phone_indices = [i[0] for i in sim_scores]

    return df[['brand_name', 'name', 'ram', 'storage', 'battery_cap', 'price']].iloc[phone_indices]

# 8. Run
print("--- Smartphone Recommender System ---")
user_input = input("Enter Smartphone Name: ").lower()

results = get_recommend(user_input)
print(results)

def evaluate_system(df, recommendations_func, k=10, sample_size=50):
    # Prevent crash if dataset is small
    sample_size = min(sample_size, len(df))

    test_samples = df.sample(sample_size)

    precision_scores = []

    for _, row in test_samples.iterrows():
        target_name = row['name'].lower()
        target_brand = row['brand_name']

        # Get recommendations
        recommendations = recommendations_func(target_name)

        if isinstance(recommendations, str):
            continue

        # Get recommended rows
        rec_indices = recommendations.index
        matches = df.loc[rec_indices]

        relevant_count = len(matches[
         (matches['brand_name'] == target_brand) &
         (matches['ram'] >= row['ram']) &
         (matches['battery_cap'] >= row['battery_cap'])
         ])

        precision_at_k = relevant_count / k
        precision_scores.append(precision_at_k)

    # Avoid empty list error
    if len(precision_scores) == 0:
        return 0

    mean_precision = np.mean(precision_scores)
    return mean_precision

m_precision = evaluate_system(df, get_recommend, k=10)

print(f"--- Evaluation Results ---")
print(f"Mean Precision@10: {m_precision:.2%}")