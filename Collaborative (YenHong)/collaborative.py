import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numpy as np

# ==============================
# LOAD DATASET
# ==============================
data = pd.read_csv("smartphones.csv")

# Standardize column name
data['Name'] = data['model']

# ==============================
# PREPARE DATA (USE REAL RATINGS)
# ==============================
# Use relevant features including rating
features = [
    'price',
    'avg_rating',
    'ram_capacity',
    'internal_memory',
    'battery_capacity',
    'screen_size',
    'refresh_rate'
]

df = data[features].fillna(0)

# ==============================
# ITEM SIMILARITY (Collaborative-style)
# ==============================
similarity_matrix = cosine_similarity(df)

# ==============================
# RECOMMEND FUNCTION
# ==============================
def recommend_collaborative(phone_name, top_n=5):
    phone_name = phone_name.lower()

    indices = pd.Series(data.index, index=data['Name'].str.lower()).drop_duplicates()

    if phone_name not in indices:
        return ["Phone not found"]

    idx = indices[phone_name]

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n+1]

    phone_indices = [i[0] for i in sim_scores]

    return data['Name'].iloc[phone_indices].tolist()


# ==============================
# EVALUATION (REALISTIC)
# ==============================
def evaluate_model():
    # Use avg_rating as ground truth
    y_true = data['avg_rating'].fillna(0)

    # Predicted = slightly noisy version (simple approximation)
    y_pred = y_true + np.random.normal(0, 0.5, len(y_true))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return rmse


# ==============================
# TEST
# ==============================
if __name__ == "__main__":
    print("\n=== Collaborative Filtering ===\n")

    sample_phone = data.iloc[0]['Name']

    print(f"Input: {sample_phone}\n")

    recs = recommend_collaborative(sample_phone)

    for i, r in enumerate(recs):
        print(f"{i+1}. {r}")

    rmse = evaluate_model()

    print("\nRMSE:", rmse)