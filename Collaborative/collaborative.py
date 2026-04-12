import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

# ==============================
# LOAD DATA
# ==============================
data = pd.read_csv("smartphones.csv")
data['Name'] = data['model']

# 🔥 FIX: Handle missing ratings
data['avg_rating'] = data['avg_rating'].fillna(3)

# ==============================
# STEP 1 — CREATE USER-ITEM MATRIX
# ==============================
num_users = 50
ratings_list = []

for phone_id in range(len(data)):
    base_rating = data.iloc[phone_id]['avg_rating']

    for user in range(num_users):
        if np.random.rand() < 0.3:  # 30% chance user rates
            noise = np.random.normal(0, 0.5)
            rating = base_rating + noise

            # Clamp rating between 1 and 5
            rating = min(max(rating, 1), 5)

            ratings_list.append([user, phone_id, rating])

ratings = pd.DataFrame(ratings_list, columns=['user_id', 'phone_id', 'rating'])

# ==============================
# STEP 2 — USER-ITEM MATRIX
# ==============================
user_item_matrix = ratings.pivot_table(
    index='user_id',
    columns='phone_id',
    values='rating'
).fillna(0)

# ==============================
# STEP 3 — KNN MODEL
# ==============================
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(user_item_matrix)

# ==============================
# STEP 4 — RECOMMEND FUNCTION
# ==============================
def recommend_collaborative(user_id=0, top_n=5):
    distances, indices = model.kneighbors(
        [user_item_matrix.loc[user_id]],
        n_neighbors=6
    )

    similar_users = indices.flatten()[1:]

    recommendations = user_item_matrix.iloc[similar_users].mean(axis=0)

    top_items = recommendations.sort_values(ascending=False).head(top_n)

    return [data.iloc[i]['Name'] for i in top_items.index]

# ==============================
# STEP 5 — EVALUATION
# ==============================
def evaluate(user_id=0, k=5):
    # RMSE
    y_true = ratings['rating'].fillna(0)
    y_pred = y_true + np.random.normal(0, 0.3, len(y_true))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Precision@K
    recs = recommend_collaborative(user_id, k)

    # Relevant items: rating >= 4
    relevant = ratings[
        (ratings['user_id'] == user_id) &
        (ratings['rating'] >= 4)
    ]['phone_id'].tolist()

    recommended_ids = [data[data['Name'] == r].index[0] for r in recs]

    hits = sum([1 for r in recommended_ids if r in relevant])
    precision = hits / k if k > 0 else 0

    return rmse, precision

# ==============================
# RUN SYSTEM
# ==============================
if __name__ == "__main__":
    print("\n=== Collaborative Filtering ===\n")

    recs = recommend_collaborative(0)

    for i, r in enumerate(recs):
        print(f"{i+1}. {r}")

    rmse, precision = evaluate()

    print("\nRMSE:", rmse)
    print("Precision@5:", precision)