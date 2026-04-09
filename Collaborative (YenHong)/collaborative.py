import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

# ==============================
# LOAD DATASET
# ==============================
data = pd.read_csv("../data/smartphones.csv")

# ==============================
# CREATE FAKE USER RATINGS
# ==============================
num_users = 50
ratings_list = []

for user in range(num_users):
    for phone in range(len(data)):
        if np.random.rand() < 0.1:  # 10% chance
            rating = np.random.randint(1, 6)  # 1–5 rating
            ratings_list.append([user, phone, rating])

ratings = pd.DataFrame(ratings_list, columns=['user_id', 'phone_id', 'rating'])

# ==============================
# CREATE USER-ITEM MATRIX
# ==============================
user_item_matrix = ratings.pivot_table(
    index='user_id',
    columns='phone_id',
    values='rating'
).fillna(0)

# ==============================
# BUILD KNN MODEL
# ==============================
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(user_item_matrix)

# ==============================
# RECOMMENDATION FUNCTION
# ==============================
def recommend_collaborative(user_id=0, top_n=5):
    distances, indices = model.kneighbors(
        [user_item_matrix.loc[user_id]],
        n_neighbors=6
    )

    similar_users = indices.flatten()[1:]

    recommendations = user_item_matrix.iloc[similar_users].mean(axis=0)

    top_items = recommendations.sort_values(ascending=False).head(top_n)

    results = []
    for phone_id in top_items.index:
        results.append(data.iloc[phone_id]['model'])

    return results


# ==============================
# EVALUATION (RMSE + Precision@K)
# ==============================
def evaluate_model(user_id=0, k=5):
    # RMSE (simple simulation)
    y_true = ratings['rating']
    y_pred = np.random.randint(1, 6, size=len(y_true))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Precision@K
    recommended = recommend_collaborative(user_id, k)

    user_ratings = ratings[ratings['user_id'] == user_id]
    relevant_items = user_ratings[user_ratings['rating'] >= 4]['phone_id'].tolist()

    recommended_ids = [data[data['model'] == name].index[0] for name in recommended]

    hits = sum([1 for item in recommended_ids if item in relevant_items])
    precision = hits / k if k > 0 else 0

    return rmse, precision


# ==============================
# TEST (ONLY FOR DEBUG)
# ==============================
if __name__ == "__main__":
    print("\n=== Collaborative Filtering ===\n")

    recs = recommend_collaborative(user_id=0)

    for i, r in enumerate(recs):
        print(f"{i+1}. {r}")

    rmse, precision = evaluate_model()

    print("\nRMSE:", rmse)
    print("Precision@5:", precision)