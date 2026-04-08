import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

# 1. Load and Prepare Data
df = pd.read_csv('smartphones_data.csv.csv')

# Combine metadata columns to create a rich description for each phone
# We use brand, OS, processor, and display type as our features
df['metadata'] = (df['brand_name'] + " " +
                  df['OS'] + " " +
                  df['processor_brand'] + " " +
                  df['display_types']).fillna('')

# 2. Produce the TF-IDF Matrix
# Initialize the Vectorizer and remove common English stop words
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['metadata'])

# 3. Calculate Cosine Similarity
# Using linear_kernel as it is faster for calculating dot products of TF-IDF vectors
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a mapping of smartphone names to their indices
indices = pd.Series(df.index, index=df['Name'].str.lower()).drop_duplicates()

# 4. Define the Recommendation Function
def get_recommendations(name, cosine_sim=cosine_sim):
    # Get the index of the smartphone that matches the name
    if name not in indices:
        return "Smartphone not found in dataset."

    idx = indices[name]
    # Handle duplicate names by taking the first occurrence
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    # Get the list of cosine similarity scores for that smartphone with all smartphones
    # Convert it into a list of tuples (index, similarity_score)
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the list of tuples based on the similarity scores (the second element)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 elements (excluding the item itself at index 0)
    sim_scores = sim_scores[1:11]

    # Get the corresponding indices
    smart_indices = [i[0] for i in sim_scores]

    # Return the names corresponding to the indices of the top elements
    return df['Name'].iloc[smart_indices]


print("--- Smartphone Recommender System ---")
user_input = input("Enter Smartphone Name: ")

results = get_recommendations(user_input)

print(f"\nRecommendations for '{user_input}':")
print(results)


def evaluate_system(df, recommendations_func, k=10, sample_size=50):
    """
    Evaluates the recommender using Precision@K based on Brand and OS match.
    """
    # Take a random sample of phones to test
    test_samples = df.sample(sample_size)

    precision_scores = []

    for _, row in test_samples.iterrows():
        target_name = row['Name'].lower()
        target_brand = row['brand_name']
        target_os = row['OS']

        # Get recommendations
        recommendations = recommendations_func(target_name)

        if isinstance(recommendations, str):  # Handle "Not found"
            continue

        # Calculate how many of the top K share the same Brand OR OS
        # We consider a match 'relevant' if it matches the brand
        rec_indices = recommendations.index
        matches = df.loc[rec_indices]

        relevant_count = len(matches[matches['brand_name'] == target_brand])

        # Precision @ K = (Relevant Items Recommended) / (Total Items Recommended)
        precision_at_k = relevant_count / k
        precision_scores.append(precision_at_k)

    mean_precision = np.mean(precision_scores)
    return mean_precision


# --- Run the Evaluation ---
# Pass your function into the evaluator
m_precision = evaluate_system(df, get_recommendations, k=10)

print(f"--- Evaluation Results ---")
print(f"Mean Precision@10: {m_precision:.2%}")