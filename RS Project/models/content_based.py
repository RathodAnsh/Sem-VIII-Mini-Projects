import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "EATFIT_DIET.csv")

# Sample fallback if CSV missing
if not os.path.exists(dataset_path):
    data = {
        "Breakfast": [
            "Oats/Fruit/Milk",
            "Poha/Tea",
            "Eggs/Bread",
            "Idli/Sambar"
        ],
        "Lunch": [
            "Rice/Dal/Vegetable",
            "Chapati/Sabzi",
            "Chicken/Rice",
            "Paneer/Roti"
        ],
        "Dinner": [
            "Soup/Salad",
            "Roti/Dal",
            "Grilled Chicken",
            "Vegetable Khichdi"
        ]
    }
    df = pd.DataFrame(data)
else:
    df = pd.read_csv(dataset_path)


def recommend_similar_meals(input_meal):
    """
    Content-based filtering using TF-IDF
    """

    # Combine meals
    df["combined"] = df["Breakfast"] + " " + df["Lunch"] + " " + df["Dinner"]

    # Add input meal
    all_meals = df["combined"].tolist()
    all_meals.append(input_meal)

    # Vectorize
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_meals)

    # Similarity
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Get best match index
    index = similarity.argsort()[0][-1]

    return df.iloc[[index]]