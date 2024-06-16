import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
import movieposters
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
import random

# Load the dataset
df = pd.read_csv('75k_Movies_90_2k_23_en_filtered_wout_anime_gt5.csv')

# Fill missing values in text columns with an empty string
text_cols = ['keywords', 'genres', 'overview', 'tagline', 'production_companies', 'production_countries']
df[text_cols] = df[text_cols].fillna('')

# Convert release_date to datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Normalize numerical features
df['vote_average_norm'] = (df['vote_average'] - df['vote_average'].min()) / (df['vote_average'].max() - df['vote_average'].min())
df['revenue_norm'] = (df['revenue'] - df['revenue'].min()) / (df['revenue'].max() - df['revenue'].min())
df['runtime_norm'] = (df['runtime'] - df['runtime'].min()) / (df['runtime'].max() - df['runtime'].min())

# Combine relevant text fields and normalized numerical features into a single string for TF-IDF
df['combined_features'] = df.apply(lambda row: ' '.join([
    row['title'], row['overview'], row['tagline'], row['genres'],
    row['production_companies'], row['production_countries'], row['keywords'],
    str(row['vote_average_norm']), str(row['revenue_norm']), str(row['runtime_norm']),
    row['release_date'].strftime('%Y-%m-%d')
]), axis=1)


# Load the pre-trained models
tfidf_vectorizer = joblib.load('Models/75k_tfidf_vector_knn_19_20_23_wout_anime_gt5.joblib')
knn_model = joblib.load('Models/75k_knn_model_19_20_23_wout_anime_gt5.joblib')
assert isinstance(knn_model, NearestNeighbors), "knn_model is not a NearestNeighbors instance"


# Function to fetch poster URLs concurrently
# @st.cache_data(show_spinner="Fetching data from API...")
def fetch_poster_urls(movie_titles):
    poster_urls = {}
    # @st.cache_data(show_spinner="Fetching data from API...")
    def fetch_poster(title):
        try:
            poster_url = movieposters.get_poster(title)  # Replace with actual poster fetching logic
            poster_urls[title] = poster_url
        except Exception as e:
            print(f"Error fetching poster for {title}: {e}")
            poster_urls[title] = "https://via.placeholder.com/150"  # Placeholder URL
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_poster, title) for title in movie_titles]
        for future in as_completed(futures):
            pass  # Wait for all threads to complete
    
    return poster_urls

# Function to get recommendations
@st.cache_data(show_spinner="🎥 Fetching movie recommendations... Grab some popcorn! 🍿")
def get_recommendations(title, _nn_model=knn_model, df=df, _tfidf=tfidf_vectorizer, num_recommendations=20):
    if title not in df['title'].values:
        return f"Title '{title}' not found in the dataset."
    
    idx = df.index[df['title'] == title].tolist()[0]
    selected_movie = df.iloc[idx]
    tfidf_vector = _tfidf.transform([selected_movie['combined_features']])
    distances, indices = _nn_model.kneighbors(tfidf_vector, n_neighbors=num_recommendations + 1)
    movie_indices = indices.flatten()[1:]
    # print(type(movie_indices))
    
    recommendations = df.iloc[movie_indices].reset_index(drop=True)  
    
    # Fetch poster URLs concurrently
    movie_titles = recommendations['title'].tolist()
    poster_urls = fetch_poster_urls(movie_titles)
    
    # Add poster URLs to recommendations DataFrame
    recommendations['poster_url'] = [poster_urls.get(title, "https://via.placeholder.com/200") for title in movie_titles]
    
    return recommendations


def get_random_compliment():
    compliments = [
        "Great taste!",
        "Excellent choice!",
        "That's a classic!",
        "Nice pick!",
        "Awesome selection!",
        "You've got good taste!",
        "Well chosen!",
        "Fantastic pick!",
        "Impressive!",
        "You know your movies!",
        "Outstanding choice!",
        "Top-notch selection!",
        "Brilliant!",
        "Way to go!",
        "Well done!",
        "Superb!",
        "Terrific choice!",
        "Spot on!",
        "A masterpiece of a choice!",
        "Exceptional taste!"
    ]
    return random.choice(compliments)


def default_recommendation():
    d_movie_indices = np.array([47, 15, 17, 52, 43, 211, 87, 25, 18, 240])
    d_recommendation = df.iloc[d_movie_indices].reset_index(drop=True)
    movie_titles = d_recommendation['title'].tolist()
    poster_urls = fetch_poster_urls(movie_titles)
    d_recommendation['poster_url'] = [poster_urls.get(title, "https://via.placeholder.com/150") for title in movie_titles]

    return d_recommendation


def convert_runtime(minutes):
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours} hrs {mins} mins"

def format_adult(adult):
    return "Yes" if adult else "No"