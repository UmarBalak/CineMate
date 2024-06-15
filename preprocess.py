import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
import movieposters
from concurrent.futures import ThreadPoolExecutor, as_completed


# Load the dataset
df = pd.read_csv('Movies_1990_2000_2023_en_filtered.csv')

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
tfidf_vectorizer = joblib.load('Models/tfidf_vectorizer_knn_19_20_23.joblib')
knn_model = joblib.load('Models/knn_model_19_20_23.joblib')
assert isinstance(knn_model, NearestNeighbors), "knn_model is not a NearestNeighbors instance"

# Function to fetch poster URLs concurrently
def fetch_poster_urls(movie_titles):
    poster_urls = {}
    
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
def get_recommendations(title, nn_model=knn_model, df=df, tfidf=tfidf_vectorizer, num_recommendations=10):
    if title not in df['title'].values:
        return f"Title '{title}' not found in the dataset."
    
    idx = df.index[df['title'] == title].tolist()[0]
    selected_movie = df.iloc[idx]
    tfidf_vector = tfidf.transform([selected_movie['combined_features']])
    distances, indices = nn_model.kneighbors(tfidf_vector, n_neighbors=num_recommendations + 1)
    movie_indices = indices.flatten()[1:]
    
    recommendations = df.iloc[movie_indices][['title', 'release_date']].reset_index(drop=True)
    
    # Sort recommendations by release date (newest first)
    recommendations['release_date'] = pd.to_datetime(recommendations['release_date'])
    recommendations = recommendations.sort_values(by='release_date', ascending=False).reset_index(drop=True)
    
    # Fetch poster URLs concurrently
    movie_titles = recommendations['title'].tolist()
    poster_urls = fetch_poster_urls(movie_titles)
    
    # Add poster URLs to recommendations DataFrame
    recommendations['poster_url'] = [poster_urls.get(title, "https://via.placeholder.com/150") for title in movie_titles]
    
    return recommendations