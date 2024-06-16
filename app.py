import streamlit as st
from preprocess import get_recommendations, get_random_compliment, default_recommendation, knn_model, tfidf_vectorizer, df
import time

# Streamlit app
st.title("Movie Recommendation System")

# User input for movie title with auto-suggestion
default_movie = 4
movie_title = st.selectbox(
    "Select your favorite movie:",
    options=df['title'].unique(),
    index = None
)

# Function to display recommendations with movie posters
def display_recommendations(movie_title):
    start_time = time.time()
    recommendations = get_recommendations(movie_title, knn_model, df, tfidf_vectorizer)
    # print(recommendations)
    
    if isinstance(recommendations, str):  # Check if the return is an error message
        st.error(recommendations)
    else:
        st.subheader(f"Top 10 movie recommendations for '{movie_title}':")
        for idx, row in recommendations.head(10).iterrows():
            st.image(row['poster_url'], caption=row['title'], width=200, use_column_width=False)
            st.write(f"{idx + 1}. {row['title']} (Release Date: {row['release_date'].strftime('%Y-%m-%d')})")
            st.write("---")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

def display_default_recommendations():
    recommendations = default_recommendation()
    if isinstance(recommendations, str):  # Check if the return is an error message
        st.error(recommendations)
    else:
        st.subheader(f"Our Top 10 Recommendations:")
        for idx, row in recommendations.head(10).iterrows():
            st.image(row['poster_url'], caption=row['title'], width=200, use_column_width=False)
            st.write(f"{idx + 1}. {row['title']} (Release Date: {row['release_date'].strftime('%Y-%m-%d')})")
            st.write("---")

# Display recommendations if a movie title is selected
if movie_title:
    compliment = get_random_compliment()
    st.info(compliment)
    display_recommendations(movie_title)
else:
    display_default_recommendations()
