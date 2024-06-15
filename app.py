import streamlit as st
from preprocess import get_recommendations, get_poster_url, knn_model, tfidf_vectorizer, df


# Streamlit app
st.title("Movie Recommendation System")

# User input for movie title with auto-suggestion
movie_title = st.selectbox(
    "Enter a movie title:",
    options=df['title'].unique()
)

if movie_title:
    recommendations = get_recommendations(movie_title, knn_model, df, tfidf_vectorizer)
    
    if isinstance(recommendations, str):  # Check if the return is an error message
        st.write(recommendations)
    else:
        st.write(f"Top 10 movie recommendations for '{movie_title}':")
        for idx, row in recommendations.iterrows():
            poster_url = get_poster_url(row['title'])
            st.image(poster_url, width=100)
            st.write(f"{idx + 1}. {row['title']} (Release Date: {row['release_date'].strftime('%Y-%m-%d')})")
