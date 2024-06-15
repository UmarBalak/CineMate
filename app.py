import streamlit as st
from preprocess import get_recommendations, get_poster_url, knn_model, tfidf_vectorizer, df


# Streamlit app
st.title("Movie Recommendation System")

# User input for movie title with auto-suggestion
movie_title = st.selectbox(
    "Enter a movie title:",
    options=df['title'].unique()
)

# Function to display recommendations with movie posters
def display_recommendations(movie_title):
    recommendations = get_recommendations(movie_title, knn_model, df, tfidf_vectorizer)
    
    if isinstance(recommendations, str):  # Check if the return is an error message
        st.error(recommendations)
    else:
        st.subheader(f"Top 10 movie recommendations for '{movie_title}':")
        for idx, row in recommendations.iterrows():
            poster_url = get_poster_url(row['title'])
            st.image(poster_url, caption=row['title'], width=200, use_column_width=False)
            st.write(f"{idx + 1}. {row['title']} (Release Date: {row['release_date'].strftime('%Y-%m-%d')})")
            st.write("---")

# Display recommendations if a movie title is selected
if movie_title:
    display_recommendations(movie_title)