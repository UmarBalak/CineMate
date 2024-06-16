import streamlit as st
from preprocess import get_recommendations, get_random_compliment, default_recommendation, convert_runtime, format_adult, knn_model, tfidf_vectorizer, df
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
            with st.container():
                col1, col2 = st.columns([1, 2])  # Adjust the ratio if needed
                with col1:
                    st.image(row['poster_url'], caption=row['title'], width=205.5, use_column_width=False)
                    # st.write("---")
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Add space above movie name
                    st.write(f"**{idx + 1}. {row['title']}**")
                    st.markdown(
                        f"<p style='font-style: italic; color: gray;'>\"{row['tagline']}\"</p>",
                        unsafe_allow_html=True
                    )
                    st.write(f"Release Date: {row['release_date'].strftime('%Y-%m-%d')}")
                    st.write(f"Rating: {row['vote_average']}")
                    st.write(f"Runtime: {convert_runtime(row['runtime'])}")
                    # st.write(f"Adult: {format_adult(row['adult'])}")
                    # st.markdown("<br>", unsafe_allow_html=True)
                    # st.markdown("<br>", unsafe_allow_html=True)
                    # st.write("---")
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
            with st.container():
                col1, col2 = st.columns([1, 2])  # Adjust the ratio if needed
                with col1:
                    st.image(row['poster_url'], caption=row['title'], width=205.5, use_column_width=False)
                    # st.write("---")
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Add space above movie name
                    st.write(f"**{idx + 1}. {row['title']}**")
                    st.markdown(
                        f"<p style='font-style: italic; color: gray;'>\"{row['tagline']}\"</p>",
                        unsafe_allow_html=True
                    )
                    st.write(f"Release Date: {row['release_date'].strftime('%Y-%m-%d')}")
                    st.write(f"Rating: {row['vote_average']}")
                    st.write(f"Runtime: {convert_runtime(row['runtime'])}")
                    # st.write(f"Adult: {format_adult(row['adult'])}")
                    # st.markdown("<br>", unsafe_allow_html=True)
                    # st.markdown("<br>", unsafe_allow_html=True)
                    # st.write("---")
            st.write("---")

# Display recommendations if a movie title is selected
if movie_title:
    compliment = get_random_compliment()
    st.info(compliment)
    display_recommendations(movie_title)
else:
    display_default_recommendations()
