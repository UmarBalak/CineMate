import streamlit as st
from streamlit_option_menu import option_menu
from preprocess1 import get_recommendations, get_random_compliment, default_recommendation, convert_runtime, knn_model, tfidf_vectorizer, df, list
from preprocess2 import knn_model2, tfidf_vectorizer2, df2, list2
import time

#########################################
hide_st_style = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""
#########################################
# Function to display recommendations with movie posters
def display_recommendations(movie_title, knn_model, tfidf_vectorizer, df):
    start_time = time.time()
    recommendations = get_recommendations(movie_title, knn_model, df, tfidf_vectorizer)
    
    if isinstance(recommendations, str):  # Check if the return is an error message
        st.error(recommendations)
    else:
        st.subheader(f"Top 10 movie recommendations for '{movie_title}':")
        for idx, row in recommendations.head(10).iterrows():
            with st.container():
                col1, col2 = st.columns([1, 2])  # Adjust the ratio if needed
                with col1:
                    st.image(row['poster_url'], caption=row['title'], width=205.5, use_column_width=False)
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Add space above movie name
                    st.write(f"**{idx + 1}. {row['title']}**")
                    try:
                        st.markdown(
                            f"<p style='font-style: italic; color: gray;'>\"{row['tagline']}\"</p>",
                            unsafe_allow_html=True
                        )
                    except:
                        pass
                    st.write(f"Release Date: {row['release_date'].strftime('%Y-%m-%d')}")
                    st.write(f"Runtime: {convert_runtime(row['runtime'])}")
            st.write("---")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

def display_recommendations2(movie_title, knn_model, tfidf_vectorizer, df):
    start_time = time.time()
    recommendations = get_recommendations(movie_title, knn_model, df, tfidf_vectorizer)
    
    if isinstance(recommendations, str):  # Check if the return is an error message
        st.error(recommendations)
    else:
        st.subheader(f"Top 10 movie recommendations for '{movie_title}':")
        for idx, row in recommendations.head(10).iterrows():
            with st.container():
                col1, col2 = st.columns([1, 2])  # Adjust the ratio if needed
                with col1:
                    st.image(row['poster_url'], caption=row['title'], width=205.5, use_column_width=False)
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Add space above movie name
                    st.write(f"**{idx + 1}. {row['title']}**")
                    st.write(f"Release Date: {row['release_year']}")   
                    st.write(f"Runtime: {row['duration']}")
                    st.write(f"Cast: {row['cast']}")
            st.write("---")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

def display_default_recommendations(list, df):
    recommendations = default_recommendation(list, df)
    if isinstance(recommendations, str):  # Check if the return is an error message
        st.error(recommendations)
    else:
        st.subheader(f"Our Top 10 Recommendations:")
        for idx, row in recommendations.head(10).iterrows():
            with st.container():
                col1, col2 = st.columns([1, 2])  # Adjust the ratio if needed
                with col1:
                    st.image(row['poster_url'], caption=row['title'], width=205.5, use_column_width=False)
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
            st.write("---")

def display_default_recommendations2(list, df):
    recommendations = default_recommendation(list, df)
    if isinstance(recommendations, str):  # Check if the return is an error message
        st.error(recommendations)
    else:
        st.subheader(f"Our Top 10 Recommendations:")
        for idx, row in recommendations.head(10).iterrows():
            with st.container():
                col1, col2 = st.columns([1, 2])  # Adjust the ratio if needed
                with col1:
                    st.image(row['poster_url'], caption=row['title'], width=205.5, use_column_width=False)
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Add space above movie name
                    st.write(f"**{idx + 1}. {row['title']}**")
                    st.write(f"Release Date: {row['release_year']}")   
                    st.write(f"Runtime: {row['duration']}")
                    st.write(f"Cast: {row['cast']}")
            st.write("---")

# Main function to run the Streamlit app
def main():
    hide_st_style = """
                    <style>
                    #MainMenu {visibility:hidden;}
                    footer {visibility:hidden;}
                    header {visibility:hidden;}
                    </style>
                    """
    with st.sidebar:
        page = option_menu(
            menu_title= "Recommendation Systems",
            options=["General Movie Recommendations", "Netflix Movie Recommendations", "About"],
                        icons=['film', 'film', 'info-circle'],
                        menu_icon="cast", default_index=0)

    if page == "General Movie Recommendations":
        st.title("Movie Recommendation System")

        movie_title = st.selectbox(
            "Select your favorite movie:",
            options=df['title'].unique(),
            index=None
        )

        if movie_title:
            compliment = get_random_compliment()
            st.info(compliment)
            display_recommendations(movie_title, knn_model, tfidf_vectorizer, df)
        else:
            display_default_recommendations(list, df)

    elif page == "Netflix Movie Recommendations":
        st.title("Netflix Movie Recommendation System")

        # Placeholder for Netflix recommendations (assuming a similar function is available)
        netflix_movie_title = st.selectbox(
            "Select your favorite Netflix movie:",
            options=df2['title'].unique(),
            index=None
        )

        if netflix_movie_title:
            compliment = get_random_compliment()
            st.info(compliment)
            display_recommendations2(netflix_movie_title, knn_model2, tfidf_vectorizer2, df2)
        else:
            display_default_recommendations2(list2, df2)

    elif page == "About":
        st.title("About this App")
        st.write("""
            This Movie Recommendation System provides personalized movie recommendations based on your selected favorite movie.
            - **General Movie Recommendations**: Get top movie suggestions from our entire database.
            - **Netflix Movie Recommendations**: Specifically tailored recommendations from Netflix's collection.
            - **Compliments**: Enjoy a random compliment while you browse movies!
            
            Built using **Streamlit** and advanced **Machine Learning** techniques.
        """)

if __name__ == '__main__':
    main()
