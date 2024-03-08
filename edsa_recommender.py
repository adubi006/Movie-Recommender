"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

st.set_page_config(page_title="SMARTAI", page_icon="::", layout="wide")

# Function to style headers and subheaders in red
def style_text_in_red(text):
    return f'<span style="color: red;">{text}</span>'


# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","SMARTai"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        # st.write('# Movie Recommender Engine')
        # st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "SMARTai":
        st.title("About SMART Solutions")
        st.write("We are a team of smart, dedicated and motivated data scientist\
                    with the drive to profer updated solutions and modern \
                    approach to solving problems across different insdustries")
        st.write("What we have here is a recommendaion system, and this solution\
                    is implemented in Streaming Industry, E-commerce and lots\
                    more")
    #------------------------------------------------------------------------
    #--------------------Dataset info----------------------------------------


        st.title("Welcome to SMART Recommender")

        def load_data():
            df_movies = pd.read_csv('resources/data/movies.csv')
            df_ratings = pd.read_csv('resources/data/ratings.csv')
            return df_movies, df_ratings

        df_movies, df_ratings = load_data()

        st.write("A snick pick into the data we are working with")

        st.header("Movies Dataset")
        st.dataframe(df_movies.head())

        st.header("Ratings Dataset")
        st.dataframe(df_ratings.head())

        # We merge the movies and ratings dataset using the 'movieId' column as the common column in both dataframes
        df_merged = pd.merge(df_ratings, df_movies, on='movieId')

        st.title('Top 10 Movies by Total Ratings and Genres')
        popularity_df = df_ratings.groupby('movieId')['rating'].count().reset_index()
        popularity_df.rename(columns={'rating': 'total_ratings'}, inplace=True)
        popularity_df = popularity_df.merge(df_movies[['movieId', 'title', 'genres']], on='movieId', how='left')
        popularity_df = popularity_df.sort_values(by='total_ratings', ascending=False)

        # Set the number of top movies to display
        top_n_movies = 10

        # Sort the dataframe in descending order based on total_ratings
        top_movies_df = popularity_df.sort_values(by='total_ratings', ascending=False).head(top_n_movies)

        # Create a bar plot to visualize the top-rated movies
        plt.figure(figsize=(12, 6))
        sns.barplot(x='total_ratings', y='title', data=top_movies_df, palette='viridis')
        plt.title(f"Top {top_n_movies} Movies by Total Ratings")
        plt.xlabel("Total Ratings")
        plt.ylabel("Movie Title")
        plt.tight_layout()  # To adjust spacing between the plot elements
        st.pyplot(plt)  # Display the Matplotlib plot in Streamlit

        st.title('Top 20 User Preferences by Genre: Distribution of Ratings')

        # Genre Preferences
        genre_preferences = df_ratings.merge(df_movies[['movieId', 'genres']], on='movieId', how='left')
        genre_preferences = genre_preferences.groupby('genres')['rating'].count().reset_index()
        genre_preferences.rename(columns={'rating': 'total_ratings'}, inplace=True)

        # Select the top twenty genres based on total ratings
        top_twenty_genres = genre_preferences.nlargest(20, 'total_ratings')

        plt.figure(figsize=(12, 6))
        sns.barplot(x='genres', y='total_ratings', data=top_twenty_genres, palette='viridis')
        plt.title("Top 20 User Preferences by Genre: Distribution of Ratings")
        plt.xlabel("Genre")
        plt.ylabel("Total Ratings")
        plt.xticks(rotation=45, ha='right')  # Rotate genre labels for better readability
        plt.tight_layout()  # Adjust plot layout to prevent label overlap
        plt.tight_layout()  # To adjust spacing between the plot elements
        st.pyplot(plt)  # Display the Matplotlib plot in Streamlit

    # or to provide your business pitch.
        # Plot the rating distribution using Seaborn
        with sns.axes_style('white'):
            g = sns.catplot(data=df_ratings, x="rating", y=None, aspect=2.0, kind='count')
            g.set_ylabels("Total number of ratings")

        # Display the Seaborn plot in Streamlit using st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False) n 
        st.title("Rating Distribution")
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False) n
        # Calculate and display the average rating in the Streamlit app
        average_rating = np.mean(df_ratings["rating"])
        st.write(f'Average rating in dataset: {average_rating:.2f}')

# Load the 'df_ratings' and 'df_movies' DataFrames (you can replace this with your data loading code)
        df_ratings = pd.read_csv('resources/data/ratings.csv')
        df_movies = pd.read_csv('resources/data/movies.csv')

        # Merge the DataFrames
        df_merged = pd.merge(df_ratings, df_movies, on='movieId')

        # Calculate the total ratings for each movie
        popularity_df = df_ratings.groupby('movieId')['rating'].count().reset_index()
        popularity_df.rename(columns={'rating': 'total_ratings'}, inplace=True)
        popularity_df = popularity_df.merge(df_movies[['movieId', 'title', 'genres']], on='movieId', how='left')
        popularity_df = popularity_df.sort_values(by='total_ratings', ascending=False)

        # Calculate the percentage of movies with ratings of 3 and above
        total_movies = len(popularity_df)
        movies_with_rating_3_or_above = len(popularity_df[popularity_df['total_ratings'] >= 3])
        percentage_movies_with_rating_3_or_above = (movies_with_rating_3_or_above / total_movies) * 100

        # Display the percentage in the Streamlit app
        st.title("Percentage of Movies with Ratings of 3 and Above")
        st.write(f"{percentage_movies_with_rating_3_or_above:.2f}%")


if __name__ == '__main__':
    # Apply background image styling
    bg_img = """
    <style>
    /* Background image for the main container */
    [data-testid="stAppViewContainer"] {
        background-image: url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=870&q=80');
        background-size: cover;
        background-position: top center;
    }

    /* Transparent background for the header */
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
    }

    /* Move the sidebar to the right and apply background image */
    [data-testid="stSidebar"] {
        right: 2rem;
        background-image: url('https://images.unsplash.com/photo-1518676590629-3dcbd9c5a5c9?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=387&q=80');
        background-size: cover;
        background-position: top;
    }
    </style>
    """
    st.markdown(bg_img, unsafe_allow_html=True)

    # Apply red text styling for headers and subheaders
    header_html = f'<h1 style="color: red;">Movie Recommender Engine</h1>'
    subheader_html = f'<h3 style="color: red;">EXPLORE Data Science Academy Unsupervised Predict</h3>'
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown(subheader_html, unsafe_allow_html=True)
    main()
