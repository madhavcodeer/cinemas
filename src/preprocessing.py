import pandas as pd
import numpy as np

class Preprocessor:
    """
    Handles data preprocessing for recommendation models.
    """
    
    def __init__(self, movies, ratings, tags):
        self.movies = movies
        self.ratings = ratings
        self.tags = tags

    def prepare_content_data(self):
        """
        Prepares data for Content-Based Filtering.
        - Combines genres and tags into a single text feature string for each movie.
        """
        # Collapse tags per movie
        print("Preparing content data...")
        movie_tags = self.tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.dropna().astype(str))).reset_index()
        
        # Merge tags with movies
        df = pd.merge(self.movies, movie_tags, on='movieId', how='left')
        df['tag'] = df['tag'].fillna('')
        
        # Clean genres (remove separators like | and replace with space)
        df['genres_str'] = df['genres'].str.replace('|', ' ')
        
        # Create a "soup" column
        df['soup'] = df['genres_str'] + ' ' + df['tag']
        
        # Clean text (simple lowercasing)
        df['soup'] = df['soup'].str.lower().str.strip()
        
        return df[['movieId', 'title', 'genres', 'soup', 'imdbId', 'tmdbId', 'avg_rating']]

    def prepare_collaborative_data(self, min_ratings_user=5, min_ratings_movie=5):
        """
        Filters data for Collaborative Filtering to ensure statistical significance.
        - Removes users with too few ratings.
        - Removes movies with too few ratings.
        """
        print("Preparing collaborative data...")
        counts_users = self.ratings['userId'].value_counts()
        counts_movies = self.ratings['movieId'].value_counts()
        
        valid_users = counts_users[counts_users >= min_ratings_user].index
        valid_movies = counts_movies[counts_movies >= min_ratings_movie].index
        
        filtered_ratings = self.ratings[
            (self.ratings['userId'].isin(valid_users)) & 
            (self.ratings['movieId'].isin(valid_movies))
        ].copy()
        
        return filtered_ratings

    def get_sparse_matrix(self, ratings_df):
        """
        Returns a sparse user-item matrix (for custom implementations if needed).
        """
        user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        return user_item_matrix
