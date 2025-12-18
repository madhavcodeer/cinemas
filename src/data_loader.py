import os
import requests
import zipfile
import io
import pandas as pd

class DataLoader:
    """
    Handles data downloading and loading for the MovieLens dataset.
    """
    DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.movies_path = os.path.join(data_dir, "ml-latest-small", "movies.csv")
        self.ratings_path = os.path.join(data_dir, "ml-latest-small", "ratings.csv")
        self.links_path = os.path.join(data_dir, "ml-latest-small", "links.csv")
        self.tags_path = os.path.join(data_dir, "ml-latest-small", "tags.csv")

    def download_data(self):
        """Downloads and extracts the MovieLens dataset if not already present."""
        if os.path.exists(self.movies_path) and os.path.exists(self.ratings_path):
            print("Data already exists.")
            return

        print(f"Downloading data from {self.DATA_URL}...")
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            r = requests.get(self.DATA_URL)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self.data_dir)
            print("Data downloaded and extracted successfully.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise

    def load_data(self):
        """
        Loads movies and ratings into pandas DataFrames.
        Preprocessing:
        - Merges movies with genres
        - Handles timestamps (optional conversion)
        
        Returns:
            movies (pd.DataFrame): DataFrame containing movieID, title, genres
            ratings (pd.DataFrame): DataFrame containing userId, movieId, rating, timestamp
        """
        self.download_data()
        
        try:
            movies = pd.read_csv(self.movies_path)
            ratings = pd.read_csv(self.ratings_path)
            tags = pd.read_csv(self.tags_path)
            links = pd.read_csv(self.links_path)
            
            # Basic preprocessing
            # Convert timestamp to datetime
            ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
            tags['timestamp'] = pd.to_datetime(tags['timestamp'], unit='s')
            
            # Calculate Average Rating and Count per Movie
            movie_stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
            movie_stats.columns = ['avg_rating', 'rating_count']
            movies = movies.merge(movie_stats, on='movieId', how='left')
            
            # Merge IMDb ID and TMDB ID
            movies = movies.merge(links[['movieId', 'imdbId', 'tmdbId']], on='movieId', how='left')
            
            # Fill NaN
            movies['avg_rating'] = movies['avg_rating'].fillna(0)
            movies['rating_count'] = movies['rating_count'].fillna(0)
            
            return movies, ratings, tags
        except FileNotFoundError:
            print("Error: specific CSV files not found after extraction.")
            raise

if __name__ == "__main__":
    dl = DataLoader()
    m, r, t = dl.load_data()
    print(f"Loaded {len(m)} movies and {len(r)} ratings.")
