import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

class CollaborativeRecommender:
    """
    Collaborative Filtering using Scikit-Learn (SVD & Cosine Similarity).
    Replaced Scikit-Surprise for better portability.
    """
    def __init__(self):
        self.matrix = None
        self.user_ids = None
        self.movie_ids = None
        self.model = None
        self.user_map = None
        self.item_map = None
        self.algorithm = None
        self.reconstructed_matrix = None

    def fit(self, ratings_df, algorithm='svd'):
        """
        Fits the collaborative filtering model.
        algorithm: 'svd'
        """
        self.algorithm = algorithm
        print(f"Training Collaborative Model ({algorithm})...")
        
        # Create User-Item Matrix
        # Pivot table: Rows=Users, Cols=Movies
        self.matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.user_ids = self.matrix.index
        self.movie_ids = self.matrix.columns
        
        # Mappings for quick lookup
        self.user_map = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_map = {mid: i for i, mid in enumerate(self.movie_ids)}
        
        if algorithm == 'svd':
            # Matrix Factorization using TruncatedSVD
            # n_components is the number of latent factors
            n_components = min(20, self.matrix.shape[1] - 1)
            self.model = TruncatedSVD(n_components=n_components, random_state=42)
            self.model.fit(self.matrix)
            
            # Pre-compute reconstructed matrix for fast prediction
            # U * Sigma * Vt
            self.reconstructed_matrix = self.model.inverse_transform(self.model.transform(self.matrix))
            
            # Map back to DataFrame for easier access if needed, requires memory but okay for small dataset
            self.reconstructed_df = pd.DataFrame(
                self.reconstructed_matrix, 
                index=self.matrix.index, 
                columns=self.matrix.columns
            )
            
        print("Collaborative Model Trained.")

    def evaluate(self, ratings_df):
        """
        Performs a distinct train-test split evaluation.
        Note: We can't easily do standard split on matrix with SVD without masking.
        We will simulate by masking some ratings in the test set.
        """
        train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
        
        # Train on train_df
        self.fit(train_df, algorithm='svd')
        
        # Predict on test_df
        y_true = []
        y_pred = []
        
        for _, row in test_df.iterrows():
            uid = row['userId']
            mid = row['movieId']
            rating = row['rating']
            
            if uid in self.reconstructed_df.index and mid in self.reconstructed_df.columns:
                pred = self.reconstructed_df.loc[uid, mid]
                y_true.append(rating)
                y_pred.append(pred)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Re-fit on full dataset
        self.fit(ratings_df, algorithm='svd')
        
        return {'RMSE': rmse, 'MAE': mae}

    def predict(self, user_id, movie_id):
        """
        Predicts the rating a user would give to a movie.
        """
        if self.algorithm == 'svd':
            if user_id in self.reconstructed_df.index and movie_id in self.reconstructed_df.columns:
                return self.reconstructed_df.loc[user_id, movie_id]
        return 0

    def recommend_for_user(self, user_id, all_movie_ids, top_k=10):
        """
        Generates top_k recommendations for a given user.
        """
        if self.algorithm == 'svd':
            if user_id not in self.reconstructed_df.index:
                return []
            
            # Get all predicted ratings for this user
            user_preds = self.reconstructed_df.loc[user_id]
            
            # Get already rated items (non-zero)
            # self.matrix has 0 for missing, so we exclude if > 0
            user_actual = self.matrix.loc[user_id]
            already_rated = user_actual[user_actual > 0].index
            
            # Filter predictions
            filtered_preds = user_preds.drop(labels=already_rated, errors='ignore')
            
            # Sort
            recommendations = filtered_preds.sort_values(ascending=False).head(top_k)
            
            # Return list of tuples
            return list(recommendations.items())
        
        return []
