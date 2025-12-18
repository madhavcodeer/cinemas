import pandas as pd
import numpy as np

class HybridRecommender:
    """
    Intelligent Hybrid Recommendation Engine.
    Blends Collaborative Filtering (SVD) and Content-Based Filtering.
    """
    def __init__(self, content_model, collab_model, movies_df, ratings_df):
        self.content_model = content_model
        self.collab_model = collab_model
        self.movies = movies_df
        self.ratings = ratings_df
        
        # Precompute popularity for cold-start
        self.popular_movies = self._get_popular_movies()

    def _get_popular_movies(self, top_k=20):
        """
        Computes weighted rating (IMDB formula) or simple average to find popular movies.
        """
        # Calculate C and m
        C = self.ratings['rating'].mean()
        m = self.ratings.groupby('movieId').count()['rating'].quantile(0.9)
        
        q_movies = self.movies.copy()
        
        # Calculate v and R
        movie_stats = self.ratings.groupby('movieId').agg({'rating': ['count', 'mean']})
        movie_stats.columns = ['vote_count', 'vote_average']
        
        q_movies = q_movies.merge(movie_stats, on='movieId', how='left')
        q_movies = q_movies[q_movies['vote_count'] >= m]
        
        def weighted_rating(x, m=m, C=C):
            v = x['vote_count']
            R = x['vote_average']
            return (v/(v+m) * R) + (m/(m+v) * C)

        q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
        q_movies = q_movies.sort_values('score', ascending=False)
        return q_movies.head(top_k)

    def recommend(self, user_id, top_k=10):
        """
        Hybrid Recommendation Logic:
        1. Check if user exists in collaborative model.
        2. If yes, get SVD estimates.
        3. Also get Content-Based recommendations based on user's high-rated movies.
        4. Blend results.
        5. If no (Cold Start), return Popular movies.
        """
        # Cold Start Check
        if user_id not in self.ratings['userId'].unique():
            print(f"User {user_id} is new. Using Popularity-based fallback.")
            recs = self.popular_movies[['movieId', 'title', 'genres', 'score']].copy()
            recs['explanation'] = "Popular with other users (Cold Start Strategy)"
            return recs.head(top_k)

        # 1. Collaborative Filtering Candidates
        all_movie_ids = self.movies['movieId'].unique()
        collab_recs = self.collab_model.recommend_for_user(user_id, all_movie_ids, top_k=top_k*2)
        collab_df = pd.DataFrame(collab_recs, columns=['movieId', 'pred_rating'])
        collab_df = collab_df.merge(self.movies, on='movieId')
        collab_df['source'] = 'Collaborative'
        
        # 2. Content-Based Augmentation
        # Find user's favorite movie (highest rated)
        user_history = self.ratings[self.ratings['userId'] == user_id].sort_values('rating', ascending=False)
        
        content_recs_list = []
        if not user_history.empty:
            top_movie_id = user_history.iloc[0]['movieId']
            # Find title
            top_movie_title = self.movies[self.movies['movieId'] == top_movie_id]['title'].values
            if len(top_movie_title) > 0:
                title = top_movie_title[0]
                content_recs_df = self.content_model.recommend(title, top_k=top_k)
                if not content_recs_df.empty:
                    # Map back to IDs
                    content_recs_df = content_recs_df.merge(self.movies[['title', 'movieId']], on='title')
                    content_recs_df['source'] = 'Content-Based'
                    content_recs_df['pred_rating'] = content_recs_df['similarity_score'] * 5 # Normalize roughly to 5 scale
                    content_recs_list.append(content_recs_df)
        
        # 3. Blending
        # Combineframes
        final_recs = collab_df.copy()
        if content_recs_list:
            content_df = content_recs_list[0]
            # We can interleave or weighted average.
            # Simple strategy: Take top n from Collab, inject some from Content if not present.
            
            # Let's just append and sort by normalized score for simplicity in this MVP
            # But standard SVD is usually better than raw content similarity for "rating" prediction.
            # We use Content to ensure diversity or "explainability".
            
            # Strategy: Boost score of items that appear in both.
            # And include items from content based that are high confidence.
            
            # For this implementation, let's mix: 70% Collab, 30% Content
            final_recs = pd.concat([collab_df.head(7), content_df.head(3)])

        # Deduplicate (keep first occurrence which is likely higher score)
        final_recs = final_recs.drop_duplicates(subset=['movieId'])
        
        # Merge metadata (avg_rating, imdbId) if not present (Content based might miss them)
        if 'avg_rating' not in final_recs.columns:
            final_recs = final_recs.merge(self.movies[['movieId', 'avg_rating', 'imdbId']], on='movieId', how='left')
        else:
             # Ensure they are filled for all rows
             final_recs['avg_rating'] = final_recs['avg_rating'].fillna(0)
             final_recs['imdbId'] = final_recs['imdbId'].fillna(0)

        # Fill explanation if missing (for Collab)
        final_recs['explanation'] = final_recs.apply(
            lambda x: x['explanation'] if 'explanation' in x and pd.notna(x['explanation']) 
            else f"Predicted Rating: {x['pred_rating']:.1f}", axis=1)
            
        return final_recs.head(top_k)[['movieId', 'title', 'genres', 'explanation', 'source', 'avg_rating', 'imdbId']]
