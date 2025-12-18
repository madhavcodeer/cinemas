import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    """
    Content-Based Filtering using TF-IDF on genres and tags.
    """
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.cosine_sim = None
        self.indices = None
        self.df = None

    def fit(self, content_df):
        """
        Fits the TF-IDF vectorizer and calculates cosine similarity matrix.
        """
        print("Training Content-Based Model...")
        self.df = content_df.reset_index(drop=True)
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()
        
        # Compute TF-IDF matrix
        tfidf_matrix = self.tfidf.fit_transform(self.df['soup'])
        
        # Compute Cosine Similarity
        # linear_kernel is equivalent to cosine_similarity for normalized vectors (TF-IDF is normalized)
        # Using linear_kernel is faster.
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        print("Content-Based Model Trained.")

    def recommend(self, title, top_k=10):
        """
        Returns recommendations based on a movie title.
        """
        if self.cosine_sim is None:
            raise Exception("Model not trained. Call fit() first.")
            
        if title not in self.indices:
            return []
            
        idx = self.indices[title]
        # Handle case where multiple movies have same title (take first)
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get scores of the top_k most similar movies (ignoring itself at 0)
        sim_scores = sim_scores[1:top_k+1]
        
        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        results = self.df.iloc[movie_indices].copy()
        results['similarity_score'] = scores
        
        # Add explanation
        results['explanation'] = results.apply(lambda x: f"High similarity in genres/tags ({x['similarity_score']:.2f})", axis=1)
        
        return results[['title', 'genres', 'similarity_score', 'explanation', 'imdbId', 'tmdbId', 'avg_rating']]
