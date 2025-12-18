import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import requests

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from preprocessing import Preprocessor
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.hybrid import HybridRecommender
from evaluation import precision_recall_at_k
import re

# -----------------
# UTILS
# -----------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_poster(imdb_id, tmdb_id=None):
    """
    Fetches the movie poster from TMDB (preferred) or IMDb.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    }

    # 1. Try TMDB (Better quality posters usually)
    if tmdb_id and str(tmdb_id).lower() not in ['nan', 'none', '0', '']:
        try:
            tmdb_id_str = str(tmdb_id).split('.')[0]
            url = f"https://www.themoviedb.org/movie/{tmdb_id_str}"
            response = requests.get(url, headers=headers, timeout=1.5)
            if response.status_code == 200:
                match = re.search(r'<meta property="og:image" content="(.*?)"', response.text)
                if match:
                    image_url = match.group(1)
                    if "w600_and_h900_bestv2" in image_url: # High quality TMDB pattern
                         return image_url
                    if "http" in image_url:
                        return image_url
        except Exception:
            pass

    # 2. Fallback to IMDb
    if imdb_id and str(imdb_id).lower() not in ['nan', 'none', '0', '']:
        try:
            imdb_id_str = str(imdb_id).split('.')[0].zfill(7)
            url = f"https://www.imdb.com/title/tt{imdb_id_str}/"
            response = requests.get(url, headers=headers, timeout=2)
            if response.status_code == 200:
                match = re.search(r'<meta property="og:image" content="(.*?)"', response.text)
                if match:
                    return match.group(1)
        except Exception:
            pass
        
    return None

# Set Page Config
st.set_page_config(
    page_title="CinemAI - Advanced Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Global Base */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.9)), 
                          url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=2070&auto=format&fit=crop');
        background-attachment: fixed;
        background-size: cover;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 15, 20, 0.85);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Titles & Headings */
    h1, h2, h3 {
        color: #fff !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        text-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    h1 {
        background: linear-gradient(90deg, #ff4b4b, #ff914d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #ff4b4b 0%, #d62f2f 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 28px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-transform: uppercase;
        font-size: 0.9rem;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 20px rgba(255, 75, 75, 0.4);
    }

    /* Metric Cards Glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 75, 75, 0.5);
    }
    .metric-value {
        font-size: 2.8em;
        font-weight: 800;
        background: linear-gradient(120deg, #00CC96, #00ffbc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.9em;
        color: #b0b0b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }

    /* Inputs/Selectboxes */
    .stSelectbox>div>div {
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: white !important;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Movie Card Container */
    .movie-card-container {
        background: rgba(20, 20, 25, 0.9);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
    }
    .movie-card-container:hover {
        border-color: #ff4b4b;
        transform: scale(1.01);
    }
</style>
""", unsafe_allow_html=True)

# -----------------
# DATA LOADING & CACHING
# -----------------
@st.cache_resource
def load_and_prep_data():
    dl = DataLoader()
    movies, ratings, tags = dl.load_data()
    
    pre = Preprocessor(movies, ratings, tags)
    content_df = pre.prepare_content_data()
    collab_df = pre.prepare_collaborative_data()
    
    return movies, ratings, tags, content_df, collab_df

@st.cache_resource
def train_models(_content_df, _collab_df, _movies, _ratings):
    # Content Based
    cb_model = ContentBasedRecommender()
    cb_model.fit(_content_df)
    
    # Collaborative
    cf_model = CollaborativeRecommender()
    cf_model.fit(_collab_df, algorithm='svd')
    
    # Hybrid
    hybrid_model = HybridRecommender(cb_model, cf_model, _movies, _ratings)
    
    return cb_model, cf_model, hybrid_model

try:
    with st.spinner('Initializing AI Core... Loading Dataset & Training Models (this may take a minute)...'):
        movies, ratings, tags, content_df, collab_df = load_and_prep_data()
        cb_model, cf_model, hybrid_model = train_models(content_df, collab_df, movies, ratings)
except Exception as e:
    st.error(f"Critical Error during initialization: {e}")
    st.stop()

# -----------------
# SIDEBAR
# -----------------
st.sidebar.title("🎬 CinemAI")
st.sidebar.image("https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80", use_column_width=True)
nav_choice = st.sidebar.radio("Navigation", ["Dashboard", "Hybrid Recommendations", "Find Similar Movies", "Model Analytics"])

st.sidebar.markdown("---")
st.sidebar.info(f"**Data Stats**\n\nMovies: {len(movies):,}\nRatings: {len(ratings):,}\nUsers: {ratings['userId'].nunique():,}")

# -----------------
# PAGES
# -----------------

if nav_choice == "Dashboard":
    # Hero Section - Pure CSS
    st.markdown("""
    <div style="text-align: center; padding: 60px 0;">
        <h1 style="font-size: 4em; margin-bottom: 0;">CINEM<span style="color:#ff4b4b">AI</span></h1>
        <p style="font-size: 1.5em; color: #a0a0a0; font-weight: 300;">Discover your next obsession with Neural Recommendation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">Hybrid</div><p class="metric-label">Neural Engine</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(movies):,}</div><p class="metric-label">Movies Indexed</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(ratings):,}</div><p class="metric-label">User Interactions</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Ecosystem Analytics")
    
    # Interactive Data Vis
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("#### Rating Distribution")
        chart_data = ratings.groupby('rating').count()['userId'].reset_index()
        fig = px.bar(chart_data, x='rating', y='userId', labels={'userId': 'Votes', 'rating': 'Stars'}, color_discrete_sequence=['#ff4b4b'])
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.markdown("#### Popular Genres")
        # Quick genre process
        all_genres = movies['genres'].str.split('|').explode()
        genre_counts = all_genres.value_counts().head(10).reset_index()
        genre_counts.columns = ['Genre', 'Count']
        fig2 = px.pie(genre_counts, values='Count', names='Genre', color_discrete_sequence=px.colors.sequential.RdBu)
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig2, use_container_width=True)

elif nav_choice == "Hybrid Recommendations":
    st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>For You</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Based on your viewing history and similarity patterns</p>", unsafe_allow_html=True)
    
    # User Input
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        user_ids = sorted(ratings['userId'].unique())
        selected_user = st.selectbox("Select User Profile", user_ids)
    
    if st.button("✨ Generate Personalized Feed", type="primary"):
        with st.spinner(f"Processing Neural Factors for User {selected_user}..."):
            # Get History
            user_history = ratings[ratings['userId'] == selected_user].merge(movies, on='movieId').sort_values('rating', ascending=False).head(3)
            
            st.markdown("### 🕒 Recently Liked")
            hist_cols = st.columns(3)
            for i, (_, row) in enumerate(user_history.iterrows()):
                with hist_cols[i]:
                    imdb_id = str(row.get('imdbId', '')).split('.')[0]
                    tmdb_id = row.get('tmdbId', '')
                    poster_url = fetch_poster(imdb_id, tmdb_id)
                    
                    if poster_url:
                        st.image(poster_url, width=150)
                    else:
                        safe_title = row['title'].replace(" ", "+")
                        st.image(f"https://placehold.co/300x450/202020/FFF?text={safe_title}", width=150)
                        
                    st.caption(f"**{row['title']}**\n\n⭐ {row['rating']}")
            
            # Get Recs
            recs = hybrid_model.recommend(selected_user, top_k=10)
            
            st.markdown("---")
            st.markdown("---")
            st.markdown("### 🍿 Top Picks for Movie Night")
            
            # Custom Grid Layout
            for i, (idx, row) in enumerate(recs.iterrows()):
                # Use the new premium container class
                st.markdown('<div class="movie-card-container">', unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([1, 4, 2])
                
                # Simulated Poster with improved styling in CSS
                with c1:
                    imdb_id = str(row.get('imdbId', '')).split('.')[0]
                    tmdb_id = row.get('tmdbId', '')
                    poster_url = fetch_poster(imdb_id, tmdb_id)
                    
                    if poster_url:
                        st.image(poster_url, width=120)
                    else:
                        safe_title = row['title'].replace(" ", "+")
                        # Placeholder with a darker, sleeker default
                        st.image(f"https://placehold.co/400x600/1e1e1e/888?text={safe_title}", width=120) 
                
                # Info
                with c2:
                    st.markdown(f"<h3 style='margin:0; padding:0; color:white;'>{i+1}. {row['title']}</h3>", unsafe_allow_html=True)
                    st.markdown(f"#### `{row['genres'].replace('|', ' • ')}`")
                    st.markdown(f"_{row['explanation']}_")
                    
                    rating_score = row.get('avg_rating', 0)
                    st.progress(min(rating_score / 5.0, 1.0))
                    
                # Action
                with c3:
                    st.caption(f"Algorithm: {row['source'].upper()}")
                    imdb_id = str(row.get('imdbId', '')).split('.')[0]
                    
                    if imdb_id and imdb_id != '0':
                         st.markdown(f"[:link: View Details](https://www.imdb.com/title/tt{imdb_id.zfill(7)}/)")
                    
                    if st.button("Add to List", key=f"btn_add_{i}"):
                        st.toast(f"✅ Added {row['title']} to your Watchlist")

                st.markdown('</div>', unsafe_allow_html=True)

elif nav_choice == "Find Similar Movies":
    st.markdown("<h1 style='text-align: center; color: #4b9eff;'>Content Discovery</h1>", unsafe_allow_html=True)
    
    col_search, _ = st.columns([1, 2])
    with col_search:
        movie_list = movies['title'].values
        selected_movie = st.selectbox("Find movies similar to...", movie_list)
    
    if st.button("🔍 Analyze Similarity"):
        results = cb_model.recommend(selected_movie, top_k=8)
        
        if len(results) == 0:
            st.warning("No recommendations found. Try another movie.")
        else:
            st.markdown(f"### Because you liked *{selected_movie}*")
            
            # Grid of 4
            for i in range(0, len(results), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i + j < len(results):
                        row = results.iloc[i + j]
                        with cols[j]:
                            safe_title = row['title'].replace(" ", "+")
                            imdb_id = str(row.get('imdbId', '')).split('.')[0]
                            tmdb_id = row.get('tmdbId', '')
                            poster_url = fetch_poster(imdb_id, tmdb_id)
                            
                            if poster_url:
                                st.image(poster_url, use_container_width=True)
                            else:
                                st.image(f"https://placehold.co/300x450/202020/FFF?text={safe_title}", use_container_width=True)
                                
                            st.caption(row['title'])
                            st.progress(min(row['similarity_score'], 1.0))

elif nav_choice == "Model Analytics":
    st.title("📈 Model Performance Evaluation")
    
    if st.button("Run Evaluation (SVD)"):
        with st.spinner("Running Cross-Validation on SVD Model..."):
            metrics = cf_model.evaluate(collab_df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE (Root Mean Squared Error)", f"{metrics['RMSE']:.4f}")
            with col2:
                st.metric("MAE (Mean Absolute Error)", f"{metrics['MAE']:.4f}")
            
            st.success("Evaluation Complete. Low RMSE indicates high predictive accuracy.")
            
            st.markdown("### Accuracy Metrics Interpretation")
            st.info("""
            **RMSE**: Standard deviation of the residuals (prediction errors). Lower is better.
            **MAE**: Average absolute difference between predicted and actual rating.
            """)
