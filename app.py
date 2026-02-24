import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Config ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

# --- Load Data ---
@st.cache_data
def load_data():
    movies_df = pd.read_csv("data/movies.csv")
    movies_df["genres"] = movies_df["genres"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["genres"])
    return movies_df, tfidf_matrix

movies_df, tfidf_matrix = load_data()

# --- Recommendation Logic ---
def get_recommendations(title, n=5):
    title_lower = title.lower()
    # Try exact match first
    matches = movies_df[movies_df["title"].str.lower() == title_lower]
    # If no match, try partial match
    if matches.empty:
        matches = movies_df[movies_df["title"].str.lower().str.contains(title_lower, regex=False)]
    if matches.empty:
        return None
    idx = matches.index[0]
    movie_vec = tfidf_matrix[idx]
    sim_scores = cosine_similarity(movie_vec, tfidf_matrix).flatten()
    sim_scores[idx] = -1
    top_indices = sim_scores.argsort()[::-1][:n]
    return movies_df[["title", "genres"]].iloc[top_indices].reset_index(drop=True)
# --- UI ---
st.title("Movie Recommender")
st.write("Enter a movie you like and get similar recommendations based on genre.")

movie_input = st.text_input("Movie title", placeholder="e.g. Toy Story (1995)")

n_results = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

if st.button("Get Recommendations"):
    if not movie_input.strip():
        st.warning("Please enter a movie title.")
    else:
        results = get_recommendations(movie_input.strip(), n=n_results)
        if results is None:
            st.error(f"Movie '{movie_input}' not found. Try including the year, e.g. 'Toy Story (1995)'")
        else:
            st.success(f"Movies similar to **{movie_input}**:")
            for i, row in results.iterrows():
                st.write(f"**{i+1}. {row['title']}** — _{row['genres'].replace('|', ', ')}_")

# --- Footer ---
st.markdown("---")
st.caption("Built with the MovieLens dataset · Content-based filtering using TF-IDF + Cosine Similarity")
