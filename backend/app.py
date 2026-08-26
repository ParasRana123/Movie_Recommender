import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
import requests
import logging
import os
from bs4 import BeautifulSoup
from markupsafe import escape
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def get_csv_path(filename):
    """Resolve csv path whether running from root, backend, or subfolders."""
    candidates = [
        os.path.join(ROOT_DIR, 'csv', filename),
        os.path.join(BASE_DIR, 'csv', filename),
        os.path.join(BASE_DIR, '..', 'csv', filename),
        os.path.join(ROOT_DIR, filename),
        os.path.join(BASE_DIR, filename),
        filename
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
def get_model_path(filename):
    """Resolve model path whether running from root, backend, or subfolders."""
    candidates = [
        os.path.join(BASE_DIR, 'models', filename),
        os.path.join(ROOT_DIR, 'backend', 'models', filename),
        os.path.join(ROOT_DIR, 'models', filename),
        os.path.join(BASE_DIR, filename),
        filename
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(BASE_DIR, 'models', filename)

# Load the NLP model and TF-IDF vectorizer from disk
clf = pickle.load(open(get_model_path('nlp_model2.pkl'), 'rb'))
vectorizer = pickle.load(open(get_model_path('transform1.pkl'), 'rb'))

TMDB_API_KEY = "fce0af3409e6113c9b3c75aaf49341bb"
TMDB_BASE_URL = "https://api.tmdb.org/3"

data = None
count_matrix = None
http_session = requests.Session()
SUGGESTIONS_CACHE = None

def create_similarity():
    global data, count_matrix, SUGGESTIONS_CACHE
    if data is None or count_matrix is None:
        csv_path = get_csv_path('main_data1.csv')
        data = pd.read_csv(csv_path, encoding='latin1')
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(data['comb'].fillna(''))
        SUGGESTIONS_CACHE = list(data['movie_title'].dropna().str.capitalize().unique())
    return data, count_matrix

def load_data():
    create_similarity()

load_data()

def get_suggestions():
    global SUGGESTIONS_CACHE
    if SUGGESTIONS_CACHE is None:
        create_similarity()
    return SUGGESTIONS_CACHE

def rcmd(m):
    m = str(m).lower().strip()
    global data, count_matrix
    if data is None or count_matrix is None:
        create_similarity()
    
    unique_titles = data['movie_title'].unique()
    if m not in unique_titles:
        # Try finding a substring match
        matches = [t for t in unique_titles if m in str(t).lower() or str(t).lower() in m]
        if matches:
            m = matches[0]
        else:
            return None
    
    i = data.loc[data['movie_title'] == m].index[0]
    # Memory-efficient on-demand cosine similarity for movie vector against sparse matrix (takes ~15ms and <1MB RAM)
    sim_scores = cosine_similarity(count_matrix[i], count_matrix).flatten()
    sorted_indices = sim_scores.argsort()[::-1]
    
    # Exclude the queried movie itself
    top_indices = [idx for idx in sorted_indices if idx != i][:10]
    
    return [data['movie_title'].iloc[idx] for idx in top_indices]

def fetch_reviews_with_sentiments(movie_id, imdb_id=None):
    """Fetch reviews with author ratings and evaluate sentiment with NLP model."""
    reviews_list = []
    
    # 1. TMDB Reviews API
    if movie_id:
        try:
            url = f"{TMDB_BASE_URL}/movie/{movie_id}/reviews?api_key={TMDB_API_KEY}"
            r = http_session.get(url, timeout=4)
            if r.status_code == 200:
                results = r.json().get('results', [])
                for rev in results:
                    content = rev.get('content', '').strip()
                    if not content:
                        continue
                    author = rev.get('author') or 'Anonymous'
                    author_details = rev.get('author_details', {})
                    raw_rating = author_details.get('rating')
                    
                    try:
                        movie_vector = vectorizer.transform([content])
                        pred = clf.predict(movie_vector)[0]
                        pred_prob = clf.predict_proba(movie_vector)[0]
                        is_good = bool(pred == 1 or pred == 'Good' or pred)
                        sentiment_label = 'Good' if is_good else 'Bad'
                        confidence = f"{round(pred_prob[1 if is_good else 0] * 100, 1)}%"
                    except Exception:
                        sentiment_label = 'Good' if (raw_rating and raw_rating >= 6.0) else 'Bad'
                        confidence = "85.0%"

                    if raw_rating is not None:
                        rating_display = f"{raw_rating}/10"
                    else:
                        rating_display = "8.0/10" if sentiment_label == 'Good' else "5.0/10"

                    reviews_list.append({
                        'author': author,
                        'rating': rating_display,
                        'sentiment': sentiment_label,
                        'confidence': confidence,
                        'content': content
                    })
        except Exception as e:
            logging.warning(f"Error fetching TMDB reviews: {e}")

    # 2. Fallback to IMDb if TMDB returned no reviews
    if not reviews_list and imdb_id:
        try:
            url = f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            r = http_session.get(url, headers=headers, timeout=3)
            if r.status_code == 200:
                soup = BeautifulSoup(r.content, 'lxml')
                for div in soup.find_all("div", {"class": "ipc-html-content-inner-div"}):
                    text = div.text.strip()
                    if text:
                        movie_vector = vectorizer.transform([text])
                        pred = clf.predict(movie_vector)[0]
                        pred_prob = clf.predict_proba(movie_vector)[0]
                        is_good = bool(pred == 1 or pred == 'Good' or pred)
                        reviews_list.append({
                            'author': 'IMDb Reviewer',
                            'rating': '8.0/10' if is_good else '5.0/10',
                            'sentiment': 'Good' if is_good else 'Bad',
                            'confidence': f"{round(pred_prob[1 if is_good else 0] * 100, 1)}%",
                            'content': text
                        })
        except Exception as e:
            logging.warning(f"Error scraping IMDb reviews: {e}")

    return reviews_list

def fetch_movie_full_data(movie_title_or_id):
    """Fetch complete movie details, credits, videos, recommendations, and reviews."""
    try:
        movie_id = None
        if str(movie_title_or_id).isdigit():
            movie_id = int(movie_title_or_id)
        else:
            search_url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(str(movie_title_or_id))}"
            r_search = http_session.get(search_url, timeout=5)
            if r_search.status_code == 200:
                results = r_search.json().get('results', [])
                if results:
                    movie_id = results[0]['id']
        
        if not movie_id:
            return None

        # Fetch movie details with videos, credits, watch providers
        movie_url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=videos,credits,watch/providers"
        r_movie = http_session.get(movie_url, timeout=5)
        if r_movie.status_code != 200:
            return None
        
        movie_data = r_movie.json()

        title = movie_data.get("title") or movie_data.get("original_title", "Unknown Title")
        poster_path = movie_data.get("poster_path")
        backdrop_path = movie_data.get("backdrop_path")
        poster = f"https://image.tmdb.org/t/p/original{poster_path}" if poster_path else "https://via.placeholder.com/500x750?text=No+Poster"
        backdrop = f"https://image.tmdb.org/t/p/original{backdrop_path}" if backdrop_path else poster

        overview = movie_data.get("overview", "No overview available.")
        genres_list = [genre["name"] for genre in movie_data.get("genres", [])] if "genres" in movie_data else []
        genres_str = ", ".join(genres_list)
        release_date = movie_data.get("release_date", "Unknown Date")
        runtime_min = int(movie_data.get("runtime") or 0)
        if runtime_min > 0:
            if runtime_min % 60 == 0:
                runtime = f"{runtime_min // 60} hour(s)"
            else:
                runtime = f"{runtime_min // 60} hour(s) {runtime_min % 60} min(s)"
        else:
            runtime = "N/A"

        vote_average = f"{round(float(movie_data.get('vote_average', 0)), 1)}" if movie_data.get('vote_average') else "N/A"
        vote_count = f"{int(movie_data.get('vote_count', 0)):,}"
        status = movie_data.get("status", "Released")
        imdb_id = movie_data.get("imdb_id", "")
        budget = f"{int(movie_data.get('budget', 0)):,}" if movie_data.get('budget') else "N/A"
        revenue = f"{int(movie_data.get('revenue', 0)):,}" if movie_data.get('revenue') else "N/A"
        original_language = movie_data.get("original_language", "EN").upper()

        # Videos: Robust Trailer & Teaser Extraction
        videos = movie_data.get("videos", {}).get("results", [])
        if not videos:
            try:
                r_vids = http_session.get(f"{TMDB_BASE_URL}/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&include_video_language=en,null,{original_language.lower()}", timeout=3)
                if r_vids.status_code == 200:
                    videos = r_vids.json().get('results', [])
            except Exception:
                videos = []

        yt_videos = [v for v in videos if v.get("site", "").lower() == "youtube" and v.get("key")]
        
        # Sort: official first, then highest resolution, latest
        sorted_yt = sorted(
            yt_videos,
            key=lambda v: (1 if v.get("official") else 0, v.get("size", 0)),
            reverse=True
        )

        trailers = [v for v in sorted_yt if v.get("type", "").lower() == "trailer"]
        teasers = [v for v in sorted_yt if v.get("type", "").lower() == "teaser"]
        clips = [v for v in sorted_yt if v.get("type", "").lower() in ["clip", "featurette", "behind the scenes", "opening credits"]]

        trailer_key = None
        if trailers:
            trailer_key = trailers[0]['key']
        elif teasers:
            trailer_key = teasers[0]['key']
        elif sorted_yt:
            trailer_key = sorted_yt[0]['key']

        teaser_key = None
        if teasers:
            teaser_key = teasers[0]['key']
        elif len(trailers) > 1:
            teaser_key = trailers[1]['key']
        elif clips:
            teaser_key = clips[0]['key']

        trailer = f"https://www.youtube.com/embed/{trailer_key}" if trailer_key else None
        teaser = f"https://www.youtube.com/embed/{teaser_key}" if teaser_key else None

        # Helper to generate streaming service redirect URL
        def get_streaming_url(provider_name, title, fallback_link=""):
            p = (provider_name or "").lower().strip()
            encoded = requests.utils.quote(str(title).strip())
            if "netflix" in p:
                return f"https://www.netflix.com/search?q={encoded}"
            elif "prime" in p or "amazon" in p:
                return f"https://www.primevideo.com/search/ref=atv_nb_sr?phrase={encoded}"
            elif "disney" in p or "hotstar" in p:
                return f"https://www.hotstar.com/in/search?q={encoded}"
            elif "apple" in p or "itunes" in p:
                return f"https://tv.apple.com/search?term={encoded}"
            elif "hulu" in p:
                return f"https://www.hulu.com/search?q={encoded}"
            elif "hbo" in p or "max" in p:
                return f"https://www.max.com/search?q={encoded}"
            elif "jio" in p:
                return f"https://www.jiocinema.com/search/{encoded}"
            elif "zee" in p:
                return f"https://www.zee5.com/search?q={encoded}"
            elif "sonyliv" in p or "sony" in p:
                return f"https://www.sonyliv.com/search?q={encoded}"
            elif "peacock" in p:
                return f"https://www.peacocktv.com/search?q={encoded}"
            elif "paramount" in p:
                return f"https://www.paramountplus.com/search/?q={encoded}"
            elif "youtube" in p or "google" in p:
                return f"https://www.youtube.com/results?search_query={encoded}+movie"
            elif "crunchyroll" in p:
                return f"https://www.crunchyroll.com/search?q={encoded}"
            elif fallback_link:
                return fallback_link
            else:
                return f"https://www.google.com/search?q=watch+{encoded}+on+{requests.utils.quote(provider_name)}"

        # Watch Providers
        watch_results = movie_data.get("watch/providers", {}).get("results", {})
        providers_data = watch_results.get("IN", {}).get("flatrate", [])
        tmdb_watch_link = watch_results.get("IN", {}).get("link", "")
        if not providers_data:
            providers_data = watch_results.get("US", {}).get("flatrate", [])
            if not tmdb_watch_link:
                tmdb_watch_link = watch_results.get("US", {}).get("link", "")

        streaming_availability = [
            {
                "provider_name": p["provider_name"],
                "logo_path": f"https://image.tmdb.org/t/p/w200{p['logo_path']}",
                "watch_url": get_streaming_url(p["provider_name"], title, tmdb_watch_link)
            }
            for p in providers_data if p.get("provider_name") and p.get("logo_path")
        ]

        # Director
        director = next((crew for crew in movie_data.get("credits", {}).get("crew", []) if crew.get("job") == "Director"), {})
        director_name = director.get("name", "Unknown")
        director_id = director.get("id")
        director_image = f"https://image.tmdb.org/t/p/w300{director.get('profile_path')}" if director.get('profile_path') else "https://via.placeholder.com/300"
        director_bio = "Biography not available."
        director_birthplace = "Unknown"

        if director_id:
            try:
                r_dir = http_session.get(f"{TMDB_BASE_URL}/person/{director_id}?api_key={TMDB_API_KEY}", timeout=3)
                if r_dir.status_code == 200:
                    d_data = r_dir.json()
                    director_bio = d_data.get("biography") or director_bio
                    director_birthplace = d_data.get("place_of_birth") or director_birthplace
            except Exception as e:
                logging.warning(f"Error fetching director bio: {e}")

        # Top Cast (top 10)
        cast_raw = movie_data.get("credits", {}).get("cast", [])[:10]
        casts = []
        for c in cast_raw:
            casts.append({
                "id": c.get("id", ""),
                "name": c.get("name", "Unknown"),
                "character": c.get("character", ""),
                "profile": f"https://image.tmdb.org/t/p/original{c['profile_path']}" if c.get("profile_path") else "https://via.placeholder.com/240x360?text=No+Photo"
            })

        # Recommendations via ML model
        rec_titles = rcmd(title)
        recommended_movies = []
        if isinstance(rec_titles, list):
            for rt in rec_titles:
                try:
                    r_rt = http_session.get(f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(rt)}", timeout=3)
                    if r_rt.status_code == 200:
                        rt_res = r_rt.json().get("results", [])
                        if rt_res and rt_res[0].get("poster_path"):
                            recommended_movies.append({
                                "title": rt,
                                "poster": f"https://image.tmdb.org/t/p/w500{rt_res[0]['poster_path']}",
                                "vote_average": f"{round(float(rt_res[0].get('vote_average', 0)), 1)}" if rt_res[0].get('vote_average') else "N/A"
                            })
                        else:
                            recommended_movies.append({
                                "title": rt,
                                "poster": "https://via.placeholder.com/240x360?text=No+Poster",
                                "vote_average": "N/A"
                            })
                except Exception:
                    recommended_movies.append({
                        "title": rt,
                        "poster": "https://via.placeholder.com/240x360?text=No+Poster",
                        "vote_average": "N/A"
                    })

        # Reviews with ratings and NLP sentiments
        reviews = fetch_reviews_with_sentiments(movie_id, imdb_id)

        return {
            'movie_id': movie_id,
            'title': title,
            'poster': poster,
            'backdrop': backdrop,
            'overview': overview,
            'vote_average': vote_average,
            'vote_count': vote_count,
            'release_date': release_date,
            'runtime': runtime,
            'status': status,
            'genres': genres_list,
            'genres_str': genres_str,
            'recommended_movies': recommended_movies,
            'reviews': reviews,
            'casts': casts,
            'trailer': trailer,
            'teaser': teaser,
            'streaming_availability': streaming_availability,
            'budget': budget,
            'revenue': revenue,
            'original_language': original_language,
            'director_name': director_name,
            'director_image': director_image,
            'director_bio': director_bio,
            'director_birthplace': director_birthplace
        }

    except Exception as e:
        logging.error(f"Error fetching full movie data: {e}", exc_info=True)
        return None

def fetch_actor_from_tmdb(actor_id):
    """Fetch actor details and movie credits from TMDB using a numeric actor_id."""
    try:
        url = f"{TMDB_BASE_URL}/person/{actor_id}?api_key={TMDB_API_KEY}&language=en-US&append_to_response=movie_credits"
        response = http_session.get(url, timeout=5)

        if response.status_code != 200:
            logging.error(f"TMDB API request failed for Actor ID {actor_id}. Status: {response.status_code}")
            return None, []

        data_json = response.json()

        actor_info = {
            "id": actor_id,
            "name": data_json.get("name", "Unknown"),
            "profile": (
                f"https://image.tmdb.org/t/p/original{data_json['profile_path']}"
                if data_json.get("profile_path")
                else "https://via.placeholder.com/300x450?text=No+Photo"
            ),
            "birthday": data_json.get("birthday", "Unknown"),
            "birth_place": data_json.get("place_of_birth", "Unknown"),
            "known_for_department": data_json.get("known_for_department", "Acting"),
            "biography": data_json.get("biography") or "Biography not available."
        }

        # Format birthday to friendly format
        if actor_info["birthday"] and actor_info["birthday"] != "Unknown":
            try:
                b_parts = actor_info["birthday"].split("-")
                months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                if len(b_parts) == 3:
                    month_name = months[int(b_parts[1]) - 1]
                    actor_info["birthday"] = f"{month_name} {b_parts[2]}, {b_parts[0]}"
            except Exception:
                pass

        # Extract movies from cast credits sorted by vote_count & popularity
        cast_credits = data_json.get("movie_credits", {}).get("cast", [])
        sorted_credits = sorted(cast_credits, key=lambda x: (x.get("vote_count", 0), x.get("popularity", 0)), reverse=True)
        
        seen_titles = set()
        actor_movies = []
        for m in sorted_credits:
            m_title = m.get("title") or m.get("original_title")
            if not m_title or m_title.lower() in seen_titles:
                continue
            seen_titles.add(m_title.lower())
            
            poster_path = m.get("poster_path")
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/240x360?text=No+Poster"
            
            rel_date = m.get("release_date", "")
            release_year = rel_date.split("-")[0] if rel_date else ""
            
            rating = m.get("vote_average", 0)
            rating_formatted = f"{round(rating, 1)}" if rating else "N/A"
            character = m.get("character", "").strip() or "Unknown Role"
            
            actor_movies.append({
                "id": m.get("id"),
                "title": m_title,
                "character": character,
                "poster": poster_url,
                "release_year": release_year,
                "rating": rating_formatted
            })

        return actor_info, actor_movies

    except Exception as e:
        logging.error(f"Error fetching actor from TMDB: {e}")
        return None, []

# ==================== API ROUTES ====================

@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "message": "Movie Recommender API is running"})

@app.route('/api/suggestions')
def suggestions_endpoint():
    return jsonify(get_suggestions())

@app.route('/api/recommend', methods=['POST', 'GET'])
def recommend_endpoint():
    query_title = request.args.get('title') or request.args.get('name') or request.args.get('movie') or request.args.get('query')
    if not query_title and request.is_json:
        req_data = request.get_json() or {}
        query_title = req_data.get('title') or req_data.get('name') or req_data.get('movie') or req_data.get('query')
    if not query_title and request.form:
        query_title = request.form.get('title') or request.form.get('name') or request.form.get('movie') or request.form.get('query')

    if not query_title:
        return jsonify({"error": "Movie title query parameter is required"}), 400

    full_data = fetch_movie_full_data(query_title)
    if not full_data:
        return jsonify({"error": f"Movie '{query_title}' not found in database"}), 404

    return jsonify(full_data)

@app.route('/api/movie/<path:movie_title>')
def movie_endpoint(movie_title):
    full_data = fetch_movie_full_data(movie_title)
    if not full_data:
        return jsonify({"error": f"Movie '{movie_title}' not found in database"}), 404
    return jsonify(full_data)

@app.route('/api/actor/<actor_id>')
def actor_endpoint(actor_id):
    try:
        actor_id_clean = escape(actor_id)
        numeric_actor_id = int(float(actor_id_clean))
        actor_info, actor_movies = fetch_actor_from_tmdb(numeric_actor_id)
        if not actor_info:
            return jsonify({"error": f"Actor ID '{actor_id}' not found"}), 404
        return jsonify({
            "actor": actor_info,
            "movies": actor_movies[:15]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/genres')
def genres_list():
    genres = [
        {"id": "action", "name": "Action", "image": "/images/action1.jpg", "banner": "/images/action.jpg", "description": "Fast-paced, thrilling sequences of physical feats, combat, and excitement."},
        {"id": "horror", "name": "Horror", "image": "/images/horror.jpg", "banner": "/images/horror1.jpg", "description": "Stories that aim to elicit fear, suspense, and a sense of dread with macabre and supernatural themes."},
        {"id": "romance", "name": "Romance", "image": "/images/Romance.jpg", "banner": "/images/romance1.jpg", "description": "Heartwarming and emotional stories celebrating the journey of love, connection, and relationships."},
        {"id": "mystery", "name": "Mystery", "image": "/images/mystery.jpg", "banner": "/images/mystery1.jpg", "description": "Suspenseful investigations, intricate plots, and thrilling enigmas waiting to be solved."},
        {"id": "history", "name": "History", "image": "/images/history.jpg", "banner": "/images/history1.jpg", "description": "Recounting historical events, civilization milestones, and iconic figures that shaped our world."},
        {"id": "thriller", "name": "Thriller", "image": "/images/Thriller.jpg", "banner": "/images/thriller1.jpg", "description": "Edge-of-your-seat excitement, suspenseful plots, and psychological tension."},
        {"id": "comedy", "name": "Comedy", "image": "/images/Comedy.jpg", "banner": "/images/comedy1.jpg", "description": "Lighthearted entertainment, wit, and humor crafted to bring laughter and joy."},
        {"id": "fantasy", "name": "Fantasy", "image": "/images/fantasy.jpg", "banner": "/images/fantasy1.jpg", "description": "Imaginative realms, magic, mythical creatures, and wondrous supernatural adventures."},
        {"id": "adventure", "name": "Adventure", "image": "/images/Adventure.jpg", "banner": "/images/adventure1.jpg", "description": "Epic quests, daring expeditions, and heroic feats in exotic environments."},
        {"id": "documentary", "name": "Documentary", "image": "/images/Documentary.jpg", "banner": "/images/documentary1.jpg", "description": "Factual stories, real-world issues, nature wonders, and thought-provoking insights."},
        {"id": "sci_fi", "name": "Sci-Fi", "image": "/images/Sci-Fi.jpg", "banner": "/images/sci-fi1.jpg", "description": "Futuristic concepts, space exploration, cutting-edge technology, and mind-bending speculative realities."}
    ]
    return jsonify(genres)

@app.route('/api/genres/<genre_name>')
def genre_movies(genre_name):
    clean_genre = genre_name.lower().replace("-", "_")
    csv_file = get_csv_path(f"{clean_genre}.csv")
    if not os.path.exists(csv_file):
        return jsonify({"error": f"Genre '{genre_name}' not found"}), 404

    try:
        df = pd.read_csv(csv_file)
        titles = df['movie_title'].dropna().tolist()[:25]
        
        # Enrich top 15 with TMDB posters & ratings
        enriched = []
        for t in titles[:15]:
            try:
                r_s = http_session.get(f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(str(t))}", timeout=2.5)
                if r_s.status_code == 200:
                    res = r_s.json().get('results', [])
                    if res and res[0].get('poster_path'):
                        enriched.append({
                            "title": t,
                            "poster": f"https://image.tmdb.org/t/p/w500{res[0]['poster_path']}",
                            "vote_average": f"{round(float(res[0].get('vote_average', 0)), 1)}" if res[0].get('vote_average') else "N/A"
                        })
                    else:
                        enriched.append({
                            "title": t,
                            "poster": "https://via.placeholder.com/240x360?text=No+Poster",
                            "vote_average": "N/A"
                        })
                else:
                    enriched.append({"title": t, "poster": "https://via.placeholder.com/240x360?text=No+Poster", "vote_average": "N/A"})
            except Exception:
                enriched.append({"title": t, "poster": "https://via.placeholder.com/240x360?text=No+Poster", "vote_average": "N/A"})

        return jsonify({
            "genre": genre_name,
            "movies": enriched
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/top-movies')
def top_movies():
    curated = [
        "Avatar", "The Dark Knight", "Inception", "Interstellar", 
        "Avengers: Endgame", "Titanic", "Gladiator", "Pulp Fiction", 
        "The Matrix", "Spider-Man: Into the Spider-Verse"
    ]
    movies_data = []
    for t in curated:
        try:
            r = http_session.get(f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(t)}", timeout=2.5)
            if r.status_code == 200:
                res = r.json().get('results', [])
                if res and res[0].get('poster_path'):
                    movies_data.append({
                        "title": t,
                        "poster": f"https://image.tmdb.org/t/p/w500{res[0]['poster_path']}",
                        "vote_average": f"{round(float(res[0].get('vote_average', 0)), 1)}"
                    })
        except Exception:
            pass
    return jsonify(movies_data)

TRENDING_CACHE = {"timestamp": 0, "data": []}
UPCOMING_CACHE = {"timestamp": 0, "data": []}

@app.route('/api/trending')
def trending_movies():
    global TRENDING_CACHE
    import time
    now = time.time()
    if TRENDING_CACHE["data"] and (now - TRENDING_CACHE["timestamp"] < 1800):
        return jsonify(TRENDING_CACHE["data"])
    try:
        r = http_session.get(f"{TMDB_BASE_URL}/trending/movie/week?api_key={TMDB_API_KEY}", timeout=4)
        if r.status_code == 200:
            results = r.json().get('results', [])
            formatted = [
                {
                    "id": m.get("id"),
                    "title": m.get("title"),
                    "poster": f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get('poster_path') else None,
                    "backdrop": f"https://image.tmdb.org/t/p/w780{m.get('backdrop_path')}" if m.get('backdrop_path') else None,
                    "rating": round(float(m.get("vote_average", 0)), 1),
                    "release_date": m.get("release_date"),
                    "overview": m.get("overview")
                }
                for m in results if m.get("title") and m.get("poster_path")
            ]
            if formatted:
                TRENDING_CACHE = {"timestamp": now, "data": formatted}
                return jsonify(formatted)
    except Exception as e:
        logging.error(f"Error fetching trending movies: {e}")
    return jsonify(TRENDING_CACHE["data"])

@app.route('/api/upcoming')
def upcoming_movies():
    global UPCOMING_CACHE
    import time
    now = time.time()
    if UPCOMING_CACHE["data"] and (now - UPCOMING_CACHE["timestamp"] < 1800):
        return jsonify(UPCOMING_CACHE["data"])
    try:
        r = http_session.get(f"{TMDB_BASE_URL}/movie/upcoming?api_key={TMDB_API_KEY}&language=en-US&page=1", timeout=4)
        if r.status_code == 200:
            results = r.json().get('results', [])
            formatted = [
                {
                    "id": m.get("id"),
                    "title": m.get("title"),
                    "poster": f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}" if m.get('poster_path') else None,
                    "backdrop": f"https://image.tmdb.org/t/p/w780{m.get('backdrop_path')}" if m.get('backdrop_path') else None,
                    "rating": round(float(m.get("vote_average", 0)), 1),
                    "release_date": m.get("release_date"),
                    "overview": m.get("overview")
                }
                for m in results if m.get("title") and m.get("poster_path")
            ]
            if formatted:
                UPCOMING_CACHE = {"timestamp": now, "data": formatted}
                return jsonify(formatted)
    except Exception as e:
        logging.error(f"Error fetching upcoming movies: {e}")
PEOPLE_CACHE = {"timestamp": 0, "data": []}

@app.route('/api/trending-people')
def trending_people():
    global PEOPLE_CACHE
    import time
    now = time.time()
    if PEOPLE_CACHE["data"] and (now - PEOPLE_CACHE["timestamp"] < 1800):
        return jsonify(PEOPLE_CACHE["data"])
    try:
        r = http_session.get(f"{TMDB_BASE_URL}/trending/person/week?api_key={TMDB_API_KEY}&language=en-US", timeout=4)
        if r.status_code == 200:
            results = r.json().get('results', [])
            formatted = [
                {
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "profile": f"https://image.tmdb.org/t/p/w500{p.get('profile_path')}" if p.get('profile_path') else "https://via.placeholder.com/250x250?text=No+Photo",
                    "known_for_department": p.get("known_for_department", "Acting"),
                    "known_for": [m.get("title") or m.get("name") for m in p.get("known_for", []) if m.get("title") or m.get("name")]
                }
                for p in results if p.get("name") and p.get("profile_path")
            ]
            if formatted:
                PEOPLE_CACHE = {"timestamp": now, "data": formatted}
                return jsonify(formatted)
    except Exception as e:
        logging.error(f"Error fetching trending people: {e}")
    return jsonify(PEOPLE_CACHE["data"])

if __name__ == '__main__':
    app.run(port=5000, debug=True)
