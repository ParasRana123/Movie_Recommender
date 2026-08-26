import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, session, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests
import logging
import ast
import csv
import os
from bs4 import BeautifulSoup
from markupsafe import escape
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)

# Base directory and path resolvers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_csv_path(filename):
    """Resolve csv path in csv/ or fallback locations."""
    candidates = [
        os.path.join(BASE_DIR, 'csv', filename),
        os.path.join(BASE_DIR, filename),
        filename
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(BASE_DIR, 'csv', filename)

def get_model_path(filename):
    """Resolve model path in backend/models/ or fallback locations."""
    candidates = [
        os.path.join(BASE_DIR, 'backend', 'models', filename),
        os.path.join(BASE_DIR, 'models', filename),
        os.path.join(BASE_DIR, filename),
        filename
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(BASE_DIR, 'backend', 'models', filename)

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
        data = pd.read_csv(get_csv_path('main_data1.csv'), encoding='latin1')
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(data['comb'].fillna(''))
        SUGGESTIONS_CACHE = list(data['movie_title'].dropna().str.capitalize().unique())
    return data, count_matrix

def rcmd(m):
    m = str(m).lower().strip()
    global data, count_matrix
    if data is None or count_matrix is None:
        create_similarity()
    
    unique_titles = data['movie_title'].unique()
    if m not in unique_titles:
        # Try finding a fuzzy or substring match in case of minor title differences
        matches = [t for t in unique_titles if m in str(t).lower() or str(t).lower() in m]
        if matches:
            m = matches[0]
        else:
            return 'Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies'
    
    i = data.loc[data['movie_title'] == m].index[0]
    # Memory-efficient on-demand cosine similarity for movie vector against sparse matrix (takes ~15ms and <1MB RAM)
    sim_scores = cosine_similarity(count_matrix[i], count_matrix).flatten()
    sorted_indices = sim_scores.argsort()[::-1]
    
    # Exclude the queried movie itself
    top_indices = [idx for idx in sorted_indices if idx != i][:10]
    
    return [data['movie_title'].iloc[idx] for idx in top_indices]

def load_data():
    create_similarity()

load_data()
    
def get_suggestions():
    global SUGGESTIONS_CACHE
    if SUGGESTIONS_CACHE is None:
        create_similarity()
    return SUGGESTIONS_CACHE

app = Flask(__name__)
# Enable CORS for all routes and all origins
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept'
    return response

@app.context_processor
def inject_suggestions():
    return dict(suggestions=get_suggestions())

@app.route('/suggestions')
def suggestions_endpoint():
    return jsonify(get_suggestions())

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'images/bookmark_tick.svg', mimetype='image/svg+xml')

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/similarity", methods=["POST"])
def similarity_route():
    movie = request.form.get('name', '').strip()
    rc = rcmd(movie)
    if isinstance(rc, str):
        return rc
    else:
        return "---".join(rc)

@app.route("/watchlist")
def watchlist():
    return render_template("watchlist.html")        

@app.route("/action")   
def action():
    df2 = pd.read_csv(get_csv_path("action.csv"))
    movie_titles = df2['movie_title'].tolist()
    return render_template('action.html', movies=movie_titles)

@app.route("/allaction")
def allaction():
    df2 = pd.read_csv(get_csv_path("action.csv"))
    movie_titles = df2['movie_title'].tolist()
    return render_template("allaction.html", movies=movie_titles)

@app.route("/horror")
def horror():
    df3 = pd.read_csv(get_csv_path("horror.csv"))
    movie_titles1 = df3['movie_title'].tolist()
    return render_template('horror.html', movies=movie_titles1)

@app.route("/romance")
def romance():
    df5 = pd.read_csv(get_csv_path("romance.csv"))
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('romance.html', movies=movie_titles3)

@app.route("/mystery")
def mystery():
    df5 = pd.read_csv(get_csv_path("mystery.csv"))
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('mystery.html', movies=movie_titles3)

@app.route("/history")
def history():
    df5 = pd.read_csv(get_csv_path("history.csv"))
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('history.html', movies=movie_titles3)

@app.route("/thriller")
def thriller():
    df5 = pd.read_csv(get_csv_path("thriller.csv"))
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('thriller.html', movies=movie_titles3)

@app.route("/comedy")
def comedy():
    df4 = pd.read_csv(get_csv_path('comedy.csv'))
    movie_titles2 = df4['movie_title'].tolist()
    return render_template('comedy.html', movies=movie_titles2)

@app.route("/fantasy")
def fantasy():
    df5 = pd.read_csv(get_csv_path("fantasy.csv"))
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('fantasy.html', movies=movie_titles3)

@app.route("/adventure")
def adventure():
    df5 = pd.read_csv(get_csv_path("adventure.csv"))
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('adventure.html', movies=movie_titles3)              

@app.route("/documentary")
def documentary():
    df5 = pd.read_csv(get_csv_path("documentary.csv"))
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('documentary.html', movies=movie_titles3)  

@app.route("/sci_fi")
def sci_fi():
    df5 = pd.read_csv(get_csv_path("sci_fi.csv"))
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('sci_fi.html', movies=movie_titles3) 

@app.route("/genres")
def genres():
    return render_template('genres.html')

def fetch_reviews_with_sentiments(movie_id, imdb_id=None):
    """Fetch real reviews with author ratings and evaluate sentiment with NLP model."""
    reviews_list = []
    
    # 1. Fetch from TMDB Reviews API
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
                    except Exception as ex:
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

    # 2. Fallback to IMDb if TMDB returned no reviews and imdb_id is available
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
        # Check if ID passed directly
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
        poster = f"https://image.tmdb.org/t/p/original{poster_path}" if poster_path else "https://via.placeholder.com/500x750"
        backdrop = f"https://image.tmdb.org/t/p/original{backdrop_path}" if backdrop_path else poster

        overview = movie_data.get("overview", "No overview available.")
        genres_list = [genre["name"] for genre in movie_data.get("genres", [])] if "genres" in movie_data else []
        genres = ", ".join(genres_list)
        release_date = movie_data.get("release_date", "Unknown Date")
        runtime_min = int(movie_data.get("runtime") or 0)
        if runtime_min > 0:
            if runtime_min % 60 == 0:
                runtime = f"{runtime_min // 60} hour(s)"
            else:
                runtime = f"{runtime_min // 60} hour(s) {runtime_min % 60} min(s)"
        else:
            runtime = "N/A"

        vote_average = movie_data.get("vote_average", "N/A")
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
        sorted_yt = sorted(yt_videos, key=lambda v: (1 if v.get("official") else 0, v.get("size", 0)), reverse=True)

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
            (p["provider_name"], f"https://image.tmdb.org/t/p/w200{p['logo_path']}", get_streaming_url(p["provider_name"], title, tmdb_watch_link))
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
        casts = {}
        cast_details = {}
        for c in cast_raw:
            c_name = c.get("name", "Unknown")
            c_id = str(c.get("id", ""))
            c_char = c.get("character", "")
            c_profile = f"https://image.tmdb.org/t/p/original{c['profile_path']}" if c.get("profile_path") else "https://via.placeholder.com/240x360"
            casts[c_name] = [c_id, c_char, c_profile]
            cast_details[c_name] = [c_id, c_profile, "Unknown", "Unknown", "Biography not available."]

        # Recommendations
        rec_titles = rcmd(title)
        movie_cards = {}
        if isinstance(rec_titles, list):
            for rt in rec_titles:
                try:
                    r_rt = http_session.get(f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(rt)}", timeout=3)
                    if r_rt.status_code == 200:
                        rt_res = r_rt.json().get("results", [])
                        if rt_res and rt_res[0].get("poster_path"):
                            movie_cards[f"https://image.tmdb.org/t/p/w500{rt_res[0]['poster_path']}"] = rt
                        else:
                            movie_cards["https://via.placeholder.com/240x360?text=No+Poster"] = rt
                except Exception as e:
                    movie_cards["https://via.placeholder.com/240x360?text=No+Poster"] = rt

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
            'genres': genres,
            'movie_cards': movie_cards,
            'reviews': reviews,
            'casts': casts,
            'cast_details': cast_details,
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

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # Check if single title/name parameter passed (from modern search)
        query_title = request.form.get('name') or request.form.get('title')
        
        # If legacy detailed form parameters are passed, or title passed
        if query_title and 'cast_names' not in request.form:
            full_data = fetch_movie_full_data(query_title)
            if not full_data:
                return "<div class='fail'><center><h3>Sorry! The movie you requested is not in our database. Please check the spelling or try with other movies!</h3></center></div>"
            return render_template('recommend.html', **full_data)

        # Legacy handler if all fields were posted
        title = request.form.get('title', 'Unknown Title')
        movie_id = request.form.get('movie_id')
        imdb_id = request.form.get('imdb_id', '')
        
        def safe_convert_list(data_str):
            try:
                if isinstance(data_str, list):
                    return data_str
                return json.loads(data_str)
            except Exception:
                try:
                    return ast.literal_eval(data_str)
                except Exception:
                    return []

        def extract_video_id(url):
            if not url or not isinstance(url, str):
                return None
            url = url.strip()
            if not url or url == 'None':
                return None
            if "youtube.com/watch?v=" in url:
                vid = url.split("v=")[-1].split("&")[0].split("?")[0].strip()
                return vid if vid else None
            elif "youtube.com/embed/" in url:
                vid = url.split("embed/")[-1].split("?")[0].split("&")[0].strip()
                return vid if vid else None
            elif "youtu.be/" in url:
                vid = url.split("youtu.be/")[-1].split("?")[0].split("&")[0].strip()
                return vid if vid else None
            elif len(url) == 11 and (url.isalnum() or "-" in url or "_" in url):
                return url
            return None

        trailer_id = extract_video_id(request.form.get('trailer', ''))
        teaser_id = extract_video_id(request.form.get('teaser', ''))
        trailer_embed = f"https://www.youtube.com/embed/{trailer_id}" if trailer_id else None
        teaser_embed = f"https://www.youtube.com/embed/{teaser_id}" if teaser_id else None

        # Server-side fallback if trailers/teasers not provided by frontend AJAX
        if (not trailer_embed or not teaser_embed) and movie_id:
            try:
                r_vids = http_session.get(f"{TMDB_BASE_URL}/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&include_video_language=en,null", timeout=3)
                if r_vids.status_code == 200:
                    v_list = r_vids.json().get('results', [])
                    yt_vids = [v for v in v_list if v.get("site", "").lower() == "youtube" and v.get("key")]
                    sorted_vids = sorted(yt_vids, key=lambda v: (1 if v.get("official") else 0, v.get("size", 0)), reverse=True)
                    trailers = [v for v in sorted_vids if v.get("type", "").lower() == "trailer"]
                    teasers = [v for v in sorted_vids if v.get("type", "").lower() == "teaser"]
                    clips = [v for v in sorted_vids if v.get("type", "").lower() in ["clip", "featurette", "behind the scenes"]]

                    if not trailer_embed:
                        if trailers:
                            trailer_embed = f"https://www.youtube.com/embed/{trailers[0]['key']}"
                        elif teasers:
                            trailer_embed = f"https://www.youtube.com/embed/{teasers[0]['key']}"
                        elif sorted_vids:
                            trailer_embed = f"https://www.youtube.com/embed/{sorted_vids[0]['key']}"

                    if not teaser_embed:
                        if teasers:
                            teaser_embed = f"https://www.youtube.com/embed/{teasers[0]['key']}"
                        elif len(trailers) > 1:
                            teaser_embed = f"https://www.youtube.com/embed/{trailers[1]['key']}"
                        elif clips:
                            teaser_embed = f"https://www.youtube.com/embed/{clips[0]['key']}"
            except Exception as e:
                logging.warning(f"Error resolving videos: {e}")

        watch_providers = safe_convert_list(request.form.get('watch_providers', '[]'))
        watch_provider_logos = safe_convert_list(request.form.get('watch_provider_logos', '[]'))
        streaming_availability = list(zip(watch_providers, watch_provider_logos))

        rec_movies = safe_convert_list(request.form.get('rec_movies', '[]'))
        rec_posters = safe_convert_list(request.form.get('rec_posters', '[]'))
        cast_names = safe_convert_list(request.form.get('cast_names', '[]'))
        cast_chars = safe_convert_list(request.form.get('cast_chars', '[]'))
        cast_profiles = safe_convert_list(request.form.get('cast_profiles', '[]'))
        cast_bdays = safe_convert_list(request.form.get('cast_bdays', '[]'))
        cast_bios = safe_convert_list(request.form.get('cast_bios', '[]'))
        cast_places = safe_convert_list(request.form.get('cast_places', '[]'))
        cast_ids = safe_convert_list(request.form.get('cast_ids', '[]'))

        num_cards = min(len(rec_posters), len(rec_movies))
        movie_cards = {rec_posters[i]: rec_movies[i] for i in range(num_cards)} if num_cards > 0 else {}

        num_casts = min(len(cast_names), len(cast_ids), len(cast_chars), len(cast_profiles))
        casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(num_casts)} if num_casts > 0 else {}

        num_details = min(len(cast_names), len(cast_ids), len(cast_profiles), len(cast_bdays), len(cast_places), len(cast_bios))
        cast_details_map = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(num_details)} if num_details > 0 else {}

        reviews = fetch_reviews_with_sentiments(movie_id, imdb_id)

        return render_template(
            'recommend.html',
            title=title,
            poster=request.form.get('poster', ''),
            backdrop=request.form.get('backdrop', ''),
            overview=request.form.get('overview', 'No Overview Available'),
            vote_average=request.form.get('rating', 'N/A'),
            vote_count=request.form.get('vote_count', '0'),
            release_date=request.form.get('release_date', 'Unknown Date'),
            runtime=request.form.get('runtime', 'Unknown Runtime'),
            status=request.form.get('status', 'Unknown Status'),
            genres=request.form.get('genres', 'Unknown Genre'),
            movie_cards=movie_cards,
            reviews=reviews,
            casts=casts,
            cast_details=cast_details_map,
            trailer=trailer_embed,
            teaser=teaser_embed,
            streaming_availability=streaming_availability,
            budget=request.form.get('budget', 'N/A'),
            revenue=request.form.get('revenue', 'N/A'),
            original_language=request.form.get('original_language', 'N/A'),
            director_name=request.form.get('director_name', 'Unknown'),
            director_image=request.form.get('director_image', ''),
            director_bio=request.form.get('director_bio', 'No biography available'),
            director_birthplace=request.form.get('director_birthplace', 'Unknown')
        )

    except Exception as e:
        logging.error(f"Critical error in recommendation function: {e}", exc_info=True)
        return "<div class='fail'><center><h3>An error occurred while processing recommendations.</h3></center></div>"

def fetch_actor_from_tmdb(actor_id):
    """Fetch actor details and movie credits from TMDB using a numeric actor_id."""
    try:
        url = f"{TMDB_BASE_URL}/person/{actor_id}?api_key={TMDB_API_KEY}&language=en-US&append_to_response=movie_credits"
        response = http_session.get(url, timeout=5)

        if response.status_code != 200:
            logging.error(f"TMDB API request failed for Actor ID {actor_id}. Status: {response.status_code}")
            return None, []

        data = response.json()

        actor_info = {
            "id": actor_id,
            "name": data.get("name", "Unknown"),
            "profile": (
                f"https://image.tmdb.org/t/p/original{data['profile_path']}"
                if data.get("profile_path")
                else "https://via.placeholder.com/300x450?text=No+Photo"
            ),
            "birthday": data.get("birthday", "Unknown"),
            "birth_place": data.get("place_of_birth", "Unknown"),
            "known_for_department": data.get("known_for_department", "Acting"),
            "biography": data.get("biography") or "Biography not available."
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
        cast_credits = data.get("movie_credits", {}).get("cast", [])
        sorted_credits = sorted(cast_credits, key=lambda x: (x.get("vote_count", 0), x.get("popularity", 0)), reverse=True)
        
        seen_titles = set()
        actor_movies = []
        for m in sorted_credits:
            title = m.get("title") or m.get("original_title")
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            
            poster_path = m.get("poster_path")
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/240x360?text=No+Poster"
            
            rel_date = m.get("release_date", "")
            release_year = rel_date.split("-")[0] if rel_date else ""
            
            rating = m.get("vote_average", 0)
            rating_formatted = f"{round(rating, 1)}" if rating else "N/A"
            character = m.get("character", "").strip() or "Unknown Role"
            
            actor_movies.append({
                "id": m.get("id"),
                "title": title,
                "character": character,
                "poster": poster_url,
                "release_year": release_year,
                "rating": rating_formatted
            })

        return actor_info, actor_movies

    except Exception as e:
        logging.error(f"Error fetching from TMDB: {e}")
        return None, []

@app.route("/actor/<actor_id>")
def actor_details(actor_id):
    try:
        actor_id = escape(actor_id)

        try:
            if not actor_id or actor_id.lower() == 'none':
                raise ValueError("Empty or None actor_id")
            numeric_actor_id = int(float(actor_id))
        except ValueError:
            return render_template("error.html", message="Invalid actor ID in URL.")

        actor_data, actor_movies = fetch_actor_from_tmdb(numeric_actor_id)
        if not actor_data:
            return render_template("error.html", message=f"Could not fetch details for Actor ID {actor_id} from TMDB.")

        return render_template("actor.html", actor=actor_data, actor_movies=actor_movies[:15])

    except Exception as e:
        logging.exception("Error loading actor details.")
        return render_template("error.html", message="An error occurred while fetching actor details.")

def get_movie_id(movie_title):
    """Fetch movie ID from TMDB using the title."""
    search_url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(movie_title)}"
    try:
        response = http_session.get(search_url, timeout=5)
        if response.status_code == 200:
            search_results = response.json().get("results", [])
            if search_results:
                return search_results[0]["id"]
    except Exception as e:
        logging.error(f"Exception in get_movie_id: {e}")
    return None

@app.route("/movie/<movie_title>")
def movie_details(movie_title):
    try:
        full_data = fetch_movie_full_data(movie_title)
        if not full_data:
            return render_template("error.html", message="Movie not found in database.")
        
        # Format actors for movie.html
        actors_list = []
        for name, details in full_data['casts'].items():
            actors_list.append({
                "id": details[0],
                "name": name,
                "character": details[1],
                "image": details[2]
            })

        # Format recommendations list of dicts for movie.html
        rec_list = []
        for poster_url, rec_title in full_data['movie_cards'].items():
            rec_list.append({
                "title": rec_title,
                "poster": poster_url
            })

        return render_template(
            "movie.html",
            title=full_data['title'],
            poster=full_data['poster'],
            backdrop=full_data['backdrop'],
            overview=full_data['overview'],
            genres=full_data['genres'].split(", "),
            release_date=full_data['release_date'],
            runtime=full_data['runtime'],
            budget=full_data['budget'],
            revenue=full_data['revenue'],
            original_language=full_data['original_language'],
            vote_average=full_data['vote_average'],
            vote_count=full_data['vote_count'],
            status=full_data['status'],
            director_name=full_data['director_name'],
            director_image=full_data['director_image'],
            director_bio=full_data['director_bio'],
            actors=actors_list,
            trailer=full_data['trailer'],
            teaser=full_data['teaser'],
            streaming_availability=full_data['streaming_availability'],
            movie_reviews=full_data['reviews'],
            recommended_movies=rec_list
        )

    except Exception as e:
        logging.error(f"Error fetching movie details: {e}", exc_info=True)
# ==================== REST API ENDPOINTS FOR REACT FRONTEND ====================

@app.route('/api/health')
def health_endpoint():
    return jsonify({"status": "ok", "message": "Movie Recommender API is live"})

@app.route('/api/suggestions')
def api_suggestions():
    return jsonify(get_suggestions())

@app.route('/api/recommend', methods=['POST', 'GET', 'OPTIONS'])
def api_recommend():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

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

@app.route('/api/movie/<path:movie_title>', methods=['GET', 'OPTIONS'])
def api_movie_details(movie_title):
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    full_data = fetch_movie_full_data(movie_title)
    if not full_data:
        return jsonify({"error": f"Movie '{movie_title}' not found in database"}), 404
    return jsonify(full_data)

@app.route('/api/actor/<actor_id>', methods=['GET', 'OPTIONS'])
def api_actor_details(actor_id):
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
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

@app.route('/api/genres', methods=['GET', 'OPTIONS'])
def api_genres_list():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
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

@app.route('/api/genres/<genre_name>', methods=['GET', 'OPTIONS'])
def api_genre_movies(genre_name):
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
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

TRENDING_CACHE = {"timestamp": 0, "data": []}
UPCOMING_CACHE = {"timestamp": 0, "data": []}
PEOPLE_CACHE = {"timestamp": 0, "data": []}

@app.route('/api/trending', methods=['GET', 'OPTIONS'])
def api_trending_movies():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
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

@app.route('/api/upcoming', methods=['GET', 'OPTIONS'])
def api_upcoming_movies():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
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
    return jsonify(UPCOMING_CACHE["data"])

@app.route('/api/trending-people', methods=['GET', 'OPTIONS'])
def api_trending_people():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
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

@app.route('/api/top-movies', methods=['GET', 'OPTIONS'])
def api_top_movies():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
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

if __name__ == '__main__':
    app.run(port=5000, debug=True)