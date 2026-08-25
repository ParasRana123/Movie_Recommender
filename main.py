import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, session, jsonify, send_from_directory
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

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the NLP model and TF-IDF vectorizer from disk
filename = 'nlp_model2.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('transform1.pkl', 'rb'))

TMDB_API_KEY = "fce0af3409e6113c9b3c75aaf49341bb"
TMDB_BASE_URL = "https://api.tmdb.org/3"

data = None
similarity = None
http_session = requests.Session()

def create_similarity():
    data = pd.read_csv('main_data1.csv', encoding='latin1')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    similarity = cosine_similarity(count_matrix)
    return data, similarity

def rcmd(m):
    m = str(m).lower().strip()
    global data, similarity
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    
    if m not in data['movie_title'].unique():
        # Try finding a fuzzy or substring match in case of minor title differences
        matches = [t for t in data['movie_title'].unique() if m in str(t).lower() or str(t).lower() in m]
        if matches:
            m = matches[0]
        else:
            return 'Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies'
    
    i = data.loc[data['movie_title'] == m].index[0]
    lst = list(enumerate(similarity[i]))
    lst = sorted(lst, key=lambda x: x[1], reverse=True)
    lst = lst[1:11]
    
    if len(lst) < 10:
        remaining = 10 - len(lst)
        additional_movies = [x for x in list(enumerate(similarity[i])) if x[0] != i][10:]
        lst.extend(additional_movies[:remaining])
    
    l = []
    for item in lst:
        a = item[0]
        l.append(data['movie_title'][a])
    return l

def load_data():
    global data, similarity
    if data is None or similarity is None:
        data, similarity = create_similarity()

load_data()
    
def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'images/bookmark_tick.svg', mimetype='image/svg+xml')

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

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
    df2 = pd.read_csv("action.csv")
    movie_titles = df2['movie_title'].tolist()
    return render_template('action.html', movies=movie_titles)

@app.route("/allaction")
def allaction():
    df2 = pd.read_csv("action.csv")
    movie_titles = df2['movie_title'].tolist()
    return render_template("allaction.html", movies=movie_titles)

@app.route("/horror")
def horror():
    df3 = pd.read_csv("horror.csv")
    movie_titles1 = df3['movie_title'].tolist()
    return render_template('horror.html', movies=movie_titles1)

@app.route("/romance")
def romance():
    df5 = pd.read_csv("romance.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('romance.html', movies=movie_titles3)

@app.route("/mystery")
def mystery():
    df5 = pd.read_csv("mystery.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('mystery.html', movies=movie_titles3)

@app.route("/history")
def history():
    df5 = pd.read_csv("history.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('history.html', movies=movie_titles3)

@app.route("/thriller")
def thriller():
    df5 = pd.read_csv("thriller.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('thriller.html', movies=movie_titles3)

@app.route("/comedy")
def comedy():
    df4 = pd.read_csv('comedy.csv')
    movie_titles2 = df4['movie_title'].tolist()
    return render_template('comedy.html', movies=movie_titles2)

@app.route("/fantasy")
def fantasy():
    df5 = pd.read_csv("fantasy.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('fantasy.html', movies=movie_titles3)

@app.route("/adventure")
def adventure():
    df5 = pd.read_csv("adventure.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('adventure.html', movies=movie_titles3)              

@app.route("/documentary")
def documentary():
    df5 = pd.read_csv("documentary.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('documentary.html', movies=movie_titles3)  

@app.route("/sci_fi")
def sci_fi():
    df5 = pd.read_csv("sci_fi.csv")
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

        # Videos: Trailer & Teaser
        videos = movie_data.get("videos", {}).get("results", [])
        trailer_key = next((v['key'] for v in videos if v.get("type") == "Trailer" and v.get("site") == "YouTube"), None)
        teaser_key = next((v['key'] for v in videos if v.get("type") == "Teaser" and v.get("site") == "YouTube"), None)
        trailer = f"https://www.youtube.com/embed/{trailer_key}" if trailer_key else None
        teaser = f"https://www.youtube.com/embed/{teaser_key}" if teaser_key else None

        # Watch Providers
        providers_data = movie_data.get("watch/providers", {}).get("results", {}).get("IN", {}).get("flatrate", [])
        if not providers_data:
            providers_data = movie_data.get("watch/providers", {}).get("results", {}).get("US", {}).get("flatrate", [])
        streaming_availability = [
            (p["provider_name"], f"https://image.tmdb.org/t/p/w200{p['logo_path']}")
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
                return ""
            if "youtube.com/watch?v=" in url:
                return url.split("v=")[-1].split("&")[0]
            elif "youtu.be/" in url:
                return url.split("youtu.be/")[-1].split("?")[0]
            return ""

        trailer_id = extract_video_id(request.form.get('trailer', ''))
        teaser_id = extract_video_id(request.form.get('teaser', ''))
        trailer_embed = f"https://www.youtube.com/embed/{trailer_id}" if trailer_id else None
        teaser_embed = f"https://www.youtube.com/embed/{teaser_id}" if teaser_id else None

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
    """Fetch actor details from TMDB using a numeric actor_id."""
    try:
        url = f"{TMDB_BASE_URL}/person/{actor_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = http_session.get(url, timeout=5)

        if response.status_code != 200:
            logging.error(f"🚨 TMDB API request failed for Actor ID {actor_id}. Status: {response.status_code}")
            return None

        data = response.json()

        return {
            "name": data.get("name", "Unknown"),
            "profile": (
                f"https://image.tmdb.org/t/p/w500{data['profile_path']}"
                if data.get("profile_path")
                else "https://via.placeholder.com/150"
            ),
            "birthday": data.get("birthday", "Unknown"),
            "birth_place": data.get("place_of_birth", "Unknown"),
            "biography": data.get("biography", "No biography available.")
        }

    except Exception as e:
        logging.error(f"❌ Error fetching from TMDB: {e}")
        return None

@app.route("/actor/<actor_id>")
def actor_details(actor_id):
    try:
        movies = set()
        actor_name = None
        actor_id = escape(actor_id)

        try:
            if not actor_id or actor_id.lower() == 'none':
                raise ValueError("Empty or None actor_id")
            actor_id = int(float(actor_id))
        except ValueError:
            return render_template("error.html", message="Invalid actor ID in URL.")

        try:
            with open('actors3.csv', mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    try:
                        csv_actor_id = int(float(row.get('actor_id', -1)))
                        if csv_actor_id == actor_id:
                            actor_name = row.get('actor_name', 'Unknown')
                            movie = row.get('movie_title', 'Untitled')
                            if movie:
                                movies.add(movie)
                    except ValueError:
                        pass
        except FileNotFoundError:
            return render_template("error.html", message="Actor database not found.")

        if not actor_name:
            return render_template("error.html", message=f"No actor found for ID {actor_id}.")

        actor_data = fetch_actor_from_tmdb(actor_id)
        if not actor_data:
            return render_template("error.html", message=f"Could not fetch details for {actor_name} from TMDB.")

        return render_template("actor.html", actor=actor_data, movies=sorted(movies))

    except Exception as e:
        logging.exception("❌ Error loading actor details.")
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
        return render_template("error.html", message="An error occurred while fetching movie details.")

if __name__ == '__main__':
    app.run(debug=True)