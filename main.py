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

def create_similarity():
    data = pd.read_csv('main_data1.csv', encoding='latin1')
    # Creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # Creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data, similarity

def rcmd(m):
    m = m.lower()
    global data, similarity
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    
    if m not in data['movie_title'].unique():
        return 'Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies'
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]  # Exclude the first item (the movie itself)
        
        # Check if we have fewer than 10 recommendations, and if so, pad the list
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
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

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
        m_str = "---".join(rc)
        return m_str

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

cast_details = {}

@app.route("/recommend", methods=["POST"])
def recommend():
    global cast_details
    try:
        # Extracting data from AJAX request safely
        title = request.form.get('title', 'Unknown Title')
        cast_ids = request.form.get('cast_ids', '[]')
        cast_names = request.form.get('cast_names', '[]')
        cast_chars = request.form.get('cast_chars', '[]')
        cast_bdays = request.form.get('cast_bdays', '[]')
        cast_bios = request.form.get('cast_bios', '[]')
        cast_places = request.form.get('cast_places', '[]')
        cast_profiles = request.form.get('cast_profiles', '[]')
        imdb_id = request.form.get('imdb_id', '')
        poster = request.form.get('poster', '')
        backdrop = request.form.get('backdrop', '')
        genres = request.form.get('genres', 'Unknown Genre')
        overview = request.form.get('overview', 'No Overview Available')
        vote_average = request.form.get('rating', 'N/A')
        vote_count = request.form.get('vote_count', '0')
        release_date = request.form.get('release_date', 'Unknown Date')
        runtime = request.form.get('runtime', 'Unknown Runtime')
        status = request.form.get('status', 'Unknown Status')
        rec_movies = request.form.get('rec_movies', '[]')
        rec_posters = request.form.get('rec_posters', '[]')
        trailer_url = request.form.get('trailer', '')
        teaser_url = request.form.get('teaser', '')
        watch_providers = request.form.get('watch_providers', '[]')
        watch_provider_logos = request.form.get('watch_provider_logos', '[]')
        budget = request.form.get('budget', 'N/A')
        revenue = request.form.get('revenue', 'N/A')

        original_language = request.form.get('original_language', 'N/A')
        director_name = request.form.get('director_name', 'Unknown')
        director_image = request.form.get('director_image', '')
        director_bio = request.form.get('director_bio', 'No biography available')
        director_birthplace = request.form.get('director_birthplace', 'Unknown')

        # Function to safely convert a string representation of a list into an actual list
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
        
        trailer_id = extract_video_id(trailer_url)
        teaser_id = extract_video_id(teaser_url)

        trailer_embed = f"https://www.youtube.com/embed/{trailer_id}" if trailer_id else None
        teaser_embed = f"https://www.youtube.com/embed/{teaser_id}" if teaser_id else None

        watch_providers = safe_convert_list(watch_providers)
        watch_provider_logos = safe_convert_list(watch_provider_logos)

        # Combine provider names and logos into tuples
        streaming_availability = list(zip(watch_providers, watch_provider_logos))

        # Convert string inputs to lists
        rec_movies = safe_convert_list(rec_movies)
        rec_posters = safe_convert_list(rec_posters)
        cast_names = safe_convert_list(cast_names)
        cast_chars = safe_convert_list(cast_chars)
        cast_profiles = safe_convert_list(cast_profiles)
        cast_bdays = safe_convert_list(cast_bdays)
        cast_bios = safe_convert_list(cast_bios)
        cast_places = safe_convert_list(cast_places)
        cast_ids = safe_convert_list(cast_ids)

        for i in range(len(cast_bios)):
            if isinstance(cast_bios[i], str):
                cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

        # Dictionary mappings with safety
        num_cards = min(len(rec_posters), len(rec_movies))
        movie_cards = {rec_posters[i]: rec_movies[i] for i in range(num_cards)} if num_cards > 0 else {}

        num_casts = min(len(cast_names), len(cast_ids), len(cast_chars), len(cast_profiles))
        casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(num_casts)} if num_casts > 0 else {}

        num_details = min(len(cast_names), len(cast_ids), len(cast_profiles), len(cast_bdays), len(cast_places), len(cast_bios))
        cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(num_details)} if num_details > 0 else {}

        # Web Scraping IMDb Reviews
        movie_reviews = {}
        if imdb_id:
            try:
                url = f'https://www.imdb.com/title/{imdb_id}/reviews/?ref_=tt_ov_rt'
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'}
                response = requests.get(url, headers=headers, timeout=4)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'lxml')
                    soup_result = soup.find_all("div", {"class": "ipc-html-content-inner-div"})
                    reviews_list = []
                    reviews_status = []
                    for reviews in soup_result:
                        review_text = reviews.text.strip()
                        if review_text:
                            reviews_list.append(review_text)
                            movie_review_list = np.array([review_text])
                            movie_vector = vectorizer.transform(movie_review_list)
                            pred = clf.predict(movie_vector)
                            reviews_status.append('Good' if pred[0] == 1 or pred[0] == 'Good' or pred[0] else 'Bad')
                    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}
            except Exception as ex:
                logging.warning(f"Error scraping IMDb reviews: {ex}")

        # Render the recommend page with all processed data
        return render_template(
            'recommend.html',
            title=title,
            poster=poster,
            backdrop=backdrop,
            overview=overview,
            vote_average=vote_average,
            vote_count=vote_count,
            release_date=release_date,
            runtime=runtime,
            status=status,
            genres=genres,
            movie_cards=movie_cards,
            reviews=movie_reviews,
            casts=casts,
            cast_details=cast_details,
            trailer=trailer_embed,
            teaser=teaser_embed,
            streaming_availability=streaming_availability,
            budget=budget,
            revenue=revenue,
            original_language=original_language,
            director_name=director_name,
            director_image=director_image,
            director_bio=director_bio,
            director_birthplace=director_birthplace
        )

    except Exception as e:
        logging.error(f"Critical error in recommendation function: {e}", exc_info=True)
        return render_template('error.html', message="An error occurred while processing your request.")
    
def get_recommendations(movie_title):
    """Get recommended movies based on similarity."""
    movie_title = movie_title.lower()

    # Ensure `data` and `similarity` are loaded
    global data, similarity
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()

    # If movie not found, return empty list
    if movie_title not in data['movie_title'].unique():
        return []

    # Get movie index and find similar movies
    i = data.loc[data['movie_title'] == movie_title].index[0]
    lst = list(enumerate(similarity[i]))
    lst = sorted(lst, key=lambda x: x[1], reverse=True)
    lst = lst[1:11]  # Exclude the first item (the movie itself)

    recommended_movies = []

    for item in lst:
        rec_title = data.iloc[item[0]]['movie_title']
        search_url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(rec_title)}"
        try:
            res = requests.get(search_url, timeout=4)
            if res.status_code == 200:
                results = res.json().get("results", [])
                if results and results[0].get("poster_path"):
                    poster_url = f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"
                else:
                    poster_url = "https://via.placeholder.com/200x300?text=No+Poster"
                recommended_movies.append({"title": rec_title, "poster": poster_url})
        except Exception as e:
            logging.warning(f"Error fetching poster for {rec_title}: {e}")
            recommended_movies.append({"title": rec_title, "poster": "https://via.placeholder.com/200x300?text=No+Poster"})

    return recommended_movies

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/"

def fetch_actor_from_wikipedia(actor_name):
    """ Fetch actor details from Wikipedia by scraping the page. """
    try:
        actor_url = WIKIPEDIA_URL + actor_name.replace(" ", "_")
        response = requests.get(actor_url, timeout=5)

        if response.status_code != 200:
            logging.error(f"🚨 Wikipedia page for {actor_name} not found! Status Code: {response.status_code}")
            return None

        # Parse the Wikipedia page
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the first paragraph
        paragraphs = soup.select("p")
        biography = "No biography available."
        for p in paragraphs:
            if p.text.strip():
                biography = p.text.strip()
                break

        # Extract image
        image_url = "https://via.placeholder.com/150"
        image_tag = soup.select_one(".infobox img")
        if image_tag and image_tag.get("src"):
            image_url = "https:" + image_tag["src"]

        return {
            "name": actor_name,
            "profile": image_url,
            "birthday": "Unknown",
            "birth_place": "Unknown",
            "biography": biography
        }

    except Exception as e:
        logging.error(f"❌ Error fetching Wikipedia details for {actor_name}: {e}")
        return None

def fetch_actor_from_tmdb(actor_id):
    """Fetch actor details from TMDB using a numeric actor_id."""
    try:
        url = f"{TMDB_BASE_URL}/person/{actor_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=5)

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

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Network error while fetching from TMDB: {e}")
        return None
    except Exception as e:
        logging.exception("❌ Unexpected error during TMDB fetch.")
        return None

@app.route("/actor/<actor_id>")
def actor_details(actor_id):
    try:
        movies = set()
        actor_name = None

        # ✅ Sanitize input
        actor_id = escape(actor_id)
        logging.debug(f"📌 Raw Actor ID from URL: {actor_id}")

        # ✅ Convert to int and validate
        try:
            if not actor_id or actor_id.lower() == 'none':
                raise ValueError("Empty or None actor_id")
            actor_id = int(float(actor_id))
        except ValueError:
            logging.error(f"❌ Invalid actor_id format: {actor_id}")
            return render_template("error.html", message="Invalid actor ID in URL.")

        # ✅ Read CSV and find actor
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
            logging.error("❌ actors3.csv file not found!")
            return render_template("error.html", message="Actor database not found.")

        # ✅ Handle actor not found in CSV
        if not actor_name:
            logging.warning(f"⚠️ Actor ID {actor_id} not found in CSV.")
            return render_template("error.html", message=f"No actor found for ID {actor_id}.")

        # ✅ Fetch actor details from TMDB using actor_id
        actor_data = fetch_actor_from_tmdb(actor_id)
        if not actor_data:
            return render_template("error.html", message=f"Could not fetch details for {actor_name} from TMDB.")

        return render_template("actor.html", actor=actor_data, movies=sorted(movies))

    except Exception as e:
        logging.exception("❌ Unexpected error while loading actor details.")
        return render_template("error.html", message="An internal error occurred while fetching actor details.")

def get_movie_id(movie_title):
    """Fetch movie ID from TMDB using the title."""
    search_url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(movie_title)}"
    try:
        response = requests.get(search_url, timeout=5)
        if response.status_code == 200:
            search_results = response.json().get("results", [])
            if search_results:
                return search_results[0]["id"]
        logging.error(f"TMDB Search API Error: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"Exception in get_movie_id: {e}")
    return None

@app.route("/movie/<movie_title>")
def movie_details(movie_title):
    try:
        # Convert movie title to TMDB movie ID
        movie_id = get_movie_id(movie_title)
        if not movie_id:
            return render_template("error.html", message="Movie not found in TMDB.")

        # Fetch movie details using the correct movie ID
        movie_url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=videos,credits,watch/providers"
        movie_response = requests.get(movie_url, timeout=5)

        if movie_response.status_code != 200:
            logging.error(f"TMDB API Error: {movie_response.status_code} - {movie_response.text}")
            return render_template("error.html", message="Error fetching movie details from TMDB API.")

        movie_data = movie_response.json()

        # Extract movie details safely
        title = movie_data.get("original_title", "Unknown Title")
        poster_path = movie_data.get("poster_path")
        backdrop_path = movie_data.get("backdrop_path")

        # Ensure valid URLs
        poster = f"https://image.tmdb.org/t/p/original{poster_path}" if poster_path else "https://via.placeholder.com/500x750"
        backdrop = f"https://image.tmdb.org/t/p/original{backdrop_path}" if backdrop_path else "https://via.placeholder.com/1280x720"

        overview = movie_data.get("overview", "No overview available.")
        genres = [genre["name"] for genre in movie_data.get("genres", [])] if "genres" in movie_data else []
        release_date = movie_data.get("release_date", "Unknown Date")
        runtime = movie_data.get("runtime", "Unknown Runtime")
        budget = f"${movie_data.get('budget', 0):,}"
        revenue = f"${movie_data.get('revenue', 0):,}"
        original_language = movie_data.get("original_language", "N/A").upper()
        vote_average = movie_data.get("vote_average", "N/A")
        vote_count = movie_data.get("vote_count", "0")
        status = movie_data.get("status", "Unknown Status")
        imdb_id = movie_data.get("imdb_id", "")

        # Extract director details
        director = next((crew for crew in movie_data.get("credits", {}).get("crew", []) if crew.get("job") == "Director"), {})
        director_name = director.get("name", "Unknown")
        director_image_path = director.get("profile_path")
        director_id = director.get("id")
        director_image = f"https://image.tmdb.org/t/p/w300{director_image_path}" if director_image_path else "https://via.placeholder.com/300"
        director_bio = "Biography not available"

        if director_id:
            director_url = f"{TMDB_BASE_URL}/person/{director_id}?api_key={TMDB_API_KEY}"
            try:
                director_response = requests.get(director_url, timeout=5)
                if director_response.status_code == 200:
                    director_data = director_response.json()
                    director_bio = director_data.get("biography", "Biography not available")
            except Exception as e:
                logging.warning(f"Error fetching director data: {e}")

        credits = movie_data.get("credits", {})
        cast = credits.get("cast", [])[:10] if isinstance(credits, dict) and "cast" in credits else []

        actors = [
            {
                "id": actor.get("id", None),
                "name": actor.get("name", "Unknown Actor"),
                "character": actor.get("character", "Unknown Character"),
                "image": f"https://image.tmdb.org/t/p/w300{actor['profile_path']}" 
                if actor.get("profile_path") else "https://via.placeholder.com/150"
            }
            for actor in cast
        ]

        # Extract trailers and teasers
        videos = movie_data.get("videos", {}).get("results", [])
        trailer = next((f"https://www.youtube.com/embed/{video['key']}" for video in videos if video.get("type") == "Trailer" and video.get("site") == "YouTube"), None)
        teaser = next((f"https://www.youtube.com/embed/{video['key']}" for video in videos if video.get("type") == "Teaser" and video.get("site") == "YouTube"), None)

        # Extract streaming providers
        providers_data = movie_data.get("watch/providers", {}).get("results", {}).get("IN", {}).get("flatrate", [])
        if not providers_data:
            providers_data = movie_data.get("watch/providers", {}).get("results", {}).get("US", {}).get("flatrate", [])
        streaming_availability = [
            (provider["provider_name"], f"https://image.tmdb.org/t/p/w200{provider['logo_path']}") 
            for provider in providers_data if provider.get("provider_name") and provider.get("logo_path")
        ]

        # Fetch IMDb reviews
        movie_reviews = fetch_imdb_reviews(imdb_id) if imdb_id else {}
        recommendations = get_recommendations(title)

        # Render the template with movie details
        return render_template(
            "movie.html",
            title=title,
            poster=poster,
            backdrop=backdrop,
            overview=overview,
            genres=genres,
            release_date=release_date,
            runtime=runtime,
            budget=budget,
            revenue=revenue,
            original_language=original_language,
            vote_average=vote_average,
            vote_count=vote_count,
            status=status,
            director_name=director_name,
            director_image=director_image,
            director_bio=director_bio,
            actors=actors,
            trailer=trailer,
            teaser=teaser,
            streaming_availability=streaming_availability,
            movie_reviews=movie_reviews,
            recommended_movies=recommendations
        )

    except Exception as e:
        logging.error(f"Error fetching movie details: {e}", exc_info=True)
        return render_template("error.html", message="An error occurred while fetching movie details.")

def fetch_imdb_reviews(imdb_id):
    """Fetch IMDb reviews using web scraping."""
    movie_reviews = {}
    if not imdb_id:
        return movie_reviews
    try:
        url = f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=4)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'lxml')
            soup_result = soup.find_all("div", {"class": "ipc-html-content-inner-div"})

            reviews_list = []
            reviews_status = []

            for reviews in soup_result:
                reviews_text = reviews.text.strip()
                if reviews_text:
                    reviews_list.append(reviews_text)
                    movie_reviews_list = np.array([reviews_text])
                    movie_vector = vectorizer.transform(movie_reviews_list)
                    pred_prob = clf.predict_proba(movie_vector)[:, 1]
                    confidence_score = round(pred_prob[0] * 100, 2)
                    reviews_status.append(f"{confidence_score}% confident positive")
            
            movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}
        return movie_reviews

    except Exception as e:
        logging.warning(f"Error fetching IMDb reviews: {e}")
        return movie_reviews

if __name__ == '__main__':
    app.run(debug=True)