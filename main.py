import numpy as np
import pandas as pd
from flask import Flask, render_template, request , redirect , session
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

# Load the NLP model and TF-IDF vectorizer from disk
filename = 'nlp_model2.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('transform1.pkl', 'rb'))

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
            # Find other movies to recommend (skipping the movie itself)
            additional_movies = [x for x in list(enumerate(similarity[i])) if x[0] != i][10:]
            lst.extend(additional_movies[:remaining])
        
        l = []
        for item in lst:
            a = item[0]
            l.append(data['movie_title'][a])
        return l
    
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

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html',suggestions=suggestions)

@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str

@app.route("/watchlist")
def watchlist():
    return render_template("watchlist.html")        

@app.route("/action")   
def action():
    df2 = pd.read_csv("action.csv")
    movie_titles = df2['movie_title'].tolist()
    return render_template('action.html' , movies = movie_titles)

@app.route("/allaction")
def allaction():
    df2 = pd.read_csv("action.csv")
    movie_titles = df2['movie_title'].tolist()
    return render_template("allaction.html" , movies = movie_titles)

@app.route("/horror")
def horror():
    df3 = pd.read_csv("horror.csv")
    movie_titles1 = df3['movie_title'].tolist()
    return render_template('horror.html' , movies=movie_titles1)

@app.route("/romance")
def romance():
    df5 = pd.read_csv("romance.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('romance.html' , movies=movie_titles3)

@app.route("/mystery")
def mystery():
    df5 = pd.read_csv("mystery.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('mystery.html' , movies=movie_titles3)

@app.route("/history")
def history():
    df5 = pd.read_csv("history.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('history.html' , movies=movie_titles3)

@app.route("/thriller")
def thriller():
    df5 = pd.read_csv("thriller.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('thriller.html' , movies=movie_titles3)

@app.route("/comedy")
def comedy():
    df4 = pd.read_csv('comedy.csv')
    movie_titles2 = df4['movie_title'].tolist()
    return render_template('comedy.html' , movies=movie_titles2)

@app.route("/fantasy")
def fantasy():
    df5 = pd.read_csv("fantasy.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('fantasy.html' , movies=movie_titles3)

@app.route("/adventure")
def adventure():
    df5 = pd.read_csv("adventure.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('adventure.html' , movies=movie_titles3)              

@app.route("/documentary")
def documentary():
    df5 = pd.read_csv("documentary.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('documentary.html' , movies=movie_titles3)  

@app.route("/sci_fi")
def sci_fi():
    df5 = pd.read_csv("sci_fi.csv")
    movie_titles3 = df5['movie_title'].tolist()
    return render_template('sci_fi.html' , movies=movie_titles3) 

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

        # Function to safely convert a string representation of a list into an actual list
        def safe_convert_list(data):
            try:
                return ast.literal_eval(data)
            except (ValueError, SyntaxError):
                return []

        # Convert string inputs to lists
        rec_movies = safe_convert_list(rec_movies)
        rec_posters = safe_convert_list(rec_posters)
        cast_names = safe_convert_list(cast_names)
        cast_chars = safe_convert_list(cast_chars)
        cast_profiles = safe_convert_list(cast_profiles)
        cast_bdays = safe_convert_list(cast_bdays)
        cast_bios = safe_convert_list(cast_bios)
        cast_places = safe_convert_list(cast_places)

        cast_ids = cast_ids.split(',')
        cast_ids[0] = cast_ids[0].replace("[","")
        cast_ids[-1] = cast_ids[-1].replace("]","")

        for i in range(len(cast_bios)):
            cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')

        # Dictionary mappings
        movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))} if rec_posters else {}
        casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))} if cast_profiles else {}
        cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))} if cast_places else {}

        # Web Scraping IMDb Reviews
        movie_reviews = {}
        if imdb_id:
            url = f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            
            imdb_request = urllib.request.Request(url, headers=headers)  # ✅ Rename 'request' to 'imdb_request'
            
            try:
                response = urllib.request.urlopen(imdb_request)  # ✅ Use 'imdb_request' here
                soup = bs.BeautifulSoup(response.read(), 'lxml')
                soup_result = soup.find_all("div", {"class": "text show-more__control"})

                # Process reviews
                reviews_list, reviews_status = [], []
                for review in soup_result:
                    if review.string:
                        reviews_list.append(review.string)
                        movie_vector = vectorizer.transform([review.string])
                        pred = clf.predict(movie_vector)
                        reviews_status.append('Good' if pred else 'Bad')

                # Create a dictionary of reviews with their sentiment status
                movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

            except urllib.error.HTTPError as e:
                logging.error(f"HTTP Error: {e.code} - {e.reason}")
                movie_reviews = {"Error": "Could not retrieve reviews"}
            except urllib.error.URLError as e:
                logging.error(f"URL Error: {e.reason}")
                movie_reviews = {"Error": "Network issue, couldn't retrieve reviews"}
            except Exception as e:
                logging.error(f"Unexpected error during IMDb scraping: {e}")
                movie_reviews = {"Error": "An unexpected error occurred"}

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
            cast_details=cast_details
        )

    except Exception as e:
        logging.error(f"Critical error in recommendation function: {e}")
        return render_template('error.html', message="An error occurred while processing your request.")

@app.route("/actor/<actor_id>")
@app.route("/actor/<actor_id>")
def actor_details(actor_id):
    try:
        global cast_details  # Declare it as global
        movies = []

        # Map actor_id to actor_name
        actor_name = None
        for name, details in cast_details.items():
            if details[0] == actor_id:
                actor_name = name
                break

        if not actor_name:
            return render_template("error.html", message=f"Actor ID {actor_id} not found in cast details.")

        # Now look up movies corresponding to the actor_name in the CSV file
        with open('actors1.csv', mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['actor_name'].strip() == actor_name.strip():  # Match actor_name from CSV
                    movies.append(row['movie_title'])

        # Debugging: Log the movies list
        logging.debug(f"Movies for {actor_name}: {movies}")

        # Find actor details from cast_details dictionary
        actor = {
            "name": actor_name,
            "profile": details[1],
            "birthday": details[2],
            "birth_place": details[3],
            "biography": details[4]
        }

        return render_template("actor.html", actor=actor, movies=movies)

    except Exception as e:
        logging.error(f"Error loading actor details: {e}")
        return render_template("error.html", message="An error occurred.")

if __name__ == '__main__':
    app.run(debug=True)