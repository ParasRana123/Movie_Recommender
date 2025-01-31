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


@app.route("/recommend", methods=["POST"])
def recommend():
    # Getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # Get movie suggestions for auto complete
    suggestions = get_suggestions()

    # Convert strings to lists
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # Convert string to list for cast IDs
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")
    
    # Render the string to Python string format
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')
    
    # Combine lists as dictionaries to pass to the HTML file
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # Web scraping to get user reviews from IMDb with User-Agent header
    url = f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    request = urllib.request.Request(url, headers=headers)
    
    try:
        sauce = urllib.request.urlopen(request).read()
        soup = bs.BeautifulSoup(sauce, 'lxml')
        soup_result = soup.find_all("div", {"class": "text show-more__control"})

        # List of reviews and status (good or bad)
        reviews_list = []
        reviews_status = []
        for reviews in soup_result:
            if reviews.string:
                reviews_list.append(reviews.string)
                # Passing the review to the model
                movie_review_list = np.array([reviews.string])
                movie_vector = vectorizer.transform(movie_review_list)
                pred = clf.predict(movie_vector)
                reviews_status.append('Good' if pred else 'Bad')

        # Combine reviews and comments into a dictionary
        movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        movie_reviews = {"Error": "Could not retrieve reviews"}
    
    # Pass all data to the HTML file
    return render_template(
        'recommend.html',
        title=title,
        poster=poster,
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

if __name__ == '__main__':
    app.run(debug=True)