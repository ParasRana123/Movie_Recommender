from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import requests
import logging

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
    
def load_data():
    global data, similarity
    if data is None or similarity is None:
        data, similarity = create_similarity()

load_data()

def get_recommendations(movie_title):
    """Get recommended movies along with posters based on similarity."""
    movie_title = movie_title.lower()

    # Ensure `data` and `similarity` are loaded
    global data, similarity
    try:
        data.head()  # Check if data is a DataFrame
        similarity.shape  # Check if similarity is a matrix
    except:
        data, similarity = create_similarity()  

    # 🔍 Debugging: Check if the movie exists
    if movie_title not in data['movie_title'].unique():
        print(f"⚠️ Movie '{movie_title}' not found in dataset!")
        return []

    # Get movie index and find similar movies
    i = data.loc[data['movie_title'] == movie_title].index[0]
    lst = list(enumerate(similarity[i]))
    lst = sorted(lst, key=lambda x: x[1], reverse=True)
    lst = lst[1:11]  # Exclude the first item (the movie itself)

    recommended_movies = []

    for item in lst:
        rec_title = data.iloc[item[0]]['movie_title']
        print(f"🔍 Fetching movie ID for: {rec_title}")

        movie_id = get_movie_id(rec_title)  # ✅ Ensure this function works
        if not movie_id:
            print(f"❌ Failed to get ID for {rec_title}")
            continue

        movie_url = f"https://api.tmdb.org/3/movie/{movie_id}?api_key=fce0af3409e6113c9b3c75aaf49341bb"
        
        try:
            movie_response = requests.get(movie_url)
            movie_response.raise_for_status()  # ✅ Catch API errors

            movie_data = movie_response.json()
            poster_path = movie_data.get("poster_path", None)
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/200x300"

            recommended_movies.append({"title": rec_title, "poster": poster_url})
            print(f"✅ Added: {rec_title}")

        except requests.exceptions.RequestException as e:
            print(f"🚨 Error fetching poster for {rec_title}: {e}")

    print(f"✅ Total recommended movies: {len(recommended_movies)}")
    return recommended_movies


def get_movie_id(movie_title):
    """Fetch movie ID from TMDB using the title."""
    search_url = f"https://api.tmdb.org/3/search/movie?api_key=fce0af3409e6113c9b3c75aaf49341bb&query={movie_title}"
    response = requests.get(search_url)

    if response.status_code == 200:
        search_results = response.json().get("results", [])
        if search_results:
            return search_results[0]["id"]  # Get the first result's ID
    logging.error(f"TMDB Search API Error: {response.status_code} - {response.text}")
    return None  # Return None if not found

print(get_recommendations("Inception"))