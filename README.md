# Movie Recommendation Platform

A multilingual movie recommendation system that leverages Natural Language Processing (NLP), cosine similarity, and web scraping to deliver accurate and engaging movie suggestions.

## Features

- 🔍 **Smart Movie Recommendations**  
  Suggests similar movies using TF-IDF, CountVectorizer, and cosine similarity based on user selections.
- 💬 **Multilingual Support**  
  Enables movie recommendations and search functionality across multiple languages.
- ✨ **Typo-Tolerant Search**  
  Intelligent search bar with auto-suggestions and typo handling for better user experience.
- 📊 **Top 20 Movies by Genre**  
  Compiled and ranked using IMDb ratings, updated via web scraping.
- 👤 **Detailed Cast & Crew Info**  
  Pop-up modals for actor/director bios, filmography, and embedded media previews.

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript (with Bootstrap for styling)
- **Backend:** Python (Flask)
- **Libraries:** 
  - NLP: `scikit-learn`, `NLTK`
  - Similarity: `cosine_similarity`, `TF-IDF`, `CountVectorizer`
  - Web Scraping: `BeautifulSoup`, `requests`
- **APIs:** TMDB API and IMDB for web scraping of movie reviews

## Project Structure

```bash
├── static/              # All CSS files and Images here          
├── templates/           # All the utility template files
├── main.py              # All flask routes present here
├── test.py              # All test code here
├── requirements.txt     # Contains all the requirements
└── README.md            # README.md file
```

### How to get the API key?

Create an account in https://www.themoviedb.org/, click on the `API` link from the left hand sidebar in your account settings and fill all the details to apply for API key. If you are asked for the website URL, just give "NA" if you don't have one. You will see the API key in your `API` sidebar once your request is approved.

## Installation

> **Note**: Python Version greater than 3.8 needed.

1. **Clone the Repository**

```bash
git clone [repository-url]
cd face
```

2. **Create and activate python virtual environment**

```bash
conda create -p venv python==3.11.0 -y
activate venv/
```

3. **Install all the requirements necessary for this project**

```bash
pip install -r requirements.txt
```

4. **Start the flask application**

```bash
python main.py
```

## Contributing

We welcome contributions from the community! Whether you're interested in improving features, fixing bugs, or adding new functionality, your input is valuable. Feel free to reach out to us with your ideas and suggestions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.