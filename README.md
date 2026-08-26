# 🎬 Movie Recommendation System (React + Flask)

A machine-learning and NLP-powered Movie Recommendation web application with a modern React SPA frontend and a Flask REST API backend.

---

## 📁 Architecture Overview

```
Movie_Recommender/
├── backend/
│   ├── app.py                  # Flask REST API server (CORS enabled)
│   ├── requirements.txt        # Backend dependencies
│   ├── nlp_model2.pkl          # Sentiment Analysis model
│   ├── transform1.pkl          # TF-IDF Vectorizer
│   ├── main_data.csv           # Autocomplete & recommendations dataset
│   ├── main_data1.csv          # Vectorized similarity dataset
│   └── *.csv                   # Genre datasets (action, comedy, sci_fi, etc.)
│
└── frontend/
    ├── package.json            # React & Vite dependencies
    ├── vite.config.js          # Vite config with backend proxy
    ├── public/images/          # High-resolution genre banners and SVG icons
    └── src/
        ├── api/                # Axios/Fetch API client
        ├── context/            # Global Watchlist state (localStorage synced)
        ├── data/               # Centralized genre metadata
        ├── components/         # Navbar (autocomplete), MovieCard, RecommendationView, CastCard, Loader, Toast
        ├── pages/              # HomePage, MovieDetailsPage, ActorPage, WatchlistPage, GenresPage, GenreDetailPage
        └── styles/             # Netflix-inspired dark mode styling
```

---

## 🚀 How to Run Locally

### 1. Start the Flask Backend
Open a terminal in the `backend/` directory:
```bash
cd backend
python app.py
```
*The API will start at `http://127.0.0.1:5000`*

### 2. Start the React Frontend
Open another terminal in the `frontend/` directory:
```bash
cd frontend
npm install
npm run dev
```
*The React app will open at `http://localhost:5173`*

---

## ✨ Features
- **Universal Live Autocomplete**: As you type in the search bar on any page (e.g. *avengers*), matched movie titles appear instantly in the dropdown.
- **Content-Based Recommendations**: Machine Learning cosine similarity engine generates 10 relevant movie recommendations.
- **NLP Sentiment Analysis**: Audience reviews are classified into *Positive* or *Critical* with probability confidence scores.
- **Trailers & Streaming Info**: YouTube trailer/teaser players and streaming platform availability (Netflix, Prime, Disney+, etc.).
- **Global Watchlist**: Saved movies persist across sessions in `localStorage` with undo notifications.
- **Dynamic Genre Hub**: 11 cinematic genres handled by a single optimized, reusable React component.
- **Actor Filmographies**: Deep-dive into actor biographies and top movie credits.