# 🎬 Movie Recommendation System (React + Flask)

A machine-learning and NLP-powered Movie Recommendation web application with a modern React SPA frontend and a high-performance Flask REST API backend.

-  **Live Website**: [https://movierecommender-navy.vercel.app/](https://movierecommender-navy.vercel.app/)
-  **Demo Video**: [https://res.cloudinary.com/d3ukbssg/video/upload/v1787560286/movie_recom_record.mp4](https://res.cloudinary.com/d3ukbssg/video/upload/v1787560286/movie_recom_record.mp4)

---

## Architecture Overview

```
Movie_Recommender/
├── csv/                        # All datasets (main_data.csv, main_data1.csv, action.csv, etc.)
├── ipynb/                      # All Jupyter Notebooks for data preprocessing & NLP training
├── backend/
│   ├── app.py                  # Flask REST API server (CORS enabled)
│   ├── requirements.txt        # Backend dependencies (Flask, pandas, scikit-learn, etc.)
│   └── models/                 # Active ML & NLP models (nlp_model2.pkl, transform1.pkl)
│
└── frontend/
    ├── package.json            # React & Vite dependencies
    ├── vite.config.js          # Vite config with backend proxy
    ├── public/images/          # High-resolution genre banners and SVG icons
    └── src/
        ├── api/                # Axios/Fetch API client
        ├── context/            # Global Theme (Dark/Light) & Watchlist state
        ├── data/               # Centralized genre metadata & instant suggestions
        ├── components/         # Navbar (autocomplete), MovieCard, RecommendationView, Loader, Toast
        ├── pages/              # HomePage, MovieDetailsPage, ActorPage, WatchlistPage, GenresPage, GenreDetailPage
        └── styles/             # Exact styled CSS with Light and Dark mode
```

---

## How to Run Locally

### 1. Start the Flask Backend
Open a terminal in the project root or `backend/` directory:
```bash
python main.py
# or
cd backend && python app.py
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

## Features
- **Universal Live Autocomplete**: As you type in the search bar on any page (e.g. *avengers*), matched movie titles appear instantly in the dropdown with matched characters highlighted.
- **Content-Based Recommendations**: Machine Learning cosine similarity engine generates 10 relevant movie recommendations.
- **NLP Sentiment Analysis**: Audience reviews are classified into *Positive* or *Critical* with probability confidence scores.
- **Trailers & Direct Streaming**: YouTube trailer/teaser players and direct click-to-watch redirection on streaming services (Netflix, Prime, Disney+ Hotstar, Apple TV, JioCinema, etc.).
- **Dark & Light Mode**: Seamless theme toggle in navbar with state persistence.
- **Global Watchlist**: Saved movies persist across sessions in `localStorage`.
- **Dynamic Genre Hub**: 11 cinematic genres handled by a single optimized, reusable React component.
- **Actor Filmographies**: Deep-dive into actor biographies and top movie credits.