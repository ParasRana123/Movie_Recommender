import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import Navbar from '../components/Navbar';
import Loader from '../components/Loader';
import RecommendationView from '../components/RecommendationView';
import MovieCard from '../components/MovieCard';
import { fetchRecommendations, fetchTopMovies } from '../api/movieApi';
import { GENRES_DATA } from '../data/genresData';

export default function HomePage() {
  const [activeMovieData, setActiveMovieData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [topMovies, setTopMovies] = useState([]);

  useEffect(() => {
    let isMounted = true;
    fetchTopMovies().then(data => {
      if (isMounted && Array.isArray(data)) {
        setTopMovies(data);
      }
    });
    return () => { isMounted = false; };
  }, []);

  const handleSearchMovie = async (title) => {
    if (!title || !title.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await fetchRecommendations(title.trim());
      setActiveMovieData(data);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      setError(err.message || 'Sorry! The movie you requested is not in our database. Please check the spelling or try with other movies!');
      setActiveMovieData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="home-page-container">
      <Navbar onSearchMovie={handleSearchMovie} />

      {/* Loading Overlay */}
      {loading && <Loader text="GENERATING AI RECOMMENDATIONS & SENTIMENT..." />}

      {/* Error Message */}
      {error && !loading && (
        <div className="search-error-banner">
          <div className="error-content">
            <h3>⚠️ Movie Not Found</h3>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Active Movie Recommendations */}
      {activeMovieData && !loading && (
        <RecommendationView
          movieData={activeMovieData}
          onSelectRecommendedMovie={handleSearchMovie}
        />
      )}

      {/* Default Landing Showcase if no movie selected */}
      {!activeMovieData && !loading && (
        <main className="home-landing-content">
          {/* Welcome Banner */}
          <section className="home-welcome-hero">
            <h1 className="welcome-title">
              Discover Your Next <span className="text-red">Favorite Movie</span>
            </h1>
            <p className="welcome-subtitle">
              Intelligent content-based movie recommendations powered by Machine Learning and NLP Sentiment Analysis.
            </p>
          </section>

          {/* Trending / Featured Showcase */}
          {topMovies.length > 0 && (
            <section className="home-showcase-section">
              <div className="section-header-row">
                <h2 className="showcase-heading">🔥 Popular & Trending</h2>
                <span className="showcase-tag">Curated Classics</span>
              </div>
              <div className="movies-grid">
                {topMovies.map((movie, idx) => (
                  <MovieCard
                    key={idx}
                    title={movie.title}
                    poster={movie.poster}
                    rating={movie.vote_average}
                    onClick={() => handleSearchMovie(movie.title)}
                  />
                ))}
              </div>
            </section>
          )}

          {/* Quick Genre Explorer */}
          <section className="home-genres-section">
            <div className="section-header-row">
              <h2 className="showcase-heading">🎬 Browse By Genre</h2>
              <Link to="/genres" className="see-all-link">View All 11 Genres →</Link>
            </div>
            <div className="genres-quick-grid">
              {GENRES_DATA.slice(0, 6).map((g) => (
                <Link
                  key={g.id}
                  to={`/genres/${g.id}`}
                  className="genre-quick-card"
                  style={{ backgroundImage: `linear-gradient(to top, rgba(0,0,0,0.85) 20%, transparent), url('${g.image}')` }}
                >
                  <span className="genre-quick-name">{g.name}</span>
                </Link>
              ))}
            </div>
          </section>
        </main>
      )}
    </div>
  );
}
