import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import Loader from '../components/Loader';
import RecommendationView from '../components/RecommendationView';
import { fetchMovieDetails } from '../api/movieApi';

export default function MovieDetailsPage() {
  const { movieTitle } = useParams();
  const navigate = useNavigate();
  const [movieData, setMovieData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!movieTitle) return;
    let isMounted = true;
    setLoading(true);
    setError(null);

    fetchMovieDetails(decodeURIComponent(movieTitle))
      .then(data => {
        if (isMounted) {
          setMovieData(data);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }
      })
      .catch(err => {
        if (isMounted) {
          setError(err.message || 'Movie not found');
          setMovieData(null);
        }
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => { isMounted = false; };
  }, [movieTitle]);

  const handleSelectMovie = (newTitle) => {
    navigate(`/movie/${encodeURIComponent(newTitle)}`);
  };

  return (
    <div className="movie-details-page-container">
      <Navbar onSearchMovie={handleSelectMovie} initialQuery={movieTitle || ''} />

      {loading && <Loader text="FETCHING MOVIE DETAILS & REVIEWS..." />}

      {error && !loading && (
        <div className="search-error-banner">
          <div className="error-content">
            <h3>⚠️ Movie Not Found</h3>
            <p>{error}</p>
          </div>
        </div>
      )}

      {movieData && !loading && (
        <RecommendationView
          movieData={movieData}
          onSelectRecommendedMovie={handleSelectMovie}
        />
      )}
    </div>
  );
}
