import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import RecommendationView from '../components/RecommendationView';
import { fetchMovieDetails } from '../api/movieApi';

export default function MovieDetailsPage() {
  const { movieTitle } = useParams();
  const navigate = useNavigate();
  const [movieData, setMovieData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!movieTitle) return;
    let isMounted = true;
    setLoading(true);
    setError(false);

    fetchMovieDetails(decodeURIComponent(movieTitle))
      .then(data => {
        if (isMounted) {
          setMovieData(data);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }
      })
      .catch(err => {
        if (isMounted) {
          console.error(err);
          setError(true);
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
    <div id="content">
      <Navbar onSearchMovie={handleSelectMovie} initialQuery={movieTitle || ''} />

      {loading && (
        <div id="loader">
          <p id="loader-text" style={{ color: '#333333' }}>LOADING...</p>
        </div>
      )}

      {error && !loading && (
        <div className="fail" style={{ display: 'block' }}>
          <center>
            <h3>Sorry! The movie you requested is not in our database. <br />
            Please check the spelling or try with other movies!</h3>
          </center>
        </div>
      )}

      {movieData && !loading && (
        <div className="results">
          <RecommendationView
            movieData={movieData}
            onSelectRecommendedMovie={handleSelectMovie}
          />
        </div>
      )}
    </div>
  );
}
