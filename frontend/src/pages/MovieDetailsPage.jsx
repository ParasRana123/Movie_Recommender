import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import RecommendationView from '../components/RecommendationView';
import Loader from '../components/Loader';
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

    const startTime = Date.now();

    fetchMovieDetails(decodeURIComponent(movieTitle))
      .then(async (data) => {
        if (isMounted) {
          const elapsed = Date.now() - startTime;
          if (elapsed < 1800) {
            await new Promise(resolve => setTimeout(resolve, 1800 - elapsed));
          }
          if (isMounted) {
            setMovieData(data);
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }
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

      {/* Dynamic Loader with rotating messages */}
      {loading && <Loader />}

      {/* Fail Message */}
      {error && !loading && (
        <div className="fail" style={{ display: 'block', margin: '40px auto', textAlign: 'center' }}>
          <center>
            <h3 style={{ color: '#333333', maxWidth: '800px', lineHeight: '1.6' }}>
              Sorry! The movie you requested is not in our database. <br />
              Please check the spelling or try with other movies!
            </h3>
          </center>
        </div>
      )}

      {/* Recommendation Results */}
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
