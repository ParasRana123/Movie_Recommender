import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import RecommendationView from '../components/RecommendationView';
import { fetchRecommendations } from '../api/movieApi';

export default function HomePage() {
  const [activeMovieData, setActiveMovieData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  const handleSearchMovie = async (title) => {
    if (!title || !title.trim()) return;
    setLoading(true);
    setError(false);
    try {
      const data = await fetchRecommendations(title.trim());
      setActiveMovieData(data);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      setError(true);
      setActiveMovieData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div id="content">
      <Navbar onSearchMovie={handleSearchMovie} />

      {/* Original Loader */}
      {loading && (
        <div id="loader">
          <p id="loader-text" style={{ color: '#333333' }}>LOADING...</p>
        </div>
      )}

      {/* Original Fail Message */}
      {error && !loading && (
        <div className="fail" style={{ display: 'block' }}>
          <center>
            <h3>Sorry! The movie you requested is not in our database. <br />
            Please check the spelling or try with other movies!</h3>
          </center>
        </div>
      )}

      {/* Recommendation Results */}
      {activeMovieData && !loading && (
        <div className="results">
          <RecommendationView
            movieData={activeMovieData}
            onSelectRecommendedMovie={handleSearchMovie}
          />
        </div>
      )}
    </div>
  );
}
