import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import RecommendationView from '../components/RecommendationView';
import Loader from '../components/Loader';
import { fetchRecommendations } from '../api/movieApi';

export default function HomePage() {
  const [activeMovieData, setActiveMovieData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  const handleSearchMovie = async (title) => {
    if (!title || !title.trim()) return;
    setLoading(true);
    setError(false);
    
    const startTime = Date.now();
    try {
      const data = await fetchRecommendations(title.trim());
      // Ensure loader displays for at least 1.5-2 seconds to show the friendly cycling loader messages
      const elapsed = Date.now() - startTime;
      if (elapsed < 1800) {
        await new Promise(resolve => setTimeout(resolve, 1800 - elapsed));
      }
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

      {/* Dynamic Loader with rotating messages */}
      {loading && <Loader />}

      {/* Original Fail Message */}
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
