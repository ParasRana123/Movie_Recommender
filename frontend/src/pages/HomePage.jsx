import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import RecommendationView from '../components/RecommendationView';
import Loader from '../components/Loader';
import { fetchRecommendations, fetchTrendingMovies, fetchUpcomingMovies } from '../api/movieApi';
import { GENRES_DATA } from '../data/genresData';
import { useTheme } from '../context/ThemeContext';

export default function HomePage() {
  const [activeMovieData, setActiveMovieData] = useState(null);
  const [trendingMovies, setTrendingMovies] = useState([]);
  const [upcomingMovies, setUpcomingMovies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [feedLoading, setFeedLoading] = useState(true);

  const { isDark } = useTheme();
  const navigate = useNavigate();

  // Load Trending & Upcoming feeds on mount
  useEffect(() => {
    let mounted = true;
    setFeedLoading(true);

    Promise.allSettled([fetchTrendingMovies(), fetchUpcomingMovies()])
      .then(([trendingRes, upcomingRes]) => {
        if (!mounted) return;
        if (trendingRes.status === 'fulfilled' && Array.isArray(trendingRes.value)) {
          setTrendingMovies(trendingRes.value);
        }
        if (upcomingRes.status === 'fulfilled' && Array.isArray(upcomingRes.value)) {
          setUpcomingMovies(upcomingRes.value);
        }
      })
      .finally(() => {
        if (mounted) setFeedLoading(false);
      });

    return () => { mounted = false; };
  }, []);

  const handleSearchMovie = async (title) => {
    if (!title || !title.trim()) return;
    setLoading(true);
    setError(false);
    
    const startTime = Date.now();
    try {
      const data = await fetchRecommendations(title.trim());
      // Friendly timing for loader animation
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

  const headingColor = isDark ? '#ffffff' : '#333333';
  const subtitleColor = isDark ? '#aaaaaa' : '#777777';
  const cardBg = isDark ? '#1c1c1c' : '#ffffff';
  const cardBorder = isDark ? '#2e2e2e' : 'rgba(0, 0, 0, 0.08)';

  return (
    <div id="content" style={{ minHeight: '100vh', paddingBottom: '60px' }}>
      <Navbar
        onSearchMovie={handleSearchMovie}
        onHomeClick={() => {
          setActiveMovieData(null);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }}
      />

      {/* Dynamic Loader */}
      {loading && <Loader />}

      {/* Fail Message */}
      {error && !loading && (
        <div className="fail" style={{ display: 'block', margin: '40px auto', textAlign: 'center' }}>
          <center>
            <h3 style={{ color: headingColor, maxWidth: '800px', lineHeight: '1.6' }}>
              Sorry! The movie you requested is not in our database. <br />
              Please check the spelling or try with other movies!
            </h3>
          </center>
        </div>
      )}

      {/* Recommendation Results (when a movie is active) */}
      {activeMovieData && !loading && (
        <div className="results">
          <RecommendationView
            movieData={activeMovieData}
            onSelectRecommendedMovie={handleSearchMovie}
          />
        </div>
      )}

      {/* Front Page Discovery Feed (Trending, Upcoming, Genres) */}
      {!activeMovieData && !loading && (
        <div className="home-feed-container" style={{ maxWidth: '1400px', margin: '0 auto', padding: '20px 25px' }}>
          
          {/* App Hero Heading */}
          <center>
            <h1 style={{ marginTop: '30px', marginBottom: '8px', fontSize: '38px', fontWeight: 'bold', letterSpacing: '1px' }}>
              MOVIE RECOMMENDATION SYSTEM
            </h1>
            <p style={{ color: subtitleColor, fontSize: '16px', marginBottom: '45px' }}>
              Discover top movies, real-time audience sentiments & AI-powered recommendations
            </p>
          </center>

          {/* 1. 🔥 TRENDING MOVIES SECTION */}
          <section className="home-section" style={{ marginBottom: '55px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '20px', borderBottom: `2px solid ${isDark ? '#2a2a2a' : '#eeeeee'}`, paddingBottom: '12px' }}>
              <div>
                <h3 style={{ color: headingColor, fontWeight: 'bold', margin: 0, fontSize: '26px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ color: '#e50914' }}>🔥</span> Trending Movies
                </h3>
                <h5 style={{ color: subtitleColor, margin: '6px 0 0 0', fontSize: '15px', fontWeight: 'normal' }}>
                  Top popular movies this week — click to get instant recommendations
                </h5>
              </div>
              <span style={{ color: '#e50914', fontSize: '14px', fontWeight: 'bold' }}>
                IMDb Style Trending
              </span>
            </div>

            <div className="home-movies-grid">
              {trendingMovies.slice(0, 10).map((movie, idx) => (
                <div
                  key={idx}
                  className="card home-movie-card"
                  style={{
                    backgroundColor: cardBg,
                    borderColor: cardBorder,
                    borderRadius: '16px',
                    overflow: 'hidden',
                    cursor: 'pointer',
                    boxShadow: isDark ? '0 6px 18px rgba(0,0,0,0.6)' : '0 4px 14px rgba(0,0,0,0.08)',
                    transition: 'transform 0.25s ease, box-shadow 0.25s ease'
                  }}
                  title={movie.title}
                  onClick={() => handleSearchMovie(movie.title)}
                >
                  <div className="imghvr" style={{ position: 'relative' }}>
                    <img
                      className="card-img-top"
                      style={{ width: '100%', height: '320px', objectFit: 'cover' }}
                      alt={`${movie.title} - poster`}
                      src={movie.poster || 'https://via.placeholder.com/240x360?text=No+Poster'}
                    />

                    {/* Rating Badge Overlay */}
                    {movie.rating && (
                      <div
                        style={{
                          position: 'absolute',
                          top: '10px',
                          right: '10px',
                          backgroundColor: 'rgba(0, 0, 0, 0.78)',
                          color: '#ffd700',
                          padding: '4px 10px',
                          borderRadius: '12px',
                          fontSize: '13px',
                          fontWeight: 'bold',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                          backdropFilter: 'blur(4px)'
                        }}
                      >
                        ★ {movie.rating}
                      </div>
                    )}

                    {/* Release Year Badge */}
                    {movie.release_date && (
                      <div
                        style={{
                          position: 'absolute',
                          top: '10px',
                          left: '10px',
                          backgroundColor: 'rgba(229, 9, 20, 0.85)',
                          color: '#ffffff',
                          padding: '3px 8px',
                          borderRadius: '8px',
                          fontSize: '12px',
                          fontWeight: 'bold'
                        }}
                      >
                        {movie.release_date.split('-')[0]}
                      </div>
                    )}

                    <figcaption className="fig">
                      <button className="card-btn btn btn-danger" style={{ backgroundColor: '#e50914', borderColor: '#e50914' }}>
                        Explore Movie
                      </button>
                    </figcaption>
                  </div>
                  
                  <div className="card-body" style={{ padding: '14px 12px', textAlign: 'center' }}>
                    <h5
                      className="card-title"
                      style={{
                        fontSize: '16px',
                        fontWeight: 'bold',
                        margin: 0,
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}
                    >
                      {movie.title}
                    </h5>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* 2. 🎬 UPCOMING MOVIES SECTION */}
          <section className="home-section" style={{ marginBottom: '55px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '20px', borderBottom: `2px solid ${isDark ? '#2a2a2a' : '#eeeeee'}`, paddingBottom: '12px' }}>
              <div>
                <h3 style={{ color: headingColor, fontWeight: 'bold', margin: 0, fontSize: '26px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ color: '#e50914' }}>🎬</span> Upcoming Movies
                </h3>
                <h5 style={{ color: subtitleColor, margin: '6px 0 0 0', fontSize: '15px', fontWeight: 'normal' }}>
                  Anticipated releases coming soon to theaters & streaming
                </h5>
              </div>
              <span style={{ color: '#e50914', fontSize: '14px', fontWeight: 'bold' }}>
                In Theaters & OTT
              </span>
            </div>

            <div className="home-movies-grid">
              {upcomingMovies.slice(0, 10).map((movie, idx) => (
                <div
                  key={idx}
                  className="card home-movie-card"
                  style={{
                    backgroundColor: cardBg,
                    borderColor: cardBorder,
                    borderRadius: '16px',
                    overflow: 'hidden',
                    cursor: 'pointer',
                    boxShadow: isDark ? '0 6px 18px rgba(0,0,0,0.6)' : '0 4px 14px rgba(0,0,0,0.08)',
                    transition: 'transform 0.25s ease, box-shadow 0.25s ease'
                  }}
                  title={movie.title}
                  onClick={() => handleSearchMovie(movie.title)}
                >
                  <div className="imghvr" style={{ position: 'relative' }}>
                    <img
                      className="card-img-top"
                      style={{ width: '100%', height: '320px', objectFit: 'cover' }}
                      alt={`${movie.title} - poster`}
                      src={movie.poster || 'https://via.placeholder.com/240x360?text=No+Poster'}
                    />

                    {/* Release Date Badge */}
                    {movie.release_date && (
                      <div
                        style={{
                          position: 'absolute',
                          top: '10px',
                          left: '10px',
                          backgroundColor: 'rgba(0, 0, 0, 0.78)',
                          color: '#ffffff',
                          padding: '4px 10px',
                          borderRadius: '10px',
                          fontSize: '12px',
                          fontWeight: '600',
                          backdropFilter: 'blur(4px)'
                        }}
                      >
                        📅 {movie.release_date}
                      </div>
                    )}

                    {movie.rating > 0 && (
                      <div
                        style={{
                          position: 'absolute',
                          top: '10px',
                          right: '10px',
                          backgroundColor: 'rgba(0, 0, 0, 0.78)',
                          color: '#ffd700',
                          padding: '4px 10px',
                          borderRadius: '12px',
                          fontSize: '13px',
                          fontWeight: 'bold',
                          backdropFilter: 'blur(4px)'
                        }}
                      >
                        ★ {movie.rating}
                      </div>
                    )}

                    <figcaption className="fig">
                      <button className="card-btn btn btn-danger" style={{ backgroundColor: '#e50914', borderColor: '#e50914' }}>
                        Explore Movie
                      </button>
                    </figcaption>
                  </div>
                  
                  <div className="card-body" style={{ padding: '14px 12px', textAlign: 'center' }}>
                    <h5
                      className="card-title"
                      style={{
                        fontSize: '16px',
                        fontWeight: 'bold',
                        margin: 0,
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}
                    >
                      {movie.title}
                    </h5>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* 3. 🎭 POPULAR GENRES SECTION */}
          <section className="home-section" style={{ marginBottom: '40px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '20px', borderBottom: `2px solid ${isDark ? '#2a2a2a' : '#eeeeee'}`, paddingBottom: '12px' }}>
              <div>
                <h3 style={{ color: headingColor, fontWeight: 'bold', margin: 0, fontSize: '26px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ color: '#e50914' }}>🎭</span> Explore by Genre
                </h3>
                <h5 style={{ color: subtitleColor, margin: '6px 0 0 0', fontSize: '15px', fontWeight: 'normal' }}>
                  Browse hand-picked collections across 11 iconic movie categories
                </h5>
              </div>
              <button
                onClick={() => navigate('/genres')}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#e50914',
                  fontSize: '15px',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  padding: 0
                }}
              >
                View All Genres →
              </button>
            </div>

            <div className="home-genres-grid">
              {GENRES_DATA.map((genre) => (
                <div
                  key={genre.id}
                  className="home-genre-card"
                  onClick={() => navigate(`/genres/${genre.id}`)}
                  style={{
                    position: 'relative',
                    borderRadius: '16px',
                    overflow: 'hidden',
                    height: '140px',
                    cursor: 'pointer',
                    boxShadow: '0 4px 15px rgba(0,0,0,0.3)',
                    transition: 'transform 0.25s ease, box-shadow 0.25s ease'
                  }}
                >
                  <img
                    src={genre.image}
                    alt={genre.name}
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      filter: 'brightness(65%)'
                    }}
                  />
                  <div
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: 'linear-gradient(to top, rgba(0,0,0,0.75) 0%, transparent 100%)'
                    }}
                  >
                    <span
                      style={{
                        color: '#ffffff',
                        fontSize: '20px',
                        fontWeight: 'bold',
                        letterSpacing: '0.5px',
                        textShadow: '0 2px 8px rgba(0,0,0,0.8)'
                      }}
                    >
                      {genre.name}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </section>

        </div>
      )}
    </div>
  );
}
