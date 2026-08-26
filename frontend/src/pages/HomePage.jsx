import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import RecommendationView from '../components/RecommendationView';
import Loader from '../components/Loader';
import { fetchRecommendations, fetchTrendingMovies, fetchUpcomingMovies, fetchTrendingPeople } from '../api/movieApi';
import { GENRES_DATA } from '../data/genresData';
import { useTheme } from '../context/ThemeContext';

export default function HomePage() {
  const [activeMovieData, setActiveMovieData] = useState(null);
  const [trendingMovies, setTrendingMovies] = useState([]);
  const [upcomingMovies, setUpcomingMovies] = useState([]);
  const [trendingPeople, setTrendingPeople] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [feedLoading, setFeedLoading] = useState(true);

  const { isDark } = useTheme();
  const navigate = useNavigate();

  // Load Trending Movies, Upcoming Releases, and Trending Celebrities on mount
  useEffect(() => {
    let mounted = true;
    setFeedLoading(true);

    Promise.allSettled([
      fetchTrendingMovies(),
      fetchUpcomingMovies(),
      fetchTrendingPeople()
    ])
      .then(([trendingRes, upcomingRes, peopleRes]) => {
        if (!mounted) return;
        if (trendingRes.status === 'fulfilled' && Array.isArray(trendingRes.value)) {
          setTrendingMovies(trendingRes.value);
        }
        if (upcomingRes.status === 'fulfilled' && Array.isArray(upcomingRes.value)) {
          setUpcomingMovies(upcomingRes.value);
        }
        if (peopleRes.status === 'fulfilled' && Array.isArray(peopleRes.value) && peopleRes.value.length > 0) {
          setTrendingPeople(peopleRes.value);
        } else {
          // Curated A-List Fallback Celebrities
          setTrendingPeople([
            { id: 6193, name: "Leonardo DiCaprio", profile: "https://image.tmdb.org/t/p/w500/wo2hxAzvBv2YXF1q2ZgWuhq2Uo5.jpg", known_for: ["Inception", "Titanic"] },
            { id: 3223, name: "Robert Downey Jr.", profile: "https://image.tmdb.org/t/p/w500/5qHNjhtjMD4YWH3ag0Y0kV99NJb.jpg", known_for: ["Iron Man", "Avengers"] },
            { id: 1245, name: "Scarlett Johansson", profile: "https://image.tmdb.org/t/p/w500/6NsMbJXRlDZuDzatNmakEBpt3Z7.jpg", known_for: ["Black Widow", "Lucy"] },
            { id: 500, name: "Tom Cruise", profile: "https://image.tmdb.org/t/p/w500/eOhwo2322aPVg44F4koc926c04f.jpg", known_for: ["Top Gun", "Mission: Impossible"] },
            { id: 234352, name: "Margot Robbie", profile: "https://image.tmdb.org/t/p/w500/euDPyqLnuagWMDo2XZAx0VStoxV.jpg", known_for: ["Barbie", "Wolf of Wall Street"] },
            { id: 2037, name: "Cillian Murphy", profile: "https://image.tmdb.org/t/p/w500/360RRAkJGzoHaVNT16Kd1w5vrhu.jpg", known_for: ["Oppenheimer", "Peaky Blinders"] },
            { id: 505710, name: "Zendaya", profile: "https://image.tmdb.org/t/p/w500/r2GQ1j3l6pM5c4cO6QZ2Y5X2d1u.jpg", known_for: ["Dune", "Euphoria"] },
            { id: 287, name: "Brad Pitt", profile: "https://image.tmdb.org/t/p/w500/cckcYc2v0yh1tc9QGRvcZ2q35Ky.jpg", known_for: ["Fight Club", "Seven"] },
            { id: 54693, name: "Emma Stone", profile: "https://image.tmdb.org/t/p/w500/cZ8a34v3v23pWpA9Z0X5Y6k2s3A.jpg", known_for: ["La La Land", "Poor Things"] },
            { id: 30614, name: "Ryan Gosling", profile: "https://image.tmdb.org/t/p/w500/4SYZwA4GZ6228sL1D2d3E4f5G6h.jpg", known_for: ["Drive", "Blade Runner 2049"] }
          ]);
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

      {/* Front Page Discovery Feed (Trending, Upcoming, Trending People, Genres) */}
      {!activeMovieData && !loading && (
        <div className="home-feed-container" style={{ maxWidth: '1400px', margin: '0 auto', padding: '20px 25px' }}>
          
          {/* App Hero Heading (Centered) */}
          <center>
            <h1 style={{ marginTop: '30px', marginBottom: '8px', fontSize: '38px', fontWeight: 'bold', letterSpacing: '1px', color: headingColor }}>
              MOVIE RECOMMENDATION SYSTEM
            </h1>
            <p style={{ color: subtitleColor, fontSize: '16px', marginBottom: '45px' }}>
              Discover top movies, real-time audience sentiments & AI-powered recommendations
            </p>
          </center>

          {/* 1. 🔥 TRENDING MOVIES SECTION (Left-aligned) */}
          <section className="home-section" style={{ marginBottom: '55px' }}>
            <div className="section-header-left" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', borderBottom: `2px solid ${isDark ? '#2a2a2a' : '#eeeeee'}`, paddingBottom: '12px' }}>
              <div style={{ textAlign: 'left' }}>
                <h3 style={{ color: headingColor, fontWeight: 'bold', margin: 0, fontSize: '26px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ color: '#e50914', fontSize: '28px', fontWeight: 'bold' }}>|</span> Trending Movies
                </h3>
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

          {/* 2. 🎬 UPCOMING MOVIES SECTION (Left-aligned) */}
          <section className="home-section" style={{ marginBottom: '55px' }}>
            <div className="section-header-left" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', borderBottom: `2px solid ${isDark ? '#2a2a2a' : '#eeeeee'}`, paddingBottom: '12px' }}>
              <div style={{ textAlign: 'left' }}>
                <h3 style={{ color: headingColor, fontWeight: 'bold', margin: 0, fontSize: '26px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ color: '#e50914', fontSize: '28px', fontWeight: 'bold' }}>|</span> Upcoming Movies
                </h3>
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

          {/* 3. ⭐ TRENDING CELEBRITIES SECTION (Left-aligned, Circular Format with Dark Hover & Mobile 1-Line Touch Scroll) */}
          {trendingPeople && trendingPeople.length > 0 && (
            <section className="home-section" style={{ marginBottom: '55px' }}>
              <div className="section-header-left" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', borderBottom: `2px solid ${isDark ? '#2a2a2a' : '#eeeeee'}`, paddingBottom: '12px' }}>
                <div style={{ textAlign: 'left' }}>
                  <h3 style={{ color: headingColor, fontWeight: 'bold', margin: 0, fontSize: '26px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ color: '#e50914', fontSize: '28px', fontWeight: 'bold' }}>|</span> Trending Celebrities
                  </h3>
                </div>
                <span style={{ color: '#e50914', fontSize: '14px', fontWeight: 'bold' }}>
                  Popular Stars
                </span>
              </div>

              <div className="movie-content cast-content-scroll home-people-scroll" style={{ justifyContent: 'flex-start' }}>
                {trendingPeople.slice(0, 12).map((person, idx) => (
                  <div
                    key={idx}
                    className="cast-card-item"
                    title={`Click to view ${person.name}'s page`}
                    onClick={() => navigate(`/actor/${person.id}`)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className="imghvr cast-imghvr">
                      <img
                        className="card-img-top cast-img"
                        alt={`${person.name} - profile`}
                        src={person.profile || 'https://via.placeholder.com/250x250?text=No+Photo'}
                      />
                      <figcaption className="img cast-fig-overlay">
                        <button
                          className="card-btn btn btn-danger"
                          style={{ backgroundColor: '#e50914', borderColor: '#e50914' }}
                        >
                          Know More
                        </button>
                      </figcaption>
                    </div>
                    <div className="card-body" style={{ textAlign: 'center', padding: '10px 4px' }}>
                      <h5 className="card-title" style={{ fontSize: '15px', fontWeight: 'bold', margin: '4px 0', color: headingColor }}>
                        {person.name}
                      </h5>
                      {person.known_for && person.known_for.length > 0 ? (
                        <h6 style={{ color: isDark ? '#aaa' : '#756969', fontSize: '12px', margin: '0 auto', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '140px' }}>
                          {person.known_for.slice(0, 2).join(', ')}
                        </h6>
                      ) : (
                        <h6 style={{ color: isDark ? '#aaa' : '#756969', fontSize: '12px', margin: 0 }}>
                          {person.known_for_department || 'Acting'}
                        </h6>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 4. 🎭 POPULAR GENRES SECTION (Left-aligned) */}
          <section className="home-section" style={{ marginBottom: '40px' }}>
            <div className="section-header-left" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', borderBottom: `2px solid ${isDark ? '#2a2a2a' : '#eeeeee'}`, paddingBottom: '12px' }}>
              <div style={{ textAlign: 'left' }}>
                <h3 style={{ color: headingColor, fontWeight: 'bold', margin: 0, fontSize: '26px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ color: '#e50914', fontSize: '28px', fontWeight: 'bold' }}>|</span> Explore by Genre
                </h3>
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
