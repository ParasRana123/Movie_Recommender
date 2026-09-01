import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import Loader from '../components/Loader';
import { useTheme } from '../context/ThemeContext';
import { fetchGenreMovies } from '../api/movieApi';
import { GENRES_DATA } from '../data/genresData';

export default function GenreDetailPage() {
  const { genreId } = useParams();
  const navigate = useNavigate();
  const { isDark } = useTheme();

  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);

  const cleanGenreId = (genreId || 'action').toLowerCase().replace('-', '_');
  const genreMeta = GENRES_DATA.find(g => g.id === cleanGenreId) || {
    id: cleanGenreId,
    name: cleanGenreId.toUpperCase(),
    banner: '/images/action1.jpg',
    heading: cleanGenreId.toUpperCase(),
    description: `The ${cleanGenreId} genre features exciting entertainment and engaging storylines.`
  };

  useEffect(() => {
    let isMounted = true;
    setLoading(true);

    fetchGenreMovies(cleanGenreId)
      .then(data => {
        if (isMounted) {
          setMovies(data.movies || []);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }
      })
      .catch(err => {
        if (isMounted) {
          console.error(err);
        }
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => { isMounted = false; };
  }, [cleanGenreId]);

  return (
    <div
      id="content"
      style={{
        backgroundColor: isDark ? '#121212' : '#f8f9fa',
        minHeight: '100vh',
        color: isDark ? '#ffffff' : '#181818',
        paddingBottom: '60px',
        transition: 'background-color 0.3s ease, color 0.3s ease'
      }}
    >
      <Navbar onSearchMovie={(title) => navigate(`/movie/${encodeURIComponent(title)}`)} />

      {loading && <Loader />}

      <div id="genre-main-content">
        {/* Genre Banner */}
        <div
          className="container1"
          style={{
            backgroundColor: isDark ? '#1a1a1a' : '#ffffff',
            border: `1px solid ${isDark ? '#2e2e2e' : '#e2e8f0'}`,
            boxShadow: isDark ? '0 10px 30px rgba(0,0,0,0.5)' : '0 6px 20px rgba(0,0,0,0.06)'
          }}
        >
          <div className="genre-banner-wrapper">
            <img
              src={genreMeta.banner || genreMeta.image}
              className="responsive-img"
              alt={genreMeta.name}
            />
          </div>
          <div className="text">
            <h2 className="heading" style={{ color: isDark ? '#ffffff' : '#181818' }}>
              {genreMeta.heading}
            </h2>
            <p className="description" style={{ color: isDark ? '#cccccc' : '#4a4a4a' }}>
              {genreMeta.description}
            </p>
          </div>
        </div>

        <hr style={{ borderColor: isDark ? '#2a2a2a' : '#e2e8f0', margin: '30px auto', maxWidth: '1400px' }} />

        <center>
          <h2 style={{ color: '#e50914', marginTop: '20px', fontWeight: 'bold' }}>Popular Movies</h2>
        </center>
        <center>
          <p style={{ color: isDark ? '#aaaaaa' : '#666666', marginBottom: '25px' }}>
            (Popular {genreMeta.name} related movies)
          </p>
        </center>

        {/* Popular Movies (Grid on Desktop, 1-Line Horizontal Scroll on Mobile) */}
        <div id="movies-container" className="genre-movies-scroll">
          {movies && movies.length > 0 && (
            movies.map((m, idx) => (
              <div
                key={idx}
                className="movie-card-genre"
                onClick={() => navigate(`/movie/${encodeURIComponent(m.title)}`)}
                title={m.title}
                style={{
                  backgroundColor: isDark ? '#1f1f1f' : '#ffffff',
                  borderColor: isDark ? '#333333' : '#e2e8f0'
                }}
              >
                <div className="imghvr">
                  <img
                    src={m.poster || 'https://via.placeholder.com/240x320?text=No+Poster'}
                    alt={`${m.title} poster`}
                    onError={(e) => { e.target.src = 'https://via.placeholder.com/240x320?text=No+Poster'; }}
                  />
                  <figcaption className="fig">
                    <button className="card-btn btn btn-danger" style={{ backgroundColor: '#e50914', borderColor: '#e50914', fontWeight: 'bold' }}>
                      Explore Movie
                    </button>
                  </figcaption>
                </div>
                {m.vote_average && m.vote_average !== 'N/A' && m.vote_average !== 0 && m.vote_average !== '0' && (
                  <p>★ {m.vote_average}</p>
                )}
                <h3 style={{ color: isDark ? '#ffffff' : '#181818' }}>{m.title}</h3>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
