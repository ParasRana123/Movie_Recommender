import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import { fetchGenreMovies } from '../api/movieApi';
import { GENRES_DATA } from '../data/genresData';

export default function GenreDetailPage() {
  const { genreId } = useParams();
  const navigate = useNavigate();

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
    <div style={{ backgroundColor: 'black', minHeight: '100vh', color: '#ffffff' }}>
      <Navbar onSearchMovie={(title) => navigate(`/movie/${encodeURIComponent(title)}`)} />

      {loading && (
        <div id="loader" style={{ display: 'block', textAlign: 'center', padding: '40px' }}>
          <p id="loader-text" style={{ color: 'white', fontSize: '20px', fontWeight: 800 }}>LOADING...</p>
        </div>
      )}

      <div id="genre-main-content">
        <div className="container1">
          <div className="img" style={{ position: 'static', opacity: 1, backdropFilter: 'none' }}>
            <img
              src={genreMeta.banner || genreMeta.image}
              className="responsive-img"
              alt={genreMeta.name}
            />
          </div>
          <div className="text">
            <h2 className="heading">{genreMeta.heading}</h2>
            <p className="description">{genreMeta.description}</p>
          </div>
        </div>

        <hr style={{ borderColor: '#333' }} />

        <center><h2 style={{ color: 'red', marginTop: '20px' }}>Popular Movies</h2></center>
        <center><p style={{ color: 'white' }}>(Popular {genreMeta.name} related movies)</p></center>

        <div id="movies-container">
          {movies && movies.length > 0 && (
            movies.map((m, idx) => (
              <div
                key={idx}
                className="movie-card-genre"
                onClick={() => navigate(`/movie/${encodeURIComponent(m.title)}`)}
              >
                <img
                  src={m.poster || 'https://via.placeholder.com/240x320?text=No+Poster'}
                  alt={`${m.title} poster`}
                />
                {m.vote_average && m.vote_average !== 'N/A' && (
                  <p>★ {m.vote_average}</p>
                )}
                <h3>{m.title}</h3>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
