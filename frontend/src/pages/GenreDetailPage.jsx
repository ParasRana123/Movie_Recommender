import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import Loader from '../components/Loader';
import MovieCard from '../components/MovieCard';
import { fetchGenreMovies } from '../api/movieApi';
import { GENRES_DATA } from '../data/genresData';

export default function GenreDetailPage() {
  const { genreId } = useParams();
  const navigate = useNavigate();

  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const cleanGenreId = (genreId || 'action').toLowerCase().replace('-', '_');
  const genreMeta = GENRES_DATA.find(g => g.id === cleanGenreId) || {
    id: cleanGenreId,
    name: cleanGenreId.toUpperCase(),
    banner: '/images/action.jpg',
    heading: cleanGenreId.toUpperCase(),
    description: `Explore our collection of popular ${cleanGenreId} movies.`
  };

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    fetchGenreMovies(cleanGenreId)
      .then(data => {
        if (isMounted) {
          setMovies(data.movies || []);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }
      })
      .catch(err => {
        if (isMounted) {
          console.error('Error fetching genre movies:', err);
          setError(err.message || 'Could not load movies for this genre');
        }
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => { isMounted = false; };
  }, [cleanGenreId]);

  return (
    <div className="genre-detail-page-container">
      <Navbar onSearchMovie={(title) => navigate(`/movie/${encodeURIComponent(title)}`)} />

      {loading && <Loader text={`LOADING POPULAR ${genreMeta.name.toUpperCase()} MOVIES...`} />}

      <main className="genre-detail-main-content">
        {/* Genre Banner Section */}
        <section className="genre-banner-card">
          <div className="genre-banner-image-wrapper">
            <img
              src={genreMeta.banner}
              alt={genreMeta.name}
              className="genre-banner-img"
              onError={(e) => { e.target.src = genreMeta.image; }}
            />
          </div>
          <div className="genre-banner-text-col">
            <h1 className="genre-banner-heading">{genreMeta.heading}</h1>
            <p className="genre-banner-description">{genreMeta.description}</p>
          </div>
        </section>

        <hr className="genre-divider" />

        {/* Popular Movies in Genre */}
        <section className="genre-movies-section">
          <center>
            <h2 className="genre-section-title">Popular Movies</h2>
            <p className="genre-section-subtitle">(Trending in {genreMeta.name} Movies)</p>
          </center>

          {error && (
            <div className="search-error-banner">
              <p>{error}</p>
            </div>
          )}

          {!loading && movies.length > 0 && (
            <div className="movies-grid">
              {movies.map((m, idx) => (
                <MovieCard
                  key={idx}
                  title={m.title}
                  poster={m.poster}
                  rating={m.vote_average}
                />
              ))}
            </div>
          )}

          {!loading && movies.length === 0 && !error && (
            <p className="empty-text">No movies found for this category.</p>
          )}
        </section>
      </main>
    </div>
  );
}
