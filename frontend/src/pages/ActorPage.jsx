import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import Loader from '../components/Loader';
import MovieCard from '../components/MovieCard';
import { fetchActorDetails } from '../api/movieApi';

export default function ActorPage() {
  const { actorId } = useParams();
  const navigate = useNavigate();
  const [actorData, setActorData] = useState(null);
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!actorId) return;
    let isMounted = true;
    setLoading(true);
    setError(null);

    fetchActorDetails(actorId)
      .then(data => {
        if (isMounted) {
          setActorData(data.actor);
          setMovies(data.movies || []);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }
      })
      .catch(err => {
        if (isMounted) {
          setError(err.message || 'Could not load actor details');
        }
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => { isMounted = false; };
  }, [actorId]);

  return (
    <div className="actor-page-container">
      <Navbar onSearchMovie={(title) => navigate(`/movie/${encodeURIComponent(title)}`)} />

      {loading && <Loader text="FETCHING ACTOR PROFILE & FILMOGRAPHY..." />}

      {error && !loading && (
        <div className="search-error-banner">
          <div className="error-content">
            <h3>⚠️ Error</h3>
            <p>{error}</p>
          </div>
        </div>
      )}

      {actorData && !loading && (
        <main className="actor-main-content">
          {/* Actor Profile Banner */}
          <section className="actor-profile-card">
            <div className="actor-profile-image-col">
              <img
                src={actorData.profile}
                alt={actorData.name}
                className="actor-large-photo"
                onError={(e) => { e.target.src = 'https://via.placeholder.com/300x450?text=No+Photo'; }}
              />
            </div>

            <div className="actor-details-col">
              <span className="actor-dept-badge">{actorData.known_for_department || 'Acting'}</span>
              <h1 className="actor-full-name">{actorData.name}</h1>

              <div className="actor-meta-row">
                {actorData.birthday && actorData.birthday !== 'Unknown' && (
                  <div className="actor-meta-item">
                    <strong>Born:</strong> {actorData.birthday}
                  </div>
                )}
                {actorData.birth_place && actorData.birth_place !== 'Unknown' && (
                  <div className="actor-meta-item">
                    <strong>Birthplace:</strong> {actorData.birth_place}
                  </div>
                )}
              </div>

              <div className="actor-bio-block">
                <h3 className="section-mini-heading">BIOGRAPHY</h3>
                <p className="actor-bio-text">{actorData.biography}</p>
              </div>
            </div>
          </section>

          {/* Filmography Grid */}
          <section className="actor-filmography-section">
            <div className="section-header-row">
              <h2 className="showcase-heading">🎬 Known For ({movies.length} Movies)</h2>
            </div>

            {movies.length > 0 ? (
              <div className="movies-grid">
                {movies.map((m, idx) => (
                  <MovieCard
                    key={idx}
                    title={m.title}
                    poster={m.poster}
                    rating={m.rating}
                  />
                ))}
              </div>
            ) : (
              <p className="empty-text">No movie credits found.</p>
            )}
          </section>
        </main>
      )}
    </div>
  );
}
