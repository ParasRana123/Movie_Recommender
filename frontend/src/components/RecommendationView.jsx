import React from 'react';
import { useWatchlist } from '../context/WatchlistContext';
import CastCard from './CastCard';
import MovieCard from './MovieCard';

export default function RecommendationView({ movieData, onSelectRecommendedMovie }) {
  const { isInWatchlist, addToWatchlist, removeFromWatchlist } = useWatchlist();

  if (!movieData) return null;

  const {
    title,
    poster,
    backdrop,
    overview,
    vote_average,
    vote_count,
    release_date,
    runtime,
    status,
    budget,
    revenue,
    original_language,
    genres = [],
    trailer,
    teaser,
    streaming_availability = [],
    director_name,
    director_image,
    director_bio,
    director_birthplace,
    casts = [],
    reviews = [],
    recommended_movies = []
  } = movieData;

  const isSaved = isInWatchlist(title);

  const handleWatchlistToggle = () => {
    if (isSaved) {
      removeFromWatchlist(title);
    } else {
      addToWatchlist({
        title,
        poster,
        rating: vote_average,
        release_date,
        genres: Array.isArray(genres) ? genres.join(', ') : genres,
        runtime
      });
    }
  };

  return (
    <div className="recommendation-view">
      {/* 1. Hero Movie Overview Section */}
      <section className="movie-hero-section">
        {/* Backdrop image with blur & gradient */}
        <div
          className="hero-backdrop"
          style={{ backgroundImage: `url('${backdrop || poster}')` }}
        />
        <div className="hero-gradient-overlay" />

        <div className="hero-content-container">
          <div className="hero-poster-wrapper">
            <img
              src={poster || 'https://via.placeholder.com/500x750?text=No+Poster'}
              alt={title}
              className="hero-poster-img"
            />
          </div>

          <div className="hero-details">
            <div className="hero-header-row">
              <h1 className="hero-movie-title">{title}</h1>
              <button
                onClick={handleWatchlistToggle}
                className={`hero-watchlist-btn ${isSaved ? 'saved' : ''}`}
                title={isSaved ? 'Remove from Watchlist' : 'Add to Watchlist'}
              >
                <img
                  src={isSaved ? '/images/bookmark_tick.svg' : '/images/add_bookmark.svg'}
                  alt="Watchlist"
                  width="22"
                  height="22"
                />
                <span>{isSaved ? 'In WatchList' : 'Add to WatchList'}</span>
              </button>
            </div>

            {/* Quick Badges Row */}
            <div className="hero-badges-row">
              {vote_average && vote_average !== 'N/A' && (
                <span className="badge badge-rating">★ {vote_average} / 10</span>
              )}
              {vote_count && vote_count !== '0' && (
                <span className="badge badge-votes">👥 {vote_count} votes</span>
              )}
              {runtime && runtime !== 'N/A' && (
                <span className="badge badge-runtime">⏱ {runtime}</span>
              )}
              {release_date && release_date !== 'Unknown Date' && (
                <span className="badge badge-date">📅 {release_date}</span>
              )}
              {status && (
                <span className="badge badge-status">🏷 {status}</span>
              )}
            </div>

            {/* Genres Tag Cloud */}
            {genres && genres.length > 0 && (
              <div className="hero-genres-cloud">
                {genres.map((g, i) => (
                  <span key={i} className="genre-pill">{g}</span>
                ))}
              </div>
            )}

            {/* Plot Overview */}
            <div className="hero-overview-block">
              <h3 className="section-mini-heading">OVERVIEW</h3>
              <p className="overview-text">{overview}</p>
            </div>

            {/* Extra Metadata Table */}
            <div className="hero-meta-grid">
              {budget && budget !== 'N/A' && (
                <div className="meta-item">
                  <span className="meta-label">Budget:</span>
                  <span className="meta-value">${budget}</span>
                </div>
              )}
              {revenue && revenue !== 'N/A' && (
                <div className="meta-item">
                  <span className="meta-label">Revenue:</span>
                  <span className="meta-value">${revenue}</span>
                </div>
              )}
              {original_language && (
                <div className="meta-item">
                  <span className="meta-label">Language:</span>
                  <span className="meta-value">{original_language}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* 2. Streaming Providers */}
      {streaming_availability && streaming_availability.length > 0 && (
        <section className="streaming-section">
          <h3 className="section-title">Where to Stream</h3>
          <div className="streaming-providers-row">
            {streaming_availability.map((prov, i) => (
              <div key={i} className="provider-badge" title={prov.provider_name}>
                <img src={prov.logo_path} alt={prov.provider_name} className="provider-logo" />
                <span>{prov.provider_name}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* 3. Official Trailers & Teasers */}
      {(trailer || teaser) && (
        <section className="trailers-section">
          <h3 className="section-title">Watch Trailers & Clips</h3>
          <div className="trailers-container">
            {trailer && (
              <div className="video-player-wrapper">
                <h4>Official Trailer</h4>
                <div className="video-responsive">
                  <iframe
                    src={trailer}
                    title={`${title} Trailer`}
                    frameBorder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                  />
                </div>
              </div>
            )}
            {teaser && (
              <div className="video-player-wrapper">
                <h4>Official Teaser</h4>
                <div className="video-responsive">
                  <iframe
                    src={teaser}
                    title={`${title} Teaser`}
                    frameBorder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                  />
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      {/* 4. Director Spotlight */}
      {director_name && director_name !== 'Unknown' && (
        <section className="director-spotlight-section">
          <div className="director-card">
            <img
              src={director_image || 'https://via.placeholder.com/300?text=No+Photo'}
              alt={director_name}
              className="director-avatar"
            />
            <div className="director-info">
              <span className="director-role-tag">DIRECTOR</span>
              <h3 className="director-name">{director_name}</h3>
              {director_birthplace && director_birthplace !== 'Unknown' && (
                <p className="director-birthplace">Born in {director_birthplace}</p>
              )}
              <p className="director-bio">{director_bio}</p>
            </div>
          </div>
        </section>
      )}

      {/* 5. Top Cast */}
      {casts && casts.length > 0 && (
        <section className="cast-section">
          <h3 className="section-title">Top Cast</h3>
          <div className="cast-grid">
            {casts.map((c, i) => (
              <CastCard
                key={i}
                id={c.id}
                name={c.name}
                character={c.character}
                profile={c.profile}
              />
            ))}
          </div>
        </section>
      )}

      {/* 6. User Reviews with NLP Sentiment Analysis */}
      {reviews && reviews.length > 0 && (
        <section className="reviews-section">
          <h3 className="section-title">Audience Reviews & Sentiment Analysis</h3>
          <div className="reviews-list">
            {reviews.map((rev, i) => (
              <div key={i} className="review-card">
                <div className="review-header">
                  <div className="reviewer-info">
                    <span className="reviewer-avatar">👤</span>
                    <strong className="reviewer-name">{rev.author}</strong>
                    {rev.rating && (
                      <span className="review-rating-badge">★ {rev.rating}</span>
                    )}
                  </div>
                  <div className="sentiment-pill-wrapper">
                    <span className={`sentiment-badge ${rev.sentiment === 'Good' ? 'good' : 'bad'}`}>
                      {rev.sentiment === 'Good' ? '👍 Positive' : '👎 Critical'}
                    </span>
                    {rev.confidence && (
                      <span className="sentiment-confidence">{rev.confidence} match</span>
                    )}
                  </div>
                </div>
                <p className="review-text">{rev.content}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* 7. Recommended Movies */}
      {recommended_movies && recommended_movies.length > 0 && (
        <section className="recommended-section">
          <h3 className="section-title">Recommended Movies (Based on Machine Learning)</h3>
          <p className="recommended-subtitle">Movies with similar plot lines, themes, directors, and genres</p>
          <div className="recommended-movies-grid">
            {recommended_movies.map((m, i) => (
              <MovieCard
                key={i}
                title={m.title}
                poster={m.poster}
                rating={m.vote_average}
                onClick={() => {
                  if (onSelectRecommendedMovie) {
                    onSelectRecommendedMovie(m.title);
                  }
                }}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
