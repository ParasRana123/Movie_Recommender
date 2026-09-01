import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import { useWatchlist } from '../context/WatchlistContext';
import { useTheme } from '../context/ThemeContext';
import { fetchMovieDetails } from '../api/movieApi';

export default function WatchlistPage() {
  const { watchlist, removeFromWatchlist } = useWatchlist();
  const { isDark } = useTheme();
  const navigate = useNavigate();

  // Cache for enriched movie details (overview, director, casts, runtime, etc.)
  const [detailsCache, setDetailsCache] = useState({});

  useEffect(() => {
    let isMounted = true;
    if (!watchlist || watchlist.length === 0) return;

    // Asynchronously fetch full details for any movie missing overview or director or runtime
    watchlist.forEach((movie) => {
      if (!movie || !movie.title) return;
      const key = movie.title.toLowerCase();

      const isMissingData = !movie.overview || !movie.director || !movie.runtime || !movie.casts;
      if (isMissingData && !detailsCache[key]) {
        fetchMovieDetails(movie.title)
          .then((data) => {
            if (isMounted && data) {
              setDetailsCache((prev) => ({
                ...prev,
                [key]: {
                  overview: data.overview || '',
                  director: data.director_name || '',
                  runtime: data.runtime || '',
                  release_date: data.release_date || '',
                  genres: data.genres || data.genres_str || '',
                  casts: Array.isArray(data.casts)
                    ? data.casts.map((c) => (typeof c === 'string' ? c : (c.name || ''))).filter(Boolean)
                    : (data.casts && typeof data.casts === 'object' ? Object.keys(data.casts) : [])
                }
              }));
            }
          })
          .catch(() => {});
      }
    });

    return () => {
      isMounted = false;
    };
  }, [watchlist, detailsCache]);

  const formatRuntime = (runtimeStr) => {
    if (!runtimeStr || runtimeStr === 'N/A') return '';
    if (typeof runtimeStr === 'number' && runtimeStr > 0) {
      const h = Math.floor(runtimeStr / 60);
      const m = runtimeStr % 60;
      return h > 0 ? `${h}h ${m}m` : `${m}m`;
    }
    const str = String(runtimeStr).trim();
    const hourMatch = str.match(/(\d+)\s*h(?:our)?/i);
    const minMatch = str.match(/(\d+)\s*m(?:in)?/i);
    if (hourMatch || minMatch) {
      const h = hourMatch ? hourMatch[1] : '0';
      const m = minMatch ? minMatch[1] : '0';
      return Number(h) > 0 ? `${h}h ${m}m` : `${m}m`;
    }
    const num = parseInt(str, 10);
    if (!isNaN(num) && num > 0) {
      const h = Math.floor(num / 60);
      const m = num % 60;
      return h > 0 ? `${h}h ${m}m` : `${m}m`;
    }
    return str;
  };

  const formatYear = (dateStr) => {
    if (!dateStr || dateStr === 'Unknown Date') return '';
    const match = String(dateStr).match(/\b(19\d\d|20\d\d)\b/);
    return match ? match[1] : dateStr;
  };

  return (
    <div
      style={{
        backgroundColor: isDark ? '#121212' : '#f8f9fa',
        minHeight: '100vh',
        color: isDark ? '#ffffff' : '#181818',
        transition: 'background-color 0.3s ease, color 0.3s ease'
      }}
    >
      <Navbar onSearchMovie={(title) => navigate(`/movie/${encodeURIComponent(title)}`)} />

      <div className="your_watchlist">
        <h1>Your Watchlist</h1>
        <p className="watchlist-subtitle">
          {watchlist && watchlist.length > 0
            ? `You have saved ${watchlist.length} movie${watchlist.length === 1 ? '' : 's'} to your list.`
            : "Here are all the movies you've saved. Search for movies to start building your personal collection."}
        </p>

        <div id="watchlist-container">
          {watchlist && watchlist.length > 0 ? (
            watchlist.map((movie, idx) => {
              const key = movie.title ? movie.title.toLowerCase() : '';
              const extra = detailsCache[key] || {};

              const title = movie.title || 'Unknown Title';
              const poster =
                movie.poster ||
                movie.poster_path ||
                extra.poster ||
                'https://via.placeholder.com/140x210?text=No+Poster';

              const releaseYear = formatYear(movie.release_date || extra.release_date);
              const runtimeFormatted = formatRuntime(movie.runtime || extra.runtime);
              const genres = movie.genres || extra.genres || '';
              const genresFormatted =
                typeof genres === 'string'
                  ? genres
                  : Array.isArray(genres)
                  ? genres.join(', ')
                  : '';

              const overview = movie.overview || extra.overview || '';
              const director = movie.director || movie.director_name || extra.director || '';
              const rawCasts = movie.casts || movie.stars || extra.casts || [];
              const starsList = Array.isArray(rawCasts)
                ? rawCasts.map((c) => (typeof c === 'string' ? c : (c.name || ''))).filter(Boolean)
                : typeof rawCasts === 'string'
                ? rawCasts.split(',').map((s) => s.trim()).filter(Boolean)
                : [];

              return (
                <div key={idx} className="watchlist-card">
                  {/* Poster Thumbnail with Gold Bookmark Ribbon */}
                  <div
                    className="watchlist-poster-wrap"
                    onClick={() => navigate(`/movie/${encodeURIComponent(title)}`)}
                    title={`View "${title}"`}
                  >
                    <div className="watchlist-poster-ribbon" title="In your Watchlist">
                      <svg
                        width="13"
                        height="13"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="3.2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                    </div>
                    <img
                      src={poster}
                      alt={title}
                      className="watchlist-poster-img"
                      loading="lazy"
                      onError={(e) => {
                        e.target.src = 'https://via.placeholder.com/140x210?text=No+Poster';
                      }}
                    />
                  </div>

                  {/* Card Content */}
                  <div className="watchlist-info-wrap">
                    {/* Header Row: Title and Top-Right Action Buttons */}
                    <div className="watchlist-header-row">
                      <h2
                        className="watchlist-card-title"
                        onClick={() => navigate(`/movie/${encodeURIComponent(title)}`)}
                      >
                        <span className="watchlist-card-idx">{idx + 1}. </span>
                        <span className="watchlist-card-name">{title}</span>
                      </h2>

                      <div className="watchlist-card-actions">
                        <button
                          type="button"
                          className="watchlist-action-btn watchlist-info-btn"
                          onClick={() => navigate(`/movie/${encodeURIComponent(title)}`)}
                          title="View Details & Recommendations"
                          aria-label="View Details"
                        >
                          <svg
                            width="18"
                            height="18"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2.2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="16" x2="12" y2="12"></line>
                            <line x1="12" y1="8" x2="12.01" y2="8"></line>
                          </svg>
                        </button>
                        <button
                          type="button"
                          className="watchlist-action-btn watchlist-delete-btn"
                          onClick={() => removeFromWatchlist(title)}
                          title="Remove from Watchlist"
                          aria-label="Remove from Watchlist"
                        >
                          <svg
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2.2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                          </svg>
                        </button>
                      </div>
                    </div>

                    {/* Metadata Subheader: Year • Runtime • Genres */}
                    {(releaseYear || runtimeFormatted || genresFormatted) && (
                      <div className="watchlist-meta-row">
                        {releaseYear && <span className="meta-year">{releaseYear}</span>}
                        {releaseYear && runtimeFormatted && <span className="meta-dot">•</span>}
                        {runtimeFormatted && <span className="meta-runtime">{runtimeFormatted}</span>}
                        {(releaseYear || runtimeFormatted) && genresFormatted && (
                          <span className="meta-dot">•</span>
                        )}
                        {genresFormatted && <span className="meta-genres">{genresFormatted}</span>}
                      </div>
                    )}

                    {/* Plot Overview */}
                    {overview ? (
                      <p className="watchlist-card-overview">{overview}</p>
                    ) : (
                      <p className="watchlist-card-overview muted-placeholder">
                        Plot summary loading or unavailable.
                      </p>
                    )}

                    {/* Key Credits: Director & Stars */}
                    {(director || starsList.length > 0) && (
                      <div className="watchlist-credits-row">
                        {director && (
                          <div className="watchlist-credit-group">
                            <span className="credit-label">Director</span>
                            <span
                              className="credit-name highlight-link"
                              onClick={() => navigate(`/movie/${encodeURIComponent(title)}`)}
                            >
                              {director}
                            </span>
                          </div>
                        )}
                        {starsList.length > 0 && (
                          <div className="watchlist-credit-group">
                            <span className="credit-label">Stars</span>
                            <span className="credit-name">
                              {starsList.slice(0, 4).map((star, sIdx) => (
                                <span
                                  key={sIdx}
                                  className="highlight-link"
                                  onClick={() => navigate(`/movie/${encodeURIComponent(title)}`)}
                                >
                                  {star}
                                  {sIdx < Math.min(starsList.length, 4) - 1 ? ', ' : ''}
                                </span>
                              ))}
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })
          ) : (
            <div className="watchlist-empty-state">
              <img
                src="/images/add_bookmark.svg"
                width="64"
                height="64"
                style={{ filter: isDark ? 'invert(1)' : 'none', opacity: 0.35, marginBottom: '20px' }}
                alt="Empty Watchlist"
              />
              <h3 style={{ color: isDark ? '#ffffff' : '#181818', marginBottom: '10px', fontWeight: 'bold' }}>
                Your Watchlist is empty
              </h3>
              <p
                style={{
                  color: isDark ? '#888888' : '#666666',
                  maxWidth: '480px',
                  margin: '0 auto 25px auto',
                  fontSize: '15px',
                  lineHeight: '1.5'
                }}
              >
                You haven't saved any movies yet. Explore recommendations and click the bookmark button on any movie to save it here!
              </p>
              <button
                className="btn btn-primary movie-button"
                style={{
                  backgroundColor: '#e50914',
                  borderColor: '#e50914',
                  borderRadius: '6px',
                  padding: '10px 28px',
                  fontWeight: 'bold',
                  fontSize: '15px',
                  cursor: 'pointer',
                  boxShadow: '0 4px 12px rgba(229, 9, 20, 0.4)'
                }}
                onClick={() => navigate('/')}
              >
                Explore Movies
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
