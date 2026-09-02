import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser, SignInButton, SignUpButton } from '@clerk/clerk-react';
import Navbar from '../components/Navbar';
import { useWatchlist } from '../context/WatchlistContext';
import { useTheme } from '../context/ThemeContext';
import { fetchMovieDetails } from '../api/movieApi';

export default function WatchlistPage() {
  const { watchlist, removeFromWatchlist } = useWatchlist();
  const { isDark } = useTheme();
  const { isSignedIn, isLoaded: isAuthLoaded } = useUser();
  const navigate = useNavigate();

  // Cache for enriched movie details (overview, director, casts, runtime, etc.)
  const [detailsCache, setDetailsCache] = useState({});

  useEffect(() => {
    let isMounted = true;
    if (!isSignedIn || !watchlist || watchlist.length === 0) return;

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
                    ? data.casts.map((c) => ({
                        name: typeof c === 'string' ? c : (c.name || ''),
                        id: c && c.id ? String(c.id) : ''
                      })).filter((c) => Boolean(c.name))
                    : (data.casts && typeof data.casts === 'object'
                        ? Object.entries(data.casts).map(([name, val]) => ({
                            name,
                            id: Array.isArray(val) && val[0] ? String(val[0]) : ''
                          }))
                        : [])
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
  }, [watchlist, detailsCache, isSignedIn]);

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
      id="content"
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

        <div id="watchlist-container">
          {!isAuthLoaded ? (
            /* Auth Loading State */
            <div style={{ textAlign: 'center', padding: '60px 20px' }}>
              <div
                style={{
                  display: 'inline-block',
                  width: '36px',
                  height: '36px',
                  border: '3px solid rgba(229, 9, 20, 0.2)',
                  borderTopColor: '#e50914',
                  borderRadius: '50%',
                  animation: 'spin 0.8s linear infinite',
                  marginBottom: '15px'
                }}
              />
              <p style={{ color: isDark ? '#aaaaaa' : '#666666', fontSize: '15px' }}>Checking authentication...</p>
            </div>
          ) : !isSignedIn ? (
            /* Auth Required Gate for Signed-Out Users */
            <div
              className="watchlist-auth-gate"
              style={{
                maxWidth: '540px',
                margin: '20px auto',
                padding: '45px 30px',
                backgroundColor: isDark ? '#1a1a1a' : '#ffffff',
                border: `1px solid ${isDark ? '#2e2e2e' : '#e2e8f0'}`,
                borderRadius: '16px',
                boxShadow: isDark ? '0 10px 30px rgba(0, 0, 0, 0.5)' : '0 10px 25px rgba(0, 0, 0, 0.06)',
                textAlign: 'center'
              }}
            >
              <div
                style={{
                  width: '64px',
                  height: '64px',
                  borderRadius: '50%',
                  backgroundColor: 'rgba(229, 9, 20, 0.12)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 20px auto'
                }}
              >
                <img
                  src="/images/add_bookmark.svg"
                  alt="Watchlist"
                  width="30"
                  height="30"
                  style={{
                    filter: isDark ? 'invert(1)' : 'none',
                    opacity: 0.9
                  }}
                />
              </div>

              <h2
                style={{
                  fontSize: '24px',
                  fontWeight: '700',
                  color: isDark ? '#ffffff' : '#181818',
                  marginBottom: '12px'
                }}
              >
                Sign in to access your Watchlist
              </h2>

              <p
                style={{
                  fontSize: '15px',
                  color: isDark ? '#aaaaaa' : '#666666',
                  lineHeight: '1.6',
                  maxWidth: '440px',
                  margin: '0 auto 28px auto'
                }}
              >
                Please sign in or create an account to view and manage your saved movies, get personal recommendations, and sync across all devices.
              </p>

              <div style={{ display: 'flex', gap: '14px', justifyContent: 'center', flexWrap: 'wrap' }}>
                <SignInButton mode="modal">
                  <button
                    type="button"
                    className="btn btn-primary"
                    style={{
                      backgroundColor: '#e50914',
                      borderColor: '#e50914',
                      padding: '10px 28px',
                      borderRadius: '6px',
                      fontWeight: '700',
                      fontSize: '15px',
                      cursor: 'pointer',
                      boxShadow: '0 4px 14px rgba(229, 9, 20, 0.4)'
                    }}
                  >
                    Sign In
                  </button>
                </SignInButton>

                <SignUpButton mode="modal">
                  <button
                    type="button"
                    className="btn"
                    style={{
                      backgroundColor: 'transparent',
                      border: `1px solid ${isDark ? '#444444' : '#cccccc'}`,
                      color: isDark ? '#ffffff' : '#181818',
                      padding: '10px 24px',
                      borderRadius: '6px',
                      fontWeight: '600',
                      fontSize: '15px',
                      cursor: 'pointer'
                    }}
                  >
                    Create Account
                  </button>
                </SignUpButton>
              </div>
            </div>
          ) : watchlist && watchlist.length > 0 ? (
            /* Render Signed-In User's Watchlist */
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
                ? rawCasts.map((c) => {
                    if (typeof c === 'string') {
                      const matched = Array.isArray(extra.casts)
                        ? extra.casts.find((ec) => ec.name && ec.name.toLowerCase() === c.toLowerCase())
                        : null;
                      return { name: c, id: matched?.id || '' };
                    }
                    return {
                      name: c.name || '',
                      id: c.id ? String(c.id) : ''
                    };
                  }).filter((c) => Boolean(c.name))
                : typeof rawCasts === 'string'
                ? rawCasts.split(',').map((s) => {
                    const name = s.trim();
                    const matched = Array.isArray(extra.casts)
                      ? extra.casts.find((ec) => ec.name && ec.name.toLowerCase() === name.toLowerCase())
                      : null;
                    return { name, id: matched?.id || '' };
                  }).filter((c) => Boolean(c.name))
                : [];

              return (
                <div
                  key={idx}
                  className="watchlist-card"
                  style={{
                    backgroundColor: isDark ? '#1a1a1a' : '#ffffff',
                    borderColor: isDark ? '#2e2e2e' : '#e2e8f0',
                    boxShadow: isDark ? '0 4px 14px rgba(0, 0, 0, 0.5)' : '0 4px 12px rgba(0, 0, 0, 0.06)'
                  }}
                >
                  {/* Poster Thumbnail with Red Bookmark Ribbon */}
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
                        stroke="#ffffff"
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
                        <span className="watchlist-card-idx" style={{ color: isDark ? '#ffffff' : '#181818' }}>
                          {idx + 1}.{' '}
                        </span>
                        <span className="watchlist-card-name" style={{ color: isDark ? '#ffffff' : '#181818' }}>
                          {title}
                        </span>
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
                      <div className="watchlist-meta-row" style={{ color: isDark ? '#a0a0a0' : '#64748b' }}>
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
                      <p
                        className="watchlist-card-overview"
                        style={{ color: isDark ? '#cccccc' : '#334155' }}
                      >
                        {overview}
                      </p>
                    ) : (
                      <p
                        className="watchlist-card-overview muted-placeholder"
                        style={{ color: isDark ? '#888888' : '#718096' }}
                      >
                        Plot summary loading or unavailable.
                      </p>
                    )}

                    {/* Key Credits: Director & Stars */}
                    {(director || starsList.length > 0) && (
                      <div className="watchlist-credits-row">
                        {director && (
                          <div className="watchlist-credit-group">
                            <span className="credit-label" style={{ color: isDark ? '#ffffff' : '#181818' }}>
                              Director
                            </span>
                            <span
                              className="credit-name highlight-link"
                              style={{ color: isDark ? '#38bdf8' : '#0284c7', cursor: 'pointer' }}
                              onClick={(e) => {
                                e.stopPropagation();
                                navigate(`/actor/${encodeURIComponent(director)}`);
                              }}
                              title={`View ${director}'s biography & filmography`}
                            >
                              {director}
                            </span>
                          </div>
                        )}
                        {starsList.length > 0 && (
                          <div className="watchlist-credit-group">
                            <span className="credit-label" style={{ color: isDark ? '#ffffff' : '#181818' }}>
                              Stars
                            </span>
                            <span className="credit-name" style={{ color: isDark ? '#94a3b8' : '#475569' }}>
                              {starsList.slice(0, 4).map((star, sIdx) => (
                                <span
                                  key={sIdx}
                                  className="highlight-link"
                                  style={{ color: isDark ? '#38bdf8' : '#0284c7', cursor: 'pointer' }}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    navigate(`/actor/${star.id || encodeURIComponent(star.name)}`);
                                  }}
                                  title={`View ${star.name}'s biography & filmography`}
                                >
                                  {star.name}
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
            /* Empty Watchlist State for Signed-In User */
            <div
              className="watchlist-empty-state"
              style={{
                backgroundColor: isDark ? '#1a1a1a' : '#ffffff',
                borderColor: isDark ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.15)'
              }}
            >
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
