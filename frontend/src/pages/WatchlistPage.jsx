import React from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import { useWatchlist } from '../context/WatchlistContext';

export default function WatchlistPage() {
  const { watchlist, removeFromWatchlist } = useWatchlist();
  const navigate = useNavigate();

  return (
    <div style={{ backgroundColor: '#121212', minHeight: '100vh', color: '#ffffff' }}>
      <Navbar onSearchMovie={(title) => navigate(`/movie/${encodeURIComponent(title)}`)} />

      <div className="your_watchlist">
        <h1>Your Watchlist</h1>
        <p>
          Here are all the movies you've saved. You can click on any movie to view recommendations or remove it from your list.
        </p>

        <div id="watchlist-container">
          {watchlist && watchlist.length > 0 ? (
            watchlist.map((movie, idx) => (
              <div key={idx} className="watchlist-card">
                <button
                  className="delete-btn"
                  onClick={() => removeFromWatchlist(movie.title)}
                  title="Remove from Watchlist"
                >
                  ✕
                </button>

                <div
                  className="movie-poster-wrap"
                  onClick={() => navigate(`/movie/${encodeURIComponent(movie.title)}`)}
                >
                  <img
                    src={movie.poster || 'https://via.placeholder.com/140x210?text=No+Poster'}
                    alt={movie.title}
                    className="movie-poster"
                  />
                </div>

                <div className="movie-info-wrap">
                  <h3
                    className="movie-title"
                    onClick={() => navigate(`/movie/${encodeURIComponent(movie.title)}`)}
                  >
                    {movie.title}
                  </h3>

                  <div className="movie-meta-row">
                    {movie.rating && movie.rating !== 'N/A' && (
                      <div className="movie-meta-item movie-rating-badge">
                        ★ {movie.rating} / 10
                      </div>
                    )}
                    {movie.runtime && movie.runtime !== 'N/A' && (
                      <div className="movie-meta-item">
                        ⏱ {movie.runtime}
                      </div>
                    )}
                    {movie.release_date && movie.release_date !== 'Unknown Date' && (
                      <div className="movie-meta-item">
                        📅 {movie.release_date}
                      </div>
                    )}
                    {movie.status && (
                      <div className="movie-meta-item">
                        🏷 {movie.status}
                      </div>
                    )}
                  </div>

                  {movie.overview && (
                    <p className="movie-overview-text">{movie.overview}</p>
                  )}
                </div>
              </div>
            ))
          ) : (
            <div style={{ textAlign: 'center', padding: '60px 20px' }}>
              <img
                src="/images/add_bookmark.svg"
                width="60"
                height="60"
                style={{ filter: 'invert(1)', opacity: 0.3, marginBottom: '20px' }}
                alt="Empty"
              />
              <h3 style={{ color: '#ffffff', marginBottom: '10px' }}>Your Watchlist is empty</h3>
              <p style={{ color: '#888888', maxWidth: '500px', margin: '0 auto 25px auto' }}>
                You haven't saved any movies yet. Search for movies and click "Add to Watchlist" to save them here!
              </p>
              <button
                className="btn btn-primary movie-button"
                style={{ backgroundColor: '#e50914', borderColor: '#e50914', borderRadius: '5px', padding: '8px 24px', cursor: 'pointer' }}
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
