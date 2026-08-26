import React from 'react';
import { useNavigate, Link } from 'react-router-dom';
import Navbar from '../components/Navbar';
import { useWatchlist } from '../context/WatchlistContext';

export default function WatchlistPage() {
  const { watchlist, removeFromWatchlist } = useWatchlist();
  const navigate = useNavigate();

  return (
    <div className="watchlist-page-container">
      <Navbar onSearchMovie={(title) => navigate(`/movie/${encodeURIComponent(title)}`)} />

      <main className="watchlist-main-content">
        <div className="watchlist-header-block">
          <h1 className="watchlist-title">Your Watchlist</h1>
          <p className="watchlist-subtitle">
            Track all the movies you want to watch. Easily jump right back into any title or view recommendations.
          </p>
        </div>

        {watchlist.length > 0 ? (
          <div className="watchlist-grid">
            {watchlist.map((movie, idx) => (
              <div key={idx} className="watchlist-movie-card">
                <button
                  className="watchlist-delete-btn"
                  onClick={() => removeFromWatchlist(movie.title)}
                  title="Remove from Watchlist"
                >
                  ✕
                </button>

                <div
                  className="watchlist-card-body"
                  onClick={() => navigate(`/movie/${encodeURIComponent(movie.title)}`)}
                >
                  <img
                    src={movie.poster || 'https://via.placeholder.com/240x360?text=No+Poster'}
                    alt={movie.title}
                    className="watchlist-poster-img"
                    onError={(e) => { e.target.src = 'https://via.placeholder.com/240x360?text=No+Poster'; }}
                  />
                  <div className="watchlist-info">
                    <h4 className="watchlist-movie-title" title={movie.title}>{movie.title}</h4>
                    {movie.rating && movie.rating !== 'N/A' && (
                      <span className="card-rating">★ {movie.rating}</span>
                    )}
                    {movie.genres && (
                      <p className="watchlist-genres-text">{movie.genres}</p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-watchlist-box">
            <img src="/images/add_bookmark.svg" width="64" height="64" alt="Empty" className="empty-icon" />
            <h3>Your Watchlist is empty</h3>
            <p>Save movies here to easily find and watch them later.</p>
            <div className="empty-actions-row">
              <Link to="/" className="btn-primary-red">Browse Popular Movies</Link>
              <Link to="/genres" className="btn-secondary-dark">Explore Genres</Link>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
