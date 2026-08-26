import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useWatchlist } from '../context/WatchlistContext';

export default function MovieCard({ title, poster, rating, onClick }) {
  const navigate = useNavigate();
  const { isInWatchlist, addToWatchlist, removeFromWatchlist } = useWatchlist();

  const isSaved = isInWatchlist(title);

  const handleClick = () => {
    if (onClick) {
      onClick(title);
    } else {
      navigate(`/movie/${encodeURIComponent(title)}`);
    }
  };

  const handleBookmarkToggle = (e) => {
    e.stopPropagation();
    if (isSaved) {
      removeFromWatchlist(title);
    } else {
      addToWatchlist({ title, poster, rating });
    }
  };

  const posterSrc = poster || 'https://via.placeholder.com/240x360?text=No+Poster';

  return (
    <div className="movie-card" onClick={handleClick}>
      <button
        className={`card-bookmark-btn ${isSaved ? 'active' : ''}`}
        onClick={handleBookmarkToggle}
        title={isSaved ? 'Remove from Watchlist' : 'Add to Watchlist'}
        aria-label="Toggle Watchlist"
      >
        <img
          src={isSaved ? '/images/bookmark_tick.svg' : '/images/add_bookmark.svg'}
          alt="Bookmark"
          width="20"
          height="20"
        />
      </button>

      <div className="poster-container">
        <img
          src={posterSrc}
          alt={`${title} poster`}
          loading="lazy"
          onError={(e) => { e.target.src = 'https://via.placeholder.com/240x360?text=No+Poster'; }}
        />
      </div>

      <div className="movie-card-info">
        {rating && rating !== 'N/A' && rating !== '0' && rating !== '0.0' && (
          <p className="card-rating">★ {rating}</p>
        )}
        <h3 className="card-title" title={title}>{title}</h3>
      </div>
    </div>
  );
}
