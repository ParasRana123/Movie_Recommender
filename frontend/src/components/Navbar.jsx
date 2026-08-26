import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { fetchSuggestions } from '../api/movieApi';
import { useWatchlist } from '../context/WatchlistContext';

export default function Navbar({ onSearchMovie, initialQuery = '' }) {
  const [query, setQuery] = useState(initialQuery);
  const [suggestions, setSuggestions] = useState([]);
  const [filteredSuggestions, setFilteredSuggestions] = useState([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);

  const { watchlist } = useWatchlist();
  const navigate = useNavigate();
  const searchContainerRef = useRef(null);

  // Load all movie suggestions once on mount
  useEffect(() => {
    let mounted = true;
    fetchSuggestions().then(data => {
      if (mounted && Array.isArray(data)) {
        setSuggestions(data);
      }
    });
    return () => { mounted = false; };
  }, []);

  // Update filtered suggestions when query changes
  useEffect(() => {
    const trimmed = query.trim().toLowerCase();
    if (trimmed.length < 2) {
      setFilteredSuggestions([]);
      setShowDropdown(false);
      return;
    }

    const matches = suggestions
      .filter(title => {
        const lower = title.toLowerCase();
        return lower.startsWith(trimmed) || lower.includes(trimmed);
      })
      .slice(0, 10);

    setFilteredSuggestions(matches);
    setShowDropdown(true);
    setSelectedIndex(-1);
  }, [query, suggestions]);

  // Click outside listener to dismiss dropdown
  useEffect(() => {
    function handleClickOutside(event) {
      if (searchContainerRef.current && !searchContainerRef.current.contains(event.target)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (selectedTitle) => {
    setQuery(selectedTitle);
    setShowDropdown(false);
    if (onSearchMovie) {
      onSearchMovie(selectedTitle);
    } else {
      navigate(`/movie/${encodeURIComponent(selectedTitle)}`);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => (prev < filteredSuggestions.length - 1 ? prev + 1 : prev));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => (prev > 0 ? prev - 1 : -1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (selectedIndex >= 0 && filteredSuggestions[selectedIndex]) {
        handleSelect(filteredSuggestions[selectedIndex]);
      } else if (query.trim()) {
        handleSelect(query.trim());
      }
    } else if (e.key === 'Escape') {
      setShowDropdown(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      handleSelect(query.trim());
    }
  };

  // Function to highlight matched substring
  const renderHighlighted = (text, highlight) => {
    if (!highlight.trim()) return text;
    const parts = text.split(new RegExp(`(${highlight})`, 'gi'));
    return (
      <span>
        {parts.map((part, i) =>
          part.toLowerCase() === highlight.toLowerCase() ? (
            <span key={i} className="autocomplete-highlight">{part}</span>
          ) : (
            part
          )
        )}
      </span>
    );
  };

  return (
    <header className="app-navbar">
      {/* GitHub Octocat Corner Link */}
      <a href="/" className="github-corner" aria-label="Home" title="Movie Recommender Home">
        <svg width="60" height="60" viewBox="0 0 250 250" style={{ fill: '#e50914', color: '#fff', position: 'fixed', zIndex: 100, top: 0, border: 0, right: 0 }} aria-hidden="true">
          <path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path>
          <path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style={{ transformOrigin: '130px 106px' }} className="octo-arm"></path>
          <path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" className="octo-body"></path>
        </svg>
      </a>

      <div className="navbar-container">
        {/* Brand Logo */}
        <Link to="/" className="navbar-brand">
          <span className="brand-red">MOVIE</span>FLIX
        </Link>

        {/* Search Bar Form */}
        <form onSubmit={handleSubmit} className="navbar-search-form" ref={searchContainerRef}>
          <div className="search-input-wrapper">
            <input
              type="text"
              id="autoComplete"
              className="movie-search-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => { if (filteredSuggestions.length > 0) setShowDropdown(true); }}
              placeholder="Enter the Movie Name (e.g. Avengers)"
              autoComplete="off"
              required
            />
            {query && (
              <button
                type="button"
                className="search-clear-btn"
                onClick={() => { setQuery(''); setFilteredSuggestions([]); setShowDropdown(false); }}
              >
                ✕
              </button>
            )}

            {/* Autocomplete Dropdown */}
            {showDropdown && filteredSuggestions.length > 0 && (
              <ul id="movie_list" className="autocomplete-dropdown">
                {filteredSuggestions.map((title, idx) => (
                  <li
                    key={idx}
                    className={`autocomplete-item ${idx === selectedIndex ? 'selected' : ''}`}
                    onClick={() => handleSelect(title)}
                    onMouseEnter={() => setSelectedIndex(idx)}
                  >
                    {renderHighlighted(title, query)}
                  </li>
                ))}
              </ul>
            )}
          </div>

          <button
            type="submit"
            className="movie-search-button"
            disabled={!query.trim()}
          >
            Enter
          </button>
        </form>

        {/* Navigation Links */}
        <div className="navbar-links">
          <div className="navbar-divider">|</div>

          <Link to="/watchlist" className="nav-watchlist-link" title="View your saved movies">
            <img src="/images/add_bookmark.svg" width="26" height="26" alt="Watchlist" />
            <span>WatchList</span>
            {watchlist.length > 0 && (
              <span className="watchlist-counter">{watchlist.length}</span>
            )}
          </Link>

          <Link to="/genres" className="nav-genres-link" title="Explore movies by genre">
            Genre-Wise
          </Link>
        </div>
      </div>
    </header>
  );
}
