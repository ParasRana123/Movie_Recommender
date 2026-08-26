import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import localSuggestions from '../data/suggestions.json';
import { fetchSuggestions } from '../api/movieApi';

export default function Navbar({ onSearchMovie, initialQuery = '' }) {
  const [query, setQuery] = useState(initialQuery);
  const [films, setFilms] = useState(localSuggestions || []);
  const [filteredSuggestions, setFilteredSuggestions] = useState([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);

  const navigate = useNavigate();
  const searchContainerRef = useRef(null);

  // Sync initialQuery prop if it changes
  useEffect(() => {
    if (initialQuery) {
      setQuery(initialQuery);
    }
  }, [initialQuery]);

  // Also fetch latest suggestions from server in background
  useEffect(() => {
    let mounted = true;
    fetchSuggestions()
      .then(data => {
        if (mounted && Array.isArray(data) && data.length > 0) {
          setFilms(data);
        }
      })
      .catch(() => {});
    return () => { mounted = false; };
  }, []);

  // Compute live search suggestions on every keystroke
  useEffect(() => {
    const trimmed = query.trim().toLowerCase();
    if (trimmed.length < 2) {
      setFilteredSuggestions([]);
      setShowDropdown(false);
      return;
    }

    // Match movies starting with query first, then movies containing query
    const startsWithMatches = [];
    const containsMatches = [];

    for (let i = 0; i < films.length; i++) {
      const title = films[i];
      const lower = title.toLowerCase();
      if (lower.startsWith(trimmed)) {
        startsWithMatches.push(title);
        if (startsWithMatches.length >= 10) break;
      } else if (lower.includes(trimmed)) {
        containsMatches.push(title);
      }
    }

    const combined = [...startsWithMatches, ...containsMatches].slice(0, 10);
    setFilteredSuggestions(combined);
    setShowDropdown(combined.length > 0);
    setSelectedIndex(-1);
  }, [query, films]);

  // Click outside to dismiss dropdown
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
    if (e) e.preventDefault();
    if (query.trim()) {
      handleSelect(query.trim());
    }
  };

  // Highlight matched query with <mark>
  const renderHighlighted = (text, highlight) => {
    if (!highlight.trim()) return text;
    const parts = text.split(new RegExp(`(${highlight})`, 'gi'));
    return (
      <span>
        {parts.map((part, i) =>
          part.toLowerCase() === highlight.toLowerCase() ? (
            <mark key={i}>{part}</mark>
          ) : (
            part
          )
        )}
      </span>
    );
  };

  return (
    <div className="ml-container" style={{ display: 'block' }}>
      {/* GitHub Octocat Ribbon */}
      <a href="https://github.com/ParasRana123/Movie-Recommender-with-Sentiment-Analysis" className="github-corner" title="View source on GitHub">
        <svg data-toggle="tooltip" data-placement="left" width="80" height="80" viewBox="0 0 250 250" style={{ fill: '#e50914', color: '#fff', position: 'fixed', zIndex: 100, top: 0, border: 0, right: 0 }} aria-hidden="true">
          <path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path>
          <path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style={{ transformOrigin: '130px 106px' }} className="octo-arm"></path>
          <path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" className="octo-body"></path>
        </svg>
      </a>

      {/* Exact Original Navigation Bar */}
      <nav
        className="form-group shadow-textarea"
        style={{
          textAlign: 'center',
          color: 'white',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: '#333333',
          height: '12vh',
          margin: 0,
          position: 'relative',
          zIndex: 1000
        }}
      >
        <div className="search-container" style={{ position: 'relative', width: '60%' }} ref={searchContainerRef}>
          <input
            type="text"
            name="movie"
            className="movie form-control"
            id="autoComplete"
            autoComplete="off"
            placeholder="Enter the Movie Name"
            style={{
              backgroundColor: '#ffffff',
              borderColor: '#ffffff',
              width: '100%',
              color: '#181818',
              borderRadius: '5px'
            }}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => { if (filteredSuggestions.length > 0) setShowDropdown(true); }}
            required="required"
          />

          {/* Autocomplete Dropdown List */}
          {showDropdown && filteredSuggestions.length > 0 && (
            <ul id="movie_list" style={{ display: 'block' }}>
              {filteredSuggestions.map((title, idx) => (
                <li
                  key={idx}
                  className={idx === selectedIndex ? 'selected' : ''}
                  onClick={() => handleSelect(title)}
                  onMouseEnter={() => setSelectedIndex(idx)}
                >
                  {renderHighlighted(title, query)}
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="form-group" style={{ textAlign: 'center', margin: 0 }}>
          <button
            className="btn btn-primary btn-block movie-button"
            style={{
              backgroundColor: '#e50914',
              textAlign: 'center',
              borderColor: '#e50914',
              width: '120px',
              position: 'relative',
              left: '-155px',
              top: '30px'
            }}
            disabled={!query.trim()}
            onClick={handleSubmit}
          >
            Enter
          </button>
          <br /><br />
        </div>

        <div style={{ color: 'white', position: 'relative', left: '-125px', top: '-5px' }}> | </div>

        <div
          className="watchlist"
          style={{ cursor: 'pointer' }}
          onClick={() => navigate('/watchlist')}
        >
          <img src="/images/add_bookmark.svg" width="30px" height="auto" alt="WatchList" />
          <p>WatchList</p>
        </div>

        <div
          className="genres"
          style={{ color: 'white', position: 'relative', left: '-120px', top: '-5px', cursor: 'pointer' }}
          onClick={() => navigate('/genres')}
        >
          Genre-Wise
        </div>
      </nav>
    </div>
  );
}
