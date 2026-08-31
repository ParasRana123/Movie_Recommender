import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { SignedIn, SignedOut, SignInButton, SignUpButton, UserButton } from '@clerk/clerk-react';
import localSuggestions from '../data/suggestions.json';
import { fetchSuggestions } from '../api/movieApi';
import { useTheme } from '../context/ThemeContext';

export default function Navbar({ onSearchMovie, onHomeClick, initialQuery = '' }) {
  const [query, setQuery] = useState(initialQuery);
  const [films, setFilms] = useState(localSuggestions || []);
  const [filteredSuggestions, setFilteredSuggestions] = useState([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [showMobileSearch, setShowMobileSearch] = useState(false);

  const { theme, toggleTheme, isDark } = useTheme();
  const navigate = useNavigate();
  const searchContainerRef = useRef(null);
  const mobileSearchRef = useRef(null);
  const mobileInputRef = useRef(null);
  const isSelectingRef = useRef(false);

  useEffect(() => {
    if (initialQuery) {
      setQuery(initialQuery);
    }
  }, [initialQuery]);

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

  // Autofocus mobile input when opened
  useEffect(() => {
    if (showMobileSearch && mobileInputRef.current) {
      setTimeout(() => {
        mobileInputRef.current?.focus();
      }, 100);
    }
  }, [showMobileSearch]);

  // Compute live search suggestions on typing
  const handleInputChange = (e) => {
    const val = e.target.value;
    setQuery(val);
    isSelectingRef.current = false;

    const trimmed = val.trim().toLowerCase();
    if (trimmed.length < 2) {
      setFilteredSuggestions([]);
      setShowDropdown(false);
      return;
    }

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
  };

  useEffect(() => {
    function handleClickOutside(event) {
      if (searchContainerRef.current && !searchContainerRef.current.contains(event.target) &&
          mobileSearchRef.current && !mobileSearchRef.current.contains(event.target)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (selectedTitle) => {
    isSelectingRef.current = true;
    setQuery(selectedTitle);
    setShowDropdown(false);
    setFilteredSuggestions([]);
    setShowMobileSearch(false);

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
      setShowMobileSearch(false);
    }
  };

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    if (query.trim()) {
      handleSelect(query.trim());
    }
  };

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
      {/* GitHub Corner (Desktop only) */}
      <a href="https://github.com/ParasRana123/Movie-Recommender-with-Sentiment-Analysis" className="github-corner desktop-only" title="View source on GitHub">
        <svg data-toggle="tooltip" data-placement="left" width="80" height="80" viewBox="0 0 250 250" style={{ fill: '#e50914', color: '#fff', position: 'fixed', zIndex: 100, top: 0, border: 0, right: 0 }} aria-hidden="true">
          <path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path>
          <path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style={{ transformOrigin: '130px 106px' }} className="octo-arm"></path>
          <path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" className="octo-body"></path>
        </svg>
      </a>

      {/* Top Navbar */}
      <nav
        className="form-group shadow-textarea app-navbar"
        style={{
          textAlign: 'center',
          color: 'white',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: isDark ? '#1a1a1a' : '#333333',
          borderBottom: isDark ? '1px solid #2e2e2e' : 'none',
          height: '12vh',
          minHeight: '65px',
          margin: 0,
          padding: '0 15px',
          gap: '14px',
          position: 'relative',
          zIndex: 1000,
          transition: 'background-color 0.3s ease'
        }}
      >
        {/* Desktop Search Bar Input Container */}
        <div className="search-container desktop-search-group" style={{ position: 'relative', width: '420px', maxWidth: '36vw', margin: 0 }} ref={searchContainerRef}>
          <input
            type="text"
            name="movie"
            className="movie form-control"
            id="autoComplete"
            autoComplete="off"
            placeholder="Enter the Movie Name"
            style={{
              backgroundColor: isDark ? '#262626' : '#ffffff',
              borderColor: isDark ? '#404040' : '#ffffff',
              width: '100%',
              color: isDark ? '#ffffff' : '#181818',
              borderRadius: '5px',
              height: '38px',
              padding: '6px 40px 6px 12px',
              backgroundImage: 'url("data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 24 24\' width=\'20\' height=\'20\' fill=\'%23e50914\'%3E%3Cpath d=\'M21.71 20.29l-5.4-5.39A8 8 0 1 0 4 11a8 8 0 0 0 12.31 6.31l5.4 5.4a1 1 0 0 0 1.41-1.41zM6 11a6 6 0 1 1 6 6 6 6 0 0 1-6-6z\'/%3E%3C/svg%3E")',
              backgroundRepeat: 'no-repeat',
              backgroundPosition: '96% center',
              backgroundSize: '20px',
              transition: 'all 0.3s ease'
            }}
            value={query}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onFocus={() => {
              if (!isSelectingRef.current && filteredSuggestions.length > 0) {
                setShowDropdown(true);
              }
            }}
            required="required"
          />

          {/* Desktop Autocomplete Dropdown List */}
          {showDropdown && filteredSuggestions.length > 0 && (
            <ul
              id="movie_list"
              style={{
                display: 'block',
                backgroundColor: isDark ? '#1f1f1f' : '#ffffff',
                borderColor: isDark ? '#333333' : 'rgba(0,0,0,0.1)'
              }}
            >
              {filteredSuggestions.map((title, idx) => (
                <li
                  key={idx}
                  className={idx === selectedIndex ? 'selected' : ''}
                  style={{
                    color: isDark ? '#e0e0e0' : '#1a1a1a',
                    borderBottomColor: isDark ? '#2c2c2c' : '#f0f0f0'
                  }}
                  onClick={() => handleSelect(title)}
                  onMouseEnter={() => setSelectedIndex(idx)}
                >
                  {renderHighlighted(title, query)}
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Desktop Enter Button */}
        <div className="desktop-search-group" style={{ margin: 0 }}>
          <button
            className="btn btn-primary movie-button"
            style={{
              backgroundColor: '#e50914',
              borderColor: '#e50914',
              width: '95px',
              height: '38px',
              borderRadius: '5px',
              fontWeight: 'bold',
              margin: 0,
              cursor: query.trim() ? 'pointer' : 'default',
              transition: 'transform 0.2s ease, background-color 0.2s ease'
            }}
            disabled={!query.trim()}
            onClick={handleSubmit}
          >
            Enter
          </button>
        </div>

        <div className="desktop-search-group nav-separator" style={{ color: 'white', fontSize: '18px', margin: '0 2px' }}> | </div>

        {/* Home Navigation */}
        <div
          className="home-nav"
          style={{
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            backgroundColor: 'transparent',
            margin: 0,
            padding: '4px 6px'
          }}
          onClick={() => {
            if (onHomeClick) {
              onHomeClick();
            }
            navigate('/');
          }}
          title="Home"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
            <polyline points="9 22 9 12 15 12 15 22"></polyline>
          </svg>
          <p style={{ margin: 0, color: 'white', fontSize: '15px', fontWeight: 500 }}>Home</p>
        </div>

        <div className="nav-separator" style={{ color: 'white', fontSize: '18px', margin: '0 2px' }}> | </div>

        {/* Watchlist Navigation */}
        <div
          className="watchlist"
          style={{
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            backgroundColor: 'transparent',
            margin: 0,
            padding: '4px 6px'
          }}
          onClick={() => navigate('/watchlist')}
          title="WatchList"
        >
          <img src="/images/add_bookmark.svg" width="22px" height="auto" alt="WatchList" style={{ filter: 'invert(1)' }} />
          <p style={{ margin: 0, color: 'white', fontSize: '15px' }}>WatchList</p>
        </div>

        <div className="nav-separator" style={{ color: 'white', fontSize: '18px', margin: '0 2px' }}> | </div>

        {/* Genre-Wise Navigation */}
        <div
          className="genres"
          style={{
            color: 'white',
            cursor: 'pointer',
            fontSize: '15px',
            fontWeight: 500,
            margin: 0,
            padding: '4px 6px',
            position: 'static'
          }}
          onClick={() => navigate('/genres')}
          title="Genre-Wise"
        >
          Genre-Wise
        </div>

        <div className="nav-separator" style={{ color: 'white', fontSize: '18px', margin: '0 2px' }}> | </div>

        {/* Dark / Light Mode Toggle Button (Icon only) */}
        <div
          className="theme-toggle-btn"
          style={{
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'transparent',
            padding: '6px',
            borderRadius: '50%',
            color: 'white',
            userSelect: 'none',
            transition: 'transform 0.2s ease, background-color 0.2s ease',
            border: 'none',
            margin: 0
          }}
          onClick={toggleTheme}
          title={`Switch to ${isDark ? 'Light' : 'Dark'} Mode`}
        >
          {isDark ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffd700" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="5"></circle>
              <line x1="12" y1="1" x2="12" y2="3"></line>
              <line x1="12" y1="21" x2="12" y2="23"></line>
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
              <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
              <line x1="1" y1="12" x2="3" y2="12"></line>
              <line x1="21" y1="12" x2="23" y2="12"></line>
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
              <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
            </svg>
          )}
        </div>

        <div className="nav-separator" style={{ color: 'white', fontSize: '18px', margin: '0 2px' }}> | </div>

        {/* Clerk Authentication Controls */}
        <div className="auth-nav-container" style={{ display: 'flex', alignItems: 'center', margin: 0 }}>
          <SignedOut>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <SignInButton mode="modal">
                <button
                  type="button"
                  style={{
                    backgroundColor: 'transparent',
                    border: isDark ? '1px solid #4a4a4a' : '1px solid rgba(255,255,255,0.6)',
                    color: '#ffffff',
                    padding: '5px 12px',
                    borderRadius: '5px',
                    fontSize: '14px',
                    fontWeight: 500,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    whiteSpace: 'nowrap'
                  }}
                  className="auth-btn-signin"
                >
                  Sign In
                </button>
              </SignInButton>
              <SignUpButton mode="modal">
                <button
                  type="button"
                  style={{
                    backgroundColor: '#e50914',
                    border: '1px solid #e50914',
                    color: '#ffffff',
                    padding: '5px 12px',
                    borderRadius: '5px',
                    fontSize: '14px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    boxShadow: '0 2px 6px rgba(229, 9, 20, 0.35)',
                    transition: 'all 0.2s ease',
                    whiteSpace: 'nowrap'
                  }}
                  className="auth-btn-signup"
                >
                  Sign Up
                </button>
              </SignUpButton>
            </div>
          </SignedOut>

          <SignedIn>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <UserButton
                afterSignOutUrl="/"
                appearance={{
                  elements: {
                    avatarBox: {
                      width: '32px',
                      height: '32px',
                      border: '2px solid #e50914'
                    }
                  }
                }}
              />
            </div>
          </SignedIn>
        </div>

        {/* Mobile Search Icon Button (Clean icon with no red circular background) */}
        <div
          className="mobile-search-btn"
          onClick={() => setShowMobileSearch(true)}
          style={{
            cursor: 'pointer',
            backgroundColor: 'transparent',
            color: '#ffffff',
            border: 'none',
            borderRadius: '0',
            width: 'auto',
            height: 'auto',
            display: 'none', // Managed by CSS media query
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: 'none',
            padding: '4px 6px',
            margin: 0
          }}
          title="Search Movies"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
        </div>
      </nav>

      {/* Mobile Search Overlay Drawer / Modal */}
      {showMobileSearch && (
        <div
          className="mobile-search-overlay"
          ref={mobileSearchRef}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            zIndex: 10001,
            backgroundColor: isDark ? '#1a1a1a' : '#ffffff',
            boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
            padding: '16px 14px',
            borderBottom: `2px solid #e50914`
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ position: 'relative', flex: 1 }}>
              <input
                ref={mobileInputRef}
                type="text"
                className="form-control"
                placeholder="Search movies (e.g. Avengers)..."
                value={query}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                style={{
                  backgroundColor: isDark ? '#2a2a2a' : '#f0f0f0',
                  color: isDark ? '#ffffff' : '#181818',
                  borderColor: isDark ? '#444' : '#ccc',
                  borderRadius: '25px',
                  padding: '10px 16px',
                  fontSize: '16px',
                  width: '100%'
                }}
              />
            </div>

            <button
              className="btn btn-danger"
              onClick={handleSubmit}
              disabled={!query.trim()}
              style={{
                backgroundColor: '#e50914',
                borderColor: '#e50914',
                borderRadius: '20px',
                padding: '8px 16px',
                fontWeight: 'bold',
                fontSize: '14px'
              }}
            >
              Search
            </button>

            <button
              onClick={() => {
                setShowMobileSearch(false);
                setShowDropdown(false);
              }}
              style={{
                background: 'none',
                border: 'none',
                color: isDark ? '#ffffff' : '#333333',
                fontSize: '24px',
                cursor: 'pointer',
                padding: '0 6px',
                lineHeight: 1
              }}
              title="Close Search"
            >
              ✕
            </button>
          </div>

          {/* Mobile Search Real-Time Suggestions */}
          {showDropdown && filteredSuggestions.length > 0 && (
            <ul
              style={{
                maxHeight: '280px',
                overflowY: 'auto',
                backgroundColor: isDark ? '#222222' : '#ffffff',
                margin: '12px 0 0 0',
                padding: 0,
                listStyle: 'none',
                borderRadius: '10px',
                border: `1px solid ${isDark ? '#333' : '#e0e0e0'}`,
                boxShadow: '0 6px 20px rgba(0,0,0,0.3)'
              }}
            >
              {filteredSuggestions.map((title, idx) => (
                <li
                  key={idx}
                  onClick={() => handleSelect(title)}
                  style={{
                    padding: '12px 16px',
                    borderBottom: `1px solid ${isDark ? '#2c2c2c' : '#f0f0f0'}`,
                    color: isDark ? '#f0f0f0' : '#1a1a1a',
                    fontSize: '15px',
                    cursor: 'pointer'
                  }}
                >
                  {renderHighlighted(title, query)}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
