import React, { createContext, useContext, useState, useEffect } from 'react';
import { useAuth, useUser } from '@clerk/clerk-react';
import { fetchDbWatchlist, addMovieToDbWatchlist, removeMovieFromDbWatchlist } from '../api/authApi';

const WatchlistContext = createContext();

export function WatchlistProvider({ children }) {
  const { isSignedIn, user, isLoaded: isAuthLoaded } = useUser();
  const { getToken } = useAuth();

  const [watchlist, setWatchlist] = useState([]);
  const [toast, setToast] = useState({ visible: false, message: '' });

  // When auth state changes or user logs in/out:
  useEffect(() => {
    let mounted = true;

    if (!isAuthLoaded) return;

    // If user is not signed in, clear the watchlist completely
    if (!isSignedIn || !user) {
      setWatchlist([]);
      try {
        localStorage.removeItem('watchlist'); // Clean up any stale legacy anonymous items
      } catch (e) {}
      return;
    }

    // If signed in, load user-specific cached list first for instant display
    const userStorageKey = `watchlist_${user.id}`;
    let initialList = [];
    try {
      const saved = localStorage.getItem(userStorageKey);
      if (saved) initialList = JSON.parse(saved);
    } catch (e) {}

    if (initialList.length > 0) {
      setWatchlist(initialList);
    }

    // Fetch user's PostgreSQL database watchlist
    async function loadDbWatchlist() {
      try {
        const dbItems = await fetchDbWatchlist(getToken);
        if (mounted && Array.isArray(dbItems)) {
          const formattedDb = dbItems.map((item) => ({
            title: item.movieTitle,
            poster: item.posterPath,
            rating: item.voteAverage !== null ? item.voteAverage : 'N/A',
            release_date: item.releaseYear || '',
            genres: item.genres?.join(', ') || '',
          }));

          // Merge unique by title
          const titles = new Set();
          const merged = [];

          for (const m of [...formattedDb, ...initialList]) {
            if (m && m.title) {
              const key = m.title.toLowerCase();
              if (!titles.has(key)) {
                titles.add(key);
                merged.push(m);
              }
            }
          }

          setWatchlist(merged);
          try {
            localStorage.setItem(userStorageKey, JSON.stringify(merged));
          } catch (e) {}
        }
      } catch (err) {
        console.warn('Could not fetch DB watchlist on startup:', err.message);
      }
    }

    loadDbWatchlist();

    return () => {
      mounted = false;
    };
  }, [isSignedIn, user, isAuthLoaded, getToken]);

  // Sync to user-specific localStorage when watchlist changes and user is signed in
  useEffect(() => {
    if (isSignedIn && user) {
      try {
        localStorage.setItem(`watchlist_${user.id}`, JSON.stringify(watchlist));
      } catch (e) {}
    }
  }, [watchlist, isSignedIn, user]);

  const showToast = (message) => {
    setToast({ visible: true, message });
    setTimeout(() => {
      setToast({ visible: false, message: '' });
    }, 2500);
  };

  const addToWatchlist = (movie) => {
    if (!movie || !movie.title) return;

    if (!isSignedIn) {
      showToast('Please sign in to add movies to your Watchlist!');
      return;
    }

    const movieObj = {
      title: movie.title,
      poster: movie.poster || movie.poster_path || '',
      rating: movie.rating !== undefined ? movie.rating : (movie.vote_average || 'N/A'),
      release_date: movie.release_date || movie.date || '',
      runtime: movie.runtime || '',
      status: movie.status || '',
      vote_count: movie.vote_count || movie.count || '',
      overview: movie.overview || '',
      genres: movie.genres || '',
      director: movie.director || movie.director_name || '',
      casts: movie.casts || movie.stars || movie.cast || [],
    };

    setWatchlist((prev) => {
      if (prev.some((m) => m.title && m.title.toLowerCase() === movie.title.toLowerCase())) {
        showToast(`"${movie.title}" is already in your Watchlist!`);
        return prev;
      }
      showToast(`Saved to Watchlist!`);
      return [movieObj, ...prev];
    });

    // Persist to PostgreSQL database
    const payload = {
      movieId: movie.title,
      movieTitle: movie.title,
      posterPath: movieObj.poster,
      releaseYear: movieObj.release_date,
      voteAverage: movieObj.rating !== 'N/A' && !isNaN(Number(movieObj.rating)) ? Number(movieObj.rating) : null,
      genres: typeof movieObj.genres === 'string'
        ? movieObj.genres.split(',').map((s) => s.trim()).filter(Boolean)
        : (Array.isArray(movieObj.genres) ? movieObj.genres : []),
    };

    addMovieToDbWatchlist(getToken, payload)
      .then((res) => {
        if (res) {
          console.log(`✅ [NeonDB] Added "${movie.title}" to PostgreSQL watchlist table:`, res);
        }
      })
      .catch((err) => {
        console.warn(`⚠️ [NeonDB] Could not sync watchlist item to backend:`, err.message);
      });
  };

  const removeFromWatchlist = (title) => {
    if (!title) return;

    if (!isSignedIn) {
      showToast('Please sign in to manage your Watchlist!');
      return;
    }

    setWatchlist((prev) => prev.filter((m) => m.title && m.title.toLowerCase() !== title.toLowerCase()));
    showToast(`Removed from Watchlist!`);

    // Remove from PostgreSQL database
    removeMovieFromDbWatchlist(getToken, title)
      .then((res) => {
        if (res) {
          console.log(`✅ [NeonDB] Removed "${title}" from PostgreSQL watchlist table:`, res);
        }
      })
      .catch((err) => {
        console.warn(`⚠️ [NeonDB] Could not remove watchlist item from backend:`, err.message);
      });
  };

  const isInWatchlist = (title) => {
    if (!title || !isSignedIn) return false;
    return watchlist.some((m) => m.title && m.title.toLowerCase() === title.toLowerCase());
  };

  return (
    <WatchlistContext.Provider
      value={{
        watchlist,
        addToWatchlist,
        removeFromWatchlist,
        isInWatchlist,
        toast,
      }}
    >
      {children}
      {toast.visible && (
        <div
          style={{
            position: 'fixed',
            bottom: '30px',
            left: '50%',
            transform: 'translateX(-50%)',
            backgroundColor: '#28a745',
            color: '#ffffff',
            padding: '12px 28px',
            borderRadius: '30px',
            fontWeight: 'bold',
            fontSize: '16px',
            zIndex: 99999,
            boxShadow: '0 4px 20px rgba(0,0,0,0.6)',
            transition: 'opacity 0.3s ease',
          }}
        >
          {toast.message}
        </div>
      )}
    </WatchlistContext.Provider>
  );
}

export function useWatchlist() {
  const context = useContext(WatchlistContext);
  if (!context) {
    throw new Error('useWatchlist must be used within a WatchlistProvider');
  }
  return context;
}
