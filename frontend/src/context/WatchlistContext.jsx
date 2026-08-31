import React, { createContext, useContext, useState, useEffect } from 'react';
import { useAuth, useUser } from '@clerk/clerk-react';
import { fetchDbWatchlist, addMovieToDbWatchlist, removeMovieFromDbWatchlist } from '../api/authApi';

const WatchlistContext = createContext();

export function WatchlistProvider({ children }) {
  const { isSignedIn } = useUser();
  const { getToken } = useAuth();

  const [watchlist, setWatchlist] = useState(() => {
    try {
      const saved = localStorage.getItem('watchlist');
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      console.error('Error reading watchlist from localStorage:', e);
      return [];
    }
  });

  const [toast, setToast] = useState({ visible: false, message: '' });

  // Sync with localStorage on state changes
  useEffect(() => {
    try {
      localStorage.setItem('watchlist', JSON.stringify(watchlist));
    } catch (e) {
      console.error('Error saving watchlist to localStorage:', e);
    }
  }, [watchlist]);

  // When user logs in, fetch their database watchlist from PostgreSQL and merge with local
  useEffect(() => {
    let mounted = true;

    async function loadDbWatchlist() {
      if (isSignedIn && getToken) {
        try {
          const dbItems = await fetchDbWatchlist(getToken);
          if (mounted && Array.isArray(dbItems) && dbItems.length > 0) {
            setWatchlist((localList) => {
              // Map DB items to frontend format
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

              for (const m of [...formattedDb, ...localList]) {
                if (m && m.title) {
                  const key = m.title.toLowerCase();
                  if (!titles.has(key)) {
                    titles.add(key);
                    merged.push(m);
                  }
                }
              }

              return merged;
            });
          }
        } catch (err) {
          console.warn('Could not fetch DB watchlist on startup:', err.message);
        }
      }
    }

    loadDbWatchlist();

    return () => {
      mounted = false;
    };
  }, [isSignedIn, getToken]);

  const showToast = (message) => {
    setToast({ visible: true, message });
    setTimeout(() => {
      setToast({ visible: false, message: '' });
    }, 2500);
  };

  const addToWatchlist = (movie) => {
    if (!movie || !movie.title) return;

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
    };

    setWatchlist((prev) => {
      if (prev.some((m) => m.title && m.title.toLowerCase() === movie.title.toLowerCase())) {
        showToast(`"${movie.title}" is already in your Watchlist!`);
        return prev;
      }
      showToast(`Saved to Watchlist!`);
      return [movieObj, ...prev];
    });

    // Asynchronously persist to PostgreSQL database if user is signed in
    if (isSignedIn && getToken) {
      addMovieToDbWatchlist(getToken, {
        movieId: movie.title,
        movieTitle: movie.title,
        posterPath: movieObj.poster,
        releaseYear: movieObj.release_date,
        voteAverage: movieObj.rating !== 'N/A' ? Number(movieObj.rating) : null,
        genres: typeof movieObj.genres === 'string' ? movieObj.genres.split(',').map((s) => s.trim()) : [],
      })
        .then((res) => {
          console.log(`✅ [NeonDB] Added "${movie.title}" to PostgreSQL watchlist table:`, res);
        })
        .catch((err) => {
          console.warn(`⚠️ [NeonDB] Could not sync watchlist item to backend:`, err.message);
        });
    }
  };

  const removeFromWatchlist = (title) => {
    if (!title) return;

    setWatchlist((prev) => prev.filter((m) => m.title && m.title.toLowerCase() !== title.toLowerCase()));
    showToast(`Removed from Watchlist!`);

    // Asynchronously remove from PostgreSQL database if user is signed in
    if (isSignedIn && getToken) {
      removeMovieFromDbWatchlist(getToken, title)
        .then((res) => {
          console.log(`✅ [NeonDB] Removed "${title}" from PostgreSQL watchlist table:`, res);
        })
        .catch((err) => {
          console.warn(`⚠️ [NeonDB] Could not remove watchlist item from backend:`, err.message);
        });
    }
  };

  const isInWatchlist = (title) => {
    if (!title) return false;
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
