import React, { createContext, useContext, useState, useEffect } from 'react';

const WatchlistContext = createContext();

export function WatchlistProvider({ children }) {
  const [watchlist, setWatchlist] = useState(() => {
    try {
      const saved = localStorage.getItem('watchlist');
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      console.error('Error reading watchlist from localStorage:', e);
      return [];
    }
  });

  const [lastRemoved, setLastRemoved] = useState(null);
  const [toast, setToast] = useState({ visible: false, message: '', canUndo: false });

  useEffect(() => {
    try {
      localStorage.setItem('watchlist', JSON.stringify(watchlist));
    } catch (e) {
      console.error('Error saving watchlist to localStorage:', e);
    }
  }, [watchlist]);

  const showToast = (message, canUndo = false) => {
    setToast({ visible: true, message, canUndo });
    setTimeout(() => {
      setToast(prev => ({ ...prev, visible: false }));
    }, 3000);
  };

  const addToWatchlist = (movie) => {
    if (!movie || !movie.title) return;
    setWatchlist(prev => {
      if (prev.some(m => m.title.toLowerCase() === movie.title.toLowerCase())) {
        showToast(`"${movie.title}" is already in your Watchlist!`, false);
        return prev;
      }
      showToast(`Added "${movie.title}" to Watchlist!`, false);
      return [
        {
          id: movie.movie_id || movie.id,
          title: movie.title,
          poster: movie.poster,
          rating: movie.vote_average || movie.rating || 'N/A',
          release_date: movie.release_date || movie.release_year || '',
          genres: movie.genres || movie.genres_str || '',
          runtime: movie.runtime || ''
        },
        ...prev
      ];
    });
  };

  const removeFromWatchlist = (title) => {
    const item = watchlist.find(m => m.title.toLowerCase() === title.toLowerCase());
    if (item) {
      setLastRemoved(item);
      setWatchlist(prev => prev.filter(m => m.title.toLowerCase() !== title.toLowerCase()));
      showToast(`Removed "${item.title}" from Watchlist`, true);
    }
  };

  const undoRemove = () => {
    if (lastRemoved) {
      setWatchlist(prev => [lastRemoved, ...prev]);
      showToast(`Restored "${lastRemoved.title}" to Watchlist`, false);
      setLastRemoved(null);
    }
  };

  const isInWatchlist = (title) => {
    if (!title) return false;
    return watchlist.some(m => m.title.toLowerCase() === title.toLowerCase());
  };

  return (
    <WatchlistContext.Provider
      value={{
        watchlist,
        addToWatchlist,
        removeFromWatchlist,
        undoRemove,
        isInWatchlist,
        toast,
        setToast
      }}
    >
      {children}
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
