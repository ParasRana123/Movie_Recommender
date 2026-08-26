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

  const [toast, setToast] = useState({ visible: false, message: '' });

  useEffect(() => {
    try {
      localStorage.setItem('watchlist', JSON.stringify(watchlist));
    } catch (e) {
      console.error('Error saving watchlist to localStorage:', e);
    }
  }, [watchlist]);

  const showToast = (message) => {
    setToast({ visible: true, message });
    setTimeout(() => {
      setToast({ visible: false, message: '' });
    }, 2500);
  };

  const addToWatchlist = (movie) => {
    if (!movie || !movie.title) return;
    setWatchlist(prev => {
      if (prev.some(m => m.title && m.title.toLowerCase() === movie.title.toLowerCase())) {
        showToast(`"${movie.title}" is already in your Watchlist!`);
        return prev;
      }
      showToast(`Saved to Watchlist!`);
      return [
        {
          title: movie.title,
          poster: movie.poster,
          rating: movie.rating || movie.vote_average || 'N/A',
          release_date: movie.release_date || movie.date || '',
          runtime: movie.runtime || '',
          status: movie.status || '',
          vote_count: movie.vote_count || movie.count || '',
          overview: movie.overview || ''
        },
        ...prev
      ];
    });
  };

  const removeFromWatchlist = (title) => {
    setWatchlist(prev => prev.filter(m => m.title && m.title.toLowerCase() !== title.toLowerCase()));
    showToast(`Removed from Watchlist!`);
  };

  const isInWatchlist = (title) => {
    if (!title) return false;
    return watchlist.some(m => m.title && m.title.toLowerCase() === title.toLowerCase());
  };

  return (
    <WatchlistContext.Provider
      value={{
        watchlist,
        addToWatchlist,
        removeFromWatchlist,
        isInWatchlist,
        toast
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
            transition: 'opacity 0.3s ease'
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
