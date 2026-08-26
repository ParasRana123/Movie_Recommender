import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { WatchlistProvider } from './context/WatchlistContext';
import Toast from './components/Toast';

import HomePage from './pages/HomePage';
import MovieDetailsPage from './pages/MovieDetailsPage';
import ActorPage from './pages/ActorPage';
import WatchlistPage from './pages/WatchlistPage';
import GenresPage from './pages/GenresPage';
import GenreDetailPage from './pages/GenreDetailPage';

import './styles/App.css';

export default function App() {
  return (
    <WatchlistProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/movie/:movieTitle" element={<MovieDetailsPage />} />
          <Route path="/actor/:actorId" element={<ActorPage />} />
          <Route path="/watchlist" element={<WatchlistPage />} />
          <Route path="/genres" element={<GenresPage />} />
          <Route path="/genres/:genreId" element={<GenreDetailPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        <Toast />
      </BrowserRouter>
    </WatchlistProvider>
  );
}
