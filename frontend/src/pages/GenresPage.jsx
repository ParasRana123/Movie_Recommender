import React from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import { useTheme } from '../context/ThemeContext';
import { GENRES_DATA } from '../data/genresData';

export default function GenresPage() {
  const navigate = useNavigate();
  const { isDark } = useTheme();

  return (
    <div
      id="content"
      style={{
        backgroundColor: isDark ? '#121212' : '#f8f9fa',
        minHeight: '100vh',
        color: isDark ? '#ffffff' : '#181818',
        paddingBottom: '60px',
        transition: 'background-color 0.3s ease, color 0.3s ease'
      }}
    >
      <Navbar onSearchMovie={(title) => navigate(`/movie/${encodeURIComponent(title)}`)} />

      <div id="genre-main-content">
        <div className="genres-page-title" style={{ textAlign: 'center', padding: '30px 20px 10px 20px' }}>
          <h1 style={{ color: '#e50914', fontSize: '36px', fontWeight: 'bold', marginBottom: '8px' }}>Movie Genres</h1>
          <p style={{ color: isDark ? '#b3b3b3' : '#666666', fontSize: '16px' }}>
            Explore curated collections across all popular genres
          </p>
        </div>

        <div className="genres-card">
          {GENRES_DATA.map((genre) => (
            <div
              key={genre.id}
              className="card-genre"
              onClick={() => navigate(`/genres/${genre.id}`)}
              style={{
                backgroundColor: isDark ? '#222222' : '#ffffff',
                borderColor: isDark ? '#333333' : '#e2e8f0',
                boxShadow: isDark ? '0 4px 20px rgba(0, 0, 0, 0.5)' : '0 4px 14px rgba(0, 0, 0, 0.08)'
              }}
            >
              <img
                src={genre.image}
                className="card-img"
                alt={genre.name}
              />
              <div
                className="genres-name"
                style={{ color: isDark ? '#ffffff' : '#181818' }}
              >
                {genre.name}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
