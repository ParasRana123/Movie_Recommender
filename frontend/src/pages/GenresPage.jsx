import React from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import { GENRES_DATA } from '../data/genresData';

export default function GenresPage() {
  const navigate = useNavigate();

  return (
    <div style={{ backgroundColor: '#121212', minHeight: '100vh', color: '#ffffff', paddingBottom: '60px' }}>
      <Navbar onSearchMovie={(title) => navigate(`/movie/${encodeURIComponent(title)}`)} />

      <div id="genre-main-content">
        <div className="genres-page-title" style={{ textAlign: 'center', padding: '30px 20px 10px 20px' }}>
          <h1 style={{ color: '#e50914', fontSize: '36px', fontWeight: 'bold', marginBottom: '8px' }}>Movie Genres</h1>
          <p style={{ color: '#b3b3b3', fontSize: '16px' }}>Explore curated collections across all popular genres</p>
        </div>

        <div className="genres-card">
          {GENRES_DATA.map((genre) => (
            <div
              key={genre.id}
              className="card-genre"
              onClick={() => navigate(`/genres/${genre.id}`)}
            >
              <img
                src={genre.image}
                className="card-img"
                alt={genre.name}
              />
              <div className="genres-name">{genre.name}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
