import React from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import { GENRES_DATA } from '../data/genresData';

export default function GenresPage() {
  const navigate = useNavigate();

  return (
    <div style={{ backgroundColor: '#121212', minHeight: '100vh', color: '#ffffff' }}>
      <Navbar onSearchMovie={(title) => navigate(`/movie/${encodeURIComponent(title)}`)} />

      <div id="genre-main-content">
        <div className="genres-page-title">
          <h1>Genres</h1>
          <p>Click on any genre to explore popular movies</p>
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
