import React from 'react';
import { useNavigate, Link } from 'react-router-dom';
import Navbar from '../components/Navbar';
import { GENRES_DATA } from '../data/genresData';

export default function GenresPage() {
  const navigate = useNavigate();

  return (
    <div className="genres-page-container">
      <Navbar onSearchMovie={(title) => navigate(`/movie/${encodeURIComponent(title)}`)} />

      <main className="genres-main-content">
        <div className="genres-header-block">
          <h1 className="genres-title">Movie Genres</h1>
          <p className="genres-subtitle">
            Explore curated collections across all major cinematic genres.
          </p>
        </div>

        <div className="genres-grid-container">
          {GENRES_DATA.map((genre) => (
            <Link
              key={genre.id}
              to={`/genres/${genre.id}`}
              className="genre-portal-card"
            >
              <div className="genre-portal-image-wrapper">
                <img
                  src={genre.image}
                  alt={genre.name}
                  className="genre-portal-img"
                  loading="lazy"
                />
                <div className="genre-portal-overlay">
                  <span className="genre-explore-tag">Explore Movies →</span>
                </div>
              </div>
              <div className="genre-portal-title-row">
                <h3 className="genre-portal-name">{genre.name}</h3>
              </div>
            </Link>
          ))}
        </div>
      </main>
    </div>
  );
}
