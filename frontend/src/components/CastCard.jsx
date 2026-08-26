import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function CastCard({ id, name, character, profile }) {
  const navigate = useNavigate();

  const handleClick = () => {
    if (id) {
      navigate(`/actor/${id}`);
    }
  };

  const imageSrc = profile || 'https://via.placeholder.com/240x360?text=No+Photo';

  return (
    <div className="cast-card" onClick={handleClick} title={`View ${name}'s filmography`}>
      <div className="cast-image-wrapper">
        <img
          src={imageSrc}
          alt={name}
          className="cast-avatar"
          loading="lazy"
          onError={(e) => { e.target.src = 'https://via.placeholder.com/240x360?text=No+Photo'; }}
        />
        <div className="cast-hover-overlay">
          <span>Know More</span>
        </div>
      </div>
      <h5 className="cast-name">{name}</h5>
      {character && character !== 'Unknown Role' && (
        <p className="cast-character">as {character}</p>
      )}
    </div>
  );
}
