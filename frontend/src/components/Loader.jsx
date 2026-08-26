import React from 'react';

export default function Loader({ text = "FINDING BEST RECOMMENDATIONS..." }) {
  return (
    <div className="loader-container">
      <div className="custom-spinner"></div>
      <p className="loader-text">{text}</p>
    </div>
  );
}
