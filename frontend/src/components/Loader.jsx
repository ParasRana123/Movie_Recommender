import React from 'react';

export default function Loader() {
  return (
    <div
      id="loader"
      style={{
        display: 'block',
        position: 'fixed',
        zIndex: 99999,
        left: 0,
        top: 0,
        width: '100%',
        height: '100%',
        backgroundImage: 'url("/loader.gif")',
        backgroundSize: '20%',
        backgroundPosition: '50% 50%',
        backgroundColor: 'rgba(255, 255, 255, 1)',
        backgroundRepeat: 'no-repeat'
      }}
    />
  );
}
