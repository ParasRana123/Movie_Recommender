import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useWatchlist } from '../context/WatchlistContext';

export default function RecommendationView({ movieData, onSelectRecommendedMovie }) {
  const [selectedCastModal, setSelectedCastModal] = useState(null);
  const navigate = useNavigate();
  const { isInWatchlist, addToWatchlist, removeFromWatchlist } = useWatchlist();

  if (!movieData) return null;

  const {
    title,
    poster,
    backdrop,
    overview,
    vote_average,
    vote_count,
    release_date,
    runtime,
    status,
    budget,
    revenue,
    original_language,
    genres = [],
    genres_str = '',
    trailer,
    teaser,
    streaming_availability = [],
    director_name,
    director_image,
    director_bio,
    casts = [],
    reviews = [],
    recommended_movies = []
  } = movieData;

  const inWatchlist = isInWatchlist(title);

  const handleWatchlistToggle = () => {
    if (inWatchlist) {
      removeFromWatchlist(title);
    } else {
      addToWatchlist({
        title,
        poster,
        rating: vote_average,
        release_date,
        genres: Array.isArray(genres) ? genres.join(', ') : (genres_str || genres),
        runtime,
        vote_count,
        status,
        overview
      });
    }
  };

  const displayGenres = genres_str || (Array.isArray(genres) ? genres.join(', ') : genres);

  return (
    <div id="mycontent" style={{ color: 'white' }}>
      {/* 1. Hero Movie Overview Section with proper top/bottom padding */}
      <div
        id="mcontent"
        style={{
          position: 'relative',
          minHeight: '67vh',
          overflow: 'hidden',
          marginTop: '0px',
          padding: '35px 0 45px 0',
          display: 'flex',
          alignItems: 'center'
        }}
      >
        {/* Background Image with Blur and Darkened Overlay */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundImage: `url('${backdrop || poster}')`,
            backgroundSize: 'cover',
            backgroundRepeat: 'no-repeat',
            backgroundPosition: 'center',
            filter: 'blur(3px)',
            width: '100%',
            height: '100%',
            zIndex: 0
          }}
        />

        {/* Left Black Gradient Overlay */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '40%',
            height: '100%',
            background: 'linear-gradient(to right, black 60%, transparent)',
            zIndex: 1
          }}
        />

        {/* Darkened Full Overlay */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.5)',
            width: '100%',
            height: '100%',
            zIndex: 1
          }}
        />

        {/* Poster Image (Large screen) with comfortable left padding */}
        <div className="poster-lg" style={{ position: 'relative', zIndex: 2, flexShrink: 0, paddingLeft: '80px' }}>
          <img
            className="poster"
            style={{ borderRadius: '40px', display: 'block' }}
            height="400"
            width="250"
            src={poster || 'https://via.placeholder.com/250x400?text=No+Poster'}
            alt={title}
          />
        </div>

        {/* Poster Image (Small screen) */}
        <div className="poster-sm text-center" style={{ position: 'relative', zIndex: 2, padding: '20px 0' }}>
          <img
            className="poster"
            style={{ borderRadius: '40px', marginBottom: '5%' }}
            height="400"
            width="250"
            src={poster || 'https://via.placeholder.com/250x400?text=No+Poster'}
            alt={title}
          />
        </div>

        {/* Details Section (Text) with generous padding */}
        <div id="details" style={{ position: 'relative', zIndex: 3, color: 'white', padding: '10px 45px', flex: 1, maxWidth: '85%' }}>
          <h6 id="title" style={{ zIndex: 3, color: 'white', fontSize: '18px', fontWeight: 'bold', marginBottom: '14px' }}>
            TITLE: &nbsp;{title}
          </h6>
          <h6 id="overview" style={{ maxWidth: '90%', zIndex: 3, color: 'white', lineHeight: '1.6', marginBottom: '14px' }}>
            OVERVIEW: <br /><br />
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{overview}
          </h6>
          <h6 id="vote_average" style={{ zIndex: 3, color: 'white', marginBottom: '14px' }}>
            RATING: &nbsp;{vote_average}/10 ({vote_count} votes)
          </h6>
          <h6 id="genres" style={{ zIndex: 3, color: 'white', marginBottom: '14px' }}>
            GENRE: &nbsp;{displayGenres}
          </h6>
          <h6 id="date" style={{ zIndex: 3, color: 'white', marginBottom: '14px' }}>
            RELEASE DATE: &nbsp;{release_date}
          </h6>
          <h6 id="runtime" style={{ zIndex: 3, color: 'white', marginBottom: '14px' }}>
            RUNTIME: &nbsp;{runtime}
          </h6>
          <h6 id="status" style={{ zIndex: 3, color: 'white', marginBottom: '14px' }}>
            STATUS: &nbsp;{status}
          </h6>

          {/* Add to Watchlist Button */}
          <div style={{ marginTop: '18px', zIndex: 4, position: 'relative' }}>
            <button
              id="watchlist-btn"
              className={`btn btn-danger ${inWatchlist ? 'in-watchlist' : ''}`}
              onClick={handleWatchlistToggle}
              style={{
                backgroundColor: inWatchlist ? '#28a745' : '#e50914',
                borderColor: inWatchlist ? '#28a745' : '#e50914',
                borderRadius: '25px',
                padding: '8px 22px',
                fontWeight: 'bold',
                fontSize: '15px',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                boxShadow: inWatchlist ? '0 4px 15px rgba(40, 167, 69, 0.4)' : '0 4px 15px rgba(229, 9, 20, 0.4)',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
            >
              <img
                id="watchlist-btn-icon"
                src={inWatchlist ? '/images/bookmark_tick.svg' : '/images/add_bookmark.svg'}
                width="20"
                height="20"
                style={{ filter: 'invert(1)' }}
                alt="Watchlist"
              />
              <span id="watchlist-btn-text">
                {inWatchlist ? 'In Watchlist ✓' : 'Add to Watchlist'}
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* 2. Streaming Platforms & Video Section */}
      <div id="streaming-platforms">
        <div id="video-section" style={{ marginTop: '30px', marginRight: '30px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {teaser && (
            <div className="teaser">
              <h3 style={{ color: 'white', marginTop: '20px' }}>🎥 Watch the Teaser</h3>
              <iframe
                style={{ marginBottom: '40px' }}
                width="250"
                height="315"
                src={teaser}
                frameBorder="0"
                allowFullScreen
                title="Teaser"
              />
            </div>
          )}
          {trailer && (
            <div className="trailer">
              <h3 style={{ color: 'white' }}>🎬 Watch the Trailer</h3>
              <iframe
                style={{ marginBottom: '20px' }}
                width="560"
                height="315"
                src={trailer}
                frameBorder="0"
                allowFullScreen
                title="Trailer"
              />
            </div>
          )}
        </div>

        <div className="vertical-line" />

        <div className="providers-container">
          <h6>Streaming on:</h6>
          <div className="streaming_platform">
            {streaming_availability && streaming_availability.length > 0 ? (
              streaming_availability.map((prov, i) => (
                <div key={i} className="provider">
                  <img src={prov.logo_path} alt={prov.provider_name} width="50" />
                  <p style={{ color: 'white' }}>{prov.provider_name}</p>
                </div>
              ))
            ) : (
              <p style={{ color: '#aaa', fontSize: '14px' }}>Check local streaming platforms</p>
            )}
          </div>

          {budget && budget !== 'N/A' && (
            <h6 id="budget" style={{ zIndex: 3 }}>BUDGET: $&nbsp;{budget}</h6>
          )}

          {revenue && revenue !== 'N/A' && (
            <h6 id="budget" style={{ zIndex: 3 }}>REVENUE: $&nbsp;{revenue}</h6>
          )}

          {original_language && (
            <h6 id="budget" style={{ zIndex: 3 }}>ORIGINAL LANGUAGE: &nbsp;{original_language}</h6>
          )}
        </div>
      </div>

      <br />

      {/* 3. Cast Modal Popup */}
      {selectedCastModal && (
        <div className="modal-overlay-custom" onClick={() => setSelectedCastModal(null)}>
          <div className="modal-content-custom" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header-custom" style={{ backgroundColor: '#e50914', color: 'white' }}>
              <h5 className="modal-title">{selectedCastModal.name}</h5>
              <button className="modal-close-x" onClick={() => setSelectedCastModal(null)}>
                &times;
              </button>
            </div>
            <div className="modal-body-custom">
              <img
                className="profile-pic"
                src={selectedCastModal.profile || 'https://via.placeholder.com/250x400?text=No+Photo'}
                alt={selectedCastModal.name}
              />
              <div style={{ marginLeft: '20px' }}>
                <p><strong>Character:</strong> {selectedCastModal.character || 'Actor'}</p>
                <p>Click below to explore full filmography and biography.</p>
                <button
                  className="btn btn-danger"
                  style={{ backgroundColor: '#e50914', borderColor: '#e50914', marginTop: '15px' }}
                  onClick={() => {
                    setSelectedCastModal(null);
                    navigate(`/actor/${selectedCastModal.id}`);
                  }}
                >
                  View Full Actor Page →
                </button>
              </div>
            </div>
            <div className="modal-footer-custom">
              <button
                type="button"
                className="btn btn-secondary"
                style={{ backgroundColor: '#6c757d', color: '#fff', border: 'none', padding: '6px 16px', borderRadius: '4px', cursor: 'pointer' }}
                onClick={() => setSelectedCastModal(null)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 4. Top Cast Section */}
      {casts && casts.length > 0 && (
        <>
          <div className="movie" style={{ color: 'black' }}>
            <center>
              <h3>TOP CAST</h3>
              <h5>(Click on the cast to know more)</h5>
            </center>
          </div>

          <div className="movie-content">
            {casts.map((c, idx) => (
              <div
                key={idx}
                title={`Click to know more about ${c.name}`}
                onClick={() => setSelectedCastModal(c)}
              >
                <div className="imghvr">
                  <img
                    className="card-img-top cast-img"
                    alt={`${c.name} - profile`}
                    src={c.profile || 'https://via.placeholder.com/250x250?text=No+Photo'}
                  />
                  <figcaption className="img">
                    <button
                      className="card-btn btn btn-danger"
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate(`/actor/${c.id}`);
                      }}
                    >
                      Know More
                    </button>
                  </figcaption>
                </div>
                <div className="card-body">
                  <h5 className="card-title">{c.name}</h5>
                  {c.character && (
                    <h5 className="card-title">
                      <span style={{ color: '#756969', fontSize: '20px' }}>
                        Character: {c.character}
                      </span>
                    </h5>
                  )}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* 5. About The Director */}
      <div>
        <h3 style={{ color: 'black', textAlign: 'center' }}>ABOUT THE DIRECTOR</h3>
      </div>

      <div className="director-section">
        {director_image && (
          <div className="director-image">
            <img src={director_image} alt="Director" />
          </div>
        )}
        <div className="director-info">
          <h2>Director: {director_name}</h2>
          <p><strong>Bio:</strong> {director_bio}</p>
        </div>
      </div>

      {/* 6. User Reviews Table */}
      <center>
        <div className="reviews-container">
          <h2 style={{ color: 'black', margin: '40px' }}>USER REVIEWS</h2>
          {reviews && reviews.length > 0 ? (
            <div className="col-md-12" style={{ margin: '0 auto', marginTop: '25px' }}>
              <table className="table table-bordered table-custom" bordercolor="white" style={{ color: 'white' }}>
                <thead>
                  <tr>
                    <th scope="col" style={{ width: '55%', color: '#333333', fontSize: '18px', fontWeight: 'bold', textAlign: 'center' }}>
                      User Comments
                    </th>
                    <th scope="col" style={{ width: '20%', color: '#333333', fontSize: '18px', fontWeight: 'bold', textAlign: 'center' }}>
                      Author & Rating
                    </th>
                    <th scope="col" style={{ width: '25%', color: '#333333', fontSize: '18px', fontWeight: 'bold', textAlign: 'center' }}>
                      Sentiment Analysis
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {reviews.map((rev, i) => (
                    <tr key={i} style={{ backgroundColor: '#e5091485' }}>
                      <td style={{ textAlign: 'left', padding: '15px', fontSize: '15px', lineHeight: '1.6', color: 'white' }}>
                        {rev.content}
                      </td>
                      <td style={{ verticalAlign: 'middle', textAlign: 'center', color: 'white' }}>
                        <strong>{rev.author}</strong><br />
                        <span style={{ color: '#ffd700', fontSize: '17px', fontWeight: 'bold' }}>
                          ★ {rev.rating}
                        </span>
                      </td>
                      <td style={{ verticalAlign: 'middle', textAlign: 'center', fontSize: '16px', color: 'white' }}>
                        <strong>{rev.sentiment}</strong> :{' '}
                        <span style={{ fontSize: '24px' }}>
                          {rev.sentiment === 'Good' ? '😃' : '😔'}
                        </span>
                        {rev.confidence && (
                          <>
                            <br />
                            <small style={{ color: '#f8f9fa' }}>({rev.confidence} confident)</small>
                          </>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{ color: 'white', margin: '30px' }}>
              <h3 style={{ color: '#333333' }}>No reviews available for this movie yet. Stay tuned!</h3>
            </div>
          )}
        </div>
      </center>

      {/* 7. Recommended Movies Grid */}
      {recommended_movies && recommended_movies.length > 0 && (
        <>
          <div className="movie" style={{ color: '#E8E8E8' }}>
            <center>
              <h3 style={{ color: '#333333' }}>RECOMMENDED MOVIES FOR YOU</h3>
              <h5>(Click any of the movies to get recommendation)</h5>
            </center>
          </div>

          <div className="movie-content">
            {recommended_movies.map((m, idx) => (
              <div
                key={idx}
                className="card"
                style={{ width: '15rem', borderRadius: '18px', boxShadow: '0 10px 10px rgba(68, 66, 66, 0.3)', margin: '10px auto' }}
                title={m.title}
                onClick={() => {
                  if (onSelectRecommendedMovie) {
                    onSelectRecommendedMovie(m.title);
                  }
                }}
              >
                <div className="imghvr">
                  <img
                    className="card-img-top"
                    height="360"
                    width="240"
                    alt={`${m.title} - poster`}
                    src={m.poster || 'https://via.placeholder.com/240x360?text=No+Poster'}
                  />
                  <figcaption className="fig">
                    <button className="card-btn btn btn-danger"> Click Me </button>
                  </figcaption>
                </div>
                <div className="card-body">
                  <h5 className="card-title">{m.title}</h5>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
