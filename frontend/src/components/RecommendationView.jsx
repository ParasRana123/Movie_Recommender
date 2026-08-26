import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useWatchlist } from '../context/WatchlistContext';
import { useTheme } from '../context/ThemeContext';

export default function RecommendationView({ movieData, onSelectRecommendedMovie }) {
  const [selectedCastModal, setSelectedCastModal] = useState(null);
  const [isDirectorBioExpanded, setIsDirectorBioExpanded] = useState(false);
  const [expandedReviews, setExpandedReviews] = useState({});

  const navigate = useNavigate();
  const { isInWatchlist, addToWatchlist, removeFromWatchlist } = useWatchlist();
  const { isDark } = useTheme();

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

  // Robust normalization: handle both List and Dict structures
  let castsList = [];
  if (Array.isArray(casts) && casts.length > 0) {
    castsList = casts;
  } else if (casts && typeof casts === 'object') {
    castsList = Object.entries(casts).map(([name, details]) => ({
      id: Array.isArray(details) ? details[0] : (details?.id || ''),
      name: name,
      character: Array.isArray(details) ? details[1] : (details?.character || ''),
      profile: Array.isArray(details) ? details[2] : (details?.profile || 'https://via.placeholder.com/250x250?text=No+Photo')
    }));
  }

  let recMoviesList = [];
  if (Array.isArray(recommended_movies) && recommended_movies.length > 0) {
    recMoviesList = recommended_movies;
  } else if (movieData.movie_cards && typeof movieData.movie_cards === 'object') {
    recMoviesList = Object.entries(movieData.movie_cards).map(([posterUrl, movieTitle]) => ({
      title: movieTitle,
      poster: posterUrl
    }));
  }

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
        genres: genres_str || (Array.isArray(genres) ? genres.join(', ') : genres)
      });
    }
  };

  const toggleReview = (idx) => {
    setExpandedReviews(prev => ({
      ...prev,
      [idx]: !prev[idx]
    }));
  };

  const headingColor = isDark ? '#ffffff' : '#333333';
  const subtitleColor = isDark ? '#aaaaaa' : '#777777';

  // Helper for direct streaming URL
  const getStreamingRedirectUrl = (provName, movieTitle, fallbackLink) => {
    const p = (provName || '').toLowerCase().trim();
    const encoded = encodeURIComponent((movieTitle || '').trim());
    if (p.includes('netflix')) return `https://www.netflix.com/search?q=${encoded}`;
    if (p.includes('prime') || p.includes('amazon')) return `https://www.primevideo.com/search/ref=atv_nb_sr?phrase=${encoded}`;
    if (p.includes('disney') || p.includes('hotstar')) return `https://www.hotstar.com/in/search?q=${encoded}`;
    if (p.includes('apple') || p.includes('itunes')) return `https://tv.apple.com/search?term=${encoded}`;
    if (p.includes('hulu')) return `https://www.hulu.com/search?q=${encoded}`;
    if (p.includes('max') || p.includes('hbo')) return `https://www.max.com/search?q=${encoded}`;
    if (p.includes('jio')) return `https://www.jiocinema.com/search/${encoded}`;
    if (p.includes('zee')) return `https://www.zee5.com/search?q=${encoded}`;
    if (p.includes('sony')) return `https://www.sonyliv.com/search?query=${encoded}`;
    return fallbackLink || `https://www.google.com/search?q=watch+${encoded}+on+${encodeURIComponent(provName || '')}`;
  };

  return (
    <div>
      {/* 1. Cinematic Hero Banner */}
      <div id="mycontent">
        <div
          id="mcontent"
          style={{
            position: 'relative',
            minHeight: '65vh',
            overflow: 'hidden',
            backgroundColor: '#000000',
            padding: '30px 0'
          }}
        >
          {/* Ambient Blurred Background Backdrop */}
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
              backgroundPosition: 'center 20%',
              filter: 'blur(10px) brightness(30%)',
              transform: 'scale(1.1)',
              zIndex: 0
            }}
          />

          {/* Left Dark Gradient Overlay for Maximum Readability */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '50%',
              height: '100%',
              background: 'linear-gradient(to right, rgba(0,0,0,0.85) 60%, transparent)',
              zIndex: 1
            }}
          />

          {/* Full Darkening Overlay */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0, 0, 0, 0.4)',
              zIndex: 1
            }}
          />

          {/* Poster (Desktop / Large Screens) */}
          <div className="poster-lg" style={{ position: 'relative', zIndex: 2, paddingLeft: '80px' }}>
            <img
              className="poster"
              style={{ borderRadius: '40px', display: 'block' }}
              height="400"
              width="260"
              src={poster}
              alt={title}
            />
          </div>

          {/* Poster (Mobile / Small Screens) */}
          <div className="poster-sm text-center" style={{ position: 'relative', zIndex: 2 }}>
            <img
              className="poster"
              style={{ borderRadius: '40px', marginTop: '20px', marginBottom: '20px' }}
              height="320"
              width="220"
              src={poster}
              alt={title}
            />
          </div>

          {/* Movie Details Text & Actions */}
          <div id="details" style={{ position: 'relative', zIndex: 3, color: 'white', padding: '20px 40px', maxWidth: '850px' }}>
            <h2 id="title" style={{ color: '#ffffff', fontWeight: 'bold', marginBottom: '15px', fontSize: '34px', letterSpacing: '0.5px' }}>
              {title}
            </h2>

            <h6 id="genres" style={{ color: '#e0e0e0', fontSize: '15px', marginBottom: '12px' }}>
              <strong>GENRE:</strong> &nbsp;{genres_str || (Array.isArray(genres) ? genres.join(', ') : genres)}
            </h6>

            <h6 id="date" style={{ color: '#e0e0e0', fontSize: '15px', marginBottom: '12px' }}>
              <strong>RELEASE DATE:</strong> &nbsp;{release_date || 'Unknown'}
            </h6>

            <h6 id="runtime" style={{ color: '#e0e0e0', fontSize: '15px', marginBottom: '12px' }}>
              <strong>RUNTIME:</strong> &nbsp;{runtime || 'Unknown'}
            </h6>

            <h6 id="status" style={{ color: '#e0e0e0', fontSize: '15px', marginBottom: '12px' }}>
              <strong>STATUS:</strong> &nbsp;{status || 'Released'}
            </h6>

            <h6 id="rating" style={{ color: '#ffd700', fontSize: '16px', marginBottom: '16px', fontWeight: 'bold' }}>
              <strong>RATING:</strong> &nbsp;★ {vote_average} / 10 &nbsp;
              <span style={{ color: '#b0b0b0', fontSize: '14px', fontWeight: 'normal' }}>({vote_count} votes)</span>
            </h6>

            <h6 style={{ color: '#ffffff', fontSize: '16px', fontWeight: 'bold', marginTop: '15px', marginBottom: '8px' }}>
              OVERVIEW:
            </h6>
            <p id="overview" style={{ color: '#d0d0d0', lineHeight: '1.7', fontSize: '14px', maxWidth: '95%' }}>
              {overview || 'No overview available.'}
            </p>

            {/* Watchlist Toggle Button */}
            <div style={{ marginTop: '20px' }}>
              <button
                id="watchlist-btn"
                className="btn btn-danger"
                onClick={handleWatchlistToggle}
                style={{
                  backgroundColor: inWatchlist ? '#28a745' : '#e50914',
                  borderColor: inWatchlist ? '#28a745' : '#e50914',
                  borderRadius: '25px',
                  padding: '8px 24px',
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
                <span>{inWatchlist ? '✓ Added to Watchlist' : '+ Add to Watchlist'}</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Streaming Platforms & Video Section */}
      <div id="streaming-platforms" className="streaming-platforms-section">
        <div id="video-section" style={{ marginTop: '30px', marginRight: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexWrap: 'wrap', gap: '20px' }}>
          {teaser && teaser !== 'None' && teaser.includes('embed/') && !teaser.endsWith('embed/') && (
            <div className="teaser movie-video-wrapper">
              <h3 style={{ color: 'white', marginTop: '20px', fontSize: '18px' }}>🎥 Watch the Teaser</h3>
              <iframe
                className="movie-video-iframe"
                src={teaser}
                title={`${title} Teaser`}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowFullScreen
                loading="lazy"
              />
            </div>
          )}
          {trailer && trailer !== 'None' && trailer.includes('embed/') && !trailer.endsWith('embed/') && (
            <div className="trailer movie-video-wrapper">
              <h3 style={{ color: 'white', fontSize: '18px' }}>🎬 Watch the Trailer</h3>
              <iframe
                className="movie-video-iframe"
                src={trailer}
                title={`${title} Trailer`}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowFullScreen
                loading="lazy"
              />
            </div>
          )}
        </div>

        <div className="vertical-line" />

        <div className="providers-container">
          <h6>Streaming on:</h6>
          <div className="streaming_platform">
            {streaming_availability && streaming_availability.length > 0 ? (
              streaming_availability.map((rawProv, i) => {
                const provName = Array.isArray(rawProv) ? rawProv[0] : (rawProv?.provider_name || 'Watch Online');
                const logoUrl = Array.isArray(rawProv) ? rawProv[1] : (rawProv?.logo_path || 'https://via.placeholder.com/60?text=Stream');
                const watchUrl = Array.isArray(rawProv) ? rawProv[2] : (rawProv?.watch_url || '');
                const targetUrl = getStreamingRedirectUrl(provName, title, watchUrl);

                return (
                  <a
                    key={i}
                    href={targetUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="provider"
                    title={`Watch "${title}" on ${provName}`}
                  >
                    <img src={logoUrl} alt={provName} className="provider-logo" />
                    <p>{provName}</p>
                  </a>
                );
              })
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

      {/* 4. Top Cast Section (Left-aligned IMDb Style, 1 Line Horizontal Scroll on Mobile, Circular Photos) */}
      {castsList && castsList.length > 0 && (
        <div className="section-container" style={{ maxWidth: '1400px', margin: '45px auto 0 auto', padding: '0 20px', width: '100%', boxSizing: 'border-box' }}>
          <div className="section-header-left" style={{ textAlign: 'left', marginBottom: '18px' }}>
            <h3 style={{ color: headingColor, fontWeight: 'bold', fontSize: '26px', margin: 0, display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ color: '#e50914', fontSize: '28px', fontWeight: 'bold' }}>|</span> Top Cast
            </h3>
          </div>

          <div className="movie-content cast-content-scroll" style={{ justifyContent: 'flex-start' }}>
            {castsList.map((c, idx) => (
              <div
                key={idx}
                className="cast-card-item"
                title={`Click to know more about ${c.name}`}
                onClick={() => setSelectedCastModal(c)}
              >
                <div className="imghvr cast-imghvr">
                  <img
                    className="card-img-top cast-img"
                    alt={`${c.name} - profile`}
                    src={c.profile || 'https://via.placeholder.com/250x250?text=No+Photo'}
                  />
                  <figcaption className="img cast-fig-overlay">
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
                <div className="card-body" style={{ textAlign: 'center', padding: '10px 4px' }}>
                  <h5 className="card-title" style={{ fontSize: '15px', fontWeight: 'bold', margin: '4px 0' }}>{c.name}</h5>
                  {c.character && (
                    <h6 style={{ color: isDark ? '#aaa' : '#756969', fontSize: '13px', margin: 0 }}>
                      {c.character}
                    </h6>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 5. About The Director (Left-aligned IMDb Style Card - Full Width) */}
      <div className="section-container" style={{ maxWidth: '1400px', margin: '45px auto 0 auto', padding: '0 20px', width: '100%', boxSizing: 'border-box' }}>
        <div className="section-header-left" style={{ textAlign: 'left', marginBottom: '18px' }}>
          <h3 style={{ color: headingColor, fontWeight: 'bold', fontSize: '26px', margin: 0, display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ color: '#e50914', fontSize: '28px', fontWeight: 'bold' }}>|</span> About the Director
          </h3>
        </div>

        <div className="director-section" style={{ width: '100%', maxWidth: '100%', margin: '0 0 30px 0', boxSizing: 'border-box' }}>
          {director_image && (
            <div className="director-image">
              <img
                src={director_image}
                alt={`Director ${director_name}`}
                onError={(e) => { e.target.src = 'https://via.placeholder.com/220x280?text=No+Photo'; }}
              />
            </div>
          )}
          <div className="director-info">
            <h2>Director: {director_name || 'Unknown'}</h2>
            <div className="director-bio-container">
              <p style={{ margin: 0 }}>
                <strong>Bio: </strong>
                {director_bio && director_bio.length > 280 && !isDirectorBioExpanded
                  ? `${director_bio.slice(0, 280)}...`
                  : (director_bio || 'Biography not available for this director.')}
                {director_bio && director_bio.length > 280 && (
                  <button
                    onClick={() => setIsDirectorBioExpanded(!isDirectorBioExpanded)}
                    className="read-more-btn"
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#e50914',
                      fontWeight: 'bold',
                      fontSize: '14px',
                      cursor: 'pointer',
                      marginLeft: '6px',
                      padding: 0,
                      textDecoration: 'underline'
                    }}
                  >
                    {isDirectorBioExpanded ? 'Show Less' : 'See More'}
                  </button>
                )}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 6. User Reviews Section (Left-aligned IMDb Style) */}
      <div className="section-container reviews-container" style={{ maxWidth: '1400px', margin: '45px auto 0 auto', padding: '0 20px', width: '100%', boxSizing: 'border-box', textAlign: 'left' }}>
        <div className="section-header-left" style={{ textAlign: 'left', marginBottom: '18px' }}>
          <h3 style={{ color: headingColor, fontWeight: 'bold', fontSize: '26px', margin: 0, display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ color: '#e50914', fontSize: '28px', fontWeight: 'bold' }}>|</span> User Reviews
          </h3>
        </div>

        {reviews && reviews.length > 0 ? (
          <div className="reviews-table-wrapper" style={{ margin: 0, padding: 0 }}>
            
            {/* Desktop / Tablet Reviews Table */}
            <div className="table-responsive reviews-desktop-table" style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch', width: '100%' }}>
              <table className="table table-bordered table-custom" bordercolor="white" style={{ color: 'white', minWidth: '650px', borderRadius: '12px', overflow: 'hidden' }}>
                <thead>
                  <tr style={{ backgroundColor: isDark ? '#1f1f1f' : '#2d2d2d' }}>
                    <th scope="col" style={{ width: '55%', color: '#ffffff', fontSize: '16px', fontWeight: 'bold', textAlign: 'left', paddingLeft: '16px' }}>
                      User Comments
                    </th>
                    <th scope="col" style={{ width: '22%', color: '#ffffff', fontSize: '16px', fontWeight: 'bold', textAlign: 'center' }}>
                      Author & Rating
                    </th>
                    <th scope="col" style={{ width: '23%', color: '#ffffff', fontSize: '16px', fontWeight: 'bold', textAlign: 'center' }}>
                      Sentiment Analysis
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {reviews.map((rev, i) => {
                    const isExpanded = !!expandedReviews[i];
                    const isLong = rev.content && rev.content.length > 220;
                    const contentToShow = isLong && !isExpanded ? `${rev.content.slice(0, 220)}...` : rev.content;

                    return (
                      <tr key={i} style={{ backgroundColor: '#e5091485' }}>
                        <td style={{ textAlign: 'left', padding: '14px 16px', fontSize: '14px', lineHeight: '1.6', color: 'white' }}>
                          <span>{contentToShow}</span>
                          {isLong && (
                            <button
                              onClick={() => toggleReview(i)}
                              style={{
                                background: 'none',
                                border: 'none',
                                color: '#ffd700',
                                fontWeight: 'bold',
                                fontSize: '13px',
                                cursor: 'pointer',
                                marginLeft: '6px',
                                padding: 0,
                                textDecoration: 'underline'
                              }}
                            >
                              {isExpanded ? 'Show Less' : 'See More'}
                            </button>
                          )}
                        </td>
                        <td style={{ verticalAlign: 'middle', textAlign: 'center', color: 'white', padding: '10px' }}>
                          <strong style={{ fontSize: '15px' }}>{rev.author || 'Anonymous'}</strong><br />
                          <span style={{ color: '#ffd700', fontSize: '15px', fontWeight: 'bold' }}>
                            ★ {rev.rating || 'N/A'}
                          </span>
                        </td>
                        <td style={{ verticalAlign: 'middle', textAlign: 'center', fontSize: '15px', color: 'white', padding: '10px' }}>
                          <strong>{rev.sentiment}</strong> :{' '}
                          <span style={{ fontSize: '22px' }}>
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
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Mobile Dedicated Reviews Cards View (Stacking clean cards with 'See More') */}
            <div className="reviews-mobile-cards">
              {reviews.map((rev, i) => {
                const isExpanded = !!expandedReviews[i];
                const isLong = rev.content && rev.content.length > 160;
                const contentToShow = isLong && !isExpanded ? `${rev.content.slice(0, 160)}...` : rev.content;

                return (
                  <div
                    key={i}
                    className="review-mobile-card"
                    style={{
                      backgroundColor: '#e5091490',
                      borderRadius: '16px',
                      padding: '16px',
                      marginBottom: '14px',
                      color: 'white',
                      textAlign: 'left',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                      border: '1px solid rgba(255,255,255,0.15)'
                    }}
                  >
                    {/* Header with author, rating, sentiment */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px', borderBottom: '1px solid rgba(255,255,255,0.2)', paddingBottom: '8px' }}>
                      <div>
                        <strong style={{ fontSize: '15px', color: '#ffffff' }}>{rev.author || 'Anonymous'}</strong>
                        <div style={{ color: '#ffd700', fontSize: '13px', fontWeight: 'bold', marginTop: '2px' }}>
                          ★ {rev.rating || 'N/A'}
                        </div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <span style={{ fontSize: '13px', fontWeight: 'bold', backgroundColor: 'rgba(0,0,0,0.4)', padding: '3px 8px', borderRadius: '10px', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                          {rev.sentiment} {rev.sentiment === 'Good' ? '😃' : '😔'}
                        </span>
                        {rev.confidence && (
                          <div style={{ fontSize: '11px', color: '#f0f0f0', marginTop: '2px' }}>
                            {rev.confidence}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Review Body with See More */}
                    <div style={{ fontSize: '14px', lineHeight: '1.6', color: '#ffffff' }}>
                      {contentToShow}
                      {isLong && (
                        <button
                          onClick={() => toggleReview(i)}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: '#ffd700',
                            fontWeight: 'bold',
                            fontSize: '13px',
                            cursor: 'pointer',
                            marginLeft: '6px',
                            padding: 0,
                            textDecoration: 'underline'
                          }}
                        >
                          {isExpanded ? 'Show Less' : 'See More'}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

          </div>
        ) : (
          <div style={{ color: headingColor, margin: '20px 0' }}>
            <h4 style={{ color: subtitleColor, fontWeight: 'normal' }}>No reviews available for this movie yet. Stay tuned!</h4>
          </div>
        )}
      </div>

      {/* 7. Recommended Movies Section */}
      {recMoviesList && recMoviesList.length > 0 && (
        <div className="section-container" style={{ maxWidth: '1400px', margin: '45px auto 30px auto', padding: '0 20px', width: '100%', boxSizing: 'border-box' }}>
          <div className="section-header-left" style={{ textAlign: 'left', marginBottom: '18px' }}>
            <h3 style={{ color: headingColor, fontWeight: 'bold', fontSize: '26px', margin: 0, display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ color: '#e50914', fontSize: '28px', fontWeight: 'bold' }}>|</span> More Like This
            </h3>
          </div>

          <div className="movie-content recommended-content-scroll">
            {recMoviesList.map((m, idx) => (
              <div
                key={idx}
                className="card recommended-card-item"
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
                    alt={`${m.title} - poster`}
                    src={m.poster || 'https://via.placeholder.com/240x360?text=No+Poster'}
                  />
                  <figcaption className="fig">
                    <button className="card-btn btn btn-danger"> Click Me </button>
                  </figcaption>
                </div>
                <div className="card-body" style={{ padding: '14px 10px', textAlign: 'center' }}>
                  <h5 className="card-title" style={{ fontSize: '15px', fontWeight: 'bold', margin: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {m.title}
                  </h5>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
