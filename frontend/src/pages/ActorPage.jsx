import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import Loader from '../components/Loader';
import { fetchActorDetails } from '../api/movieApi';

export default function ActorPage() {
  const { actorId } = useParams();
  const navigate = useNavigate();
  const [actorData, setActorData] = useState(null);
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!actorId) return;
    let isMounted = true;
    setLoading(true);
    setError(false);

    fetchActorDetails(actorId)
      .then(data => {
        if (isMounted) {
          setActorData(data.actor);
          setMovies(data.movies || []);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }
      })
      .catch(err => {
        if (isMounted) {
          console.error(err);
          setError(true);
        }
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => { isMounted = false; };
  }, [actorId]);

  return (
    <div id="content" style={{ backgroundColor: '#ffffff', minHeight: '100vh' }}>
      <Navbar onSearchMovie={(title) => navigate(`/movie/${encodeURIComponent(title)}`)} />

      {loading && <Loader />}

      {error && !loading && (
        <div className="fail" style={{ display: 'block', textAlign: 'center', marginTop: '30px' }}>
          <h3 style={{ color: '#333333' }}>Sorry! Actor details could not be loaded.</h3>
        </div>
      )}

      {actorData && !loading && (
        <div id="actor-main-content">
          {/* Cinematic Hero Banner */}
          <div id="mycontent">
            <div id="mcontent" style={{ position: 'relative', minHeight: '65vh', overflow: 'hidden', backgroundColor: '#000000' }}>
              {/* Blurred Background Backdrop */}
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  backgroundImage: `url('${actorData.profile}')`,
                  backgroundSize: 'cover',
                  backgroundRepeat: 'no-repeat',
                  backgroundPosition: 'center 25%',
                  filter: 'blur(8px) brightness(35%)',
                  transform: 'scale(1.1)',
                  zIndex: 0
                }}
              />

              {/* Left Dark Gradient Overlay */}
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

              {/* Actor Profile Poster (Large screens) */}
              <div className="poster-lg" style={{ position: 'relative', zIndex: 2 }}>
                <img
                  className="poster"
                  style={{ borderRadius: '40px', marginLeft: '90px', marginTop: '40px' }}
                  height="400"
                  width="260"
                  src={actorData.profile}
                  alt={actorData.name}
                />
              </div>

              {/* Actor Profile Poster (Mobile screens) */}
              <div className="poster-sm text-center" style={{ position: 'relative', zIndex: 2 }}>
                <img
                  className="poster"
                  style={{ borderRadius: '40px', marginTop: '20px', marginBottom: '20px' }}
                  height="320"
                  width="220"
                  src={actorData.profile}
                  alt={actorData.name}
                />
              </div>

              {/* Actor Details Text Section */}
              <div id="details" style={{ position: 'relative', zIndex: 3, color: 'white', padding: '30px', maxWidth: '850px' }}>
                <h2 id="title" style={{ color: '#ffffff', fontWeight: 'bold', marginBottom: '15px', fontSize: '34px', letterSpacing: '0.5px' }}>
                  {actorData.name}
                </h2>
                <h6 style={{ color: '#e50914', fontWeight: 'bold', fontSize: '15px', marginBottom: '12px' }}>
                  PROFESSION: &nbsp;<span style={{ color: '#ffffff', fontWeight: 'normal' }}>{actorData.known_for_department || 'Acting'}</span>
                </h6>

                {actorData.birthday && actorData.birthday !== 'Unknown' && (
                  <h6 style={{ fontSize: '15px', marginBottom: '12px', fontWeight: 'bold' }}>
                    BORN: &nbsp;<span style={{ color: '#e0e0e0', fontWeight: 'normal' }}>{actorData.birthday}</span>
                  </h6>
                )}

                {actorData.birth_place && actorData.birth_place !== 'Unknown' && (
                  <h6 style={{ fontSize: '15px', marginBottom: '12px', fontWeight: 'bold' }}>
                    PLACE OF BIRTH: &nbsp;<span style={{ color: '#e0e0e0', fontWeight: 'normal' }}>{actorData.birth_place}</span>
                  </h6>
                )}

                <h6 style={{ maxWidth: '95%', fontSize: '15px', fontWeight: 'bold', marginTop: '15px' }}>
                  BIOGRAPHY: <br /><br />
                  <span className="actor-bio-scroll">
                    {actorData.biography || 'Biography not available for this actor.'}
                  </span>
                </h6>
              </div>
            </div>
          </div>

          {/* Known For Movies Section */}
          {movies && movies.length > 0 && (
            <>
              <div className="movie" style={{ color: '#E8E8E8', marginTop: '45px' }}>
                <center>
                  <h3 style={{ color: '#333333', fontWeight: 'bold', letterSpacing: '0.5px' }}>KNOWN FOR MOVIES</h3>
                  <h5 style={{ color: '#777777', fontSize: '15px' }}>(Click any of the movies to view full details and recommendations)</h5>
                </center>
              </div>

              <div className="movie-content">
                {movies.map((m, idx) => (
                  <div
                    key={idx}
                    className="card"
                    style={{ width: '15rem', borderRadius: '18px', boxShadow: '0 10px 10px rgba(68, 66, 66, 0.3)', margin: '10px auto' }}
                    title={m.title}
                    onClick={() => navigate(`/movie/${encodeURIComponent(m.title)}`)}
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
      )}
    </div>
  );
}
