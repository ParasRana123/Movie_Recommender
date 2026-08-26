const API_BASE = '/api';

/**
 * Fetch suggestions list for autocomplete
 */
export async function fetchSuggestions() {
  try {
    const res = await fetch(`${API_BASE}/suggestions`);
    if (!res.ok) throw new Error('Failed to fetch suggestions');
    return await res.json();
  } catch (err) {
    console.error('Error in fetchSuggestions:', err);
    return [];
  }
}

/**
 * Fetch full movie recommendations & metadata
 */
export async function fetchRecommendations(title) {
  try {
    const res = await fetch(`${API_BASE}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, name: title })
    });
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.error || 'Movie not found in database');
    }
    return await res.json();
  } catch (err) {
    console.error('Error in fetchRecommendations:', err);
    throw err;
  }
}

/**
 * Fetch movie details by title
 */
export async function fetchMovieDetails(title) {
  try {
    const res = await fetch(`${API_BASE}/movie/${encodeURIComponent(title)}`);
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.error || 'Movie details not found');
    }
    return await res.json();
  } catch (err) {
    console.error('Error in fetchMovieDetails:', err);
    throw err;
  }
}

/**
 * Fetch actor details & filmography
 */
export async function fetchActorDetails(actorId) {
  try {
    const res = await fetch(`${API_BASE}/actor/${encodeURIComponent(actorId)}`);
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.error || 'Actor details not found');
    }
    return await res.json();
  } catch (err) {
    console.error('Error in fetchActorDetails:', err);
    throw err;
  }
}

/**
 * Fetch genres list
 */
export async function fetchGenres() {
  try {
    const res = await fetch(`${API_BASE}/genres`);
    if (!res.ok) throw new Error('Failed to fetch genres');
    return await res.json();
  } catch (err) {
    console.error('Error in fetchGenres:', err);
    return [];
  }
}

/**
 * Fetch movies for a specific genre
 */
export async function fetchGenreMovies(genreName) {
  try {
    const res = await fetch(`${API_BASE}/genres/${encodeURIComponent(genreName)}`);
    if (!res.ok) throw new Error(`Failed to fetch movies for genre ${genreName}`);
    return await res.json();
  } catch (err) {
    console.error('Error in fetchGenreMovies:', err);
    throw err;
  }
}

/**
 * Fetch trending movies
 */
export async function fetchTrendingMovies() {
  try {
    const res = await fetch(`${API_BASE}/trending`);
    if (!res.ok) throw new Error('Failed to fetch trending movies');
    return await res.json();
  } catch (err) {
    console.error('Error in fetchTrendingMovies:', err);
    return [];
  }
}

/**
 * Fetch upcoming movies
 */
export async function fetchUpcomingMovies() {
  try {
    const res = await fetch(`${API_BASE}/upcoming`);
    if (!res.ok) throw new Error('Failed to fetch upcoming movies');
    return await res.json();
  } catch (err) {
    console.error('Error in fetchUpcomingMovies:', err);
    return [];
  }
}

/**
 * Fetch curated top movies
 */
export async function fetchTopMovies() {
  try {
    const res = await fetch(`${API_BASE}/top-movies`);
    if (!res.ok) throw new Error('Failed to fetch top movies');
    return await res.json();
  } catch (err) {
    console.error('Error in fetchTopMovies:', err);
    return [];
  }
}
