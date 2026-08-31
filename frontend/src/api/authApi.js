const AUTH_API_BASE = import.meta.env.VITE_AUTH_API_URL 
  ? `${import.meta.env.VITE_AUTH_API_URL.replace(/\/$/, '')}/api` 
  : (typeof window !== 'undefined' && window.location.hostname !== 'localhost'
      ? 'https://movie-recommender-bcaj.vercel.app/api'
      : 'http://localhost:5001/api');

/**
 * Helper to build auth headers with Clerk JWT token
 */
async function getAuthHeaders(getToken) {
  const token = typeof getToken === 'function' ? await getToken() : getToken;
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

/**
 * Sync authenticated Clerk user with PostgreSQL database
 */
export async function syncUserWithBackend(getToken, clerkUser) {
  if (!clerkUser) return null;
  try {
    const headers = await getAuthHeaders(getToken);
    const primaryEmail =
      clerkUser.primaryEmailAddress?.emailAddress ||
      clerkUser.emailAddresses?.[0]?.emailAddress ||
      (clerkUser.id ? `${clerkUser.id}@clerk.user` : 'user@clerk.user');

    const payload = {
      email: primaryEmail,
      firstName: clerkUser.firstName || '',
      lastName: clerkUser.lastName || '',
      username: clerkUser.username || '',
      imageUrl: clerkUser.imageUrl || '',
    };

    const res = await fetch(`${AUTH_API_BASE}/auth/sync`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP ${res.status}: Failed to sync user`);
    }

    const data = await res.json();
    console.log('✅ [PostgreSQL Sync]: User profile synced successfully:', data);
    return data;
  } catch (err) {
    console.warn(`⚠️ [PostgreSQL Sync]: Express Auth Backend unreachable at ${AUTH_API_BASE} (${err.message}). If running locally, start the server with "npm run dev" inside /server. If in production, ensure your Express backend is deployed and VITE_AUTH_API_URL is configured.`);
    return null;
  }
}

/**
 * Get current user profile from PostgreSQL
 */
export async function fetchCurrentUser(getToken) {
  try {
    const headers = await getAuthHeaders(getToken);
    const res = await fetch(`${AUTH_API_BASE}/auth/me`, {
      method: 'GET',
      headers,
    });

    if (!res.ok) {
      throw new Error('Failed to fetch user profile');
    }

    return await res.json();
  } catch (err) {
    console.error('Error in fetchCurrentUser:', err);
    return null;
  }
}

/**
 * Update user preferences
 */
export async function updateUserPreferences(getToken, preferences) {
  try {
    const headers = await getAuthHeaders(getToken);
    const res = await fetch(`${AUTH_API_BASE}/auth/preferences`, {
      method: 'PUT',
      headers,
      body: JSON.stringify(preferences),
    });

    if (!res.ok) throw new Error('Failed to update preferences');
    return await res.json();
  } catch (err) {
    console.error('Error in updateUserPreferences:', err);
    throw err;
  }
}

/**
 * Fetch authenticated user's watchlist from PostgreSQL
 */
export async function fetchDbWatchlist(getToken) {
  try {
    const headers = await getAuthHeaders(getToken);
    const res = await fetch(`${AUTH_API_BASE}/user/watchlist`, {
      method: 'GET',
      headers,
    });

    if (!res.ok) throw new Error('Failed to fetch watchlist from database');
    const data = await res.json();
    return data.data || [];
  } catch (err) {
    console.error('Error in fetchDbWatchlist:', err);
    return [];
  }
}

/**
 * Add movie to user's database watchlist
 */
export async function addMovieToDbWatchlist(getToken, movie) {
  try {
    const headers = await getAuthHeaders(getToken);
    const res = await fetch(`${AUTH_API_BASE}/user/watchlist`, {
      method: 'POST',
      headers,
      body: JSON.stringify(movie),
    });

    if (!res.ok) throw new Error('Failed to add movie to database watchlist');
    return await res.json();
  } catch (err) {
    console.error('Error in addMovieToDbWatchlist:', err);
    throw err;
  }
}

/**
 * Remove movie from user's database watchlist
 */
export async function removeMovieFromDbWatchlist(getToken, movieId) {
  try {
    const headers = await getAuthHeaders(getToken);
    const res = await fetch(`${AUTH_API_BASE}/user/watchlist/${encodeURIComponent(movieId)}`, {
      method: 'DELETE',
      headers,
    });

    if (!res.ok) throw new Error('Failed to remove movie from database watchlist');
    return await res.json();
  } catch (err) {
    console.error('Error in removeMovieFromDbWatchlist:', err);
    throw err;
  }
}
