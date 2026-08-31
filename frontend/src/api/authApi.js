const AUTH_API_BASE = import.meta.env.VITE_AUTH_API_URL 
  ? `${import.meta.env.VITE_AUTH_API_URL.replace(/\/$/, '')}/api` 
  : (typeof window !== 'undefined' && window.location.hostname !== 'localhost'
      ? 'https://movie-recommender-bcaj.vercel.app/api'
      : 'http://localhost:5001/api');

/**
 * Helper to build auth headers with Clerk JWT token
 */
async function getAuthHeaders(getToken) {
  let token = null;
  if (typeof getToken === 'function') {
    try {
      token = await getToken();
    } catch (e) {
      console.warn('Error getting Clerk token:', e);
    }
  } else if (typeof getToken === 'string') {
    token = getToken;
  }

  // Fallback to window.Clerk session if available
  if (!token && typeof window !== 'undefined' && window.Clerk?.session) {
    try {
      token = await window.Clerk.session.getToken();
    } catch (e) {}
  }

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

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}: Failed to sync user`);
    }

    console.log('✅ [PostgreSQL Sync]: User profile synced successfully:', data);
    return data;
  } catch (err) {
    console.warn(`⚠️ [PostgreSQL Sync]: Express Auth Backend unreachable at ${AUTH_API_BASE} (${err.message}).`);
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

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}: Failed to fetch user profile`);
    }

    return data;
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

    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || 'Failed to update preferences');
    return data;
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
    // If no auth token is present, user is signed out
    if (!headers.Authorization) return [];

    const res = await fetch(`${AUTH_API_BASE}/user/watchlist`, {
      method: 'GET',
      headers,
    });

    if (!res.ok) return [];
    const data = await res.json().catch(() => ({}));
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
    if (!headers.Authorization) {
      // User is not signed in; stored locally only
      return null;
    }

    const res = await fetch(`${AUTH_API_BASE}/user/watchlist`, {
      method: 'POST',
      headers,
      body: JSON.stringify(movie),
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}: Failed to add movie`);
    }

    return data;
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
    if (!headers.Authorization) {
      return null;
    }

    const res = await fetch(`${AUTH_API_BASE}/user/watchlist/${encodeURIComponent(movieId)}`, {
      method: 'DELETE',
      headers,
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}: Failed to remove movie`);
    }

    return data;
  } catch (err) {
    console.error('Error in removeMovieFromDbWatchlist:', err);
    throw err;
  }
}
