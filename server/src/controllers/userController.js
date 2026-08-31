import prisma from '../config/db.js';
import { getClerkAuth } from '../middleware/auth.js';

/**
 * Get the authenticated user's watchlist from PostgreSQL
 */
export const getUserWatchlist = async (req, res, next) => {
  try {
    const auth = getClerkAuth(req);
    const clerkId = req.clerkUserId || auth?.userId;

    if (!clerkId) {
      return res.status(401).json({
        success: false,
        error: 'Unauthorized: Missing Clerk session token.',
      });
    }

    const user = await prisma.user.findUnique({
      where: { clerkId },
      include: {
        watchlist: {
          orderBy: { addedAt: 'desc' },
        },
      },
    });

    if (!user) {
      return res.status(200).json({
        success: true,
        data: [],
      });
    }

    return res.status(200).json({
      success: true,
      data: user.watchlist,
    });
  } catch (error) {
    console.error('Error fetching watchlist:', error);
    next(error);
  }
};

/**
 * Add a movie to the user's watchlist in PostgreSQL
 */
export const addToWatchlist = async (req, res, next) => {
  try {
    const auth = getClerkAuth(req);
    const clerkId = req.clerkUserId || auth?.userId;

    if (!clerkId) {
      return res.status(401).json({
        success: false,
        error: 'Unauthorized: Missing Clerk session token.',
      });
    }

    const body = req.body || {};
    const movieTitle = body.movieTitle || body.title || body.name;
    const movieId = String(body.movieId || body.id || movieTitle || '');
    const posterPath = body.posterPath || body.poster || body.poster_path || null;
    const releaseYear = body.releaseYear || body.release_date || body.date || body.year || null;
    const voteAverageRaw = body.voteAverage !== undefined ? body.voteAverage : (body.rating !== undefined ? body.rating : body.vote_average);
    const voteAverage = voteAverageRaw !== undefined && !isNaN(Number(voteAverageRaw)) ? Number(voteAverageRaw) : null;
    
    let genres = [];
    if (Array.isArray(body.genres)) {
      genres = body.genres;
    } else if (typeof body.genres === 'string') {
      genres = body.genres.split(',').map((g) => g.trim()).filter(Boolean);
    }

    if (!movieTitle) {
      return res.status(400).json({
        success: false,
        error: 'Movie title is required.',
      });
    }

    // Ensure user exists in database
    let user = await prisma.user.findUnique({
      where: { clerkId },
    });

    if (!user) {
      user = await prisma.user.create({
        data: {
          clerkId,
          email: `${clerkId}@clerk.user`,
          preferences: {
            create: {
              theme: 'dark',
              favoriteGenres: [],
            },
          },
        },
      });
    }

    const item = await prisma.watchlistItem.upsert({
      where: {
        userId_movieId: {
          userId: user.id,
          movieId,
        },
      },
      update: {
        movieTitle,
        posterPath: posterPath ? String(posterPath) : null,
        releaseYear: releaseYear ? String(releaseYear) : null,
        voteAverage,
        genres,
      },
      create: {
        userId: user.id,
        movieId,
        movieTitle,
        posterPath: posterPath ? String(posterPath) : null,
        releaseYear: releaseYear ? String(releaseYear) : null,
        voteAverage,
        genres,
      },
    });

    console.log(`[Watchlist] ✅ Saved "${movieTitle}" for user ${clerkId} in PostgreSQL.`);

    return res.status(201).json({
      success: true,
      message: 'Movie added to database watchlist.',
      data: item,
    });
  } catch (error) {
    console.error('Error adding to watchlist:', error);
    next(error);
  }
};

/**
 * Remove a movie from the user's watchlist in PostgreSQL
 */
export const removeFromWatchlist = async (req, res, next) => {
  try {
    const auth = getClerkAuth(req);
    const clerkId = req.clerkUserId || auth?.userId;

    if (!clerkId) {
      return res.status(401).json({
        success: false,
        error: 'Unauthorized: Missing Clerk session token.',
      });
    }

    const { movieId } = req.params;

    const user = await prisma.user.findUnique({
      where: { clerkId },
    });

    if (!user) {
      return res.status(200).json({
        success: true,
        message: 'No watchlist to remove from.',
      });
    }

    const decodedParam = decodeURIComponent(movieId);

    await prisma.watchlistItem.deleteMany({
      where: {
        userId: user.id,
        OR: [
          { movieId: decodedParam },
          { movieTitle: { equals: decodedParam, mode: 'insensitive' } },
        ],
      },
    });

    console.log(`[Watchlist] 🗑️ Removed "${decodedParam}" for user ${clerkId} from PostgreSQL.`);

    return res.status(200).json({
      success: true,
      message: 'Movie removed from database watchlist.',
    });
  } catch (error) {
    console.error('Error removing from watchlist:', error);
    next(error);
  }
};
