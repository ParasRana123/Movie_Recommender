import prisma from '../config/db.js';
import { getAuth } from '@clerk/express';

/**
 * Get the authenticated user's watchlist from PostgreSQL
 */
export const getUserWatchlist = async (req, res, next) => {
  try {
    const auth = getAuth(req);
    const clerkId = auth?.userId;

    const user = await prisma.user.findUnique({
      where: { clerkId },
      include: {
        watchlist: {
          orderBy: { addedAt: 'desc' },
        },
      },
    });

    if (!user) {
      return res.status(404).json({
        success: false,
        error: 'User not found in database.',
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
    const auth = getAuth(req);
    const clerkId = auth?.userId;

    const { movieId, movieTitle, posterPath, releaseYear, voteAverage, genres } = req.body;

    if (!movieId || !movieTitle) {
      return res.status(400).json({
        success: false,
        error: 'movieId and movieTitle are required.',
      });
    }

    const user = await prisma.user.findUnique({
      where: { clerkId },
    });

    if (!user) {
      return res.status(404).json({
        success: false,
        error: 'User not found in database. Please sync first.',
      });
    }

    const item = await prisma.watchlistItem.upsert({
      where: {
        userId_movieId: {
          userId: user.id,
          movieId: String(movieId),
        },
      },
      update: {
        movieTitle,
        posterPath: posterPath || null,
        releaseYear: releaseYear ? String(releaseYear) : null,
        voteAverage: voteAverage !== undefined ? Number(voteAverage) : null,
        genres: Array.isArray(genres) ? genres : [],
      },
      create: {
        userId: user.id,
        movieId: String(movieId),
        movieTitle,
        posterPath: posterPath || null,
        releaseYear: releaseYear ? String(releaseYear) : null,
        voteAverage: voteAverage !== undefined ? Number(voteAverage) : null,
        genres: Array.isArray(genres) ? genres : [],
      },
    });

    return res.status(201).json({
      success: true,
      message: 'Movie added to watchlist.',
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
    const auth = getAuth(req);
    const clerkId = auth?.userId;
    const { movieId } = req.params;

    const user = await prisma.user.findUnique({
      where: { clerkId },
    });

    if (!user) {
      return res.status(404).json({
        success: false,
        error: 'User not found in database.',
      });
    }

    await prisma.watchlistItem.deleteMany({
      where: {
        userId: user.id,
        movieId: String(movieId),
      },
    });

    return res.status(200).json({
      success: true,
      message: 'Movie removed from watchlist.',
    });
  } catch (error) {
    console.error('Error removing from watchlist:', error);
    next(error);
  }
};
