import prisma from '../config/db.js';
import { getClerkAuth } from '../middleware/auth.js';

/**
 * Syncs the authenticated Clerk user with PostgreSQL database.
 * Upserts the user record to keep profile data synchronized.
 */
export const syncUser = async (req, res, next) => {
  try {
    const auth = getClerkAuth(req);
    const clerkId = req.clerkUserId || auth?.userId;

    if (!clerkId) {
      return res.status(401).json({
        success: false,
        error: 'Unauthorized: Missing Clerk session token.',
      });
    }

    const { email, firstName, lastName, username, imageUrl } = req.body;
    const userEmail = email || req.body?.emailAddress || `${clerkId}@clerk.user`;

    // Upsert user in Postgres
    const user = await prisma.user.upsert({
      where: { clerkId },
      update: {
        email: userEmail,
        firstName: firstName || null,
        lastName: lastName || null,
        username: username || null,
        imageUrl: imageUrl || null,
      },
      create: {
        clerkId,
        email: userEmail,
        firstName: firstName || null,
        lastName: lastName || null,
        username: username || null,
        imageUrl: imageUrl || null,
        preferences: {
          create: {
            theme: 'dark',
            favoriteGenres: [],
          },
        },
      },
      include: {
        preferences: true,
        _count: {
          select: { watchlist: true },
        },
      },
    });

    console.log(`[Auth Sync] ✅ Synced user ${clerkId} (${userEmail}) into PostgreSQL database.`);

    return res.status(200).json({
      success: true,
      message: 'User synchronized successfully with PostgreSQL database.',
      data: user,
    });
  } catch (error) {
    console.error('Error syncing user in database:', error);
    next(error);
  }
};

/**
 * Retrieves the currently authenticated user's profile and preferences.
 */
export const getMe = async (req, res, next) => {
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
        preferences: true,
        watchlist: {
          orderBy: { addedAt: 'desc' },
          take: 20,
        },
        _count: {
          select: { watchlist: true },
        },
      },
    });

    if (!user) {
      return res.status(404).json({
        success: false,
        error: 'User not found in database. Please call /api/auth/sync first.',
      });
    }

    return res.status(200).json({
      success: true,
      data: user,
    });
  } catch (error) {
    console.error('Error getting current user:', error);
    next(error);
  }
};

/**
 * Updates user preferences (theme, favorite genres).
 */
export const updatePreferences = async (req, res, next) => {
  try {
    const auth = getClerkAuth(req);
    const clerkId = req.clerkUserId || auth?.userId;

    if (!clerkId) {
      return res.status(401).json({
        success: false,
        error: 'Unauthorized.',
      });
    }

    const { theme, favoriteGenres } = req.body;

    const user = await prisma.user.findUnique({
      where: { clerkId },
    });

    if (!user) {
      return res.status(404).json({
        success: false,
        error: 'User not found.',
      });
    }

    const updatedPreference = await prisma.userPreference.upsert({
      where: { userId: user.id },
      update: {
        ...(theme ? { theme } : {}),
        ...(Array.isArray(favoriteGenres) ? { favoriteGenres } : {}),
      },
      create: {
        userId: user.id,
        theme: theme || 'dark',
        favoriteGenres: Array.isArray(favoriteGenres) ? favoriteGenres : [],
      },
    });

    return res.status(200).json({
      success: true,
      message: 'Preferences updated successfully.',
      data: updatedPreference,
    });
  } catch (error) {
    console.error('Error updating preferences:', error);
    next(error);
  }
};
