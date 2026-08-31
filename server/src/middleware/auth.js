import { getAuth } from '@clerk/express';
import prisma from '../config/db.js';

/**
 * Safe helper to extract Clerk authentication object without mutating req.auth
 */
export const getClerkAuth = (req) => {
  try {
    if (typeof req.auth === 'function') {
      return req.auth();
    }
    return getAuth(req);
  } catch (err) {
    console.error('Error resolving Clerk auth:', err.message);
    return null;
  }
};

/**
 * Middleware to enforce authentication using Clerk.
 * If user is not authenticated, returns 401 Unauthorized.
 */
export const requireUserAuth = (req, res, next) => {
  const auth = getClerkAuth(req);

  if (!auth || !auth.userId) {
    return res.status(401).json({
      success: false,
      error: 'Unauthorized. Valid Clerk authentication session required.',
    });
  }

  req.clerkUserId = auth.userId;
  req.clerkAuthData = auth;
  next();
};

/**
 * Middleware to attach the database user record to the request.
 * If the user does not exist in PostgreSQL yet, it creates the initial record.
 */
export const attachDbUser = async (req, res, next) => {
  try {
    const auth = getClerkAuth(req);
    const clerkId = req.clerkUserId || auth?.userId;

    if (!clerkId) {
      return res.status(401).json({
        success: false,
        error: 'Unauthorized: No active session found.',
      });
    }

    let user = await prisma.user.findUnique({
      where: { clerkId },
      include: {
        preferences: true,
      },
    });

    // If user is authenticated in Clerk but not yet in PostgreSQL, auto-provision record
    if (!user) {
      const email = req.body?.email || `${clerkId}@clerk.user`;
      user = await prisma.user.create({
        data: {
          clerkId,
          email,
          firstName: req.body?.firstName || null,
          lastName: req.body?.lastName || null,
          imageUrl: req.body?.imageUrl || null,
          preferences: {
            create: {
              theme: 'dark',
              favoriteGenres: [],
            },
          },
        },
        include: {
          preferences: true,
        },
      });
    }

    req.dbUser = user;
    next();
  } catch (error) {
    console.error('Error in attachDbUser middleware:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to authenticate user against database.',
    });
  }
};
