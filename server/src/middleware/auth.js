import { getAuth } from '@clerk/express';
import prisma from '../config/db.js';

/**
 * Middleware to enforce authentication using Clerk.
 * If user is not authenticated, returns 401 Unauthorized.
 */
export const requireUserAuth = (req, res, next) => {
  const auth = getAuth(req);

  if (!auth || !auth.userId) {
    return res.status(401).json({
      success: false,
      error: 'Unauthorized. Valid Clerk authentication session required.',
    });
  }

  req.auth = auth;
  next();
};

/**
 * Middleware to attach the database user record to the request.
 * If the user does not exist in PostgreSQL yet, it creates the initial record.
 */
export const attachDbUser = async (req, res, next) => {
  try {
    const auth = getAuth(req);
    if (!auth || !auth.userId) {
      return res.status(401).json({
        success: false,
        error: 'Unauthorized: No active session found.',
      });
    }

    const clerkId = auth.userId;
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
