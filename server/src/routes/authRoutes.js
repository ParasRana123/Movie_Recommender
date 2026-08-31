import { Router } from 'express';
import { requireUserAuth } from '../middleware/auth.js';
import { syncUser, getMe, updatePreferences } from '../controllers/authController.js';

const router = Router();

/**
 * @route   POST /api/auth/sync
 * @desc    Sync authenticated Clerk user with PostgreSQL database
 * @access  Protected (Requires Clerk session token)
 */
router.post('/sync', requireUserAuth, syncUser);

/**
 * @route   GET /api/auth/me
 * @desc    Get current user profile and preferences from PostgreSQL
 * @access  Protected (Requires Clerk session token)
 */
router.get('/me', requireUserAuth, getMe);

/**
 * @route   PUT /api/auth/preferences
 * @desc    Update user preferences (theme, favorite genres)
 * @access  Protected (Requires Clerk session token)
 */
router.put('/preferences', requireUserAuth, updatePreferences);

export default router;
