import { Router } from 'express';
import { requireUserAuth } from '../middleware/auth.js';
import {
  getUserWatchlist,
  addToWatchlist,
  removeFromWatchlist,
} from '../controllers/userController.js';

const router = Router();

/**
 * @route   GET /api/user/watchlist
 * @desc    Get authenticated user's watchlist from PostgreSQL
 * @access  Protected
 */
router.get('/watchlist', requireUserAuth, getUserWatchlist);

/**
 * @route   POST /api/user/watchlist
 * @desc    Add movie to authenticated user's watchlist
 * @access  Protected
 */
router.post('/watchlist', requireUserAuth, addToWatchlist);

/**
 * @route   DELETE /api/user/watchlist/:movieId
 * @desc    Remove movie from authenticated user's watchlist
 * @access  Protected
 */
router.delete('/watchlist/:movieId', requireUserAuth, removeFromWatchlist);

export default router;
