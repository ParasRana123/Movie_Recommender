import { Router } from 'express';
import { handleClerkWebhook } from '../controllers/webhookController.js';

const router = Router();

/**
 * @route   POST /api/webhooks/clerk
 * @desc    Clerk webhook event endpoint for user sync
 * @access  Public (Signature verified via Svix)
 */
router.post('/clerk', handleClerkWebhook);

export default router;
