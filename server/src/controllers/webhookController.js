import { Webhook } from 'svix';
import prisma from '../config/db.js';

/**
 * Handles Clerk Webhook events (user.created, user.updated, user.deleted)
 */
export const handleClerkWebhook = async (req, res, next) => {
  const WEBHOOK_SECRET = process.env.CLERK_WEBHOOK_SECRET;

  if (!WEBHOOK_SECRET) {
    console.warn('CLERK_WEBHOOK_SECRET is not set. Webhook verification skipped (dev mode).');
  }

  const svixId = req.headers['svix-id'];
  const svixTimestamp = req.headers['svix-timestamp'];
  const svixSignature = req.headers['svix-signature'];

  let evt;

  // Verify webhook signature if secret is present
  if (WEBHOOK_SECRET) {
    if (!svixId || !svixTimestamp || !svixSignature) {
      return res.status(400).json({
        success: false,
        error: 'Missing Svix verification headers.',
      });
    }

    // Clerk webhooks require raw body or stringified payload for signature verification
    const payload = typeof req.body === 'string' ? req.body : JSON.stringify(req.body);
    const wh = new Webhook(WEBHOOK_SECRET);

    try {
      evt = wh.verify(payload, {
        'svix-id': svixId,
        'svix-timestamp': svixTimestamp,
        'svix-signature': svixSignature,
      });
    } catch (err) {
      console.error('Error verifying Clerk webhook signature:', err.message);
      return res.status(400).json({
        success: false,
        error: 'Invalid webhook signature.',
      });
    }
  } else {
    evt = req.body;
  }

  const { type, data } = evt;
  console.log(`[Clerk Webhook Received]: ${type}`);

  try {
    switch (type) {
      case 'user.created':
      case 'user.updated': {
        const clerkId = data.id;
        const primaryEmailId = data.primary_email_address_id;
        const emailObj = data.email_addresses?.find((e) => e.id === primaryEmailId) || data.email_addresses?.[0];
        const email = emailObj?.email_address || `${clerkId}@clerk.user`;
        const firstName = data.first_name || null;
        const lastName = data.last_name || null;
        const username = data.username || null;
        const imageUrl = data.image_url || null;

        await prisma.user.upsert({
          where: { clerkId },
          update: {
            email,
            firstName,
            lastName,
            username,
            imageUrl,
          },
          create: {
            clerkId,
            email,
            firstName,
            lastName,
            username,
            imageUrl,
            preferences: {
              create: {
                theme: 'dark',
                favoriteGenres: [],
              },
            },
          },
        });
        console.log(`[Clerk Webhook]: User ${clerkId} (${email}) upserted in PostgreSQL.`);
        break;
      }

      case 'user.deleted': {
        const clerkId = data.id;
        if (clerkId) {
          await prisma.user.deleteMany({
            where: { clerkId },
          });
          console.log(`[Clerk Webhook]: User ${clerkId} removed from PostgreSQL.`);
        }
        break;
      }

      default:
        console.log(`[Clerk Webhook]: Unhandled event type ${type}`);
    }

    return res.status(200).json({
      success: true,
      message: 'Webhook processed successfully.',
    });
  } catch (error) {
    console.error('Error processing Clerk webhook in database:', error);
    next(error);
  }
};
