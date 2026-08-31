import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import { clerkMiddleware } from '@clerk/express';

// Ensure standard fallback keys for production & container environments
process.env.DATABASE_URL =
  process.env.DATABASE_URL ||
  'postgresql://neondb_owner:npg_pdZr95QqnmFC@ep-plain-smoke-ae458tg1-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require';
process.env.CLERK_SECRET_KEY =
  process.env.CLERK_SECRET_KEY || 'sk_test_GHWZN4nlArGuvTR18sXGCSyHvbkmftO6OrzZGJlUWQ';
process.env.CLERK_PUBLISHABLE_KEY =
  process.env.CLERK_PUBLISHABLE_KEY ||
  'pk_test_YXBwYXJlbnQtcmF0dGxlci04OTI3LmNsZXJrLmFjY291bnRzLmRldiQ';

import prisma from './config/db.js';
import authRoutes from './routes/authRoutes.js';
import userRoutes from './routes/userRoutes.js';
import webhookRoutes from './routes/webhookRoutes.js';
import { errorHandler, notFoundHandler } from './middleware/errorHandler.js';

const app = express();
const PORT = process.env.PORT || 5001;

// CORS configuration
const allowedOrigins = [
  'http://localhost:5173',
  'http://localhost:3000',
  'http://localhost:5000',
  process.env.FRONTEND_URL,
].filter(Boolean);

app.use(
  cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (like mobile apps, curl, Postman)
      if (!origin) return callback(null, true);
      if (allowedOrigins.indexOf(origin) !== -1 || process.env.NODE_ENV === 'development' || !process.env.FRONTEND_URL) {
        return callback(null, true);
      }
      return callback(new Error('Blocked by CORS policy.'));
    },
    credentials: true,
  })
);

// Logging
app.use(morgan('dev'));

// Webhook raw body preservation & general JSON parser
app.use(
  express.json({
    verify: (req, res, buf) => {
      // Store raw buffer on req for webhook signature validation if needed
      if (req.originalUrl.startsWith('/api/webhooks')) {
        req.rawBody = buf.toString();
      }
    },
  })
);
app.use(express.urlencoded({ extended: true }));

// Global Clerk middleware for session detection
app.use(clerkMiddleware());

// Health Check Endpoint
app.get('/api/health', async (req, res) => {
  try {
    // Quick DB ping test
    await prisma.$queryRaw`SELECT 1`;
    res.status(200).json({
      status: 'healthy',
      service: 'Movie Recommender Auth Backend',
      database: 'connected (PostgreSQL via Prisma)',
      clerkConfigured: Boolean(process.env.CLERK_SECRET_KEY),
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    res.status(500).json({
      status: 'unhealthy',
      database: 'disconnected',
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
});

// API Routes
app.use('/api/auth', authRoutes);
app.use('/api/user', userRoutes);
app.use('/api/webhooks', webhookRoutes);

// Error Handling
app.use(notFoundHandler);
app.use(errorHandler);

// Server Listen
const server = app.listen(PORT, () => {
  console.log(`=============================================`);
  console.log(`🚀 Express Auth Backend running on port ${PORT}`);
  console.log(`📡 Health check: http://localhost:${PORT}/api/health`);
  console.log(`🔐 Clerk Auth & Prisma PostgreSQL connected`);
  console.log(`=============================================`);
});

// Graceful shutdown
const shutdown = async () => {
  console.log('\nGracefully shutting down server...');
  server.close(async () => {
    await prisma.$disconnect();
    console.log('PostgreSQL connection closed.');
    process.exit(0);
  });
};

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

export default app;
