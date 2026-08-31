# Express + Prisma + PostgreSQL + Clerk Authentication Backend

Dedicated authentication and user profile service for the Movie Recommender application.

## Tech Stack
- **Runtime & Framework**: Node.js + Express.js (ES Modules)
- **Database**: PostgreSQL (Neon Serverless)
- **ORM**: Prisma ORM v6
- **Authentication**: Clerk (`@clerk/express`)
- **Webhooks**: Svix signature validation

---

## Getting Started

### 1. Install Dependencies
```bash
cd server
npm install
```

### 2. Environment Variables
Configure `.env` in the `server/` directory:
```env
# Neon PostgreSQL Connection URL
DATABASE_URL="postgresql://neondb_owner:npg_pdZr95QqnmFC@ep-plain-smoke-ae458tg1-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require"

# Clerk Authentication Keys
CLERK_PUBLISHABLE_KEY="pk_test_..."
CLERK_SECRET_KEY="sk_test_..."

# Clerk Webhook Secret (optional - from Clerk Dashboard)
CLERK_WEBHOOK_SECRET="whsec_..."

# Server Port
PORT=5001
FRONTEND_URL=http://localhost:5173
```

### 3. Generate Prisma Client & Push Schema to Database
```bash
npx prisma generate
npx prisma db push
```

### 4. Run the Server
```bash
# Development (auto-reload with nodemon)
npm run dev

# Production
npm start
```

---

## API Endpoints

### 🩺 Health
- `GET /api/health` - Check API and PostgreSQL database health status.

### 🔐 Authentication (`/api/auth`)
- `POST /api/auth/sync` - Syncs logged in Clerk user with PostgreSQL database. *(Protected)*
- `GET /api/auth/me` - Get profile, preferences, and watchlist count for authenticated user. *(Protected)*
- `PUT /api/auth/preferences` - Update user theme and favorite genres. *(Protected)*

### 🎬 User Watchlist (`/api/user`)
- `GET /api/user/watchlist` - Fetch user's personal movie watchlist from PostgreSQL. *(Protected)*
- `POST /api/user/watchlist` - Add or update a movie in user's watchlist. *(Protected)*
- `DELETE /api/user/watchlist/:movieId` - Remove movie from user's watchlist. *(Protected)*

### 🪝 Webhooks (`/api/webhooks`)
- `POST /api/webhooks/clerk` - Clerk webhook endpoint for `user.created`, `user.updated`, and `user.deleted` events. *(Signature Verified)*
