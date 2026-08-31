import React from 'react';
import ReactDOM from 'react-dom/client';
import { ClerkProvider } from '@clerk/clerk-react';
import App from './App.jsx';

// Default to the provided Clerk Publishable Key or environment variable
const PUBLISHABLE_KEY =
  import.meta.env.VITE_CLERK_PUBLISHABLE_KEY ||
  'pk_test_YXBwYXJlbnQtcmF0dGxlci04OTI3LmNsZXJrLmFjY291bnRzLmRldiQ';

const RootComponent = () => {
  if (PUBLISHABLE_KEY && PUBLISHABLE_KEY.startsWith('pk_')) {
    return (
      <ClerkProvider publishableKey={PUBLISHABLE_KEY} afterSignOutUrl="/">
        <App />
      </ClerkProvider>
    );
  }

  // Graceful fallback if no valid Clerk key is provided
  return <App />;
};

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <RootComponent />
  </React.StrictMode>
);
