import React, { createContext, useContext, useEffect, useState } from 'react';
import { useUser, useAuth } from '@clerk/clerk-react';
import { syncUserWithBackend, fetchCurrentUser } from '../api/authApi';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const { user, isLoaded: isUserLoaded, isSignedIn } = useUser();
  const { getToken } = useAuth();
  const [dbUser, setDbUser] = useState(null);
  const [isSyncing, setIsSyncing] = useState(false);

  useEffect(() => {
    let mounted = true;

    async function sync() {
      if (isSignedIn && user) {
        setIsSyncing(true);
        try {
          const syncResult = await syncUserWithBackend(getToken, user);
          if (mounted && syncResult?.data) {
            setDbUser(syncResult.data);
          } else {
            // Fallback: try fetching me
            const meResult = await fetchCurrentUser(getToken);
            if (mounted && meResult?.data) {
              setDbUser(meResult.data);
            }
          }
        } catch (err) {
          console.warn('Auth sync notice:', err.message);
        } finally {
          if (mounted) setIsSyncing(false);
        }
      } else if (!isSignedIn) {
        setDbUser(null);
      }
    }

    if (isUserLoaded) {
      sync();
    }

    return () => {
      mounted = false;
    };
  }, [isSignedIn, user, isUserLoaded, getToken]);

  return (
    <AuthContext.Provider
      value={{
        user,
        dbUser,
        isSignedIn,
        isUserLoaded,
        isSyncing,
        getToken,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAppAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAppAuth must be used within an AuthProvider');
  }
  return context;
}
