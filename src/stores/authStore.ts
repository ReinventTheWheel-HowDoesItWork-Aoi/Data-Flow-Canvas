/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { create } from 'zustand';
import { supabase } from '@/lib/supabase';
import type { User, Session } from '@supabase/supabase-js';

export interface UserProfile {
  firstName: string;
  lastName: string;
  email: string;
  company?: string;
}

interface AuthState {
  // State
  user: User | null;
  session: Session | null;
  profile: UserProfile | null;
  isLoading: boolean;
  isInitialized: boolean;
  needsProfileCompletion: boolean;
  error: string | null;

  // Actions
  initialize: () => Promise<void>;
  signInWithMagicLink: (
    email: string
  ) => Promise<{ success: boolean; error?: string }>;
  signUpWithMagicLink: (
    email: string,
    firstName: string,
    lastName: string,
    company?: string
  ) => Promise<{ success: boolean; error?: string }>;
  completeProfile: (
    firstName: string,
    lastName: string,
    company?: string
  ) => Promise<{ success: boolean; error?: string }>;
  signOut: () => Promise<void>;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  // Initial state
  user: null,
  session: null,
  profile: null,
  isLoading: false,
  isInitialized: false,
  needsProfileCompletion: false,
  error: null,

  // Initialize auth state and listen for changes
  initialize: async () => {
    try {
      // Set up auth state change listener FIRST
      // This is critical for handling magic link callbacks
      const { data: { subscription } } = supabase.auth.onAuthStateChange(
        async (event, session) => {
          console.log('Auth state changed:', event);

          if (session?.user) {
            const profile = extractProfile(session.user);
            const needsProfile = !profile.firstName || !profile.lastName;

            // Sync profile to database when user signs in or signs up
            if (event === 'SIGNED_IN' || event === 'USER_UPDATED') {
              await syncProfileToDatabase(session.user);
            }

            set({
              user: session.user,
              session,
              profile,
              needsProfileCompletion: needsProfile,
              isInitialized: true,
              isLoading: false,
            });
          } else {
            set({
              user: null,
              session: null,
              profile: null,
              needsProfileCompletion: false,
              isInitialized: true,
              isLoading: false,
            });
          }
        }
      );

      // Then get initial session (for page refreshes when already logged in)
      const { data: { session }, error } = await supabase.auth.getSession();

      if (error) {
        console.error('Error getting session:', error);
        set({ isInitialized: true, isLoading: false });
        return;
      }

      if (session?.user) {
        const profile = extractProfile(session.user);
        const needsProfile = !profile.firstName || !profile.lastName;
        set({
          user: session.user,
          session,
          profile,
          needsProfileCompletion: needsProfile,
          isInitialized: true,
          isLoading: false,
        });
      } else {
        set({ isInitialized: true, isLoading: false });
      }
    } catch (error) {
      console.error('Error initializing auth:', error);
      set({ isInitialized: true, isLoading: false });
    }
  },

  // Sign in with magic link (passwordless) - for existing users
  signInWithMagicLink: async (email) => {
    set({ isLoading: true, error: null });

    try {
      const { error } = await supabase.auth.signInWithOtp({
        email,
        options: {
          emailRedirectTo: `${window.location.origin}/editor`,
        },
      });

      if (error) {
        set({ isLoading: false, error: error.message });
        return { success: false, error: error.message };
      }

      set({ isLoading: false });
      return { success: true };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to send magic link';
      set({ isLoading: false, error: message });
      return { success: false, error: message };
    }
  },

  // Sign up with magic link (passwordless) - for new users with profile data
  signUpWithMagicLink: async (email, firstName, lastName, company) => {
    set({ isLoading: true, error: null });

    try {
      const { error } = await supabase.auth.signInWithOtp({
        email,
        options: {
          emailRedirectTo: `${window.location.origin}/editor`,
          data: {
            first_name: firstName,
            last_name: lastName,
            company: company || null,
          },
        },
      });

      if (error) {
        set({ isLoading: false, error: error.message });
        return { success: false, error: error.message };
      }

      set({ isLoading: false });
      return { success: true };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to send magic link';
      set({ isLoading: false, error: message });
      return { success: false, error: message };
    }
  },

  // Complete profile after first magic link sign-in
  completeProfile: async (firstName, lastName, company) => {
    set({ isLoading: true, error: null });

    try {
      const { error } = await supabase.auth.updateUser({
        data: {
          first_name: firstName,
          last_name: lastName,
          company: company || null,
        },
      });

      if (error) {
        set({ isLoading: false, error: error.message });
        return { success: false, error: error.message };
      }

      // Update local profile state and sync to database
      const { user } = get();
      if (user) {
        // Sync completed profile to database
        await syncProfileToDatabase({
          ...user,
          user_metadata: {
            ...user.user_metadata,
            first_name: firstName,
            last_name: lastName,
            company: company || null,
          },
        });

        set({
          profile: {
            firstName,
            lastName,
            email: user.email || '',
            company: company || undefined,
          },
          needsProfileCompletion: false,
          isLoading: false,
        });
      } else {
        set({ isLoading: false });
      }

      return { success: true };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to update profile';
      set({ isLoading: false, error: message });
      return { success: false, error: message };
    }
  },

  // Sign out
  signOut: async () => {
    set({ isLoading: true });

    try {
      await supabase.auth.signOut();
      set({
        user: null,
        session: null,
        profile: null,
        needsProfileCompletion: false,
        isLoading: false,
      });
    } catch (error) {
      console.error('Error signing out:', error);
      set({ isLoading: false });
    }
  },

  // Clear error
  clearError: () => set({ error: null }),
}));

// Helper function to extract profile from user metadata
function extractProfile(user: User): UserProfile {
  const metadata = user.user_metadata || {};
  return {
    firstName: metadata.first_name || '',
    lastName: metadata.last_name || '',
    email: user.email || '',
    company: metadata.company || undefined,
  };
}

// Sync user profile to the profiles table in the database
async function syncProfileToDatabase(user: User): Promise<void> {
  const profile = extractProfile(user);

  // Only sync if we have the required profile data
  if (!profile.firstName || !profile.lastName) {
    return;
  }

  try {
    const { error } = await supabase
      .from('profiles')
      .upsert({
        id: user.id,
        first_name: profile.firstName,
        last_name: profile.lastName,
        email: profile.email,
        company: profile.company || null,
        updated_at: new Date().toISOString(),
      }, {
        onConflict: 'id',
      });

    if (error) {
      console.error('Error syncing profile to database:', error);
    }
  } catch (error) {
    console.error('Error syncing profile to database:', error);
  }
}
