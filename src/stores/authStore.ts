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
  error: string | null;

  // Actions
  initialize: () => Promise<void>;
  signUp: (
    email: string,
    password: string,
    firstName: string,
    lastName: string,
    company?: string
  ) => Promise<{ success: boolean; error?: string }>;
  signIn: (
    email: string,
    password: string
  ) => Promise<{ success: boolean; error?: string }>;
  signOut: () => Promise<void>;
  resetPassword: (email: string) => Promise<{ success: boolean; error?: string }>;
  updatePassword: (newPassword: string) => Promise<{ success: boolean; error?: string }>;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  // Initial state
  user: null,
  session: null,
  profile: null,
  isLoading: false,
  isInitialized: false,
  error: null,

  // Initialize auth state and listen for changes
  initialize: async () => {
    try {
      // Get initial session
      const { data: { session }, error } = await supabase.auth.getSession();

      if (error) {
        console.error('Error getting session:', error);
        set({ isInitialized: true, isLoading: false });
        return;
      }

      if (session?.user) {
        const profile = extractProfile(session.user);
        set({
          user: session.user,
          session,
          profile,
          isInitialized: true,
          isLoading: false,
        });
      } else {
        set({ isInitialized: true, isLoading: false });
      }

      // Listen for auth state changes
      supabase.auth.onAuthStateChange((_event, session) => {
        if (session?.user) {
          const profile = extractProfile(session.user);
          set({
            user: session.user,
            session,
            profile,
          });
        } else {
          set({
            user: null,
            session: null,
            profile: null,
          });
        }
      });
    } catch (error) {
      console.error('Error initializing auth:', error);
      set({ isInitialized: true, isLoading: false });
    }
  },

  // Sign up with email and password
  signUp: async (email, password, firstName, lastName, company) => {
    set({ isLoading: true, error: null });

    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
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

      if (data.user) {
        const profile = extractProfile(data.user);
        set({
          user: data.user,
          session: data.session,
          profile,
          isLoading: false,
        });
        return { success: true };
      }

      set({ isLoading: false });
      return { success: true };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Sign up failed';
      set({ isLoading: false, error: message });
      return { success: false, error: message };
    }
  },

  // Sign in with email and password
  signIn: async (email, password) => {
    set({ isLoading: true, error: null });

    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) {
        set({ isLoading: false, error: error.message });
        return { success: false, error: error.message };
      }

      if (data.user) {
        const profile = extractProfile(data.user);
        set({
          user: data.user,
          session: data.session,
          profile,
          isLoading: false,
        });
        return { success: true };
      }

      set({ isLoading: false });
      return { success: false, error: 'Sign in failed' };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Sign in failed';
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
        isLoading: false,
      });
    } catch (error) {
      console.error('Error signing out:', error);
      set({ isLoading: false });
    }
  },

  // Reset password - sends email with reset link
  resetPassword: async (email: string) => {
    set({ isLoading: true, error: null });

    try {
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/reset-password`,
      });

      if (error) {
        set({ isLoading: false, error: error.message });
        return { success: false, error: error.message };
      }

      set({ isLoading: false });
      return { success: true };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to send reset email';
      set({ isLoading: false, error: message });
      return { success: false, error: message };
    }
  },

  // Update password - called after user clicks reset link
  updatePassword: async (newPassword: string) => {
    set({ isLoading: true, error: null });

    try {
      const { error } = await supabase.auth.updateUser({ password: newPassword });

      if (error) {
        set({ isLoading: false, error: error.message });
        return { success: false, error: error.message };
      }

      set({ isLoading: false });
      return { success: true };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to update password';
      set({ isLoading: false, error: message });
      return { success: false, error: message };
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
