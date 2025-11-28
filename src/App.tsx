/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { Suspense, lazy, useEffect, useRef } from 'react';
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Spinner } from './components/ui/Spinner';
import { ToastProvider } from './components/ui/Toast';
import { useAuthStore } from './stores/authStore';

const LandingPage = lazy(() => import('./pages/LandingPage'));
const EditorPage = lazy(() => import('./pages/EditorPage'));
const ProjectsPage = lazy(() => import('./pages/ProjectsPage'));
const HelpPage = lazy(() => import('./pages/HelpPage'));
const PrivacyPage = lazy(() => import('./pages/PrivacyPage'));
const TermsPage = lazy(() => import('./pages/TermsPage'));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
    },
  },
});

function LoadingScreen() {
  return (
    <div className="min-h-screen bg-deep-navy flex items-center justify-center">
      <div className="text-center">
        <Spinner size="lg" />
        <p className="mt-4 text-slate-400">Loading Data Flow Canvas...</p>
      </div>
    </div>
  );
}

// Component to handle auth initialization and magic link redirects
function AuthHandler({ children }: { children: React.ReactNode }) {
  const { initialize, user, isInitialized } = useAuthStore();
  const navigate = useNavigate();
  const location = useLocation();
  const authInitialized = useRef(false);

  // Initialize auth on mount - this handles magic link callbacks
  useEffect(() => {
    if (authInitialized.current) return;
    authInitialized.current = true;
    initialize();
  }, [initialize]);

  // Redirect to editor after successful magic link sign-in
  useEffect(() => {
    if (isInitialized && user && location.pathname === '/') {
      // User just signed in via magic link and landed on root
      // Redirect them to the editor
      navigate('/editor', { replace: true });
    }
  }, [isInitialized, user, location.pathname, navigate]);

  return <>{children}</>;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <BrowserRouter>
          <AuthHandler>
            <Suspense fallback={<LoadingScreen />}>
              <Routes>
                <Route path="/" element={<LandingPage />} />
                <Route path="/editor" element={<EditorPage />} />
                <Route path="/editor/:projectId" element={<EditorPage />} />
                <Route path="/projects" element={<ProjectsPage />} />
                <Route path="/help" element={<HelpPage />} />
                <Route path="/privacy" element={<PrivacyPage />} />
                <Route path="/terms" element={<TermsPage />} />
              </Routes>
            </Suspense>
          </AuthHandler>
        </BrowserRouter>
      </ToastProvider>
    </QueryClientProvider>
  );
}

export default App;
