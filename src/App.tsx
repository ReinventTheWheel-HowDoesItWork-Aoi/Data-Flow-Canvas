/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Spinner } from './components/ui/Spinner';
import { ToastProvider } from './components/ui/Toast';

const LandingPage = lazy(() => import('./pages/LandingPage'));
const EditorPage = lazy(() => import('./pages/EditorPage'));
const ProjectsPage = lazy(() => import('./pages/ProjectsPage'));
const HelpPage = lazy(() => import('./pages/HelpPage'));

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

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <BrowserRouter>
          <Suspense fallback={<LoadingScreen />}>
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route path="/editor" element={<EditorPage />} />
              <Route path="/editor/:projectId" element={<EditorPage />} />
              <Route path="/projects" element={<ProjectsPage />} />
              <Route path="/help" element={<HelpPage />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </ToastProvider>
    </QueryClientProvider>
  );
}

export default App;
