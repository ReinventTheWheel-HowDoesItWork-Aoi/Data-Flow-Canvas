/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { ReactFlowProvider } from '@xyflow/react';
import { TopBar } from '@/components/layout/TopBar';
import { Sidebar } from '@/components/layout/Sidebar';
import { RightPanel } from '@/components/layout/RightPanel';
import { BottomPanel } from '@/components/layout/BottomPanel';
import { Canvas } from '@/components/canvas/Canvas';
import { PyodideLoadingOverlay } from '@/components/PyodideLoadingOverlay';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { TooltipProvider } from '@/components/ui/Tooltip';
import { Spinner } from '@/components/ui/Spinner';
import { PyodideProvider } from '@/lib/pyodide';
import { useUIStore } from '@/stores/uiStore';
import { useAuthStore } from '@/stores/authStore';
import { useProject } from '@/hooks/useProject';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';
import { cn } from '@/lib/utils/cn';
import { AuthModal } from '@/components/auth';

export default function EditorPage() {
  const { projectId } = useParams<{ projectId?: string }>();
  const isBottomPanelOpen = useUIStore((state) => state.isBottomPanelOpen);
  const isDarkMode = useUIStore((state) => state.isDarkMode);
  const setDarkMode = useUIStore((state) => state.setDarkMode);
  const { loadProject, createNewProject, currentProject } = useProject();
  const { user, isInitialized, initialize } = useAuthStore();
  const initialized = useRef(false);
  const authInitialized = useRef(false);

  // Register keyboard shortcuts
  useKeyboardShortcuts();

  // Initialize auth on mount
  useEffect(() => {
    if (authInitialized.current) return;
    authInitialized.current = true;
    initialize();
  }, [initialize]);

  // Initialize dark mode on mount only
  useEffect(() => {
    setDarkMode(isDarkMode);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load project if projectId is provided - only run once
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    if (projectId) {
      loadProject(projectId);
    } else if (!currentProject) {
      createNewProject();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  // Show loading while auth is initializing
  if (!isInitialized) {
    return (
      <div className="min-h-screen bg-bg-primary flex items-center justify-center">
        <div className="text-center">
          <Spinner size="lg" />
          <p className="mt-4 text-text-secondary">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <PyodideProvider>
        <TooltipProvider delayDuration={300}>
          <ReactFlowProvider>
            <div className="h-screen flex flex-col bg-bg-primary">
              <TopBar />

              <div className="flex-1 flex overflow-hidden">
                <Sidebar />

                <main
                  className={cn(
                    'flex-1 flex flex-col overflow-hidden',
                    'bg-bg-primary relative'
                  )}
                >
                  <div className="flex-1 relative">
                    <Canvas />
                    <PyodideLoadingOverlay />
                  </div>
                  {isBottomPanelOpen && <BottomPanel />}
                </main>

                <RightPanel />
              </div>
            </div>

            {/* Auth Modal - shows when user is not logged in */}
            {!user && <AuthModal />}
          </ReactFlowProvider>
        </TooltipProvider>
      </PyodideProvider>
    </ErrorBoundary>
  );
}
