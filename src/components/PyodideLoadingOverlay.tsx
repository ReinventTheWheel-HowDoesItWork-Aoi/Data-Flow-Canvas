/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { usePyodide } from '@/lib/pyodide';
import { cn } from '@/lib/utils/cn';

export function PyodideLoadingOverlay() {
  const { isLoading, isReady, error, loadProgress } = usePyodide();

  // Only show when actively loading (not before initialization starts)
  if (!isLoading && !error) {
    return null;
  }

  return (
    <AnimatePresence>
      {(isLoading || error) && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className={cn(
            'absolute inset-0 z-50',
            'bg-bg-primary/80 backdrop-blur-sm',
            'flex items-center justify-center'
          )}
        >
          <div className="bg-bg-secondary rounded-xl border border-border-default p-8 max-w-sm w-full mx-4 shadow-lg">
            <div className="text-center">
              {error ? (
                <>
                  <div className="w-12 h-12 rounded-full bg-warm-coral/20 flex items-center justify-center mx-auto mb-4">
                    <AlertCircle size={24} className="text-warm-coral" />
                  </div>
                  <h3 className="text-h3 text-text-primary mb-2">
                    Failed to Load Python Engine
                  </h3>
                  <p className="text-small text-text-secondary mb-4">{error}</p>
                  <button
                    onClick={() => window.location.reload()}
                    className="text-electric-indigo hover:underline text-small"
                  >
                    Reload page to try again
                  </button>
                </>
              ) : isReady ? (
                <>
                  <div className="w-12 h-12 rounded-full bg-fresh-teal/20 flex items-center justify-center mx-auto mb-4">
                    <CheckCircle size={24} className="text-fresh-teal" />
                  </div>
                  <h3 className="text-h3 text-text-primary">Python Ready</h3>
                </>
              ) : (
                <>
                  <div className="w-12 h-12 rounded-full bg-electric-indigo/20 flex items-center justify-center mx-auto mb-4">
                    <Loader2 size={24} className="text-electric-indigo animate-spin" />
                  </div>
                  <h3 className="text-h3 text-text-primary mb-2">
                    Loading Python Engine
                  </h3>
                  <p className="text-small text-text-muted mb-4">
                    {getLoadingMessage(loadProgress)}
                  </p>

                  {/* Progress bar */}
                  <div className="h-2 bg-bg-tertiary rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-electric-indigo"
                      initial={{ width: 0 }}
                      animate={{ width: `${loadProgress}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                  <p className="text-small text-text-muted mt-2">
                    {Math.round(loadProgress)}%
                  </p>
                </>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function getLoadingMessage(progress: number): string {
  if (progress < 30) {
    return 'Downloading Pyodide WebAssembly...';
  } else if (progress < 60) {
    return 'Initializing Python runtime...';
  } else if (progress < 80) {
    return 'Loading pandas & numpy...';
  } else if (progress < 95) {
    return 'Loading scikit-learn...';
  } else {
    return 'Almost ready...';
  }
}
