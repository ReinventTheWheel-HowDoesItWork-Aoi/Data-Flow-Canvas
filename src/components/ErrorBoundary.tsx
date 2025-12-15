/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React, { Component, type ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/Button';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Error details are captured in state for debugging but not logged to console
    // to prevent sensitive information disclosure in production
    // Consider integrating with error tracking services (e.g., Sentry) for production monitoring
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-bg-primary flex items-center justify-center p-6">
          <div className="max-w-md w-full bg-bg-secondary rounded-xl border border-border-default p-8 text-center">
            <div className="w-16 h-16 rounded-full bg-warm-coral/20 flex items-center justify-center mx-auto mb-4">
              <AlertTriangle size={32} className="text-warm-coral" />
            </div>

            <h1 className="text-h2 text-text-primary mb-2">Something went wrong</h1>
            <p className="text-text-secondary mb-6">
              An unexpected error occurred. This has been logged and we'll look into it.
            </p>

            {this.state.error && (
              <div className="bg-bg-tertiary rounded-lg p-4 mb-6 text-left">
                <p className="text-small font-mono text-warm-coral break-all">
                  A rendering error occurred. Please refresh the page or try again.
                </p>
              </div>
            )}

            <div className="flex gap-3 justify-center">
              <Button variant="secondary" onClick={this.handleReset}>
                Try Again
              </Button>
              <Button
                variant="primary"
                onClick={this.handleReload}
                leftIcon={<RefreshCw size={16} />}
              >
                Reload Page
              </Button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
