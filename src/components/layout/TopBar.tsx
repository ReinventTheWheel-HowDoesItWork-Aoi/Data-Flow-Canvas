/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React from 'react';
import { Link } from 'react-router-dom';
import {
  Play,
  Square,
  Save,
  Users,
  Settings,
  Moon,
  Sun,
  ChevronLeft,
  Share2,
  Loader2,
  Terminal,
  HelpCircle,
  Workflow,
  LogOut,
  User,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from '@/components/ui/Tooltip';
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from '@/components/ui/Dropdown';
import { useUIStore } from '@/stores/uiStore';
import { useProjectStore } from '@/stores/projectStore';
import { useAuthStore } from '@/stores/authStore';
import { useExecution } from '@/hooks/useExecution';
import { useProject } from '@/hooks/useProject';

export function TopBar() {
  const {
    isDarkMode,
    toggleDarkMode,
    openCollaborationModal,
    openSettingsModal,
    toggleBottomPanel,
    isBottomPanelOpen,
  } = useUIStore();
  const { currentProject, isSaving, hasUnsavedChanges } = useProjectStore();
  const { profile, signOut } = useAuthStore();
  const { saveCurrentProject } = useProject();
  const {
    runPipeline,
    cancelExecution,
    isRunning,
    isPyodideReady,
    isPyodideLoading,
  } = useExecution();

  const handleSignOut = async () => {
    await signOut();
  };

  const handleRun = () => {
    if (isRunning) {
      cancelExecution();
    } else {
      runPipeline('run-all');
    }
  };

  const handleSave = async () => {
    await saveCurrentProject();
  };

  return (
    <header className="h-14 bg-bg-secondary border-b border-border-default flex items-center justify-between px-4">
      {/* Left section */}
      <div className="flex items-center gap-4">
        <Link
          to="/projects"
          className="flex items-center gap-2 text-text-muted hover:text-text-primary transition-colors"
        >
          <ChevronLeft size={20} />
        </Link>

        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-electric-indigo to-soft-violet flex items-center justify-center">
            <Workflow size={18} className="text-white" />
          </div>
          <div>
            <h1 className="text-body font-semibold text-text-primary">
              Data Flow Canvas
            </h1>
            {hasUnsavedChanges && (
              <span className="text-small text-text-muted">Unsaved changes</span>
            )}
          </div>
        </div>
      </div>

      {/* Center section - Run controls */}
      <div className="flex items-center gap-2">
        {isPyodideLoading && (
          <div className="flex items-center gap-2 text-text-muted text-small mr-2">
            <Loader2 size={14} className="animate-spin" />
            <span>Loading Python...</span>
          </div>
        )}

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={isRunning ? 'danger' : 'primary'}
              size="sm"
              onClick={handleRun}
              disabled={!isPyodideReady && !isRunning}
              leftIcon={
                isRunning ? (
                  <Square size={16} />
                ) : isPyodideLoading ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Play size={16} />
                )
              }
            >
              {isRunning ? 'Stop' : 'Run'}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            {isPyodideReady
              ? 'Run pipeline (Ctrl+Enter)'
              : 'Waiting for Python engine to load...'}
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={isBottomPanelOpen ? 'secondary' : 'ghost'}
              size="sm"
              onClick={toggleBottomPanel}
            >
              <Terminal size={18} />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Toggle logs (Ctrl+J)</TooltipContent>
        </Tooltip>
      </div>

      {/* Right section */}
      <div className="flex items-center gap-2">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleSave}
              isLoading={isSaving}
            >
              <Save size={18} />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Save project (Ctrl+S)</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="sm" onClick={openCollaborationModal}>
              <Users size={18} />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Collaborate</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="sm" onClick={() => {}}>
              <Share2 size={18} />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Share</TooltipContent>
        </Tooltip>

        <div className="w-px h-6 bg-border-default mx-1" />

        <Tooltip>
          <TooltipTrigger asChild>
            <Link to="/help">
              <Button variant="ghost" size="sm">
                <HelpCircle size={18} />
              </Button>
            </Link>
          </TooltipTrigger>
          <TooltipContent>Help & Documentation</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="sm" onClick={toggleDarkMode}>
              {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            {isDarkMode ? 'Light mode' : 'Dark mode'}
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="sm" onClick={openSettingsModal}>
              <Settings size={18} />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Settings</TooltipContent>
        </Tooltip>

        {/* User Menu */}
        {profile && (
          <>
            <div className="w-px h-6 bg-border-default mx-1" />
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="flex items-center gap-2 px-2 py-1.5 rounded-lg text-text-secondary hover:text-text-primary hover:bg-bg-tertiary transition-colors">
                  <div className="w-7 h-7 rounded-full bg-gradient-to-br from-electric-indigo to-soft-violet flex items-center justify-center">
                    <span className="text-xs font-medium text-white">
                      {profile.firstName.charAt(0).toUpperCase()}
                      {profile.lastName.charAt(0).toUpperCase()}
                    </span>
                  </div>
                  <span className="text-small font-medium hidden sm:inline">
                    {profile.firstName}
                  </span>
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <div className="px-3 py-2 border-b border-border-default">
                  <p className="text-small font-medium text-text-primary">
                    {profile.firstName} {profile.lastName}
                  </p>
                  <p className="text-small text-text-muted truncate">
                    {profile.email}
                  </p>
                  {profile.company && (
                    <p className="text-small text-text-muted">
                      {profile.company}
                    </p>
                  )}
                </div>
                <DropdownMenuItem onClick={handleSignOut} className="text-warm-coral">
                  <LogOut size={16} className="mr-2" />
                  Sign out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </>
        )}
      </div>
    </header>
  );
}
