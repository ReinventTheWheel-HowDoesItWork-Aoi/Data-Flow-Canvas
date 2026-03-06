/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import {
  Play,
  Square,
  Save,
  Settings,
  Moon,
  Sun,
  ChevronLeft,
  Share2,
  Loader2,
  Terminal,
  HelpCircle,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from '@/components/ui/Tooltip';
import { LanguageSelector } from '@/components/ui/LanguageSelector';
import { useUIStore } from '@/stores/uiStore';
import { useProjectStore } from '@/stores/projectStore';
import { useExecution } from '@/hooks/useExecution';
import { useProject } from '@/hooks/useProject';

export function TopBar() {
  const { t } = useTranslation();
  const {
    isDarkMode,
    toggleDarkMode,
    openSettingsModal,
    toggleBottomPanel,
    isBottomPanelOpen,
  } = useUIStore();
  const { hasUnsavedChanges } = useProjectStore();
  const { saveCurrentProject } = useProject();
  const {
    runPipeline,
    cancelExecution,
    isRunning,
    isPyodideReady,
    isPyodideLoading,
  } = useExecution();

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
          <img src="/logo.svg" alt="Data Flow Canvas" className="w-8 h-8 rounded-lg" />
          <div>
            <h1 className="text-body font-semibold text-text-primary">
              Data Flow Canvas
            </h1>
            {hasUnsavedChanges && (
              <span className="text-small text-text-muted">{t('topBar.unsavedChanges')}</span>
            )}
          </div>
        </div>
      </div>

      {/* Center section - Run controls */}
      <div className="flex items-center gap-2">
        {isPyodideLoading && (
          <div className="flex items-center gap-2 text-text-muted text-small mr-2">
            <Loader2 size={14} className="animate-spin" />
            <span>{t('topBar.loadingPython')}</span>
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
              {isRunning ? t('topBar.stop') : t('topBar.run')}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            {isPyodideReady
              ? t('topBar.runPipelineTooltip')
              : t('topBar.waitingForPython')}
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
          <TooltipContent>{t('topBar.toggleLogs')}</TooltipContent>
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
            >
              <Save size={18} />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{t('topBar.saveProject')}</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="sm" onClick={() => {}}>
              <Share2 size={18} />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{t('topBar.share')}</TooltipContent>
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
          <TooltipContent>{t('topBar.helpDocs')}</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="sm" onClick={toggleDarkMode}>
              {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            {isDarkMode ? t('topBar.lightMode') : t('topBar.darkMode')}
          </TooltipContent>
        </Tooltip>

        <LanguageSelector />

        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="sm" onClick={openSettingsModal}>
              <Settings size={18} />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{t('topBar.settings')}</TooltipContent>
        </Tooltip>
      </div>
    </header>
  );
}
