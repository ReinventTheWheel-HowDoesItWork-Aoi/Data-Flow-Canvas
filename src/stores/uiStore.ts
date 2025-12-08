/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { create } from 'zustand';

interface UIState {
  // Panels
  isSidebarOpen: boolean;
  isRightPanelOpen: boolean;
  isBottomPanelOpen: boolean;
  rightPanelTab: 'config' | 'preview' | 'visualization';
  bottomPanelTab: 'logs' | 'output';

  // Modals
  isCollaborationModalOpen: boolean;
  isSettingsModalOpen: boolean;
  isNewProjectModalOpen: boolean;
  isExportModalOpen: boolean;
  isJumpToBlockOpen: boolean;

  // Theme
  isDarkMode: boolean;

  // Actions
  toggleSidebar: () => void;
  toggleRightPanel: () => void;
  toggleBottomPanel: () => void;
  setRightPanelTab: (tab: 'config' | 'preview' | 'visualization') => void;
  setBottomPanelTab: (tab: 'logs' | 'output') => void;
  openCollaborationModal: () => void;
  closeCollaborationModal: () => void;
  openSettingsModal: () => void;
  closeSettingsModal: () => void;
  openNewProjectModal: () => void;
  closeNewProjectModal: () => void;
  openExportModal: () => void;
  closeExportModal: () => void;
  openJumpToBlock: () => void;
  closeJumpToBlock: () => void;
  toggleDarkMode: () => void;
  setDarkMode: (isDark: boolean) => void;
}

export const useUIStore = create<UIState>((set) => ({
  // Panels
  isSidebarOpen: true,
  isRightPanelOpen: true,
  isBottomPanelOpen: false,
  rightPanelTab: 'config',
  bottomPanelTab: 'logs',

  // Modals
  isCollaborationModalOpen: false,
  isSettingsModalOpen: false,
  isNewProjectModalOpen: false,
  isExportModalOpen: false,
  isJumpToBlockOpen: false,

  // Theme - default to dark, respect explicit light preference
  isDarkMode: typeof window !== 'undefined'
    ? !window.matchMedia('(prefers-color-scheme: light)').matches
    : true,

  // Actions
  toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
  toggleRightPanel: () => set((state) => ({ isRightPanelOpen: !state.isRightPanelOpen })),
  toggleBottomPanel: () => set((state) => ({ isBottomPanelOpen: !state.isBottomPanelOpen })),
  setRightPanelTab: (tab) => set({ rightPanelTab: tab }),
  setBottomPanelTab: (tab) => set({ bottomPanelTab: tab }),
  openCollaborationModal: () => set({ isCollaborationModalOpen: true }),
  closeCollaborationModal: () => set({ isCollaborationModalOpen: false }),
  openSettingsModal: () => set({ isSettingsModalOpen: true }),
  closeSettingsModal: () => set({ isSettingsModalOpen: false }),
  openNewProjectModal: () => set({ isNewProjectModalOpen: true }),
  closeNewProjectModal: () => set({ isNewProjectModalOpen: false }),
  openExportModal: () => set({ isExportModalOpen: true }),
  closeExportModal: () => set({ isExportModalOpen: false }),
  openJumpToBlock: () => set({ isJumpToBlockOpen: true }),
  closeJumpToBlock: () => set({ isJumpToBlockOpen: false }),
  toggleDarkMode: () => set((state) => {
    const newMode = !state.isDarkMode;
    if (typeof document !== 'undefined') {
      document.documentElement.classList.toggle('dark', newMode);
    }
    return { isDarkMode: newMode };
  }),
  setDarkMode: (isDark) => {
    if (typeof document !== 'undefined') {
      document.documentElement.classList.toggle('dark', isDark);
    }
    set({ isDarkMode: isDark });
  },
}));
