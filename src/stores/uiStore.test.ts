/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { useUIStore } from './uiStore';

describe('uiStore', () => {
  beforeEach(() => {
    // Reset store to default values
    useUIStore.setState({
      isSidebarOpen: true,
      isRightPanelOpen: true,
      isBottomPanelOpen: false,
      rightPanelTab: 'config',
      bottomPanelTab: 'logs',
      isCollaborationModalOpen: false,
      isSettingsModalOpen: false,
      isNewProjectModalOpen: false,
      isExportModalOpen: false,
      isDarkMode: false,
    });
  });

  describe('sidebar', () => {
    it('should toggle sidebar', () => {
      const store = useUIStore.getState();
      expect(store.isSidebarOpen).toBe(true);

      store.toggleSidebar();
      expect(useUIStore.getState().isSidebarOpen).toBe(false);

      store.toggleSidebar();
      expect(useUIStore.getState().isSidebarOpen).toBe(true);
    });
  });

  describe('right panel', () => {
    it('should toggle right panel', () => {
      const store = useUIStore.getState();
      expect(store.isRightPanelOpen).toBe(true);

      store.toggleRightPanel();
      expect(useUIStore.getState().isRightPanelOpen).toBe(false);
    });

    it('should set right panel tab', () => {
      const store = useUIStore.getState();
      expect(store.rightPanelTab).toBe('config');

      store.setRightPanelTab('preview');
      expect(useUIStore.getState().rightPanelTab).toBe('preview');

      store.setRightPanelTab('visualization');
      expect(useUIStore.getState().rightPanelTab).toBe('visualization');
    });
  });

  describe('bottom panel', () => {
    it('should toggle bottom panel', () => {
      const store = useUIStore.getState();
      expect(store.isBottomPanelOpen).toBe(false);

      store.toggleBottomPanel();
      expect(useUIStore.getState().isBottomPanelOpen).toBe(true);
    });

    it('should set bottom panel tab', () => {
      const store = useUIStore.getState();
      expect(store.bottomPanelTab).toBe('logs');

      store.setBottomPanelTab('output');
      expect(useUIStore.getState().bottomPanelTab).toBe('output');
    });
  });

  describe('modals', () => {
    it('should open and close collaboration modal', () => {
      const store = useUIStore.getState();
      expect(store.isCollaborationModalOpen).toBe(false);

      store.openCollaborationModal();
      expect(useUIStore.getState().isCollaborationModalOpen).toBe(true);

      store.closeCollaborationModal();
      expect(useUIStore.getState().isCollaborationModalOpen).toBe(false);
    });

    it('should open and close settings modal', () => {
      const store = useUIStore.getState();
      expect(store.isSettingsModalOpen).toBe(false);

      store.openSettingsModal();
      expect(useUIStore.getState().isSettingsModalOpen).toBe(true);

      store.closeSettingsModal();
      expect(useUIStore.getState().isSettingsModalOpen).toBe(false);
    });

    it('should open and close new project modal', () => {
      const store = useUIStore.getState();
      expect(store.isNewProjectModalOpen).toBe(false);

      store.openNewProjectModal();
      expect(useUIStore.getState().isNewProjectModalOpen).toBe(true);

      store.closeNewProjectModal();
      expect(useUIStore.getState().isNewProjectModalOpen).toBe(false);
    });

    it('should open and close export modal', () => {
      const store = useUIStore.getState();
      expect(store.isExportModalOpen).toBe(false);

      store.openExportModal();
      expect(useUIStore.getState().isExportModalOpen).toBe(true);

      store.closeExportModal();
      expect(useUIStore.getState().isExportModalOpen).toBe(false);
    });
  });

  describe('dark mode', () => {
    it('should toggle dark mode', () => {
      const store = useUIStore.getState();
      const initialMode = store.isDarkMode;

      store.toggleDarkMode();
      expect(useUIStore.getState().isDarkMode).toBe(!initialMode);

      store.toggleDarkMode();
      expect(useUIStore.getState().isDarkMode).toBe(initialMode);
    });

    it('should set dark mode explicitly', () => {
      const store = useUIStore.getState();

      store.setDarkMode(true);
      expect(useUIStore.getState().isDarkMode).toBe(true);

      store.setDarkMode(false);
      expect(useUIStore.getState().isDarkMode).toBe(false);
    });
  });
});
