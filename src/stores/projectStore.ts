/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { create } from 'zustand';
import type { Project, ProjectMetadata } from '@/types';

interface ProjectState {
  currentProject: Project | null;
  projectList: ProjectMetadata[];
  isLoading: boolean;
  isSaving: boolean;
  lastSaved: Date | null;
  hasUnsavedChanges: boolean;

  // Actions
  setCurrentProject: (project: Project | null) => void;
  updateCurrentProject: (updates: Partial<Project>) => void;
  setProjectList: (projects: ProjectMetadata[]) => void;
  addToProjectList: (project: ProjectMetadata) => void;
  removeFromProjectList: (id: string) => void;
  setLoading: (isLoading: boolean) => void;
  setSaving: (isSaving: boolean) => void;
  setLastSaved: (date: Date) => void;
  setHasUnsavedChanges: (hasChanges: boolean) => void;
  clearCurrentProject: () => void;
}

export const useProjectStore = create<ProjectState>((set) => ({
  currentProject: null,
  projectList: [],
  isLoading: false,
  isSaving: false,
  lastSaved: null,
  hasUnsavedChanges: false,

  setCurrentProject: (project) =>
    set({ currentProject: project, hasUnsavedChanges: false }),

  updateCurrentProject: (updates) =>
    set((state) => ({
      currentProject: state.currentProject
        ? { ...state.currentProject, ...updates }
        : null,
      hasUnsavedChanges: true,
    })),

  setProjectList: (projects) => set({ projectList: projects }),

  addToProjectList: (project) =>
    set((state) => ({
      projectList: [project, ...state.projectList],
    })),

  removeFromProjectList: (id) =>
    set((state) => ({
      projectList: state.projectList.filter((p) => p.id !== id),
    })),

  setLoading: (isLoading) => set({ isLoading }),

  setSaving: (isSaving) => set({ isSaving }),

  setLastSaved: (date) => set({ lastSaved: date, hasUnsavedChanges: false }),

  setHasUnsavedChanges: (hasChanges) => set({ hasUnsavedChanges: hasChanges }),

  clearCurrentProject: () =>
    set({ currentProject: null, hasUnsavedChanges: false }),
}));
