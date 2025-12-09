/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useCanvasStore } from '@/stores/canvasStore';
import { useProjectStore } from '@/stores/projectStore';
import {
  saveProject,
  getProject,
  getAllProjects,
  deleteProject as deleteProjectFromDb,
} from '@/lib/storage';
import type { Project } from '@/types';
import { v4 as uuidv4 } from 'uuid';

export function useProject() {
  const navigate = useNavigate();
  const { blocks, edges, viewport, setBlocks, setEdges, setViewport, clearCanvas } = useCanvasStore();
  const {
    currentProject,
    setCurrentProject,
    updateCurrentProject,
    projectList,
    setProjectList,
    addToProjectList,
    removeFromProjectList,
    setLoading,
    setSaving,
    setLastSaved,
    setHasUnsavedChanges,
  } = useProjectStore();

  // Load all projects on mount
  useEffect(() => {
    loadProjectList();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync canvas state to project
  useEffect(() => {
    if (currentProject) {
      const hasChanges =
        JSON.stringify(currentProject.blocks) !== JSON.stringify(blocks) ||
        JSON.stringify(currentProject.edges) !== JSON.stringify(edges);

      if (hasChanges) {
        setHasUnsavedChanges(true);
      }
    }
  }, [blocks, edges, currentProject, setHasUnsavedChanges]);

  const loadProjectList = useCallback(async () => {
    try {
      const projects = await getAllProjects();
      setProjectList(projects);
    } catch (error) {
      // Failed to load projects silently
    }
  }, [setProjectList]);

  const createNewProject = useCallback(
    (name: string = 'Untitled Project') => {
      const project: Project = {
        id: uuidv4(),
        name,
        createdAt: new Date(),
        updatedAt: new Date(),
        blocks: [],
        edges: [],
        viewport: { x: 0, y: 0, zoom: 1 },
      };

      clearCanvas();
      setCurrentProject(project);
      navigate(`/editor/${project.id}`);

      return project;
    },
    [clearCanvas, setCurrentProject, navigate]
  );

  const loadProject = useCallback(
    async (projectId: string) => {
      setLoading(true);
      try {
        const project = await getProject(projectId);
        if (project) {
          setCurrentProject(project);
          setBlocks(project.blocks);
          setEdges(project.edges);
          setViewport(project.viewport);
          return project;
        } else {
          // Project not found, create new
          return createNewProject();
        }
      } catch (error) {
        return null;
      } finally {
        setLoading(false);
      }
    },
    [setLoading, setCurrentProject, setBlocks, setEdges, setViewport, createNewProject]
  );

  const saveCurrentProject = useCallback(async () => {
    if (!currentProject) return;

    setSaving(true);
    try {
      const projectToSave: Project = {
        ...currentProject,
        blocks,
        edges,
        viewport,
        updatedAt: new Date(),
      };

      await saveProject(projectToSave);
      setCurrentProject(projectToSave);
      setLastSaved(new Date());

      // Update project list
      const existingIndex = projectList.findIndex((p) => p.id === projectToSave.id);
      if (existingIndex === -1) {
        addToProjectList({
          id: projectToSave.id,
          name: projectToSave.name,
          description: projectToSave.description,
          createdAt: projectToSave.createdAt,
          updatedAt: projectToSave.updatedAt,
          blockCount: projectToSave.blocks.length,
        });
      } else {
        await loadProjectList();
      }

      return true;
    } catch (error) {
      return false;
    } finally {
      setSaving(false);
    }
  }, [
    currentProject,
    blocks,
    edges,
    viewport,
    setSaving,
    setCurrentProject,
    setLastSaved,
    projectList,
    addToProjectList,
    loadProjectList,
  ]);

  const deleteProject = useCallback(
    async (projectId: string) => {
      try {
        await deleteProjectFromDb(projectId);
        removeFromProjectList(projectId);

        if (currentProject?.id === projectId) {
          clearCanvas();
          setCurrentProject(null);
          navigate('/projects');
        }

        return true;
      } catch (error) {
        return false;
      }
    },
    [removeFromProjectList, currentProject, clearCanvas, setCurrentProject, navigate]
  );

  const renameProject = useCallback(
    async (newName: string) => {
      if (!currentProject) return;

      updateCurrentProject({ name: newName });
      await saveCurrentProject();
    },
    [currentProject, updateCurrentProject, saveCurrentProject]
  );

  const duplicateProject = useCallback(
    async (projectId: string) => {
      const project = await getProject(projectId);
      if (!project) return null;

      const newProject: Project = {
        ...project,
        id: uuidv4(),
        name: `${project.name} (Copy)`,
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      await saveProject(newProject);
      await loadProjectList();

      return newProject;
    },
    [loadProjectList]
  );

  return {
    currentProject,
    projectList,
    createNewProject,
    loadProject,
    saveCurrentProject,
    deleteProject,
    renameProject,
    duplicateProject,
    loadProjectList,
  };
}
