/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { create } from 'zustand';
import type { Collaborator, ConnectionStatus } from '@/types';

interface CollaborationState {
  sessionId: string | null;
  localUserId: string | null;
  localUserName: string;
  collaborators: Collaborator[];
  connectionStatus: ConnectionStatus;

  // Actions
  setSessionId: (id: string | null) => void;
  setLocalUser: (id: string, name: string) => void;
  setLocalUserName: (name: string) => void;
  addCollaborator: (collaborator: Collaborator) => void;
  removeCollaborator: (id: string) => void;
  updateCollaboratorCursor: (id: string, cursor: { x: number; y: number } | null) => void;
  setCollaborators: (collaborators: Collaborator[]) => void;
  setConnectionStatus: (status: ConnectionStatus) => void;
  reset: () => void;
}

export const useCollaborationStore = create<CollaborationState>((set) => ({
  sessionId: null,
  localUserId: null,
  localUserName: 'Anonymous',
  collaborators: [],
  connectionStatus: 'disconnected',

  setSessionId: (id) => set({ sessionId: id }),

  setLocalUser: (id, name) => set({ localUserId: id, localUserName: name }),

  setLocalUserName: (name) => set({ localUserName: name }),

  addCollaborator: (collaborator) =>
    set((state) => ({
      collaborators: [...state.collaborators, collaborator],
    })),

  removeCollaborator: (id) =>
    set((state) => ({
      collaborators: state.collaborators.filter((c) => c.id !== id),
    })),

  updateCollaboratorCursor: (id, cursor) =>
    set((state) => ({
      collaborators: state.collaborators.map((c) =>
        c.id === id ? { ...c, cursor } : c
      ),
    })),

  setCollaborators: (collaborators) => set({ collaborators }),

  setConnectionStatus: (status) => set({ connectionStatus: status }),

  reset: () =>
    set({
      sessionId: null,
      collaborators: [],
      connectionStatus: 'disconnected',
    }),
}));
