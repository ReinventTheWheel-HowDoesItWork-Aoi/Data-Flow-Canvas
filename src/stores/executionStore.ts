/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { create } from 'zustand';
import type { ExecutionResult, ExecutionLog, ExecutionMode } from '@/types';
import { v4 as uuidv4 } from 'uuid';

interface ExecutionState {
  isRunning: boolean;
  currentBlockId: string | null;
  progress: number;
  results: Map<string, ExecutionResult>;
  logs: ExecutionLog[];
  mode: ExecutionMode;

  // Actions
  startExecution: (mode: ExecutionMode) => void;
  stopExecution: () => void;
  setCurrentBlock: (blockId: string | null) => void;
  setProgress: (progress: number) => void;
  addResult: (blockId: string, result: ExecutionResult) => void;
  clearResults: () => void;
  addLog: (level: ExecutionLog['level'], message: string, blockId?: string) => void;
  clearLogs: () => void;
}

export const useExecutionStore = create<ExecutionState>((set, get) => ({
  isRunning: false,
  currentBlockId: null,
  progress: 0,
  results: new Map(),
  logs: [],
  mode: 'run-all',

  startExecution: (mode) => {
    set({
      isRunning: true,
      mode,
      progress: 0,
      currentBlockId: null,
    });
    get().addLog('info', `Starting execution in ${mode} mode`);
  },

  stopExecution: () => {
    set({
      isRunning: false,
      currentBlockId: null,
      progress: 100,
    });
    get().addLog('info', 'Execution stopped');
  },

  setCurrentBlock: (blockId) => {
    set({ currentBlockId: blockId });
  },

  setProgress: (progress) => {
    set({ progress });
  },

  addResult: (blockId, result) => {
    set((state) => {
      const newResults = new Map(state.results);
      newResults.set(blockId, result);
      return { results: newResults };
    });
  },

  clearResults: () => {
    set({ results: new Map() });
  },

  addLog: (level, message, blockId) => {
    set((state) => ({
      logs: [
        ...state.logs,
        {
          id: uuidv4(),
          timestamp: new Date(),
          level,
          message,
          blockId,
        },
      ],
    }));
  },

  clearLogs: () => {
    set({ logs: [] });
  },
}));
