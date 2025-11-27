/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { useEffect, useCallback, useMemo } from 'react';
import { useCanvasStore } from '@/stores/canvasStore';
import { useExecutionStore } from '@/stores/executionStore';
import { useUIStore } from '@/stores/uiStore';

interface ShortcutHandler {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  meta?: boolean;
  handler: () => void;
  description: string;
}

export function useKeyboardShortcuts() {
  const { undo, redo } = useCanvasStore.temporal.getState();
  const removeBlock = useCanvasStore((state) => state.removeBlock);
  const selectedBlockIds = useCanvasStore((state) => state.selectedBlockIds);
  const clearSelection = useCanvasStore((state) => state.clearSelection);
  const isRunning = useExecutionStore((state) => state.isRunning);
  const startExecution = useExecutionStore((state) => state.startExecution);
  const stopExecution = useExecutionStore((state) => state.stopExecution);
  const toggleBottomPanel = useUIStore((state) => state.toggleBottomPanel);

  const shortcuts: ShortcutHandler[] = useMemo(
    () => [
      {
        key: 'z',
        ctrl: true,
        handler: undo,
        description: 'Undo',
      },
      {
        key: 'z',
        ctrl: true,
        shift: true,
        handler: redo,
        description: 'Redo',
      },
      {
        key: 'y',
        ctrl: true,
        handler: redo,
        description: 'Redo',
      },
      {
        key: 'Enter',
        ctrl: true,
        handler: () => {
          // Get current state at time of execution
          const running = useExecutionStore.getState().isRunning;
          if (running) {
            stopExecution();
          } else {
            startExecution('run-all');
          }
        },
        description: 'Run/Stop pipeline',
      },
      {
        key: 'Backspace',
        handler: () => {
          // Get current state at time of execution
          const ids = useCanvasStore.getState().selectedBlockIds;
          ids.forEach((id) => removeBlock(id));
        },
        description: 'Delete selected',
      },
      {
        key: 'Delete',
        handler: () => {
          // Get current state at time of execution
          const ids = useCanvasStore.getState().selectedBlockIds;
          ids.forEach((id) => removeBlock(id));
        },
        description: 'Delete selected',
      },
      {
        key: 'Escape',
        handler: clearSelection,
        description: 'Clear selection',
      },
      {
        key: 'j',
        ctrl: true,
        handler: toggleBottomPanel,
        description: 'Toggle bottom panel',
      },
    ],
    [undo, redo, startExecution, stopExecution, removeBlock, clearSelection, toggleBottomPanel]
  );

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      for (const shortcut of shortcuts) {
        const ctrlMatch = shortcut.ctrl
          ? event.ctrlKey || event.metaKey
          : !event.ctrlKey && !event.metaKey;
        const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey;

        if (
          event.key.toLowerCase() === shortcut.key.toLowerCase() &&
          ctrlMatch &&
          shiftMatch
        ) {
          event.preventDefault();
          shortcut.handler();
          return;
        }
      }
    },
    [shortcuts]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return shortcuts;
}
