/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { useCallback } from 'react';
import { useCanvasStore } from '@/stores/canvasStore';
import { useExecutionStore } from '@/stores/executionStore';
import { usePyodide } from '@/lib/pyodide';
import { ExecutionEngine } from '@/lib/execution';
import type { ExecutionMode } from '@/types';

export function useExecution() {
  const { pyodide, isReady, isLoading, loadProgress, error: pyodideError } = usePyodide();
  const { blocks, edges, setBlockState, updateBlock } = useCanvasStore();
  const {
    isRunning,
    startExecution,
    stopExecution,
    setCurrentBlock,
    setProgress,
    addResult,
    addLog,
    clearResults,
  } = useExecutionStore();

  const runPipeline = useCallback(
    async (mode: ExecutionMode = 'run-all', selectedBlockId?: string) => {
      if (!pyodide || !isReady) {
        addLog('error', 'Pyodide is not ready. Please wait for initialization.');
        return;
      }

      if (blocks.length === 0) {
        addLog('warning', 'No blocks in the pipeline to execute.');
        return;
      }

      try {
        startExecution(mode);
        clearResults();

        // Reset all block states
        blocks.forEach((block) => setBlockState(block.id, 'idle'));

        const engine = new ExecutionEngine(pyodide);

        await engine.execute(
          blocks,
          edges,
          mode,
          selectedBlockId,
          // onBlockStart
          (blockId) => {
            setCurrentBlock(blockId);
            setBlockState(blockId, 'executing');
            addLog('info', `Executing block: ${blocks.find(b => b.id === blockId)?.data.label}`);
          },
          // onBlockComplete
          (blockId, result) => {
            if (result.success) {
              updateBlock(blockId, { state: 'success', error: undefined });
              addLog('info', `Block completed in ${result.executionTime.toFixed(0)}ms`);
            } else {
              updateBlock(blockId, { state: 'error', error: result.error });
              addLog('error', `Block failed: ${result.error}`, blockId);
            }
            addResult(blockId, result);
          },
          // onProgress
          (progress) => {
            setProgress(progress);
          }
        );

        addLog('info', 'Pipeline execution completed');
      } catch (error) {
        addLog('error', `Pipeline execution failed: ${error instanceof Error ? error.message : String(error)}`);
      } finally {
        stopExecution();
        setCurrentBlock(null);
      }
    },
    [pyodide, isReady, blocks, edges, startExecution, stopExecution, setCurrentBlock, setProgress, addResult, addLog, clearResults, setBlockState, updateBlock]
  );

  const runSelected = useCallback(
    (blockId: string) => {
      runPipeline('run-selected', blockId);
    },
    [runPipeline]
  );

  const runToBlock = useCallback(
    (blockId: string) => {
      runPipeline('run-to-here', blockId);
    },
    [runPipeline]
  );

  const cancelExecution = useCallback(() => {
    stopExecution();
    addLog('warning', 'Execution cancelled by user');
    blocks.forEach((block) => {
      if (block.data.state === 'executing') {
        setBlockState(block.id, 'idle');
      }
    });
  }, [stopExecution, addLog, blocks, setBlockState]);

  return {
    runPipeline,
    runSelected,
    runToBlock,
    cancelExecution,
    isRunning,
    isPyodideReady: isReady,
    isPyodideLoading: isLoading,
    pyodideLoadProgress: loadProgress,
    pyodideError,
  };
}
