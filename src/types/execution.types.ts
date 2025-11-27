/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

export interface ExecutionResult {
  blockId: string;
  success: boolean;
  data?: unknown;
  error?: string;
  executionTime: number;
  rowCount?: number;
  columnCount?: number;
}

export interface ExecutionState {
  isRunning: boolean;
  currentBlockId: string | null;
  progress: number;
  results: Map<string, ExecutionResult>;
  logs: ExecutionLog[];
}

export interface ExecutionLog {
  id: string;
  timestamp: Date;
  level: 'info' | 'warning' | 'error';
  blockId?: string;
  message: string;
}

export type ExecutionMode = 'run-all' | 'run-selected' | 'run-to-here';
