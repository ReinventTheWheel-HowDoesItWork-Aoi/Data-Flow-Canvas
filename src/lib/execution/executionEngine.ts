/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { topologicalSort, getUpstreamBlocks } from './topologicalSort';
import { blockDefinitions } from '@/constants';
import type {
  PipelineBlock,
  PipelineEdge,
  ExecutionResult,
  ExecutionMode,
} from '@/types';

// Deep convert Pyodide data structures to plain JS objects
function deepConvertToJS(data: unknown): unknown {
  // Handle null/undefined
  if (data === null || data === undefined) {
    return data;
  }

  // Handle Map -> Object
  if (data instanceof Map) {
    const obj: Record<string, unknown> = {};
    data.forEach((value, key) => {
      obj[String(key)] = deepConvertToJS(value);
    });
    return obj;
  }

  // Handle Array
  if (Array.isArray(data)) {
    return data.map((item) => deepConvertToJS(item));
  }

  // Handle BigInt (common with pandas int64)
  if (typeof data === 'bigint') {
    return Number(data);
  }

  // Handle plain objects (but not Date, etc.)
  if (typeof data === 'object' && data.constructor === Object) {
    const obj: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(data)) {
      obj[key] = deepConvertToJS(value);
    }
    return obj;
  }

  // Return primitives and other types as-is
  return data;
}

export class ExecutionEngine {
  private pyodide: any;
  private resultCache: Map<string, ExecutionResult> = new Map();
  private abortController: AbortController | null = null;

  constructor(pyodide: any) {
    this.pyodide = pyodide;
  }

  async execute(
    blocks: PipelineBlock[],
    edges: PipelineEdge[],
    mode: ExecutionMode,
    selectedBlockId?: string,
    onBlockStart?: (blockId: string) => void,
    onBlockComplete?: (blockId: string, result: ExecutionResult) => void,
    onProgress?: (progress: number) => void
  ): Promise<Map<string, ExecutionResult>> {
    this.abortController = new AbortController();
    const results = new Map<string, ExecutionResult>();

    // Determine execution order
    let executionOrder: string[];

    switch (mode) {
      case 'run-all':
        executionOrder = topologicalSort(blocks, edges);
        break;
      case 'run-selected':
        executionOrder = this.getBlockWithDependencies(
          selectedBlockId!,
          blocks,
          edges
        );
        break;
      case 'run-to-here':
        executionOrder = this.getBlocksUpTo(selectedBlockId!, blocks, edges);
        break;
    }

    // Execute blocks in order
    for (let i = 0; i < executionOrder.length; i++) {
      if (this.abortController.signal.aborted) {
        break;
      }

      const blockId = executionOrder[i];
      const block = blocks.find((b) => b.id === blockId);

      if (!block) continue;

      onBlockStart?.(blockId);
      onProgress?.((i / executionOrder.length) * 100);

      const result = await this.executeBlock(block, blocks, edges, results);
      results.set(blockId, result);

      onBlockComplete?.(blockId, result);
    }

    onProgress?.(100);
    return results;
  }

  cancel(): void {
    this.abortController?.abort();
  }

  private async executeBlock(
    block: PipelineBlock,
    blocks: PipelineBlock[],
    edges: PipelineEdge[],
    previousResults: Map<string, ExecutionResult>
  ): Promise<ExecutionResult> {
    const startTime = performance.now();

    try {
      // Get input data from upstream blocks
      const inputEdges = edges.filter((e) => e.target === block.id);
      const inputData: any[] = [];

      for (const edge of inputEdges) {
        const sourceResult = previousResults.get(edge.source);
        if (sourceResult?.success && sourceResult.data) {
          inputData.push(sourceResult.data);
        }
      }

      const code = this.generatePythonCode(block, inputData);
      const result = await this.pyodide.runPythonAsync(code);

      // Result is a JSON string - parse it to get clean JS data
      let data: unknown;
      if (typeof result === 'string') {
        data = JSON.parse(result);
      } else if (result && typeof result.toJs === 'function') {
        // Fallback for non-JSON results
        data = result.toJs({ dict_converter: Object.fromEntries });
        if (typeof result.destroy === 'function') {
          result.destroy();
        }
        data = deepConvertToJS(data);
      } else {
        data = result;
      }

      const executionTime = performance.now() - startTime;

      return {
        blockId: block.id,
        success: true,
        data,
        executionTime,
      };
    } catch (error) {
      return {
        blockId: block.id,
        success: false,
        error: error instanceof Error ? error.message : String(error),
        executionTime: performance.now() - startTime,
      };
    }
  }

  private generatePythonCode(block: PipelineBlock, inputData: any[]): string {
    const { type, config } = block.data;
    const definition = blockDefinitions[type];

    // Create Python code with config and input data
    const configJson = JSON.stringify(config);
    const inputJson = JSON.stringify(inputData.length === 1 ? inputData[0] : inputData);

    return `
import json
import pandas as pd
import numpy as np

config = json.loads('''${configJson}''')
input_data_raw = json.loads('''${inputJson}''')

# Convert input to DataFrame if needed
if isinstance(input_data_raw, list) and len(input_data_raw) > 0:
    if isinstance(input_data_raw[0], dict):
        input_data = pd.DataFrame(input_data_raw)
    else:
        input_data = [pd.DataFrame(d) if isinstance(d, list) else d for d in input_data_raw]
else:
    input_data = input_data_raw

# Block execution code
${definition.pythonTemplate}

# Convert output to JSON-serializable format and return as JSON string
def convert_output(obj):
    if hasattr(obj, 'to_dict'):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {k: convert_output(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_output(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

_result = convert_output(output)
json.dumps(_result)
`;
  }

  private getBlockWithDependencies(
    blockId: string,
    blocks: PipelineBlock[],
    edges: PipelineEdge[]
  ): string[] {
    const dependencies = new Set<string>([blockId]);

    const collectDependencies = (id: string) => {
      for (const edge of edges) {
        if (edge.target === id && !dependencies.has(edge.source)) {
          dependencies.add(edge.source);
          collectDependencies(edge.source);
        }
      }
    };

    collectDependencies(blockId);

    return topologicalSort(
      blocks.filter((b) => dependencies.has(b.id)),
      edges.filter((e) => dependencies.has(e.source) && dependencies.has(e.target))
    );
  }

  private getBlocksUpTo(
    blockId: string,
    blocks: PipelineBlock[],
    edges: PipelineEdge[]
  ): string[] {
    return this.getBlockWithDependencies(blockId, blocks, edges);
  }
}
