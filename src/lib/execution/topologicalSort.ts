/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import type { PipelineBlock, PipelineEdge } from '@/types';

export function topologicalSort(
  blocks: PipelineBlock[],
  edges: PipelineEdge[]
): string[] {
  const graph = new Map<string, string[]>();
  const inDegree = new Map<string, number>();

  // Initialize
  for (const block of blocks) {
    graph.set(block.id, []);
    inDegree.set(block.id, 0);
  }

  // Build graph
  for (const edge of edges) {
    const neighbors = graph.get(edge.source) || [];
    neighbors.push(edge.target);
    graph.set(edge.source, neighbors);
    inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
  }

  // Find all nodes with no incoming edges
  const queue: string[] = [];
  for (const [nodeId, degree] of inDegree) {
    if (degree === 0) {
      queue.push(nodeId);
    }
  }

  const result: string[] = [];

  while (queue.length > 0) {
    const node = queue.shift()!;
    result.push(node);

    const neighbors = graph.get(node) || [];
    for (const neighbor of neighbors) {
      const newDegree = (inDegree.get(neighbor) || 0) - 1;
      inDegree.set(neighbor, newDegree);

      if (newDegree === 0) {
        queue.push(neighbor);
      }
    }
  }

  // Check for cycles
  if (result.length !== blocks.length) {
    throw new Error('Pipeline contains a cycle');
  }

  return result;
}

export function hasCycle(
  blocks: PipelineBlock[],
  edges: PipelineEdge[]
): boolean {
  try {
    topologicalSort(blocks, edges);
    return false;
  } catch {
    return true;
  }
}

export function getUpstreamBlocks(
  blockId: string,
  edges: PipelineEdge[]
): string[] {
  const upstream: Set<string> = new Set();

  function collectUpstream(id: string) {
    for (const edge of edges) {
      if (edge.target === id && !upstream.has(edge.source)) {
        upstream.add(edge.source);
        collectUpstream(edge.source);
      }
    }
  }

  collectUpstream(blockId);
  return Array.from(upstream);
}

export function getDownstreamBlocks(
  blockId: string,
  edges: PipelineEdge[]
): string[] {
  const downstream: Set<string> = new Set();

  function collectDownstream(id: string) {
    for (const edge of edges) {
      if (edge.source === id && !downstream.has(edge.target)) {
        downstream.add(edge.target);
        collectDownstream(edge.target);
      }
    }
  }

  collectDownstream(blockId);
  return Array.from(downstream);
}
