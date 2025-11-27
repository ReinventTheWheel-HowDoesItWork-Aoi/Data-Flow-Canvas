/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { describe, it, expect } from 'vitest';
import {
  topologicalSort,
  hasCycle,
  getUpstreamBlocks,
  getDownstreamBlocks,
} from './topologicalSort';
import type { PipelineBlock, PipelineEdge } from '@/types';

// Helper to create mock blocks
function createBlock(id: string): PipelineBlock {
  return {
    id,
    type: 'custom',
    position: { x: 0, y: 0 },
    data: {
      type: 'sample-data',
      category: 'data-input',
      label: 'Test Block',
      config: {},
      state: 'idle',
    },
  };
}

// Helper to create mock edges
function createEdge(source: string, target: string): PipelineEdge {
  return {
    id: `${source}-${target}`,
    source,
    target,
  };
}

describe('topologicalSort', () => {
  it('should return empty array for empty input', () => {
    const result = topologicalSort([], []);
    expect(result).toEqual([]);
  });

  it('should return single block for single block with no edges', () => {
    const blocks = [createBlock('A')];
    const result = topologicalSort(blocks, []);
    expect(result).toEqual(['A']);
  });

  it('should sort a linear chain correctly', () => {
    const blocks = [createBlock('A'), createBlock('B'), createBlock('C')];
    const edges = [createEdge('A', 'B'), createEdge('B', 'C')];

    const result = topologicalSort(blocks, edges);

    expect(result).toEqual(['A', 'B', 'C']);
  });

  it('should sort a diamond dependency correctly', () => {
    // A -> B -> D
    // A -> C -> D
    const blocks = [
      createBlock('A'),
      createBlock('B'),
      createBlock('C'),
      createBlock('D'),
    ];
    const edges = [
      createEdge('A', 'B'),
      createEdge('A', 'C'),
      createEdge('B', 'D'),
      createEdge('C', 'D'),
    ];

    const result = topologicalSort(blocks, edges);

    // A should come first, D should come last
    expect(result[0]).toBe('A');
    expect(result[result.length - 1]).toBe('D');
    // B and C should come before D
    expect(result.indexOf('B')).toBeLessThan(result.indexOf('D'));
    expect(result.indexOf('C')).toBeLessThan(result.indexOf('D'));
  });

  it('should handle multiple root nodes', () => {
    // A -> C
    // B -> C
    const blocks = [createBlock('A'), createBlock('B'), createBlock('C')];
    const edges = [createEdge('A', 'C'), createEdge('B', 'C')];

    const result = topologicalSort(blocks, edges);

    // C should come last
    expect(result[result.length - 1]).toBe('C');
    // A and B should both appear before C
    expect(result.indexOf('A')).toBeLessThan(result.indexOf('C'));
    expect(result.indexOf('B')).toBeLessThan(result.indexOf('C'));
  });

  it('should throw error for cyclic dependencies', () => {
    const blocks = [createBlock('A'), createBlock('B'), createBlock('C')];
    const edges = [
      createEdge('A', 'B'),
      createEdge('B', 'C'),
      createEdge('C', 'A'), // Creates cycle
    ];

    expect(() => topologicalSort(blocks, edges)).toThrow('Pipeline contains a cycle');
  });
});

describe('hasCycle', () => {
  it('should return false for empty graph', () => {
    expect(hasCycle([], [])).toBe(false);
  });

  it('should return false for acyclic graph', () => {
    const blocks = [createBlock('A'), createBlock('B'), createBlock('C')];
    const edges = [createEdge('A', 'B'), createEdge('B', 'C')];

    expect(hasCycle(blocks, edges)).toBe(false);
  });

  it('should return true for cyclic graph', () => {
    const blocks = [createBlock('A'), createBlock('B'), createBlock('C')];
    const edges = [
      createEdge('A', 'B'),
      createEdge('B', 'C'),
      createEdge('C', 'A'),
    ];

    expect(hasCycle(blocks, edges)).toBe(true);
  });

  it('should detect self-loop', () => {
    const blocks = [createBlock('A')];
    const edges = [createEdge('A', 'A')];

    expect(hasCycle(blocks, edges)).toBe(true);
  });
});

describe('getUpstreamBlocks', () => {
  it('should return empty array for root block', () => {
    const edges = [createEdge('A', 'B'), createEdge('B', 'C')];

    expect(getUpstreamBlocks('A', edges)).toEqual([]);
  });

  it('should return direct upstream block', () => {
    const edges = [createEdge('A', 'B')];

    const upstream = getUpstreamBlocks('B', edges);

    expect(upstream).toContain('A');
  });

  it('should return all upstream blocks recursively', () => {
    const edges = [createEdge('A', 'B'), createEdge('B', 'C'), createEdge('C', 'D')];

    const upstream = getUpstreamBlocks('D', edges);

    expect(upstream).toContain('A');
    expect(upstream).toContain('B');
    expect(upstream).toContain('C');
    expect(upstream).toHaveLength(3);
  });

  it('should handle diamond dependencies', () => {
    const edges = [
      createEdge('A', 'B'),
      createEdge('A', 'C'),
      createEdge('B', 'D'),
      createEdge('C', 'D'),
    ];

    const upstream = getUpstreamBlocks('D', edges);

    expect(upstream).toContain('A');
    expect(upstream).toContain('B');
    expect(upstream).toContain('C');
    expect(upstream).toHaveLength(3);
  });
});

describe('getDownstreamBlocks', () => {
  it('should return empty array for leaf block', () => {
    const edges = [createEdge('A', 'B'), createEdge('B', 'C')];

    expect(getDownstreamBlocks('C', edges)).toEqual([]);
  });

  it('should return direct downstream block', () => {
    const edges = [createEdge('A', 'B')];

    const downstream = getDownstreamBlocks('A', edges);

    expect(downstream).toContain('B');
  });

  it('should return all downstream blocks recursively', () => {
    const edges = [createEdge('A', 'B'), createEdge('B', 'C'), createEdge('C', 'D')];

    const downstream = getDownstreamBlocks('A', edges);

    expect(downstream).toContain('B');
    expect(downstream).toContain('C');
    expect(downstream).toContain('D');
    expect(downstream).toHaveLength(3);
  });

  it('should handle branching', () => {
    const edges = [
      createEdge('A', 'B'),
      createEdge('A', 'C'),
      createEdge('B', 'D'),
      createEdge('C', 'D'),
    ];

    const downstream = getDownstreamBlocks('A', edges);

    expect(downstream).toContain('B');
    expect(downstream).toContain('C');
    expect(downstream).toContain('D');
    expect(downstream).toHaveLength(3);
  });
});
