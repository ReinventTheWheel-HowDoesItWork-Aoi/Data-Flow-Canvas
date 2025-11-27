/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { useCanvasStore } from './canvasStore';

describe('canvasStore', () => {
  beforeEach(() => {
    // Reset store before each test
    useCanvasStore.getState().clearCanvas();
  });

  describe('addBlock', () => {
    it('should add a block with correct properties', () => {
      const store = useCanvasStore.getState();
      const id = store.addBlock('sample-data', { x: 100, y: 200 });

      expect(id).toBeDefined();
      expect(typeof id).toBe('string');

      const blocks = useCanvasStore.getState().blocks;
      expect(blocks).toHaveLength(1);
      expect(blocks[0].id).toBe(id);
      expect(blocks[0].position).toEqual({ x: 100, y: 200 });
      expect(blocks[0].data.type).toBe('sample-data');
      expect(blocks[0].data.category).toBe('data-input');
      expect(blocks[0].data.label).toBe('Sample Data');
      expect(blocks[0].data.state).toBe('idle');
    });

    it('should add blocks with default config', () => {
      const store = useCanvasStore.getState();
      store.addBlock('sample-data', { x: 0, y: 0 });

      const blocks = useCanvasStore.getState().blocks;
      expect(blocks[0].data.config).toEqual({ dataset: 'iris' });
    });

    it('should add multiple blocks', () => {
      const store = useCanvasStore.getState();
      store.addBlock('sample-data', { x: 0, y: 0 });
      store.addBlock('filter-rows', { x: 200, y: 0 });
      store.addBlock('chart', { x: 400, y: 0 });

      const blocks = useCanvasStore.getState().blocks;
      expect(blocks).toHaveLength(3);
    });
  });

  describe('updateBlock', () => {
    it('should update block data', () => {
      const store = useCanvasStore.getState();
      const id = store.addBlock('sample-data', { x: 0, y: 0 });

      store.updateBlock(id, { config: { dataset: 'wine' } });

      const blocks = useCanvasStore.getState().blocks;
      expect(blocks[0].data.config).toEqual({ dataset: 'wine' });
    });

    it('should preserve other block data when updating', () => {
      const store = useCanvasStore.getState();
      const id = store.addBlock('sample-data', { x: 0, y: 0 });

      store.updateBlock(id, { label: 'My Data' });

      const blocks = useCanvasStore.getState().blocks;
      expect(blocks[0].data.label).toBe('My Data');
      expect(blocks[0].data.type).toBe('sample-data');
      expect(blocks[0].data.config).toEqual({ dataset: 'iris' });
    });
  });

  describe('removeBlock', () => {
    it('should remove a block', () => {
      const store = useCanvasStore.getState();
      const id = store.addBlock('sample-data', { x: 0, y: 0 });

      store.removeBlock(id);

      expect(useCanvasStore.getState().blocks).toHaveLength(0);
    });

    it('should remove associated edges when removing a block', () => {
      const store = useCanvasStore.getState();
      const id1 = store.addBlock('sample-data', { x: 0, y: 0 });
      const id2 = store.addBlock('filter-rows', { x: 200, y: 0 });
      store.addEdge(id1, id2);

      store.removeBlock(id1);

      expect(useCanvasStore.getState().edges).toHaveLength(0);
    });

    it('should remove block from selection', () => {
      const store = useCanvasStore.getState();
      const id = store.addBlock('sample-data', { x: 0, y: 0 });
      store.setSelectedBlocks([id]);

      store.removeBlock(id);

      expect(useCanvasStore.getState().selectedBlockIds).toHaveLength(0);
    });
  });

  describe('setBlockState', () => {
    it('should update block state', () => {
      const store = useCanvasStore.getState();
      const id = store.addBlock('sample-data', { x: 0, y: 0 });

      store.setBlockState(id, 'executing');
      expect(useCanvasStore.getState().blocks[0].data.state).toBe('executing');

      store.setBlockState(id, 'success');
      expect(useCanvasStore.getState().blocks[0].data.state).toBe('success');

      store.setBlockState(id, 'error');
      expect(useCanvasStore.getState().blocks[0].data.state).toBe('error');
    });
  });

  describe('addEdge', () => {
    it('should add an edge between blocks', () => {
      const store = useCanvasStore.getState();
      const id1 = store.addBlock('sample-data', { x: 0, y: 0 });
      const id2 = store.addBlock('filter-rows', { x: 200, y: 0 });

      store.addEdge(id1, id2);

      const edges = useCanvasStore.getState().edges;
      expect(edges).toHaveLength(1);
      expect(edges[0].source).toBe(id1);
      expect(edges[0].target).toBe(id2);
    });

    it('should not add duplicate edges', () => {
      const store = useCanvasStore.getState();
      const id1 = store.addBlock('sample-data', { x: 0, y: 0 });
      const id2 = store.addBlock('filter-rows', { x: 200, y: 0 });

      store.addEdge(id1, id2);
      store.addEdge(id1, id2);

      expect(useCanvasStore.getState().edges).toHaveLength(1);
    });
  });

  describe('removeEdge', () => {
    it('should remove an edge', () => {
      const store = useCanvasStore.getState();
      const id1 = store.addBlock('sample-data', { x: 0, y: 0 });
      const id2 = store.addBlock('filter-rows', { x: 200, y: 0 });
      store.addEdge(id1, id2);

      store.removeEdge(`${id1}-${id2}`);

      expect(useCanvasStore.getState().edges).toHaveLength(0);
    });
  });

  describe('selection', () => {
    it('should set selected blocks', () => {
      const store = useCanvasStore.getState();
      const id1 = store.addBlock('sample-data', { x: 0, y: 0 });
      const id2 = store.addBlock('filter-rows', { x: 200, y: 0 });

      store.setSelectedBlocks([id1, id2]);

      expect(useCanvasStore.getState().selectedBlockIds).toEqual([id1, id2]);
    });

    it('should clear selection', () => {
      const store = useCanvasStore.getState();
      const id = store.addBlock('sample-data', { x: 0, y: 0 });
      store.setSelectedBlocks([id]);

      store.clearSelection();

      expect(useCanvasStore.getState().selectedBlockIds).toHaveLength(0);
    });
  });

  describe('viewport', () => {
    it('should set viewport', () => {
      const store = useCanvasStore.getState();

      store.setViewport({ x: 100, y: 200, zoom: 1.5 });

      expect(useCanvasStore.getState().viewport).toEqual({ x: 100, y: 200, zoom: 1.5 });
    });
  });

  describe('importPipeline', () => {
    it('should import blocks and edges', () => {
      const store = useCanvasStore.getState();
      const blocks = [
        {
          id: 'block-1',
          type: 'custom' as const,
          position: { x: 0, y: 0 },
          data: {
            type: 'sample-data' as const,
            category: 'data-input' as const,
            label: 'Sample Data',
            config: { dataset: 'iris' },
            state: 'idle' as const,
          },
        },
      ];
      const edges = [{ id: 'edge-1', source: 'block-1', target: 'block-2' }];

      store.importPipeline(blocks, edges);

      expect(useCanvasStore.getState().blocks).toEqual(blocks);
      expect(useCanvasStore.getState().edges).toEqual(edges);
    });
  });

  describe('clearCanvas', () => {
    it('should clear all blocks, edges, and selection', () => {
      const store = useCanvasStore.getState();
      const id1 = store.addBlock('sample-data', { x: 0, y: 0 });
      const id2 = store.addBlock('filter-rows', { x: 200, y: 0 });
      store.addEdge(id1, id2);
      store.setSelectedBlocks([id1]);

      store.clearCanvas();

      const state = useCanvasStore.getState();
      expect(state.blocks).toHaveLength(0);
      expect(state.edges).toHaveLength(0);
      expect(state.selectedBlockIds).toHaveLength(0);
    });
  });
});
