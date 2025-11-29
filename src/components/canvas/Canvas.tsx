/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { useCallback, useRef, useEffect, DragEvent, MouseEvent } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useReactFlow,
  applyNodeChanges,
  applyEdgeChanges,
  type Connection,
  type Node,
  type NodeChange,
  type EdgeChange,
  BackgroundVariant,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useCanvasStore } from '@/stores/canvasStore';
import { useCollaboration } from '@/hooks/useCollaboration';
import { BaseBlock } from '@/components/blocks/BaseBlock';
import { AnimatedEdge } from './AnimatedEdge';
import { CollaboratorCursors } from '@/components/collaboration';
import { cn } from '@/lib/utils/cn';
import type { BlockType, BlockData, PipelineBlock, PipelineEdge } from '@/types';

// Define node and edge types outside component to prevent recreation
const nodeTypes = {
  custom: BaseBlock,
} as const;

const edgeTypes = {
  animated: AnimatedEdge,
} as const;

export function Canvas() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const lastDropTime = useRef<number>(0);
  const { screenToFlowPosition } = useReactFlow();

  // Use individual selectors to prevent unnecessary re-renders
  const blocks = useCanvasStore((state) => state.blocks);
  const edges = useCanvasStore((state) => state.edges);
  const setBlocks = useCanvasStore((state) => state.setBlocks);
  const setEdges = useCanvasStore((state) => state.setEdges);
  const setSelectedBlocks = useCanvasStore((state) => state.setSelectedBlocks);
  const addEdgeToStore = useCanvasStore((state) => state.addEdge);
  const setViewport = useCanvasStore((state) => state.setViewport);
  const addBlock = useCanvasStore((state) => state.addBlock);

  // Collaboration hook
  const {
    isConnected,
    remoteCursors,
    updateCursor,
    broadcastBlockAdd,
    broadcastBlockUpdate,
    broadcastBlockDelete,
    broadcastEdgeAdd,
    broadcastEdgeDelete,
  } = useCollaboration();

  // Track previous blocks/edges to detect changes
  const prevBlocksRef = useRef<PipelineBlock[]>(blocks);
  const prevEdgesRef = useRef<PipelineEdge[]>(edges);

  // Detect and broadcast block changes
  useEffect(() => {
    if (!isConnected) {
      prevBlocksRef.current = blocks;
      return;
    }

    const prevBlocks = prevBlocksRef.current;
    const prevBlockIds = new Set(prevBlocks.map((b) => b.id));
    const currentBlockIds = new Set(blocks.map((b) => b.id));

    // Detect added blocks
    blocks.forEach((block) => {
      if (!prevBlockIds.has(block.id)) {
        broadcastBlockAdd(block);
      }
    });

    // Detect removed blocks
    prevBlocks.forEach((block) => {
      if (!currentBlockIds.has(block.id)) {
        broadcastBlockDelete(block.id);
      }
    });

    // Detect updated blocks (position or data changes)
    blocks.forEach((block) => {
      const prevBlock = prevBlocks.find((b) => b.id === block.id);
      if (prevBlock) {
        const posChanged =
          prevBlock.position.x !== block.position.x ||
          prevBlock.position.y !== block.position.y;
        const dataChanged = JSON.stringify(prevBlock.data) !== JSON.stringify(block.data);

        if (posChanged || dataChanged) {
          broadcastBlockUpdate(block.id, {
            position: block.position,
            data: block.data,
          });
        }
      }
    });

    prevBlocksRef.current = blocks;
  }, [blocks, isConnected, broadcastBlockAdd, broadcastBlockDelete, broadcastBlockUpdate]);

  // Detect and broadcast edge changes
  useEffect(() => {
    if (!isConnected) {
      prevEdgesRef.current = edges;
      return;
    }

    const prevEdges = prevEdgesRef.current;
    const prevEdgeIds = new Set(prevEdges.map((e) => e.id));
    const currentEdgeIds = new Set(edges.map((e) => e.id));

    // Detect added edges
    edges.forEach((edge) => {
      if (!prevEdgeIds.has(edge.id)) {
        broadcastEdgeAdd(edge);
      }
    });

    // Detect removed edges
    prevEdges.forEach((edge) => {
      if (!currentEdgeIds.has(edge.id)) {
        broadcastEdgeDelete(edge.id);
      }
    });

    prevEdgesRef.current = edges;
  }, [edges, isConnected, broadcastEdgeAdd, broadcastEdgeDelete]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      // Use React Flow's optimized change application
      setBlocks(applyNodeChanges(changes, blocks) as PipelineBlock[]);
    },
    [blocks, setBlocks]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      // Use React Flow's optimized change application
      setEdges(applyEdgeChanges(changes, edges) as PipelineEdge[]);
    },
    [edges, setEdges]
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      if (connection.source && connection.target) {
        addEdgeToStore(
          connection.source,
          connection.target,
          connection.sourceHandle || undefined,
          connection.targetHandle || undefined
        );
      }
    },
    [addEdgeToStore]
  );

  const onSelectionChange = useCallback(
    ({ nodes }: { nodes: Node[] }) => {
      setSelectedBlocks(nodes.map((n) => n.id));
    },
    [setSelectedBlocks]
  );

  const onMoveEnd = useCallback(
    (_: unknown, viewport: { x: number; y: number; zoom: number }) => {
      setViewport(viewport);
    },
    [setViewport]
  );

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      event.stopPropagation();

      // Prevent duplicate drops within 100ms
      const now = Date.now();
      if (now - lastDropTime.current < 100) {
        return;
      }
      lastDropTime.current = now;

      const type = event.dataTransfer.getData('application/dataflow-block');
      if (!type) return;

      // Use React Flow's coordinate transformation
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      addBlock(type as BlockType, position);
    },
    [addBlock, screenToFlowPosition]
  );

  // Track mouse movement for cursor collaboration
  const onMouseMove = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      if (!isConnected || !reactFlowWrapper.current) return;

      const bounds = reactFlowWrapper.current.getBoundingClientRect();
      const x = event.clientX - bounds.left;
      const y = event.clientY - bounds.top;

      updateCursor(x, y);
    },
    [isConnected, updateCursor]
  );

  return (
    <div
      ref={reactFlowWrapper}
      className="w-full h-full relative"
      onDragOver={onDragOver}
      onDrop={onDrop}
      onMouseMove={onMouseMove}
    >
      <ReactFlow
        nodes={blocks}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        defaultEdgeOptions={{ type: 'animated' }}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onSelectionChange={onSelectionChange}
        onMoveEnd={onMoveEnd}
        fitView
        snapToGrid
        snapGrid={[20, 20]}
        deleteKeyCode={['Backspace', 'Delete']}
        multiSelectionKeyCode={['Shift', 'Meta']}
        className="canvas-grid"
        proOptions={{ hideAttribution: true }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="rgb(var(--color-border-default) / 0.5)"
        />
        <Controls
          className={cn(
            '!bg-bg-secondary !border-border-default !rounded-lg !shadow-md',
            '[&>button]:!bg-bg-secondary [&>button]:!border-border-default',
            '[&>button]:hover:!bg-bg-tertiary',
            '[&>button>svg]:!fill-text-secondary'
          )}
        />
        <MiniMap
          className={cn(
            '!bg-bg-secondary/80 !border-border-default !rounded-lg !shadow-md',
            'backdrop-blur-sm'
          )}
          nodeColor={(node) => {
            const category = (node.data as BlockData)?.category;
            const colors: Record<string, string> = {
              'data-input': '#6366f1',
              'transform': '#8b5cf6',
              'analysis': '#14b8a6',
              'visualization': '#f59e0b',
              'output': '#f43f5e',
            };
            return colors[category] || '#64748b';
          }}
          maskColor="rgb(var(--color-bg-primary) / 0.8)"
        />
      </ReactFlow>

      {/* Remote collaborator cursors */}
      {isConnected && <CollaboratorCursors cursors={remoteCursors} />}
    </div>
  );
}
