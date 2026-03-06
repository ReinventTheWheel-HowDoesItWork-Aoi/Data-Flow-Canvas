/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { useCallback, useRef, DragEvent } from 'react';
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
import { BaseBlock } from '@/components/blocks/BaseBlock';
import { AnimatedEdge } from './AnimatedEdge';
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

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      setBlocks(applyNodeChanges(changes, blocks) as PipelineBlock[]);
    },
    [blocks, setBlocks]
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
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

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      addBlock(type as BlockType, position);
    },
    [addBlock, screenToFlowPosition]
  );

  return (
    <div
      ref={reactFlowWrapper}
      className="w-full h-full relative"
      onDragOver={onDragOver}
      onDrop={onDrop}
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
              'data-input': '#FF2E97',
              'transform': '#BF00FF',
              'analysis': '#00D4FF',
              'visualization': '#DFFF00',
              'output': '#f43f5e',
            };
            return colors[category] || '#64748b';
          }}
          maskColor="rgb(var(--color-bg-primary) / 0.8)"
        />
      </ReactFlow>
    </div>
  );
}
