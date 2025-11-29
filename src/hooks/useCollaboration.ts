/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { useEffect, useCallback, useRef, useState } from 'react';
import { useCanvasStore } from '@/stores/canvasStore';
import { useCollaborationStore } from '@/stores/collaborationStore';
import { collaborationManager } from '@/lib/collaboration/supabaseRealtime';
import type { PipelineBlock, PipelineEdge } from '@/types';

interface CursorData {
  userId: string;
  userName: string;
  color: string;
  x: number;
  y: number;
}

export function useCollaboration() {
  const [remoteCursors, setRemoteCursors] = useState<Map<string, CursorData>>(new Map());
  const cursorTimeouts = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  // Canvas store
  const blocks = useCanvasStore((state) => state.blocks);
  const edges = useCanvasStore((state) => state.edges);
  const setBlocks = useCanvasStore((state) => state.setBlocks);
  const setEdges = useCanvasStore((state) => state.setEdges);
  const addBlockDirect = useCanvasStore((state) => state.addBlockDirect);
  const removeBlock = useCanvasStore((state) => state.removeBlock);
  const removeEdge = useCanvasStore((state) => state.removeEdge);
  const addEdgeDirect = useCanvasStore((state) => state.addEdgeDirect);

  // Collaboration store
  const connectionStatus = useCollaborationStore((state) => state.connectionStatus);
  const setCollaborators = useCollaborationStore((state) => state.setCollaborators);
  const setConnectionStatus = useCollaborationStore((state) => state.setConnectionStatus);

  const isConnected = connectionStatus === 'connected';
  const blocksRef = useRef(blocks);
  const edgesRef = useRef(edges);

  // Keep refs updated
  useEffect(() => {
    blocksRef.current = blocks;
  }, [blocks]);

  useEffect(() => {
    edgesRef.current = edges;
  }, [edges]);

  // Set up collaboration manager callbacks
  useEffect(() => {
    collaborationManager.onStatusChange = setConnectionStatus;
    collaborationManager.onPeersChange = setCollaborators;

    collaborationManager.onCursorUpdate = (cursor) => {
      // Update cursor position
      setRemoteCursors((prev) => {
        const newMap = new Map(prev);
        newMap.set(cursor.userId, {
          userId: cursor.userId,
          userName: cursor.userName,
          color: cursor.color,
          x: cursor.x,
          y: cursor.y,
        });
        return newMap;
      });

      // Clear existing timeout for this user
      const existingTimeout = cursorTimeouts.current.get(cursor.userId);
      if (existingTimeout) {
        clearTimeout(existingTimeout);
      }

      // Remove cursor after 3 seconds of inactivity
      const timeout = setTimeout(() => {
        setRemoteCursors((prev) => {
          const newMap = new Map(prev);
          newMap.delete(cursor.userId);
          return newMap;
        });
        cursorTimeouts.current.delete(cursor.userId);
      }, 3000);

      cursorTimeouts.current.set(cursor.userId, timeout);
    };

    collaborationManager.onCanvasSync = (state) => {
      // Apply full canvas state from another user
      if (state.blocks && state.blocks.length > 0) {
        setBlocks(state.blocks);
      }
      if (state.edges) {
        setEdges(state.edges);
      }
    };

    collaborationManager.onBlockAdd = (block) => {
      // Check if block already exists to avoid duplicates
      const exists = blocksRef.current.some((b) => b.id === block.id);
      if (!exists && addBlockDirect) {
        addBlockDirect(block);
      }
    };

    collaborationManager.onBlockUpdate = (blockId, changes) => {
      setBlocks(
        blocksRef.current.map((block) =>
          block.id === blockId ? { ...block, ...changes } : block
        )
      );
    };

    collaborationManager.onBlockDelete = (blockId) => {
      if (removeBlock) {
        removeBlock(blockId);
      }
    };

    collaborationManager.onEdgeAdd = (edge) => {
      const exists = edgesRef.current.some((e) => e.id === edge.id);
      if (!exists && addEdgeDirect) {
        addEdgeDirect(edge);
      }
    };

    collaborationManager.onEdgeDelete = (edgeId) => {
      if (removeEdge) {
        removeEdge(edgeId);
      }
    };

    return () => {
      collaborationManager.onStatusChange = null;
      collaborationManager.onPeersChange = null;
      collaborationManager.onCursorUpdate = null;
      collaborationManager.onCanvasSync = null;
      collaborationManager.onBlockAdd = null;
      collaborationManager.onBlockUpdate = null;
      collaborationManager.onBlockDelete = null;
      collaborationManager.onEdgeAdd = null;
      collaborationManager.onEdgeDelete = null;

      // Clear all cursor timeouts
      cursorTimeouts.current.forEach((timeout) => clearTimeout(timeout));
      cursorTimeouts.current.clear();
    };
  }, [setConnectionStatus, setCollaborators, setBlocks, setEdges, addBlockDirect, removeBlock, addEdgeDirect, removeEdge]);

  // Broadcast local changes when connected
  const broadcastBlockAdd = useCallback((block: PipelineBlock) => {
    if (isConnected) {
      collaborationManager.broadcastBlockAdd(block);
    }
  }, [isConnected]);

  const broadcastBlockUpdate = useCallback((blockId: string, changes: Partial<PipelineBlock>) => {
    if (isConnected) {
      collaborationManager.broadcastBlockUpdate(blockId, changes);
    }
  }, [isConnected]);

  const broadcastBlockDelete = useCallback((blockId: string) => {
    if (isConnected) {
      collaborationManager.broadcastBlockDelete(blockId);
    }
  }, [isConnected]);

  const broadcastEdgeAdd = useCallback((edge: PipelineEdge) => {
    if (isConnected) {
      collaborationManager.broadcastEdgeAdd(edge);
    }
  }, [isConnected]);

  const broadcastEdgeDelete = useCallback((edgeId: string) => {
    if (isConnected) {
      collaborationManager.broadcastEdgeDelete(edgeId);
    }
  }, [isConnected]);

  const updateCursor = useCallback((x: number, y: number) => {
    if (isConnected) {
      collaborationManager.updateCursor(x, y);
    }
  }, [isConnected]);

  // Send current canvas state when requested by other users
  const respondToSyncRequest = useCallback(() => {
    if (isConnected) {
      collaborationManager.respondToSyncRequest(blocksRef.current, edgesRef.current);
    }
  }, [isConnected]);

  return {
    isConnected,
    remoteCursors,
    updateCursor,
    broadcastBlockAdd,
    broadcastBlockUpdate,
    broadcastBlockDelete,
    broadcastEdgeAdd,
    broadcastEdgeDelete,
    respondToSyncRequest,
  };
}
