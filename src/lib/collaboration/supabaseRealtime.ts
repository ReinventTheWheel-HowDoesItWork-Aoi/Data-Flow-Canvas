/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { supabase } from '@/lib/supabase/client';
import type { RealtimeChannel } from '@supabase/supabase-js';
import type { Collaborator, ConnectionStatus, PipelineBlock, PipelineEdge } from '@/types';

const MAX_GLOBAL_CONNECTIONS = 30;
const MAX_PER_SESSION = 3;
const HEARTBEAT_INTERVAL = 30000; // 30 seconds
const CURSOR_THROTTLE = 50; // ms between cursor updates

interface JoinSessionResult {
  success: boolean;
  error?: string;
  message?: string;
  sessionId?: string;
  participantCount?: number;
}

interface UserPresence {
  id: string;
  name: string;
  color: string;
  cursor: { x: number; y: number } | null;
  joinedAt: string;
}

interface CanvasState {
  blocks: PipelineBlock[];
  edges: PipelineEdge[];
}

interface CursorPayload {
  userId: string;
  userName: string;
  color: string;
  x: number;
  y: number;
}

export class SupabaseCollaborationManager {
  private channel: RealtimeChannel | null = null;
  private roomId: string | null = null;
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  private userId: string | null = null;
  private userName: string = 'Anonymous';
  private userColor: string = '#6366f1';
  private lastCursorUpdate: number = 0;
  private isInitialSync: boolean = true;

  // Callbacks
  public onStatusChange: ((status: ConnectionStatus) => void) | null = null;
  public onPeersChange: ((peers: Collaborator[]) => void) | null = null;
  public onError: ((error: string) => void) | null = null;
  public onCursorUpdate: ((cursor: CursorPayload) => void) | null = null;
  public onCanvasSync: ((state: CanvasState) => void) | null = null;
  public onBlockAdd: ((block: PipelineBlock) => void) | null = null;
  public onBlockUpdate: ((blockId: string, changes: Partial<PipelineBlock>) => void) | null = null;
  public onBlockDelete: ((blockId: string) => void) | null = null;
  public onEdgeAdd: ((edge: PipelineEdge) => void) | null = null;
  public onEdgeDelete: ((edgeId: string) => void) | null = null;

  constructor() {
    this.userColor = this.generateColor();
  }

  getUserId(): string | null {
    return this.userId;
  }

  getUserColor(): string {
    return this.userColor;
  }

  async startSession(userName: string): Promise<{ success: boolean; roomId?: string; error?: string }> {
    const roomId = crypto.randomUUID().slice(0, 8); // Short room ID for easy sharing
    const result = await this.joinSession(roomId, userName);

    if (result.success) {
      return { success: true, roomId };
    }
    return { success: false, error: result.error };
  }

  async joinSession(roomId: string, userName: string): Promise<JoinSessionResult> {
    // Get current user
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      return {
        success: false,
        error: 'NOT_AUTHENTICATED',
        message: 'You must be signed in to collaborate',
      };
    }

    this.userId = user.id;
    this.userName = userName;
    this.isInitialSync = true;

    // Check limits via database function
    const { data, error } = await supabase.rpc('try_join_session', {
      p_room_id: roomId,
      p_max_global: MAX_GLOBAL_CONNECTIONS,
      p_max_per_session: MAX_PER_SESSION,
    });

    if (error) {
      return {
        success: false,
        error: 'DATABASE_ERROR',
        message: 'Failed to join session. Please try again.',
      };
    }

    if (!data.success) {
      return {
        success: false,
        error: data.error,
        message: data.message,
      };
    }

    // Disconnect from any existing session
    if (this.channel) {
      await this.disconnect();
    }

    this.roomId = roomId;
    this.onStatusChange?.('connecting');

    // Create Supabase Realtime channel
    this.channel = supabase.channel(`collab:${roomId}`, {
      config: {
        presence: {
          key: this.userId,
        },
        broadcast: {
          self: false, // Don't receive own broadcasts
        },
      },
    });

    // Set up presence tracking
    this.channel
      .on('presence', { event: 'sync' }, () => {
        this.handlePresenceSync();
      })
      .on('presence', { event: 'join' }, () => {
        // User joined - handled by presence sync
      })
      .on('presence', { event: 'leave' }, () => {
        // User left - handled by presence sync
      })
      // Canvas sync events
      .on('broadcast', { event: 'cursor' }, ({ payload }) => {
        this.handleCursorUpdate(payload as CursorPayload);
      })
      .on('broadcast', { event: 'canvas_sync' }, ({ payload }) => {
        this.handleCanvasSync(payload as CanvasState);
      })
      .on('broadcast', { event: 'block_add' }, ({ payload }) => {
        this.onBlockAdd?.(payload as PipelineBlock);
      })
      .on('broadcast', { event: 'block_update' }, ({ payload }) => {
        const { blockId, changes } = payload as { blockId: string; changes: Partial<PipelineBlock> };
        this.onBlockUpdate?.(blockId, changes);
      })
      .on('broadcast', { event: 'block_delete' }, ({ payload }) => {
        this.onBlockDelete?.(payload.blockId as string);
      })
      .on('broadcast', { event: 'edge_add' }, ({ payload }) => {
        this.onEdgeAdd?.(payload as PipelineEdge);
      })
      .on('broadcast', { event: 'edge_delete' }, ({ payload }) => {
        this.onEdgeDelete?.(payload.edgeId as string);
      })
      .on('broadcast', { event: 'request_sync' }, ({ payload }) => {
        // Another user is requesting current canvas state
        // Only the first user (session creator) should respond
        this.handleSyncRequest(payload as { requesterId: string });
      });

    // Subscribe to channel
    await this.channel.subscribe(async (status) => {
      if (status === 'SUBSCRIBED') {
        // Track our presence
        await this.channel?.track({
          id: this.userId,
          name: this.userName,
          color: this.userColor,
          cursor: null,
          joinedAt: new Date().toISOString(),
        } as UserPresence);

        this.onStatusChange?.('connected');
        this.startHeartbeat();

        // Request canvas sync from other users after a short delay
        setTimeout(() => {
          this.requestCanvasSync();
        }, 500);
      } else if (status === 'CHANNEL_ERROR') {
        this.onStatusChange?.('disconnected');
        this.onError?.('Failed to connect to collaboration session');
      }
    });

    return {
      success: true,
      sessionId: data.session_id,
      message: data.message,
      participantCount: data.participant_count,
    };
  }

  async disconnect(): Promise<void> {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    if (this.roomId) {
      // Update database to mark user as left
      await supabase.rpc('leave_session', { p_room_id: this.roomId });
    }

    if (this.channel) {
      await this.channel.untrack();
      await supabase.removeChannel(this.channel);
      this.channel = null;
    }

    this.roomId = null;
    this.isInitialSync = true;
    this.onStatusChange?.('disconnected');
    this.onPeersChange?.([]);
  }

  // Cursor updates with throttling
  updateCursor(x: number, y: number): void {
    if (!this.channel || !this.userId) return;

    const now = Date.now();
    if (now - this.lastCursorUpdate < CURSOR_THROTTLE) return;
    this.lastCursorUpdate = now;

    // Broadcast cursor position to others
    this.channel.send({
      type: 'broadcast',
      event: 'cursor',
      payload: {
        userId: this.userId,
        userName: this.userName,
        color: this.userColor,
        x,
        y,
      },
    });
  }

  // Canvas sync methods
  broadcastCanvasState(blocks: PipelineBlock[], edges: PipelineEdge[]): void {
    if (!this.channel) return;

    this.channel.send({
      type: 'broadcast',
      event: 'canvas_sync',
      payload: { blocks, edges },
    });
  }

  broadcastBlockAdd(block: PipelineBlock): void {
    if (!this.channel) return;

    this.channel.send({
      type: 'broadcast',
      event: 'block_add',
      payload: block,
    });
  }

  broadcastBlockUpdate(blockId: string, changes: Partial<PipelineBlock>): void {
    if (!this.channel) return;

    this.channel.send({
      type: 'broadcast',
      event: 'block_update',
      payload: { blockId, changes },
    });
  }

  broadcastBlockDelete(blockId: string): void {
    if (!this.channel) return;

    this.channel.send({
      type: 'broadcast',
      event: 'block_delete',
      payload: { blockId },
    });
  }

  broadcastEdgeAdd(edge: PipelineEdge): void {
    if (!this.channel) return;

    this.channel.send({
      type: 'broadcast',
      event: 'edge_add',
      payload: edge,
    });
  }

  broadcastEdgeDelete(edgeId: string): void {
    if (!this.channel) return;

    this.channel.send({
      type: 'broadcast',
      event: 'edge_delete',
      payload: { edgeId },
    });
  }

  requestCanvasSync(): void {
    if (!this.channel || !this.userId) return;

    this.channel.send({
      type: 'broadcast',
      event: 'request_sync',
      payload: { requesterId: this.userId },
    });
  }

  // Send current canvas state to a specific requester
  respondToSyncRequest(blocks: PipelineBlock[], edges: PipelineEdge[]): void {
    if (!this.channel) return;

    this.channel.send({
      type: 'broadcast',
      event: 'canvas_sync',
      payload: { blocks, edges },
    });
  }

  getRoomId(): string | null {
    return this.roomId;
  }

  isConnected(): boolean {
    return this.channel !== null && this.roomId !== null;
  }

  private handlePresenceSync(): void {
    if (!this.channel) return;

    const state = this.channel.presenceState<UserPresence>();
    const peers: Collaborator[] = [];

    Object.entries(state).forEach(([key, presences]) => {
      if (presences && presences.length > 0) {
        const presence = presences[0];
        peers.push({
          id: presence.id || key,
          name: presence.name || 'Anonymous',
          color: presence.color || '#6366f1',
          cursor: presence.cursor || null,
          isOnline: true,
          joinedAt: new Date(presence.joinedAt || Date.now()),
        });
      }
    });

    this.onPeersChange?.(peers);
  }

  private handleCursorUpdate(payload: CursorPayload): void {
    // Skip our own cursor updates
    if (payload.userId === this.userId) return;
    this.onCursorUpdate?.(payload);
  }

  private handleCanvasSync(state: CanvasState): void {
    // Only apply sync if we just joined (initial sync)
    if (this.isInitialSync) {
      this.isInitialSync = false;
      this.onCanvasSync?.(state);
    }
  }

  private handleSyncRequest(payload: { requesterId: string }): void {
    // This will be handled by the hook which has access to canvas state
    // The hook should call respondToSyncRequest with current state
  }

  private startHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    this.heartbeatInterval = setInterval(async () => {
      if (this.roomId) {
        await supabase.rpc('collaboration_heartbeat', { p_room_id: this.roomId });
      }
    }, HEARTBEAT_INTERVAL);
  }

  private generateColor(): string {
    const colors = [
      '#6366f1', // indigo
      '#8b5cf6', // violet
      '#14b8a6', // teal
      '#f59e0b', // amber
      '#ec4899', // pink
      '#10b981', // emerald
      '#3b82f6', // blue
      '#f43f5e', // rose
    ];
    return colors[Math.floor(Math.random() * colors.length)];
  }
}

export const collaborationManager = new SupabaseCollaborationManager();
