/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import { supabase } from '@/lib/supabase/client';
import type { RealtimeChannel } from '@supabase/supabase-js';
import type { Collaborator, ConnectionStatus } from '@/types';

const MAX_GLOBAL_CONNECTIONS = 30;
const MAX_PER_SESSION = 3;
const HEARTBEAT_INTERVAL = 30000; // 30 seconds

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

export class SupabaseCollaborationManager {
  private channel: RealtimeChannel | null = null;
  private roomId: string | null = null;
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  private userId: string | null = null;
  private userName: string = 'Anonymous';
  private userColor: string = '#6366f1';

  public onStatusChange: ((status: ConnectionStatus) => void) | null = null;
  public onPeersChange: ((peers: Collaborator[]) => void) | null = null;
  public onError: ((error: string) => void) | null = null;

  constructor() {
    this.userColor = this.generateColor();
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

    // Check limits via database function
    const { data, error } = await supabase.rpc('try_join_session', {
      p_room_id: roomId,
      p_max_global: MAX_GLOBAL_CONNECTIONS,
      p_max_per_session: MAX_PER_SESSION,
    });

    if (error) {
      console.error('Error joining session:', error);
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
      },
    });

    // Set up presence tracking
    this.channel
      .on('presence', { event: 'sync' }, () => {
        this.handlePresenceSync();
      })
      .on('presence', { event: 'join' }, ({ key, newPresences }) => {
        console.log('User joined:', key, newPresences);
      })
      .on('presence', { event: 'leave' }, ({ key, leftPresences }) => {
        console.log('User left:', key, leftPresences);
      })
      .on('broadcast', { event: 'cursor' }, ({ payload }) => {
        this.handleCursorUpdate(payload);
      });

    // Subscribe to channel
    const status = await this.channel.subscribe(async (status) => {
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
    this.onStatusChange?.('disconnected');
    this.onPeersChange?.([]);
  }

  updateCursor(x: number, y: number): void {
    if (!this.channel || !this.userId) return;

    // Broadcast cursor position to others
    this.channel.send({
      type: 'broadcast',
      event: 'cursor',
      payload: {
        userId: this.userId,
        x,
        y,
      },
    });

    // Update our presence with cursor position
    this.channel.track({
      id: this.userId,
      name: this.userName,
      color: this.userColor,
      cursor: { x, y },
      joinedAt: new Date().toISOString(),
    } as UserPresence);
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

  private handleCursorUpdate(payload: { userId: string; x: number; y: number }): void {
    // This is handled by presence sync, but we could use this for more frequent updates
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
