/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import * as Y from 'yjs';
import { WebrtcProvider } from 'y-webrtc';
import type { Collaborator, ConnectionStatus } from '@/types';

const STUN_SERVERS = [
  'stun:stun.l.google.com:19302',
  'stun:openrelay.metered.ca:80',
];

export class CollaborationManager {
  private doc: Y.Doc;
  private provider: WebrtcProvider | null = null;
  private roomId: string | null = null;

  public onStatusChange: ((status: ConnectionStatus) => void) | null = null;
  public onPeersChange: ((peers: Collaborator[]) => void) | null = null;

  constructor() {
    this.doc = new Y.Doc();
  }

  async startSession(userName: string): Promise<string> {
    const roomId = crypto.randomUUID();
    await this.joinSession(roomId, userName);
    return roomId;
  }

  async joinSession(roomId: string, userName: string): Promise<void> {
    if (this.provider) {
      this.disconnect();
    }

    this.roomId = roomId;

    this.provider = new WebrtcProvider(roomId, this.doc, {
      signaling: ['wss://signaling.yjs.dev'],
      password: undefined,
      awareness: undefined,
      maxConns: 5,
      filterBcConns: true,
      peerOpts: {
        config: {
          iceServers: STUN_SERVERS.map((url) => ({ urls: url })),
        },
      },
    });

    // Set local awareness
    this.provider.awareness.setLocalStateField('user', {
      name: userName,
      color: this.generateColor(),
      cursor: null,
    });

    // Listen for awareness changes
    this.provider.awareness.on('change', () => {
      this.emitPeersChange();
    });

    // Connection status
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    this.provider.on('status', (event: any) => {
      const status = event.status || (event.connected ? 'connected' : 'disconnected');
      this.onStatusChange?.(status as ConnectionStatus);
    });
  }

  disconnect(): void {
    if (this.provider) {
      this.provider.destroy();
      this.provider = null;
    }
    this.roomId = null;
  }

  updateCursor(x: number, y: number): void {
    if (this.provider) {
      this.provider.awareness.setLocalStateField('user', {
        ...this.provider.awareness.getLocalState()?.user,
        cursor: { x, y },
      });
    }
  }

  getSharedMap<T>(name: string): Y.Map<T> {
    return this.doc.getMap(name);
  }

  getSharedArray<T>(name: string): Y.Array<T> {
    return this.doc.getArray(name);
  }

  getRoomId(): string | null {
    return this.roomId;
  }

  private emitPeersChange(): void {
    if (!this.provider) return;

    const peers: Collaborator[] = [];
    this.provider.awareness.getStates().forEach((state, clientId) => {
      if (state.user) {
        peers.push({
          id: String(clientId),
          name: state.user.name,
          color: state.user.color,
          cursor: state.user.cursor,
          isOnline: true,
          joinedAt: new Date(),
        });
      }
    });

    this.onPeersChange?.(peers);
  }

  private generateColor(): string {
    const colors = [
      '#6366f1',
      '#8b5cf6',
      '#14b8a6',
      '#f59e0b',
      '#ec4899',
      '#10b981',
      '#3b82f6',
      '#f43f5e',
    ];
    return colors[Math.floor(Math.random() * colors.length)];
  }
}

export const collaborationManager = new CollaborationManager();
