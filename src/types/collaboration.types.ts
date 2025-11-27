/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

export interface Collaborator {
  id: string;
  name: string;
  color: string;
  cursor: { x: number; y: number } | null;
  isOnline: boolean;
  joinedAt: Date;
}

export interface CollaborationSession {
  id: string;
  initiatorId: string;
  collaborators: Collaborator[];
  createdAt: Date;
  isActive: boolean;
}

export type ConnectionStatus =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting';
