/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import Dexie, { type Table } from 'dexie';
import type { Project, ProjectMetadata } from '@/types';

interface DataSnapshot {
  id: string;
  projectId: string;
  blockId: string;
  data: unknown;
  createdAt: Date;
}

export class DataFlowDatabase extends Dexie {
  projects!: Table<Project, string>;
  dataSnapshots!: Table<DataSnapshot, string>;

  constructor() {
    super('DataFlowCanvas');

    this.version(1).stores({
      projects: 'id, name, updatedAt',
      dataSnapshots: 'id, projectId, blockId, createdAt',
    });
  }
}

export const db = new DataFlowDatabase();

// Storage utilities
export async function saveProject(project: Project): Promise<void> {
  project.updatedAt = new Date();
  await db.projects.put(project);
}

export async function getProject(id: string): Promise<Project | undefined> {
  return db.projects.get(id);
}

export async function getAllProjects(): Promise<ProjectMetadata[]> {
  const projects = await db.projects.orderBy('updatedAt').reverse().toArray();
  return projects.map(({ id, name, description, createdAt, updatedAt, blocks }) => ({
    id,
    name,
    description,
    createdAt,
    updatedAt,
    blockCount: blocks.length,
  }));
}

export async function deleteProject(id: string): Promise<void> {
  await db.transaction('rw', [db.projects, db.dataSnapshots], async () => {
    await db.dataSnapshots.where('projectId').equals(id).delete();
    await db.projects.delete(id);
  });
}

export async function saveDataSnapshot(
  projectId: string,
  blockId: string,
  data: unknown
): Promise<string> {
  const id = crypto.randomUUID();
  await db.dataSnapshots.add({
    id,
    projectId,
    blockId,
    data,
    createdAt: new Date(),
  });
  return id;
}

export async function getDataSnapshot(id: string): Promise<DataSnapshot | undefined> {
  return db.dataSnapshots.get(id);
}

export async function getStorageUsage(): Promise<{
  used: number;
  quota: number;
  percentage: number;
}> {
  if ('storage' in navigator && 'estimate' in navigator.storage) {
    const estimate = await navigator.storage.estimate();
    const used = estimate.usage || 0;
    const quota = estimate.quota || 0;
    return {
      used,
      quota,
      percentage: quota > 0 ? (used / quota) * 100 : 0,
    };
  }
  return { used: 0, quota: 0, percentage: 0 };
}
