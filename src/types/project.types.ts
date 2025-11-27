/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

import type { PipelineBlock, PipelineEdge } from './block.types';

export interface Project {
  id: string;
  name: string;
  description?: string;
  createdAt: Date;
  updatedAt: Date;
  blocks: PipelineBlock[];
  edges: PipelineEdge[];
  viewport: {
    x: number;
    y: number;
    zoom: number;
  };
}

export interface ProjectMetadata {
  id: string;
  name: string;
  description?: string;
  createdAt: Date;
  updatedAt: Date;
  blockCount: number;
  thumbnailUrl?: string;
}
