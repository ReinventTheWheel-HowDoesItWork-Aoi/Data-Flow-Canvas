/**
 * @author Lavelle Hatcher Jr
 * @copyright Copyright (c) 2025 Lavelle Hatcher Jr. All rights reserved.
 */

export interface Viewport {
  x: number;
  y: number;
  zoom: number;
}

export interface Position {
  x: number;
  y: number;
}

export interface CanvasSettings {
  snapToGrid: boolean;
  gridSize: number;
  showMinimap: boolean;
  showControls: boolean;
}
