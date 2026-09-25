export type Pt = [number, number];

export interface Room {
  id: number;
  label: string;
  polygon: Pt[];
  holes: Pt[][];
  area_px: number;
  area_m2: number | null;
  centroid: Pt;
}

export interface Wall {
  id: number;
  p1: Pt;
  p2: Pt;
  thickness: number;
  exterior: boolean;
}

export interface Opening {
  id: number;
  kind: "door" | "window";
  polygon: Pt[];
  center: Pt;
  width: number;
  depth: number;
  angle: number;
  wall_id: number | null;
}

export interface Geometry {
  width: number;
  height: number;
  rooms: Room[];
  walls: Wall[];
  openings: Opening[];
  wall_polygons: Pt[][];
  px_per_meter: number | null;
  meta: Record<string, unknown>;
}

export interface Segmentation {
  /** class index per pixel at working resolution */
  label: Uint8Array;
  width: number;
  height: number;
  /** working / original scale */
  scale: number;
  meanConfidence: number;
  ms: number;
}

export const CLASSES = ["background", "wall", "room", "door", "window"] as const;
export const CLASS_COLORS: [number, number, number][] = [
  [255, 255, 255],
  [38, 38, 38],
  [129, 199, 235],
  [236, 112, 99],
  [88, 214, 141],
];
export const WALL = 1;
export const ROOM = 2;
export const DOOR = 3;
export const WINDOW = 4;

export const ROOM_TYPES = [
  "room", "living room", "kitchen", "bedroom", "bath", "entry", "hallway",
  "dining", "office", "storage", "closet", "utility", "garage", "balcony",
];
