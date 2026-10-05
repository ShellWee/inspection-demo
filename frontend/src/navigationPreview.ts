import type { FloorMap, Pose2D } from "./types";

export const robotClearance = { jackal: 0.41, husky: 0.60, "robot-dog": 0.44 } as const;

/** Circular footprint/cell intersection; the backend remains authoritative. */
export function isPoseFree(floor: FloorMap, pose: Pose2D, radius: number): boolean {
  const { x, y, yaw } = pose;
  const r = floor.resolution_m;
  if (![x, y, yaw].every(Number.isFinite) || !floor.occupancy_rows.length) return false;
  if (x - radius < 0 || y - radius < 0 || x + radius >= floor.width_m || y + radius >= floor.height_m) return false;
  for (let row = Math.floor((y - radius) / r); row <= Math.floor((y + radius) / r); row++) {
    for (let col = Math.floor((x - radius) / r); col <= Math.floor((x + radius) / r); col++) {
      if (floor.occupancy_rows[row]?.[col] === ".") continue;
      const dx = x - Math.max(col * r, Math.min(x, (col + 1) * r));
      const dy = y - Math.max(row * r, Math.min(y, (row + 1) * r));
      if (dx * dx + dy * dy <= radius * radius) return false;
    }
  }
  return true;
}

export function interpolateYaw(start: number, end: number, ratio: number): number {
  return start + Math.atan2(Math.sin(end - start), Math.cos(end - start)) * ratio;
}
