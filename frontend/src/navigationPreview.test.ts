import { expect, test } from "vitest";
import { interpolateYaw, isPoseFree } from "./navigationPreview";
import type { FloorMap } from "./types";

test("pose preview rejects obstacles, boundaries, missing maps, and nonfinite input", () => {
  const floor = { occupancy_rows: ["....", ".#..", "....", "...."], resolution_m: 1, width_m: 4, height_m: 4 } as FloorMap;
  expect(isPoseFree(floor, { x: 2.8, y: 2.8, yaw: 0 }, 0.4)).toBe(true);
  expect(isPoseFree(floor, { x: 1.5, y: 1.5, yaw: 0 }, 0.4)).toBe(false);
  expect(isPoseFree(floor, { x: 2.2, y: 1.5, yaw: 0 }, 0.4)).toBe(false);
  expect(isPoseFree(floor, { x: 0, y: 1, yaw: 0 }, 0.4)).toBe(false);
  expect(isPoseFree(floor, { x: NaN, y: 1, yaw: 0 }, 0.4)).toBe(false);
  expect(isPoseFree({ ...floor, occupancy_rows: [] }, { x: 2, y: 2, yaw: 0 }, 0.4)).toBe(false);
});

test("replay interpolates the shortest rotation across the angle wrap", () => {
  const result = interpolateYaw(179 * Math.PI / 180, -179 * Math.PI / 180, 0.5);
  expect(result).toBeCloseTo(Math.PI);
});
