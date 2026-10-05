import { describe, expect, test } from "vitest";

import { suggestedTaskOrder } from "./taskOrdering";
import type { GroundedTask } from "./types";

const task = (id: string, risk: number, distance: number, information: number, score: number) => ({
  task_id: id,
  enabled: true,
  validation_status: "certified",
  binding_index: Number(id.slice(1)),
  action_spec: {
    cost: {
      risk_score: risk,
      distance_m: distance,
      information_gain: information,
      duration_s: 5,
      model_score: score,
    },
  },
}) as GroundedTask;

describe("suggestedTaskOrder", () => {
  test("applies safety, distance and information gain in that order", () => {
    const tasks = [
      task("t2", 0.2, 1, 1, 1),
      task("t1", 0.1, 8, 0.2, 0.9),
      task("t0", 0.1, 2, 0.1, 0.8),
      task("t3", 0.1, 2, 0.8, 0.7),
    ];

    expect(suggestedTaskOrder(tasks).map((item) => item.task_id)).toEqual([
      "t3",
      "t0",
      "t1",
      "t2",
    ]);
  });
});
