import type { GroundedTask } from "./types";

export function suggestedTaskOrder(tasks: GroundedTask[]): GroundedTask[] {
  return [...tasks].sort((left, right) => {
    const a = left.action_spec.cost;
    const b = right.action_spec.cost;
    return (
      Number(!left.enabled || left.validation_status !== "certified") -
        Number(!right.enabled || right.validation_status !== "certified") ||
      a.risk_score - b.risk_score ||
      a.distance_m - b.distance_m ||
      b.information_gain - a.information_gain ||
      a.duration_s - b.duration_s ||
      b.model_score - a.model_score ||
      left.binding_index - right.binding_index
    );
  });
}

export function moveTask(tasks: GroundedTask[], from: number, to: number): GroundedTask[] {
  if (from === to || from < 0 || to < 0 || from >= tasks.length || to >= tasks.length) return tasks;
  const next = [...tasks];
  const [moved] = next.splice(from, 1);
  next.splice(to, 0, moved);
  return next;
}
