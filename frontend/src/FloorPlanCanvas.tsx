import { memo, useEffect, useMemo, useRef, useState } from "react";
import { Arrow, Circle, Layer, Line, Rect, Stage, Text } from "react-konva";
import type { ExecutionFrame, FloorMap, GroundedTask, Pose2D } from "./types";

interface Props {
  floor: FloorMap; tasks?: GroundedTask[]; pose?: Pose2D;
  onPoseChange?: (pose: Pose2D) => void; frame?: ExecutionFrame;
  trajectory?: ExecutionFrame[]; highlightedNodeId?: string | null;
  robotRadius?: number; poseValid?: boolean;
}
const PAD = 38;
interface GeometryProps { floor: FloorMap; width: number; height: number; scale: number; offsetX: number; offsetY: number }
const FloorGeometry = memo(function FloorGeometry({ floor, width, height, scale, offsetX, offsetY }: GeometryProps) {
  return <Layer listening={false}><Rect width={width} height={height} fill="#f7fafb" />{floor.polygons.map(polygon => <Line key={polygon.id} points={polygon.points.flatMap(point => [offsetX + point.x * scale, height - offsetY - point.y * scale])} closed fill={polygon.kind === "slab" ? "#e8eef1" : polygon.kind === "space" ? "#f5f8f9" : polygon.kind === "column" ? "#607b8d" : "#8ba0ad"} stroke={polygon.kind === "space" ? "#cbd8df" : "#a9bbc6"} strokeWidth={0.8} />)}</Layer>;
});
export default function FloorPlanCanvas({ floor, tasks = [], pose, onPoseChange, frame, trajectory = [], highlightedNodeId, robotRadius = 0.41, poseValid }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const stage = useRef<import("konva/lib/Stage").Stage>(null);
  const [width, setWidth] = useState(760);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [dragEnd, setDragEnd] = useState<{ x: number; y: number } | null>(null);
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(entries => { if (entries[0].contentRect.width > 0) setWidth(entries[0].contentRect.width); });
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  const height = Math.min(580, Math.max(380, width * 0.64));
  const scale = Math.min((width - PAD * 2) / floor.width_m, (height - PAD * 2) / floor.height_m);
  const offsetX = (width - floor.width_m * scale) / 2;
  const offsetY = (height - floor.height_m * scale) / 2;
  const toCanvas = (x: number, y: number) => ({ x: offsetX + x * scale, y: height - offsetY - y * scale });
  const robot = frame?.pose ?? pose;
  const robotPoint = robot ? toCanvas(robot.x, robot.y) : undefined;
  const path = useMemo(() => trajectory.flatMap(item => [offsetX + item.pose.x * scale, height - offsetY - item.pose.y * scale]), [trajectory, offsetX, offsetY, height, scale]);
  const updatePose = () => {
    const pointer = stage.current?.getPointerPosition();
    if (!dragStart || !pointer || !onPoseChange) return;
    onPoseChange({ x: (dragStart.x - offsetX) / scale, y: (height - offsetY - dragStart.y) / scale, yaw: Math.atan2(dragStart.y - pointer.y, pointer.x - dragStart.x) });
    setDragStart(null); setDragEnd(null);
  };
  const beginDrag = () => { if (onPoseChange) { const pointer = stage.current?.getPointerPosition() ?? null; setDragStart(pointer); setDragEnd(pointer); } };
  return <div ref={container} className="floorplan-frame" aria-label={`${floor.floor_id} floor plan`}>
    <Stage ref={stage} width={width} height={height} onMouseDown={beginDrag} onMouseMove={() => { if (dragStart) setDragEnd(stage.current?.getPointerPosition() ?? null); }} onMouseUp={updatePose} onMouseLeave={() => { setDragStart(null); setDragEnd(null); }} onTouchStart={beginDrag} onTouchMove={() => { if (dragStart) setDragEnd(stage.current?.getPointerPosition() ?? null); }} onTouchEnd={updatePose}>
      <FloorGeometry floor={floor} width={width} height={height} scale={scale} offsetX={offsetX} offsetY={offsetY} />
      <Layer listening={false}>
      {path.length >= 4 && <Line points={path} stroke="#b67a2e" strokeWidth={2.5} dash={[5, 4]} lineCap="round" />}
      {tasks.filter(task => task.target_xy).map((task, index) => { const p = toCanvas(...task.target_xy!); const highlighted = task.node_id === highlightedNodeId; return <Circle key={task.task_id} x={p.x} y={p.y} radius={highlighted ? 12 : 8} fill="#137e79" stroke="#fff" strokeWidth={2} name={`target-${index}`} />; })}
      {tasks.filter(task => task.target_xy).map((task, index) => { const p = toCanvas(...task.target_xy!); return <Text key={task.task_id} x={p.x + 11} y={p.y - 8} text={`${index + 1}. ${task.target_name}`} fill="#23515c" fontFamily="IBM Plex Sans" fontSize={11} width={140} ellipsis wrap="none" />; })}
      {robotPoint && robot && <><Circle x={robotPoint.x} y={robotPoint.y} radius={Math.max(7, robotRadius * scale)} fill={frame || poseValid ? "#b67a2e" : "#b5473b"} opacity={0.85} stroke="#fff" strokeWidth={2} /><Arrow points={[robotPoint.x, robotPoint.y, robotPoint.x + Math.cos(robot.yaw) * 24, robotPoint.y - Math.sin(robot.yaw) * 24]} fill="#66441c" stroke="#66441c" pointerLength={6} pointerWidth={6} /></>}
      {dragStart && dragEnd && <Arrow points={[dragStart.x, dragStart.y, dragEnd.x, dragEnd.y]} stroke="#1b586b" fill="#1b586b" strokeWidth={2} dash={[4, 3]} pointerLength={7} pointerWidth={7} />}
      <Text x={18} y={16} text={`${floor.width_m.toFixed(1)} × ${floor.height_m.toFixed(1)} m`} fill="#718692" fontFamily="IBM Plex Mono" fontSize={10} />
      </Layer>
    </Stage><p className="map-instruction">{onPoseChange ? "Click + drag to set position and heading" : "Building geometry · metres"}</p>
  </div>;
}
