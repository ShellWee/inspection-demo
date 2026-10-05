import cytoscape, { type Core } from "cytoscape";
import { useEffect, useRef } from "react";

import type { GroundingResult } from "./types";

interface Props {
  subgraph: NonNullable<GroundingResult["subgraph"]>;
  highlightedNodeId: string | null;
  onHighlight: (nodeId: string | null) => void;
}

export default function SubgraphView({ subgraph, highlightedNodeId, onHighlight }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const instance = useRef<Core | null>(null);

  useEffect(() => {
    if (!container.current) return;
    instance.current?.destroy();
    const headless = navigator.userAgent.toLowerCase().includes("jsdom");
    const cy = cytoscape({
      container: headless ? undefined : container.current,
      headless,
      elements: [
        ...subgraph.nodes.map((node) => ({
          data: { id: node.id, label: node.label, kind: node.kind, target: node.is_target ? 1 : 0 },
          position: { x: (node.x ?? 1) * 28, y: (node.y ?? 1) * 24 },
        })),
        ...subgraph.edges.map((edge, index) => ({
          data: { id: `edge-${index}`, source: edge.source, target: edge.target, label: edge.relation },
        })),
      ],
      layout: headless ? { name: "preset" } : !subgraph.edges.length
        ? { name: "grid", cols: 5, avoidOverlap: true, avoidOverlapPadding: 24, nodeDimensionsIncludeLabels: true, fit: true, padding: 45 }
        : { name: "cose", animate: false, fit: true, padding: 55, nodeRepulsion: () => 8000, idealEdgeLength: () => 90, nodeDimensionsIncludeLabels: true, componentSpacing: 70, randomize: false },
      style: [
        { selector: "node", style: { "background-color": "#c0d0db", "border-color": "#8fa8b9", "border-width": 1.5, color: "#456174", label: "data(label)", "font-size": 11, "font-family": "IBM Plex Sans", "text-wrap": "ellipsis", "text-max-width": "110px", width: 28, height: 28, "text-valign": "bottom", "text-margin-y": 7 } },
        { selector: "node[target = 1]", style: { "background-color": "#137e79", "border-color": "#b2dad4", color: "#18384f", width: 40, height: 40, "border-width": 4 } },
        { selector: "node.highlighted", style: { "border-color": "#bd893e", "border-width": 5, "overlay-color": "#bd893e", "overlay-opacity": 0.12 } },
        { selector: "edge", style: { width: 1, "line-color": "#becdd6", "target-arrow-color": "#becdd6", "target-arrow-shape": "triangle", "curve-style": "bezier" } },
        { selector: "edge:selected", style: { label: "data(label)", color: "#456174", "font-size": 11, "text-background-color": "#fff", "text-background-opacity": 1 } },
      ],
      minZoom: 0.05,
      maxZoom: 3,
    });
    cy.on("mouseover", "node", (event) => onHighlight(event.target.id()));
    cy.on("mouseout", "node", () => onHighlight(null));
    instance.current = cy;
    const observer = new ResizeObserver(() => { cy.resize(); });
    observer.observe(container.current);
    return () => { observer.disconnect(); cy.destroy(); };
  }, [onHighlight, subgraph]);

  useEffect(() => {
    const cy = instance.current;
    if (!cy) return;
    cy.nodes().removeClass("highlighted");
    if (highlightedNodeId) cy.getElementById(highlightedNodeId).addClass("highlighted");
  }, [highlightedNodeId]);

  return <div className="graph-container"><div className="subgraph-canvas" ref={container} aria-label="Grounding subgraph" /><button className="secondary graph-fit" onClick={() => instance.current?.fit(undefined, 45)}>Fit graph</button></div>;
}
