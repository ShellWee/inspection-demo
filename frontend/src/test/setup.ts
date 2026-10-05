import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { afterEach, vi } from "vitest";

vi.mock("react-konva", () => {
  const Container = ({ children }: { children?: ReactNode }) =>
    createElement("div", null, children);
  const Shape = () => createElement("span");
  return {
    Stage: Container,
    Layer: Container,
    Rect: Shape,
    Line: Shape,
    Circle: Shape,
    Arrow: Shape,
    Text: Shape,
  };
});

afterEach(cleanup);

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

globalThis.ResizeObserver = ResizeObserverStub as typeof ResizeObserver;
