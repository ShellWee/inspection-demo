from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main() -> int:
    output = (
        Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("artifacts/e2e.png").resolve()
    )
    engine = sys.argv[2] if len(sys.argv) > 2 else "chromium"
    base_url = os.getenv("INSPECTION_DEMO_E2E_URL", "http://127.0.0.1:8765")
    output.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as playwright:
        if engine == "edge":
            browser = playwright.chromium.launch(channel="msedge")
        elif engine == "webkit":
            browser = playwright.webkit.launch()
        else:
            browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on(
            "console",
            lambda message: errors.append(message.text) if message.type == "error" else None,
        )
        page.goto(base_url, wait_until="networkidle")
        page.get_by_role("heading", name="Inspection control desk").wait_for()
        page.get_by_label("OpenAI API key").fill("demo-key")
        page.get_by_role("button", name="Connect models").click()
        model = page.get_by_label("Reasoning model")
        page.wait_for_function("element => !element.disabled", arg=model.element_handle())
        model.select_option("gpt-4.1")
        page.get_by_role("button", name="Run target grounding").click()
        page.get_by_text("Room 404 · Cyber-Physical Systems Lab").wait_for()
        page.get_by_text("all_bindings_certified").wait_for()
        target_card = page.locator(".target-card")
        target_card.hover()
        expect(target_card).to_have_class(re.compile(r"\bhighlighted\b"))
        page.get_by_role("button", name="Configure execution").click()
        page.get_by_role("heading", name="Mission setup").wait_for()
        canvas = page.locator(".floorplan-frame canvas").first
        box = canvas.bounding_box()
        if box is None:
            raise AssertionError("floor-plan canvas was not rendered")
        page.mouse.move(box["x"] + 120, box["y"] + 300)
        page.mouse.down()
        page.mouse.move(box["x"] + 165, box["y"] + 280)
        page.mouse.up()
        page.get_by_role("button", name="Start simulation").click()
        page.get_by_text("Mission complete").wait_for(timeout=20_000)
        frames = page.locator(".telemetry").get_by_text("TRAJECTORY FRAMES")
        if frames.count() != 1:
            raise AssertionError("trajectory telemetry is missing")
        page.screenshot(path=str(output), full_page=True)

        page.get_by_role("button", name="01 · Setup").click()
        page.get_by_label("OpenAI API key").fill("demo-key")
        page.get_by_role("button", name="Connect models").click()
        page.get_by_label("Natural-language query").fill(
            "Scan the electrical panel in conference room 444, then inspect the AV display "
            "in the same room."
        )
        page.get_by_role("button", name="Run target grounding").click()
        page.get_by_text("Execution blocked").wait_for()
        if not page.get_by_role("button", name="Configure execution").is_disabled():
            raise AssertionError("q043 abstention did not block task execution")
        negative = output.with_name(f"{output.stem}-q043{output.suffix}")
        page.screenshot(path=str(negative), full_page=True)
        browser.close()
    if errors:
        raise AssertionError(f"browser console errors: {errors}")
    print(f"E2E passed; screenshot={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
