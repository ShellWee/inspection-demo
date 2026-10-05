from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main() -> int:
    base_url = os.environ.get("INSPECTION_DEMO_E2E_URL", "http://127.0.0.1:8769")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the live E2E")
    screenshot = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/e2e-live-q015.png")
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    query = os.environ.get(
        "INSPECTION_DEMO_E2E_QUERY",
        "Inspect the outlet switch in the Aero Capstone Studio on Level 4.",
    )
    expected_target = os.environ.get(
        "INSPECTION_DEMO_E2E_EXPECTED_TARGET",
        'EF_Outlet-Switch:36" AFF Quad:17969373',
    )
    map_width = float(os.environ.get("INSPECTION_DEMO_E2E_MAP_WIDTH", "105.7"))
    map_height = float(os.environ.get("INSPECTION_DEMO_E2E_MAP_HEIGHT", "73.1"))
    initial_x = float(os.environ.get("INSPECTION_DEMO_E2E_START_X", "65.5"))
    initial_y = float(os.environ.get("INSPECTION_DEMO_E2E_START_Y", "5.5"))
    console_errors: list[str] = []
    not_found: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on(
            "console",
            lambda message: (
                console_errors.append(message.text) if message.type == "error" else None
            ),
        )
        page.on(
            "response",
            lambda response: not_found.append(response.url) if response.status == 404 else None,
        )
        page.goto(base_url, wait_until="networkidle")
        expect(page.get_by_role("heading", name="Inspection control desk")).to_be_visible()

        page.get_by_label("OpenAI API key").fill(api_key)
        page.get_by_role("button", name="Connect models").click()
        expect(page.get_by_label("Reasoning model")).to_have_value("gpt-4.1", timeout=30_000)
        page.get_by_label("Natural-language query").fill(query)
        page.get_by_role("button", name="Run target grounding").click()

        configure = page.get_by_role("button", name="Configure execution")
        expect(configure).to_be_enabled(timeout=600_000)
        expect(page.locator(".certificate", has_text="all_bindings_certified")).to_be_visible()
        expect(page.get_by_text(expected_target, exact=False).first).to_be_visible()
        configure.click()

        expect(page.get_by_role("heading", name="Mission setup")).to_be_visible(timeout=30_000)
        canvas = page.locator(".floorplan-frame canvas").first
        box = canvas.bounding_box()
        if box is None:
            raise RuntimeError("execution floor canvas has no bounding box")
        scale = min((760 - 56) / map_width, (430 - 56) / map_height)
        start_x = box["x"] + 28 + initial_x * scale
        start_y = box["y"] + 430 - 28 - initial_y * scale
        page.mouse.move(start_x, start_y)
        page.mouse.down()
        page.mouse.move(start_x + 24, start_y)
        page.mouse.up()
        pose_text = page.locator(".pose-readout code").inner_text()
        pose_x = re.search(r"x\s+([0-9.]+)", pose_text)
        if pose_x is None or abs(float(pose_x.group(1)) - initial_x) > 0.2:
            raise AssertionError(f"pose drag did not set x={initial_x}: {pose_text}")

        page.get_by_role("button", name="Start simulation").click()
        expect(page.get_by_text("Mission complete")).to_be_visible(timeout=180_000)
        page.screenshot(path=str(screenshot.resolve()), full_page=True)
        browser.close()

    if not_found:
        raise AssertionError(f"Live UI produced 404 responses: {not_found}")
    if console_errors:
        raise AssertionError(f"Live UI produced console errors: {console_errors}")
    print(f"LIVE_UI_E2E=passed screenshot={screenshot.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
