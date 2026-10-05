from __future__ import annotations

import argparse
import json
from pathlib import Path

from .comparison import compare_evaluations
from .engine import ToGSystem
from .index import GraphIndexManager
from .models import ToGConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="IFC Think-on-Graph baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build or validate an IFC graph index")
    index_parser.add_argument("ifc_path")
    index_parser.add_argument("--cache-dir", default="outputs/tog/indexes")
    index_parser.add_argument("--rebuild", action="store_true")

    ask_parser = subparsers.add_parser("ask", help="Ask using deterministic ToG fallback")
    ask_parser.add_argument("ifc_path")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--category", type=int, choices=[1, 2, 3, 4], default=1)
    ask_parser.add_argument("--variant", choices=["canonical", "bim"], default="bim")
    ask_parser.add_argument(
        "--profile",
        choices=["legacy", "lean-grounding"],
        default="legacy",
        help="Execution/cost profile (default preserves historical behaviour)",
    )
    ask_parser.add_argument("--cache-dir", default="outputs/tog/indexes")
    ask_parser.add_argument("--debug", action="store_true")

    compare_parser = subparsers.add_parser(
        "compare", help="Compare evaluation JSON files on their fair question intersection"
    )
    compare_parser.add_argument("results", nargs="+", help="Two or more evaluation JSON files")

    args = parser.parse_args()
    if args.command == "compare":
        if len(args.results) < 2:
            parser.error("compare requires at least two result files")
        print(json.dumps(compare_evaluations(args.results), indent=2, ensure_ascii=False))
        return 0

    manager = GraphIndexManager(Path(args.cache_dir))
    if args.command == "index":
        report = manager.ensure_index(args.ifc_path, rebuild=args.rebuild)
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0

    system = ToGSystem(
        manager,
        config=ToGConfig(profile=args.profile, variant=args.variant, debug=args.debug),
    )
    try:
        response = system.ask(args.question, args.category, args.ifc_path)
        print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
    finally:
        system.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
