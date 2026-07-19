"""Emit an explicit nonzero BLOCKED result for gated V7 stages."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("reason")
    args = parser.parse_args()
    print(json.dumps({"experiment": args.experiment, "status": "BLOCKED", "reason": args.reason}))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
