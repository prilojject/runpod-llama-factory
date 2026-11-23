"""Validate a request locally without a GPU or weights. Refuses anything that is not a dry run.

    python scripts/dry_run.py test_input.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    with open(sys.argv[1] if len(sys.argv) > 1 else "test_input.json", encoding="utf-8") as f:
        job = json.load(f)
    if not job.get("input", {}).get("dry_run"):
        sys.exit('refusing to run: set "dry_run": true (this script never trains)')
    import handler

    res = handler.handler({"id": "local-dry-run", **job})
    print(json.dumps(res, indent=2, default=str))
    sys.exit(1 if res.get("error") else 0)
