"""Run an independent DNS only after verifying a completed mean-field case."""
import argparse
import json
from pathlib import Path
import numpy as np
from .dns import run
from .verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--max-time", type=float, default=1000.)
    parser.add_argument("--dt", type=float)
    args = parser.parse_args()
    result = args.directory/"mean_field.npz"
    verified = verify(result)
    result.with_suffix(".verification.json").write_text(json.dumps(verified, indent=2)+"\n")
    with np.load(result, allow_pickle=False) as data:
        config = json.loads(str(data["metadata_json"]))["config"]
    # Only the problem parameters pass to DNS. No MF fields or kets are input.
    run(config["n"], args.directory/"dns.npz", reynolds=config["reynolds"],
        max_time=args.max_time, dt=args.dt)


if __name__ == "__main__":
    main()
