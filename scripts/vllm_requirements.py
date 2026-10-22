#!/usr/bin/env python3
"""
Print the installed vllm's dependencies that can join the training environment.

vllm 0.19.1 requires opencv-python-headless >= 4.13, and every such opencv
needs numpy >= 2, while the training pins hold numpy 1.26.4 (pandas 2.0.3,
pyarrow 14 and datasets 2.16 predate numpy 2). opencv only serves vllm's
image/video inputs, which text PPO rollouts never use, so it is left out
(mistral_common loses its [image] extra for the same reason):

    pip install --no-deps vllm==0.19.1
    python scripts/vllm_requirements.py > vllm_deps.txt
    pip install -r vllm_deps.txt -c <training pins without extras>
"""

import argparse
from importlib.metadata import requires
from typing import Iterable, List

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

EXCLUDED = {"opencv-python-headless"}


def runtime_requirements(specs: Iterable[str], excluded=EXCLUDED) -> List[str]:
    """Requirement strings that apply here (no optional extras), minus excluded and without extras."""
    kept = []
    for spec in specs:
        requirement = Requirement(spec)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        if canonicalize_name(requirement.name) in excluded:
            continue
        requirement.extras = set()
        requirement.marker = None
        kept.append(str(requirement))
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", default="vllm")
    args = parser.parse_args()
    print("\n".join(runtime_requirements(requires(args.package) or [])))


if __name__ == "__main__":
    main()
