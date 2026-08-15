from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from backend.app.scoring import (
    TennisScoringEngine,
)


def print_score(
    engine: TennisScoringEngine,
) -> None:
    state = engine.get_state()

    print(
        json.dumps(
            state,
            indent=4,
        )
    )


def main():
    engine = (
        TennisScoringEngine()
    )

    sequence = [
        "Player A",
        "Player A",
        "Player B",
        "Player B",
        "Player A",
        "Player B",
        "Player A",
        "Player A",
    ]

    for index, winner in enumerate(
        sequence,
        start=1,
    ):
        print(
            f"\nPoint {index}: "
            f"{winner} wins"
        )

        engine.award_point(
            winner
        )

        print_score(
            engine
        )


if __name__ == "__main__":
    main()