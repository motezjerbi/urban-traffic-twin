"""Rank candidate intersections with the weighted grid of spec section 2.2."""

from __future__ import annotations

import argparse

import pandas as pd

from traffic_twin.config import INTERSECTIONS_KEYS, load_config
from traffic_twin.paths import INTERSECTIONS_CONFIG

MAX_SCORE = 5


def score_candidates(cfg: dict) -> pd.DataFrame:
    weights: dict[str, float] = cfg["weights"]
    eliminatory = set(cfg.get("eliminatory", []))
    total_weight = sum(weights.values())
    rows = []
    for key, cand in cfg["candidates"].items():
        scores = cand["scores"]
        missing = set(weights) - set(scores)
        if missing:
            raise ValueError(f"{key}: missing scores for {sorted(missing)}")
        bad = {k: v for k, v in scores.items() if not 0 <= v <= MAX_SCORE}
        if bad:
            raise ValueError(f"{key}: scores must be 0-{MAX_SCORE}, got {bad}")
        weighted = sum(weights[c] * scores[c] for c in weights)
        rows.append(
            {
                "candidate": key,
                "label": cand.get("label", key),
                "score_pct": round(100 * weighted / (MAX_SCORE * total_weight), 1),
                "eliminated": any(scores[c] == 0 for c in eliminatory),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["eliminated", "score_pct"], ascending=[True, False]).reset_index(
        drop=True
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=INTERSECTIONS_CONFIG)
    args = parser.parse_args(argv)
    ranking = score_candidates(load_config(args.config, INTERSECTIONS_KEYS))
    print(ranking.to_string(index=False))
    kept = ranking[~ranking["eliminated"]]
    if kept.empty:
        print(
            "\nNo candidate passes the sensor criterion yet:"
            " fill the 'sensor' scores after `make explore`."
        )
    else:
        print(f"\nChosen: {kept.iloc[0]['label']}")


if __name__ == "__main__":
    main()
