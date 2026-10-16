"""Pre-registered seed map of Parts 5-6 (D38): every range in configs/eta.yaml,
configs/rl.yaml and configs/api.yaml is disjoint from the others and from the Part 3-4
seeds of configs/evaluation.yaml (evaluation 1-10, calibration 101-103, development
201-207), which is not edited; RL training starts at or after `rl_training_from`."""

import itertools

import pytest

from traffic_twin.config import load_config
from traffic_twin.paths import API_CONFIG, ETA_CONFIG, EVALUATION_CONFIG, RL_CONFIG

# D38, as registered
ETA_SEEDS = {
    "train": (2000, 2299), "val": (2300, 2349), "calib": (2350, 2449),
    "reserve": (2450, 2899), "test": (2900, 2999),
}  # fmt: skip
RL_SEEDS = {
    "train": (1000, 1999), "selection": (3000, 3009), "burned": (3010, 3019),
    "dev_test": (3020, 3049),
}  # fmt: skip
DEMO_SEEDS = (4000, 4009)


def ranges() -> dict[str, range]:
    """Every registered Parts 5-6 range (inclusive bounds) by a readable name."""
    eta = load_config(ETA_CONFIG)["seeds"]
    rl = load_config(RL_CONFIG)["seeds"]
    api = load_config(API_CONFIG)["seeds"]
    out = {f"eta.{k}": v for k, v in eta.items()}
    out |= {f"rl.{k}": v for k, v in rl.items()}
    out |= {f"api.{k}": v for k, v in api.items()}
    for name, (lo, hi) in out.items():
        assert isinstance(lo, int) and isinstance(hi, int) and lo <= hi, name
    return {name: range(lo, hi + 1) for name, (lo, hi) in out.items()}


def part34_seeds() -> dict[str, set[int]]:
    seeds = load_config(EVALUATION_CONFIG)["seeds"]
    return {k: set(v) for k, v in seeds.items() if isinstance(v, list)}


def test_ranges_are_the_registered_ones():
    assert {k: tuple(v) for k, v in load_config(ETA_CONFIG)["seeds"].items()} == ETA_SEEDS
    assert {k: tuple(v) for k, v in load_config(RL_CONFIG)["seeds"].items()} == RL_SEEDS
    assert tuple(load_config(API_CONFIG)["seeds"]["demo"]) == DEMO_SEEDS


def test_eta_splits_are_contiguous_and_cover_2000_2999():
    eta = sorted(ETA_SEEDS.values())
    assert eta[0][0] == 2000 and eta[-1][1] == 2999
    for (_, hi), (lo, _) in itertools.pairwise(eta):
        assert lo == hi + 1


@pytest.mark.parametrize(("a", "b"), list(itertools.combinations(sorted(ranges()), 2)))
def test_registered_ranges_are_disjoint(a, b):
    r = ranges()
    assert not set(r[a]) & set(r[b]), f"{a} and {b} overlap"


def test_disjoint_from_the_part3_4_seeds():
    fixed = part34_seeds()
    assert fixed["evaluation"] == set(range(1, 11))
    assert fixed["calibration"] == {101, 102, 103}
    assert fixed["development"] == set(range(201, 208))
    used = set().union(*fixed.values())
    for name, r in ranges().items():
        assert not used & set(r), f"{name} reuses a Part 3-4 seed"


def test_rl_training_starts_at_the_registered_floor():
    floor = load_config(EVALUATION_CONFIG)["seeds"]["rl_training_from"]
    assert min(ranges()["rl.train"]) >= floor == 1000
