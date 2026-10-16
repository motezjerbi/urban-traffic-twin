"""Pre-registered seed map of Parts 5-7 (D38, D70): every range in configs/eta.yaml,
configs/rl.yaml, configs/api.yaml and the own `seeds` block of configs/rl7.yaml is
disjoint from the others and from the Part 3-4 seeds of configs/evaluation.yaml
(evaluation 1-10, calibration 101-103, development 201-207), which is not edited; RL
training starts at or after `rl_training_from`.

Part 7 (D70) reads the raw rl7.yaml `seeds` block (only the keys it registers itself;
`burned` and `dev_test` are inherited from rl.yaml through `extends`): its new training
range 5000-9999 and its selection / select2 / holdout / sensitivity / stress / reserve
ranges come from the D38 reserve 3050-3999. Part 7 looks at dev-test 3020-3029 once
(D79), inside rl.yaml's dev_test. configs/rl.yaml stays the frozen Part 6 record."""

import itertools

import pytest
import yaml

from traffic_twin.config import load_config, text_sha256
from traffic_twin.paths import (
    API_CONFIG,
    ETA_CONFIG,
    EVALUATION_CONFIG,
    RL7_CONFIG,
    RL7_CONFIG_DIR,
    RL_CONFIG,
)

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
# D70, as registered (rl7.yaml's own keys)
RL7_SEEDS = {
    "train": (5000, 9999), "selection": (3050, 3089), "select2": (3090, 3109),
    "holdout": (3110, 3159), "sensitivity": (3160, 3179), "stress": (3180, 3199),
    "reserve": (3200, 3999),
}  # fmt: skip
RL7_DEVTEST_LOOK = (3020, 3029)  # D79: looked at once
RL_YAML_SHA8 = "c0b7dd7b"  # text_sha256 of the frozen Part 6 configs/rl.yaml


def rl7_raw_seeds() -> dict:
    """The `seeds` block written in configs/rl7.yaml itself (no `extends` merge)."""
    with open(RL7_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)["seeds"]


def ranges() -> dict[str, range]:
    """Every registered Parts 5-7 range (inclusive bounds) by a readable name."""
    eta = load_config(ETA_CONFIG)["seeds"]
    rl = load_config(RL_CONFIG)["seeds"]
    api = load_config(API_CONFIG)["seeds"]
    out = {f"eta.{k}": v for k, v in eta.items()}
    out |= {f"rl.{k}": v for k, v in rl.items()}
    out |= {f"api.{k}": v for k, v in api.items()}
    out |= {f"rl7.{k}": v for k, v in rl7_raw_seeds().items()}
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
    assert min(ranges()["rl7.train"]) >= floor


# -- Part 7 (D70) ------------------------------------------------------------------------
def test_rl7_ranges_are_the_registered_ones():
    assert {k: tuple(v) for k, v in rl7_raw_seeds().items()} == RL7_SEEDS


def test_rl_yaml_is_the_frozen_part6_record():
    assert text_sha256(RL_CONFIG).startswith(RL_YAML_SHA8)
    assert {k: tuple(v) for k, v in load_config(RL_CONFIG)["seeds"].items()} == RL_SEEDS


def test_rl7_inherits_burned_and_dev_test_from_rl_yaml():
    merged = {k: tuple(v) for k, v in load_config(RL7_CONFIG)["seeds"].items()}
    inherited = {k: v for k, v in RL_SEEDS.items() if k not in RL7_SEEDS}
    assert set(inherited) == {"burned", "dev_test"}
    assert merged == RL7_SEEDS | inherited


def test_rl7_devtest_look_is_inside_dev_test():
    rl7 = load_config(RL7_CONFIG)
    look = tuple(rl7["protocol"]["devtest"]["seeds"])
    assert look == RL7_DEVTEST_LOOK
    lo, hi = RL_SEEDS["dev_test"]
    assert lo <= look[0] <= look[1] <= hi
    full = rl7["protocol"]["holdout"]["full"]
    h_lo, h_hi = RL7_SEEDS["holdout"]
    assert h_lo <= full[0] <= full[1] <= h_hi


@pytest.mark.parametrize("path", sorted(RL7_CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_rl7_run_configs_keep_the_seed_map(path):
    """Run configs carry only the keys they change (D69): never their own seeds."""
    with open(path, encoding="utf-8") as f:
        assert "seeds" not in yaml.safe_load(f)
    assert load_config(path)["seeds"] == load_config(RL7_CONFIG)["seeds"]


def test_rl7_run_configs_are_the_registered_ones():
    """One file per stage-T config and sensitivity setting (D73, D76), each naming
    itself as its `variant`."""
    protocol = load_config(RL7_CONFIG)["protocol"]
    names = set(protocol["stage_t"]["configs"]) | set(protocol["sensitivity"]["settings"])
    assert {p.stem for p in RL7_CONFIG_DIR.glob("*.yaml")} == names
    for name in names:
        assert load_config(RL7_CONFIG_DIR / f"{name}.yaml")["variant"] == name
