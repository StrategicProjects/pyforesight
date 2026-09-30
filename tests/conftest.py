import csv
import json
import math
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


def column(name, i):
    with open(DATA / name) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    return [float(r[i]) for r in rows[1:]]


def logs(values):
    return [math.log(v) for v in values]


def recorded(name):
    with open(DATA / name) as f:
        return json.load(f)


def near(ours, theirs, tolerance, what=""):
    """Relative difference, as in the Go and Rust editions."""
    if isinstance(theirs, (list, tuple)):
        assert len(ours) == len(theirs), f"{what}: {len(ours)} values, want {len(theirs)}"
        for i, (a, b) in enumerate(zip(ours, theirs)):
            near(a, b, tolerance, f"{what}[{i}]")
        return
    assert abs(ours - theirs) <= tolerance * max(abs(theirs), 1), f"{what}: {ours}, want {theirs}"


# what a search may differ by between platforms, and what arithmetic may
SEARCHED = 1e-5
COMPUTED = 1e-9


@pytest.fixture(scope="session")
def icms():
    return column("piaui_revenue.csv", 1)


@pytest.fixture(scope="session")
def fpe():
    return column("piaui_revenue.csv", 2)


@pytest.fixture(scope="session")
def ipca():
    return column("piaui_revenue.csv", 3)


@pytest.fixture(scope="session")
def air():
    return column("air_passengers.csv", 1)


@pytest.fixture(scope="session")
def series(icms, fpe, air):
    """name -> (values, first month, 1-based)"""
    return {"air": (air, 1), "icms": (icms, 3), "fpe": (fpe, 3)}
