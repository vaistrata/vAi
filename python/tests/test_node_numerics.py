"""Every spec in node_cases.py, run through the same checks."""

import numpy as np
import pytest
from node_cases import SPECS, Run

IDS = [s.id for s in SPECS]
RESHAPED = [s for s in SPECS if s.alt_inputs]


def _check(spec, got, want, tol=None):
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    rtol, atol = tol or spec.tol
    if spec.out == "u32":
        np.testing.assert_array_equal(got, want.astype(np.uint32))
    elif spec.out == "f16":
        np.testing.assert_array_equal(got, want.astype(np.float16))
    else:
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_matches_reference(spec, rng):
    run = Run(spec, rng)
    _check(spec, run.forward(), run.expected())


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_second_forward_is_identical(spec, rng):
    """The second forward replays the recorded commands."""
    run = Run(spec, rng)
    np.testing.assert_array_equal(run.forward(), run.forward())


@pytest.mark.parametrize("spec", [s for s in SPECS if s.out != "u32"], ids=lambda s: s.id)
def test_output_is_finite(spec, rng):
    assert np.isfinite(Run(spec, rng).forward().astype(np.float64)).all()


@pytest.mark.parametrize("spec", RESHAPED, ids=[s.id for s in RESHAPED])
def test_new_shape_matches_reference(spec, rng):
    """A shape change re-prepares the node; the rerun must still be right."""
    run = Run(spec, rng)
    run.forward()
    run.draw(spec.alt_inputs, rng)
    _check(spec, run.forward(), run.expected(), spec.alt_tol)
