import types

import numpy as np
import pytest

from persiste.core.data import ObservedTransitions
from persiste.core import inference as inference_mod
from persiste.core.inference import ConstraintInference, ConstraintResult


class DummyBaseline:
    def __init__(self, theta):
        self.theta = theta


class DummyConstraintModel:
    def __init__(self, theta0, allow_facilitation=True):
        self.theta0 = np.array(theta0, dtype=float)
        self.allow_facilitation = allow_facilitation
        self.constraint_structure = "per_transition"
        self.sparsity = None
        self.strength = 1.0

    def get_constrained_baseline(self, parameters=None):
        theta_vec = parameters["theta"] if parameters is not None else self.theta0
        return DummyBaseline(theta_vec)

    def initial_parameters(self):
        return self.theta0

    def unpack(self, theta_vec):
        return {"theta": np.array(theta_vec, dtype=float)}

    def num_free_parameters(self, parameters=None):
        if parameters is None:
            return len(self.theta0)
        theta = parameters.get("theta", [])
        return len(theta)


class DummyObservationModel:
    def __init__(self):
        self.graph = object()
        self.calls = []

    def log_likelihood(self, data, baseline, graph):
        self.calls.append((data, baseline, graph))
        theta = baseline.theta
        # Prefer theta close to 2.0 for first entry
        return -float((theta[0] - 2.0) ** 2)


@pytest.fixture
def observed_data():
    return ObservedTransitions.from_dict({(0, 1): 5, (1, 2): 3}, exposure=1.0)


def test_log_likelihood_passes_baseline_to_obs_model(observed_data):
    model = DummyConstraintModel(theta0=[1.0])
    obs_model = DummyObservationModel()
    inference = ConstraintInference(model, obs_model)

    params = {"theta": np.array([1.5])}
    ll = inference.log_likelihood(observed_data, params)

    assert ll == pytest.approx(-0.25)
    assert obs_model.calls, "Observation model should be invoked"
    _, baseline, graph = obs_model.calls[-1]
    assert isinstance(baseline, DummyBaseline)
    assert graph is obs_model.graph
    np.testing.assert_allclose(baseline.theta, [1.5])


def test_fit_uses_facilitation_bounds(monkeypatch, observed_data):
    model = DummyConstraintModel(theta0=[0.5, 0.5], allow_facilitation=False)
    obs_model = DummyObservationModel()
    inference = ConstraintInference(model, obs_model)

    captured = {}

    def fake_minimize(func, x0, bounds=None, options=None):
        captured["bounds"] = bounds
        captured["x0"] = x0
        # Return neutral parameters
        return types.SimpleNamespace(
            success=True,
            x=np.array([0.75, 0.25]),
            message="ok",
            nfev=5,
            fun=-1.0,
        )

    monkeypatch.setattr(inference_mod.optimize, "minimize", fake_minimize)

    result = inference.fit(observed_data)

    assert captured["bounds"] == [(0.0, 1.0), (0.0, 1.0)]
    np.testing.assert_allclose(result.parameters["theta"], [0.75, 0.25])
    assert result.metadata["success"] is True
    assert result.log_likelihood == pytest.approx(1.0)


def test_fit_raises_for_unknown_method(observed_data):
    model = DummyConstraintModel(theta0=[1.0])
    obs_model = DummyObservationModel()
    inference = ConstraintInference(model, obs_model)

    with pytest.raises(NotImplementedError):
        inference.fit(observed_data, method="MCMC")


def test_likelihood_ratio_test_statistics():
    model = DummyConstraintModel(theta0=[1.0, 1.0])

    null_result = ConstraintResult(
        model=model,
        parameters={"theta": np.array([1.0])},
        method="MLE",
        log_likelihood=-10.0,
    )
    alt_result = ConstraintResult(
        model=model,
        parameters={"theta": np.array([1.0, 0.5])},
        method="MLE",
        log_likelihood=-9.0,
    )

    obs_model = DummyObservationModel()
    inference = ConstraintInference(model, obs_model)

    test_result = inference.test(
        data=ObservedTransitions.from_dict({(0, 1): 2}, exposure=1.0),
        null_result=null_result,
        alternative_result=alt_result,
    )

    assert test_result.statistic == pytest.approx(2.0)
    assert test_result.metadata["df"] == 1
    assert 0.0 <= test_result.pvalue <= 1.0
