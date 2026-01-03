import numpy as np
import pytest
from scipy.special import gammaln

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.observation.presence_model import FrequencyWeightedPresenceModel
from persiste.plugins.assembly.states.assembly_state import AssemblyState


def make_primitive_state(symbol: str) -> AssemblyState:
    return AssemblyState.from_parts([symbol], depth=0)


class TestAssemblyConstraints:
    def test_constraint_scales_join_rate(self):
        primitives = ["A"]
        graph = AssemblyGraph(primitives, max_depth=2, min_rate_threshold=0.0)
        baseline = AssemblyBaseline(kappa=1.0)
        constraint = AssemblyConstraint(feature_weights={"reuse_count": 2.0})

        state = make_primitive_state("A")
        neighbors = graph.get_neighbors(state, baseline, constraint)
        join_neighbors = [n for n in neighbors if n[2].value == "join"]

        assert len(join_neighbors) == 1
        _, rate, _ = join_neighbors[0]

        # Baseline rate is 1.0 for this join; constraint should scale by exp(2)
        assert pytest.approx(rate, rel=1e-6) == np.exp(2.0)

    def test_constraint_neutral_without_weights(self):
        primitives = ["A"]
        graph = AssemblyGraph(primitives, max_depth=2, min_rate_threshold=0.0)
        baseline = AssemblyBaseline(kappa=1.0)
        constraint = AssemblyConstraint()  # null weights

        state = make_primitive_state("A")
        neighbors = graph.get_neighbors(state, baseline, constraint)
        join_neighbors = [n for n in neighbors if n[2].value == "join"]

        assert len(join_neighbors) == 1
        _, rate, _ = join_neighbors[0]

        assert rate == pytest.approx(1.0)


class TestAssemblyObservation:
    def test_frequency_weighted_presence_matches_poisson_form(self):
        model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.1)

        state_present = AssemblyState.from_parts(["A"], depth=1)
        state_absent = AssemblyState.from_parts(["B"], depth=1)
        latent_states = {
            state_present: 0.6,
            state_absent: 0.4,
        }
        observed_counts = {"A": 10}

        log_lik = model.compute_log_likelihood(observed_counts, latent_states)

        n_samples = 10
        lambda_a = n_samples * 0.6 * model.detection_prob + model.false_positive_rate
        expected = 10 * np.log(lambda_a) - lambda_a - gammaln(11)
        lambda_b = n_samples * 0.4 * model.detection_prob + model.false_positive_rate
        expected += -lambda_b

        assert pytest.approx(log_lik, rel=1e-6) == expected

    def test_predict_presence_matches_detection_formula(self):
        model = FrequencyWeightedPresenceModel(detection_prob=0.8, false_positive_rate=0.05)
        state_present = AssemblyState.from_parts(["C"], depth=1)
        latent_states = {state_present: 0.7}
        prob = model.predict_presence(latent_states, "C")
        expected = model.detection_prob * 0.7 + model.false_positive_rate * (1 - 0.7)
        assert pytest.approx(prob, rel=1e-9) == expected
