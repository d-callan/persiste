import numpy as np
import pytest
from scipy.special import gammaln

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.observation.counts_model import AssemblyCountsModel
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
    def test_counts_model_likelihood_matches_poisson_form(self):
        model = AssemblyCountsModel(detection_efficiency=0.9, background_noise=0.1)

        # Use stable IDs (ints) for latent states
        state_a_id = 101
        state_b_id = 102
        
        latent_states = {
            state_a_id: 0.6,
            state_b_id: 0.4,
        }
        observed_counts = {state_a_id: 10}

        log_lik = model.compute_log_likelihood(observed_counts, latent_states)

        total_units = 10.0
        lambda_a = total_units * 0.6 * model.detection_efficiency + model.background_noise
        expected = 10 * np.log(lambda_a) - lambda_a - gammaln(11)
        
        lambda_b = total_units * 0.4 * model.detection_efficiency + model.background_noise
        expected += -lambda_b # count=0 for state_b

        assert pytest.approx(log_lik, rel=1e-6) == expected
