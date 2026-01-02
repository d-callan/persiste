import numpy as np

from persiste.core.trees import TreeStructure
from persiste.plugins.genecontent.constraints.gene_constraint import (
    ConstraintEffect,
    GeneContentConstraint,
)
from persiste.plugins.genecontent.inference.gene_inference import (
    GeneContentData,
    GeneContentModel,
)


def make_star_tree():
    # Simple 2-tip tree with unit branches
    newick = "(A:1.0,B:1.0);"
    return TreeStructure.from_newick(newick, backend="simple")


def make_data():
    tree = make_star_tree()
    presence_matrix = np.array([[1, 0], [0, 1]], dtype=np.int8)
    taxon_names = ["A", "B"]
    family_names = ["fam1", "fam2"]
    return GeneContentData(tree, presence_matrix, taxon_names, family_names)


def test_genecontent_model_log_likelihood_without_constraint():
    data = make_data()
    model = GeneContentModel(data, use_rust=False)
    params = {"log_gain": np.log(0.5), "log_loss": np.log(0.5)}
    ll = model.log_likelihood(params)
    assert np.isfinite(ll)


class ToyConstraint(GeneContentConstraint):
    def __init__(self):
        self.called = False

    def get_effect(self, family_id, context=None):
        self.called = True
        return ConstraintEffect(gain_effect=0.1, loss_effect=-0.1, family_id=family_id)

    def get_parameters(self):
        return {}

    def set_parameters(self, params):
        pass

    def n_parameters(self):
        return 0


def test_genecontent_model_respects_constraint_effects():
    data = make_data()
    constraint = ToyConstraint()
    model = GeneContentModel(data, constraint=constraint, use_rust=False)
    params = {"log_gain": 0.0, "log_loss": 0.0}
    ll = model.log_likelihood(params)
    assert np.isfinite(ll)
    assert constraint.called
