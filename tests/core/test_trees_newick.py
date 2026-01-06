import numpy as np

from persiste.core.trees import TreeStructure


def test_tree_to_newick_roundtrip_quotes_and_escapes_tip_names():
    # Include underscore (common in strain IDs) and a single quote (needs escaping)
    original_newick = "('A_strain':0.100000,'B\\'s':0.200000);"
    tree = TreeStructure.from_newick(original_newick, backend="simple")

    assert set(tree.tip_names) == {"A_strain", "B's"}

    serialized = tree.to_newick()

    # Default serializer quotes tips and omits root branch length.
    assert serialized.endswith(";")
    assert "'A_strain':0.100000" in serialized
    assert "'B\\'s':0.200000" in serialized
    assert "):" not in serialized

    tree2 = TreeStructure.from_newick(serialized, backend="simple")
    assert set(tree2.tip_names) == {"A_strain", "B's"}

    bl_by_name_1 = {tree.nodes[i].name: float(tree.branch_lengths[i]) for i in tree.tip_indices}
    bl_by_name_2 = {tree2.nodes[i].name: float(tree2.branch_lengths[i]) for i in tree2.tip_indices}

    for name in bl_by_name_1:
        assert np.isclose(bl_by_name_1[name], bl_by_name_2[name])
