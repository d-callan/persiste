"""Tests for phylogenetic tree wrapper."""

import sys


def test_tree_from_newick_string():
    """Test loading tree from Newick string."""
    print("Testing tree from Newick string...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    
    # Simple 4-taxon tree
    newick = "((A:0.1,B:0.2):0.3,(C:0.15,D:0.25):0.35):0.0;"
    
    tree = PhylogeneticTree.from_string(newick)
    
    assert tree.n_taxa == 4, f"Expected 4 taxa, got {tree.n_taxa}"
    assert set(tree.taxa) == {'A', 'B', 'C', 'D'}, f"Unexpected taxa: {tree.taxa}"
    
    # Should have 4 leaves + 3 internal nodes = 7 total
    assert tree.n_nodes == 7, f"Expected 7 nodes, got {tree.n_nodes}"
    
    # Check total tree length
    total_length = tree.total_tree_length()
    expected = 0.1 + 0.2 + 0.3 + 0.15 + 0.25 + 0.35  # All branch lengths
    assert abs(total_length - expected) < 1e-6, f"Expected length {expected}, got {total_length}"
    
    print(f"  ✓ Tree loaded: {tree}")
    print(f"  ✓ Taxa: {tree.taxa}")
    print(f"  ✓ Total length: {total_length:.4f}")


def test_tree_traversal():
    """Test tree traversal orders."""
    print("Testing tree traversal...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    
    newick = "((A:0.1,B:0.2):0.3,(C:0.15,D:0.25):0.35):0.0;"
    tree = PhylogeneticTree.from_string(newick)
    
    # Postorder: leaves first, root last
    postorder_labels = []
    for node in tree.postorder_nodes():
        label = tree.get_node_label(node)
        if label:
            postorder_labels.append(label)
    
    assert len(postorder_labels) == 4, "Should visit all 4 leaves in postorder"
    assert set(postorder_labels) == {'A', 'B', 'C', 'D'}
    
    # Count leaf vs internal nodes
    n_leaves = sum(1 for node in tree.leaf_nodes())
    n_internal = sum(1 for node in tree.internal_nodes())
    
    assert n_leaves == 4, f"Expected 4 leaves, got {n_leaves}"
    assert n_internal == 3, f"Expected 3 internal nodes, got {n_internal}"
    
    print(f"  ✓ Postorder visits all leaves: {postorder_labels}")
    print(f"  ✓ Leaf nodes: {n_leaves}, Internal nodes: {n_internal}")


def test_branch_lengths():
    """Test branch length access."""
    print("Testing branch lengths...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    
    newick = "((A:0.1,B:0.2):0.3,(C:0.15,D:0.25):0.35):0.0;"
    tree = PhylogeneticTree.from_string(newick)
    
    # Get branch lengths for leaves
    branch_lengths = {}
    for node in tree.leaf_nodes():
        label = tree.get_node_label(node)
        length = tree.get_branch_length(node)
        branch_lengths[label] = length
    
    assert abs(branch_lengths['A'] - 0.1) < 1e-6, f"A branch length should be 0.1, got {branch_lengths['A']}"
    assert abs(branch_lengths['B'] - 0.2) < 1e-6, f"B branch length should be 0.2, got {branch_lengths['B']}"
    assert abs(branch_lengths['C'] - 0.15) < 1e-6, f"C branch length should be 0.15, got {branch_lengths['C']}"
    assert abs(branch_lengths['D'] - 0.25) < 1e-6, f"D branch length should be 0.25, got {branch_lengths['D']}"
    
    print(f"  ✓ Branch lengths: {branch_lengths}")


def test_tree_manipulation():
    """Test tree manipulation (scaling, rerooting)."""
    print("Testing tree manipulation...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    
    newick = "((A:0.1,B:0.2):0.3,(C:0.15,D:0.25):0.35):0.0;"
    tree = PhylogeneticTree.from_string(newick)
    
    original_length = tree.total_tree_length()
    
    # Scale branches by 2
    scaled_tree = tree.scale_branches(2.0)
    scaled_length = scaled_tree.total_tree_length()
    
    assert abs(scaled_length - 2 * original_length) < 1e-6, \
        f"Scaled length should be 2x original: {scaled_length} vs {2 * original_length}"
    
    # Original tree should be unchanged
    assert abs(tree.total_tree_length() - original_length) < 1e-6, \
        "Original tree should not be modified"
    
    print(f"  ✓ Original length: {original_length:.4f}")
    print(f"  ✓ Scaled length (2x): {scaled_length:.4f}")
    print(f"  ✓ Tree manipulation preserves original")


def test_newick_roundtrip():
    """Test Newick string roundtrip."""
    print("Testing Newick roundtrip...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    
    newick = "((A:0.1,B:0.2):0.3,(C:0.15,D:0.25):0.35);"
    tree = PhylogeneticTree.from_string(newick)
    
    # Convert back to Newick
    newick_out = tree.to_newick()
    
    # Parse again
    tree2 = PhylogeneticTree.from_string(newick_out)
    
    # Should have same structure
    assert tree2.n_taxa == tree.n_taxa, "Taxa count should match"
    assert set(tree2.taxa) == set(tree.taxa), "Taxa should match"
    assert abs(tree2.total_tree_length() - tree.total_tree_length()) < 1e-6, \
        "Tree length should match"
    
    print(f"  ✓ Roundtrip successful")
    print(f"  ✓ Input:  {newick}")
    print(f"  ✓ Output: {newick_out}")


def main():
    """Run all tree tests."""
    print("=" * 60)
    print("PERSISTE Phylo Plugin - Tree Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_tree_from_newick_string,
        test_tree_traversal,
        test_branch_lengths,
        test_tree_manipulation,
        test_newick_roundtrip,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
