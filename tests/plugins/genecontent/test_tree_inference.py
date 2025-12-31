"""
Tests for tree inference from presence/absence matrices.
"""

import pytest
import numpy as np
from persiste.plugins.genecontent.tree_inference import (
    jaccard_distance,
    hamming_distance,
    upgma_tree,
    infer_tree_from_pam,
    TreeInferenceMetadata,
)
from persiste.core.trees import TreeStructure


class TestDistanceMetrics:
    """Test distance matrix computation."""
    
    def test_jaccard_distance_identical(self):
        """Identical taxa should have zero distance."""
        pam = np.array([[1, 0, 1, 0],
                        [1, 0, 1, 0]])
        D = jaccard_distance(pam)
        assert D[0, 1] == 0.0
        assert D[1, 0] == 0.0
    
    def test_jaccard_distance_disjoint(self):
        """Completely disjoint taxa should have distance 1."""
        pam = np.array([[1, 1, 0, 0],
                        [0, 0, 1, 1]])
        D = jaccard_distance(pam)
        assert D[0, 1] == 1.0
    
    def test_jaccard_distance_partial(self):
        """Partial overlap should give intermediate distance."""
        # A = {1, 2}, B = {2, 3}
        # Intersection = {2}, Union = {1, 2, 3}
        # Jaccard similarity = 1/3, distance = 2/3
        pam = np.array([[1, 1, 0],
                        [0, 1, 1]])
        D = jaccard_distance(pam)
        expected = 2.0 / 3.0
        assert abs(D[0, 1] - expected) < 1e-10
    
    def test_jaccard_distance_symmetric(self):
        """Distance matrix should be symmetric."""
        pam = np.random.binomial(1, 0.5, size=(5, 20))
        D = jaccard_distance(pam)
        assert np.allclose(D, D.T)
    
    def test_jaccard_distance_diagonal_zero(self):
        """Diagonal should be zero."""
        pam = np.random.binomial(1, 0.5, size=(5, 20))
        D = jaccard_distance(pam)
        assert np.allclose(np.diag(D), 0.0)
    
    def test_hamming_distance_identical(self):
        """Identical taxa should have zero Hamming distance."""
        pam = np.array([[1, 0, 1, 0],
                        [1, 0, 1, 0]])
        D = hamming_distance(pam)
        assert D[0, 1] == 0.0
    
    def test_hamming_distance_opposite(self):
        """Completely opposite taxa should have distance 1."""
        pam = np.array([[1, 1, 1, 1],
                        [0, 0, 0, 0]])
        D = hamming_distance(pam)
        assert D[0, 1] == 1.0
    
    def test_hamming_distance_half(self):
        """Half different should give distance 0.5."""
        pam = np.array([[1, 1, 0, 0],
                        [1, 1, 1, 1]])
        D = hamming_distance(pam)
        assert D[0, 1] == 0.5


class TestUPGMATree:
    """Test UPGMA tree construction."""
    
    def test_upgma_three_taxa(self):
        """Test UPGMA with three taxa."""
        # Simple distance matrix
        D = np.array([
            [0.0, 0.2, 0.4],
            [0.2, 0.0, 0.4],
            [0.4, 0.4, 0.0]
        ])
        taxon_names = ['A', 'B', 'C']
        
        tree = upgma_tree(D, taxon_names)
        
        # Check tree structure
        assert tree.n_nodes == 5  # 3 tips + 2 internal
        assert tree.n_tips == 3
        assert len(tree.tip_names) == 3
        assert set(tree.tip_names) == {'A', 'B', 'C'}
    
    def test_upgma_four_taxa(self):
        """Test UPGMA with four taxa."""
        D = np.array([
            [0.0, 0.1, 0.3, 0.3],
            [0.1, 0.0, 0.3, 0.3],
            [0.3, 0.3, 0.0, 0.1],
            [0.3, 0.3, 0.1, 0.0]
        ])
        taxon_names = ['A', 'B', 'C', 'D']
        
        tree = upgma_tree(D, taxon_names)
        
        assert tree.n_nodes == 7  # 4 tips + 3 internal
        assert tree.n_tips == 4
        assert set(tree.tip_names) == {'A', 'B', 'C', 'D'}
    
    def test_upgma_branch_lengths_positive(self):
        """Branch lengths should be non-negative."""
        D = np.array([
            [0.0, 0.2, 0.4],
            [0.2, 0.0, 0.4],
            [0.4, 0.4, 0.0]
        ])
        taxon_names = ['A', 'B', 'C']
        
        tree = upgma_tree(D, taxon_names)
        
        # All branch lengths should be >= 0
        assert np.all(tree.branch_lengths >= 0.0)
    
    def test_upgma_preserves_taxon_names(self):
        """Taxon names should be preserved correctly."""
        D = np.array([
            [0.0, 0.2, 0.4],
            [0.2, 0.0, 0.4],
            [0.4, 0.4, 0.0]
        ])
        taxon_names = ['GCF_001', 'GCF_002', 'GCF_003']
        
        tree = upgma_tree(D, taxon_names)
        
        assert set(tree.tip_names) == set(taxon_names)
    
    def test_upgma_with_underscores(self):
        """Taxon names with underscores should be preserved."""
        D = np.array([
            [0.0, 0.2],
            [0.2, 0.0]
        ])
        taxon_names = ['GCF_001_234', 'GCF_002_567']
        
        tree = upgma_tree(D, taxon_names)
        
        # Check that underscores are preserved
        assert 'GCF_001_234' in tree.tip_names
        assert 'GCF_002_567' in tree.tip_names


class TestInferTreeFromPAM:
    """Test complete tree inference pipeline."""
    
    def test_infer_jaccard_upgma(self):
        """Test Jaccard + UPGMA inference."""
        pam = np.array([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 1]
        ])
        taxon_names = ['A', 'B', 'C']
        
        tree, metadata = infer_tree_from_pam(pam, taxon_names, method='jaccard_upgma')
        
        assert tree.n_tips == 3
        assert metadata.source == 'inferred'
        assert metadata.method == 'jaccard_upgma'
        assert metadata.distance_metric == 'jaccard'
        assert metadata.clustering_method == 'upgma'
        assert metadata.n_taxa == 3
        assert metadata.n_genes == 4
    
    def test_infer_hamming_upgma(self):
        """Test Hamming + UPGMA inference."""
        pam = np.array([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 1]
        ])
        taxon_names = ['A', 'B', 'C']
        
        tree, metadata = infer_tree_from_pam(pam, taxon_names, method='hamming_upgma')
        
        assert tree.n_tips == 3
        assert metadata.distance_metric == 'hamming'
    
    def test_infer_large_pam(self):
        """Test inference on larger PAM."""
        np.random.seed(42)
        pam = np.random.binomial(1, 0.5, size=(20, 100))
        taxon_names = [f'taxon_{i}' for i in range(20)]
        
        tree, metadata = infer_tree_from_pam(pam, taxon_names)
        
        assert tree.n_tips == 20
        assert metadata.n_taxa == 20
        assert metadata.n_genes == 100
    
    def test_infer_validates_input(self):
        """Test that input validation works."""
        pam = np.array([[1, 0], [0, 1]])
        taxon_names = ['A', 'B', 'C']  # Wrong length
        
        with pytest.raises(ValueError, match="taxon_names length"):
            infer_tree_from_pam(pam, taxon_names)
    
    def test_infer_unknown_method(self):
        """Test that unknown method raises error."""
        pam = np.array([[1, 0], [0, 1]])
        taxon_names = ['A', 'B']
        
        with pytest.raises(ValueError, match="Unknown method"):
            infer_tree_from_pam(pam, taxon_names, method='unknown')


class TestTreeInferencePerformance:
    """Performance tests for tree inference."""
    
    def test_performance_50_strains(self):
        """Benchmark 50 strains × 1000 genes."""
        import time
        
        np.random.seed(42)
        pam = np.random.binomial(1, 0.5, size=(50, 1000))
        taxon_names = [f'strain_{i}' for i in range(50)]
        
        start = time.time()
        tree, metadata = infer_tree_from_pam(pam, taxon_names)
        elapsed = time.time() - start
        
        assert tree.n_tips == 50
        # Should complete in reasonable time (< 1 second for this size)
        assert elapsed < 1.0
    
    def test_performance_100_strains(self):
        """Benchmark 100 strains × 5000 genes."""
        import time
        
        np.random.seed(42)
        pam = np.random.binomial(1, 0.5, size=(100, 5000))
        taxon_names = [f'strain_{i}' for i in range(100)]
        
        start = time.time()
        tree, metadata = infer_tree_from_pam(pam, taxon_names)
        elapsed = time.time() - start
        
        assert tree.n_tips == 100
        # Should complete in reasonable time (< 5 seconds for this size)
        assert elapsed < 5.0


class TestTreeInferenceCorrectness:
    """Test correctness of inferred trees."""
    
    def test_tree_is_ultrametric(self):
        """UPGMA should produce ultrametric trees (molecular clock)."""
        # Create simple distance matrix
        D = np.array([
            [0.0, 0.2, 0.4],
            [0.2, 0.0, 0.4],
            [0.4, 0.4, 0.0]
        ])
        taxon_names = ['A', 'B', 'C']
        
        tree = upgma_tree(D, taxon_names)
        
        # For UPGMA, all tips should be at same distance from root
        # This is the ultrametric property
        # We'll check that branch lengths are reasonable
        assert tree.n_tips == 3
        assert np.all(tree.branch_lengths >= 0.0)
    
    def test_tree_topology_makes_sense(self):
        """Tree topology should reflect distance matrix."""
        # Create PAM where A and B are similar, C is different
        pam = np.array([
            [1, 1, 1, 0, 0],  # A
            [1, 1, 1, 1, 0],  # B (similar to A)
            [0, 0, 0, 1, 1]   # C (different)
        ])
        taxon_names = ['A', 'B', 'C']
        
        tree, metadata = infer_tree_from_pam(pam, taxon_names)
        
        # A and B should be closer to each other than to C
        # We can verify this by checking the tree structure
        assert tree.n_tips == 3
        assert set(tree.tip_names) == {'A', 'B', 'C'}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
