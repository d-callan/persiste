"""
Test optional tree functionality for CopyNumberDynamics.

Verifies that trees can be inferred automatically from CN matrices.
"""

import numpy as np
import pytest

from persiste.plugins.copynumber.cn_interface import fit, fit_null_model


class TestOptionalTree:
    """Test that trees are optional and can be inferred from CN data."""
    
    def test_fit_without_tree(self):
        """fit() should work without providing a tree."""
        # Create simple CN matrix
        np.random.seed(42)
        cn_matrix = np.random.randint(0, 4, size=(10, 5))  # 10 families, 5 taxa
        
        family_names = [f"fam_{i}" for i in range(10)]
        taxon_names = [f"taxon_{i}" for i in range(5)]
        
        # Should work without tree parameter
        result = fit(
            cn_matrix=cn_matrix,
            family_names=family_names,
            taxon_names=taxon_names,
            # No tree parameter!
            baseline_type='global',
            verbose=False,
        )
        
        # Should produce valid result
        assert result is not None
        assert np.isfinite(result.log_likelihood)
        assert result.log_likelihood < 0  # Log-likelihood should be negative
    
    def test_fit_null_without_tree(self):
        """fit_null_model() should work without providing a tree."""
        np.random.seed(42)
        cn_matrix = np.random.randint(0, 4, size=(10, 5))
        
        family_names = [f"fam_{i}" for i in range(10)]
        taxon_names = [f"taxon_{i}" for i in range(5)]
        
        result = fit_null_model(
            cn_matrix=cn_matrix,
            family_names=family_names,
            taxon_names=taxon_names,
            # No tree parameter!
            baseline_type='global',
            verbose=False,
        )
        
        assert result is not None
        assert np.isfinite(result.log_likelihood)
    
    def test_tree_inference_methods(self):
        """Different tree inference methods should work."""
        np.random.seed(42)
        cn_matrix = np.random.randint(0, 4, size=(10, 5))
        
        family_names = [f"fam_{i}" for i in range(10)]
        taxon_names = [f"taxon_{i}" for i in range(5)]
        
        methods = ["jaccard_upgma", "hamming_upgma"]
        
        for method in methods:
            result = fit(
                cn_matrix=cn_matrix,
                family_names=family_names,
                taxon_names=taxon_names,
                tree_method=method,
                baseline_type='global',
                verbose=False,
            )
            
            assert result is not None
            assert np.isfinite(result.log_likelihood)
    
    def test_inferred_tree_has_correct_tips(self):
        """Inferred tree should have correct number of tips."""
        np.random.seed(42)
        n_taxa = 8
        cn_matrix = np.random.randint(0, 4, size=(10, n_taxa))
        
        family_names = [f"fam_{i}" for i in range(10)]
        taxon_names = [f"taxon_{i}" for i in range(n_taxa)]
        
        result = fit(
            cn_matrix=cn_matrix,
            family_names=family_names,
            taxon_names=taxon_names,
            baseline_type='global',
            verbose=False,
        )
        
        # Tree should have correct number of tips
        assert result.tree.n_tips == n_taxa
        
        # Tree should have correct tip names
        tree_tip_names = set(result.tree.tip_names)
        expected_names = set(taxon_names)
        assert tree_tip_names == expected_names
    
    def test_with_constraint_no_tree(self):
        """Constraints should work with inferred trees."""
        np.random.seed(42)
        cn_matrix = np.random.randint(0, 4, size=(15, 6))
        
        family_names = [f"fam_{i}" for i in range(15)]
        taxon_names = [f"taxon_{i}" for i in range(6)]
        
        # Fit with dosage stability constraint
        result = fit(
            cn_matrix=cn_matrix,
            family_names=family_names,
            taxon_names=taxon_names,
            constraint_type='dosage_stability',
            theta=-0.5,
            baseline_type='global',
            verbose=False,
        )
        
        assert result is not None
        assert result.constraint_type == 'dosage_stability'
        assert result.theta == -0.5
        assert np.isfinite(result.log_likelihood)
    
    def test_recipes_work_without_tree(self):
        """Recipe functions should work without trees."""
        from persiste.plugins.copynumber.recipes import null_cn_dynamics
        
        np.random.seed(42)
        cn_matrix = np.random.randint(0, 4, size=(20, 8))
        
        family_names = [f"fam_{i}" for i in range(20)]
        taxon_names = [f"taxon_{i}" for i in range(8)]
        
        # Recipe should work without tree
        report = null_cn_dynamics(
            cn_matrix=cn_matrix,
            family_names=family_names,
            taxon_names=taxon_names,
            # No tree parameter!
            baseline_type='global',
            verbose=False,
        )
        
        assert report is not None
        assert report.stationary_distribution is not None
        assert len(report.stationary_distribution) == 4  # 4 CN states
        assert np.isclose(np.sum(report.stationary_distribution), 1.0)


class TestTreeInferenceFromCore:
    """Test that tree inference from core works correctly."""
    
    def test_core_tree_building_import(self):
        """Should be able to import tree building from core."""
        from persiste.core.tree_building import (
            infer_tree_from_binary_matrix,
            jaccard_distance,
            hamming_distance,
            upgma_tree,
            TreeInferenceMetadata,
        )
        
        # All imports should succeed
        assert infer_tree_from_binary_matrix is not None
        assert jaccard_distance is not None
        assert hamming_distance is not None
        assert upgma_tree is not None
        assert TreeInferenceMetadata is not None
    
    def test_cn_to_pam_conversion(self):
        """CN matrix should convert to PAM correctly."""
        cn_matrix = np.array([
            [0, 1, 2, 0],
            [1, 0, 3, 1],
            [2, 2, 0, 0],
        ])
        
        # Convert to PAM (transpose to taxa × families)
        pam = (cn_matrix.T > 0).astype(int)
        
        expected_pam = np.array([
            [0, 1, 1],  # taxon 0
            [1, 0, 1],  # taxon 1
            [1, 1, 0],  # taxon 2
            [0, 1, 0],  # taxon 3
        ])
        
        np.testing.assert_array_equal(pam, expected_pam)
    
    def test_infer_tree_from_cn(self):
        """Should be able to infer tree from CN matrix using core."""
        from persiste.core.tree_building import infer_tree_from_binary_matrix
        
        np.random.seed(42)
        cn_matrix = np.random.randint(0, 4, size=(10, 6))
        taxon_names = [f"taxon_{i}" for i in range(6)]
        
        # Convert to PAM
        pam = (cn_matrix.T > 0).astype(int)
        
        # Infer tree
        tree, metadata = infer_tree_from_binary_matrix(
            binary_matrix=pam,
            taxon_names=taxon_names,
            method="jaccard_upgma",
        )
        
        assert tree is not None
        assert tree.n_tips == 6
        assert metadata.source == "inferred"
        assert metadata.method == "jaccard_upgma"
        assert metadata.n_taxa == 6
        assert metadata.n_features == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
