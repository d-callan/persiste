//! Fast distance matrix computation for phylogenetic tree inference.
//!
//! Implements parallel computation of Jaccard and Hamming distances
//! from presence/absence matrices.

use ndarray::{Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Compute Jaccard distance between two binary vectors.
///
/// Jaccard distance = 1 - (intersection / union)
/// where intersection = number of positions where both are 1
/// and union = number of positions where at least one is 1
///
/// # Arguments
/// * `a` - First binary vector
/// * `b` - Second binary vector
///
/// # Returns
/// Jaccard distance in [0, 1]
#[inline]
fn jaccard_distance_pair(a: ArrayView1<u8>, b: ArrayView1<u8>) -> f64 {
    let mut intersection = 0u32;
    let mut union = 0u32;
    
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let a_present = ai != 0;
        let b_present = bi != 0;
        
        if a_present || b_present {
            union += 1;
            if a_present && b_present {
                intersection += 1;
            }
        }
    }
    
    if union == 0 {
        // Both vectors are all zeros - define distance as 0
        0.0
    } else {
        1.0 - (intersection as f64) / (union as f64)
    }
}

/// Compute Hamming distance between two binary vectors.
///
/// Hamming distance = proportion of positions that differ
///
/// # Arguments
/// * `a` - First binary vector
/// * `b` - Second binary vector
///
/// # Returns
/// Hamming distance in [0, 1]
#[inline]
fn hamming_distance_pair(a: ArrayView1<u8>, b: ArrayView1<u8>) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    
    let differences: u32 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| if ai != bi { 1 } else { 0 })
        .sum();
    
    differences as f64 / n as f64
}

/// Compute pairwise Jaccard distances for all pairs of taxa.
///
/// # Arguments
/// * `pam` - Presence/absence matrix (n_taxa × n_genes)
///
/// # Returns
/// Symmetric distance matrix (n_taxa × n_taxa)
///
/// # Performance
/// Uses parallel computation with rayon for O(n²m) complexity
/// where n = number of taxa, m = number of genes.
pub fn jaccard_distance_matrix(pam: ArrayView2<u8>) -> Array2<f64> {
    let n_taxa = pam.nrows();
    let mut distances = Array2::<f64>::zeros((n_taxa, n_taxa));
    
    // Compute upper triangle in parallel
    let indices: Vec<(usize, usize)> = (0..n_taxa)
        .flat_map(|i| (i+1..n_taxa).map(move |j| (i, j)))
        .collect();
    
    let dists: Vec<((usize, usize), f64)> = indices
        .par_iter()
        .map(|&(i, j)| {
            let row_i = pam.row(i);
            let row_j = pam.row(j);
            let dist = jaccard_distance_pair(row_i, row_j);
            ((i, j), dist)
        })
        .collect();
    
    // Fill matrix (symmetric)
    for ((i, j), dist) in dists {
        distances[[i, j]] = dist;
        distances[[j, i]] = dist;
    }
    
    distances
}

/// Compute pairwise Hamming distances for all pairs of taxa.
///
/// # Arguments
/// * `pam` - Presence/absence matrix (n_taxa × n_genes)
///
/// # Returns
/// Symmetric distance matrix (n_taxa × n_taxa)
pub fn hamming_distance_matrix(pam: ArrayView2<u8>) -> Array2<f64> {
    let n_taxa = pam.nrows();
    let mut distances = Array2::<f64>::zeros((n_taxa, n_taxa));
    
    // Compute upper triangle in parallel
    let indices: Vec<(usize, usize)> = (0..n_taxa)
        .flat_map(|i| (i+1..n_taxa).map(move |j| (i, j)))
        .collect();
    
    let dists: Vec<((usize, usize), f64)> = indices
        .par_iter()
        .map(|&(i, j)| {
            let row_i = pam.row(i);
            let row_j = pam.row(j);
            let dist = hamming_distance_pair(row_i, row_j);
            ((i, j), dist)
        })
        .collect();
    
    // Fill matrix (symmetric)
    for ((i, j), dist) in dists {
        distances[[i, j]] = dist;
        distances[[j, i]] = dist;
    }
    
    distances
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_jaccard_identical() {
        let a = array![1u8, 0, 1, 0];
        let b = array![1u8, 0, 1, 0];
        let dist = jaccard_distance_pair(a.view(), b.view());
        assert_eq!(dist, 0.0);
    }
    
    #[test]
    fn test_jaccard_disjoint() {
        let a = array![1u8, 1, 0, 0];
        let b = array![0u8, 0, 1, 1];
        let dist = jaccard_distance_pair(a.view(), b.view());
        assert_eq!(dist, 1.0);
    }
    
    #[test]
    fn test_jaccard_partial() {
        // A = {0, 1}, B = {1, 2}
        // Intersection = {1}, Union = {0, 1, 2}
        // Similarity = 1/3, Distance = 2/3
        let a = array![1u8, 1, 0];
        let b = array![0u8, 1, 1];
        let dist = jaccard_distance_pair(a.view(), b.view());
        assert!((dist - 2.0/3.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_jaccard_all_zeros() {
        let a = array![0u8, 0, 0, 0];
        let b = array![0u8, 0, 0, 0];
        let dist = jaccard_distance_pair(a.view(), b.view());
        assert_eq!(dist, 0.0);
    }
    
    #[test]
    fn test_hamming_identical() {
        let a = array![1u8, 0, 1, 0];
        let b = array![1u8, 0, 1, 0];
        let dist = hamming_distance_pair(a.view(), b.view());
        assert_eq!(dist, 0.0);
    }
    
    #[test]
    fn test_hamming_opposite() {
        let a = array![1u8, 1, 1, 1];
        let b = array![0u8, 0, 0, 0];
        let dist = hamming_distance_pair(a.view(), b.view());
        assert_eq!(dist, 1.0);
    }
    
    #[test]
    fn test_hamming_half() {
        let a = array![1u8, 1, 0, 0];
        let b = array![1u8, 1, 1, 1];
        let dist = hamming_distance_pair(a.view(), b.view());
        assert_eq!(dist, 0.5);
    }
    
    #[test]
    fn test_jaccard_matrix() {
        let pam = array![
            [1u8, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ];
        
        let dist = jaccard_distance_matrix(pam.view());
        
        // Check diagonal is zero
        assert_eq!(dist[[0, 0]], 0.0);
        assert_eq!(dist[[1, 1]], 0.0);
        assert_eq!(dist[[2, 2]], 0.0);
        
        // Check symmetry
        assert_eq!(dist[[0, 1]], dist[[1, 0]]);
        assert_eq!(dist[[0, 2]], dist[[2, 0]]);
        assert_eq!(dist[[1, 2]], dist[[2, 1]]);
        
        // Check specific values
        assert_eq!(dist[[0, 1]], 0.0); // Identical
        assert_eq!(dist[[0, 2]], 1.0); // Disjoint
    }
    
    #[test]
    fn test_hamming_matrix() {
        let pam = array![
            [1u8, 1, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0]
        ];
        
        let dist = hamming_distance_matrix(pam.view());
        
        // Check diagonal is zero
        assert_eq!(dist[[0, 0]], 0.0);
        
        // Check symmetry
        assert_eq!(dist[[0, 1]], dist[[1, 0]]);
        
        // Check specific values
        assert_eq!(dist[[0, 1]], 0.5); // Half different
        assert_eq!(dist[[1, 2]], 1.0); // Completely different
    }
}
