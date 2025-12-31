use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;

mod tree;
mod pruning;
mod distance;

use tree::Tree;
use pruning::felsenstein_pruning;
use distance::{jaccard_distance_matrix, hamming_distance_matrix};

/// Compute log-likelihoods for all gene families in parallel.
///
/// This is the main entry point from Python. It parallelizes the per-family
/// likelihood computation using Rayon, providing 5-10x speedup on multi-core CPUs.
///
/// # Arguments
/// * `parent_indices` - Parent index for each node (-1 for root)
/// * `branch_lengths` - Branch length for each node
/// * `presence_matrix` - (n_tips, n_families) binary presence/absence matrix
/// * `gain_rates` - (n_families,) gain rates for each family
/// * `loss_rates` - (n_families,) loss rates for each family
/// * `n_tips` - Number of tip nodes
///
/// # Returns
/// * Vector of log-likelihoods, one per family
#[pyfunction]
fn compute_likelihoods_parallel<'py>(
    py: Python<'py>,
    parent_indices: PyReadonlyArray1<i32>,
    branch_lengths: PyReadonlyArray1<f64>,
    presence_matrix: PyReadonlyArray2<i8>,
    gain_rates: PyReadonlyArray1<f64>,
    loss_rates: PyReadonlyArray1<f64>,
    n_tips: usize,
) -> PyResult<&'py PyArray1<f64>> {
    // Build tree from structure
    let tree = Tree::from_structure(
        parent_indices.as_array().to_vec(),
        branch_lengths.as_array().to_vec(),
        n_tips,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    // Get array views
    let presence = presence_matrix.as_array();
    let gains = gain_rates.as_array();
    let losses = loss_rates.as_array();
    
    let n_families = presence.shape()[1];
    
    // Parallel computation over families
    let log_likelihoods: Vec<f64> = (0..n_families)
        .into_par_iter()
        .map(|fam_idx| {
            // Extract presence data for this family
            let family_data: Vec<i8> = (0..presence.shape()[0])
                .map(|tip_idx| presence[[tip_idx, fam_idx]])
                .collect();
            
            // Compute likelihood for this family
            felsenstein_pruning(
                &tree,
                &family_data,
                gains[fam_idx],
                losses[fam_idx],
            )
        })
        .collect();
    
    // Convert to numpy array
    Ok(PyArray1::from_vec(py, log_likelihoods))
}

/// Compute Jaccard distance matrix from presence/absence matrix.
///
/// # Arguments
/// * `pam` - (n_taxa, n_genes) binary presence/absence matrix
///
/// # Returns
/// * (n_taxa, n_taxa) symmetric distance matrix
#[pyfunction]
fn compute_jaccard_distance<'py>(
    py: Python<'py>,
    pam: PyReadonlyArray2<u8>,
) -> PyResult<&'py PyArray2<f64>> {
    let pam_view = pam.as_array();
    let distances = jaccard_distance_matrix(pam_view);
    Ok(PyArray2::from_owned_array(py, distances))
}

/// Compute Hamming distance matrix from presence/absence matrix.
///
/// # Arguments
/// * `pam` - (n_taxa, n_genes) binary presence/absence matrix
///
/// # Returns
/// * (n_taxa, n_taxa) symmetric distance matrix
#[pyfunction]
fn compute_hamming_distance<'py>(
    py: Python<'py>,
    pam: PyReadonlyArray2<u8>,
) -> PyResult<&'py PyArray2<f64>> {
    let pam_view = pam.as_array();
    let distances = hamming_distance_matrix(pam_view);
    Ok(PyArray2::from_owned_array(py, distances))
}

/// Python module definition
#[pymodule]
fn persiste_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_likelihoods_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(compute_jaccard_distance, m)?)?;
    m.add_function(wrap_pyfunction!(compute_hamming_distance, m)?)?;
    Ok(())
}
