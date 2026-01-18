use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;
use std::collections::HashMap;

mod tree;
mod pruning;
mod distance;
mod assembly;

use tree::Tree;
use pruning::felsenstein_pruning;
use distance::{jaccard_distance_matrix, hamming_distance_matrix};
use assembly::{AssemblyState, AssemblyBaseline, AssemblyConstraint, SimulationConfig, ContextClassConfig, FounderBiasConfig, CacheConfig, CacheManager, CacheStatus, InvalidationReason, TopologyGuard};

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

/// Simulate assembly trajectories in parallel and return path statistics.
///
/// # Arguments
/// * `primitives` - List of primitive building blocks
/// * `initial_parts` - Parts for the initial state
/// * `theta` - Feature weights (constraint parameters)
/// * `n_samples` - Number of trajectories to simulate
/// * `t_max` - Maximum simulation time
/// * `burn_in` - Burn-in time
/// * `max_depth` - Maximum assembly depth
/// * `seed` - RNG seed
/// * `kappa` - Baseline rate constant
/// * `join_exponent` - Join rate exponent
/// * `split_exponent` - Split rate exponent
/// * `decay_rate` - Decay rate
/// * `depth_gate_threshold` - Optional depth threshold for non-stationarity (Symmetry Break A)
/// * `depth_gate_theta` - Reuse modifier strength when depth >= threshold
///
/// # Returns
/// * List of dicts, each containing path statistics
#[pyfunction]
#[pyo3(signature = (primitives, initial_parts, theta, n_samples, t_max, burn_in, max_depth, seed, kappa=1.0, join_exponent=-0.5, split_exponent=0.3, decay_rate=0.01, depth_gate_threshold=None, depth_gate_theta=0.0, context_class_config=None, founder_bias_config=None))]
fn simulate_assembly_trajectories<'py>(
    py: Python<'py>,
    primitives: Vec<String>,
    initial_parts: Vec<String>,
    theta: &PyDict,
    n_samples: usize,
    t_max: f64,
    burn_in: f64,
    max_depth: u32,
    seed: u64,
    kappa: f64,
    join_exponent: f64,
    split_exponent: f64,
    decay_rate: f64,
    depth_gate_threshold: Option<u32>,
    depth_gate_theta: f64,
    context_class_config: Option<&PyDict>,
    founder_bias_config: Option<&PyDict>,
) -> PyResult<Vec<PyObject>> {
    // Convert theta from PyDict to HashMap
    let mut theta_map: HashMap<String, f64> = HashMap::new();
    for (key, value) in theta.iter() {
        let k: String = key.extract()?;
        let v: f64 = value.extract()?;
        theta_map.insert(k, v);
    }

    // Build components
    let baseline = AssemblyBaseline::new(kappa, join_exponent, split_exponent, decay_rate);
    let constraint = AssemblyConstraint::new(theta_map.clone());
    let context_config = match context_class_config {
        Some(cfg) => {
            let primitive_classes = match cfg.get_item("primitive_classes")? {
                Some(item) => item.extract::<HashMap<String, String>>()?,
                None => HashMap::new(),
            };

            let same_class_theta = match cfg.get_item("same_class_theta")? {
                Some(value) => value.extract::<f64>()?,
                None => 0.0,
            };

            let cross_class_theta = match cfg.get_item("cross_class_theta")? {
                Some(value) => value.extract::<f64>()?,
                None => 0.0,
            };

            Some(ContextClassConfig {
                primitive_classes,
                same_class_theta,
                cross_class_theta,
            })
        }
        None => None,
    };

    let founder_config = match founder_bias_config {
        Some(cfg) => {
            let founder_rank_threshold = match cfg.get_item("founder_rank_threshold")? {
                Some(value) => value.extract::<u32>()?,
                None => 1,
            };

            let founder_bonus_theta = match cfg.get_item("founder_bonus_theta")? {
                Some(value) => value.extract::<f64>()?,
                None => 0.0,
            };

            let late_penalty_theta = match cfg.get_item("late_penalty_theta")? {
                Some(value) => value.extract::<f64>()?,
                None => 0.0,
            };

            Some(FounderBiasConfig {
                founder_rank_threshold,
                founder_bonus_theta,
                late_penalty_theta,
            })
        }
        None => None,
    };

    let config = SimulationConfig {
        t_max,
        burn_in,
        max_depth,
        min_rate_threshold: 1e-6,
        primitives,
        depth_gate_threshold,
        depth_gate_theta,
        context_class_config: context_config,
        founder_bias_config: founder_config,
    };

    // Create initial state
    let initial_parts_refs: Vec<&str> = initial_parts.iter().map(|s| s.as_str()).collect();
    let initial_state = AssemblyState::new(&initial_parts_refs, 0, None);

    // Simulate
    let path_stats = assembly::simulate_trajectories_parallel(
        &baseline,
        &constraint,
        &config,
        &initial_state,
        n_samples,
        seed,
    );

    // Convert results to Python dicts
    let results: Vec<PyObject> = path_stats
        .into_iter()
        .map(|ps| {
            let dict = PyDict::new(py);
            
            // Feature counts
            let counts_dict = PyDict::new(py);
            for (k, v) in &ps.feature_counts {
                counts_dict.set_item(k, v).unwrap();
            }
            dict.set_item("feature_counts", counts_dict).unwrap();
            
            dict.set_item("final_state_id", ps.final_state_id).unwrap();
            dict.set_item("duration", ps.duration).unwrap();
            dict.set_item("n_transitions", ps.n_transitions).unwrap();
            dict.set_item("max_depth_reached", ps.max_depth_reached).unwrap();
            let reuse_count = *ps.feature_counts.get("reuse_count").unwrap_or(&0);
            dict.set_item("reuse_count", reuse_count).unwrap();
            dict.set_item("founder_rank", ps.founder_rank).unwrap();
            dict.set_item("first_visit_time", ps.first_visit_time).unwrap();

            dict.into()
        })
        .collect();

    Ok(results)
}

/// Compute importance weights for cached paths at a new theta.
///
/// # Arguments
/// * `path_feature_counts` - List of feature count dicts from simulate_assembly_trajectories
/// * `theta` - New feature weights
/// * `theta_ref` - Reference feature weights (from simulation)
///
/// # Returns
/// * Tuple of (normalized_weights, effective_sample_size)
#[pyfunction]
fn compute_importance_weights<'py>(
    py: Python<'py>,
    path_feature_counts: Vec<&PyDict>,
    theta: &PyDict,
    theta_ref: &PyDict,
) -> PyResult<(&'py PyArray1<f64>, f64)> {
    // Convert theta dicts
    let theta_map: HashMap<String, f64> = theta
        .iter()
        .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<f64>()?)))
        .collect::<PyResult<_>>()?;
    
    let theta_ref_map: HashMap<String, f64> = theta_ref
        .iter()
        .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<f64>()?)))
        .collect::<PyResult<_>>()?;

    // Convert path feature counts
    let paths: Vec<HashMap<String, u32>> = path_feature_counts
        .iter()
        .map(|d| {
            d.iter()
                .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<u32>()?)))
                .collect::<PyResult<HashMap<String, u32>>>()
        })
        .collect::<PyResult<_>>()?;

    if paths.is_empty() {
        return Ok((PyArray1::from_vec(py, vec![]), 0.0));
    }

    // Compute log weights
    let log_weights: Vec<f64> = paths
        .iter()
        .map(|fc| {
            let mut log_w = 0.0;
            for (feature, &count) in fc {
                let theta_k = theta_map.get(feature).unwrap_or(&0.0);
                let theta_k_ref = theta_ref_map.get(feature).unwrap_or(&0.0);
                log_w += (count as f64) * (theta_k - theta_k_ref);
            }
            log_w
        })
        .collect();

    // Log-sum-exp normalization
    let max_log_w = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_log_w.is_infinite() {
        let n = paths.len();
        let uniform = vec![1.0 / n as f64; n];
        return Ok((PyArray1::from_vec(py, uniform), n as f64));
    }

    let sum_exp: f64 = log_weights.iter().map(|&lw| (lw - max_log_w).exp()).sum();
    let log_sum = max_log_w + sum_exp.ln();

    let weights: Vec<f64> = log_weights
        .iter()
        .map(|&lw| (lw - log_sum).exp())
        .collect();

    // ESS = 1 / Σ w_i²
    let sum_sq: f64 = weights.iter().map(|&w| w * w).sum();
    let ess = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };

    Ok((PyArray1::from_vec(py, weights), ess))
}

/// Evaluate cache validity for a given theta.
///
/// Implements the three safety valves:
/// 1. L∞ trust region check
/// 2. Weight variance explosion (early warning)  
/// 3. ESS threshold check
///
/// # Arguments
/// * `path_feature_counts` - List of feature count dicts from simulate_assembly_trajectories
/// * `theta` - New feature weights to evaluate
/// * `theta_ref` - Reference feature weights (from simulation)
/// * `trust_radius` - L∞ trust region radius (default: 1.0)
/// * `ess_threshold` - ESS threshold ratio (default: 0.3)
/// * `max_weight_variance` - Maximum weight variance (default: 100.0)
///
/// # Returns
/// * Dict with keys: "valid" (bool), "ess" (float), "reason" (str or None),
///   "latent_states" (dict[state_id -> prob] if valid)
#[pyfunction]
#[pyo3(signature = (path_feature_counts, final_state_ids, theta, theta_ref, trust_radius=1.0, ess_threshold=0.3, max_weight_variance=100.0))]
fn evaluate_cache<'py>(
    py: Python<'py>,
    path_feature_counts: Vec<&PyDict>,
    final_state_ids: Vec<u64>,
    theta: &PyDict,
    theta_ref: &PyDict,
    trust_radius: f64,
    ess_threshold: f64,
    max_weight_variance: f64,
) -> PyResult<PyObject> {
    use assembly::PathStats;

    // Convert theta dicts
    let theta_map: HashMap<String, f64> = theta
        .iter()
        .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<f64>()?)))
        .collect::<PyResult<_>>()?;

    let theta_ref_map: HashMap<String, f64> = theta_ref
        .iter()
        .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<f64>()?)))
        .collect::<PyResult<_>>()?;

    // Build PathStats from Python data
    let paths: Vec<PathStats> = path_feature_counts
        .iter()
        .zip(final_state_ids.iter())
        .map(|(fc_dict, &state_id)| {
            let feature_counts: HashMap<String, u32> = fc_dict
                .iter()
                .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<u32>()?)))
                .collect::<PyResult<_>>()?;
            
            let mut ps = PathStats::new(state_id, 0.0);
            ps.feature_counts = feature_counts;
            Ok(ps)
        })
        .collect::<PyResult<_>>()?;

    // Create cache manager
    let config = CacheConfig {
        trust_radius,
        ess_threshold,
        max_weight_variance,
    };
    let cache = CacheManager::new(paths, theta_ref_map, config);

    // Evaluate
    let result = PyDict::new(py);
    match cache.evaluate(&theta_map) {
        CacheStatus::Valid { latent_states, ess } => {
            result.set_item("valid", true)?;
            result.set_item("ess", ess)?;
            result.set_item("reason", py.None())?;
            
            let states_dict = PyDict::new(py);
            for (state_id, prob) in latent_states {
                states_dict.set_item(state_id, prob)?;
            }
            result.set_item("latent_states", states_dict)?;
        }
        CacheStatus::Invalid { reason } => {
            result.set_item("valid", false)?;
            result.set_item("ess", py.None())?;
            result.set_item("reason", format!("{}", reason))?;
            result.set_item("latent_states", py.None())?;
        }
    }

    Ok(result.into())
}

/// Check for topology-changing theta features.
///
/// # Arguments
/// * `theta` - Current feature weights
/// * `theta_ref` - Reference feature weights
/// * `sensitive_features` - List of feature names that affect graph topology
/// * `threshold` - Change threshold for triggering resimulation (default: 2.0)
///
/// # Returns
/// * List of affected feature names, or empty list if topology preserved
#[pyfunction]
#[pyo3(signature = (theta, theta_ref, sensitive_features, threshold=2.0))]
fn check_topology_change(
    theta: &PyDict,
    theta_ref: &PyDict,
    sensitive_features: Vec<String>,
    threshold: f64,
) -> PyResult<Vec<String>> {
    let theta_map: HashMap<String, f64> = theta
        .iter()
        .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<f64>()?)))
        .collect::<PyResult<_>>()?;

    let theta_ref_map: HashMap<String, f64> = theta_ref
        .iter()
        .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<f64>()?)))
        .collect::<PyResult<_>>()?;

    let sensitive_refs: Vec<&str> = sensitive_features.iter().map(|s| s.as_str()).collect();
    let guard = TopologyGuard::new(&sensitive_refs, threshold, None);

    match guard.check(&theta_map, &theta_ref_map) {
        Some(affected) => Ok(affected),
        None => Ok(vec![]),
    }
}

/// Python module definition
#[pymodule]
fn persiste_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_likelihoods_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(compute_jaccard_distance, m)?)?;
    m.add_function(wrap_pyfunction!(compute_hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_assembly_trajectories, m)?)?;
    m.add_function(wrap_pyfunction!(compute_importance_weights, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_cache, m)?)?;
    m.add_function(wrap_pyfunction!(check_topology_change, m)?)?;
    Ok(())
}
