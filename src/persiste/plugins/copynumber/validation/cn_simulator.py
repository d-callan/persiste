"""
Simulation framework for copy number evolution.

Simulates CN evolution on a phylogenetic tree under specified models.
Used for validation and power analysis.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from persiste.core.trees import TreeStructure
from persiste.plugins.copynumber.baselines.cn_baseline import (
    GlobalBaseline,
    HierarchicalBaseline,
)
from persiste.plugins.copynumber.constraints.cn_constraints import (
    AmplificationBiasConstraint,
    DosageStabilityConstraint,
    HostConditionedVolatilityConstraint,
    apply_constraint,
)


class CNRegime(Enum):
    """Copy number evolutionary regimes (biological realism)."""
    STABLE_SINGLE = "stable_single"  # Housekeeping, essential genes
    VOLATILE_MULTI = "volatile_multi"  # Antigen families, variable genes
    RARE_AMPLIFYING = "rare_amplifying"  # Drug resistance, adaptive CNV


class SimulationScenario(Enum):
    """Predefined simulation scenarios for validation."""
    NULL = "null"
    DOSAGE_BUFFERING = "dosage_buffering"
    AMPLIFICATION_BIAS = "amplification_bias"
    HIGH_VOLATILITY_LINEAGE = "high_volatility_lineage"


@dataclass
class RegimeParams:
    """
    Parameters defining a CN evolutionary regime.
    
    Attributes:
        name: Regime name
        gain_rate: 0→1 rate
        loss_rate: 1→0 rate
        amplify_rate: 1→2, 2→3 rate
        contract_rate: 2→1, 3→2 rate
        proportion: Proportion of families in this regime
        description: Biological interpretation
    """
    name: str
    gain_rate: float
    loss_rate: float
    amplify_rate: float
    contract_rate: float
    proportion: float
    description: str


@dataclass
class SimulationConfig:
    """
    Configuration for CN evolution simulation.
    
    Attributes:
        scenario: Predefined scenario or custom
        baseline_type: 'global' or 'hierarchical'
        baseline_params: Parameters for baseline model
        constraint_type: Type of constraint (if any)
        theta: Constraint parameter (if any)
        root_state: Initial state at root (or None for stationary)
        n_families: Number of gene families to simulate
        seed: Random seed for reproducibility
        use_regimes: Whether to use regime-based heterogeneity
        regime_params: List of regime parameters (if use_regimes=True)
    """
    scenario: SimulationScenario
    baseline_type: str = 'global'
    baseline_params: dict | None = None
    constraint_type: str | None = None
    theta: float | None = None
    root_state: int | None = None
    n_families: int = 100
    seed: int = 42
    use_regimes: bool = False
    regime_params: list | None = None


def get_stationary_distribution(rate_matrix: np.ndarray) -> np.ndarray:
    """
    Compute stationary distribution of rate matrix.
    
    Solves πQ = 0 with Σπ = 1.
    
    Args:
        Q: (4, 4) rate matrix
    
    Returns:
        (4,) stationary distribution
    """
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(rate_matrix.T)

    # Find eigenvector corresponding to eigenvalue ≈ 0
    idx = np.argmin(np.abs(eigenvalues))
    stationary = np.real(eigenvectors[:, idx])

    # Normalize
    stationary = stationary / stationary.sum()

    # Ensure non-negative (numerical issues)
    stationary = np.maximum(stationary, 0)
    stationary = stationary / stationary.sum()

    return stationary


def simulate_along_branch(
    start_state: int,
    branch_length: float,
    rate_matrix: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """
    Simulate CN evolution along a single branch.
    
    Uses Gillespie algorithm for exact simulation.
    
    Args:
        start_state: Starting state (0-3)
        branch_length: Branch length in substitutions
        Q: (4, 4) rate matrix
        rng: Random number generator
    
    Returns:
        End state (0-3)
    """
    current_state = start_state
    current_time = 0.0

    while current_time < branch_length:
        # Get rate of leaving current state
        rate_out = -rate_matrix[current_state, current_state]

        if rate_out <= 0:
            # Absorbing state (shouldn't happen with valid Q)
            break

        # Sample waiting time
        waiting_time = rng.exponential(1.0 / rate_out)

        if current_time + waiting_time >= branch_length:
            # No more events on this branch
            break

        current_time += waiting_time

        # Sample destination state
        transition_rates = rate_matrix[current_state, :].copy()
        transition_rates[current_state] = 0  # Can't stay
        transition_probs = transition_rates / transition_rates.sum()

        current_state = rng.choice(4, p=transition_probs)

    return current_state


def simulate_on_tree(
    tree: TreeStructure,
    rate_matrix: np.ndarray,
    root_state: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, int]:
    """
    Simulate CN evolution on a phylogenetic tree.
    
    Args:
        tree: Phylogenetic tree
        rate_matrix: (4, 4) rate matrix
        root_state: Initial state at root (or None for stationary)
        rng: Random number generator
    
    Returns:
        Dictionary mapping taxon name → state
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize root state
    if root_state is None:
        # Sample from stationary distribution
        pi = get_stationary_distribution(rate_matrix)
        root_state = rng.choice(4, p=pi)

    # Store states at all nodes
    node_states = np.zeros(tree.n_nodes, dtype=int)
    node_states[tree.root_index] = root_state

    # Traverse tree (preorder)
    def traverse(node_id: int):
        parent_state = node_states[node_id]
        node = tree.nodes[node_id]

        for child_id in node.children_ids:
            branch_length = tree.branch_lengths[child_id]

            child_state = simulate_along_branch(parent_state, branch_length, rate_matrix, rng)
            node_states[child_id] = child_state

            if not tree.nodes[child_id].is_tip:
                traverse(child_id)

    traverse(tree.root_index)

    # Extract tip states
    tip_states = {}
    for tip_idx in tree.tip_indices:
        taxon_name = tree.nodes[tip_idx].name or f"tip_{tip_idx}"
        tip_states[taxon_name] = node_states[tip_idx]

    return tip_states


def get_default_regimes() -> list:
    """
    Get default CN regime definitions (biological realism).
    
    Returns:
        List of RegimeParams defining realistic CN regimes
    """
    return [
        RegimeParams(
            name="stable_single",
            gain_rate=0.05,      # Low gain (rare gene birth)
            loss_rate=0.05,      # Low loss (stable presence)
            amplify_rate=0.01,   # Very rare amplification
            contract_rate=0.02,  # Rare contraction
            proportion=0.60,     # 60% of genes (housekeeping, essential)
            description="Stable single-copy genes (housekeeping, essential)"
        ),
        RegimeParams(
            name="volatile_multi",
            gain_rate=0.20,      # Higher gain
            loss_rate=0.15,      # Moderate loss
            amplify_rate=0.15,   # Frequent amplification
            contract_rate=0.15,  # Frequent contraction
            proportion=0.30,     # 30% of genes (antigen families, variable)
            description="Volatile multi-copy genes (antigens, variable)"
        ),
        RegimeParams(
            name="rare_amplifying",
            gain_rate=0.10,      # Moderate gain
            loss_rate=0.08,      # Lower loss (retained)
            amplify_rate=0.25,   # High amplification
            contract_rate=0.05,  # Low contraction (biased)
            proportion=0.10,     # 10% of genes (drug resistance, adaptive)
            description="Rare amplifying genes (drug resistance, adaptive CNV)"
        ),
    ]


def simulate_cn_evolution(
    tree: TreeStructure,
    config: SimulationConfig,
) -> tuple[np.ndarray, dict]:
    """
    Simulate copy number evolution for multiple families.
    
    Supports regime-based heterogeneity for biological realism.
    
    Args:
        tree: Phylogenetic tree
        config: Simulation configuration
    
    Returns:
        (cn_matrix, metadata) where:
            cn_matrix: (n_families, n_taxa) array of simulated states
            metadata: Dictionary with simulation details
    """
    rng = np.random.default_rng(config.seed)

    # Get taxon names from tree
    taxon_names = [tree.nodes[idx].name or f"tip_{idx}" for idx in tree.tip_indices]
    n_taxa = len(taxon_names)

    # Initialize output
    cn_matrix = np.zeros((config.n_families, n_taxa), dtype=int)

    # Regime-based simulation (realistic heterogeneity)
    if config.use_regimes:
        regimes = config.regime_params if config.regime_params else get_default_regimes()

        # Assign families to regimes
        regime_proportions = [r.proportion for r in regimes]
        regime_assignments = rng.choice(
            len(regimes),
            size=config.n_families,
            p=regime_proportions,
        )

        # Simulate each family with regime-specific rates
        for fam_idx in range(config.n_families):
            regime = regimes[regime_assignments[fam_idx]]

            # Build rate matrix for this regime
            baseline = GlobalBaseline(
                gain_rate=regime.gain_rate,
                loss_rate=regime.loss_rate,
                amplify_rate=regime.amplify_rate,
                contract_rate=regime.contract_rate,
            )
            baseline_matrix = baseline.build_rate_matrix()

            # Apply constraint if specified
            if config.constraint_type is not None and config.theta is not None:
                if config.constraint_type == 'dosage_stability':
                    constraint = DosageStabilityConstraint()
                elif config.constraint_type == 'amplification_bias':
                    constraint = AmplificationBiasConstraint()
                elif config.constraint_type == 'host_conditioned':
                    constraint = HostConditionedVolatilityConstraint()
                else:
                    raise ValueError(f"Unknown constraint type: {config.constraint_type}")

                rate_matrix = apply_constraint(
                    baseline_matrix,
                    constraint,
                    config.theta,
                    family_idx=fam_idx,
                )
            else:
                rate_matrix = baseline_matrix

            # Simulate on tree
            tip_states = simulate_on_tree(tree, rate_matrix, config.root_state, rng)

            # Store in matrix
            for taxon_idx, taxon_name in enumerate(taxon_names):
                cn_matrix[fam_idx, taxon_idx] = tip_states[taxon_name]

        metadata: dict = {
            'scenario': config.scenario.value,
            'use_regimes': True,
            'regimes': [r.name for r in regimes],
            'regime_proportions': regime_proportions,
            'constraint_type': config.constraint_type,
            'theta': config.theta,
            'n_families': config.n_families,
            'n_taxa': n_taxa,
            'seed': config.seed,
            'taxon_names': taxon_names,
        }

        return cn_matrix, metadata

    # Original single-baseline simulation (for comparison)
    # Create baseline model
    if config.baseline_params is None:
        baseline_params = {}
    else:
        baseline_params = config.baseline_params.copy()

    if config.baseline_type == 'global':
        baseline = GlobalBaseline(**baseline_params)
    else:  # hierarchical
        baseline = HierarchicalBaseline(**baseline_params)
        baseline.sample_family_rates(config.n_families, rng)

    # Simulate each family
    for fam_idx in range(config.n_families):
        # Get baseline rate matrix for this family
        baseline_matrix = baseline.build_rate_matrix(
            family_idx=fam_idx if config.baseline_type == 'hierarchical' else None
        )

        # Apply constraint if specified
        if config.constraint_type is not None and config.theta is not None:
            if config.constraint_type == 'dosage_stability':
                constraint = DosageStabilityConstraint()
            elif config.constraint_type == 'amplification_bias':
                constraint = AmplificationBiasConstraint()
            elif config.constraint_type == 'host_conditioned':
                constraint = HostConditionedVolatilityConstraint()
            else:
                raise ValueError(f"Unknown constraint type: {config.constraint_type}")

            rate_matrix = apply_constraint(
                baseline_matrix,
                constraint,
                config.theta,
                family_idx=fam_idx,
            )
        else:
            rate_matrix = baseline_matrix

        # Simulate on tree
        tip_states = simulate_on_tree(tree, rate_matrix, config.root_state, rng)

        # Store in matrix
        for taxon_idx, taxon_name in enumerate(taxon_names):
            cn_matrix[fam_idx, taxon_idx] = tip_states[taxon_name]

    # Metadata
    metadata = {
        'scenario': config.scenario.value,
        'baseline_type': config.baseline_type,
        'constraint_type': config.constraint_type,
        'theta': config.theta,
        'n_families': config.n_families,
        'n_taxa': n_taxa,
        'seed': config.seed,
        'taxon_names': taxon_names,
    }

    return cn_matrix, metadata


def create_scenario_config(
    scenario: SimulationScenario,
    n_families: int = 200,
    seed: int = 42,
    use_regimes: bool = True,  # Use realistic regime heterogeneity by default
) -> SimulationConfig:
    """
    Create simulation config for a predefined scenario.
    
    Uses regime-based heterogeneity for biological realism:
    - 60% stable single-copy (housekeeping)
    - 30% volatile multi-copy (antigens)
    - 10% rare amplifying (drug resistance)
    
    Args:
        scenario: Scenario to simulate
        n_families: Number of families (default 200)
        seed: Random seed
        use_regimes: Use regime-based heterogeneity (default True)
    
    Returns:
        SimulationConfig
    """
    if scenario == SimulationScenario.NULL:
        # Null: regime heterogeneity, no constraint
        return SimulationConfig(
            scenario=scenario,
            constraint_type=None,
            theta=None,
            n_families=n_families,
            seed=seed,
            use_regimes=use_regimes,
        )

    elif scenario == SimulationScenario.DOSAGE_BUFFERING:
        # Dosage buffering: regime heterogeneity + moderate constraint
        # Effect: suppresses all CN changes across all regimes
        return SimulationConfig(
            scenario=scenario,
            constraint_type='dosage_stability',
            theta=-0.7,  # Moderate buffering (not overfitting)
            n_families=n_families,
            seed=seed,
            use_regimes=use_regimes,
        )

    elif scenario == SimulationScenario.AMPLIFICATION_BIAS:
        # Amplification bias: regime heterogeneity + moderate constraint
        # Effect: boosts 1→2, 2→3; suppresses 2→1, 3→2 (bidirectional)
        # Does NOT affect 0→1 (gene birth ≠ amplification)
        return SimulationConfig(
            scenario=scenario,
            constraint_type='amplification_bias',
            theta=0.7,  # Moderate amplification (not overfitting)
            n_families=n_families,
            seed=seed,
            use_regimes=use_regimes,
        )

    elif scenario == SimulationScenario.HIGH_VOLATILITY_LINEAGE:
        # High volatility: regime heterogeneity + lineage-specific constraint
        return SimulationConfig(
            scenario=scenario,
            constraint_type='host_conditioned',
            theta=0.7,
            n_families=n_families,
            seed=seed,
            use_regimes=use_regimes,
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def simulate_scenario(
    scenario: SimulationScenario,
    tree: TreeStructure,
    n_families: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """
    Convenience function to simulate a predefined scenario.
    
    Args:
        scenario: Scenario to simulate
        tree: Phylogenetic tree
        n_families: Number of families
        seed: Random seed
    
    Returns:
        (cn_matrix, metadata)
    """
    config = create_scenario_config(scenario, n_families, seed)
    return simulate_cn_evolution(tree, config)
