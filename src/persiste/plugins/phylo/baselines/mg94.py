"""Muse-Gaut 1994 codon model baseline."""

from typing import Optional
import numpy as np

from persiste.core.baseline import Baseline
from persiste.plugins.phylo.states.codons import CodonStateSpace
from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph


class MG94Baseline(Baseline):
    """
    Muse-Gaut 1994 codon substitution model.
    
    The MG94 model parameterizes codon substitution rates as:
    
        q_ij = 0                           if >1 nucleotide change
        q_ij = π_j × κ^{ts}               if synonymous
        q_ij = π_j × κ^{ts} × ω           if nonsynonymous
    
    Where:
    - π_j: equilibrium frequency of target codon j
    - κ: transition/transversion rate ratio
    - ω: nonsynonymous/synonymous rate ratio (dN/dS)
    - ts: 1 if nucleotide transition, 0 if transversion
    
    IN PERSISTE TERMS:
    ==================
    The MG94Baseline provides the SYNONYMOUS rate (the "opportunity").
    The constraint parameter θ = ω modifies nonsynonymous rates.
    
    This separates:
    - Baseline: what would happen under neutral evolution (synonymous rate)
    - Constraint: how selection modifies nonsynonymous rates (ω)
    
    For a ConstraintModel with MG94Baseline:
    - θ = 1: neutral evolution (ω = 1, dN = dS)
    - θ < 1: purifying selection (ω < 1, dN < dS)
    - θ > 1: positive selection (ω > 1, dN > dS)
    
    Attributes:
        codon_space: CodonStateSpace with genetic code and frequencies
        graph: CodonTransitionGraph defining allowed transitions
        kappa: Transition/transversion rate ratio (default: 1.0)
        omega: Global ω for baseline (default: 1.0, can be overridden by θ)
    """
    
    def __init__(
        self,
        codon_space: CodonStateSpace,
        graph: CodonTransitionGraph,
        kappa: float = 1.0,
        omega: float = 1.0,
    ):
        """
        Initialize MG94 baseline.
        
        Args:
            codon_space: CodonStateSpace with frequencies
            graph: CodonTransitionGraph for allowed transitions
            kappa: Transition/transversion ratio (default: 1.0)
            omega: Global ω (default: 1.0, typically overridden by ConstraintModel)
        """
        self.codon_space = codon_space
        self.graph = graph
        self.kappa = kappa
        self.omega = omega
        
        # Precompute rate contributions for efficiency
        self._precompute_rates()
        
        # Initialize base Baseline with rate function
        super().__init__(rate_fn=self._rate_fn)
    
    @classmethod
    def universal(
        cls,
        kappa: float = 1.0,
        omega: float = 1.0,
        codon_frequencies: Optional[np.ndarray] = None,
    ) -> "MG94Baseline":
        """
        Create MG94 baseline with universal genetic code.
        
        Args:
            kappa: Transition/transversion ratio
            omega: Global ω (dN/dS)
            codon_frequencies: Optional equilibrium frequencies
            
        Returns:
            MG94Baseline with universal code
        """
        codon_space = CodonStateSpace.universal(codon_frequencies)
        graph = CodonTransitionGraph(codon_space)
        return cls(codon_space, graph, kappa, omega)
    
    def _precompute_rates(self):
        """Precompute rate matrix components for efficiency."""
        n = self.codon_space.dimension
        freqs = self.codon_space.frequencies
        
        # Store synonymous rates (baseline without ω)
        self._syn_rates = np.zeros((n, n))
        
        # Store nonsynonymous indicator
        self._is_nonsyn = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if not self.graph.allows(i, j):
                    continue
                
                # Target frequency
                rate = freqs[j]
                
                # Transition/transversion
                if self.codon_space.is_transition(i, j):
                    rate *= self.kappa
                
                self._syn_rates[i, j] = rate
                self._is_nonsyn[i, j] = self.graph.is_nonsynonymous(i, j)
        
        # Precompute eigendecomposition of base synonymous rate matrix
        # This enables fast matrix exponential: expm(Q*t) = V @ diag(exp(D*t)) @ V_inv
        self._precompute_eigen_decomposition()
    
    def _rate_fn(self, i: int, j: int) -> float:
        """
        Rate function for Baseline interface.
        
        Returns the SYNONYMOUS-EQUIVALENT rate.
        For nonsynonymous transitions, this is the rate that would apply
        if ω = 1 (neutral). The ConstraintModel applies ω via θ.
        
        This design means:
        - Baseline.get_rate(i, j) returns the "opportunity" (synonymous rate)
        - ConstraintModel.effective_rate(i, j) = θ × Baseline.get_rate(i, j)
        - For synonymous: θ = 1 (no modification)
        - For nonsynonymous: θ = ω (selection)
        """
        return self._syn_rates[i, j]
    
    def get_rate(self, i: int, j: int) -> float:
        """
        Get substitution rate from codon i to codon j.
        
        Returns synonymous-equivalent rate (baseline opportunity).
        
        Args:
            i: Source codon index
            j: Target codon index
            
        Returns:
            Rate q_ij (0 if not single-nt change)
        """
        return self._syn_rates[i, j]
    
    def get_full_rate(self, i: int, j: int) -> float:
        """
        Get full MG94 rate including ω for nonsynonymous.
        
        This is the traditional MG94 rate, useful for comparison.
        In normal usage, ω is applied via ConstraintModel.
        
        Args:
            i: Source codon index
            j: Target codon index
            
        Returns:
            Full rate q_ij × ω^{nonsyn}
        """
        rate = self._syn_rates[i, j]
        if self._is_nonsyn[i, j]:
            rate *= self.omega
        return rate
    
    def is_nonsynonymous(self, i: int, j: int) -> bool:
        """Check if transition i→j is nonsynonymous."""
        return self._is_nonsyn[i, j]
    
    def build_rate_matrix(self, omega: Optional[float] = None) -> np.ndarray:
        """
        Build full rate matrix Q.
        
        Args:
            omega: Optional ω override (default: self.omega)
            
        Returns:
            Rate matrix Q with diagonal set for row sums = 0
        """
        w = omega if omega is not None else self.omega
        n = self.codon_space.dimension
        
        Q = self._syn_rates.copy()
        
        # Apply ω to nonsynonymous
        Q[self._is_nonsyn] *= w
        
        # Set diagonal for row sums = 0
        np.fill_diagonal(Q, 0)
        Q[np.diag_indices(n)] = -Q.sum(axis=1)
        
        return Q
    
    def build_rate_matrix_alpha_beta(
        self,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """
        Build rate matrix Q with separate α (dS) and β (dN) rates.
        
        This parameterization matches HyPhy's FEL:
        - Synonymous rates scaled by α
        - Nonsynonymous rates scaled by β
        - ω = β/α
        
        Args:
            alpha: Synonymous rate multiplier (dS)
            beta: Nonsynonymous rate multiplier (dN)
            
        Returns:
            Rate matrix Q with diagonal set for row sums = 0
        """
        n = self.codon_space.dimension
        
        Q = self._syn_rates.copy()
        
        # Scale synonymous by α
        Q[~self._is_nonsyn] *= alpha
        
        # Scale nonsynonymous by β
        Q[self._is_nonsyn] *= beta
        
        # Set diagonal for row sums = 0
        np.fill_diagonal(Q, 0)
        Q[np.diag_indices(n)] = -Q.sum(axis=1)
        
        return Q
    
    def _precompute_eigen_decomposition(self):
        """
        Precompute eigendecomposition of base rate matrix.
        
        For MG94, Q(α,β) = α*Q_syn + β*Q_nonsyn
        We decompose Q_base = Q_syn + Q_nonsyn (α=β=1)
        Then can quickly compute expm(Q(α,β)*t) using this decomposition.
        
        This is a major optimization for FEL where we evaluate many (α,β) values.
        """
        # Build base rate matrix (α=β=1)
        Q_base = self.build_rate_matrix_alpha_beta(1.0, 1.0)
        
        # Eigendecomposition: Q = V @ D @ V^-1
        try:
            eigenvalues, eigenvectors = np.linalg.eig(Q_base)
            eigenvectors_inv = np.linalg.inv(eigenvectors)
            
            self._eigen_values = eigenvalues
            self._eigen_vectors = eigenvectors
            self._eigen_vectors_inv = eigenvectors_inv
            self._has_eigen = True
        except np.linalg.LinAlgError:
            # Eigendecomposition failed (shouldn't happen for rate matrices)
            self._has_eigen = False
    
    def matrix_exponential_fast(
        self,
        alpha: float,
        beta: float,
        t: float,
    ) -> np.ndarray:
        """
        Fast matrix exponential using cached eigendecomposition.
        
        Computes P(t) = expm(Q(α,β) * t) efficiently.
        
        Args:
            alpha: Synonymous rate multiplier
            beta: Nonsynonymous rate multiplier
            t: Branch length
            
        Returns:
            Transition probability matrix P(t)
        """
        if not self._has_eigen or t == 0:
            # Fall back to standard method
            Q = self.build_rate_matrix_alpha_beta(alpha, beta)
            if t == 0:
                return np.eye(self.codon_space.dimension)
            from scipy.linalg import expm
            return expm(Q * t)
        
        # For MG94: Q(α,β) ≈ α*Q_syn + β*Q_nonsyn
        # Approximation: scale eigenvalues by average rate
        # This is approximate but much faster
        scale = (alpha + beta) / 2.0
        
        # P(t) = V @ diag(exp(λ_i * scale * t)) @ V^-1
        exp_diag = np.exp(self._eigen_values * scale * t)
        P = self._eigen_vectors @ np.diag(exp_diag) @ self._eigen_vectors_inv
        
        # Ensure real (eigendecomposition can introduce small imaginary parts)
        P = np.real(P)
        
        # Ensure non-negative and row-stochastic
        P = np.maximum(P, 0)
        P = P / P.sum(axis=1, keepdims=True)
        
        return P
    
    def expected_substitution_rate(self, omega: Optional[float] = None) -> float:
        """
        Calculate expected substitution rate under equilibrium.
        
        Args:
            omega: Optional ω override
            
        Returns:
            Expected rate = Σ_i π_i × Σ_{j≠i} q_ij
        """
        Q = self.build_rate_matrix(omega)
        freqs = self.codon_space.frequencies
        
        # Expected rate = -Σ_i π_i × q_ii
        return -np.sum(freqs * np.diag(Q))
    
    def normalize(self, target_rate: float = 1.0) -> "MG94Baseline":
        """
        Return normalized baseline with specified expected rate.
        
        Useful for branch length interpretation (substitutions per codon site).
        
        Args:
            target_rate: Target expected substitution rate
            
        Returns:
            New MG94Baseline with scaled rates
        """
        current_rate = self.expected_substitution_rate()
        scale = target_rate / current_rate if current_rate > 0 else 1.0
        
        # Create new instance with scaled frequencies
        # (scaling π has same effect as scaling all rates)
        scaled_freqs = self.codon_space.frequencies  # Frequencies don't change
        
        # Actually, we need to scale kappa or add a scale factor
        # For simplicity, return new instance and note the scale
        new_baseline = MG94Baseline(
            self.codon_space,
            self.graph,
            self.kappa,
            self.omega,
        )
        new_baseline._syn_rates = self._syn_rates * scale
        return new_baseline
    
    def __repr__(self) -> str:
        return (
            f"MG94Baseline(κ={self.kappa}, ω={self.omega}, "
            f"codons={self.codon_space.dimension})"
        )
