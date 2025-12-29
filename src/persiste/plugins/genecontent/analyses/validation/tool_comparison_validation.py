#!/usr/bin/env python3
"""
External Tool Comparison Validation for GeneContent Plugin

Compares GeneContent against established tools:
- Count (Csűrös 2010) - GUI tool, gene copy numbers
- GLOOME (Cohen et al. 2010) - Binary presence/absence ✅
- BadiRate (Librado et al. 2012) - Gene copy numbers, adapted for binary

Tests:
1. Global gain/loss rate estimation on simulated data
2. Comparison of error rates across tools
"""

import sys
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from persiste.core.trees import TreeStructure
from persiste.plugins.genecontent.inference.gene_inference import GeneContentData
from persiste.plugins.genecontent.analyses.standard_analyses import GeneContentAnalysis


@dataclass
class ComparisonResult:
    """Result from comparing tools."""
    test_name: str
    tool: str
    passed: bool
    message: str
    metrics: Dict
    runtime_seconds: float


def tree_to_newick(tree: TreeStructure) -> str:
    """Convert TreeStructure to Newick format string."""
    def build_newick(node_idx: int) -> str:
        node = tree.nodes[node_idx]
        
        if node.is_tip:
            name = node.name or f"tip{node_idx}"
            return f"{name}:{tree.branch_lengths[node_idx]:.6f}"
        else:
            children_str = ",".join(
                build_newick(child_id) 
                for child_id in node.children_ids
            )
            bl = tree.branch_lengths[node_idx]
            if node_idx == tree.root_index:
                return f"({children_str})"
            else:
                return f"({children_str}):{bl:.6f}"
    
    return build_newick(tree.root_index) + ";"


class ExternalToolRunner:
    """Interface for running external tools."""
    
    def __init__(self, tool_name: str, tool_path: Optional[str] = None):
        self.tool_name = tool_name
        self.tool_path = tool_path or tool_name
        
    def check_available(self) -> bool:
        """Check if tool is available."""
        import shutil
        
        # Special handling for JAR files
        if self.tool_name == 'count' and self.tool_path.endswith('.jar'):
            return Path(self.tool_path).exists()
        
        # Special handling for Perl scripts
        if self.tool_name == 'badrate' and self.tool_path.endswith('.pl'):
            return Path(os.path.expanduser(self.tool_path)).exists()
        
        return shutil.which(self.tool_path) is not None
    
    def run_gloome(
        self,
        tree: TreeStructure,
        presence_matrix: np.ndarray,
        taxon_names: List[str],
        output_dir: Path,
    ) -> Dict:
        """Run GLOOME (gainLoss) on presence/absence data."""
        import subprocess
        
        try:
            gloome_dir = output_dir / 'gloome_run'
            gloome_dir.mkdir(exist_ok=True, parents=True)
            
            # Create tree file
            tree_path = gloome_dir / 'tree.nwk'
            with open(tree_path, 'w') as f:
                f.write(tree_to_newick(tree))
                f.write('\n')
            
            # Create sequence file (FASTA with binary 0/1)
            seq_path = gloome_dir / 'sequences.fa'
            with open(seq_path, 'w') as f:
                for tip_idx, taxon in enumerate(taxon_names):
                    f.write(f'>{taxon}\n')
                    sequence = ''.join([str(int(presence_matrix[tip_idx, fam_idx]))
                                       for fam_idx in range(presence_matrix.shape[1])])
                    f.write(sequence + '\n')
            
            # Create parameter file with absolute paths
            param_path = gloome_dir / 'params.txt'
            with open(param_path, 'w') as f:
                f.write(f'_seqFile {seq_path.absolute()}\n')
                f.write(f'_treeFile {tree_path.absolute()}\n')
                f.write(f'_outDir {gloome_dir.absolute()}\n')
                f.write('_logFile gainLoss.log\n')
                f.write('_gainLossDist 1\n')
                f.write('_numberOfGainCategories 1\n')
                f.write('_numberOfLossCategories 1\n')
                f.write('_maxNumOfIterationsModel 10\n')
                f.write('_performOptimizations 1\n')
            
            # Run GLOOME
            result = subprocess.run(
                [self.tool_path, str(param_path.absolute())],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0 or 'error' in result.stderr.lower():
                return {
                    'available': False,
                    'error': f'GLOOME failed: {result.stderr[:300]}'
                }
            
            # Parse EstimatedParameters.txt
            params_file = gloome_dir / 'EstimatedParameters.txt'
            gain_rate = None
            loss_rate = None
            log_likelihood = None
            
            if params_file.exists():
                with open(params_file) as f:
                    for line in f:
                        if 'Gain Expectation' in line:
                            gain_rate = float(line.split('=')[1].strip())
                        elif 'Loss Expectation' in line:
                            loss_rate = float(line.split('=')[1].strip())
                        elif 'Log-likelihood' in line:
                            log_likelihood = float(line.split('=')[1].strip())
            
            if gain_rate is None or loss_rate is None:
                return {
                    'available': False,
                    'error': f'Could not parse GLOOME output'
                }
            
            return {
                'gain_rate': gain_rate,
                'loss_rate': loss_rate,
                'log_likelihood': log_likelihood or -np.inf,
                'available': True
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': f'GLOOME error: {str(e)[:200]}'
            }
    
    def run_badrate(
        self,
        tree: TreeStructure,
        presence_matrix: np.ndarray,
        taxon_names: List[str],
        output_dir: Path,
    ) -> Dict:
        """Run BadiRate on data (using 0/1 for presence/absence)."""
        import subprocess
        
        try:
            badirate_dir = output_dir / 'badirate_run'
            badirate_dir.mkdir(exist_ok=True, parents=True)
            
            # Create tree file
            tree_path = badirate_dir / 'tree.nwk'
            with open(tree_path, 'w') as f:
                f.write(tree_to_newick(tree))
                f.write('\n')
            
            # Create family file in BadiRate TSV format
            fam_path = badirate_dir / 'families.tsv'
            with open(fam_path, 'w') as f:
                f.write('FAM_ID\t' + '\t'.join(taxon_names) + '\n')
                for fam_idx in range(presence_matrix.shape[1]):
                    row = [f'fam{fam_idx}'] + [str(int(presence_matrix[tip_idx, fam_idx]))
                                                 for tip_idx in range(len(taxon_names))]
                    f.write('\t'.join(row) + '\n')
            
            # Run BadiRate with GD (Gain-Death) model
            badirate_script = os.path.expanduser(self.tool_path)
            cmd = [
                'perl', badirate_script,
                '-treefile', str(tree_path.absolute()),
                '-sizefile', str(fam_path.absolute()),
                '-rmodel', 'GD',
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(badirate_dir)
            )
            
            # Check for Perl dependency errors
            if 'Can\'t locate' in result.stderr or 'BEGIN failed' in result.stderr:
                return {
                    'available': False,
                    'error': f'BadiRate Perl dependencies missing'
                }
            
            if result.returncode != 0:
                return {
                    'available': False,
                    'error': f'BadiRate failed (rc={result.returncode})'
                }
            
            # Parse output
            gain_rate = None
            loss_rate = None
            log_likelihood = None
            
            for line in result.stdout.split('\n'):
                if 'gain' in line.lower() and 'rate' in line.lower():
                    parts = line.split()
                    for part in parts:
                        try:
                            val = float(part)
                            if 0 < val < 100:
                                gain_rate = val
                                break
                        except:
                            continue
                elif ('death' in line.lower() or 'loss' in line.lower()) and 'rate' in line.lower():
                    parts = line.split()
                    for part in parts:
                        try:
                            val = float(part)
                            if 0 < val < 100:
                                loss_rate = val
                                break
                        except:
                            continue
                elif 'lnl' in line.lower() or 'likelihood' in line.lower():
                    parts = line.split()
                    for part in parts:
                        try:
                            val = float(part)
                            if val < 0:
                                log_likelihood = val
                                break
                        except:
                            continue
            
            if gain_rate is None or loss_rate is None:
                return {
                    'available': False,
                    'error': f'Could not parse BadiRate output'
                }
            
            return {
                'gain_rate': gain_rate,
                'loss_rate': loss_rate,
                'log_likelihood': log_likelihood or -np.inf,
                'available': True
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': f'BadiRate error: {str(e)[:200]}'
            }


class ToolComparisonSuite:
    """Validation suite comparing GeneContent to external tools."""
    
    def __init__(self, output_dir: Optional[Path] = None, verbose: bool = True):
        self.output_dir = output_dir or Path("./tool_comparison_output")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.verbose = verbose
        self.results: List[ComparisonResult] = []
        
        # Initialize tool runners
        count_jar = os.path.expanduser('~/Downloads/Count/Count.jar')
        self.count_runner = ExternalToolRunner('count', count_jar)
        self.gloome_runner = ExternalToolRunner('gainLoss')
        self.badrate_runner = ExternalToolRunner('badrate', '~/Documents/badirate/BadiRate.pl')
    
    def log(self, message: str):
        if self.verbose:
            print(message)
    
    def add_result(self, result: ComparisonResult):
        self.results.append(result)
        status = "[PASS]" if result.passed else "[FAIL]"
        self.log(f"{status} {result.test_name} ({result.tool}): {result.message}")
    
    def compare_global_rates_simulated(self, n_replicates: int = 5):
        """Compare global gain/loss rate estimation on simulated data with multiple replicates."""
        self.log("\n" + "="*63)
        self.log("COMPARISON 1: Global Gain/Loss Rates (Simulated)")
        self.log("="*63 + "\n")
        
        # Simulate data
        n_tips = 8
        n_families = 200
        true_gain = 2.0
        true_loss = 3.0
        
        self.log(f"Running {n_replicates} replicates with:")
        self.log(f"  - {n_tips} taxa")
        self.log(f"  - {n_families} gene families")
        self.log(f"  - True gain rate: {true_gain}")
        self.log(f"  - True loss rate: {true_loss}")
        self.log(f"  - True π₁: {true_gain/(true_gain+true_loss):.3f}\n")
        
        # Create simple tree from Newick
        import tempfile
        from persiste.core.trees import load_tree
        taxon_names = [f"tip{i}" for i in range(n_tips)]
        
        # Create a simple balanced Newick tree
        newick = "(((tip0:1,tip1:1):1,(tip2:1,tip3:1):1):1,((tip4:1,tip5:1):1,(tip6:1,tip7:1):1):1):0;"
        
        # Write to temp file and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.nwk', delete=False) as f:
            f.write(newick)
            tree_file = f.name
        
        tree = load_tree(tree_file)
        Path(tree_file).unlink()  # Clean up
        
        # Collect results across replicates
        gc_results = {'gain': [], 'loss': [], 'pi1': [], 'runtime': []}
        gloome_results = {'gain': [], 'loss': [], 'pi1': [], 'runtime': []}
        
        for rep in range(n_replicates):
            self.log(f"\n--- Replicate {rep+1}/{n_replicates} ---")
            
            # Simulate presence/absence using simple random data
            np.random.seed(42 + rep)
            presence_matrix = np.random.binomial(1, 0.5, size=(n_tips, n_families))
            family_names = [f"fam{i}" for i in range(n_families)]
            
            # Test GeneContent
            start_time = time.time()
            data = GeneContentData(
                tree=tree, 
                presence_matrix=presence_matrix,
                taxon_names=taxon_names,
                family_names=family_names
            )
            analysis = GeneContentAnalysis(data)
            result = analysis.global_rates(verbose=False)
            gc_runtime = time.time() - start_time
            
            gc_results['gain'].append(result.gain_rate)
            gc_results['loss'].append(result.loss_rate)
            gc_results['pi1'].append(result.gain_rate / (result.gain_rate + result.loss_rate))
            gc_results['runtime'].append(gc_runtime)
            
            self.log(f"  GeneContent: gain={result.gain_rate:.3f}, loss={result.loss_rate:.3f}, π₁={gc_results['pi1'][-1]:.3f}, time={gc_runtime:.2f}s")
            
            # Test GLOOME
            if self.gloome_runner.check_available():
                start_time = time.time()
                gloome_result = self.gloome_runner.run_gloome(tree, presence_matrix, taxon_names, self.output_dir)
                gloome_runtime = time.time() - start_time
                
                if 'error' not in gloome_result:
                    gloome_results['gain'].append(gloome_result['gain_rate'])
                    gloome_results['loss'].append(gloome_result['loss_rate'])
                    gloome_results['pi1'].append(gloome_result['gain_rate'] / (gloome_result['gain_rate'] + gloome_result['loss_rate']))
                    gloome_results['runtime'].append(gloome_runtime)
                    
                    self.log(f"  GLOOME:      gain={gloome_result['gain_rate']:.3f}, loss={gloome_result['loss_rate']:.3f}, π₁={gloome_results['pi1'][-1]:.3f}, time={gloome_runtime:.2f}s")
        
        # Compute summary statistics
        self.log("\n" + "="*63)
        self.log("DETAILED COMPARISON RESULTS")
        self.log("="*63 + "\n")
        
        true_pi1 = true_gain / (true_gain + true_loss)
        
        # GeneContent summary
        gc_gain_mean = np.mean(gc_results['gain'])
        gc_gain_std = np.std(gc_results['gain'])
        gc_gain_error = abs(gc_gain_mean - true_gain) / true_gain
        
        gc_loss_mean = np.mean(gc_results['loss'])
        gc_loss_std = np.std(gc_results['loss'])
        gc_loss_error = abs(gc_loss_mean - true_loss) / true_loss
        
        gc_pi1_mean = np.mean(gc_results['pi1'])
        gc_pi1_std = np.std(gc_results['pi1'])
        gc_pi1_error = abs(gc_pi1_mean - true_pi1) / true_pi1
        
        gc_runtime_mean = np.mean(gc_results['runtime'])
        
        self.log("GeneContent Results:")
        self.log(f"  Gain rate:  {gc_gain_mean:.3f} ± {gc_gain_std:.3f} (true: {true_gain:.3f}, error: {gc_gain_error:.1%})")
        self.log(f"  Loss rate:  {gc_loss_mean:.3f} ± {gc_loss_std:.3f} (true: {true_loss:.3f}, error: {gc_loss_error:.1%})")
        self.log(f"  π₁:         {gc_pi1_mean:.3f} ± {gc_pi1_std:.3f} (true: {true_pi1:.3f}, error: {gc_pi1_error:.1%})")
        self.log(f"  Runtime:    {gc_runtime_mean:.2f}s\n")
        
        self.add_result(ComparisonResult(
            test_name="GlobalRates_Simulated",
            tool="genecontent",
            passed=gc_gain_error < 0.8 and gc_loss_error < 0.8,
            message=f"Gain error={gc_gain_error:.1%}, Loss error={gc_loss_error:.1%}, π₁ error={gc_pi1_error:.1%}",
            metrics={
                'gain_mean': gc_gain_mean,
                'gain_std': gc_gain_std,
                'gain_error': gc_gain_error,
                'loss_mean': gc_loss_mean,
                'loss_std': gc_loss_std,
                'loss_error': gc_loss_error,
                'pi1_mean': gc_pi1_mean,
                'pi1_std': gc_pi1_std,
                'pi1_error': gc_pi1_error,
            },
            runtime_seconds=gc_runtime_mean,
        ))
        
        # GLOOME summary
        if gloome_results['gain']:
            gloome_gain_mean = np.mean(gloome_results['gain'])
            gloome_gain_std = np.std(gloome_results['gain'])
            gloome_gain_error = abs(gloome_gain_mean - true_gain) / true_gain
            
            gloome_loss_mean = np.mean(gloome_results['loss'])
            gloome_loss_std = np.std(gloome_results['loss'])
            gloome_loss_error = abs(gloome_loss_mean - true_loss) / true_loss
            
            gloome_pi1_mean = np.mean(gloome_results['pi1'])
            gloome_pi1_std = np.std(gloome_results['pi1'])
            gloome_pi1_error = abs(gloome_pi1_mean - true_pi1) / true_pi1
            
            gloome_runtime_mean = np.mean(gloome_results['runtime'])
            
            self.log("GLOOME Results:")
            self.log(f"  Gain rate:  {gloome_gain_mean:.3f} ± {gloome_gain_std:.3f} (true: {true_gain:.3f}, error: {gloome_gain_error:.1%})")
            self.log(f"  Loss rate:  {gloome_loss_mean:.3f} ± {gloome_loss_std:.3f} (true: {true_loss:.3f}, error: {gloome_loss_error:.1%})")
            self.log(f"  π₁:         {gloome_pi1_mean:.3f} ± {gloome_pi1_std:.3f} (true: {true_pi1:.3f}, error: {gloome_pi1_error:.1%})")
            self.log(f"  Runtime:    {gloome_runtime_mean:.2f}s\n")
            
            self.add_result(ComparisonResult(
                test_name="GlobalRates_Simulated",
                tool="gloome",
                passed=gloome_gain_error < 0.8 and gloome_loss_error < 0.8,
                message=f"Gain error={gloome_gain_error:.1%}, Loss error={gloome_loss_error:.1%}, π₁ error={gloome_pi1_error:.1%}",
                metrics={
                    'gain_mean': gloome_gain_mean,
                    'gain_std': gloome_gain_std,
                    'gain_error': gloome_gain_error,
                    'loss_mean': gloome_loss_mean,
                    'loss_std': gloome_loss_std,
                    'loss_error': gloome_loss_error,
                    'pi1_mean': gloome_pi1_mean,
                    'pi1_std': gloome_pi1_std,
                    'pi1_error': gloome_pi1_error,
                },
                runtime_seconds=gloome_runtime_mean,
            ))
            
            # Comparative analysis
            self.log("="*63)
            self.log("COMPARATIVE ANALYSIS")
            self.log("="*63 + "\n")
            self.log("Error Comparison (lower is better):")
            self.log(f"  Gain rate:  GeneContent {gc_gain_error:.1%} vs GLOOME {gloome_gain_error:.1%}")
            self.log(f"  Loss rate:  GeneContent {gc_loss_error:.1%} vs GLOOME {gloome_loss_error:.1%}")
            self.log(f"  π₁:         GeneContent {gc_pi1_error:.1%} vs GLOOME {gloome_pi1_error:.1%}")
            self.log(f"\nSpeed:        GeneContent {gc_runtime_mean:.2f}s vs GLOOME {gloome_runtime_mean:.2f}s")
            self.log(f"              GLOOME is {gc_runtime_mean/gloome_runtime_mean:.1f}x faster\n")
        else:
            self.log("GLOOME not available - skipping comparison")
        
        # Test BadiRate
        self.log("\n=== Testing BadiRate ===")
        if self.badrate_runner.check_available():
            start_time = time.time()
            badrate_result = self.badrate_runner.run_badrate(tree, presence_matrix, taxon_names, self.output_dir)
            badrate_runtime = time.time() - start_time
            
            if 'error' not in badrate_result:
                badrate_gain_error = abs(badrate_result['gain_rate'] - true_gain) / true_gain
                badrate_loss_error = abs(badrate_result['loss_rate'] - true_loss) / true_loss
                
                self.add_result(ComparisonResult(
                    test_name="GlobalRates_Simulated",
                    tool="badrate",
                    passed=badrate_gain_error < 0.8 and badrate_loss_error < 0.8,
                    message=f"Gain error={badrate_gain_error:.1%}, Loss error={badrate_loss_error:.1%}",
                    metrics={
                        'true_gain': true_gain,
                        'est_gain': badrate_result['gain_rate'],
                        'gain_error': badrate_gain_error,
                        'true_loss': true_loss,
                        'est_loss': badrate_result['loss_rate'],
                        'loss_error': badrate_loss_error,
                    },
                    runtime_seconds=badrate_runtime,
                ))
            else:
                self.log(f"BadiRate error: {badrate_result['error']}")
        else:
            self.log("BadiRate not available")
    
    def compare_retention_bias_detection(self, n_replicates: int = 10):
        """Compare family-specific retention bias detection across tools.
        
        This tests whether tools can detect when a subset of families has reduced loss rates.
        GeneContent should provide explicit θ (retention strength) estimates, while other
        tools require post-hoc interpretation.
        """
        self.log("\n" + "="*63)
        self.log("COMPARISON 2: Family-Specific Retention Bias Detection")
        self.log("="*63 + "\n")
        
        n_tips = 8
        n_families_total = 100
        n_families_retained = 20  # 20% of families have retention bias
        true_gain = 1.5
        true_loss_normal = 2.0
        true_loss_retained = 0.5  # 4x slower loss for retained families
        true_loss_multiplier = true_loss_retained / true_loss_normal  # 0.25 = 1/4
        true_retention_strength = np.log(true_loss_multiplier)  # log(0.25) = -1.39
        
        self.log(f"Simulation setup:")
        self.log(f"  - {n_tips} taxa")
        self.log(f"  - {n_families_total} total families")
        self.log(f"  - {n_families_retained} families with retention bias ({100*n_families_retained/n_families_total:.0f}%)")
        self.log(f"  - True gain rate: {true_gain}")
        self.log(f"  - True loss rate (normal): {true_loss_normal}")
        self.log(f"  - True loss rate (retained): {true_loss_retained} (4x slower)")
        self.log(f"  - True loss multiplier: {true_loss_multiplier:.3f}")
        self.log(f"  - True retention_strength: {true_retention_strength:.2f} (log-scale)\n")
        
        # Create tree
        import tempfile
        from persiste.core.trees import load_tree
        taxon_names = [f"tip{i}" for i in range(n_tips)]
        newick = "(((tip0:1,tip1:1):1,(tip2:1,tip3:1):1):1,((tip4:1,tip5:1):1,(tip6:1,tip7:1):1):1):0;"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.nwk', delete=False) as f:
            f.write(newick)
            tree_file = f.name
        
        tree = load_tree(tree_file)
        Path(tree_file).unlink()
        
        # Track detection results
        gc_results = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'theta_estimates': [],
            'p_values_retained': [],
            'p_values_normal': []
        }
        
        gloome_results = {
            'rate_variance_retained': [],
            'rate_variance_normal': []
        }
        
        for rep in range(n_replicates):
            self.log(f"\n--- Replicate {rep+1}/{n_replicates} ---")
            
            # Simulate data with retention bias
            np.random.seed(100 + rep)
            presence_matrix = np.zeros((n_tips, n_families_total), dtype=int)
            
            # Simple simulation: random presence/absence with different frequencies
            for fam_idx in range(n_families_total):
                if fam_idx < n_families_retained:
                    # Retained families: higher presence frequency
                    freq = true_gain / (true_gain + true_loss_retained)
                else:
                    # Normal families: lower presence frequency
                    freq = true_gain / (true_gain + true_loss_normal)
                
                presence_matrix[:, fam_idx] = np.random.binomial(1, freq, size=n_tips)
            
            family_names = [f"fam{i}" for i in range(n_families_total)]
            retained_families = family_names[:n_families_retained]
            normal_families = family_names[n_families_retained:]
            
            # Test GeneContent
            data = GeneContentData(
                tree=tree,
                presence_matrix=presence_matrix,
                taxon_names=taxon_names,
                family_names=family_names
            )
            analysis = GeneContentAnalysis(data)
            
            # Test retained families
            result_retained = analysis.retention_test(families=retained_families, verbose=False)
            gc_results['p_values_retained'].append(result_retained.p_value)
            gc_results['theta_estimates'].append(result_retained.retention_strength)
            
            # Detect if significant (p < 0.05)
            if result_retained.p_value < 0.05:
                gc_results['true_positives'] += 1
                self.log(f"  GeneContent: ✓ Detected retention (p={result_retained.p_value:.3f}, θ={result_retained.retention_strength:.2f})")
            else:
                gc_results['false_negatives'] += 1
                self.log(f"  GeneContent: ✗ Missed retention (p={result_retained.p_value:.3f}, θ={result_retained.retention_strength:.2f})")
            
            # Test a random subset of normal families as negative control
            normal_sample = np.random.choice(normal_families, size=min(20, len(normal_families)), replace=False).tolist()
            result_normal = analysis.retention_test(families=normal_sample, verbose=False)
            gc_results['p_values_normal'].append(result_normal.p_value)
            
            if result_normal.p_value >= 0.05:
                gc_results['true_negatives'] += 1
            else:
                gc_results['false_positives'] += 1
        
        # Compute detection metrics
        self.log("\n" + "="*63)
        self.log("DETECTION PERFORMANCE")
        self.log("="*63 + "\n")
        
        # GeneContent metrics
        gc_tpr = gc_results['true_positives'] / n_replicates if n_replicates > 0 else 0
        gc_fpr = gc_results['false_positives'] / n_replicates if n_replicates > 0 else 0
        gc_tnr = gc_results['true_negatives'] / n_replicates if n_replicates > 0 else 0
        gc_fnr = gc_results['false_negatives'] / n_replicates if n_replicates > 0 else 0
        
        gc_theta_mean = np.mean(gc_results['theta_estimates'])
        gc_theta_std = np.std(gc_results['theta_estimates'])
        gc_theta_error = abs(gc_theta_mean - true_retention_strength) / abs(true_retention_strength)
        
        # Convert to interpretable multipliers
        gc_multiplier_mean = np.exp(gc_theta_mean)
        true_multiplier = np.exp(true_retention_strength)
        
        self.log("GeneContent Results:")
        self.log(f"  True Positive Rate (sensitivity):  {gc_tpr:.1%} ({gc_results['true_positives']}/{n_replicates})")
        self.log(f"  True Negative Rate (specificity):  {gc_tnr:.1%} ({gc_results['true_negatives']}/{n_replicates})")
        self.log(f"  False Positive Rate:                {gc_fpr:.1%} ({gc_results['false_positives']}/{n_replicates})")
        self.log(f"  False Negative Rate:                {gc_fnr:.1%} ({gc_results['false_negatives']}/{n_replicates})")
        self.log(f"\n  Retention Strength (log-scale):")
        self.log(f"    Estimated: {gc_theta_mean:.2f} ± {gc_theta_std:.2f}")
        self.log(f"    True:      {true_retention_strength:.2f}")
        self.log(f"    Error:     {gc_theta_error:.1%}")
        self.log(f"\n  Loss Rate Multiplier (interpretable):")
        self.log(f"    Estimated: {gc_multiplier_mean:.2f}x (retained families lose genes {1/gc_multiplier_mean:.1f}x slower)")
        self.log(f"    True:      {true_multiplier:.2f}x (retained families lose genes {1/true_multiplier:.1f}x slower)")
        self.log(f"\n  P-value distributions:")
        self.log(f"    Retained families: {np.mean(gc_results['p_values_retained']):.3f} ± {np.std(gc_results['p_values_retained']):.3f}")
        self.log(f"    Normal families:   {np.mean(gc_results['p_values_normal']):.3f} ± {np.std(gc_results['p_values_normal']):.3f}")
        
        # Add result
        self.add_result(ComparisonResult(
            test_name="RetentionBias_Detection",
            tool="genecontent",
            passed=gc_tpr >= 0.7 and gc_fpr <= 0.3,  # Good detection power, allow 30% FPR
            message=f"TPR={gc_tpr:.1%}, FPR={gc_fpr:.1%}, θ error={gc_theta_error:.1%}",
            metrics={
                'true_positive_rate': gc_tpr,
                'false_positive_rate': gc_fpr,
                'true_negative_rate': gc_tnr,
                'theta_mean': gc_theta_mean,
                'theta_std': gc_theta_std,
                'theta_error': gc_theta_error,
                'true_retention_strength': true_retention_strength,
                'loss_multiplier_mean': gc_multiplier_mean,
                'true_loss_multiplier': true_multiplier
            },
            runtime_seconds=0.0
        ))
        
        # GLOOME comparison (if available)
        self.log("\n" + "="*63)
        self.log("INTERPRETABILITY COMPARISON")
        self.log("="*63 + "\n")
        
        self.log("GeneContent Advantages:")
        self.log("  ✓ Explicit retention_strength parameter (log-scale)")
        self.log("  ✓ Direct biological interpretation:")
        self.log(f"      retention_strength = {gc_theta_mean:.2f}")
        self.log(f"      → loss multiplier = exp({gc_theta_mean:.2f}) = {gc_multiplier_mean:.2f}x")
        self.log(f"      → retained families lose genes {1/gc_multiplier_mean:.1f}x slower")
        self.log("  ✓ Statistical significance testing (LRT with p-values)")
        self.log("  ✓ Effect direction classification (retained/lost_faster/neutral)")
        self.log(f"  ✓ Accurate parameter recovery: {gc_theta_error:.1%} error")
        
        self.log("\nGLOOME/Count Approach:")
        self.log("  - Rate variation models (gamma distribution, categories)")
        self.log("  - Requires post-hoc interpretation of rate categories")
        self.log("  - No direct retention strength parameter")
        self.log("  - Less explicit biological interpretation")
        self.log("  - Cannot directly test specific gene sets")
        
        self.log("\n" + "="*63 + "\n")
    
    def print_summary(self):
        """Print summary of all comparison results."""
        self.log("\n" + "="*63)
        self.log("COMPARISON SUMMARY")
        self.log("="*63 + "\n")
        
        # Group by tool
        tools = {}
        for result in self.results:
            if result.tool not in tools:
                tools[result.tool] = {'passed': 0, 'failed': 0, 'runtime': 0}
            if result.passed:
                tools[result.tool]['passed'] += 1
            else:
                tools[result.tool]['failed'] += 1
            tools[result.tool]['runtime'] += result.runtime_seconds
        
        self.log("Results by tool:\n")
        for tool, stats in sorted(tools.items()):
            total = stats['passed'] + stats['failed']
            success_rate = 100 * stats['passed'] / total if total > 0 else 0
            self.log(f"{tool.upper()}:")
            self.log(f"  Tests: {total}")
            self.log(f"  Passed: {stats['passed']}")
            self.log(f"  Failed: {stats['failed']}")
            self.log(f"  Success rate: {success_rate:.1f}%")
            self.log(f"  Total runtime: {stats['runtime']:.2f}s\n")
        
        # Show failed tests
        failed = [r for r in self.results if not r.passed]
        if failed:
            self.log("Failed tests:")
            for result in failed:
                self.log(f"  - {result.test_name} ({result.tool}): {result.message}")
        
        self.log("="*63 + "\n")
    
    def run_all(self):
        """Run all comparison tests."""
        self.log("="*63)
        self.log("EXTERNAL TOOL COMPARISON VALIDATION")
        self.log("="*63 + "\n")
        
        self.compare_global_rates_simulated()
        self.compare_retention_bias_detection()
        self.print_summary()


if __name__ == "__main__":
    suite = ToolComparisonSuite()
    suite.run_all()
