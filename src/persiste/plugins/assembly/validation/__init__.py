"""
Validation and robustness experiments for the Assembly plugin.

This package houses the current (Reboot) iteration of validation tooling:

- validation.scripts: Reusable CLI utilities (dataset generation, benchmarking)
- validation.experiments: Notebook-like scripts that run full validation suites
- validation.results: Canonical outputs captured from the latest runs

Historical V1/V2 artifacts live under src/persiste/plugins/assembly/wip/.
"""

__all__ = ["experiments", "results", "scripts"]
