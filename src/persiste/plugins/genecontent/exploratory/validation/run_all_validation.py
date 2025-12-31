#!/usr/bin/env python
"""Streamlined validation runner for GeneContent plugin."""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from level1_mechanical import MechanicalTests


def run_validation():
    """Run all validation tests."""
    print("=" * 80)
    print("GENECONTENT PLUGIN VALIDATION")
    print("=" * 80)
    print()
    
    all_results = []
    total_passed = 0
    total_failed = 0
    
    # Level 1: Mechanical Correctness
    print("\n" + "=" * 80)
    print("LEVEL 1: MECHANICAL CORRECTNESS")
    print("=" * 80)
    level1 = MechanicalTests()
    passed, failed, results = level1.run_all()
    all_results.extend(results)
    total_passed += passed
    total_failed += failed
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print("Total tests: {0}".format(total_passed + total_failed))
    print("Passed:      {0}".format(total_passed))
    print("Failed:      {0}".format(total_failed))
    print()
    
    if total_failed == 0:
        print("[PASS] ALL VALIDATION TESTS PASSED")
        print("System is ready for use.")
    else:
        print("[FAIL] SOME TESTS FAILED")
        print("Review failures before using with real data.")
    
    print("=" * 80)
    
    # Save report
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / "validation_report_{0}.txt".format(timestamp)
    
    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GENECONTENT PLUGIN VALIDATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write("Generated: {0}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write("Total tests: {0}\n".format(total_passed + total_failed))
        f.write("Passed:      {0}\n".format(total_passed))
        f.write("Failed:      {0}\n".format(total_failed))
        f.write("\n")
        
        if total_failed == 0:
            f.write("[PASS] ALL VALIDATION TESTS PASSED\n\n")
            f.write("The GeneContent plugin has passed validation.\n")
            f.write("The system is ready for use.\n\n")
            f.write("Next steps:\n")
            f.write("1. Run inference on real datasets\n")
            f.write("2. Compare results with existing methods\n")
            f.write("3. Document findings for publication\n")
        else:
            f.write("[FAIL] SOME VALIDATION TESTS FAILED\n\n")
            f.write("Review failures before using with real data.\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for line in all_results:
            f.write(line + "\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print("\nValidation report saved to: {0}".format(report_file))
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_validation())
