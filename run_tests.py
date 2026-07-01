#!/usr/bin/env python
"""
Run all tests for the Resume Screening System
"""

import pytest
import sys
import os


def run_tests():
    """Run all tests with coverage"""
    print("=" * 60)
    print("🧪 Running Tests for Resume Screening System")
    print("=" * 60)
    print()

    # Run pytest with coverage
    args = [
        "pytest",
        "tests/",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term",
        "-v",
        "--tb=short",
    ]

    # Add markers if specified
    if len(sys.argv) > 1:
        if sys.argv[1] == "unit":
            args.extend(["-m", "unit"])
        elif sys.argv[1] == "integration":
            args.extend(["-m", "integration"])
        elif sys.argv[1] == "smoke":
            args.extend(["-m", "smoke"])

    exit_code = pytest.main(args)

    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        print("\n📊 Coverage Report: htmlcov/index.html")
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed. Please check the output above.")
        print("=" * 60)

    return exit_code


if __name__ == "__main__":
    sys.exit(run_tests())
