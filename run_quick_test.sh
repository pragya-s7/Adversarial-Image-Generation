#!/bin/bash
# Quick End-to-End Test Script
# This runs all the baseline tests to validate the setup

echo "======================================================================"
echo "Grounded Attention MVP - Quick Validation Test"
echo "======================================================================"
echo ""

# Change to project directory
cd "$(dirname "$0")"

echo "📁 Current directory: $(pwd)"
echo ""

# Test 1: Core grounded attention
echo "Test 1/3: Testing Grounded Attention Module..."
echo "----------------------------------------------------------------------"
python src/models/grounded_attention.py
if [ $? -eq 0 ]; then
    echo "✅ Grounded Attention: PASSED"
else
    echo "❌ Grounded Attention: FAILED"
    exit 1
fi
echo ""

# Test 2: Loss functions
echo "Test 2/3: Testing Loss Functions..."
echo "----------------------------------------------------------------------"
python src/training/losses.py
if [ $? -eq 0 ]; then
    echo "✅ Loss Functions: PASSED"
else
    echo "❌ Loss Functions: FAILED"
    exit 1
fi
echo ""

# Test 3: Dataset module
echo "Test 3/3: Testing Dataset Module..."
echo "----------------------------------------------------------------------"
python src/data/datasets.py
if [ $? -eq 0 ]; then
    echo "✅ Dataset Module: PASSED"
else
    echo "❌ Dataset Module: FAILED"
    exit 1
fi
echo ""

echo "======================================================================"
echo "✅ ALL TESTS PASSED!"
echo "======================================================================"
echo ""
echo "Your Grounded Attention MVP is ready to use!"
echo ""
echo "Next steps:"
echo "  1. Run inference: python scripts/evaluate_simple.py --image_path <your_image.jpg>"
echo "  2. Download COCO: See MVP_README.md for instructions"
echo "  3. Train model: python scripts/train_minimal.py --help"
echo ""
echo "Documentation:"
echo "  - Quick start: MVP_README.md"
echo "  - Test results: MVP_TEST_RESULTS.md"
echo "  - Full summary: BASELINE_SUMMARY.md"
echo ""
echo "======================================================================"
