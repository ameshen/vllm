#!/bin/bash
# Quick setup script for vLLM development with P/D disaggregation debugging

set -e  # Exit on error

echo "=========================================="
echo "vLLM Development Setup"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📍 Current directory: $SCRIPT_DIR"
echo ""

# Check if we're in the vLLM directory
if [ ! -f "setup.py" ] || [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: This doesn't appear to be the vLLM root directory"
    echo "   Expected to find setup.py and pyproject.toml"
    exit 1
fi

echo "✅ Found vLLM source directory"
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $PYTHON_VERSION"
echo ""

# Check if vLLM is already installed
if python -c "import vllm" 2>/dev/null; then
    VLLM_PATH=$(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")
    echo "📦 vLLM is currently installed at:"
    echo "   $VLLM_PATH"
    echo ""

    if [ "$VLLM_PATH" == "$SCRIPT_DIR/vllm" ]; then
        echo "✅ Already using editable install from this directory!"
        echo ""
        echo "Your changes will be immediately active."
        echo "Just edit the files and run your scripts."
        exit 0
    else
        echo "⚠️  vLLM is installed but not in editable mode"
        echo ""
        read -p "Uninstall current vLLM and reinstall in editable mode? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Uninstalling current vLLM..."
            pip uninstall vllm -y
        else
            echo "Aborting. You can manually run:"
            echo "  pip uninstall vllm -y"
            echo "  pip install -e ."
            exit 0
        fi
    fi
else
    echo "📦 vLLM is not currently installed"
    echo ""
fi

# Install build dependencies
echo "📥 Installing build dependencies..."
if [ -f "requirements/build.txt" ]; then
    pip install -q -r requirements/build.txt
    echo "✅ Build dependencies installed"
else
    echo "⚠️  requirements/build.txt not found, installing basic deps"
    pip install -q cmake ninja wheel packaging setuptools
fi
echo ""

# Install vLLM in editable mode
echo "🔨 Installing vLLM in editable mode..."
pip install -e .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ vLLM installed successfully in editable mode!"
else
    echo ""
    echo "❌ Installation failed. Please check the error messages above."
    exit 1
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
VLLM_PATH=$(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)

if [ "$VLLM_PATH" == "$SCRIPT_DIR/vllm" ]; then
    echo "✅ SUCCESS! vLLM is using your local source:"
    echo "   $VLLM_PATH"
    echo ""
    echo "✅ Debug breakpoints are active:"
    if grep -q "_DEBUG_P2P" "$SCRIPT_DIR/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py" 2>/dev/null; then
        echo "   Found _DEBUG_P2P in p2p_nccl_engine.py"
    else
        echo "   ⚠️  _DEBUG_P2P not found (you may need to apply changes)"
    fi
else
    echo "⚠️  Warning: vLLM path doesn't match expected source directory"
    echo "   Expected: $SCRIPT_DIR/vllm"
    echo "   Got:      $VLLM_PATH"
fi

echo ""
echo "=========================================="
echo "🎉 Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Make your changes to files in:"
echo "   $SCRIPT_DIR/vllm/"
echo ""
echo "2. Your changes are immediately active! Just run your scripts:"
echo "   export VLLM_P2P_DEBUG=1"
echo "   python your_pd_disagg_script.py"
echo ""
echo "3. To verify your changes are active:"
echo "   python -c 'import vllm; print(vllm.__file__)'"
echo "   Should show: $SCRIPT_DIR/vllm/__init__.py"
echo ""
echo "Happy debugging! 🐛"

