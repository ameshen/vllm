# How to Apply Your P/D Disaggregation Changes to vLLM

## 🎯 The Problem

You've made changes to vLLM source files (added debug breakpoints to `p2p_nccl_engine.py`), but after installing vLLM via pip, those changes aren't reflected when you run your code.

## 📦 Understanding vLLM Installation

When you do `pip install vllm` or `pip install -r requirements/xxx.txt`, pip installs vLLM as a package in your Python site-packages directory, NOT from your local source directory.

Your changes are in: `/Users/amesshen/MLSys/vllm/vllm/distributed/kv_transfer/...`

But Python is using the installed package from: `~/anaconda3/envs/your_env/lib/python3.x/site-packages/vllm/...`

## ✅ Solution 1: Install in Editable/Development Mode (RECOMMENDED)

This is the **best approach** for development - your changes are immediately reflected without reinstalling.

### Step 1: Uninstall the existing vLLM
```bash
pip uninstall vllm -y
```

### Step 2: Install vLLM in editable mode from your local source
```bash
cd /Users/amesshen/MLSys/vllm

# Install in editable mode with dependencies
pip install -e .

# Or if you need specific extras (like CUDA, development tools, etc.)
pip install -e ".[cuda]"
```

### What This Does:
- Creates a symbolic link from your Python environment to your source directory
- Any changes you make to the source files are **immediately active**
- No need to reinstall after making changes
- Perfect for development and debugging

### Verify It's Working:
```python
import vllm
print(vllm.__file__)
# Should show: /Users/amesshen/MLSys/vllm/vllm/__init__.py
```

## 🔄 Solution 2: Reinstall After Each Change

If you don't want editable mode, you can reinstall after making changes:

```bash
cd /Users/amesshen/MLSys/vllm
pip install --force-reinstall --no-deps .
```

⚠️ **Downsides:**
- Must reinstall after every change
- Takes time to rebuild
- Easy to forget and run stale code

## 📋 Solution 3: Copy Files to Installed Location

If you can't use editable mode, you can manually copy modified files:

### Step 1: Find where vLLM is installed
```bash
python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))"
```

This will output something like:
```
/Users/amesshen/anaconda3/envs/your_env/lib/python3.11/site-packages/vllm
```

### Step 2: Copy your modified file
```bash
# Set the installed location (from step 1)
VLLM_INSTALLED="/Users/amesshen/anaconda3/envs/your_env/lib/python3.11/site-packages/vllm"

# Copy the modified file
cp /Users/amesshen/MLSys/vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py \
   $VLLM_INSTALLED/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py

# Make sure permissions are correct
chmod 644 $VLLM_INSTALLED/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py
```

### Step 3: Verify the change
```bash
grep "_DEBUG_P2P" $VLLM_INSTALLED/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py
```

⚠️ **Downsides:**
- Must copy after every change
- Easy to lose track of what you've modified
- Changes lost if you reinstall vLLM
- **NOT RECOMMENDED for development**

## 🛠️ Step-by-Step: Setting Up for Development

Here's the complete workflow I recommend:

### 1. Clone/Navigate to vLLM source
```bash
cd /Users/amesshen/MLSys/vllm
```

### 2. Create a virtual environment (if not already done)
```bash
# Using conda (recommended)
conda create -n vllm-dev python=3.11
conda activate vllm-dev

# Or using venv
python -m venv vllm-dev
source vllm-dev/bin/activate  # On macOS/Linux
```

### 3. Install build dependencies
```bash
pip install -r requirements/build.txt
```

### 4. Install vLLM in editable mode
```bash
# Basic installation
pip install -e .

# Or with CUDA support
pip install -e ".[cuda]"

# Or with all development tools
pip install -e ".[dev]"
```

### 5. Verify installation
```bash
python -c "import vllm; print('vLLM version:', vllm.__version__); print('Location:', vllm.__file__)"
```

Should output:
```
vLLM version: 0.x.x+xxx
Location: /Users/amesshen/MLSys/vllm/vllm/__init__.py
```

### 6. Make your changes and test immediately
```bash
# Edit the file
vim vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py

# Run your test - changes are already active!
export VLLM_P2P_DEBUG=1
python your_pd_disagg_script.py
```

## 🔍 Verifying Your Changes Are Active

### Quick Check Script
Create a file `check_vllm_source.py`:

```python
#!/usr/bin/env python3
"""Check if vLLM is using your modified source code."""

import os
import sys

def check_vllm_installation():
    try:
        import vllm
        vllm_path = os.path.dirname(vllm.__file__)
        
        print("=" * 80)
        print("vLLM Installation Check")
        print("=" * 80)
        print(f"vLLM version: {vllm.__version__}")
        print(f"vLLM location: {vllm_path}")
        
        # Check if it's your source directory
        expected_source = "/Users/amesshen/MLSys/vllm/vllm"
        if vllm_path == expected_source:
            print("✅ Using LOCAL SOURCE (editable install)")
            print("   Changes to source files will be immediately active!")
        elif "site-packages" in vllm_path:
            print("❌ Using INSTALLED PACKAGE")
            print("   Changes to source files will NOT be active!")
            print("   Solution: Run 'pip install -e .' in your vLLM directory")
        else:
            print("⚠️  Unknown installation type")
        
        # Check for debug breakpoints
        engine_file = os.path.join(vllm_path, 
                                   "distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py")
        if os.path.exists(engine_file):
            with open(engine_file, 'r') as f:
                content = f.read()
                if "_DEBUG_P2P" in content:
                    print("✅ Debug breakpoints are present in the file")
                else:
                    print("❌ Debug breakpoints NOT found!")
                    print("   The file may not have your changes")
        else:
            print("❌ p2p_nccl_engine.py not found!")
        
        print("=" * 80)
        
    except ImportError:
        print("❌ vLLM is not installed!")
        sys.exit(1)

if __name__ == "__main__":
    check_vllm_installation()
```

Run it:
```bash
python check_vllm_source.py
```

## 🐛 Troubleshooting

### Issue: "pip install -e ." fails

**Solution**: Make sure you have build dependencies:
```bash
pip install -r requirements/build.txt
pip install cmake ninja wheel
```

### Issue: ImportError after editable install

**Solution**: Rebuild the C++ extensions:
```bash
cd /Users/amesshen/MLSys/vllm
pip install -e . --force-reinstall --no-deps
```

### Issue: Changes not reflected even with editable install

**Solution**: Python may have cached the module:
```python
# In your script, before importing vllm:
import sys
# Remove any cached vllm modules
for module in list(sys.modules.keys()):
    if module.startswith('vllm'):
        del sys.modules[module]

# Now import
import vllm
```

Or restart your Python interpreter.

### Issue: Multiple vLLM installations

**Solution**: Check all installations:
```bash
pip list | grep vllm
conda list | grep vllm

# Uninstall all
pip uninstall vllm -y
conda uninstall vllm -y

# Reinstall in editable mode
cd /Users/amesshen/MLSys/vllm
pip install -e .
```

## 📝 Quick Reference Commands

```bash
# Install vLLM in editable mode (do this once)
cd /Users/amesshen/MLSys/vllm
pip install -e .

# Check what vLLM Python is using
python -c "import vllm; print(vllm.__file__)"

# Check if debug code is present
grep "_DEBUG_P2P" vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py

# Run your script with changes active
export VLLM_P2P_DEBUG=1
python your_script.py

# If you need to rebuild C++ extensions (after changing .cpp/.cu files)
pip install -e . --force-reinstall --no-deps
```

## 🎯 Recommended Workflow

1. **One-time setup:**
   ```bash
   cd /Users/amesshen/MLSys/vllm
   pip uninstall vllm -y
   pip install -e .
   ```

2. **Daily development:**
   ```bash
   # Edit files in /Users/amesshen/MLSys/vllm/vllm/...
   vim vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py
   
   # Run immediately - changes are active!
   python your_script.py
   ```

3. **If you change C++/CUDA files (`.cpp`, `.cu`):**
   ```bash
   pip install -e . --force-reinstall --no-deps
   ```

## ✨ Benefits of Editable Install

- ✅ Changes are immediately active
- ✅ No need to reinstall after Python file changes
- ✅ Easy to test and debug
- ✅ Can use version control (git) on your changes
- ✅ Standard practice for Python development
- ✅ Works with debuggers and IDEs

## 🚀 Summary

**Best Practice**: Use editable install (`pip install -e .`)

```bash
cd /Users/amesshen/MLSys/vllm
pip uninstall vllm -y
pip install -e .
```

Then your debug breakpoints and any other changes are immediately active! 🎉

