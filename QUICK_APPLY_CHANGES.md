# Quick Start: Apply Your P/D Disaggregation Changes

## 🚀 TL;DR - Do This Now

```bash
cd /Users/amesshen/MLSys/vllm
pip uninstall vllm -y
pip install -e .
```

✅ Done! Your changes are now active.

**Note:** This also installs ALL required dependencies (PyTorch, pyzmq, msgspec, transformers, etc.) automatically! You don't need to separately install anything.

## 📋 What This Does

- Uninstalls any existing vLLM installation
- Installs vLLM in "editable mode" (development mode)
- Creates a symbolic link from Python to your source directory
- **All your changes are immediately active** - no need to reinstall!

## ✅ Verify It Worked

```bash
python -c "import vllm; print(vllm.__file__)"
```

Should output: `/Users/amesshen/MLSys/vllm/vllm/__init__.py`

If it shows something with `site-packages`, it didn't work - see troubleshooting below.

## 🎯 Now Test Your Changes

```bash
# Your debug breakpoints are now active!
export VLLM_P2P_DEBUG=1
export NCCL_DEBUG=INFO

python your_pd_disagg_script.py
```

You should see logs like:
```
INFO [p2p_nccl_engine.py:105] 🔍 [Rank 0] P2pNcclEngine.__init__ START | local_rank=0, pid=12345
```

## 📝 What Changed

Before: `pip install vllm` → Python uses `/path/to/site-packages/vllm/`
After: `pip install -e .` → Python uses `/Users/amesshen/MLSys/vllm/vllm/`

## 🔄 Workflow

### One-Time Setup (already done above)
```bash
pip install -e .
```

### Daily Development (repeat as needed)
```bash
# 1. Edit files
vim vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py

# 2. Run immediately - changes are active!
python your_script.py
```

That's it! No reinstall needed for Python files.

## ⚠️ When You DO Need to Reinstall

Only if you change C++/CUDA files (`.cpp`, `.cu`, `.cuh`):

```bash
pip install -e . --force-reinstall --no-deps
```

For Python files (`.py`): **No reinstall needed!**

## 🐛 Troubleshooting

### "pip install -e ." fails with build errors

Install build dependencies first:
```bash
pip install -r requirements/build.txt
pip install cmake ninja wheel
pip install -e .
```

### Changes still not reflected

1. Check installation type:
   ```bash
   python -c "import vllm; print(vllm.__file__)"
   ```
   Must show your source directory, not site-packages.

2. Restart Python interpreter (exit and reopen)

3. Clear Python cache:
   ```python
   import sys
   for mod in list(sys.modules.keys()):
       if mod.startswith('vllm'):
           del sys.modules[mod]
   import vllm  # Now reimport
   ```

### Multiple Python environments

Make sure you're in the right environment:
```bash
which python
# Should show your conda/venv path

conda activate your_env_name
# or
source your_venv/bin/activate
```

## 🎁 Bonus: Automated Setup

Run this script to automate everything:
```bash
cd /Users/amesshen/MLSys/vllm
bash setup_dev_mode.sh
```

## 📚 More Details

See `HOW_TO_APPLY_CHANGES.md` for comprehensive documentation.

## ✨ Summary

**Before your changes:**
- vLLM installed via pip → using site-packages
- Your source changes NOT active ❌

**After `pip install -e .`:**
- vLLM installed in editable mode → using your source directory
- Your source changes IMMEDIATELY active ✅
- Edit files and run - that's it! 🎉

---

**Remember**: You've already added debug breakpoints to `p2p_nccl_engine.py`, so after doing `pip install -e .`, just set `VLLM_P2P_DEBUG=1` and run your script!

