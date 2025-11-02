# Will `pip install -e .` Install Required Packages?

## 🎯 Short Answer

**YES!** ✅

`pip install -e .` will install **all required dependencies** automatically.

## 📦 What Gets Installed

When you run `pip install -e .`, setuptools reads `setup.py` which calls `get_requirements()` and installs:

### 1. Core Dependencies (from `requirements/common.txt`)
- **torch** (PyTorch)
- **transformers** >= 4.56.0
- **tokenizers** >= 0.21.1
- **numpy**
- **fastapi** >= 0.115.0
- **aiohttp**
- **pydantic** >= 2.12.0
- **pillow** (for image processing)
- **pyzmq** >= 25.0.0 ⭐ (Required for P/D disaggregation!)
- **msgspec**
- **tiktoken**
- **outlines_core**
- **xgrammar**
- **opencv-python-headless** (for video processing)
- **pyyaml**
- **einops**
- **prometheus_client**
- **openai** >= 1.99.1
- **sentencepiece**
- **requests**
- And 40+ more packages...

### 2. Platform-Specific Dependencies

Depending on your hardware, it also installs from:

| Hardware | Requirements File | Key Packages |
|----------|------------------|--------------|
| CUDA GPU | `requirements/cuda.txt` | vllm-flash-attn, nvidia packages |
| ROCm (AMD) | `requirements/rocm.txt` | ROCm-specific packages |
| CPU only | `requirements/cpu.txt` | CPU-optimized packages |
| TPU | `requirements/tpu.txt` | TPU packages |
| Intel XPU | `requirements/xpu.txt` | Intel XPU packages |

### 3. Build Dependencies

The build process uses (from `pyproject.toml`):
- **cmake** >= 3.26.1
- **ninja**
- **wheel**
- **setuptools** >= 77.0.3
- **packaging** >= 24.2

## 🔍 How It Works

Here's what happens under the hood:

```python
# In setup.py line 582-623:
def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR / "requirements"
    
    # Detects your hardware (CUDA/ROCm/CPU/etc.)
    if _is_cuda():
        requirements = _read_requirements("cuda.txt")
    elif _is_hip():
        requirements = _read_requirements("rocm.txt")
    # ... etc
    
    return requirements

# In setup.py line 708:
setup(
    install_requires=get_requirements(),  # ← All dependencies installed here!
    extras_require={
        "bench": ["pandas", "matplotlib", "seaborn"],
        "tensorizer": ["tensorizer==2.10.1"],
        "audio": ["librosa", "soundfile"],
        # ... optional extras
    },
)
```

## ✅ Verification

After running `pip install -e .`, verify packages are installed:

```bash
# Check if key P/D disaggregation dependencies are installed
python -c "import zmq; print('✅ pyzmq:', zmq.__version__)"
python -c "import msgspec; print('✅ msgspec:', msgspec.__version__)"
python -c "import torch; print('✅ torch:', torch.__version__)"
python -c "import vllm; print('✅ vllm:', vllm.__version__)"

# List all vllm-related packages
pip list | grep -i vllm
```

## 📋 Complete Installation Command

```bash
cd /Users/amesshen/MLSys/vllm

# Option 1: Basic installation (recommended for most users)
pip install -e .

# Option 2: With optional extras
pip install -e ".[bench,audio]"

# Option 3: If you need CUDA-specific packages
pip install -e ".[cuda]"
```

## 🎁 What About Optional Dependencies?

Some packages are **optional extras** and NOT installed by default:

| Extra | Install Command | Packages |
|-------|----------------|----------|
| `bench` | `pip install -e ".[bench]"` | pandas, matplotlib, seaborn, datasets |
| `audio` | `pip install -e ".[audio]"` | librosa, soundfile |
| `tensorizer` | `pip install -e ".[tensorizer]"` | tensorizer==2.10.1 |
| `petit-kernel` | `pip install -e ".[petit-kernel]"` | petit-kernel (AMD FP4 quantization) |

### For P/D Disaggregation

You **DON'T need any extras**! The core dependencies include everything needed:
- ✅ `pyzmq` >= 25.0.0 (for ZMQ communication)
- ✅ `msgspec` (for message serialization)
- ✅ NCCL (comes with PyTorch/CUDA)

## 🔧 Troubleshooting

### Issue: "No matching distribution found for ..."

**Cause:** Some platform-specific dependencies aren't available.

**Solution:**
```bash
# Install build dependencies first
pip install -r requirements/build.txt

# Then install vLLM
pip install -e .
```

### Issue: CUDA version mismatch

**Cause:** Your CUDA version doesn't match PyTorch's CUDA version.

**Solution:**
```bash
# Check versions
python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"
nvcc --version

# If mismatched, reinstall PyTorch for your CUDA version
# Example for CUDA 12.1:
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Build fails with "cmake not found"

**Solution:**
```bash
pip install cmake ninja wheel
pip install -e .
```

## 📊 Dependency Tree for P/D Disaggregation

Here's what P/D disaggregation specifically needs:

```
vLLM P/D Disaggregation
├── vllm (core)
│   ├── torch >= 2.0 (provides NCCL)
│   ├── pyzmq >= 25.0.0 (ZMQ for control messages)
│   ├── msgspec (fast serialization)
│   └── transformers (model loading)
│
├── CUDA/NCCL (tensor communication)
│   └── Installed with torch+cu121/cu118
│
└── Your changes
    └── p2p_nccl_engine.py (debug breakpoints)
```

All installed automatically with `pip install -e .` ✅

## 🎓 Summary

### Question: Will `pip install -e .` install required packages?

**Answer: YES!** It installs:

1. ✅ All core dependencies (~50 packages from `requirements/common.txt`)
2. ✅ Platform-specific dependencies (CUDA/ROCm/CPU packages)
3. ✅ Build tools (cmake, ninja, etc.)
4. ✅ Everything needed for P/D disaggregation (`pyzmq`, `msgspec`, NCCL)
5. ✅ Your source code in editable mode (changes immediately active)

### What it does NOT install (by default):

- ❌ Optional extras like `bench`, `audio`, `tensorizer`
- ❌ Development tools like linters (install with `pip install -e ".[dev]"`)

### For P/D Disaggregation:

```bash
cd /Users/amesshen/MLSys/vllm
pip install -e .  # ← This is ALL you need!

# Verify
python -c "import pyzmq, msgspec, torch; print('✅ All P/D disaggregation deps installed!')"

# Run your script
export VLLM_P2P_DEBUG=1
python your_pd_disagg_script.py
```

**No need to install anything else!** 🎉

## 💡 Pro Tip

If you previously installed vLLM via `pip install vllm`, uninstall it first to avoid conflicts:

```bash
pip uninstall vllm -y
pip install -e .
```

This ensures you're using your modified source with all dependencies properly installed.

