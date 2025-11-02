# Quick Start: Debugging Your P/D Disaggregation Issue

## 🚨 THE BUG IN YOUR CODE

**Line that's causing the hang:**
```python
ktc = KVTransferConfig(
    kv_connector="P2pNcclConnector",
    kv_role="kv_consumer",
    kv_rank=2,  # ❌ WRONG! This should be 1
    kv_parallel_size=2,
)
```

**Fixed version:**
```python
ktc = KVTransferConfig(
    kv_connector="P2pNcclConnector",
    kv_role="kv_consumer",
    kv_rank=1,  # ✅ CORRECT
    kv_parallel_size=2,
)
```

## Why This Causes a Hang

The P2P NCCL connector uses `ncclCommInitRank()`, which is a **collective operation**. Both sides must:
1. Call it with the same `nprocs` (2 in your case)
2. Use valid ranks: 0 and 1 for a 2-node setup

When you use `kv_rank=2`:
- Prefill calls `ncclCommInitRank(2, unique_id, rank=0)` ✅ Valid
- Decode calls `ncclCommInitRank(2, unique_id, rank=2)` ❌ Invalid! (rank must be < nprocs)
- NCCL waits forever for rank 1 to join, causing both sides to hang

## Quick Fix

1. **Change your decode function:**
```python
def run_decode(prefill_done, decode_done, model_path, prompt, sampling_params, trace_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_consumer",
        kv_rank=1,  # Changed from 2 to 1
        kv_parallel_size=2,
    )
    # ... rest of your code
```

2. **Test it:**
```bash
python your_script.py
```

## If It Still Hangs

### Step 1: Enable Debug Logging
```bash
export VLLM_P2P_DEBUG=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### Step 2: Run Your Script
The instrumented code I added to vLLM will now print detailed debug messages showing:
- When each node initializes
- When ZMQ connections are established
- When NCCL initialization starts (critical hang point)
- When NCCL initialization completes

### Step 3: Check the Logs
Look for these messages:

**Expected flow (working):**
```
[Rank 0] P2pNcclEngine.__init__ START
[Rank 0] ZMQ Router socket bound successfully
[Rank 0] Listener thread started
[Rank 1] P2pNcclEngine.__init__ START
[Rank 1] ZMQ Router socket bound successfully
[Rank 1] Listener thread started
[Rank 1] Listener thread RUNNING, starting to poll
[Rank 0] create_connect START
[Rank 0] ZMQ DEALER connected
[Rank 0] Sending NEW command via ZMQ
[Rank 1] DECODE: Received NEW command
[Rank 0] PREFILL: About to call ncclCommInitRank  👈 CRITICAL POINT
[Rank 1] DECODE: About to call ncclCommInitRank  👈 CRITICAL POINT
[Rank 1] DECODE: ncclCommInitRank SUCCESS  👈 Should happen quickly
[Rank 0] PREFILL: ncclCommInitRank SUCCESS  👈 Should happen quickly
```

**If it hangs:**
Look for where it stops. Common patterns:
- Stops before "Listener thread RUNNING" → Decode node's listener didn't start
- Stops at "Sending NEW command" → ZMQ connection failed
- Stops at "About to call ncclCommInitRank" → Most likely the kv_rank bug

## Files I Created for You

1. **`DEBUG_PD_DISAGG.md`** - Comprehensive debugging guide with:
   - Architecture overview
   - All critical debugging points
   - Common hang scenarios and solutions

2. **`debug_breakpoints.py`** - Python module with instrumented debugging functions
   - Can be imported into vLLM code for detailed tracing
   - Provides timeline tracking
   - Thread analysis utilities

3. **`corrected_pd_disagg_example.py`** - Working example with the fix applied
   - Shows correct configuration
   - Has helpful debug output
   - Ready to run

4. **Instrumented vLLM code** - I added debug logging to:
   - `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py`
   - 13 strategic breakpoints at critical execution points
   - Enabled with `VLLM_P2P_DEBUG=1`

## Debugging Workflow

### Quick Check (5 minutes)
1. Fix the kv_rank bug (change 2 → 1)
2. Run your script
3. If it works, you're done! 🎉

### If Still Stuck (15 minutes)
1. Set environment variables:
   ```bash
   export VLLM_P2P_DEBUG=1
   export NCCL_DEBUG=INFO
   ```
2. Run your script
3. Check where it hangs in the log output
4. Refer to `DEBUG_PD_DISAGG.md` for that specific hang scenario

### Deep Debugging (30+ minutes)
1. Read `DEBUG_PD_DISAGG.md` sections:
   - "Architecture Overview"
   - "Critical Debugging Points"
   - Specific hang scenario matching your issue
2. Use a Python debugger (pdb/PyCharm/VSCode):
   ```python
   import pdb; pdb.set_trace()  # Add at suspected hang point
   ```
3. Check thread states with the utility in `debug_breakpoints.py`:
   ```python
   from debug_breakpoints import check_hanging_threads
   check_hanging_threads()
   ```

## Common Issues and Solutions

### Issue: "Address already in use"
**Cause:** Previous run didn't clean up properly
**Solution:**
```bash
# Kill any remaining processes
pkill -f "python.*vllm"
# Or reboot if necessary
```

### Issue: NCCL timeout
**Cause:** Firewall or network issue
**Solution:**
```bash
# Check if localhost works
ping 127.0.0.1
# Try using explicit IP in config
ktc = KVTransferConfig(
    kv_connector="P2pNcclConnector",
    kv_role="kv_consumer",
    kv_rank=1,
    kv_parallel_size=2,
    kv_ip="127.0.0.1",  # Explicit localhost
)
```

### Issue: CUDA out of memory
**Cause:** Buffer size too large
**Solution:**
```python
ktc = KVTransferConfig(
    kv_connector="P2pNcclConnector",
    kv_role="kv_consumer",
    kv_rank=1,
    kv_parallel_size=2,
    kv_buffer_size=1e8,  # Reduce buffer size (default is 1e9)
)
```

## Understanding the Architecture

### What is P/D Disaggregation?
- **Prefill Node**: Processes input tokens, generates KV cache
- **Decode Node**: Receives KV cache, generates output tokens
- **Benefit**: Can optimize each stage independently (e.g., larger batch for prefill)

### How Communication Works
```
Prefill (GPU 0)                    Decode (GPU 1)
===============                    ==============
1. Initialize                      1. Initialize
2. Bind ZMQ router (:14579)       2. Bind ZMQ router (:14580)
3. Start listener thread          3. Start listener thread
4. Generate input                  4. Wait...
5. Connect to decode (:14580) →   5. Receive connection
6. Send NEW command via ZMQ   →   6. Receive NEW, extract unique_id
7. Call ncclCommInitRank(0)   ↔   7. Call ncclCommInitRank(1)
   [Both block until both call]
8. NCCL comms established      ✓   8. NCCL comms established
9. Send KV via NCCL           →   9. Receive KV via NCCL
10. Signal done                →  10. Start decoding
```

### Key Synchronization Points
1. **ZMQ Connection**: Prefill → Decode
2. **NCCL Init**: Both must call together (collective operation) ⚠️
3. **KV Transfer**: Prefill sends, Decode receives
4. **Completion**: Both must wait for each other

## Next Steps

1. **Apply the fix** (change kv_rank=2 to kv_rank=1)
2. **Test it**
3. **If it works**: Celebrate! 🎉
4. **If it doesn't**: Enable debug logging and check where it hangs
5. **Still stuck?**: Read the relevant section in `DEBUG_PD_DISAGG.md`

## Getting Help

When asking for help, include:
1. The debug logs (with `VLLM_P2P_DEBUG=1` and `NCCL_DEBUG=INFO`)
2. Where exactly it hangs (last log message printed)
3. Your configuration (kv_rank, kv_parallel_size, etc.)
4. GPU setup (how many GPUs, which ones are visible)

Good luck! The fix should be simple - just change that one number from 2 to 1. 🚀

