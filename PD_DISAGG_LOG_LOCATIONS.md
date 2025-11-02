# P/D Disaggregation Log Locations

## 📍 Where Logs Are Written

### Standard Output (stdout/stderr)

By default, all P/D disaggregation logs are written to **standard output (stdout)**. This means:

1. **If you run from terminal**: Logs appear in your terminal window
2. **If you redirect output**: `python script.py > output.log 2>&1`
3. **If using subprocess/multiprocessing**: Each process writes to its own stdout

### Log Configuration

The logging is configured in **`vllm/logger.py`**:

```python
DEFAULT_LOGGING_CONFIG = {
    "handlers": {
        "vllm": {
            "class": "logging.StreamHandler",  # ← Writes to stream
            "formatter": "vllm",
            "level": VLLM_LOGGING_LEVEL,       # ← Set via env var
            "stream": VLLM_LOGGING_STREAM,     # ← Default: sys.stderr
        },
    },
    # ...
}
```

### Environment Variables That Control Logging

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `VLLM_LOGGING_LEVEL` | Log level | `"INFO"` | `export VLLM_LOGGING_LEVEL=DEBUG` |
| `VLLM_CONFIGURE_LOGGING` | Enable vLLM logging | `1` | `export VLLM_CONFIGURE_LOGGING=1` |
| `VLLM_LOGGING_CONFIG_PATH` | Custom config path | None | `export VLLM_LOGGING_CONFIG_PATH=/path/to/config.json` |
| `VLLM_P2P_DEBUG` | Enable P2P debug logs | `0` | `export VLLM_P2P_DEBUG=1` |
| `NCCL_DEBUG` | NCCL logging level | None | `export NCCL_DEBUG=INFO` |

## 🔍 P/D Disaggregation Log Sources

### 1. P2P NCCL Engine Logs

**File**: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py`

**Logger name**: `vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine`

**Key log messages**:
```python
# Line 30: Logger initialization
logger = logging.getLogger(__name__)

# Line 226: Engine initialization
logger.info(
    "💯P2pNcclEngine init, rank:%d, local_rank:%d, http_address:%s, "
    "zmq_address:%s, proxy_address:%s, send_type:%s, buffer_size_"
    "threshold:%.2f, nccl_num_channels:%s",
    self.rank, self.local_rank, ...
)

# Line 287: NCCL comm success
logger.info(
    "🤝ncclCommInitRank Success, %s👉%s, MyRank:%s",
    self.zmq_address, remote_address, rank,
)

# Line 473: NCCL comm success (decode side)
logger.info(
    "🤝ncclCommInitRank Success, %s👈%s, MyRank:%s",
    self.zmq_address, remote_address.decode(), rank,
)
```

**Debug logs** (enabled with `VLLM_P2P_DEBUG=1`):
```python
# Line 42: Debug helper function
def _debug_log(rank: int, location: str, **kwargs):
    """Helper function for debug logging."""
    if _DEBUG_P2P:
        details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"🔍 [Rank {rank}] {location} | {details}")

# Called at 13 critical breakpoints throughout the code
```

### 2. P2P NCCL Connector Logs

**File**: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`

**Logger name**: `vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector`

**Key log messages**:
```python
# Line 30: Logger initialization
logger = init_logger(__name__)

# Various warning/debug messages for KV cache operations
logger.warning("🚧kv_cache does not match, block_ids:%d, num_block:%d, request_id:%s", ...)
logger.warning("🚧kv_cache is None, %s", request_id)
```

## 📝 Log Output Format

### Standard vLLM Log Format
```
INFO 11-01 10:30:45 [p2p_nccl_engine.py:226] 💯P2pNcclEngine init, rank:0, local_rank:0, ...
│    │    │          │                  │     │
│    │    │          │                  │     └─ Log message
│    │    │          │                  └─ Line number
│    │    │          └─ File name
│    │    └─ Time (HH:MM:SS)
│    └─ Date (MM-DD)
└─ Log level
```

### Debug Log Format (when `VLLM_P2P_DEBUG=1`)
```
INFO 11-01 10:30:45 [p2p_nccl_engine.py:105] 🔍 [Rank 0] P2pNcclEngine.__init__ START | local_rank=0, pid=12345
│    │    │          │                  │     │  │        │                                │
│    │    │          │                  │     │  │        │                                └─ Details
│    │    │          │                  │     │  │        └─ Location description
│    │    │          │                  │     │  └─ Rank identifier
│    │    │          │                  │     └─ Debug marker
│    │    │          │                  └─ Line number
│    │    │          └─ File name
│    │    └─ Time
│    └─ Date
└─ Log level
```

## 🎯 How to Capture Logs

### Option 1: Direct to Terminal (Default)
```bash
python your_script.py
```
Output appears in terminal.

### Option 2: Redirect to File
```bash
# Redirect all output to file
python your_script.py > vllm_output.log 2>&1

# Redirect and view simultaneously (tee)
python your_script.py 2>&1 | tee vllm_output.log
```

### Option 3: Separate Logs for Each Process
When using multiprocessing (prefill + decode):

```python
import sys
from multiprocessing import Process

def run_prefill(...):
    # Redirect prefill logs to file
    sys.stdout = open('/tmp/prefill.log', 'w')
    sys.stderr = sys.stdout
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # ... rest of prefill code

def run_decode(...):
    # Redirect decode logs to file
    sys.stdout = open('/tmp/decode.log', 'w')
    sys.stderr = sys.stdout
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # ... rest of decode code

def main():
    prefill_process = Process(target=run_prefill, args=(...))
    decode_process = Process(target=run_decode, args=(...))
    
    prefill_process.start()
    decode_process.start()
    
    # Logs will be in:
    # - /tmp/prefill.log
    # - /tmp/decode.log
```

### Option 4: Use Python Logging to File

```python
import logging

# Add file handler to root vllm logger
vllm_logger = logging.getLogger('vllm')
file_handler = logging.FileHandler('/tmp/vllm_pd_disagg.log')
file_handler.setFormatter(logging.Formatter(
    '%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s'
))
vllm_logger.addHandler(file_handler)

# Now all vllm logs will also go to file
```

## 🚀 Quick Start: View P/D Disaggregation Logs

### Minimal Debug Setup
```bash
# Enable debug mode
export VLLM_P2P_DEBUG=1
export VLLM_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=INFO

# Run and save logs
python your_script.py 2>&1 | tee pd_disagg_debug.log
```

### Separate Files for Each Process
```bash
# Run prefill in background, save logs
python run_prefill.py > prefill.log 2>&1 &

# Wait a moment
sleep 2

# Run decode, save logs
python run_decode.py > decode.log 2>&1
```

### View Logs in Real-Time
```bash
# Terminal 1: Run script
python your_script.py > pd_disagg.log 2>&1

# Terminal 2: Tail the log
tail -f pd_disagg.log
```

## 🔎 Finding Specific Logs

### Search for Debug Breakpoints
```bash
# Find all debug breakpoint logs
grep "🔍" pd_disagg.log

# Find NCCL initialization logs
grep "ncclCommInitRank" pd_disagg.log

# Find connection establishment
grep "create_connect" pd_disagg.log

# Find ZMQ socket operations
grep "ZMQ" pd_disagg.log
```

### Filter by Rank
```bash
# Show only prefill (rank 0) logs
grep "\[Rank 0\]" pd_disagg.log

# Show only decode (rank 1) logs
grep "\[Rank 1\]" pd_disagg.log
```

### Extract Timeline
```bash
# Get just the timestamps and key events
grep -E "(🔍|💯|🤝|🚨)" pd_disagg.log | \
    sed 's/INFO//' | \
    sed 's/WARNING//' > timeline.txt
```

## 📊 Example Log Output

### With `VLLM_P2P_DEBUG=1`:
```
INFO 11-01 10:30:45 [p2p_nccl_engine.py:105] 🔍 [Rank 0] P2pNcclEngine.__init__ START | local_rank=0, pid=12345
INFO 11-01 10:30:45 [p2p_nccl_engine.py:122] 🔍 [Rank 0] ZMQ address configured | zmq_address=127.0.0.1:14579
INFO 11-01 10:30:45 [p2p_nccl_engine.py:155] 🔍 [Rank 0] ZMQ Router socket bound successfully | address=127.0.0.1:14579
INFO 11-01 10:30:45 [p2p_nccl_engine.py:212] 🔍 [Rank 0] Listener thread started | thread_name=Thread-1, thread_alive=True
INFO 11-01 10:30:45 [p2p_nccl_engine.py:226] 💯P2pNcclEngine init, rank:0, local_rank:0, http_address:, zmq_address:127.0.0.1:14579, proxy_address:, send_type:PUT_ASYNC, buffer_size_threshold:1000000000.00, nccl_num_channels:8
INFO 11-01 10:30:47 [p2p_nccl_engine.py:105] 🔍 [Rank 1] P2pNcclEngine.__init__ START | local_rank=0, pid=12346
INFO 11-01 10:30:47 [p2p_nccl_engine.py:122] 🔍 [Rank 1] ZMQ address configured | zmq_address=127.0.0.1:14580
INFO 11-01 10:30:47 [p2p_nccl_engine.py:155] 🔍 [Rank 1] ZMQ Router socket bound successfully | address=127.0.0.1:14580
INFO 11-01 10:30:47 [p2p_nccl_engine.py:212] 🔍 [Rank 1] Listener thread started | thread_name=Thread-1, thread_alive=True
INFO 11-01 10:30:47 [p2p_nccl_engine.py:226] 💯P2pNcclEngine init, rank:1, local_rank:0, http_address:, zmq_address:127.0.0.1:14580, proxy_address:, send_type:PUT_ASYNC, buffer_size_threshold:1000000000.00, nccl_num_channels:8
INFO 11-01 10:30:47 [p2p_nccl_engine.py:432] 🔍 [Rank 1] Listener thread RUNNING, starting to poll | zmq_address=127.0.0.1:14580
INFO 11-01 10:30:50 [p2p_nccl_engine.py:234] 🔍 [Rank 0] create_connect START | remote_address=127.0.0.1:14580, existing_comms=[]
INFO 11-01 10:30:50 [p2p_nccl_engine.py:243] 🔍 [Rank 0] ZMQ DEALER connected | remote_address=127.0.0.1:14580
INFO 11-01 10:30:50 [p2p_nccl_engine.py:257] 🔍 [Rank 0] Sending NEW command via ZMQ | remote_address=127.0.0.1:14580, unique_id_hex=a1b2c3d4e5f67890
INFO 11-01 10:30:50 [p2p_nccl_engine.py:443] 🔍 [Rank 1] ✉️ DECODE: Received NEW command | remote_address=127.0.0.1:14579, unique_id_hex=a1b2c3d4e5f67890
INFO 11-01 10:30:50 [p2p_nccl_engine.py:265] 🔍 [Rank 0] ⚠️ PREFILL: About to call ncclCommInitRank | nprocs=2, my_rank=0, remote_address=127.0.0.1:14580
WARNING 11-01 10:30:50 [p2p_nccl_engine.py:267] 🚨 [Rank 0] ENTERING ncclCommInitRank - This will block until decode side also calls it!
INFO 11-01 10:30:50 [p2p_nccl_engine.py:452] 🔍 [Rank 1] ⚠️ DECODE: About to call ncclCommInitRank | nprocs=2, my_rank=1, remote_address=127.0.0.1:14579
WARNING 11-01 10:30:50 [p2p_nccl_engine.py:454] 🚨 [Rank 1] ENTERING ncclCommInitRank - This will block until prefill side completes!
INFO 11-01 10:30:51 [p2p_nccl_engine.py:462] 🔍 [Rank 1] ✅ DECODE: ncclCommInitRank SUCCESS | remote_address=127.0.0.1:14579
INFO 11-01 10:30:51 [p2p_nccl_engine.py:473] 🤝ncclCommInitRank Success, 127.0.0.1:14580👈127.0.0.1:14579, MyRank:1
INFO 11-01 10:30:51 [p2p_nccl_engine.py:273] 🔍 [Rank 0] ✅ PREFILL: ncclCommInitRank SUCCESS | remote_address=127.0.0.1:14580
INFO 11-01 10:30:51 [p2p_nccl_engine.py:287] 🤝ncclCommInitRank Success, 127.0.0.1:14579👉127.0.0.1:14580, MyRank:0
```

## 🎓 Summary

**Default Location**: `stdout` (terminal or redirected file)

**To enable detailed logs**:
```bash
export VLLM_P2P_DEBUG=1
export VLLM_LOGGING_LEVEL=DEBUG
```

**To save to file**:
```bash
python your_script.py > output.log 2>&1
```

**To view real-time**:
```bash
python your_script.py 2>&1 | tee output.log
```

The logs contain emoji markers for easy identification:
- 🔍 = Debug breakpoint
- 💯 = Engine initialization
- 🤝 = NCCL connection success
- 🚨 = Critical warning
- ✅ = Success
- ✉️ = Message received
- ⚠️ = Warning/attention needed

