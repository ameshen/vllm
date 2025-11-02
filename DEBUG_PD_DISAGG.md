# P/D Disaggregation Debugging Guide for vLLM

## Critical Issue Found in Your Code

**🚨 PROBLEM: Your decode node has `kv_rank=2` but it should be `kv_rank=1`**

```python
# WRONG - This will cause your system to hang:
ktc = KVTransferConfig(
    kv_connector="P2pNcclConnector",
    kv_role="kv_consumer",
    kv_rank=2,  # ❌ WRONG! Should be 1
    kv_parallel_size=2,
)

# CORRECT:
ktc = KVTransferConfig(
    kv_connector="P2pNcclConnector",
    kv_role="kv_consumer",
    kv_rank=1,  # ✅ CORRECT
    kv_parallel_size=2,
)
```

For P2P NCCL with 2 instances:
- Prefill (producer): `kv_rank=0`
- Decode (consumer): `kv_rank=1`

## Architecture Overview

### Key Components

1. **P2pNcclConnector** (`vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`)
   - High-level connector interface
   - Manages KV cache save/load operations
   - Handles request metadata

2. **P2pNcclEngine** (`vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py`)
   - Low-level engine for P2P communication
   - Uses ZMQ for control messages
   - Uses NCCL for tensor transfers
   - Runs listener thread and optional async send thread

3. **KVTransferConfig** (`vllm/config/kv_transfer.py`)
   - Configuration for KV transfer setup

### Communication Flow

```
Prefill (Producer)                          Decode (Consumer)
=====================                       =====================
1. Generate KV cache                        1. Wait for prefill
2. save_kv_layer() called                  2. Receives "NEW" cmd via ZMQ
   per layer                                3. Initializes NCCL comm
3. send_tensor() via engine                4. start_load_kv() called
4. PUT cmd sent via ZMQ                    5. Receives PUT cmd via ZMQ
5. Tensor sent via NCCL                    6. Tensor received via NCCL
6. wait_for_save()                         7. inject_kv_into_layer()
```

## Critical Debugging Points

### 1. Engine Initialization

**File**: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py`

**Breakpoint Location**: Line ~75 (in `__init__`)
```python
def __init__(
    self,
    local_rank: int,
    config: KVTransferConfig,
    hostname: str = "",
    port_offset: int = 0,
    library_path: str | None = None,
) -> None:
    # BREAKPOINT HERE
    self.config = config
    self.rank = port_offset
    # ...
```

**What to check**:
- `self.rank` value (should be 0 for prefill, 1 for decode)
- `self.zmq_address` (should be unique for each instance)
- `self._port` (should be different for each rank)

### 2. ZMQ Router Socket Binding

**File**: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py`

**Breakpoint Location**: Line ~128
```python
self.router_socket = self.context.socket(zmq.ROUTER)
# BREAKPOINT HERE - Check if bind succeeds
self.router_socket.bind(f"tcp://{self.zmq_address}")
```

**What to check**:
- No `Address already in use` errors
- `self.zmq_address` is correct format (e.g., "127.0.0.1:14579")

### 3. Listener Thread Start

**File**: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py`

**Breakpoint Location**: Line ~165
```python
self._listener_thread = threading.Thread(
    target=self.listen_for_requests, daemon=True
)
# BREAKPOINT HERE - Verify thread starts
self._listener_thread.start()
```

**What to check**:
- Thread actually starts (check `self._listener_thread.is_alive()`)

### 4. Connection Creation (Prefill initiates)

**File**: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py`

**Breakpoint Location**: Line ~199
```python
def create_connect(self, remote_address: str | None = None):
    # BREAKPOINT HERE
    assert remote_address is not None
    if remote_address not in self.socks:
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        # BREAKPOINT HERE - Before connect
        sock.connect(f"tcp://{remote_address}")
        self.socks[remote_address] = sock
```

**What to check**:
- `remote_address` format is correct
- Connection succeeds without timeout
- Socket identity is set correctly

### 5. NCCL Comm Initialization (Both sides)

**File**: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py`

**Prefill side** - Line ~213:
```python
unique_id = self.nccl.ncclGetUniqueId()
data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
# BREAKPOINT HERE - Before sending NEW
sock.send(msgpack.dumps(data))

with torch.cuda.device(self.device):
    rank = 0
    with set_p2p_nccl_context(self.nccl_num_channels):
        # BREAKPOINT HERE - This might hang if decode is not ready
        comm: ncclComm_t = self.nccl.ncclCommInitRank(2, unique_id, rank)
```

**Decode side (listener)** - Line ~384:
```python
def listen_for_requests(self):
    while True:
        # BREAKPOINT HERE - Check if polling
        socks = dict(self.poller.poll())
        if self.router_socket not in socks:
            continue

        remote_address, message = self.router_socket.recv_multipart()
        data = msgpack.loads(message)
        if data["cmd"] == "NEW":
            # BREAKPOINT HERE - When NEW received
            unique_id = self.nccl.unique_id_from_bytes(bytes(data["unique_id"]))
            with torch.cuda.device(self.device):
                rank = 1
                with set_p2p_nccl_context(self.nccl_num_channels):
                    # BREAKPOINT HERE - This must complete for prefill to proceed
                    comm: ncclComm_t = self.nccl.ncclCommInitRank(
                        2, unique_id, rank
                    )
```

**What to check**:
- Both sides reach `ncclCommInitRank` (this is a collective operation)
- `rank` is 0 on prefill, 1 on decode
- NCCL environment variables are set correctly

### 6. Sending KV Cache (Prefill)

**File**: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`

**Breakpoint Location**: Line ~248
```python
def save_kv_layer(
    self,
    layer_name: str,
    kv_layer: torch.Tensor,
    attn_metadata: "AttentionMetadata",
    **kwargs: Any,
) -> None:
    # BREAKPOINT HERE - Per layer save
    if not self.is_producer:
        return
    
    # ... extract KV ...
    
    for request in connector_metadata.requests:
        request_id = request.request_id
        ip, port = self.parse_request_id(request_id, True)
        # BREAKPOINT HERE - Check remote_address
        remote_address = ip + ":" + str(port + self._rank)
        
        kv_cache = extract_kv_from_layer(kv_layer, request.block_ids)
        # BREAKPOINT HERE - Before send
        self.p2p_nccl_engine.send_tensor(
            request_id + "#" + layer_name, kv_cache, remote_address
        )
```

**What to check**:
- `request_id` format and content
- `remote_address` is correct
- `kv_cache` shape and dtype
- Number of layers being saved

### 7. Receiving KV Cache (Decode)

**File**: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`

**Breakpoint Location**: Line ~114
```python
def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
    # BREAKPOINT HERE
    if self.is_producer:
        return
    
    # ... metadata setup ...
    
    for request in metadata.requests:
        request_id = request.request_id
        ip, port = self.parse_request_id(request_id, False)
        # BREAKPOINT HERE - Check remote_address
        remote_address = ip + ":" + str(port + self._rank)
        for layer_name in forward_context.no_compile_layers:
            # ... 
            # BREAKPOINT HERE - Before recv
            kv_cache = self.p2p_nccl_engine.recv_tensor(
                request_id + "#" + layer_name, remote_address
            )
```

**What to check**:
- Metadata has requests
- `remote_address` matches prefill's `zmq_address`
- `recv_tensor` doesn't hang

### 8. Tensor Send/Recv Operations (Engine level)

**File**: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py`

**Send (PUT mode)** - Line ~535:
```python
def send_sync(self, item: SendQueueItem) -> bool:
    # BREAKPOINT HERE
    sock = self.socks[item.remote_address]
    comm, rank = self.comms[item.remote_address]
    data = {
        "cmd": "PUT",
        "tensor_id": item.tensor_id,
        "shape": tensor.shape,
        "dtype": str(tensor.dtype).replace("torch.", ""),
    }
    # BREAKPOINT HERE - Before ZMQ send
    sock.send(msgpack.dumps(data))
    
    # BREAKPOINT HERE - Waiting for ack
    response = sock.recv()
    
    # BREAKPOINT HERE - Before NCCL send
    self.send(comm, tensor.to(self.device), rank ^ 1, self.send_stream)
```

**Recv (listener)** - Line ~398:
```python
elif data["cmd"] == "PUT":
    # BREAKPOINT HERE
    tensor_id = data["tensor_id"]
    try:
        with torch.cuda.stream(self.recv_stream):
            tensor = torch.empty(
                data["shape"],
                dtype=getattr(torch, data["dtype"]),
                device=self.device,
            )
        # BREAKPOINT HERE - Before ack
        self.router_socket.send_multipart([remote_address, b"0"])
        comm, rank = self.comms[remote_address.decode()]
        # BREAKPOINT HERE - Before NCCL recv (this might hang)
        self.recv(comm, tensor, rank ^ 1, self.recv_stream)
```

**What to check**:
- ZMQ messages are exchanged successfully
- Buffer size doesn't exceed threshold
- NCCL send/recv complete without hanging

### 9. Request ID Format

**File**: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`

**Breakpoint Location**: Line ~472
```python
@staticmethod
def parse_request_id(request_id: str, is_prefill=True) -> tuple[str, int]:
    # BREAKPOINT HERE
    if is_prefill:
        pattern = r"___decode_addr_(.*):(\d+)"
    else:
        pattern = r"___prefill_addr_(.*):(\d+)___"
    
    match = re.search(pattern, request_id)
```

**What to check**:
- `request_id` contains embedded address info
- Pattern matches correctly
- IP and port are extracted properly

## Common Hang Scenarios

### 1. NCCL Initialization Hang
**Symptom**: Both processes hang at `ncclCommInitRank`

**Root Causes**:
- Wrong `kv_rank` values (e.g., your bug with rank=2)
- Wrong `kv_parallel_size`
- Firewall blocking connections
- Different NCCL versions or settings

**Debug**:
```python
# Add these breakpoints BEFORE ncclCommInitRank on both sides:
print(f"[Rank {rank}] About to call ncclCommInitRank with nprocs=2")
print(f"[Rank {rank}] unique_id={unique_id}")
print(f"[Rank {rank}] local_rank={self.local_rank}")

# Add timeout wrapper (in test code):
import signal
def timeout_handler(signum, frame):
    raise TimeoutError("ncclCommInitRank timed out!")
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout
```

### 2. ZMQ Connection Hang
**Symptom**: Prefill hangs on `sock.send()` or `sock.recv()`

**Root Causes**:
- Decode listener thread not running
- Wrong address/port
- ZMQ socket not properly initialized

**Debug**:
```python
# In decode listener, add:
print(f"Listener thread started, polling on {self.zmq_address}")

# In prefill, before connect:
print(f"Attempting to connect to {remote_address}")
sock.setsockopt(zmq.SNDTIMEO, 5000)  # 5 sec timeout
sock.setsockopt(zmq.RCVTIMEO, 5000)
```

### 3. Tensor Transfer Hang
**Symptom**: Hangs during NCCL send/recv

**Root Causes**:
- Buffer size exhausted
- CUDA out of memory
- Mismatched tensor shapes
- NCCL communicator not initialized

**Debug**:
```python
# Before each NCCL operation:
print(f"[{self.rank}] About to send/recv tensor_id={tensor_id}, "
      f"shape={tensor.shape}, dtype={tensor.dtype}")
print(f"[{self.rank}] Buffer usage: {self.buffer_size}/{self.buffer_size_threshold}")
```

### 4. Request ID Mismatch
**Symptom**: Consumer can't find KV cache for requests

**Root Causes**:
- Request IDs don't contain address info
- Pattern matching fails
- Different request IDs between prefill and decode

**Debug**:
```python
# In save_kv_layer and start_load_kv:
print(f"Request ID: {request_id}")
print(f"Parsed address: {remote_address}")
```

## Debugging Workflow

### Step 1: Verify Configuration
```python
# In both prefill and decode, after LLM creation:
print("=" * 80)
print(f"KV Transfer Config:")
print(f"  kv_connector: {ktc.kv_connector}")
print(f"  kv_role: {ktc.kv_role}")
print(f"  kv_rank: {ktc.kv_rank}")
print(f"  kv_parallel_size: {ktc.kv_parallel_size}")
print(f"  kv_ip: {ktc.kv_ip}")
print(f"  kv_port: {ktc.kv_port}")
print("=" * 80)
```

### Step 2: Monitor Engine Initialization
```python
# Add logging in P2pNcclEngine.__init__:
logger.setLevel(logging.DEBUG)
```

### Step 3: Track Connection Establishment
```python
# In create_connect and listen_for_requests:
logger.info(f"[{self.rank}] Connection state: {list(self.comms.keys())}")
```

### Step 4: Monitor Tensor Transfers
```python
# In send_tensor and recv_tensor:
logger.info(f"[{self.rank}] Transfer {tensor_id}: "
            f"{tensor.shape if tensor else None}")
```

### Step 5: Check Synchronization Points
```python
# In wait_for_save and wait_for_load:
logger.info(f"[{self.rank}] Waiting at synchronization point...")
```

## Recommended Debugging Setup

Create a file `debug_pd_disagg.py`:

```python
import logging
import sys

# Enable verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'vllm_debug_rank{os.getenv("CUDA_VISIBLE_DEVICES", "0")}.log')
    ]
)

# Set specific loggers to DEBUG
logging.getLogger('vllm.distributed.kv_transfer').setLevel(logging.DEBUG)
logging.getLogger('vllm.v1').setLevel(logging.DEBUG)

# Your run_prefill and run_decode functions here...
```

## Quick Fix for Your Code

Change your decode function:
```python
def run_decode(prefill_done, decode_done, model_path, prompt, sampling_params, trace_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_consumer",
        kv_rank=1,  # ✅ Changed from 2 to 1
        kv_parallel_size=2,
    )
    # ... rest of your code
```

## Environment Variables to Check

```bash
# NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# vLLM debugging
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_TRACE_FUNCTION=1

# CUDA
export CUDA_LAUNCH_BLOCKING=1  # For synchronous CUDA ops
```

## Next Steps

1. **Fix the kv_rank bug first** (change from 2 to 1)
2. Add logging to track execution flow
3. Set NCCL_DEBUG=INFO to see NCCL operations
4. If still stuck, add breakpoints at the critical points listed above
5. Check the log files to see where each process stops

Good luck with debugging!

