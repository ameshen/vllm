"""
Debug Breakpoints for P/D Disaggregation in vLLM

This file contains instrumented functions that you can add to vLLM's code
to help debug P/D disaggregation issues.

Usage:
1. Import this module in the files you want to debug
2. Call the relevant debug functions at key points
3. Run with a debugger (pdb, PyCharm, VSCode, etc.)
"""

import logging
import os
import sys
import threading
import time
import traceback
from typing import Any

logger = logging.getLogger(__name__)

# Global state for tracking execution
_execution_timeline = []
_lock = threading.Lock()


def log_execution_point(
    location: str,
    rank: int | None = None,
    details: dict[str, Any] | None = None,
    breakpoint_here: bool = True
):
    """
    Log an execution point with optional breakpoint.

    Args:
        location: Description of where we are (e.g., "P2pNcclEngine.__init__")
        rank: The rank/role (0 for prefill, 1 for decode)
        details: Additional information to log
        breakpoint_here: If True, will trigger a breakpoint
    """
    rank_str = f"[Rank {rank}]" if rank is not None else "[Unknown Rank]"
    timestamp = time.time()

    msg = f"{rank_str} {location}"
    if details:
        msg += f" | Details: {details}"

    logger.info(msg)

    # Track in timeline
    with _lock:
        _execution_timeline.append({
            'timestamp': timestamp,
            'rank': rank,
            'location': location,
            'details': details,
            'thread': threading.current_thread().name,
        })

    # Optional breakpoint
    if breakpoint_here and os.getenv("VLLM_DEBUG_BREAKPOINTS", "0") == "1":
        import pdb
        pdb.set_trace()


def print_timeline():
    """Print the execution timeline."""
    with _lock:
        print("\n" + "=" * 80)
        print("EXECUTION TIMELINE")
        print("=" * 80)
        for i, event in enumerate(_execution_timeline):
            print(f"{i:3d}. {event['timestamp']:.3f} | "
                  f"Rank {event['rank']} | "
                  f"{event['thread']:20s} | "
                  f"{event['location']}")
            if event['details']:
                print(f"     Details: {event['details']}")
        print("=" * 80 + "\n")


# ============================================================================
# Breakpoint Functions to Insert into vLLM Code
# ============================================================================

def debug_p2p_nccl_engine_init(
    self,
    local_rank: int,
    rank: int,
    zmq_address: str,
    port: int,
):
    """
    Insert at the beginning of P2pNcclEngine.__init__

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:75
    """
    log_execution_point(
        location="P2pNcclEngine.__init__",
        rank=rank,
        details={
            'local_rank': local_rank,
            'zmq_address': zmq_address,
            'port': port,
            'pid': os.getpid(),
        }
    )


def debug_zmq_bind(self, zmq_address: str, rank: int):
    """
    Insert after ZMQ router socket bind

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:128
    """
    log_execution_point(
        location="ZMQ Router Socket Bound",
        rank=rank,
        details={
            'zmq_address': zmq_address,
            'socket_type': 'ROUTER',
        }
    )


def debug_listener_thread_start(self, rank: int, thread):
    """
    Insert after listener thread starts

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:168
    """
    log_execution_point(
        location="Listener Thread Started",
        rank=rank,
        details={
            'thread_name': thread.name,
            'thread_alive': thread.is_alive(),
        }
    )


def debug_create_connect_start(
    self,
    rank: int,
    remote_address: str,
    existing_comms: list[str]
):
    """
    Insert at the start of create_connect

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:199
    """
    log_execution_point(
        location="create_connect() START",
        rank=rank,
        details={
            'remote_address': remote_address,
            'existing_comms': existing_comms,
        }
    )


def debug_nccl_init_rank_prefill(
    self,
    rank: int,
    unique_id: Any,
    remote_address: str
):
    """
    Insert BEFORE ncclCommInitRank on prefill side (rank 0)

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:218

    ⚠️ This is a CRITICAL synchronization point that can hang!
    """
    log_execution_point(
        location="PREFILL: About to call ncclCommInitRank",
        rank=rank,
        details={
            'nprocs': 2,
            'my_rank': 0,
            'remote_address': remote_address,
            'unique_id_bytes': bytes(unique_id.internal)[:16].hex(),
        }
    )


def debug_nccl_init_rank_prefill_done(
    self,
    rank: int,
    comm: Any,
    remote_address: str
):
    """
    Insert AFTER ncclCommInitRank on prefill side (rank 0)

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:220
    """
    log_execution_point(
        location="PREFILL: ncclCommInitRank COMPLETED",
        rank=rank,
        details={
            'comm': str(comm),
            'remote_address': remote_address,
        },
        breakpoint_here=False  # Don't break here, just log
    )


def debug_listener_polling(self, rank: int, iteration: int):
    """
    Insert at the start of listen_for_requests loop

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:373

    Only log every 10 iterations to avoid spam
    """
    if iteration % 10 == 0:
        log_execution_point(
            location=f"Listener polling (iteration {iteration})",
            rank=rank,
            details={'iteration': iteration},
            breakpoint_here=False
        )


def debug_listener_received_new(
    self,
    rank: int,
    remote_address: bytes,
    unique_id: Any
):
    """
    Insert when NEW command is received in listener

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:384
    """
    log_execution_point(
        location="DECODE: Received NEW command",
        rank=rank,
        details={
            'remote_address': remote_address.decode(),
            'unique_id_bytes': bytes(unique_id.internal)[:16].hex(),
        }
    )


def debug_nccl_init_rank_decode(
    self,
    rank: int,
    unique_id: Any,
    remote_address: str
):
    """
    Insert BEFORE ncclCommInitRank on decode side (rank 1)

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:389

    ⚠️ This is a CRITICAL synchronization point that can hang!
    """
    log_execution_point(
        location="DECODE: About to call ncclCommInitRank",
        rank=rank,
        details={
            'nprocs': 2,
            'my_rank': 1,
            'remote_address': remote_address,
            'unique_id_bytes': bytes(unique_id.internal)[:16].hex(),
        }
    )


def debug_nccl_init_rank_decode_done(
    self,
    rank: int,
    comm: Any,
    remote_address: str
):
    """
    Insert AFTER ncclCommInitRank on decode side (rank 1)

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:392
    """
    log_execution_point(
        location="DECODE: ncclCommInitRank COMPLETED",
        rank=rank,
        details={
            'comm': str(comm),
            'remote_address': remote_address,
        },
        breakpoint_here=False
    )


def debug_save_kv_layer(
    self,
    layer_name: str,
    num_requests: int,
    rank: int
):
    """
    Insert at start of save_kv_layer

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:248
    """
    log_execution_point(
        location="save_kv_layer() called",
        rank=rank,
        details={
            'layer_name': layer_name,
            'num_requests': num_requests,
        },
        breakpoint_here=False
    )


def debug_save_kv_layer_send(
    self,
    request_id: str,
    layer_name: str,
    tensor_shape: tuple,
    remote_address: str,
    rank: int
):
    """
    Insert before send_tensor in save_kv_layer

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:288
    """
    log_execution_point(
        location="save_kv_layer: About to send tensor",
        rank=rank,
        details={
            'request_id': request_id,
            'layer_name': layer_name,
            'tensor_shape': tensor_shape,
            'remote_address': remote_address,
        }
    )


def debug_start_load_kv(
    self,
    num_requests: int,
    rank: int
):
    """
    Insert at start of start_load_kv

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:114
    """
    log_execution_point(
        location="start_load_kv() called",
        rank=rank,
        details={
            'num_requests': num_requests,
        }
    )


def debug_start_load_kv_recv(
    self,
    request_id: str,
    layer_name: str,
    remote_address: str,
    rank: int
):
    """
    Insert before recv_tensor in start_load_kv

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:214
    """
    log_execution_point(
        location="start_load_kv: About to recv tensor",
        rank=rank,
        details={
            'request_id': request_id,
            'layer_name': layer_name,
            'remote_address': remote_address,
        }
    )


def debug_send_sync_start(
    self,
    tensor_id: str,
    tensor_shape: tuple,
    remote_address: str,
    rank: int
):
    """
    Insert at start of send_sync

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:537
    """
    log_execution_point(
        location="send_sync() START",
        rank=rank,
        details={
            'tensor_id': tensor_id,
            'tensor_shape': tensor_shape,
            'remote_address': remote_address,
        }
    )


def debug_send_sync_before_zmq_send(
    self,
    tensor_id: str,
    rank: int
):
    """
    Insert before sock.send in send_sync

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:549
    """
    log_execution_point(
        location="send_sync: Before ZMQ send (PUT cmd)",
        rank=rank,
        details={'tensor_id': tensor_id},
        breakpoint_here=False
    )


def debug_send_sync_before_zmq_recv(
    self,
    tensor_id: str,
    rank: int
):
    """
    Insert before sock.recv in send_sync

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:552
    """
    log_execution_point(
        location="send_sync: Waiting for ZMQ ack",
        rank=rank,
        details={'tensor_id': tensor_id}
    )


def debug_send_sync_before_nccl_send(
    self,
    tensor_id: str,
    rank: int
):
    """
    Insert before self.send (NCCL) in send_sync

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:565
    """
    log_execution_point(
        location="send_sync: Before NCCL send",
        rank=rank,
        details={'tensor_id': tensor_id}
    )


def debug_listener_received_put(
    self,
    tensor_id: str,
    tensor_shape: tuple,
    remote_address: bytes,
    rank: int
):
    """
    Insert when PUT command is received in listener

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:400
    """
    log_execution_point(
        location="DECODE: Received PUT command",
        rank=rank,
        details={
            'tensor_id': tensor_id,
            'tensor_shape': tensor_shape,
            'remote_address': remote_address.decode(),
        }
    )


def debug_listener_before_nccl_recv(
    self,
    tensor_id: str,
    rank: int
):
    """
    Insert before self.recv (NCCL) in listener PUT handler

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:415
    """
    log_execution_point(
        location="DECODE: Before NCCL recv",
        rank=rank,
        details={'tensor_id': tensor_id}
    )


def debug_parse_request_id(
    request_id: str,
    is_prefill: bool,
    parsed_ip: str | None,
    parsed_port: int | None
):
    """
    Insert after parse_request_id

    Location: vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:472
    """
    log_execution_point(
        location="parse_request_id()",
        rank=None,
        details={
            'request_id': request_id,
            'is_prefill': is_prefill,
            'parsed_ip': parsed_ip,
            'parsed_port': parsed_port,
        },
        breakpoint_here=False
    )


# ============================================================================
# Utility Functions
# ============================================================================

def setup_debug_logging(rank: int | None = None):
    """
    Set up comprehensive debug logging.

    Call this at the start of your prefill/decode functions.
    """
    # Create logs directory
    os.makedirs("/tmp/vllm_debug", exist_ok=True)

    # Set up file handler
    rank_suffix = f"_rank{rank}" if rank is not None else ""
    log_file = f"/tmp/vllm_debug/vllm_debug{rank_suffix}.log"

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d [%(process)d:%(thread)d] '
               '%(name)s:%(lineno)d %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w')
        ]
    )

    # Set specific loggers to DEBUG
    for logger_name in [
        'vllm.distributed.kv_transfer',
        'vllm.v1',
        '__main__',
    ]:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)

    logger.info(f"Debug logging initialized. Log file: {log_file}")
    logger.info(f"Timeline will be saved to: /tmp/vllm_debug/timeline{rank_suffix}.txt")


def save_timeline(rank: int | None = None):
    """Save the execution timeline to a file."""
    rank_suffix = f"_rank{rank}" if rank is not None else ""
    timeline_file = f"/tmp/vllm_debug/timeline{rank_suffix}.txt"

    with open(timeline_file, 'w') as f:
        with _lock:
            for i, event in enumerate(_execution_timeline):
                f.write(f"{i:3d}. {event['timestamp']:.3f} | "
                       f"Rank {event['rank']} | "
                       f"{event['thread']:20s} | "
                       f"{event['location']}\n")
                if event['details']:
                    f.write(f"     Details: {event['details']}\n")

    logger.info(f"Timeline saved to {timeline_file}")


def check_hanging_threads():
    """Print information about all running threads."""
    logger.info("\n" + "=" * 80)
    logger.info("ACTIVE THREADS")
    logger.info("=" * 80)

    for thread in threading.enumerate():
        logger.info(f"Thread: {thread.name}")
        logger.info(f"  Daemon: {thread.daemon}")
        logger.info(f"  Alive: {thread.is_alive()}")

        # Try to get stack trace
        frame = sys._current_frames().get(thread.ident)
        if frame:
            logger.info("  Stack trace:")
            for line in traceback.format_stack(frame):
                logger.info(f"    {line.strip()}")

    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    print("This module provides debugging utilities for P/D disaggregation.")
    print("\nUsage:")
    print("1. Import this module in vLLM code files you want to debug")
    print("2. Insert the relevant debug_* function calls at key points")
    print("3. Set VLLM_DEBUG_BREAKPOINTS=1 to enable breakpoints")
    print("4. Run your prefill/decode script")
    print("\nExample:")
    print("  export VLLM_DEBUG_BREAKPOINTS=1")
    print("  python your_disagg_script.py")

