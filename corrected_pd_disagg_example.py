#!/usr/bin/env python3
"""
Corrected P/D Disaggregation Example with Debug Logging

This script demonstrates the CORRECTED setup for P/D disaggregation.
The main fix: kv_rank should be 1 (not 2) for the decode node.

To enable debug logging, set:
    export VLLM_P2P_DEBUG=1
    export NCCL_DEBUG=INFO
"""

import os
import time
from datetime import datetime
from multiprocessing import Event, Process

import torch
from torch.profiler import profile, ProfilerActivity

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def run_prefill(prefill_done, decode_done, model_path, prompt, multi_modal_data, trace_dir):
    """Prefill node (producer, rank 0)"""
    # We use GPU 0 for prefill node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("=" * 80)
    print("PREFILL NODE STARTING")
    print("=" * 80)

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # ✅ CORRECT CONFIGURATION
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_producer",
        kv_rank=0,  # ✅ Prefill is rank 0
        kv_parallel_size=2,  # ✅ 2 nodes: prefill + decode
    )

    print(f"KV Transfer Config:")
    print(f"  kv_connector: {ktc.kv_connector}")
    print(f"  kv_role: {ktc.kv_role}")
    print(f"  kv_rank: {ktc.kv_rank}")
    print(f"  kv_parallel_size: {ktc.kv_parallel_size}")
    print(f"  kv_ip: {ktc.kv_ip}")
    print(f"  kv_port: {ktc.kv_port}")
    print("=" * 80)

    llm = LLM(
        model=model_path,
        kv_transfer_config=ktc,
        max_model_len=2000,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"video": 1},
        disable_log_stats=False,
    )

    print("Prefill LLM initialized. Starting generation...")

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
    ) as prof:
        llm.generate(prompt, sampling_params, multi_modal_data=multi_modal_data)

    print("✅ Prefill node is finished.")

    if trace_dir:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_filename = f"prefill_profile_trace_{timestamp}.json"
        trace_path = os.path.join(trace_dir, trace_filename)
        os.makedirs(trace_dir, exist_ok=True)
        prof.export_chrome_trace(trace_path)
        print(f"Prefill profiler trace saved to {trace_path}")

    prefill_done.set()
    print("Prefill node is waiting for decode to finish...")
    decode_done.wait()
    print("Prefill node is exiting.")


def run_decode(prefill_done, decode_done, model_path, prompt, sampling_params, trace_dir):
    """Decode node (consumer, rank 1)"""
    # We use GPU 1 for decode node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print("=" * 80)
    print("DECODE NODE STARTING")
    print("=" * 80)

    # ✅ CORRECT CONFIGURATION - Changed kv_rank from 2 to 1
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_consumer",
        kv_rank=1,  # ✅ FIXED: Was 2, should be 1 for P2P with 2 nodes
        kv_parallel_size=2,  # ✅ 2 nodes: prefill + decode
    )

    print(f"KV Transfer Config:")
    print(f"  kv_connector: {ktc.kv_connector}")
    print(f"  kv_role: {ktc.kv_role}")
    print(f"  kv_rank: {ktc.kv_rank} (CORRECTED from 2)")
    print(f"  kv_parallel_size: {ktc.kv_parallel_size}")
    print(f"  kv_ip: {ktc.kv_ip}")
    print(f"  kv_port: {ktc.kv_port}")
    print("=" * 80)

    llm = LLM(
        model=model_path,
        kv_transfer_config=ktc,
        max_model_len=2000,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"video": 1},
        disable_log_stats=False,
    )

    print("Decode LLM initialized. Waiting for prefill node to finish...")
    prefill_done.wait()
    print("Prefill is done. Starting decode generation...")

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
    ) as prof:
        torch.cuda.synchronize()
        tic = time.time()
        outputs = llm.generate(prompt, sampling_params)
        torch.cuda.synchronize()
        toc = time.time()

    decoding_time = toc - tic

    if trace_dir:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_filename = f"decode_profile_trace_{timestamp}.json"
        trace_path = os.path.join(trace_dir, trace_filename)
        os.makedirs(trace_dir, exist_ok=True)
        prof.export_chrome_trace(trace_path)
        print(f"Decode profiler trace saved to {trace_path}")

    print("\n")
    print("-------Disaggregated Decoding with vLLM-------")
    print("Decoding Time:", decoding_time)

    output = outputs[0]
    input_token_length = len(output.prompt_token_ids)
    output_token_length = len(output.outputs[0].token_ids)
    print(f"Input token length: {input_token_length}")
    print(f"Output token length: {output_token_length}")

    output_text = outputs[0].outputs[0].text
    print("Output:")
    print(output_text)
    print("\n")

    print("✅ Decode node finished successfully!")
    decode_done.set()


def main():
    """
    Main function to run P/D disaggregation example.

    Usage:
        # Enable debug logging (recommended for troubleshooting)
        export VLLM_P2P_DEBUG=1
        export NCCL_DEBUG=INFO

        # Run the script
        python corrected_pd_disagg_example.py
    """
    import sys

    # Check if debug mode is enabled
    if os.getenv("VLLM_P2P_DEBUG") == "1":
        print("🔍 Debug mode enabled (VLLM_P2P_DEBUG=1)")
    else:
        print("💡 Tip: Set VLLM_P2P_DEBUG=1 to enable detailed debug logging")

    if os.getenv("NCCL_DEBUG") == "INFO":
        print("🔍 NCCL debug mode enabled")
    else:
        print("💡 Tip: Set NCCL_DEBUG=INFO to see NCCL communication details")

    print("\n")

    # Configuration
    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Change as needed
    prompt = "Hello, my name is"
    multi_modal_data = None  # Set if using multimodal model
    trace_dir = None  # Set to a path if you want profiling traces

    # Sampling params for decode
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    # Synchronization events
    prefill_done = Event()
    decode_done = Event()

    # Create processes
    prefill_process = Process(
        target=run_prefill,
        args=(prefill_done, decode_done, model_path, prompt, multi_modal_data, trace_dir)
    )
    decode_process = Process(
        target=run_decode,
        args=(prefill_done, decode_done, model_path, prompt, sampling_params, trace_dir)
    )

    # Start both processes
    print("Starting prefill process...")
    prefill_process.start()

    # Give prefill a moment to initialize
    time.sleep(2)

    print("Starting decode process...")
    decode_process.start()

    # Wait for both to complete
    decode_process.join()
    prefill_process.join()

    print("\n" + "=" * 80)
    print("✅ P/D Disaggregation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

