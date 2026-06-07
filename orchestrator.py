# Copyright (C) 2026 mlx-orchestrator Authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response, Security, HTTPException
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.security import APIKeyHeader

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mlx-orchestrator")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Default generic configuration structure
DEFAULT_CONFIG = {
    "proxy_port": 12500,
    "scan_interval_seconds": 5,
    "scan_ports_min": 12501,
    "scan_ports_max": 12520,
    "api_key": None,
    "log_token_usage": True,
    "global_opencode": False,
    "debug_mode": False,
    "max_log_size_bytes": 10485760,  # Limit logs to 10MB by default
    "ssl_certfile": None,
    "ssl_keyfile": None,
    "managed_servers": [
        {
            "model": "mlx-community/gemma-4-e2b-it-4bit",
            "port": 12501,
            "type": "vlm",
            "enabled": True
        },
        {
            "model": "mlx-community/gemma-4-e4b-it-4bit",
            "port": 12502,
            "type": "vlm",
            "enabled": False
        },
        {
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "port": 12503,
            "type": "lm",
            "enabled": False
        },
        {
            "model": "mlx-community/gemma-4-31b-bf16",
            "port": 12504,
            "type": "vlm",
            "enabled": False
        }
    ]
}

# Global state
MODEL_ROUTES = {}         # model_id -> backend_url (updated by port scanner)
DISCOVERED_MODELS = {}    # model_id -> original model JSON metadata
RUNNING_PROCESSES = {}    # model_name -> (subprocess.Popen, log_file_handle, log_file_path)
GLOBAL_OPENCODE_OVERRIDE = None
GLOBAL_DEBUG_OVERRIDE = None

client = None

def load_config():
    """Loads configuration from config.json, creating it if it doesn't exist."""
    if not os.path.exists(CONFIG_PATH):
        # Try to clone config.json.example to start off with a helpful template
        example_path = os.path.join(BASE_DIR, "config.json.example")
        config_to_write = DEFAULT_CONFIG
        
        if os.path.exists(example_path):
            try:
                with open(example_path, "r") as f:
                    config_to_write = json.load(f)
                logger.info(f"Cloned template from {example_path} to initialize config.json")
            except Exception as e:
                logger.warning(f"Failed to read example config template: {e}. Using empty default config.")
                
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(config_to_write, f, indent=2)
            logger.info(f"Created configuration file at: {CONFIG_PATH}")
            return config_to_write
        except Exception as e:
            logger.error(f"Failed to write config file: {e}")
            return config_to_write
            
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}. Using defaults.")
        return DEFAULT_CONFIG

def write_opencode_config(config, port=None):
    """Automatically generates or updates opencode.json to match the current
    enabled models, port, and security credentials, avoiding redundant writes.
    Supports local and/or global (~/.opencode/opencode.json) targets.
    """
    port = port or config.get("proxy_port", 12500)
    
    # Determine if we should also write globally
    global_opencode = GLOBAL_OPENCODE_OVERRIDE if GLOBAL_OPENCODE_OVERRIDE is not None else config.get("global_opencode", False)
    
    target_paths = []
    
    # 1. Local path
    local_path = os.path.join(os.path.dirname(CONFIG_PATH), "opencode.json")
    target_paths.append(local_path)
    
    # 2. Global path
    if global_opencode:
        global_dir = os.path.expanduser("~/.opencode")
        try:
            os.makedirs(global_dir, exist_ok=True)
            global_path = os.path.join(global_dir, "opencode.json")
            target_paths.append(global_path)
        except Exception as e:
            logger.error(f"Failed to create directory {global_dir}: {e}")
            
    # Check if HTTPS is configured and certs exist
    ssl_certfile = config.get("ssl_certfile")
    ssl_keyfile = config.get("ssl_keyfile")
    use_https = ssl_certfile and ssl_keyfile and os.path.exists(ssl_certfile) and os.path.exists(ssl_keyfile)
    scheme = "https" if use_https else "http"
    
    base_url = f"{scheme}://127.0.0.1:{port}/v1"
    
    headers = {}
    api_key = config.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    models_dict = {}
    managed_servers = config.get("managed_servers", [])
    if isinstance(managed_servers, list):
        for server in managed_servers:
            if not isinstance(server, dict):
                continue
            if server.get("enabled", False):
                repo_id = server.get("model")
                if not repo_id:
                    continue
            # Generate a friendly title name from repo name
            name_parts = repo_id.split("/")[-1].replace("-", " ").replace("_", " ").split()
            words = []
            for w in name_parts:
                cap = w.capitalize()
                replacements = {
                    "It": "Instruct",
                    "Lm": "LM",
                    "Vlm": "VLM",
                    "Bf16": "BF16",
                    "4bit": "4-bit",
                    "8bit": "8-bit",
                    "31b": "31B",
                    "35b": "35B",
                    "E2b": "2B",
                    "E4b": "4B",
                    "A3b": "A3B"
                }
                words.append(replacements.get(cap, cap))
            friendly_name = " ".join(words)
            models_dict[repo_id] = {"name": friendly_name}
            
    opencode_data = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "local-mlx": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "MLX Local Server",
                "options": {
                    "baseURL": base_url
                },
                "models": models_dict
            }
        }
    }
    
    if headers:
        opencode_data["provider"]["local-mlx"]["options"]["headers"] = headers
        
    for opencode_path in target_paths:
        write_needed = True
        if os.path.exists(opencode_path):
            try:
                with open(opencode_path, "r") as f:
                    existing = json.load(f)
                if existing == opencode_data:
                    write_needed = False
            except Exception:
                pass
                
        if write_needed:
            try:
                with open(opencode_path, "w") as f:
                    json.dump(opencode_data, f, indent=2)
                logger.info(f"Generated/updated OpenCode configuration profile at: {opencode_path}")
            except Exception as e:
                logger.error(f"Failed to write opencode.json at {opencode_path}: {e}")

async def monitor_and_scan_loop(interval):
    """
    Background worker that runs continuously to:
    1. Check/reload config.json.
    2. Manage subprocess lifetimes (starts enabled servers, stops disabled ones, restarts crashed ones).
    3. Scan ports to register active routing endpoints.
    """
    global MODEL_ROUTES, DISCOVERED_MODELS, RUNNING_PROCESSES
    
    # Dedicated HTTP client for scanning with very short timeouts
    async with httpx.AsyncClient() as scan_client:
        while True:
            config = load_config()
            write_opencode_config(config)
            managed_servers = config.get("managed_servers", [])
            if not isinstance(managed_servers, list):
                logger.warning("Config error: 'managed_servers' must be a list in config.json.")
                managed_servers = []
            
            # Step 1: Manage subprocesses based on configuration
            desired_enabled = {}
            for server in managed_servers:
                if not isinstance(server, dict):
                    logger.warning(f"Config warning: Skip non-dictionary server entry: {server}")
                    continue
                model_name = server.get("model")
                port = server.get("port")
                if not model_name or port is None:
                    logger.warning(f"Config warning: Skip malformed server entry (missing model or port): {server}")
                    continue
                enabled = server.get("enabled", False)
                server_type = server.get("type", "lm")
                desired_enabled[model_name] = (enabled, port, server_type)
                
            # Stop servers that are now disabled
            for model_name in list(RUNNING_PROCESSES.keys()):
                enabled, _, _ = desired_enabled.get(model_name, (False, None, None))
                if not enabled:
                    proc, log_file, _ = RUNNING_PROCESSES[model_name]
                    logger.info(f"Stopping model '{model_name}' (PID: {proc.pid}) as it was disabled in config...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    log_file.close()
                    del RUNNING_PROCESSES[model_name]
                    
            # Start or recover servers that are enabled
            for model_name, (enabled, port, server_type) in desired_enabled.items():
                if not enabled:
                    continue
                    
                proc_info = RUNNING_PROCESSES.get(model_name)
                is_running = False
                
                if proc_info:
                    proc, log_file, log_file_path = proc_info
                    if proc.poll() is None:
                        is_running = True
                        # Check log file size and truncate in-place if it exceeds limit
                        max_log_size = config.get("max_log_size_bytes", 10 * 1024 * 1024)
                        try:
                            if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > max_log_size:
                                with open(log_file_path, "r+") as f:
                                    f.truncate(0)
                                logger.info(f"Truncated active log file {log_file_path} for model '{model_name}' (exceeded {max_log_size} bytes)")
                        except Exception as e:
                            logger.error(f"Failed to truncate active log file for model '{model_name}': {e}")
                    else:
                        logger.warning(f"Model server '{model_name}' (port {port}) terminated unexpectedly with exit code {proc.poll()}. Restarting...")
                        log_file.close()
                        del RUNNING_PROCESSES[model_name]
                        
                if not is_running:
                    # Formulate command to launch model server with a strict-override monkeypatch
                    # to support newer models with redundant KV-sharing parameters (like Gemma 4).
                    if server_type == "lm":
                        python_code = (
                            "import sys; "
                            "import mlx.nn as nn; "
                            "orig_load_weights = nn.Module.load_weights; "
                            "nn.Module.load_weights = lambda self, weights, *args, **kwargs: "
                            "orig_load_weights(self, weights, *args, **{**kwargs, 'strict': False}); "
                            "from mlx_lm.server import main; "
                            "sys.argv = ['mlx_lm.server'] + sys.argv[1:]; "
                            "main()"
                        )
                    else:
                        python_code = (
                            "import sys; "
                            "import mlx.nn as nn; "
                            "orig_load_weights = nn.Module.load_weights; "
                            "nn.Module.load_weights = lambda self, weights, *args, **kwargs: "
                            "orig_load_weights(self, weights, *args, **{**kwargs, 'strict': False}); "
                            "from mlx_vlm.server.cli import main; "
                            "sys.argv = ['mlx_vlm.server'] + sys.argv[1:]; "
                            "main()"
                        )
                    cmd = [
                        sys.executable, "-c", python_code,
                        "--model", model_name,
                        "--port", str(port)
                    ]
                    
                    safe_name = model_name.replace("/", "_")
                    log_file_path = os.path.join(LOGS_DIR, f"{safe_name}.log")
                    
                    # Rotate old log on startup if it exceeds the limit
                    max_log_size = config.get("max_log_size_bytes", 10 * 1024 * 1024)
                    if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > max_log_size:
                        backup_path = log_file_path + ".1"
                        try:
                            if os.path.exists(backup_path):
                                os.remove(backup_path)
                            os.rename(log_file_path, backup_path)
                            logger.info(f"Rotated log file {log_file_path} to {backup_path} (exceeded {max_log_size} bytes)")
                        except Exception as e:
                            logger.error(f"Failed to rotate log file {log_file_path}: {e}")
                    
                    try:
                        log_file = open(log_file_path, "a")
                        proc = subprocess.Popen(
                            cmd, 
                            stdout=log_file, 
                            stderr=log_file,
                            start_new_session=True
                        )
                        RUNNING_PROCESSES[model_name] = (proc, log_file, log_file_path)
                        logger.info(f"Launched model '{model_name}' on port {port} (PID: {proc.pid}, Logs: logs/{safe_name}.log)")
                    except Exception as e:
                        logger.error(f"Failed to launch model '{model_name}': {e}")

            # Step 2: Scan active ports to build routing table
            new_routes = {}
            new_models = {}
            
            scan_min = config.get("scan_ports_min", 12501)
            scan_max = config.get("scan_ports_max", 12520)
            
            for port in range(scan_min, scan_max + 1):
                backend_url = f"http://127.0.0.1:{port}"
                try:
                    response = await scan_client.get(
                        f"{backend_url}/v1/models",
                        timeout=httpx.Timeout(0.4, connect=0.1)
                    )
                    if response.status_code == 200:
                        data = response.json()
                        models = data.get("data", [])
                        for m in models:
                            model_id = m.get("id")
                            if model_id:
                                new_routes[model_id] = backend_url
                                new_models[model_id] = m
                except Exception:
                    pass
            
            # Log updates
            added = set(new_routes.keys()) - set(MODEL_ROUTES.keys())
            removed = set(MODEL_ROUTES.keys()) - set(new_routes.keys())
            
            if added:
                logger.info(f"Discovered active endpoints: {list(added)}")
            if removed:
                logger.info(f"Endpoints went offline: {list(removed)}")
                
            MODEL_ROUTES = new_routes
            DISCOVERED_MODELS = new_models
            
            await asyncio.sleep(interval)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0))
    logger.info("Orchestrator client initialized.")
    
    config = load_config()
    interval = config.get("scan_interval_seconds", 5)
    
    monitor_task = asyncio.create_task(monitor_and_scan_loop(interval))
    yield
    
    logger.info("Shutting down... Cleaning up background model servers...")
    monitor_task.cancel()
    
    for model_name, (proc, log_file, _) in RUNNING_PROCESSES.items():
        logger.info(f"Terminating subprocess for model '{model_name}' (PID: {proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()
        
    await client.aclose()
    logger.info("Orchestrator stopped successfully.")

# API Key header verification dependency
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def verify_api_key(authorization: str = Security(api_key_header)):
    """Validates the Authorization header against the config.json api_key if enabled."""
    config = load_config()
    required_key = config.get("api_key")
    
    if not required_key:
        # Authentication is disabled
        return
        
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Authentication required. Missing Authorization header.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "missing_api_key"
                }
            }
        )
        
    # Handle Bearer token format
    token = authorization
    if token.startswith("Bearer "):
        token = token[7:]
        
    if token != required_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Incorrect API key provided.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            }
        )

app = FastAPI(
    title="MLX Multi-Model Orchestration Proxy",
    description="Manages backend MLX server subprocesses dynamically and routes OpenAI queries.",
    lifespan=lifespan
)

@app.get("/", include_in_schema=False)
async def redirect_to_swagger():
    """Redirects root requests to Swagger UI documentation."""
    return RedirectResponse(url="/docs")

@app.get("/v1/models")
async def list_models(api_key: str = Security(verify_api_key)):
    """Unified models endpoint."""
    config = load_config()
    debug_mode = GLOBAL_DEBUG_OVERRIDE if GLOBAL_DEBUG_OVERRIDE is not None else config.get("debug_mode", False)
    
    models_data = list(DISCOVERED_MODELS.values())
    if debug_mode:
        logger.info(
            f"🔍 [DEBUG REQUEST] GET /v1/models endpoint called.\n"
            f"🔍 [DEBUG RESPONSE] GET /v1/models returning: {json.dumps(models_data)}"
        )
    return {
        "object": "list",
        "data": models_data
    }

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_request(request: Request, path: str, api_key: str = Security(verify_api_key)):
    """Routes client calls to the active downstream server."""
    body = await request.body()
    model_name = None

    if request.method == "POST" and body:
        try:
            payload = await request.json()
            model_name = payload.get("model")
        except Exception as e:
            logger.warning(f"Failed to parse request JSON: {e}")

    # Fallback routing
    if not model_name and MODEL_ROUTES:
        model_name = next(iter(MODEL_ROUTES.keys()))
        logger.info(f"No model name specified. Routing to default active model: {model_name}")

    backend_url = MODEL_ROUTES.get(model_name)
    if not backend_url:
        active_models = list(MODEL_ROUTES.keys())
        if active_models:
            error_msg = (
                f"Error: Model '{model_name}' is not currently running.\n"
                f"Available active models: {', '.join(active_models)}"
            )
        else:
            error_msg = (
                f"Error: No model servers are running.\n"
                f"Please enable models in 'config.json' (the orchestrator will auto-load them)."
            )
        logger.error(error_msg)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": error_msg,
                    "type": "service_unavailable",
                    "param": None,
                    "code": None
                }
            }
        )

    # Load preferences
    config = load_config()
    log_token_usage = config.get("log_token_usage", True)
    debug_mode = GLOBAL_DEBUG_OVERRIDE if GLOBAL_DEBUG_OVERRIDE is not None else config.get("debug_mode", False)

    target_url = f"{backend_url}/v1/{path}"
    logger.info(f"Routing request for model '{model_name}' -> {target_url}")

    # Log incoming request in debug mode (mask sensitive Authorization headers)
    if debug_mode:
        masked_headers = dict(request.headers)
        if "authorization" in masked_headers:
            masked_headers["authorization"] = "Bearer " + "*" * 8
        logger.info(
            f"🔍 [DEBUG REQUEST] Routing from client:\n"
            f"  Method:  {request.method} /v1/{path}\n"
            f"  Headers: {masked_headers}\n"
            f"  Params:  {dict(request.query_params)}\n"
            f"  Body:    {body.decode('utf-8', errors='ignore')}"
        )

    headers = dict(request.headers)
    headers.pop("host", None)
    params = dict(request.query_params)

    try:
        proxy_req = client.build_request(
            method=request.method,
            url=target_url,
            headers=headers,
            params=params,
            content=body
        )
        
        start_time = time.time()
        downstream_response = await client.send(proxy_req, stream=True)

        if debug_mode:
            logger.info(
                f"🔍 [DEBUG RESPONSE] Downstream Response metadata:\n"
                f"  Status:  {downstream_response.status_code}\n"
                f"  Headers: {dict(downstream_response.headers)}"
            )
        
        if downstream_response.headers.get("content-type") == "text/event-stream" or downstream_response.status_code == 200:
            async def generate_chunks():
                total_completion_tokens = 0
                total_prompt_tokens = 0
                chunk_token_count = 0
                buffer = ""
                is_event_stream = downstream_response.headers.get("content-type") == "text/event-stream"
                accumulated_body = []
                
                try:
                    async for chunk in downstream_response.aiter_bytes():
                        yield chunk
                        
                        if debug_mode:
                            if is_event_stream:
                                logger.info(f"🔍 [DEBUG STREAM CHUNK] {len(chunk)} bytes: {chunk.decode('utf-8', errors='ignore')}")
                            else:
                                accumulated_body.append(chunk.decode('utf-8', errors='ignore'))
                        
                        if log_token_usage:
                            # Accumulate bytes in buffer to parse complete lines
                            buffer += chunk.decode("utf-8", errors="ignore")
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if line.startswith("data:"):
                                    data_content = line[5:].strip()
                                    if data_content == "[DONE]":
                                        continue
                                    try:
                                        data_json = json.loads(data_content)
                                        
                                        # 1. Look for explicit usage block returned in stream
                                        usage = data_json.get("usage")
                                        if usage:
                                            total_prompt_tokens = usage.get("prompt_tokens", 0)
                                            total_completion_tokens = usage.get("completion_tokens", 0)
                                            
                                        # 2. Count tokens as backup
                                        choices = data_json.get("choices", [])
                                        if choices:
                                            delta = choices[0].get("delta", {})
                                            if "content" in delta and delta["content"]:
                                                chunk_token_count += 1
                                    except Exception:
                                        pass
                finally:
                    if debug_mode and not is_event_stream and accumulated_body:
                        logger.info(
                            f"🔍 [DEBUG RESPONSE] Downstream Non-streaming body:\n"
                            f"  Body: {''.join(accumulated_body)}"
                        )
                    await downstream_response.aclose()
                    
                    # Log usage metrics
                    if log_token_usage:
                        elapsed = time.time() - start_time
                        
                        # Fallback to chunk count if server didn't include usage block
                        final_gen_tokens = total_completion_tokens or chunk_token_count
                        
                        if final_gen_tokens > 0:
                            tps = final_gen_tokens / elapsed if elapsed > 0 else 0
                            logger.info(
                                f"📈 [USAGE] Model: '{model_name}' | Prompt: {total_prompt_tokens} tokens | "
                                f"Generated: {final_gen_tokens} tokens | Speed: {tps:.1f} tokens/sec | "
                                f"Time: {elapsed:.2f}s"
                            )

            response_headers = dict(downstream_response.headers)
            response_headers.pop("transfer-encoding", None)

            return StreamingResponse(
                generate_chunks(),
                status_code=downstream_response.status_code,
                headers=response_headers
            )
        else:
            await downstream_response.aread()
            
            if debug_mode:
                logger.info(
                    f"🔍 [DEBUG RESPONSE] Downstream Non-streaming body:\n"
                    f"  Body: {downstream_response.content.decode('utf-8', errors='ignore')}"
                )

            # Log usage metrics for non-streaming response
            if log_token_usage:
                elapsed = time.time() - start_time
                try:
                    res_json = downstream_response.json()
                    usage = res_json.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    if completion_tokens > 0:
                        tps = completion_tokens / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"📈 [USAGE] Model: '{model_name}' | Prompt: {prompt_tokens} tokens | "
                            f"Generated: {completion_tokens} tokens | Speed: {tps:.1f} tokens/sec | "
                            f"Time: {elapsed:.2f}s"
                        )
                except Exception:
                    pass
                    
            response_headers = dict(downstream_response.headers)
            response_headers.pop("transfer-encoding", None)
            res = Response(
                content=downstream_response.content,
                status_code=downstream_response.status_code,
                headers=response_headers
            )
            await downstream_response.aclose()
            return res

    except httpx.ConnectError:
        error_msg = f"Error: Connection to backend {backend_url} failed."
        logger.error(error_msg)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": error_msg,
                    "type": "service_unavailable",
                    "param": None,
                    "code": None
                }
            }
        )

def download_model(model_id: str):
    """Downloads model snapshot directly from Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ Error: 'huggingface_hub' is not installed. Please run 'pip install -r requirements.txt'")
        sys.exit(1)
        
    print(f"📥 Starting download for: {model_id}")
    print("This will download weights and configs to your local Hugging Face cache folder.")
    try:
        # snapshot_download displays a live progress bar automatically when run in a shell
        local_dir = snapshot_download(repo_id=model_id)
        print("\n✅ Download Complete!")
        print(f"Stored at cache directory: {local_dir}")
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        sys.exit(1)

from pathlib import Path

def get_system_ram_gb() -> float:
    """Gets total physical RAM of the system in Gigabytes."""
    try:
        return (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')) / (1024**3)
    except Exception:
        # Sensible fallback
        return 16.0

def get_optimal_gpu_limit_mb(system_ram_gb: float) -> tuple[int, float, int]:
    """Given system RAM in GB, returns (optimal_limit_mb, optimal_limit_gb, closest_tier_gb).
    Allocates high limits (85% to 94%) depending on the total RAM capacity.
    """
    tiers = [8, 16, 24, 32, 36, 48, 64, 96, 128, 192]
    closest_tier = min(tiers, key=lambda x: abs(x - system_ram_gb))
    
    tier_mapping = {
        8: 5120,      # 5.0 GB limit (leaves 3.0 GB for system/browser/IDE)
        16: 11776,    # 11.5 GB limit (leaves 4.5 GB for system/browser/IDE)
        24: 18432,    # 18.0 GB limit (leaves 6.0 GB for system/browser/IDE)
        32: 25600,    # 25.0 GB limit (leaves 7.0 GB for system/browser/IDE)
        36: 28672,    # 28.0 GB limit (leaves 8.0 GB for system/browser/IDE)
        48: 38912,    # 38.0 GB limit (leaves 10.0 GB for system/browser/IDE)
        64: 53248,    # 52.0 GB limit (leaves 12.0 GB for system/browser/IDE)
        96: 81920,    # 80.0 GB limit (leaves 16.0 GB for system/browser/IDE)
        128: 110592,  # 108.0 GB limit (leaves 20.0 GB for system/browser/IDE)
        192: 172032   # 168.0 GB limit (leaves 24.0 GB for system/browser/IDE)
    }
    
    limit_mb = tier_mapping.get(closest_tier, int(max(system_ram_gb - 8.0, system_ram_gb * 0.7) * 1024))
    return limit_mb, limit_mb / 1024.0, closest_tier

def get_gpu_wired_limit_gb() -> float:
    """Gets the active macOS iogpu.wired_limit_mb setting converted to Gigabytes.
    If 0 (default), macOS dynamically allocates memory up to approximately 75% of RAM.
    """
    try:
        import subprocess
        result = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True, check=True)
        mb = int(result.stdout.strip())
        if mb > 0:
            return mb / 1024.0
    except Exception:
        pass
    
    # Fallback to 75% of physical system RAM if unset (0) or error
    return get_system_ram_gb() * 0.75

def is_custom_gpu_limit_active() -> bool:
    """Checks if a custom non-default GPU wired limit is set on the Mac."""
    try:
        import subprocess
        result = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True, check=True)
        return int(result.stdout.strip()) > 0
    except Exception:
        return False

def get_local_model_size_gb(repo_id: str) -> float:
    """Calculates size of the model locally downloaded in Hugging Face blobs cache."""
    if "/" not in repo_id:
        return 0.0
    org, name = repo_id.split("/")
    cache_dir = Path.home() / f".cache/huggingface/hub/models--{org}--{name}"
    blobs_dir = cache_dir / "blobs"
    if not blobs_dir.exists():
        return 0.0
    try:
        total_size = sum(f.stat().st_size for f in blobs_dir.glob("*") if f.is_file())
        return total_size / (1024**3)
    except Exception:
        return 0.0

def check_memory_limits(config):
    """Checks if the enabled models exceed the active GPU allocation limit,
    spilling over to CPU causing high CPU heat and slow inference.
    """
    system_ram = get_system_ram_gb()
    gpu_limit = get_gpu_wired_limit_gb()
    has_custom_limit = is_custom_gpu_limit_active()
    
    total_enabled_size = 0.0
    enabled_models = []
    
    managed_servers = config.get("managed_servers", [])
    if isinstance(managed_servers, list):
        for server in managed_servers:
            if not isinstance(server, dict):
                continue
            if server.get("enabled", False):
                model_name = server.get("model")
                if not model_name:
                    continue
            model_size = get_local_model_size_gb(model_name)
            if model_size > 0:
                total_enabled_size += model_size
                enabled_models.append(f"{model_name} ({model_size:.1f} GB)")
                
    if total_enabled_size > gpu_limit:
        limit_desc = f"Custom GPU Limit: {gpu_limit:.1f} GB" if has_custom_limit else f"Default Dynamic Limit: ~{gpu_limit:.1f} GB (75% of RAM)"
        opt_mb, opt_gb, closest_tier = get_optimal_gpu_limit_mb(system_ram)
        
        logger.warning(
            f"\n🚨  MEMORY OVERFLOW WARNING:\n"
            f"  Total size of enabled models is ~{total_enabled_size:.1f} GB.\n"
            f"  Your active limit is: {limit_desc}.\n"
            f"  ⚠️  The model weights exceed this limit. This will force your Mac to spill over\n"
            f"     and run inference on the CPU. This causes extremely slow generation speeds\n"
            f"     and will cause your Mac CPU to run hot and spin up fans.\n"
            f"  👉 To fix this, allocate more memory to the GPU by running this command:\n"
            f"     sudo sysctl iogpu.wired_limit_mb={opt_mb}\n"
            f"     (This optimizes your {closest_tier}GB RAM Mac to allow up to {opt_gb:.1f}GB for GPU)\n"
        )
    elif enabled_models:
        limit_desc = f"custom GPU limit of {gpu_limit:.1f} GB" if has_custom_limit else f"default dynamic limit of ~{gpu_limit:.1f} GB"
        logger.info(
            f"Memory Check: Loading ~{total_enabled_size:.1f} GB of models onto a system with a {limit_desc}. "
            f"Model fits entirely within GPU memory."
        )

def recommend_models():
    """Prints system memory details and tailored model running recommendations."""
    system_ram = get_system_ram_gb()
    gpu_limit = get_gpu_wired_limit_gb()
    has_custom_limit = is_custom_gpu_limit_active()
    
    print("=" * 60)
    print("🖥️  SYSTEM DIAGNOSTICS & RECOMMENDATIONS")
    print("=" * 60)
    print(f"Total System RAM:        {system_ram:.1f} GB")
    if has_custom_limit:
        print(f"Active GPU Limit:        {gpu_limit:.1f} GB (Custom User Setting)")
    else:
        print(f"Active GPU Limit:        ~{gpu_limit:.1f} GB (macOS default, 75% of RAM)")
    
    print("\n📦 Downloaded Models Size (Local Cache):")
    config = load_config()
    has_downloaded = False
    
    for server in config.get("managed_servers", []):
        model_name = server["model"]
        model_size = get_local_model_size_gb(model_name)
        if model_size > 0:
            has_downloaded = True
            status = "ENABLED" if server.get("enabled", False) else "DISABLED"
            print(f"  - {model_name:<45} | Size: {model_size:>5.1f} GB | Status: {status}")
        else:
            print(f"  - {model_name:<45} | Size:  N/A  (Not in cache)")
            
    if not has_downloaded:
        print("  (No configured models have been downloaded yet. Run 'python orchestrator.py download <model_id>')")
        
    print("\n💡 Memory Allocation Advice:")
    if system_ram < 32:
        print("  - Your Mac has less than 32 GB of Unified Memory.")
        print("  - We recommend running 4-bit or 8-bit quantized models only.")
        print("  - Avoid enabling multiple servers at the same time.")
        print("  - Target models under 14B parameters (e.g. Qwen 7B/14B, Llama 8B) for best speed.")
    elif system_ram < 64:
        print("  - Your Mac has between 32 GB and 64 GB of Unified Memory.")
        print("  - You can comfortably run models up to 32B parameters (e.g., Qwen 32B 4-bit/8-bit).")
        print("  - Running two 8B models simultaneously (e.g. Qwen 8B + Llama 8B) is supported.")
        print("  - Avoid running BF16 versions of models larger than 14B parameter sizes.")
    else:
        print("  - Your Mac has 64 GB+ of Unified Memory.")
        print("  - You can comfortably run large models (like Gemma 4 31B BF16) or serve multiple smaller models concurrently.")
        print("  - If running multiple models, monitor the combined active size to stay under your RAM limit.")
        
    print("\n🚀 Apple Silicon GPU Memory Optimization Tip:")
    if not has_custom_limit:
        print("  - macOS is currently using its default dynamic limits (~75% RAM).")
    else:
        print(f"  - A custom GPU limit of {gpu_limit:.1f} GB is currently active.")
        
    opt_mb, opt_gb, closest_tier = get_optimal_gpu_limit_mb(system_ram)
    print(f"  - To maximize GPU memory on your {closest_tier} GB RAM Mac, run:")
    print(f"    sudo sysctl iogpu.wired_limit_mb={opt_mb}")
    print(f"    (This allocates up to {opt_gb:.1f} GB of Unified Memory for the GPU)")
    print("  - To reset to macOS defaults:")
    print("    sudo sysctl iogpu.wired_limit_mb=0")
    print("=" * 60)

def run_server(port_override=None, global_opencode=False, debug=False):
    """Launches the orchestrator proxy server with optional SSL/HTTPS."""
    global GLOBAL_OPENCODE_OVERRIDE, GLOBAL_DEBUG_OVERRIDE
    GLOBAL_OPENCODE_OVERRIDE = global_opencode
    GLOBAL_DEBUG_OVERRIDE = debug
    
    config = load_config()
    port = port_override or config.get("proxy_port", 12500)
    
    # Auto-generate/update opencode.json matching the config
    write_opencode_config(config, port)
    
    # Run memory limit diagnostics before startup
    check_memory_limits(config)
    
    # SSL/HTTPS configuration
    ssl_certfile = config.get("ssl_certfile")
    ssl_keyfile = config.get("ssl_keyfile")
    
    debug_mode = debug or config.get("debug_mode", False)
    logger.info(f"Starting MLX Orchestrator on port {port} (access logs: {'enabled' if debug_mode else 'disabled'})...")
    if debug_mode:
        logger.info("🔍 Debug logging mode is active. Request/response details will be logged.")
    
    # Build uvicorn arguments dynamically
    uvicorn_kwargs = {
        "app": app,
        "host": "127.0.0.1",
        "port": port,
        "reload": False,
        "access_log": debug_mode
    }
    
    if ssl_certfile and ssl_keyfile:
        if os.path.exists(ssl_certfile) and os.path.exists(ssl_keyfile):
            uvicorn_kwargs["ssl_certfile"] = ssl_certfile
            uvicorn_kwargs["ssl_keyfile"] = ssl_keyfile
            logger.info("🔒 SSL/HTTPS enabled.")
        else:
            logger.error(
                f"❌ SSL Config Error: Certificate or key file not found.\n"
                f"  Certfile path: {ssl_certfile}\n"
                f"  Keyfile path:  {ssl_keyfile}\n"
                f"  Falling back to HTTP."
            )
            
    uvicorn.run(**uvicorn_kwargs)

def main():
    parser = argparse.ArgumentParser(
        description="MLX Multi-Model Orchestrator & CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py serve
  python orchestrator.py serve --port 12500
  python orchestrator.py serve --global-opencode
  python orchestrator.py serve --debug
  python orchestrator.py download mlx-community/gemma-4-31b-bf16
  python orchestrator.py recommend
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")
    
    # Serve Parser
    serve_parser = subparsers.add_parser("serve", help="Start the orchestrator proxy and managed MLX servers")
    serve_parser.add_argument("--port", type=int, help="Override proxy port (defaults to config value or 12500)")
    serve_parser.add_argument("--global-opencode", action="store_true", help="Generate the OpenCode configuration profile globally at ~/.opencode/opencode.json")
    serve_parser.add_argument("--debug", action="store_true", help="Enable debug logging of request/response interactions")
    
    # Download Parser
    download_parser = subparsers.add_parser("download", help="Pre-download a model from Hugging Face")
    download_parser.add_argument("model_id", type=str, help="Hugging Face model repository ID (e.g. mlx-community/gemma-4-31b-bf16)")
    
    # Recommend Parser
    subparsers.add_parser("recommend", help="Display memory diagnostics and model recommendations")
    
    args = parser.parse_args()
    
    if args.command == "download":
        download_model(args.model_id)
    elif args.command == "serve":
        run_server(args.port, args.global_opencode, args.debug)
    elif args.command == "recommend":
        recommend_models()
    else:
        # If no arguments provided, display help
        parser.print_help()

if __name__ == "__main__":
    main()
