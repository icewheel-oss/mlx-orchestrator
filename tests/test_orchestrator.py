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

import json
import os
from unittest.mock import MagicMock, patch, ANY, AsyncMock
import pytest
from fastapi.testclient import TestClient

# Import the orchestrator script
import orchestrator

@pytest.fixture
def mock_config_dir(tmp_path, monkeypatch):
    """
    Sets up a temporary directory structure for configuration file testing.
    Mocks the BASE_DIR, CONFIG_PATH, and LOGS_DIR variables in orchestrator.py
    to isolate tests from the host system configuration.
    """
    # Create temp paths
    temp_config = tmp_path / "config.json"
    temp_example = tmp_path / "config.json.example"
    temp_logs = tmp_path / "logs"
    temp_logs.mkdir()

    # Monkeypatch paths inside orchestrator module
    monkeypatch.setattr(orchestrator, "CONFIG_PATH", str(temp_config))
    monkeypatch.setattr(orchestrator, "LOGS_DIR", str(temp_logs))
    monkeypatch.setattr(orchestrator, "BASE_DIR", str(tmp_path))

    return {
        "config": temp_config,
        "example": temp_example,
        "logs": temp_logs,
        "base": tmp_path
    }

def test_load_config_creates_default_when_missing(mock_config_dir):
    """
    Checks that if config.json does not exist, the orchestrator generates
    a fallback configuration automatically.
    """
    # Ensure config.json is not present
    assert not mock_config_dir["config"].exists()

    config = orchestrator.load_config()

    # Verify config matches the DEFAULT_CONFIG template
    assert config == orchestrator.DEFAULT_CONFIG
    assert mock_config_dir["config"].exists()

def test_load_config_clones_example_if_available(mock_config_dir):
    """
    Validates that if config.json is missing but config.json.example is present,
    the loader automatically copies the example settings to config.json.
    """
    example_data = {
        "proxy_port": 9000,
        "scan_interval_seconds": 10,
        "managed_servers": [{"model": "test-model", "port": 9001, "enabled": True}]
    }
    
    # Write the mock template file
    with open(mock_config_dir["example"], "w") as f:
        json.dump(example_data, f)

    config = orchestrator.load_config()

    # Verify example values are copied
    assert config["proxy_port"] == 9000
    assert config["managed_servers"][0]["model"] == "test-model"
    assert mock_config_dir["config"].exists()

def test_get_local_model_size_calculations(tmp_path, monkeypatch):
    """
    Tests get_local_model_size_gb function. Uses a temporary directory mimicking
    the Hugging Face cache structure to ensure file size calculations are accurate.
    """
    # Create fake HF cache directory structure
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    
    org_name, repo_name = "mlx-community", "gemma-test"
    fake_cache_dir = fake_home / f".cache/huggingface/hub/models--{org_name}--{repo_name}/blobs"
    fake_cache_dir.mkdir(parents=True)

    # Create dummy weight files: one of 10 MB, another of 20 MB (total 30 MB)
    file_1 = fake_cache_dir / "blob1"
    file_2 = fake_cache_dir / "blob2"
    file_1.write_bytes(b"\x00" * 10 * 1024 * 1024)
    file_2.write_bytes(b"\x00" * 20 * 1024 * 1024)

    # Monkeypatch Path.home() to point to our fake home directory
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    # Run size check (should be 30 MB / 1024^3 GB = ~0.0286 GB)
    size_gb = orchestrator.get_local_model_size_gb(f"{org_name}/{repo_name}")
    assert pytest.approx(size_gb, rel=1e-3) == 30.0 / 1024.0

def test_get_system_ram_detection():
    """
    Verifies that system physical RAM lookup correctly reports a value
    greater than 0.
    """
    ram = orchestrator.get_system_ram_gb()
    assert ram > 0.0
    assert isinstance(ram, float)

@patch("orchestrator.get_system_ram_gb")
@patch("orchestrator.get_gpu_wired_limit_gb")
@patch("orchestrator.is_custom_gpu_limit_active")
@patch("orchestrator.get_local_model_size_gb")
def test_memory_limit_warnings_trigger_correctly(
    mock_local_size, mock_custom_limit_active, mock_gpu_limit, mock_ram, caplog
):
    """
    Checks that the orchestrator logs a warning if the size of the enabled models
    exceeds the active macOS GPU wired limit.
    """
    # Simulate a 16 GB RAM Mac with default 75% limit (12 GB GPU limit)
    mock_ram.return_value = 16.0
    mock_gpu_limit.return_value = 12.0
    mock_custom_limit_active.return_value = False

    # Mock model sizes (our mock model size is 14 GB)
    mock_local_size.return_value = 14.0

    mock_config = {
        "managed_servers": [
            {"model": "org/big-model", "enabled": True}
        ]
    }

    # Run memory limits check capturing logs
    with caplog.at_level("WARNING"):
        orchestrator.check_memory_limits(mock_config)

    # Assert that warning was issued since 14 GB > 12 GB limit
    assert any("MEMORY OVERFLOW WARNING" in record.message for record in caplog.records)

def test_unified_models_list_endpoint():
    """
    Tests that the FastAPI unified models endpoint returns a standard list
    format containing registered models in the orchestrator memory.
    """
    # Setup test models in global variables
    orchestrator.DISCOVERED_MODELS = {
        "model-A": {"id": "model-A", "object": "model"},
        "model-B": {"id": "model-B", "object": "model"}
    }

    client = TestClient(orchestrator.app)
    response = client.get("/v1/models")

    assert response.status_code == 200
    json_data = response.json()
    assert json_data["object"] == "list"
    assert len(json_data["data"]) == 2
    assert json_data["data"][0]["id"] == "model-A"

@pytest.mark.asyncio
@patch("orchestrator.client")
async def test_proxy_routing_to_backends(mock_http_client):
    """
    Tests the core routing mechanism of the API proxy:
    - Verifies that calls for model A route to A's port.
    - Verifies that calls for model B route to B's port.
    """
    # Setup the discovery routes
    orchestrator.MODEL_ROUTES = {
        "model-qwen": "http://127.0.0.1:12501",
        "model-gemma": "http://127.0.0.1:12502"
    }

    # Setup mock HTTP response for the downstream calls
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.content = b'{"choices":[]}'
    mock_response.aread = AsyncMock()
    mock_response.aclose = AsyncMock()
    
    # Correctly mock the async generator aiter_bytes
    async def mock_aiter_bytes():
        yield b'{"choices":[]}'
    mock_response.aiter_bytes = mock_aiter_bytes
    
    # Configure mock client send method
    async def mock_send(req, stream=False):
        return mock_response
    mock_http_client.send = mock_send
    
    # Configure mock client request builder
    mock_req = MagicMock()
    mock_http_client.build_request.return_value = mock_req

    client = TestClient(orchestrator.app)

    # 1. Test routing to Qwen
    payload_qwen = {"model": "model-qwen", "messages": []}
    client.post("/v1/chat/completions", json=payload_qwen)
    
    # Check request builder was called with Qwen target URL
    mock_http_client.build_request.assert_called_with(
        method="POST",
        url="http://127.0.0.1:12501/v1/chat/completions",
        headers=ANY,
        params={},
        content=ANY
    )

    # 2. Test routing to Gemma
    payload_gemma = {"model": "model-gemma", "messages": []}
    client.post("/v1/chat/completions", json=payload_gemma)
    
    # Check request builder was called with Gemma target URL
    mock_http_client.build_request.assert_called_with(
        method="POST",
        url="http://127.0.0.1:12502/v1/chat/completions",
        headers=ANY,
        params={},
        content=ANY
    )

@patch("orchestrator.load_config")
def test_api_key_security_success(mock_load_config):
    """
    Verifies that requests succeed when the correct API key is provided.
    """
    mock_load_config.return_value = {
        "api_key": "my-secret-token",
        "managed_servers": []
    }
    
    client = TestClient(orchestrator.app)
    response = client.get("/v1/models", headers={"Authorization": "Bearer my-secret-token"})
    
    assert response.status_code == 200

@patch("orchestrator.load_config")
def test_api_key_security_missing_unauthorized(mock_load_config):
    """
    Verifies that requests fail with 401 when authentication is enabled
    but the Authorization header is missing.
    """
    mock_load_config.return_value = {
        "api_key": "my-secret-token",
        "managed_servers": []
    }
    
    client = TestClient(orchestrator.app)
    response = client.get("/v1/models")
    
    assert response.status_code == 401
    assert "Missing Authorization header" in response.json()["detail"]["error"]["message"]

@patch("orchestrator.load_config")
def test_api_key_security_incorrect_unauthorized(mock_load_config):
    """
    Verifies that requests fail with 401 when an incorrect API key is provided.
    """
    mock_load_config.return_value = {
        "api_key": "my-secret-token",
        "managed_servers": []
    }
    
    client = TestClient(orchestrator.app)
    response = client.get("/v1/models", headers={"Authorization": "Bearer wrong-token"})
    
    assert response.status_code == 401
    assert "Incorrect API key provided" in response.json()["detail"]["error"]["message"]

@pytest.mark.asyncio
@patch("orchestrator.client")
@patch("orchestrator.load_config")
async def test_token_usage_logging_in_streams(mock_load_config, mock_http_client, caplog):
    """
    Verifies that the proxy parses stream chunks, counts tokens, calculates TPS,
    and logs token usage statistics at the end of the streaming response.
    """
    mock_load_config.return_value = {
        "api_key": None,
        "log_token_usage": True,
        "managed_servers": []
    }
    
    orchestrator.MODEL_ROUTES = {
        "model-qwen": "http://127.0.0.1:12501"
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_response.aread = AsyncMock()
    mock_response.aclose = AsyncMock()
    
    # Simulate a stream with two text content chunks and one usage chunk
    async def mock_aiter_bytes():
        yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
        yield b'data: {"choices": [{"delta": {"content": " World"}}]}\n\n'
        yield b'data: {"usage": {"prompt_tokens": 10, "completion_tokens": 2}}\n\n'
        yield b'data: [DONE]\n\n'
    mock_response.aiter_bytes = mock_aiter_bytes

    async def mock_send(req, stream=False):
        return mock_response
    mock_http_client.send = mock_send
    mock_http_client.build_request.return_value = MagicMock()

    client = TestClient(orchestrator.app)
    
    with caplog.at_level("INFO"):
        response = client.post("/v1/chat/completions", json={"model": "model-qwen"})
        
    assert response.status_code == 200
    # The response content should have the streamed chunks
    assert b"Hello" in response.content
    
    # Assert that token usage log metrics were outputted by proxy
    assert any("📈 [USAGE]" in record.message for record in caplog.records)
    assert any("Prompt: 10 tokens" in record.message for record in caplog.records)
    assert any("Generated: 2 tokens" in record.message for record in caplog.records)

def test_root_redirect_to_swagger(mock_config_dir):
    """Verifies that accessing the root URL redirects to /docs (Swagger/OpenAPI docs)."""
    client = TestClient(orchestrator.app)
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"

def test_write_opencode_config(mock_config_dir):
    """Verifies that write_opencode_config generates the correct opencode.json content."""
    config = {
        "proxy_port": 12500,
        "api_key": "test-key",
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
            }
        ]
    }
    
    orchestrator.write_opencode_config(config)
    
    opencode_path = os.path.join(os.path.dirname(orchestrator.CONFIG_PATH), "opencode.json")
    assert os.path.exists(opencode_path)
    
    with open(opencode_path, "r") as f:
        data = json.load(f)
        
    provider = data["provider"]["local-mlx"]
    assert provider["options"]["baseURL"] == "http://127.0.0.1:12500/v1"
    assert provider["options"]["headers"]["Authorization"] == "Bearer test-key"
    
    # Should only contain enabled models
    assert "mlx-community/gemma-4-e2b-it-4bit" in provider["models"]
    assert "mlx-community/gemma-4-e4b-it-4bit" not in provider["models"]
    
    # Check friendly name formatting
    assert provider["models"]["mlx-community/gemma-4-e2b-it-4bit"]["name"] == "Gemma 4 2B Instruct 4-bit"

def test_write_global_opencode_config(mock_config_dir, monkeypatch):
    """Verifies that write_opencode_config generates global opencode.json when enabled."""
    config = {
        "proxy_port": 12500,
        "global_opencode": True,
        "managed_servers": []
    }
    
    temp_global_dir = os.path.join(mock_config_dir["base"], "global_opencode_dir")
    monkeypatch.setattr(os.path, "expanduser", lambda path: temp_global_dir if path == "~/.opencode" else path)

    # Ensure clean override state
    orchestrator.GLOBAL_OPENCODE_OVERRIDE = None

    orchestrator.write_opencode_config(config)
    
    global_opencode_path = os.path.join(temp_global_dir, "opencode.json")
    assert os.path.exists(global_opencode_path)
    
    with open(global_opencode_path, "r") as f:
        data = json.load(f)
    assert data["provider"]["local-mlx"]["options"]["baseURL"] == "http://127.0.0.1:12500/v1"

def test_write_global_opencode_config_by_override(mock_config_dir, monkeypatch):
    """Verifies that write_opencode_config generates global opencode.json when CLI override is active."""
    config = {
        "proxy_port": 12500,
        "global_opencode": False,
        "managed_servers": []
    }
    
    temp_global_dir = os.path.join(mock_config_dir["base"], "global_opencode_dir")
    monkeypatch.setattr(os.path, "expanduser", lambda path: temp_global_dir if path == "~/.opencode" else path)

    # Set override state
    orchestrator.GLOBAL_OPENCODE_OVERRIDE = True

    orchestrator.write_opencode_config(config)
    
    global_opencode_path = os.path.join(temp_global_dir, "opencode.json")
    assert os.path.exists(global_opencode_path)
    
    # Reset override state
    orchestrator.GLOBAL_OPENCODE_OVERRIDE = None

def test_write_junie_config(mock_config_dir):
    """Verifies that write_junie_config generates the correct Junie profile json content."""
    config = {
        "proxy_port": 12500,
        "api_key": "test-key",
        "managed_servers": [
            {
                "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
                "port": 12503,
                "type": "lm",
                "enabled": True,
                "temperature": 0.3,
                "faster_model": "mlx-community/custom-helper"
            },
            {
                "model": "mlx-community/gemma-4-e4b-it-4bit",
                "port": 12502,
                "type": "vlm",
                "enabled": True
            },
            {
                "model": "mlx-community/gemma-4-e2b-it-4bit",
                "port": 12501,
                "type": "vlm",
                "enabled": True
            }
        ]
    }
    
    orchestrator.write_junie_config(config)
    
    junie_local_dir = os.path.join(os.path.dirname(orchestrator.CONFIG_PATH), ".junie", "models")
    qwen_profile_path = os.path.join(junie_local_dir, "mlx-qwen3.6-35b-a3b-4bit.json")
    gemma4_profile_path = os.path.join(junie_local_dir, "mlx-gemma-4-e4b-it-4bit.json")
    gemma2_profile_path = os.path.join(junie_local_dir, "mlx-gemma-4-e2b-it-4bit.json")
    
    assert os.path.exists(qwen_profile_path)
    assert os.path.exists(gemma4_profile_path)
    assert os.path.exists(gemma2_profile_path)
    
    # 1. Assert Qwen profile (explicitly configured)
    with open(qwen_profile_path, "r") as f:
        qwen_data = json.load(f)
    assert qwen_data["baseUrl"] == "http://127.0.0.1:12500/v1"
    assert qwen_data["id"] == "mlx-community/Qwen3.6-35B-A3B-4bit"
    assert qwen_data["apiType"] == "OpenAICompletion"
    assert qwen_data["apiKey"] == "test-key"
    assert qwen_data["temperature"] == 0.3
    assert qwen_data["primaryModel"] == {
        "id": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "temperature": 0.3
    }
    assert qwen_data["fasterModel"] == {
        "id": "mlx-community/custom-helper"
    }
    
    # 2. Assert Gemma-4B profile (auto-detects Gemma-2B as faster)
    with open(gemma4_profile_path, "r") as f:
        gemma4_data = json.load(f)
    assert gemma4_data["id"] == "mlx-community/gemma-4-e4b-it-4bit"
    assert "temperature" not in gemma4_data
    assert gemma4_data["primaryModel"] == {
        "id": "mlx-community/gemma-4-e4b-it-4bit"
    }
    assert gemma4_data["fasterModel"] == {
        "id": "mlx-community/gemma-4-e2b-it-4bit"
    }
    
    # 3. Assert Gemma-2B profile (no active model is smaller, so no fasterModel)
    with open(gemma2_profile_path, "r") as f:
        gemma2_data = json.load(f)
    assert gemma2_data["id"] == "mlx-community/gemma-4-e2b-it-4bit"
    assert "fasterModel" not in gemma2_data

def test_write_global_junie_config(mock_config_dir, monkeypatch):
    """Verifies that write_junie_config generates global Junie profiles when global_opencode is enabled."""
    config = {
        "proxy_port": 12500,
        "global_opencode": True,
        "managed_servers": [
            {
                "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
                "port": 12503,
                "type": "lm",
                "enabled": True
            }
        ]
    }
    
    temp_global_dir = os.path.join(mock_config_dir["base"], "global_junie_dir")
    monkeypatch.setattr(os.path, "expanduser", lambda path: os.path.join(temp_global_dir, "models") if path == "~/.junie/models" else path)

    # Clean override state
    orchestrator.GLOBAL_OPENCODE_OVERRIDE = None

    orchestrator.write_junie_config(config)
    
    global_profile_path = os.path.join(temp_global_dir, "models", "mlx-qwen3.6-35b-a3b-4bit.json")
    assert os.path.exists(global_profile_path)
    
    with open(global_profile_path, "r") as f:
        data = json.load(f)
    assert data["baseUrl"] == "http://127.0.0.1:12500/v1"

@pytest.mark.asyncio
@patch("orchestrator.load_config")
@patch("orchestrator.write_opencode_config")
@patch("orchestrator.write_junie_config")
@patch("subprocess.Popen")
@patch("asyncio.sleep")
async def test_monitor_and_scan_loop_manages_subprocesses(
    mock_sleep, mock_popen, mock_write_junie_config, mock_write_config, mock_load_config
):
    """
    Verifies that monitor_and_scan_loop dynamically starts enabled servers,
    and terminates/stops disabled servers without restarting the orchestrator.
    """
    # 1. First iteration: model-A is enabled, model-B is disabled
    config_1 = {
        "scan_interval_seconds": 1,
        "managed_servers": [
            {"model": "model-A", "port": 12501, "type": "lm", "enabled": True},
            {"model": "model-B", "port": 12502, "type": "vlm", "enabled": False}
        ]
    }
    
    # 2. Second iteration: model-A is disabled, model-B is enabled
    config_2 = {
        "scan_interval_seconds": 1,
        "managed_servers": [
            {"model": "model-A", "port": 12501, "type": "lm", "enabled": False},
            {"model": "model-B", "port": 12502, "type": "vlm", "enabled": True}
        ]
    }
    
    # Load config returns config_1 first, then config_2
    mock_load_config.side_effect = [config_1, config_2]
    
    # Mock subprocess.Popen processes
    mock_proc_A = MagicMock()
    mock_proc_A.poll.return_value = None
    mock_proc_A.pid = 12345
    
    mock_proc_B = MagicMock()
    mock_proc_B.poll.return_value = None
    mock_proc_B.pid = 12346
    
    # subprocess.Popen returns mock_proc_A first, then mock_proc_B
    mock_popen.side_effect = [mock_proc_A, mock_proc_B]
    
    # Make asyncio.sleep raise KeyboardInterrupt on the second call to break the loop
    sleep_calls = 0
    async def mock_sleep_fn(secs):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise KeyboardInterrupt()
    mock_sleep.side_effect = mock_sleep_fn
    
    # Ensure clear initial state
    orchestrator.RUNNING_PROCESSES = {}
    
    # Run the monitor loop until it raises KeyboardInterrupt
    try:
        await orchestrator.monitor_and_scan_loop(1)
    except KeyboardInterrupt:
        pass
        
    # Assertions:
    # Check that model-A was launched as a subprocess
    mock_popen.assert_any_call(
        ANY,
        stdout=ANY,
        stderr=ANY,
        start_new_session=True
    )
    
    # Check that model-A is now disabled and was terminated
    mock_proc_A.terminate.assert_called_once()
    assert "model-A" not in orchestrator.RUNNING_PROCESSES
    
    # Check that model-B was launched
    assert "model-B" in orchestrator.RUNNING_PROCESSES

def test_malformed_config_handling_gracefully(mock_config_dir):
    """
    Checks that malformed config lists or server dictionaries do not crash
    write_opencode_config, check_memory_limits, or load_config.
    """
    # 1. Non-list managed_servers
    malformed_config_1 = {
        "proxy_port": 12500,
        "managed_servers": "not-a-list"
    }
    
    # These should complete without throwing exceptions
    orchestrator.write_opencode_config(malformed_config_1)
    orchestrator.check_memory_limits(malformed_config_1)
    
    # 2. List containing non-dictionary entries
    malformed_config_2 = {
        "proxy_port": 12500,
        "managed_servers": ["not-a-dict", 123, None]
    }
    orchestrator.write_opencode_config(malformed_config_2)
    orchestrator.check_memory_limits(malformed_config_2)
    
    # 3. Dictionaries missing required keys
    malformed_config_3 = {
        "proxy_port": 12500,
        "managed_servers": [
            {"enabled": True}, # missing model
            {"model": "test-model", "enabled": True} # missing port
        ]
    }
    orchestrator.write_opencode_config(malformed_config_3)
    orchestrator.check_memory_limits(malformed_config_3)

@pytest.mark.asyncio
@patch("orchestrator.client")
@patch("orchestrator.load_config")
async def test_debug_mode_logging(mock_load_config, mock_http_client, caplog):
    """
    Verifies that enabling debug mode logs request and response interactions.
    """
    mock_load_config.return_value = {
        "debug_mode": True,
        "managed_servers": []
    }
    
    orchestrator.MODEL_ROUTES = {
        "model-qwen": "http://127.0.0.1:12501"
    }

    # Setup mock HTTP response for downstream calls
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.content = b'{"choices":[{"text":"output-text"}]}'
    mock_response.aread = AsyncMock()
    mock_response.aclose = AsyncMock()
    
    async def mock_aiter_bytes():
        yield mock_response.content
    mock_response.aiter_bytes = mock_aiter_bytes
    
    async def mock_send(req, stream=False):
        return mock_response
    mock_http_client.send = mock_send
    mock_http_client.build_request.return_value = MagicMock()

    client = TestClient(orchestrator.app)
    
    # Clean override state
    orchestrator.GLOBAL_DEBUG_OVERRIDE = None
    
    with caplog.at_level("INFO"):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-qwen", "messages": [{"role": "user", "content": "hi"}]}
        )
        
    assert response.status_code == 200
    
    # Assertions for debug mode logs:
    assert any("[DEBUG REQUEST] Routing from client" in record.message for record in caplog.records)
    assert any("[DEBUG RESPONSE] Downstream Response metadata" in record.message for record in caplog.records)
    assert any("[DEBUG RESPONSE] Downstream Non-streaming body" in record.message for record in caplog.records)

@pytest.mark.asyncio
@patch("orchestrator.client")
@patch("orchestrator.load_config")
async def test_debug_mode_logging_streaming(mock_load_config, mock_http_client, caplog):
    """
    Verifies that enabling debug mode logs stream chunks in a streaming response.
    """
    mock_load_config.return_value = {
        "debug_mode": True,
        "managed_servers": []
    }
    
    orchestrator.MODEL_ROUTES = {
        "model-qwen": "http://127.0.0.1:12501"
    }

    # Setup mock HTTP response for streaming downstream calls
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_response.aread = AsyncMock()
    mock_response.aclose = AsyncMock()
    
    async def mock_aiter_bytes():
        yield b'data: {"choices": [{"delta": {"content": "chunk-1"}}]}\n\n'
    mock_response.aiter_bytes = mock_aiter_bytes

    async def mock_send(req, stream=False):
        return mock_response
    mock_http_client.send = mock_send
    mock_http_client.build_request.return_value = MagicMock()

    client = TestClient(orchestrator.app)
    
    # Clean override state
    orchestrator.GLOBAL_DEBUG_OVERRIDE = None
    
    with caplog.at_level("INFO"):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-qwen"}
        )
        
    assert response.status_code == 200
    assert b"chunk-1" in response.content
    
    # Assertions for debug stream logging:
    assert any("[DEBUG REQUEST] Routing from client" in record.message for record in caplog.records)
    assert any("[DEBUG RESPONSE] Downstream Response metadata" in record.message for record in caplog.records)
    assert any("[DEBUG STREAM CHUNK]" in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_log_file_rotation_and_truncation(monkeypatch, tmp_path):
    import asyncio
    # Setup config path and temporary file
    temp_config = tmp_path / "config.json"
    monkeypatch.setattr(orchestrator, "CONFIG_PATH", str(temp_config))
    
    config_data = {
        "proxy_port": 12500,
        "scan_interval_seconds": 0.1,
        "max_log_size_bytes": 10,
        "managed_servers": [
            {
                "model": "test-model-rotation",
                "port": 12501,
                "type": "lm",
                "enabled": True
            }
        ]
    }
    with open(temp_config, "w") as f:
        json.dump(config_data, f)
        
    # Setup log directory and file
    log_dir = tmp_path / "logs"
    os.makedirs(log_dir, exist_ok=True)
    monkeypatch.setattr(orchestrator, "LOGS_DIR", str(log_dir))
    
    log_path = log_dir / "test-model-rotation.log"
    with open(log_path, "w") as f:
        f.write("A" * 15)  # 15 bytes exceeds 10 bytes limit
        
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # running
    
    with patch("subprocess.Popen", return_value=mock_proc):
        async def mock_sleep(delay):
            raise ValueError("stop loop")
            
        monkeypatch.setattr(asyncio, "sleep", mock_sleep)
        
        with patch("httpx.AsyncClient.get", side_effect=Exception("mocked offline")):
            try:
                await orchestrator.monitor_and_scan_loop(0.1)
            except ValueError as e:
                assert str(e) == "stop loop"
                
    backup_path = str(log_path) + ".1"
    assert os.path.exists(backup_path)
    with open(backup_path, "r") as f:
        assert f.read() == "A" * 15
        
    # Test active truncation
    mock_file_handle = open(log_path, "a")
    mock_file_handle.write("B" * 20)
    mock_file_handle.flush()
    
    orchestrator.RUNNING_PROCESSES = {
        "test-model-rotation": (mock_proc, mock_file_handle, str(log_path))
    }
    
    with patch("subprocess.Popen", return_value=mock_proc):
        with patch("httpx.AsyncClient.get", side_effect=Exception("mocked offline")):
            try:
                await orchestrator.monitor_and_scan_loop(0.1)
            except ValueError as e:
                assert str(e) == "stop loop"
                
    mock_file_handle.close()
    assert os.path.exists(log_path)
    assert os.path.getsize(log_path) == 0




