# MLX Orchestrator

A lightweight, self-healing process manager and unified proxy for serving multiple native **Apple Silicon MLX** models concurrently.

MLX (Apple's machine learning framework) is exceptionally fast on Apple Silicon. However, native serving wrapper tools (`mlx-lm.server` and `mlx-vlm.server`) only support running a single model per process. **MLX Orchestrator** solves this by providing a unified gateway that automatically manages backend model servers, watches their health, auto-restarts them if they crash, and dynamically routes OpenAI-compatible API requests.

---

## ✨ Features

- **Unified OpenAI Endpoint:** Access all your MLX models from a single base URL (e.g., `http://127.0.0.1:12500/v1`).
- **Process Manager Watchdog:** Launches your model servers as background subprocesses, monitors their status, and restarts them automatically if they exit or crash (ideal for tight-memory swapping limits).
- **Auto-Discovery Port Scanner:** Scans local ports (`12501–12520` by default) every few seconds, auto-discovers active OpenAI-compatible model backends, and registers their routes dynamically.
- **Dynamic Config Handling:** Turn models on or off on the fly by changing the `"enabled"` flag in `config.json`—the orchestrator will automatically spin up or tear down subprocesses.
- **Unified `/v1/models`:** Serves an aggregated models registry endpoint showing only models that are currently online.
- **Built-in Model Downloader:** Easily download models from Hugging Face via the command line with standard resume support and a live progress bar.

---

## 🚀 Quick Start (Step-by-Step for Everyone)

### 1. Clone & Install Dependencies
Open your **Terminal** app (press `Cmd + Space`, type "Terminal", and press Enter) and copy-paste these commands line-by-line:

```bash
# 1. Clone the project code
git clone https://github.com/icewheel-oss/mlx-orchestrator.git
cd mlx-orchestrator

# 2. Create a clean Python environment (virtualenv)
python3 -m venv venv

# 3. Activate the environment
source venv/bin/activate

# 4. Install the required libraries
pip install -r requirements.txt
```

### 2. Pre-Download a Model (Optional)
Use the built-in CLI downloader to fetch any model from Hugging Face (such as the Gemma 4 or Qwen models):

```bash
python orchestrator.py download mlx-community/Qwen3.6-35B-A3B-4bit
```

### 3. Configure Your Models
When you first run the server, a default `config.json` will be generated in your project directory:

```json
{
  "proxy_port": 12500,
  "scan_interval_seconds": 5,
  "scan_ports_min": 12501,
  "scan_ports_max": 12520,
  "api_key": null,
  "log_token_usage": true,
  "global_opencode": false,
  "debug_mode": false,
  "max_log_size_bytes": 10485760,
  "ssl_certfile": null,
  "ssl_keyfile": null,
  "managed_servers": [
    {
      "model": "mlx-community/gemma-4-e2b-it-4bit",
      "port": 12501,
      "type": "vlm",
      "enabled": true
    },
    {
      "model": "mlx-community/gemma-4-e4b-it-4bit",
      "port": 12502,
      "type": "vlm",
      "enabled": false
    },
    {
      "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
      "port": 12503,
      "type": "lm",
      "enabled": false
    },
    {
      "model": "mlx-community/gemma-4-31b-bf16",
      "port": 12504,
      "type": "vlm",
      "enabled": false
    }
  ]
}
```

* **Toggle models live:** Set `"enabled": true` or `false` to spin servers up or down.
* **Security (`api_key`):** Set this to a string (e.g. `"my-secret-key"`) to require authorization. Clients must then supply the `Authorization: Bearer <key>` header. Set to `null` to disable auth (default).
* **Observability (`log_token_usage`):** Set to `true` (default) to output token metrics (prompt size, generated tokens, tokens/sec generation speed, and elapsed time) to the proxy's terminal after every streaming/non-streaming completion.
* **Global OpenCode Config (`global_opencode`):** Set to `true` to generate the OpenCode configuration profile globally at `~/.opencode/opencode.json` in addition to the project directory. Defaults to `false`.
* **Debug Mode (`debug_mode`):** Set to `true` to log all incoming request headers, query parameters, payloads, downstream requests, downstream response status/headers, streaming chunks in real-time, and non-streaming bodies (with sensitive `Authorization` headers masked). Defaults to `false`.
* **Log Limits (`max_log_size_bytes`):** Maximum size in bytes (default: `10485760` / 10MB) before rotating logs to `.log.1` on startup or truncating active model server log files in-place during runtime.
* **HTTPS/SSL (`ssl_certfile` & `ssl_keyfile`):** Provide paths to your `.pem` / `.crt` certificate and key files to serve the proxy securely over HTTPS. Defaults to `null` (plain HTTP).
* **Types:** Use `"lm"` or `"vlm"` depending on the model's architecture:
  * **`lm` (Language Model):** Text-only models. Runs under the `mlx-lm` backend wrapper. Best for coding, text chat, and reasoning (e.g. Qwen 3.6, Llama 3.1, DeepSeek Distill).
  * **`vlm` (Vision-Language Model):** Multimodal/vision-enabled models. Runs under the `mlx-vlm` backend wrapper. Necessary for analyzing screenshots, photos, and diagrams (e.g. Gemma 4, Llama 3.2 Vision).
* **Model-Specific Integration Parameters (for Junie profiles):**
  * **`temperature`:** Optional number (e.g. `0.3`). Custom sampling temperature to write into the Junie profile.
  * **`faster_model`:** Optional string. Explicit model ID to use as the `"fasterModel"` role in Junie. If omitted, the orchestrator automatically detects and selects the smallest enabled model (whose parameter size is smaller than the current model) as the helper!

### 4. Run the Orchestrator
Start the main gateway:

```bash
python orchestrator.py serve
```

Now, configure your client applications (like OpenCode or Goose) to point to the unified endpoint `http://127.0.0.1:12500/v1` (or `https://127.0.0.1:12500/v1` if SSL is configured).

### 🤖 OpenCode & IntelliJ IDEA Integration

1. **Auto-Generated OpenCode Config File:**
   Running the orchestrator server (via `python orchestrator.py serve`) automatically generates or updates the `opencode.json` configuration profile in your project directory.
   *(This dynamically synchronizes the `baseURL`, HTTPS configuration, authorization keys, and the list of currently enabled models. You can also generate the profile globally at `~/.opencode/opencode.json` by running the server with the `--global-opencode` CLI flag or by setting `"global_opencode": true` in `config.json`).*

2. **Open OpenCode in the Project Directory:**
   Launch OpenCode within your project directory so it reads the local `opencode.json` profile:
   ```bash
   opencode .
   ```

3. **Install OpenCode Agent in IntelliJ IDEA:**
   - Open **IntelliJ IDEA**.
   - Navigate to **Settings** (on macOS: `IntelliJ IDEA` -> `Settings...` or `Cmd + ,`).
   - Go to the **AI Chat settings** (located under `Tools` -> `AI Chat` or similar).
   - Select and configure/install the **OpenCode Agent** from the list.

4. **Select OpenCode Agent in IntelliJ:**
   - Open the **AI Chat window** inside IntelliJ IDEA.
   - Choose **OpenCode Agent** from the list of agents.
   - The agent will connect to your local MLX Orchestrator, dynamically listing all of your active models!

### 🤖 Junie Integration

JetBrains Junie is a terminal-based AI assistant. The orchestrator automatically generates and updates JSON profiles for Junie to let you chat with your active local models:

1. **Auto-Generated Profiles:**
   Running the orchestrator server automatically generates `.json` profile files for each enabled model under `.junie/models/` in your project directory.
   *(This synchronizes the `baseUrl`, API type (`OpenAICompletion`), and keys. If `"global_opencode": true` is configured or the `--global-opencode` CLI flag is passed, it will also write these profiles globally at `~/.junie/models/` so they are available system-wide).*

2. **Cleaning Inactive Models:**
   The orchestrator dynamically cleans up old generated model profiles from local and global directories when they are disabled in `config.json`, so they don't clutter your model selection.

3. **Using Custom Profiles in Junie:**
   Run the `junie` CLI pointing to the generated model profile. Custom models are referenced with the `custom:` prefix followed by the profile filename (without the `.json` extension):
   ```bash
   # Run Junie with the local Qwen model profile
   junie --model custom:mlx-qwen3.6-35b-a3b-4bit
   ```
---

## 🛠️ CLI Commands

```text
positional arguments:
  {serve,download,recommend}  Subcommands
    serve           Start the orchestrator proxy and managed MLX servers
    download        Pre-download a model from Hugging Face
    recommend       Display memory diagnostics and model recommendations
```

### Serve
```bash
python orchestrator.py serve [--port PORT] [--global-opencode] [--debug]
```
Starts the FastAPI routing proxy and process watchdog. *During startup, it will run a diagnostic check on your physical RAM vs the size of enabled models and print a warning if you exceed safe limits (70% of total unified memory). The `--global-opencode` flag writes the OpenCode configuration profile globally at `~/.opencode/opencode.json`. The `--debug` flag activates verbose request/response logging to terminal stdout.*

### Download
```bash
python orchestrator.py download <model_id>
```
Downloads a model snapshot directly from Hugging Face into your local cache folder.

### Recommend
```bash
python orchestrator.py recommend
```
Checks your Mac's total Unified Memory, lists the exact size on disk of all configured/downloaded models, and gives advice on which model sizes (4-bit vs 8-bit vs BF16) you can comfortably run.

---

## ⚙️ Advanced Features & Architecture

### 1. Interactive API Playground (Swagger UI)
Since the [orchestrator.py](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/orchestrator.py) server is built on top of the **FastAPI** web framework, it hosts automatic interactive OpenAPI documentation:
- **Access URL:** Simply open your web browser and navigate to the proxy root (e.g., `http://127.0.0.1:12500/`).
- **Redirect behavior:** The root URL automatically redirects you to `/docs` (e.g., `http://127.0.0.1:12500/docs`), which renders the Swagger UI.
- **Use case:** You can interactively test the routing proxy endpoints (like `GET /v1/models` or `POST /v1/chat/completions`), review the required JSON schemas, and test authorized requests by setting your `api_key` using the **Authorize** button.

### 2. Verbose Debug Mode & Security Masking
When diagnosing connection errors, custom client configurations, or token-generation bugs, you can enable verbose request/response logging:
- **How to enable:**
  - Set `"debug_mode": true` in [config.json](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/config.json), OR
  - Run the server command with the `--debug` CLI flag (`python orchestrator.py serve --debug`).
- **What it logs:**
  - **Client Request Headers:** Displays incoming headers (with the sensitive `Authorization` token automatically masked as `Bearer ********` to prevent security leaks).
  - **Request Body & Parameters:** Logs query parameters and raw JSON payloads sent by client IDEs/applications.
  - **Downstream Requests:** Prints the exact backend server port and URL target for routed payloads.
  - **Downstream Responses:** Logs HTTP response status code, response headers, and non-streaming response body.
  - **Real-Time Stream Chunks:** Prints individual event-stream chunks (e.g., `data: {...}`) as they arrive from downstream servers.
  - **Uvicorn Access Logs:** Automatically enables `access_log` in Uvicorn to trace every HTTP transaction.
  
All of these details are logged to both the terminal stdout and the `logs/orchestrator.log` file.

### 3. Automated Log Rotation & In-Place Active Truncation
Because running large language models generates substantial output logs during load and evaluation, MLX Orchestrator prevents log files from consuming all your disk space:
- **Startup Rotation:** Whenever a backend model server starts, the orchestrator checks if its existing log in `logs/{model_name}.log` exceeds the `"max_log_size_bytes"` limit (default: 10MB). If it does, it moves the file to `logs/{model_name}.log.1`, overwriting any previous backup.
- **Runtime Active Truncation:** While the model is running, the background watchdog thread continuously monitors active log files. If a log exceeds the limit at runtime, the watchdog calls Python's `truncate(0)` on the open file handle. This resets the log file size to 0 bytes dynamically in-place **without** closing the stream or restarting the backend process, protecting your disk space while maintaining open file handles.

### 4. Gemma 4 & Strict-Override weight compatibility
- **The Issue:** Newer architectures (like Gemma 4) include redundant Key-Value parameters or slightly mismatched weights structures compared to standard base architectures. Loading these models using raw CLI tools (`mlx-lm.server` or `mlx-vlm.server`) often throws a strict schema validation exception, preventing the server from starting.
- **The Solution:** Inside its process manager [monitor_and_scan_loop](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/orchestrator.py#L231), the MLX Orchestrator dynamically wraps weight loading by monkeypatching the MLX weight loader (`mlx.nn.Module.load_weights`) to execute with `strict=False`. This transparently enables out-of-the-box support for Gemma 4 models and other advanced architectures without requiring manual codebase modifications or model-conversion hacks.

### 5. Graceful Self-Healing Config Management
The orchestrator reads and updates its state from [config.json](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/config.json) every 5 seconds:
- **Typo Tolerance:** If you make a syntax error (such as a missing bracket, comma, or quotation mark) while editing [config.json](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/config.json) during server runtime, the config loader [load_config](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/orchestrator.py#L93) will catch the exception, output a clean warning to the terminal logs, and fall back to the last valid settings or default values.
- **Auto-Reloading:** The orchestrator will not crash. Once you fix the typo in [config.json](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/config.json) and save, it will automatically detect and load the updated settings on the very next background cycle.

---

## 🗂️ Logs

The MLX Orchestrator separates its logs to keep your terminal output readable:
1. **Gateway Logs (Terminal stdout & `logs/orchestrator.log`):** Shows orchestrator startup, configuration updates, active port scans, API routing targets, request forwarding, token usage statistics, and debug output (including full request/response JSON details).
2. **Model Process Logs (`logs/` directory):** The raw stdout and stderr outputs of each managed backend subprocess are routed to dedicated files under `logs/`.
   - File format: `logs/<org>_<model_name>.log` (e.g., `logs/mlx-community_Qwen3.6-35B-A3B-4bit.log`).
   - If a model fails to start or crashes, **always check these files first** to see the python traceback from the underlying server wrapper.

---

## 🚨 Troubleshooting & Handling Failures

Here is how to diagnose and resolve issues you might encounter:

### 1. Extremely slow response speeds / Mac runs very hot
* **Symptom:** Tokens generate very slowly (1-2 tokens per second), your Mac's fans spin up loudly, and your CPU goes to 100% load.
* **Cause:** The model is too large for your GPU memory limit, forcing macOS to run the model on the CPU instead.
* **Solution:** 
  1. Run the diagnostics tool [recommend_models](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/orchestrator.py#L857) to inspect your system limits:
     ```bash
     python orchestrator.py recommend
     ```
  2. Increase your Mac's maximum GPU memory allocation. Find your Mac's total RAM capacity in the table below and run the corresponding command. These recommended limits explicitly reserve enough system headroom (3 GB to 24 GB) to run macOS, web browsers (like Chrome/Safari), and IDEs (like VS Code/Cursor/IntelliJ) comfortably alongside MLX models:

     | Total RAM | Recommended GPU Limit | Reserved Headroom | Allocation (MB) | Terminal Command |
     | :--- | :--- | :--- | :--- | :--- |
     | **8 GB** | 5.0 GB | 3.0 GB | `5120` | `sudo sysctl iogpu.wired_limit_mb=5120` |
     | **16 GB** | 11.5 GB | 4.5 GB | `11776` | `sudo sysctl iogpu.wired_limit_mb=11776` |
     | **24 GB** | 18.0 GB | 6.0 GB | `18432` | `sudo sysctl iogpu.wired_limit_mb=18432` |
     | **32 GB** | 25.0 GB | 7.0 GB | `25600` | `sudo sysctl iogpu.wired_limit_mb=25600` |
     | **36 GB** | 28.0 GB | 8.0 GB | `28672` | `sudo sysctl iogpu.wired_limit_mb=28672` |
     | **48 GB** | 38.0 GB | 10.0 GB | `38912` | `sudo sysctl iogpu.wired_limit_mb=38912` |
     | **64 GB** | 52.0 GB | 12.0 GB | `53248` | `sudo sysctl iogpu.wired_limit_mb=53248` |
     | **96 GB** | 80.0 GB | 16.0 GB | `81920` | `sudo sysctl iogpu.wired_limit_mb=81920` |
     | **128 GB** | 108.0 GB | 20.0 GB | `110592` | `sudo sysctl iogpu.wired_limit_mb=110592` |
     | **192 GB** | 168.0 GB | 24.0 GB | `172032` | `sudo sysctl iogpu.wired_limit_mb=172032` |

     > [!TIP]
     > To restore the default dynamic macOS GPU allocation behavior, run:
     > ```bash
     > sudo sysctl iogpu.wired_limit_mb=0
     > ```

### 2. "ModuleNotFoundError: No module named '...'"
* **Symptom:** Running the script outputs an import error.
* **Cause:** You forgot to activate the virtual environment (`venv`) or your terminal session restarted.
* **Solution:** Reactivate your environment and run:
  ```bash
  source venv/bin/activate
  ```

### 3. Port Conflict / Killing Stuck Processes ("Address already in use")
* **Symptom:** The proxy or model servers fail to start, or logs report port conflict errors like `address already in use`.
* **Cause:** Another application is using the port, or a prior orchestrator/model server process was orphaned and is still running in the background.
* **Solution:**
  * To immediately terminate **all** running orchestrator and model server processes in one command, run:
    ```bash
    pkill -f "orchestrator.py|mlx_lm|mlx_vlm"
    ```
  * Alternatively, find the exact process ID (PID) holding a specific port and kill it manually:
    ```bash
    # 1. Find the PID running on port 12500 or 12501
    lsof -i :12500 -i :12501
    
    # 2. Kill the process using its PID
    kill -9 <PID_NUMBER>
    ```
  * You can also change the proxy port by opening [config.json](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/config.json) and changing `"proxy_port": 12500` to a different number (e.g. `13500`).

### 4. Model download fails or hangs
* **Symptom:** The `download` command times out or gets cut off due to network disconnection.
* **Cause:** Unstable network or Hugging Face Hub rate limits.
* **Solution:** Re-run the download command. Hugging Face downloads are fully resumable; they query local cache files and download only the missing chunks.

### 5. Backend model fails to start / Service Unavailable (503)
* **Symptom:** Requests for a specific model return a `503 Service Unavailable` error, or the model fails to load.
* **Solution:**
  1. Check the logs in the `logs/` directory for the specific model (e.g., `logs/mlx-community_Qwen3.6-35B-A3B-4bit.log`).
  2. If the log file is empty or does not exist, the process might have failed before writing output. Verify:
     - The model name is exactly correct in [config.json](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/config.json).
     - The model is download-complete (has weights and a config in Hugging Face cache).
     - You have `mlx-lm` or `mlx-vlm` installed (run `pip show mlx-lm mlx-vlm` to confirm).
  3. If you see a python traceback inside the model's log:
     - **Out of Memory:** The weights size exceeds system capacity. Disable unused models in [config.json](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/config.json) and adjust `sysctl iogpu.wired_limit_mb` as shown in step 1.
     - **Mismatched architecture:** Ensure the model is converted to MLX format (model name contains `mlx`). The orchestrator does not serve raw PyTorch models directly.

### 6. Typo or malformed settings in `config.json`
* **Symptom:** You made a typo or entered malformed settings in `config.json` while the orchestrator was running.
* **Solution:** The MLX Orchestrator is designed to be fully self-healing. It will output a clean warning to the terminal logs and safely fall back to the last valid settings or defaults without crashing. Correct the typo in [config.json](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/config.json), and it will auto-load your corrections dynamically in the next background cycle (every 5 seconds).

### 7. Interactive API docs (Swagger UI) failing to load
* **Symptom:** Navigating to `http://127.0.0.1:12500/` or `/docs` returns a connection error in the browser.
* **Solution:**
  - Confirm that the MLX Orchestrator process is running in the terminal.
  - Check if you have enabled HTTPS/SSL in [config.json](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/config.json) (by setting `"ssl_certfile"` and `"ssl_keyfile"`). If HTTPS is enabled, you must access the API using `https://127.0.0.1:12500/` instead of `http`.

---

## 📄 License

This project is licensed under the GNU General Public License Version 3 (GPLv3) - see the [LICENSE.txt](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/LICENSE.txt) file for details.
