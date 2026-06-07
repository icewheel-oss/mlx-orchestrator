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
* **HTTPS/SSL (`ssl_certfile` & `ssl_keyfile`):** Provide paths to your `.pem` / `.crt` certificate and key files to serve the proxy securely over HTTPS. Defaults to `null` (plain HTTP).
* **Types:** Use `"lm"` or `"vlm"` depending on the model's architecture:
  * **`lm` (Language Model):** Text-only models. Runs under the `mlx-lm` backend wrapper. Best for coding, text chat, and reasoning (e.g. Qwen 3.6, Llama 3.1, DeepSeek Distill).
  * **`vlm` (Vision-Language Model):** Multimodal/vision-enabled models. Runs under the `mlx-vlm` backend wrapper. Necessary for analyzing screenshots, photos, and diagrams (e.g. Gemma 4, Llama 3.2 Vision).

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
python orchestrator.py serve [--port PORT] [--global-opencode]
```
Starts the FastAPI routing proxy and process watchdog. *During startup, it will run a diagnostic check on your physical RAM vs the size of enabled models and print a warning if you exceed safe limits (70% of total unified memory). The `--global-opencode` flag writes the OpenCode configuration profile globally at `~/.opencode/opencode.json`.*

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

## 🗂️ Logs
To keep your terminal clean, the stdout/stderr outputs of all managed model processes are routed to log files in the `logs/` directory:
- `logs/mlx-community_Qwen3.6-35B-A3B-4bit.log`
- `logs/mlx-community_gemma-4-31b-bf16.log`

## 🚨 Troubleshooting & Handling Failures

Here is how to solve common issues you might encounter:

### 1. Extremely slow response speeds / Mac runs very hot
* **Symptom:** Tokens generate very slowly (1-2 tokens per second), your Mac's fans spin up loudly, and your CPU goes to 100% load.
* **Cause:** The model is too large for your GPU memory limit, forcing macOS to run the model on the CPU instead.
* **Solution:** 
  1. Run the diagnostics tool to inspect your system limits:
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
  * You can also change the proxy port by opening `config.json` and changing `"proxy_port": 12500` to a different number (e.g. `13500`).

### 4. Model download fails or hangs
* **Symptom:** The `download` command times out or gets cut off due to network disconnection.
* **Solution:** Simply re-run the download command. Hugging Face downloads are fully resumable and will pick up right where they left off.

### 5. Backend model fails to start
* **Symptom:** Proxy starts, but requests for a specific model fail with a `503 Service Unavailable` error.
* **Solution:** Check the log files in the `logs/` directory (e.g., `logs/mlx-community_Qwen3.6-35B-A3B-4bit.log`) to read the exact traceback. Common causes include:
  * Typo in the model name in your config file.
  * Trying to run a standard PyTorch model instead of an MLX-converted format (make sure it has `mlx` in the name).

### 6. Typo or malformed settings in `config.json`
* **Symptom:** You made a typo or entered malformed settings in `config.json` while the orchestrator was running.
* **Solution:** The MLX Orchestrator is designed to be fully self-healing and gracefully handles malformed syntax (e.g., incorrect JSON brackets) or schema errors (e.g., missing dictionary entries or type mismatches). It will output a clean warning to the terminal logs and safely fall back to the last valid settings or defaults without crashing. Simply correct the typo in `config.json` and it will auto-load your corrections dynamically in the next background cycle (every 5 seconds).

---

## 📄 License

This project is licensed under the GNU General Public License Version 3 (GPLv3) - see the [LICENSE.txt](file:///Users/tony/programming/workspace/oss/ai/mlx-orchestrator/LICENSE.txt) file for details.
