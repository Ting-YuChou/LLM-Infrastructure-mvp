# LLM Infrastructure MVP

A complete end-to-end infrastructure for large language model training and deployment, including Supervised Fine-Tuning (SFT), Reward Model Training, RLHF Alignment Training, high-performance inference serving, model registry, and monitoring systems.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Serving & Deployment](#serving--deployment)
- [Configuration](#configuration)
- [Development Guide](#development-guide)
- [Contributing](#contributing)

## ✨ Features

### 🎯 Core Capabilities

- **Complete Training Pipeline**
  - ✅ Supervised Fine-Tuning (SFT)
  - ✅ Reward Model Training
  - ✅ RLHF with PPO - Reinforcement Learning Alignment (Full Implementation)
  - ✅ LoRA Support - Parameter-Efficient Fine-Tuning

- **High-Performance Inference**
  - ✅ vLLM Inference Engine (PagedAttention + Continuous Batching)
  - ✅ OpenAI-Compatible API
  - ✅ Streaming Generation Support
  - ✅ Model Quantization (INT8/INT4)

- **Production-Ready Infrastructure**
  - ✅ API Gateway (Authentication, Rate Limiting, Routing)
  - ✅ MLflow Model Registry & Versioning
  - ✅ Prometheus + Grafana Monitoring
  - ✅ Kubernetes Deployment Configs
  - ✅ Docker Compose for Local Development

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
├─────────────────────────────────────────────────────────┤
│  SFT → Reward Model → RLHF (PPO) → Model Registry       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   Serving Infrastructure                 │
├─────────────────────────────────────────────────────────┤
│  API Gateway → vLLM Server → Monitoring                 │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
llm-infrastructure-mvp/
├── config/                 # Configuration files
│   ├── sft_config.yaml     # SFT training config
│   ├── reward_config.yaml  # Reward model config
│   ├── rlhf_config.yaml    # RLHF training config
│   └── serving_config.yaml # Serving config
│
├── src/                    # Core source code
│   ├── training/          # Training modules
│   │   ├── sft_trainer.py      # SFT trainer
│   │   ├── reward_trainer.py   # Reward model trainer
│   │   └── rlhf_trainer.py     # RLHF trainer (Full PPO implementation)
│   │
│   ├── serving/           # Inference serving
│   │   └── vllm_server.py     # vLLM inference server
│   │
│   ├── api/              # API layer
│   │   └── gateway.py         # API Gateway
│   │
│   ├── registry/         # Model registry
│   │   └── mlflow_client.py   # MLflow client
│   │
│   ├── monitoring/       # Monitoring
│   │   └── metrics.py         # Prometheus metrics
│   │
│   └── optimization/     # Optimization tools
│       └── quantization.py    # Model quantization
│
├── scripts/              # Utility scripts
│   ├── train_sft.py      # SFT training entry point
│   ├── prepare_data.py   # Data preparation
│   ├── deploy.py         # Deployment scripts
│   ├── benchmark.py      # Performance benchmarking
│   └── load_test.py      # Load testing
│
├── tests/                # Tests
│   ├── unit/             # Unit tests
│   └── integration/     # Integration tests
│
├── docker/               # Docker configuration
│   └── Dockerfile.base
│
├── k8s/                  # Kubernetes configuration
│   ├── deployments/      # Deployment manifests
│   └── monitoring/       # Monitoring configs
│
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU training and inference)
- Docker & Docker Compose (optional, for service deployment)
- At least 16GB GPU memory (24GB+ recommended)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd llm-infrastructure-mvp
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt

# Optional: Flash Attention (for faster training)
pip install flash-attn --no-build-isolation

# Optional: For RLHF training
pip install git+https://github.com/volcengine/verl.git
```

4. **Prepare data**
```bash
python scripts/prepare_data.py \
    --output-dir ./data \
    --sft-examples 1000 \
    --preference-examples 500 \
    --rl-prompts 200
```

## 🎓 Training Pipeline

### 1. Supervised Fine-Tuning (SFT)

```bash
python scripts/train_sft.py \
    --config config/sft_config.yaml \
    --use-lora  # Optional: Use LoRA for parameter-efficient fine-tuning
```

**Configuration**:
- Modify model path, data path, and hyperparameters in `config/sft_config.yaml`
- Supports both LoRA and full parameter fine-tuning
- Automatic logging to TensorBoard and MLflow

### 2. Reward Model Training

```bash
python -m src.training.reward_trainer \
    --config config/reward_config.yaml
```

**Data Format**:
```json
[
  {
    "prompt": "Question or instruction",
    "chosen": "Better response",
    "rejected": "Worse response"
  }
]
```

### 3. RLHF Training (PPO)

```bash
python -m src.training.rlhf_trainer \
    --config config/rlhf_config.yaml \
    --prompts Anthropic.Dataset/rl_prompts.json \
    --eval-prompts Anthropic.Dataset/rl_prompts.json  # Optional
```

**Complete Features**:
- ✅ Full PPO algorithm implementation
- ✅ GAE (Generalized Advantage Estimation)
- ✅ Value Function training
- ✅ Adaptive KL Penalty
- ✅ Multiple PPO epochs
- ✅ Complete training metrics tracking

**Training Flow**:
1. Rollout: Generate responses and compute rewards
2. Advantage Computation: Calculate advantages using GAE
3. PPO Update: Multiple policy update epochs
4. Adaptive KL: Dynamically adjust KL coefficient

## 🚢 Serving & Deployment

### Local Development on Mac / CPU-only

```bash
# Start mock serving, gateway, redis, Prometheus, and Grafana
docker compose -f docker-compose.local.yml up -d --build

# If localhost:3000 is already in use
GRAFANA_PORT=3001 docker compose -f docker-compose.local.yml up -d --build

# Verify gateway contract, streaming, usage, and metrics
python scripts/compose_smoke_test.py --base-url http://localhost:8080

# View gateway logs
docker compose -f docker-compose.local.yml logs -f api-gateway

# Stop services
docker compose -f docker-compose.local.yml down
```

**Service Ports**:
- Mock LLM API: `http://localhost:8000`
- API Gateway: `http://localhost:8080`
- Prometheus: `http://localhost:9091`
- Grafana: `http://localhost:${GRAFANA_PORT:-3000}` (admin/admin)

### GPU Development on Linux + NVIDIA

```bash
# Start vLLM, gateway, redis, Prometheus, and Grafana
docker compose -f docker-compose.gpu.yml up -d --build

# If localhost:3000 is already in use
GRAFANA_PORT=3001 docker compose -f docker-compose.gpu.yml up -d --build

# View vLLM logs
docker compose -f docker-compose.gpu.yml logs -f vllm-server

# Stop services
docker compose -f docker-compose.gpu.yml down
```

**Service Ports**:
- vLLM API: `http://localhost:8000`
- API Gateway: `http://localhost:8080`
- Prometheus: `http://localhost:9091`
- Grafana: `http://localhost:${GRAFANA_PORT:-3000}` (admin/admin)

### Full GPU Stack

```bash
# Start the default GPU-oriented compose stack (includes MLflow)
docker compose up -d --build
```

### Production Deployment (Kubernetes)

```bash
# Deploy vLLM service
kubectl apply -f k8s/deployments/vllm-deployment.yaml

# Deploy monitoring
kubectl apply -f k8s/monitoring/

# Install Prometheus Adapter if you use the HPA custom metrics in the manifest
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring --create-namespace \
  -f k8s/autoscaling/prometheus-adapter-values.yaml
```

The provided Kubernetes manifest scrapes `/metrics` from the main vLLM HTTP port. Its HPA uses queue depth first (`llm_request_queue_size`), GPU saturation second (`llm_gpu_utilization_percent`), and CPU/memory as backup signals. Queue/GPU/custom-metric autoscaling is also available as an optional KEDA manifest in `k8s/autoscaling/vllm-keda-scaledobject.yaml`; do not run the KEDA ScaledObject and the manifest HPA against the same Deployment at the same time.

### Start Inference Server

```bash
# Run directly
python -m src.serving.vllm_server \
    --config config/serving_config.yaml \
    --model /path/to/awq-model

# AWQ serving uses vLLM weight quantization, typically 4-bit weights with
# FP16/BF16 activations. It is not the same thing as FP8 quantization.
QUANTIZATION=awq DTYPE=float16 MODEL_NAME=/path/to/awq-model \
python -m src.serving.vllm_server --config config/serving_config.yaml

# Or use API Gateway
JWT_SECRET=change-me AUTH_USERS=local:local \
python -m src.api.gateway --host 0.0.0.0 --port 8080
```

Both the vLLM server and the API gateway expose Prometheus metrics on their main HTTP ports:
- vLLM metrics: `http://localhost:8000/metrics`
- API Gateway metrics: `http://localhost:8080/metrics`

The API gateway also provides production-serving controls:
- `RATE_LIMIT_REQUESTS_PER_MINUTE` / `RATE_LIMIT_REQUESTS_PER_HOUR`: request quota.
- `RATE_LIMIT_TOKENS_PER_MINUTE` / `RATE_LIMIT_TOKENS_PER_HOUR`: estimated prompt plus max-token quota before dispatch.
- `MODEL_ROUTING_CONFIG`: JSON route manifest for registry-driven stable/canary model routing.
- `/usage`: authenticated per-user totals split by model and endpoint.
- Streaming metrics: `llm_time_to_first_token_seconds`, `llm_inter_token_latency_seconds`, and `llm_tokens_per_second`.

### Platformization Controls

Priority order for production platform hardening:

1. Registry-driven routing: configure logical model names in `config/model_routing.local.json` or `config/model_routing.gpu.json`. Each logical model can route to weighted `stable` and `canary` targets with separate backend URLs and physical model IDs.
2. Canary and rollback: raise the canary target `weight` to shift deterministic per-user traffic; set it back to `0` for immediate rollback without changing client-facing model names.
3. Eval gate: block promotion unless candidate metrics pass absolute thresholds and baseline regression guards.
4. Alerting: Prometheus loads `docker/alerts.yml` locally and `k8s/monitoring/alerts.yml` in Kubernetes-style deployments.
5. Autoscaling: `k8s/autoscaling/vllm-keda-scaledobject.yaml` is an optional KEDA template for queue, GPU utilization, and active-request based scaling.

Example eval gate:

```bash
python scripts/eval_gate.py \
    --metrics outputs/evals/candidate.json \
    --baseline outputs/evals/production.json \
    --config config/eval_gate.json
```

Optional KEDA scale-out manifest:

```bash
kubectl apply -f k8s/autoscaling/vllm-keda-scaledobject.yaml
```

### API Usage Example

```python
import requests

# The Docker Compose stacks configure AUTH_USERS=local:local,loadtest:loadtest123.
# Get authentication token
response = requests.post("http://localhost:8080/auth/token", json={
    "username": "local",
    "password": "local"
})
token = response.json()["access_token"]

# Send inference request
response = requests.post(
    "http://localhost:8080/v1/completions",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "prompt": "Explain what machine learning is",
        "max_tokens": 256,
        "temperature": 0.7
    }
)
print(response.json())
```

## ⚙️ Configuration

### SFT Configuration (`config/sft_config.yaml`)

Key parameters:
- `model.name`: Base model path
- `data.train_file`: Training data path
- `training.num_train_epochs`: Number of training epochs
- `training.learning_rate`: Learning rate
- `lora.enabled`: Whether to use LoRA

### RLHF Configuration (`config/rlhf_config.yaml`)

Key parameters:
- `model.policy.name`: Policy model (SFT model)
- `model.reward.name`: Reward model path
- `algorithm.ppo.clip_range`: PPO clipping range
- `algorithm.ppo.target_kl`: Target KL divergence
- `training.total_steps`: Total training steps

### Serving Configuration (`config/serving_config.yaml`)

Key parameters:
- `model.name`: Model path for serving
- `gpu.tensor_parallel_size`: Tensor parallel size
- `gpu.gpu_memory_utilization`: GPU memory utilization

## 🛠️ Development Guide

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Coverage report
pytest --cov=src tests/
```

### Benchmarking and Load Testing

```bash
# Benchmark through the API gateway
python scripts/benchmark.py \
    --endpoint http://localhost:8080 \
    --username local \
    --password local \
    --sweep concurrent=1,4,8 \
    --sweep max_tokens=64,256 \
    --output-dir outputs/benchmarks/local-gateway

# Locust load test through the API gateway
locust -f scripts/load_test.py --host=http://localhost:8080

# Optional: target a specific backend model directly
LOAD_TEST_MODEL=/workspace/models/aligned/checkpoint-final \
locust -f scripts/load_test.py --host=http://localhost:8080
```

### Code Style

```bash
# Format code
black src/ scripts/

# Lint
flake8 src/
mypy src/
```

### Model Quantization

```bash
python -m src.optimization.quantization \
    --model-path /path/to/model \
    --output-path /path/to/quantized \
    --method int8  # or int4, dynamic, better-transformer
```

## 📊 Monitoring & Tracking

### MLflow Model Registry

```python
from src.registry.mlflow_client import LLMModelRegistry

registry = LLMModelRegistry(
    tracking_uri="http://localhost:5000",
    experiment_name="llm-training"
)

# Register model
version = registry.register_model(
    model_path="./outputs/sft/final",
    model_name="llama2-7b-sft",
    model_type="sft",
    metrics={"eval_loss": 0.5},
    tags={"task": "instruction-following"}
)

# Deploy to production
registry.transition_model_stage(
    model_name="llama2-7b-sft",
    version=version,
    stage="Production"
)
```

### Prometheus Metrics

Prometheus UI: `http://localhost:9091`

Service metrics endpoints:
- vLLM server: `http://localhost:8000/metrics`
- API gateway: `http://localhost:8080/metrics`

Key metrics:
- `llm_requests_total`: Total number of requests
- `llm_request_duration_seconds`: Request latency
- `llm_tokens_processed_total`: Number of tokens processed
- `llm_active_requests`: In-flight requests

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [vLLM](https://github.com/vllm-project/vllm)
- [MLflow](https://mlflow.org/)
- [DeepSpeed](https://www.deepspeed.ai/)

## 📧 Contact

For questions or suggestions, please open an Issue or contact the maintainers.

---

**Note**: This project is an MVP version for learning and prototyping. Please thoroughly test and optimize before using in production environments.
