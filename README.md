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
- CUDA 12.9.1 runtime for the production serving image
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

### Aggregated SFT -> Reward -> RLHF Pipeline

```bash
python scripts/train_pipeline.py \
    --config config/training_pipeline.yaml \
    --stages all
```

The aggregated entrypoint runs the fixed stage order:
`sft -> reward -> rlhf -> training_eval -> mlflow_promotion -> artifact_manifest -> awq_handoff -> serving_handoff`.
It injects the SFT final checkpoint into reward/RLHF policy inputs, injects the
reward final checkpoint into RLHF, writes promotion metrics to
`outputs/training_pipeline/eval/training_candidate.json`, emits
`model_artifact_manifest.json` with checkpoint checksums, dataset fingerprints,
git SHA, MLflow lineage, and model-card metadata, and writes AWQ/serving
handoff manifests for deployment automation.

Useful options:

```bash
python scripts/train_pipeline.py --config config/training_pipeline.yaml --dry-run
python scripts/train_pipeline.py --config config/training_pipeline.yaml --output-dir ./outputs/run-001
python scripts/train_pipeline.py --config config/training_pipeline.yaml --baseline-metrics ./outputs/baseline.json
python scripts/train_pipeline.py --config config/training_pipeline.yaml --skip-promotion
python scripts/train_pipeline.py --config config/training_pipeline.yaml --resume
```

This pipeline still does not install a heavy AWQ converter by default. The AWQ
handoff contract validates the source checkpoint, expected target path, and
optional conversion command so a GPU conversion job can be run and audited
without confusing bitsandbytes INT4 output for vLLM-supported AWQ weights.

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
# Create gateway credentials before installing the chart. Use real values.
kubectl create namespace llm-inference
kubectl create secret generic gateway-auth \
  --namespace llm-inference \
  --from-literal=JWT_SECRET='<strong-random-secret>' \
  --from-literal=AUTH_USERS='<username>:<strong-password>'

# Deploy the serving path with the Helm chart:
# Client -> Ingress -> API gateway -> internal vLLM service -> vLLM pods
helm upgrade --install vllm-inference charts/vllm-inference \
  --namespace llm-inference \
  --create-namespace \
  --atomic \
  --wait \
  -f charts/vllm-inference/values-production.yaml

# Override image tags for a release candidate
helm upgrade --install vllm-inference charts/vllm-inference \
  --namespace llm-inference \
  --create-namespace \
  --atomic \
  --wait \
  -f charts/vllm-inference/values.yaml \
  -f charts/vllm-inference/values-production.yaml \
  --set-string image.tag=release-candidate \
  --set-string gateway.image.tag=release-candidate \
  --set-string baseImage.tag=release-candidate

# Or use the deployment helper, which defaults to Helm mode
python scripts/deploy.py --action deploy

# Install Prometheus Adapter if you use HPA custom metrics
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring --create-namespace \
  -f k8s/autoscaling/prometheus-adapter-values.yaml

# Optional GPU exporter; Prometheus/KEDA themselves remain external dependencies
kubectl apply -f k8s/monitoring/dcgm-exporter.yaml
```

The Helm chart renders the vLLM backend, API gateway, Redis, routing ConfigMap, parameterized Ingress, NetworkPolicies, PDBs, HPAs, and ServiceMonitors. The vLLM Service is `ClusterIP` and should not be exposed directly; the gateway is the serving entrypoint. Both gateway and vLLM expose `/metrics` on their main HTTP ports. vLLM HPA uses queue depth first (`llm_request_queue_size`), GPU saturation second (`llm_gpu_utilization_percent`), and CPU/memory as backup signals.

Production values assume External Secrets, cert-manager TLS, and managed Redis:
`values-production.yaml` renders an `ExternalSecret` for `gateway-auth`, a
cert-manager `Certificate`, TLS ingress, and Redis connection settings for an
external Redis endpoint. For local/dev clusters, keep the defaults and create
`gateway-auth` manually.

Canary serving uses a separate backend release and gateway routing weights:

```bash
helm upgrade --install vllm-inference-canary charts/vllm-inference \
  --namespace llm-inference \
  -f charts/vllm-inference/values-production.yaml \
  -f charts/vllm-inference/values-canary.yaml \
  --atomic --wait

python scripts/deploy.py --action deploy --strategy canary --deploy-mode helm \
  --values charts/vllm-inference/values.yaml \
  --values charts/vllm-inference/values-production.yaml \
  --canary-values charts/vllm-inference/values-canary.yaml \
  --canary-percentage 10
```

The chart also includes a disabled-by-default `trainingPipeline` Kubernetes Job
spec. Enable it only with a training image that contains this repo, training
dependencies, MLflow credentials, dataset/object-store access, and GPU capacity.
Build the reference image after the base image with
`docker build -f docker/Dockerfile.training --build-arg BASE_IMAGE=llm-base:latest -t llm-training:latest .`.
After an AWQ checkpoint is available, run `scripts/vllm_awq_smoke.py` against a
GPU-backed staging endpoint before promoting traffic.

KEDA autoscaling is available through chart values. Do not enable HPA and KEDA together because both manage the same Deployment:

```bash
helm upgrade --install vllm-inference charts/vllm-inference \
  --namespace llm-inference \
  --create-namespace \
  -f charts/vllm-inference/values.yaml \
  --set hpa.enabled=false \
  --set keda.enabled=true
```

Legacy raw manifests remain under `k8s/` for reference/dev fallback. If
`MODEL_NAME` is an absolute path such as `/workspace/models/awq/checkpoint-final`,
the `model-pvc` volume must already contain a vLLM-supported AWQ checkpoint at
that path before the vLLM Pod starts. Production deployments should use the Helm
chart above.

```bash
kubectl create namespace llm-inference
kubectl create secret generic gateway-auth \
  --namespace llm-inference \
  --from-literal=JWT_SECRET='<strong-random-secret>' \
  --from-literal=AUTH_USERS='<username>:<strong-password>'
kubectl apply -f k8s/deployments/
kubectl apply -f k8s/autoscaling/vllm-keda-scaledobject.yaml  # optional; do not combine with the HPA
```

Production serving runtime matrix:

| Component | Version |
| --- | --- |
| CUDA image | `12.9.1-cudnn-runtime-ubuntu22.04` |
| Python | `3.10` |
| PyTorch | `2.10.0` |
| Transformers | `4.57.6` |
| vLLM | `0.19.1` |
| FastAPI / Pydantic | `0.136.1` / `2.13.3` |

CI enforces this matrix with `python scripts/check_serving_runtime_matrix.py`.

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
In Kubernetes, use the gateway Ingress for client traffic. Direct vLLM access is for debugging only, for example `kubectl port-forward -n llm-inference svc/vllm-service 8000:80`.

The API gateway also provides production-serving controls:
- `RATE_LIMIT_REQUESTS_PER_MINUTE` / `RATE_LIMIT_REQUESTS_PER_HOUR`: request quota.
- `RATE_LIMIT_TOKENS_PER_MINUTE` / `RATE_LIMIT_TOKENS_PER_HOUR`: estimated prompt plus max-token quota before dispatch.
- `MODEL_ROUTING_CONFIG`: JSON route manifest for registry-driven stable/canary model routing.
- `/ready`: unauthenticated Kubernetes readiness check for routing config, Redis, and active backend `/ready`.
- `/usage`: authenticated per-user totals split by model and endpoint.
- Streaming metrics: `llm_time_to_first_token_seconds`, `llm_inter_token_latency_seconds`, and `llm_tokens_per_second`.

### Platformization Controls

Priority order for production platform hardening:

1. Registry-driven routing: configure logical model names in `config/model_routing.local.json` or `config/model_routing.gpu.json` for Compose, or `gateway.routing` values in Helm for Kubernetes. Each logical model can route to weighted `stable` and `canary` targets with separate backend URLs and physical model IDs.
2. Canary and rollback: deploy `vllm-inference-canary` with `values-canary.yaml`, then shift deterministic per-user traffic through gateway weights. Helm `--atomic --wait` and `helm rollback` are the rollback path.
3. Eval gate: `config/eval_gate.training.json` protects training handoff metrics; `config/eval_gate.production.json` adds task, safety, latency, cost, checksum, and baseline regression requirements for real promotion.
4. Alerting: Prometheus loads `docker/alerts.yml` locally and `k8s/monitoring/alerts.yml` in Kubernetes-style deployments.
5. Autoscaling: Helm values can switch between the default HPA and optional KEDA queue/GPU/active-request based scaling.

Example eval gate:

```bash
python scripts/eval_gate.py \
    --metrics outputs/evals/candidate.json \
    --baseline outputs/evals/production.json \
    --config config/eval_gate.json
```

Optional KEDA scale-out with Helm:

```bash
helm upgrade --install vllm-inference charts/vllm-inference \
    --namespace llm-inference \
    --create-namespace \
    -f charts/vllm-inference/values.yaml \
    --set hpa.enabled=false \
    --set keda.enabled=true
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
LOAD_TEST_MODEL=/workspace/models/awq/checkpoint-final \
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
    model_name="llama3-8b-sft",
    model_type="sft",
    metrics={"eval_loss": 0.5},
    tags={"task": "instruction-following"}
)

# Deploy to production
registry.transition_model_stage(
    model_name="llama3-8b-sft",
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
