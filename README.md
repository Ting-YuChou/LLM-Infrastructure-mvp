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
│   ├── autoscaling/      # Prometheus Adapter custom metric rules
│   ├── deployments/      # Deployment manifests
│   └── monitoring/       # Monitoring configs
│
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### Requirements

- Python 3.10+
- CUDA 12.9-compatible NVIDIA driver for the provided Docker images
- Docker & Docker Compose (optional, for service deployment)
- At least 16GB GPU memory (24GB+ recommended)

Serving runtime matrix:
- vLLM `0.19.1`
- PyTorch `2.10.0` / torchvision `0.25.0` / torchaudio `2.10.0`
- CUDA `12.9.1` cuDNN Docker base image

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

### Local Development (Docker Compose)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f vllm-server

# Stop services
docker-compose down
```

**Service Ports**:
- vLLM API: `http://localhost:8000`
- API Gateway: `http://localhost:8080`
- MLflow UI: `http://localhost:5000`
- Prometheus: `http://localhost:9091`
- Grafana: `http://localhost:3000` (admin/admin)

### Production Deployment (Kubernetes)

Prerequisites for GPU autoscaling:
- NVIDIA device plugin and GPU Feature Discovery labels on GPU nodes
- `metrics-server` for CPU/memory HPA metrics
- Prometheus or Prometheus Operator scraping `llm-inference` and `monitoring`
- Prometheus Adapter installed with `k8s/autoscaling/prometheus-adapter-values.yaml`
- DCGM exporter for node/GPU visibility
- GPU nodepool quota or cluster autoscaler capacity for the HPA max replica count

```bash
# Deploy vLLM service and HPA
kubectl apply -f k8s/deployments/vllm-deployment.yaml

# Deploy GPU exporter
kubectl apply -f k8s/monitoring/dcgm-exporter.yaml

# Install Prometheus Adapter with the custom metric mappings
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring --create-namespace \
  -f k8s/autoscaling/prometheus-adapter-values.yaml
```

`k8s/monitoring/prometheus.yml` and `k8s/monitoring/alerts.yml` are Prometheus config files, not Kubernetes objects. Mount them through your Prometheus deployment or translate them into the equivalent Helm values for your monitoring stack.

The default Kubernetes manifest uses one GPU per vLLM pod and scales horizontally from 3 to 30 replicas. For tensor-parallel serving inside a single pod, set `TENSOR_PARALLEL_SIZE` and the pod `nvidia.com/gpu` request/limit to the same value, remove any cluster-specific one-GPU scheduling assumptions, and schedule onto nodes with enough local GPUs.

The HPA uses queue depth first (`llm_request_queue_size`), GPU saturation second (`llm_gpu_utilization_percent`), and CPU/memory as backup signals. For 10x traffic spikes, size `minReplicas` and `maxReplicas` from measured baseline capacity:

```text
required_gpus = ceil((baseline_peak_rps * 10 * avg_tokens_per_request) / measured_tokens_per_second_per_gpu)
```

Keep `minReplicas` high enough to absorb the first minute of traffic while GPU nodes and model pods warm up.

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
python -m src.api.gateway --host 0.0.0.0 --port 8080
```

### API Usage Example

```python
import requests

# Get authentication token
response = requests.post("http://localhost:8080/auth/token", json={
    "username": "user",
    "password": "pass"
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
- `server.max_concurrent_requests`: Requests admitted before queueing
- `monitoring.metrics_port`: Prometheus exporter port, default `9090`

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

Metrics endpoint: `http://localhost:9090/metrics`

Key metrics:
- `llm_requests_total`: Total number of requests
- `llm_request_duration_seconds`: Request latency
- `llm_active_requests`: Requests currently admitted for generation
- `llm_request_queue_size`: Requests waiting behind the admission limit
- `llm_tokens_processed_total`: Number of tokens processed
- `llm_gpu_utilization_percent`: GPU utilization

Autoscaling validation:
```bash
# Confirm HPA can read custom metrics
kubectl get --raw \
  "/apis/custom.metrics.k8s.io/v1beta1/namespaces/llm-inference/pods/*/llm_request_queue_size"

# Build a baseline, then ramp to 10x with the load test
python scripts/benchmark.py --endpoint http://localhost:8000
locust -f scripts/load_test.py --host http://localhost:8080
```

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
