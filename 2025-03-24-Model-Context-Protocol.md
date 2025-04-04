---
layout: post
title: Introduction to Model Context Protocol
description: The Model Context Protocol (MCP) is a standardized framework for managing, transmitting, and utilizing contextual information in machine learning systems. At its core, MCP defines how context—the set of relevant information surrounding a model's operation—should be captured, structured, passed to models, and used during inference.
date:   2025-03-24 01:42:44 -0500
---
# Building Your Own Model Context Protocol (MCP) Server: Comprehensive Guide

## 1. Introduction to MCP

### What is the Model Context Protocol (MCP)?

The Model Context Protocol (MCP) is a standardized framework for managing, transmitting, and utilizing contextual information in machine learning systems. At its core, MCP defines how context—the set of relevant information surrounding a model's operation—should be captured, structured, passed to models, and used during inference.

Unlike traditional ML deployment approaches where models operate as isolated black boxes, MCP creates an ecosystem where models are constantly aware of their operational environment, historical interactions, and user-specific requirements. This context-aware approach enables models to make more informed, personalized, and accurate predictions.

### The Importance of Context Management

Context management addresses a fundamental limitation in traditional ML deployments: the assumption that a model's input alone contains all information needed for an optimal response. In reality, several contextual factors affect how a model should perform:

- **Environmental context**: Information about the deployment environment, including time, location, system resources, and operational constraints
- **User context**: User preferences, history, demographics, interaction patterns, and specific requirements
- **Task context**: The broader goal the model is helping to achieve, including prior steps in a multi-step process
- **Data context**: Information about the data's source, quality, recency, and potential biases

By managing this context effectively, MCP allows models to:
- Personalize responses based on user history
- Adapt to environmental changes
- Maintain conversation coherence across multiple interactions
- Understand the intent behind ambiguous requests
- Follow evolving guidelines or constraints

### Benefits of MCP

#### Scalability
- **Horizontal Scaling**: MCP's standardized context format allows for seamless distribution of model workloads across multiple servers
- **Decoupled Architecture**: Context management can be scaled independently from model inference
- **Stateless Design**: Models can be spun up or down as needed without losing contextual information

#### Flexibility
- **Model Interchangeability**: Different models can access the same context data through a standardized interface
- **Progressive Enhancement**: New context attributes can be added without breaking existing functionality
- **Context Filtering**: Only relevant context is passed to each model, improving efficiency

#### Model Lifecycle Management
- **Version Control**: Context includes model version information, enabling graceful transitions between versions
- **Performance Monitoring**: Context tracking allows for detailed analysis of model behavior across different scenarios
- **Continuous Improvement**: Historical context enables targeted retraining based on actual usage patterns

## 2. Prerequisites for Building Your Own MCP Server

### Hardware Requirements

#### Compute Resources
- **CPU**: Minimum 8 cores (16+ recommended for production), preferably server-grade processors like Intel Xeon or AMD EPYC
- **GPU**: For transformer-based models, NVIDIA GPUs with at least 16GB VRAM (A100, V100, or RTX 3090/4090); multiple GPUs recommended for high workloads
- **Memory**: 32GB RAM minimum (64-128GB recommended for production)
- **Storage**:
  - 500GB+ SSD for OS and applications (NVMe preferred)
  - 1TB+ storage for model artifacts and context data (scalable based on expected usage)
  - High IOPS capability for context retrieval operations

#### Networking
- **Bandwidth**: 10Gbps+ network interfaces for high-throughput model serving
- **Latency**: Low-latency connections, especially if context data is stored separately from models

### Software Requirements

#### Operating System
- **Linux Distributions**: Ubuntu 20.04/22.04 LTS or CentOS 8/9 (preferred for ML workloads)
- **Windows**: Windows Server 2019/2022 (if required by organizational constraints)

#### Containerization
- **Docker**: Engine 20.10+ for containerizing individual components
- **Kubernetes**: v1.24+ for orchestrating multi-container deployments
- **Helm**: For managing Kubernetes applications

#### Model Management
- **TensorFlow Serving**: For TensorFlow models
- **TorchServe**: For PyTorch models
- **Triton Inference Server**: For multi-framework model serving
- **MLflow**: For model lifecycle management
- **KServe/Seldon Core**: For Kubernetes-native model serving

#### Database Systems
- **Vector Database**: ChromaDB, Pinecone, or Milvus for storing and retrieving embeddings
- **Relational Database**: PostgreSQL 14+ for structured context data and metadata
- **Redis**: For high-speed context caching and session management
- **MongoDB**: For schema-flexible context storage

#### Networking and APIs
- **REST Framework**: FastAPI or Flask for creating REST endpoints
- **gRPC**: For high-performance internal communication
- **Envoy/Istio**: For API gateway and service mesh capabilities
- **Protocol Buffers**: For efficient data serialization

#### Monitoring and Logging
- **Prometheus**: For metrics collection
- **Grafana**: For metrics visualization
- **Elasticsearch, Logstash, Kibana (ELK)**: For comprehensive logging
- **Jaeger/Zipkin**: For distributed tracing

## 3. Installation and Setup

### Operating System Setup

```bash
# Example for Ubuntu Server 22.04 LTS
# 1. Download Ubuntu Server ISO from ubuntu.com
# 2. Create bootable USB and install Ubuntu Server
# 3. Update system packages
sudo apt update && sudo apt upgrade -y

# 4. Install basic utilities
sudo apt install -y build-essential curl wget git software-properties-common
```

### Docker Installation

```bash
# Install Docker on Ubuntu
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Add current user to docker group
sudo usermod -aG docker $USER

# Verify installation
newgrp docker
docker --version
```

### Kubernetes Setup

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install minikube for local development
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start minikube
minikube start --driver=docker --memory=8g --cpus=4

# For production, consider using kubeadm or managed Kubernetes services
```

### GPU Support

```bash
# Install NVIDIA drivers
sudo apt install -y nvidia-driver-535  # Choose appropriate version

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU is accessible to Docker
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Database Setup

```bash
# PostgreSQL for structured context data
sudo apt install -y postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database for MCP
sudo -u postgres psql -c "CREATE DATABASE mcp_context;"
sudo -u postgres psql -c "CREATE USER mcp_user WITH ENCRYPTED PASSWORD 'your_secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE mcp_context TO mcp_user;"

# Redis for caching
sudo apt install -y redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# ChromaDB (Vector Database) using Docker
docker run -d -p 8000:8000 --name chromadb chromadb/chroma
```

### Setting Up Model Serving Infrastructure

```bash
# TensorFlow Serving with Docker
docker pull tensorflow/serving:latest

# TorchServe
git clone https://github.com/pytorch/serve.git
cd serve
docker build -t torchserve:latest .

# Triton Inference Server
docker pull nvcr.io/nvidia/tritonserver:22.12-py3

# MLflow
pip install mlflow
mlflow server --host 0.0.0.0 --port 5000
```

## 4. Configuring the Model Context Protocol

### Defining Context Parameters

The MCP server needs to track various context parameters. Create a schema that includes:

```python
# Example context schema (context_schema.py)
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

class UserContext(BaseModel):
    user_id: str
    preferences: Dict[str, Any] = {}
    session_history: List[str] = []
    demographics: Optional[Dict[str, Any]] = None
    
class EnvironmentContext(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    deployment_environment: str = "production"  # or "staging", "development"
    server_load: float = 0.0
    available_resources: Dict[str, float] = {}
    
class ModelContext(BaseModel):
    model_id: str
    model_version: str
    parameters: Dict[str, Any] = {}
    constraints: Dict[str, Any] = {}
    
class DataContext(BaseModel):
    data_source: str = "unknown"
    data_timestamp: Optional[datetime] = None
    data_quality_metrics: Dict[str, float] = {}
    
class MCPContext(BaseModel):
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user: UserContext
    environment: EnvironmentContext = Field(default_factory=EnvironmentContext)
    model: ModelContext
    data: DataContext = Field(default_factory=DataContext)
    custom_attributes: Dict[str, Any] = {}
```

### API Integration

Create a FastAPI server to handle MCP context:

```python
# app.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any

from context_schema import MCPContext
from database import SessionLocal, engine, Base
import context_store
import model_manager

# Initialize database
Base.metadata.create_all(bind=engine)

app = FastAPI(title="MCP Server")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/v1/context", response_model=Dict[str, Any])
def create_context(context: MCPContext, db: Session = Depends(get_db)):
    """Create a new context record"""
    context_id = context_store.save_context(db, context)
    return {"context_id": context_id, "status": "created"}

@app.get("/api/v1/context/{context_id}")
def get_context(context_id: str, db: Session = Depends(get_db)):
    """Retrieve a specific context by ID"""
    context = context_store.get_context(db, context_id)
    if not context:
        raise HTTPException(status_code=404, detail="Context not found")
    return context

@app.post("/api/v1/inference/{model_id}")
async def model_inference(
    model_id: str,
    input_data: Dict[str, Any],
    context_id: str = None,
    db: Session = Depends(get_db)
):
    """Run model inference with context"""
    context = None
    if context_id:
        context = context_store.get_context(db, context_id)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")
    
    # Get model and run inference
    result = await model_manager.run_inference(model_id, input_data, context)
    
    # Update context with this interaction if needed
    if context_id:
        context_store.update_context_after_inference(db, context_id, input_data, result)
    
    return result

@app.put("/api/v1/context/{context_id}")
def update_context(
    context_id: str,
    updates: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Update specific fields in the context"""
    success = context_store.update_context(db, context_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Context not found")
    return {"status": "updated"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Context Handling Implementation

Create the context store:

```python
# context_store.py
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, JSON, DateTime
import json
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

from database import Base

class ContextRecord(Base):
    __tablename__ = "contexts"
    
    id = Column(String, primary_key=True, index=True)
    user_context = Column(JSON)
    environment_context = Column(JSON)
    model_context = Column(JSON) 
    data_context = Column(JSON)
    custom_attributes = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def save_context(db: Session, context_data) -> str:
    """Save context to database"""
    context_dict = context_data.dict()
    context_id = context_dict.get("context_id", str(uuid.uuid4()))
    
    db_context = ContextRecord(
        id=context_id,
        user_context=context_dict.get("user"),
        environment_context=context_dict.get("environment"),
        model_context=context_dict.get("model"),
        data_context=context_dict.get("data"),
        custom_attributes=context_dict.get("custom_attributes", {})
    )
    
    db.add(db_context)
    db.commit()
    db.refresh(db_context)
    return context_id

def get_context(db: Session, context_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve context from database"""
    db_context = db.query(ContextRecord).filter(ContextRecord.id == context_id).first()
    if not db_context:
        return None
        
    return {
        "context_id": db_context.id,
        "user": db_context.user_context,
        "environment": db_context.environment_context,
        "model": db_context.model_context,
        "data": db_context.data_context,
        "custom_attributes": db_context.custom_attributes,
        "created_at": db_context.created_at,
        "updated_at": db_context.updated_at
    }

def update_context(db: Session, context_id: str, updates: Dict[str, Any]) -> bool:
    """Update specific fields in the context"""
    db_context = db.query(ContextRecord).filter(ContextRecord.id == context_id).first()
    if not db_context:
        return False
    
    # Update appropriate fields based on the structure
    for key, value in updates.items():
        if key == "user":
            db_context.user_context = value
        elif key == "environment":
            db_context.environment_context = value
        elif key == "model":
            db_context.model_context = value
        elif key == "data":
            db_context.data_context = value
        elif key == "custom_attributes":
            db_context.custom_attributes = value
    
    db_context.updated_at = datetime.utcnow()
    db.commit()
    return True

def update_context_after_inference(
    db: Session, 
    context_id: str, 
    input_data: Dict[str, Any],
    result: Dict[str, Any]
) -> bool:
    """Update context after model inference"""
    db_context = db.query(ContextRecord).filter(ContextRecord.id == context_id).first()
    if not db_context:
        return False
    
    # Add this interaction to user history
    user_context = db_context.user_context
    if "session_history" not in user_context:
        user_context["session_history"] = []
    
    # Add interaction record
    user_context["session_history"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "output": result
    })
    
    # Limit history size
    if len(user_context["session_history"]) > 100:  # Example limit
        user_context["session_history"] = user_context["session_history"][-100:]
    
    db_context.user_context = user_context
    db_context.updated_at = datetime.utcnow()
    db.commit()
    return True
```

### Model Manager Implementation

```python
# model_manager.py
import os
import json
import asyncio
import httpx
from typing import Dict, Any, Optional
import numpy as np
import tensorflow as tf
import torch
import redis

# Redis client for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Model registry - in production this would be a database
MODEL_REGISTRY = {
    "gpt-model": {
        "type": "http",
        "endpoint": "http://localhost:8001/v1/models/gpt:predict",
        "version": "1.0.0"
    },
    "bert-embedding": {
        "type": "tensorflow",
        "path": "/models/bert",
        "version": "2.1.0"
    },
    "image-classifier": {
        "type": "pytorch",
        "path": "/models/image_classifier.pt",
        "version": "1.2.0"
    }
}

async def run_inference(model_id: str, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
    """Run model inference with context awareness"""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_id} not found in registry")
    
    model_info = MODEL_REGISTRY[model_id]
    
    # Prepare input with context
    inference_input = prepare_input_with_context(model_id, input_data, context)
    
    # Check cache for identical request if appropriate
    cache_key = None
    if model_info.get("cacheable", False):
        cache_key = f"{model_id}:{hash(json.dumps(inference_input, sort_keys=True))}"
        cached_result = redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
    
    # Run model based on type
    if model_info["type"] == "http":
        result = await http_inference(model_info["endpoint"], inference_input)
    elif model_info["type"] == "tensorflow":
        result = tf_inference(model_info["path"], inference_input)
    elif model_info["type"] == "pytorch":
        result = pytorch_inference(model_info["path"], inference_input)
    else:
        raise ValueError(f"Unsupported model type: {model_info['type']}")
    
    # Store in cache if appropriate
    if cache_key:
        redis_client.setex(
            cache_key,
            model_info.get("cache_ttl", 3600),  # Default 1 hour TTL
            json.dumps(result)
        )
    
    return result

def prepare_input_with_context(model_id: str, input_data: Dict[str, Any], context: Optional[Dict[str, Any]]):
    """Prepare model input with relevant context information"""
    if not context:
        return input_data
    
    # Deep copy to avoid modifying original
    enhanced_input = input_data.copy()
    
    # Add context based on model requirements
    if model_id == "gpt-model":
        # For a GPT-like model, we might include conversation history
        if "user" in context and "session_history" in context["user"]:
            # Format history appropriately for the model
            history = context["user"]["session_history"][-5:]  # Last 5 interactions
            enhanced_input["conversation_history"] = history
            
        # Add user preferences if available
        if "user" in context and "preferences" in context["user"]:
            enhanced_input["user_preferences"] = context["user"]["preferences"]
    
    elif model_id == "bert-embedding":
        # For embeddings, maybe we add language preference
        if "user" in context and "preferences" in context["user"]:
            enhanced_input["language"] = context["user"]["preferences"].get("language", "en")
    
    # Add model-specific parameters from context
    if "model" in context and "parameters" in context["model"]:
        enhanced_input["parameters"] = context["model"]["parameters"]
    
    # Add environmental context if relevant
    if "environment" in context:
        enhanced_input["environment"] = {
            "timestamp": context["environment"].get("timestamp"),
            "deployment": context["environment"].get("deployment_environment")
        }
    
    return enhanced_input

async def http_inference(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Call a model exposed via HTTP endpoint"""
    async with httpx.AsyncClient() as client:
        response = await client.post(endpoint, json=data)
        response.raise_for_status()
        return response.json()

def tf_inference(model_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on a TensorFlow model"""
    # Load model (in production, this would be cached)
    model = tf.saved_model.load(model_path)
    
    # Prepare tensors
    input_tensors = {}
    for key, value in data.items():
        if key != "parameters" and key != "environment":
            if isinstance(value, list):
                input_tensors[key] = tf.constant(value)
            else:
                input_tensors[key] = tf.constant([value])
    
    # Run inference
    results = model.signatures["serving_default"](**input_tensors)
    
    # Convert results to Python types
    output = {}
    for key, tensor in results.items():
        output[key] = tensor.numpy().tolist()
    
    return output

def pytorch_inference(model_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on a PyTorch model"""
    # Load model (in production, this would be cached)
    model = torch.load(model_path)
    model.eval()
    
    # Prepare tensors
    input_tensors = {}
    for key, value in data.items():
        if key != "parameters" and key != "environment":
            if isinstance(value, list):
                input_tensors[key] = torch.tensor(value)
            else:
                input_tensors[key] = torch.tensor([value])
    
    # Run inference
    with torch.no_grad():
        results = model(**input_tensors)
    
    # Convert results to Python types
    if isinstance(results, tuple):
        output = {}
        for i, result in enumerate(results):
            output[f"output_{i}"] = result.numpy().tolist()
    else:
        output = {"output": results.numpy().tolist()}
    
    return output
```

### Contextual Adaptation

Implement a context adapter that modifies model behavior based on context:

```python
# context_adapter.py
from typing import Dict, Any, List, Optional

class ContextAdapter:
    """Adapts model behavior based on context"""
    
    @staticmethod
    def adapt_model_parameters(model_id: str, default_params: Dict[str, Any], 
                               context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Modify model parameters based on context"""
        if not context:
            return default_params
        
        params = default_params.copy()
        
        # User-specific adaptations
        if "user" in context:
            user = context["user"]
            
            # Adapt language model temperature based on user preference
            if model_id.startswith("gpt") and "preferences" in user:
                if "creativity" in user["preferences"]:
                    creativity = user["preferences"]["creativity"]
                    # Map creativity preference to temperature
                    if creativity == "high":
                        params["temperature"] = max(params.get("temperature", 0.7), 0.9)
                    elif creativity == "low":
                        params["temperature"] = min(params.get("temperature", 0.7), 0.3)
            
            # Adapt response length based on user preference
            if "preferences" in user and "verbosity" in user["preferences"]:
                verbosity = user["preferences"]["verbosity"]
                if verbosity == "concise":
                    params["max_tokens"] = min(params.get("max_tokens", 1024), 256)
                elif verbosity == "detailed":
                    params["max_tokens"] = max(params.get("max_tokens", 1024), 1024)
        
        # Environment-specific adaptations
        if "environment" in context:
            env = context["environment"]
            
            # Reduce complexity under high server load
            if "server_load" in env and env["server_load"] > 0.8:
                params["max_tokens"] = min(params.get("max_tokens", 1024), 512)
                if "top_k" in params:
                    params["top_k"] = min(params["top_k"], 10)
            
            # Adapt based on deployment environment
            if "deployment_environment" in env:
                if env["deployment_environment"] == "development":
                    # More logging in development
                    params["verbose"] = True
                elif env["deployment_environment"] == "production":
                    # Safer settings in production
                    params["safety_filter"] = True
        
        # Data-specific adaptations
        if "data" in context and "data_quality_metrics" in context["data"]:
            quality = context["data"]["data_quality_metrics"]
            
            # If input data quality is low, be more conservative
            if "noise_level" in quality and quality["noise_level"] > 0.6:
                params["temperature"] = min(params.get("temperature", 0.7), 0.4)
                if "top_p" in params:
                    params["top_p"] = min(params["top_p"], 0.92)
        
        # Model-specific adaptations from context
        if "model" in context and "parameters" in context["model"]:
            # Explicit parameter overrides from context
            for k, v in context["model"]["parameters"].items():
                params[k] = v
        
        return params
    
    @staticmethod
    def adapt_response(model_id: str, response: Any, 
                       context: Optional[Dict[str, Any]]) -> Any:
        """Post-process model response based on context"""
        if not context:
            return response
        
        # For text responses
        if isinstance(response, str) or (isinstance(response, dict) and "text" in response):
            text = response if isinstance(response, str) else response["text"]
            
            # Apply user language preference
            if "user" in context and "preferences" in context["user"]:
                prefs = context["user"]["preferences"]
                if "language" in prefs and prefs["language"] != "en":
                    # In a real system, this would call a translation service
                    pass
                
                # Apply formality preference
                if "formality" in prefs:
                    if prefs["formality"] == "formal" and not text.startswith("Dear"):
                        text = "I would like to inform you that " + text
                    elif prefs["formality"] == "casual" and text.startswith("Dear"):
                        text = text.replace("Dear", "Hey").replace("Sincerely", "Cheers")
            
            # Format response appropriately
            if isinstance(response, dict):
                response["text"] = text
            else:
                response = text
        
        return response
```

## 5. Testing and Validation

### Unit Tests

Create a test suite to validate the MCP server:

```python
# test_mcp_server.py
import unittest
import json
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import app, get_db
from database import Base
from context_schema import MCPContext, UserContext, ModelContext

# Create in-memory database for testing
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

class TestMCPServer(unittest.TestCase):
    def test_create_context(self):
        """Test creating a new context"""
        context_data = {
            "user": {
                "user_id": "test_user_1",
                "preferences": {"language": "en"}
            },
            "model": {
                "model_id": "gpt-model",
                "model_version": "1.0.0"
            }
        }
        
        response = client.post("/api/v1/context", json=context_data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("context_id", result)
        self.assertEqual(result["status"], "created")
        
        # Verify we can retrieve it
        context_id = result["context_id"]
        get_response = client.get(f"/api/v1/context/{context_id}")
        self.assertEqual(get_response.status_code, 200)
        context = get_response.json()
        self.assertEqual(context["user"]["user_id"], "test_user_1")
    
    def test_update_context(self):
        """Test updating an existing context"""
        # First create a context
        context_data = {
            "user": {
                "user_id": "test_user_2",
                "preferences": {"language": "en"}
            },
            "model": {
                "model_id": "gpt-model",
                "model_version": "1.0.0"
            }
        }
        
        response = client.post("/api/v1/context", json=context_data)
        context_id = response.json()["context_id"]
        
        # Now update it
        update_data = {
            "user": {
                "user_id": "test_user_2",
                "preferences": {"language": "fr", "formality": "formal"}
            }
        }
        
        update_response = client.put(f"/api/v1/context/{context_id}", json=update_data)
        self.assertEqual(update_response.status_code, 200)
        
        # Verify the update
        get_response = client.get(f"/api/v1/context/{context_id}")
        context = get_response.json()
        self.assertEqual(context["user"]["preferences"]["language"], "fr")
        self.assertEqual(context["user"]["preferences"]["formality"], "formal")
    
    def test_model_inference_with_context(self):
        """Test model inference with context (mocked)"""
        # This would be a more complex mock in a real test
        # For now, we'll just verify the API structure
        
        # First create a context
        context_data = {
            "user": {
                "user_id": "test_user_3",
                "preferences": {"language": "en", "creativity": "high"}
            },
            "model": {
                "model_id": "gpt-model",
                "model_version": "1.0.0",
                "parameters": {"max_tokens": 100}
            }
        }
        
        response = client.post("/api/v1/context", json=context_data)
        context_id = response.json()["context_id"]
        
        # Mock inference request
        inference_data = {
            "prompt": "Tell me a story about a dragon"
        }
        
        # This will fail in the test environment without mocking
        # But we can test the API structure
        try:
            inference_response = client.post(
                f"/api/v1/inference/gpt-model?context_id={context_id}",
                json=inference_data
            )
        except Exception as e:
            # Expected to fail without proper mock
            pass

if __name__ == "__main__":
    unittest.main()
```

### Integration Tests

Create a script to test the entire system:

```python
# integration_test.py
import asyncio
import httpx
import json
import time
from typing import Dict, Any

async def test_full_workflow():
    """Test the entire MCP workflow from context creation to inference"""
    async with httpx.AsyncClient() as client:
        print("Testing MCP Server Integration...")
        
        # 1. Create a context
        context_data = {
            "user": {
                "user_id": "integration_test_user",
                "preferences": {
                    "language": "en",
                    "creativity": "high",
                    "verbosity": "concise"
                }
            },
            "environment": {
                "deployment_environment": "test",
                "server_load": 0.2
            },
            "model": {
                "model_id": "gpt-model",
                "model_version": "1.0.0",
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            },
            "data": {
                "data_source": "integration_test",
                "data_quality_metrics": {
                    "noise_level": 0.1
                }
            }
        }
        
        print("Step 1: Creating context...")
        response = await client.post(
            "http://localhost:8080/api/v1/context",
            json=context_data
        )
        
        assert response.status_code == 200, f"Failed to create context: {response.text}"
        result = response.json()
        context_id = result["context_id"]
        print(f"Context created with ID: {context_id}")
        
        # 2. Retrieve the context to verify
        print("Step 2: Retrieving context...")
        get_response = await client.get(f"http://localhost:8080/api/v1/context/{context_id}")
        assert get_response.status_code == 200, f"Failed to retrieve context: {get_response.text}"
        context = get_response.json()
        assert context["user"]["user_id"] == "integration_test_user"
        print("Context retrieved successfully")
        
        # 3. Update the context with new preferences
        print("Step 3: Updating context...")
        update_data = {
            "user": {
                "user_id": "integration_test_user",
                "preferences": {
                    "language": "en",
                    "creativity": "low",  # Changed from high to low
                    "verbosity": "concise"
                }
            }
        }
        
        update_response = await client.put(
            f"http://localhost:8080/api/v1/context/{context_id}",
            json=update_data
        )
        
        assert update_response.status_code == 200, f"Failed to update context: {update_response.text}"
        print("Context updated successfully")
        
        # 4. Run model inference with context
        print("Step 4: Running model inference with context...")
        inference_data = {
            "prompt": "Write a short poem about artificial intelligence"
        }
        
        inference_response = await client.post(
            f"http://localhost:8080/api/v1/inference/gpt-model?context_id={context_id}",
            json=inference_data
        )
        
        # This might fail in a test environment without actual models
        # For a real test, check the response structure
        if inference_response.status_code == 200:
            result = inference_response.json()
            print(f"Model response: {result}")
        else:
            print(f"Inference failed (expected in test environment): {inference_response.text}")
        
        # 5. Verify context was updated after inference
        print("Step 5: Verifying context update after inference...")
        final_get_response = await client.get(f"http://localhost:8080/api/v1/context/{context_id}")
        final_context = final_get_response.json()
        
        # Check if session history was updated
        if "session_history" in final_context["user"]:
            history = final_context["user"]["session_history"]
            if history:
                print(f"Session history updated: {len(history)} interactions recorded")
            else:
                print("Session history exists but no interactions recorded")
        else:
            print("No session history found in context")
        
        print("Integration test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_full_workflow())
```

### Performance Testing

Create a script to test performance:

```python
# performance_test.py
import asyncio
import httpx
import time
import random
import statistics
from typing import List, Dict, Any
import uuid

async def create_context(client: httpx.AsyncClient, user_id: str) -> str:
    """Create a context and return its ID"""
    context_data = {
        "user": {
            "user_id": user_id,
            "preferences": {
                "language": "en",
                "creativity": random.choice(["high", "medium", "low"])
            }
        },
        "model": {
            "model_id": "gpt-model",
            "model_version": "1.0.0"
        }
    }
    
    response = await client.post("http://localhost:8080/api/v1/context", json=context_data)
    if response.status_code != 200:
        raise Exception(f"Failed to create context: {response.text}")
    
    return response.json()["context_id"]

async def run_inference(client: httpx.AsyncClient, context_id: str, prompt: str) -> float:
    """Run inference and return time taken"""
    inference_data = {"prompt": prompt}
    
    start_time = time.time()
    response = await client.post(
        f"http://localhost:8080/api/v1/inference/gpt-model?context_id={context_id}",
        json=inference_data
    )
    end_time = time.time()
    
    if response.status_code != 200:
        print(f"Inference failed: {response.text}")
    
    return end_time - start_time

async def performance_test(num_users: int, requests_per_user: int):
    """Run performance test with multiple simulated users"""
    async with httpx.AsyncClient() as client:
        print(f"Starting performance test with {num_users} users, {requests_per_user} requests each")
        
        # Create contexts for all users
        print("Creating contexts...")
        context_ids = []
        for i in range(num_users):
            user_id = f"perf_test_user_{uuid.uuid4()}"
            context_id = await create_context(client, user_id)
            context_ids.append(context_id)
        
        # Test prompts
        prompts = [
            "Tell me a story about a robot",
            "Explain quantum computing",
            "Write a poem about the ocean",
            "Give me a recipe for chocolate cake",
            "Describe the solar system"
        ]
        
        # Run inference requests concurrently
        print("Running inference requests...")
        tasks = []
        for i in range(num_users):
            for j in range(requests_per_user):
                prompt = random.choice(prompts)
                tasks.append(run_inference(client, context_ids[i], prompt))
        
        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate statistics
        successful_times = [t for t in results if isinstance(t, float)]
        errors = [e for e in results if isinstance(e, Exception)]
        
        if successful_times:
            avg_time = statistics.mean(successful_times)
            min_time = min(successful_times)
            max_time = max(successful_times)
            p95_time = sorted(successful_times)[int(len(successful_times) * 0.95)]
            
            print(f"Performance results:")
            print(f"  Total requests: {len(results)}")
            print(f"  Successful: {len(successful_times)}")
            print(f"  Failed: {len(errors)}")
            print(f"  Average response time: {avg_time:.4f}s")
            print(f"  Min response time: {min_time:.4f}s")
            print(f"  Max response time: {max_time:.4f}s")
            print(f"  95th percentile: {p95_time:.4f}s")
            print(f"  Requests per second: {len(successful_times) / sum(successful_times):.2f}")
        else:
            print("No successful requests to analyze")

if __name__ == "__main__":
    asyncio.run(performance_test(num_users=10, requests_per_user=5))
```

## 6. Scaling and Optimization

### Scaling with Kubernetes

Create Kubernetes deployment files for your MCP server:

```yaml
# kubernetes/mcp-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
  labels:
    app: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: mcp-server:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: database-url
        - name: REDIS_HOST
          value: "redis-service"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server-service
spec:
  selector:
    app: mcp-server
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-server
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Database Scaling

```yaml
# kubernetes/database-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: "postgres"
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:14
        ports:
        - containerPort: 5432
          name: postgres
        env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: postgres-password
        - name: POSTGRES_USER
          value: mcp_user
        - name: POSTGRES_DB
          value: mcp_context
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  clusterIP: None
```

### Redis for Caching

```yaml
# kubernetes/redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:6.2-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

### Optimizing Performance

Implement a request throttler:

```python
# throttler.py
import time
import asyncio
from typing import Dict, Any, Callable, Awaitable
import redis

class RequestThrottler:
    """Throttles requests to manage load"""
    
    def __init__(self, redis_client: redis.Redis, limits: Dict[str, int]):
        """
        Initialize with Redis client and limits dict
        
        limits: Dict mapping model IDs to requests per minute
        """
        self.redis = redis_client
        self.limits = limits
        self.default_limit = 60  # Default to 60 RPM
    
    async def throttle(self, model_id: str) -> bool:
        """
        Check if request should be throttled
        
        Returns:
            True if request is allowed, False if it should be throttled
        """
        limit = self.limits.get(model_id, self.default_limit)
        key = f"throttle:{model_id}:{int(time.time() / 60)}"  # Key expires each minute
        
        # Increment counter for this minute
        current = self.redis.incr(key)
        
        # Set expiry to ensure cleanup
        if current == 1:
            self.redis.expire(key, 120)  # 2 minutes (to handle edge cases)
        
        # Check if under limit
        return current <= limit
    
    async def with_throttling(
        self,
        model_id: str,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Execute function with throttling
        
        Raises:
            Exception if throttled
        """
        if await self.throttle(model_id):
            return await func(*args, **kwargs)
        else:
            # In a real implementation, you might queue the request instead
            raise Exception(f"Request throttled for model {model_id}")
```

Add connection pooling for database:

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Get DB URL from environment or use default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://mcp_user:password@localhost/mcp_context")

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=20,             # Maximum number of connections
    max_overflow=10,          # Allow 10 connections beyond pool_size
    pool_timeout=30,          # Wait up to 30 seconds for a connection
    pool_recycle=1800,        # Recycle connections after 30 minutes
    pool_pre_ping=True        # Check connection validity before using
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
```

## 7. Security and Maintenance

### Authentication and Authorization

Implement JWT authentication:

```python
# auth.py
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import os

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-for-development")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    role: str = "user"  # "user", "admin", "model_developer"

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# User database - in production, this would be in a real database
USERS_DB = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": pwd_context.hash("secret"),
        "disabled": False,
        "role": "user"
    },
    "alice": {
        "username": "alice",
        "full_name": "Alice Smith",
        "email": "alice@example.com",
        "hashed_password": pwd_context.hash("password"),
        "disabled": False,
        "role": "admin"
    }
}

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(USERS_DB, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def has_role(required_role: str):
    async def role_checker(current_user: User = Depends(get_current_active_user)):
        if current_user.role != required_role and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires role: {required_role}"
            )
        return current_user
    return role_checker
```

Update your app to use authentication:

```python
# Update app.py to include authentication

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from typing import Dict, Any

from auth import (
    User, Token, authenticate_user, create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES, get_current_active_user, has_role, USERS_DB
)

# ... other imports as before

app = FastAPI(title="MCP Server")

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(USERS_DB, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

# Now protect your API endpoints with authentication
@app.post("/api/v1/context", response_model=Dict[str, Any])
def create_context(
    context: MCPContext, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new context record (requires authentication)"""
    context_id = context_store.save_context(db, context)
    return {"context_id": context_id, "status": "created"}

# Admin-only endpoint
@app.delete("/api/v1/context/{context_id}")
def delete_context(
    context_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(has_role("admin"))
):
    """Delete a context (admin only)"""
    success = context_store.delete_context(db, context_id)
    if not success:
        raise HTTPException(status_code=404, detail="Context not found")
    return {"status": "deleted"}

# ... other endpoints
```

### Security Best Practices

Implement data encryption for sensitive context data:

```python
# encryption.py
from cryptography.fernet import Fernet
import os
import json
from typing import Dict, Any, Optional

class ContextEncryption:
    """Handles encryption of sensitive context data"""
    
    def __init__(self, key_path: Optional[str] = None):
        """Initialize with encryption key"""
        if key_path and os.path.exists(key_path):
            with open(key_path, "rb") as key_file:
                self.key = key_file.read()
        else:
            # Generate a key and save it
            self.key = Fernet.generate_key()
            if key_path:
                os.makedirs(os.path.dirname(key_path), exist_ok=True)
                with open(key_path, "wb") as key_file:
                    key_file.write(self.key)
        
        self.cipher = Fernet(self.key)
    
    def encrypt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive parts of the context"""
        # Create a deep copy to avoid modifying the original
        encrypted_context = context.copy()
        
        # Encrypt user data if present
        if "user" in encrypted_context:
            user_data = encrypted_context["user"]
            
            # Encrypt user preferences
            if "preferences" in user_data:
                user_data["preferences"] = self._encrypt_data(user_data["preferences"])
            
            # Encrypt demographics
            if "demographics" in user_data:
                user_data["demographics"] = self._encrypt_data(user_data["demographics"])
        
        # Encrypt custom attributes
        if "custom_attributes" in encrypted_context:
            encrypted_context["custom_attributes"] = self._encrypt_data(
                encrypted_context["custom_attributes"]
            )
        
        return encrypted_context
    
    def decrypt_context(self, encrypted_context: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt encrypted parts of the context"""
        # Create a deep copy to avoid modifying the original
        decrypted_context = encrypted_context.copy()
        
        # Decrypt user data if present
        if "user" in decrypted_context:
            user_data = decrypted_context["user"]
            
            # Decrypt user preferences
            if "preferences" in user_data and isinstance(user_data["preferences"], str):
                user_data["preferences"] = self._decrypt_data(user_data["preferences"])
            
            # Decrypt demographics
            if "demographics" in user_data and isinstance(user_data["demographics"], str):
                user_data["demographics"] = self._decrypt_data(user_data["demographics"])
        
        # Decrypt custom attributes
        if "custom_attributes" in decrypted_context and isinstance(decrypted_context["custom_attributes"], str):
            decrypted_context["custom_attributes"] = self._decrypt_data(
                decrypted_context["custom_attributes"]
            )
        
        return decrypted_context
    
    def _encrypt_data(self, data: Any) -> str:
        """Encrypt any data by converting to JSON and encrypting"""
        json_data = json.dumps(data)
        encrypted_bytes = self.cipher.encrypt(json_data.encode('utf-8'))
        return encrypted_bytes.decode('utf-8')
    
    def _decrypt_data(self, encrypted_str: str) -> Any:
        """Decrypt data and convert from JSON"""
        decrypted_bytes = self.cipher.decrypt(encrypted_str.encode('utf-8'))
        return json.loads(decrypted_bytes.decode('utf-8'))
```

### Monitoring and Logging

```python
# monitoring.py
import logging
import time
from functools import wraps
from typing import Dict, Any, Callable, Optional
import json
import prometheus_client as prom
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_server")

# Prometheus metrics
MODEL_REQUESTS = Counter(
    'model_requests_total', 
    'Total model inference requests',
    ['model_id', 'status']
)

CONTEXT_OPERATIONS = Counter(
    'context_operations_total',
    'Context CRUD operations',
    ['operation']
)

RESPONSE_TIME = Histogram(
    'response_time_seconds',
    'Response time in seconds',
    ['endpoint', 'method'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of active requests',
    ['endpoint']
)

def log_context_operation(operation: str, context_id: str, data: Optional[Dict[str, Any]] = None):
    """Log context operations with standardized format"""
    log_data = {
        "operation": operation,
        "context_id": context_id,
        "timestamp": time.time()
    }
    
    if data:
        # Exclude sensitive data from logging
        sanitized_data = data.copy()
        if "user" in sanitized_data and "preferences" in sanitized_data["user"]:
            sanitized_data["user"]["preferences"] = "[REDACTED]"
        if "user" in sanitized_data and "demographics" in sanitized_data["user"]:
            sanitized_data["user"]["demographics"] = "[REDACTED]"
        
        log_data["data"] = sanitized_data
    
    logger.info(f"Context {operation}: {json.dumps(log_data)}")
    CONTEXT_OPERATIONS.labels(operation=operation).inc()

def log_model_request(model_id: str, status: str, context_id: Optional[str] = None, 
                       input_data: Optional[Dict[str, Any]] = None, 
                       response: Optional[Dict[str, Any]] = None,
                       error: Optional[str] = None):
    """Log model inference requests"""
    log_data = {
        "model_id": model_id,
        "status": status,
        "timestamp": time.time()
    }
    
    if context_id:
        log_data["context_id"] = context_id
    
    if input_data:
        # Sanitize input data for logging
        log_data["input"] = "[DATA]"
    
    if response and status == "success":
        # Only log response structure, not content
        log_data["response_type"] = type(response).__name__
    
    if error:
        log_data["error"] = error
    
    logger.info(f"Model request: {json.dumps(log_data)}")
    MODEL_REQUESTS.labels(model_id=model_id, status=status).inc()

def timer_decorator(endpoint: str, method: str):
    """Decorator to time and log endpoint execution"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()
            try:
                result = await func(*args, **kwargs)
                status = "success"
            except Exception as e:
                logger.error(f"Error in {endpoint}: {str(e)}")
                status = "error"
                raise
            finally:
                ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()
                
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Record metrics
            RESPONSE_TIME.labels(endpoint=endpoint, method=method).observe(execution_time)
            
            # Log request
            logger.info(f"{method} {endpoint} - {status} - {execution_time:.4f}s")
            
            return result
        return wrapper
    return decorator
```

### System Health Checks

```python
# health.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text
import redis
import psutil
import os
from typing import Dict, Any

from database import get_db

health_router = APIRouter()

@health_router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "ok"}

@health_router.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """Readiness check - ensures database connection works"""
    try:
        # Execute a simple query to check if database is responsive
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    try:
        # Check Redis connection
        r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"))
        r.ping()
        redis_status = "ok"
    except Exception as e:
        redis_status = f"error: {str(e)}"
    
    # Get system resources
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    system_status = {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "disk_percent": disk.percent
    }
    
    # Determine overall status
    if db_status == "ok" and redis_status == "ok":
        status = "ready"
    else:
        status = "not ready"
    
    return {
        "status": status,
        "database": db_status,
        "redis": redis_status,
        "system": system_status
    }

@health_router.get("/metrics")
async def metrics():
    """Endpoint for Prometheus metrics"""
    from prometheus_client import generate_latest
    
    # Generate metrics in Prometheus format
    metrics_data = generate_latest()
    return metrics_data
```

## 8. Best Practices and Pitfalls to Avoid

### Best Practices for MCP Server Deployment

#### Context Management
1. **Structured Context Schema**: Always use a well-defined, versioned schema for context data. This ensures compatibility across models and prevents unexpected behavior.

2. **Context Lifetime Management**: Implement policies to expire context data after a certain period of inactivity to prevent context bloat:

```python
# Example context cleanup job (scheduled job)
async def cleanup_stale_contexts():
    """Remove contexts that haven't been used for X days"""
    cutoff_date = datetime.utcnow() - timedelta(days=30)
    
    async with async_session() as session:
        # Find and delete old contexts
        result = await session.execute(
            delete(ContextRecord).where(ContextRecord.updated_at < cutoff_date)
        )
        await session.commit()
        
        num_deleted = result.rowcount
        logger.info(f"Cleaned up {num_deleted} stale contexts")
```

3. **Progressive Context Enhancement**: Design your system to allow new context attributes to be added without breaking existing functionality.

```python
# Example of context schema versioning and migration
class ContextMigration:
    @staticmethod
    def migrate_context(context_data, from_version, to_version):
        """Migrate context data between schema versions"""
        if from_version == 1 and to_version == 2:
            # Add new fields for version 2
            if "user" in context_data and "preferences" not in context_data["user"]:
                context_data["user"]["preferences"] = {}
            
            # Restructure existing fields
            if "environment" in context_data and "location" in context_data["environment"]:
                location = context_data["environment"].pop("location")
                if "geography" not in context_data:
                    context_data["geography"] = {}
                context_data["geography"]["location"] = location
            
            # Update version
            context_data["schema_version"] = 2
        
        # Add more migration paths as needed
        return context_data
```

#### Performance Optimization
1. **Efficient Context Retrieval**: Ensure your database schema is properly indexed for quick context lookups:

```sql
-- Example SQL for adding indexes to the contexts table
CREATE INDEX idx_contexts_user_id ON contexts ((user_context->>'user_id'));
CREATE INDEX idx_contexts_updated_at ON contexts (updated_at);
```

2. **Implement Caching Layers**: Use Redis or a similar in-memory store to cache frequently accessed contexts:

```python
# Example context caching implementation
async def get_cached_context(context_id: str, db: Session):
    """Get context with caching"""
    # Try cache first
    cache_key = f"context:{context_id}"
    cached = redis_client.get(cache_key)
    
    if cached:
        return json.loads(cached)
    
    # If not in cache, get from database
    context = context_store.get_context(db, context_id)
    if context:
        # Store in cache with TTL
        redis_client.setex(cache_key, 3600, json.dumps(context))
    
    return context
```

3. **Batch Processing**: For high-volume applications, implement batch processing for context updates:

```python
# Example batch context update
async def batch_update_contexts(updates: List[Dict[str, Any]]):
    """Update multiple contexts in a single transaction"""
    async with async_session() as session:
        async with session.begin():
            for update in updates:
                context_id = update["context_id"]
                data = update["data"]
                
                stmt = (
                    update(ContextRecord)
                    .where(ContextRecord.id == context_id)
                    .values(updated_at=datetime.utcnow(), **data)
                )
                await session.execute(stmt)
```

#### Security and Compliance
1. **Implement Role-Based Access Control**: Ensure only authorized users can access context data:

```python
# Extend the has_role function to check context ownership
def has_context_access(context_id: str):
    async def context_access_checker(current_user: User = Depends(get_current_active_user),
                                    db: Session = Depends(get_db)):
        # Admin always has access
        if current_user.role == "admin":
            return current_user
            
        # Get the context
        context = context_store.get_context(db, context_id)
        if not context:
            raise HTTPException(status_code=404, detail="Context not found")
            
        # Check if user owns this context
        if (context["user"]["user_id"] != current_user.username and 
            current_user.role != "model_developer"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this context"
            )
            
        return current_user
    return context_access_checker
```

2. **Implement Data Retention Policies**: Ensure compliance with regulations like GDPR by implementing proper data retention:

```python
# Example GDPR compliance handler
async def delete_user_data(user_id: str):
    """Delete all contexts associated with a user"""
    async with async_session() as session:
        # Find contexts with this user
        result = await session.execute(
            select(ContextRecord)
            .where(ContextRecord.user_context.contains({"user_id": user_id}))
        )
        contexts = result.scalars().all()
        
        # Delete each context
        for context in contexts:
            await session.delete(context)
            
        await session.commit()
        
        return len(contexts)
```

### Common Pitfalls to Avoid

#### Context Management Issues
1. **Overly Large Contexts**: Contexts that grow unbounded can cause performance issues and increased latency:

```python
# Implement context size limits
def validate_context_size(context):
    """Check if context is within size limits"""
    context_json = json.dumps(context)
    size_kb = len(context_json) / 1024
    
    if size_kb > 100:  # Example: 100KB limit
        logger.warning(f"Context size {size_kb:.2f}KB exceeds recommended limit of 100KB")
        
        # Take action - could truncate history, remove less important fields, etc.
        if "user" in context and "session_history" in context["user"]:
            # Keep only last 10 interactions
            context["user"]["session_history"] = context["user"]["session_history"][-10:]
    
    return context
```

2. **Inconsistent Context Formats**: Ensure all services create and consume context in consistent formats:

```python
# Use validation middleware
@app.middleware("http")
async def validate_context_middleware(request: Request, call_next):
    """Validate context structure in requests"""
    if request.url.path.startswith("/api/v1/context"):
        body = await request.json()
        try:
            # Validate against schema
            MCPContext(**body)
        except ValidationError as e:
            return JSONResponse(
                status_code=422,
                content={"detail": "Invalid context format", "errors": e.errors()}
            )
    
    response = await call_next(request)
    return response
```

#### Performance Pitfalls
1. **N+1 Query Problems**: Avoid making multiple database queries in loops:

```python
# BAD EXAMPLE - DON'T DO THIS
async def get_multiple_contexts(context_ids: List[str]):
    contexts = []
    for context_id in context_ids:
        # N+1 problem - one query per context
        context = await get_context(context_id)
        contexts.append(context)
    return contexts

# GOOD EXAMPLE - DO THIS INSTEAD
async def get_multiple_contexts_efficiently(context_ids: List[str]):
    # Single query to get all contexts
    async with async_session() as session:
        result = await session.execute(
            select(ContextRecord)
            .where(ContextRecord.id.in_(context_ids))
        )
        contexts = result.scalars().all()
        
        # Process results
        return [context.to_dict() for context in contexts]
```

2. **Synchronous I/O in Async Code**: Ensure all I/O operations are properly async:

```python
# BAD EXAMPLE - DON'T DO THIS
async def get_model_result(model_id: str, input_data: Dict[str, Any]):
    # Blocking I/O in async function!
    with open(f"/models/{model_id}/config.json", "r") as f:
        config = json.load(f)
    
    # More async code...

# GOOD EXAMPLE - DO THIS INSTEAD
async def get_model_result_correctly(model_id: str, input_data: Dict[str, Any]):
    # Use aiofiles for async file I/O
    import aiofiles
    
    async with aiofiles.open(f"/models/{model_id}/config.json", "r") as f:
        content = await f.read()
        config = json.loads(content)
    
    # More async code...
```

#### Security Pitfalls
1. **Inadequate Input Validation**: Always validate all inputs, especially context data:

```python
# Implement strict validation
class UpdateContextRequest(BaseModel):
    """Schema for context update requests"""
    user: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None
    model: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    custom_attributes: Optional[Dict[str, Any]] = None
    
    class Config:
        # Prevent additional fields
        extra = "forbid"
```

2. **Using Passwords or Keys in Context**: Never store sensitive authentication data in context:

```python
# Context sanitizer example
def sanitize_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive data from context"""
    sanitized = context.copy()
    
    # Remove any fields that might contain sensitive data
    sensitive_fields = [
        "password", "token", "key", "secret", "auth", "credential", "api_key"
    ]
    
    # Check in custom attributes
    if "custom_attributes" in sanitized:
        for field in sensitive_fields:
            if field in sanitized["custom_attributes"]:
                del sanitized["custom_attributes"][field]
    
    # Check in user preferences
    if "user" in sanitized and "preferences" in sanitized["user"]:
        for field in sensitive_fields:
            if field in sanitized["user"]["preferences"]:
                del sanitized["user"]["preferences"][field]
    
    return sanitized
```

## 9. Conclusion

### MCP Server Benefits

Building your own MCP-compliant server provides several key advantages:

1. **Complete Control**: With a custom MCP server, you have full control over how context is managed, stored, and utilized by your models.

2. **Tailored Performance**: You can optimize performance based on your specific workloads and deployment environment, from resource allocation to caching strategies.

3. **Customized Security**: Implement security measures that align with your organization's requirements and compliance needs.

4. **Integration Flexibility**: Connect your MCP server to your existing systems, databases, and services with custom integrations.

5. **Cost Optimization**: Avoid vendor lock-in and potentially reduce costs by implementing exactly what you need, especially for high-volume deployments.
