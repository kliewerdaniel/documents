---
layout: post
title: Simulacra
description: This comprehensive guide demonstrates how to integrate the official OpenAI Agents SDK with Ollama to create AI agents that run entirely on local infrastructure. By the end, you'll understand both the theoretical foundations and practical implementation of locally-hosted AI agents.
date:   2025-03-13 01:42:44 -0500
---
# Comprehensive Guide to Simulacra01

This guide provides detailed documentation on how to use, customize, and extend Simulacra01, a framework that integrates the OpenAI Agents SDK with Ollama for local AI agent capabilities.

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Installation & Setup](#installation--setup)
4. [Using Document Analysis Agent](#using-document-analysis-agent)
5. [Working with the Command-Line Interface](#working-with-the-command-line-interface)
6. [Creating Custom Agents](#creating-custom-agents)
7. [Advanced Customization](#advanced-customization)
8. [Debugging and Troubleshooting](#debugging-and-troubleshooting)
9. [Performance Optimization](#performance-optimization)
10. [Contributing and Development](#contributing-and-development)

## Introduction

Simulacra01 is a powerful framework that brings together the structured agent capabilities of OpenAI's Agents SDK with the privacy and cost benefits of local LLM inference through Ollama. This integration enables you to build sophisticated AI agents that run entirely on your local infrastructure.

### Key Benefits

- **Complete Data Privacy**: All processing happens locally, with no data sent to external services
- **Cost Efficiency**: No per-token API costs associated with cloud-based LLM services
- **Customizability**: Full control over model selection, fine-tuning, and behavior
- **Network Independence**: Agents function without requiring internet access
- **Reduced Latency**: Eliminate network roundtrips for faster responses

### Core Components

- **OpenAI Agents SDK**: Provides the structured framework for building AI agents
- **Ollama**: Enables local running of various open-source LLMs
- **Adapter Layer**: Connects the two technologies seamlessly
- **Specialized Agents**: Pre-built agents for document analysis and other tasks
- **Command-Line Interface**: Interactive way to engage with agents

## Understanding the Architecture

Simulacra01 employs a layered architecture designed for flexibility and extensibility:

### Ollama Layer

The base layer provides LLM inference capabilities:

- Handles model loading and management
- Processes raw prompts into completions
- Manages system resources for inference
- Provides API endpoints that mimic OpenAI's structure

### Adapter Layer

The bridge between Ollama and the OpenAI Agents SDK:

- `OllamaClient`: Routes requests to Ollama's API endpoints
- `AgentAdapter`: Makes OpenAI's Agent class compatible with the Ollama backend
- `ResponseFormatter`: Ensures responses match expected formats
- `ToolCallProcessor`: Handles function/tool calls with local models

### Agents SDK Layer

Provides the agent framework and abstractions:

- Agent lifecycle management
- Tool definition and integration
- Conversation handling
- Response processing

### Application Layer

Implements specialized agents and interfaces:

- Document Analysis Agent
- Command-Line Interface
- Document Memory system
- Other specialized agent types

## Installation & Setup

### System Requirements

- Python 3.9 or higher
- 8GB+ RAM recommended (model dependent)
- 2GB+ free disk space for model storage

### Step 1: Install Ollama

For macOS and Linux:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

For Windows, download from [Ollama's website](https://ollama.com/download).

Verify installation:

```bash
ollama --version
```

### Step 2: Download Required Models

```bash
# Pull the Mistral model (recommended starting model)
ollama pull mistral

# Optional: Pull additional models
ollama pull llama3
ollama pull mixtral
```

Verify model installation:

```bash
ollama list
```

### Step 3: Clone and Install Simulacra01

```bash
git clone https://github.com/kliewerdaniel/simulacra01.git
cd simulacra01
pip install -e .
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

Run the basic test script:

```bash
python -c "from ollama_client import OllamaClient; client = OllamaClient(); response = client.chat.completions.create(model='mistral', messages=[{'role': 'user', 'content': 'Hello, world!'}]); print(response.choices[0].message.content)"
```

You should see a response from the model.

## Using Document Analysis Agent

The Document Analysis Agent is a powerful tool for extracting information from documents, answering questions about content, and managing a document repository.

### Basic Usage

Run the document agent:

```bash
python main.py
```

This will start an interactive session with the agent.

### Available Commands

- `exit`: Exit the agent
- `help`: Show help information
- `list`: List documents in memory

### Example Interactions

Analyze a webpage:
```
You: Please analyze the article at https://en.wikipedia.org/wiki/Artificial_intelligence and tell me when AI was first developed.
```

Extract specific information:
```
You: Extract all the dates mentioned in the last document.
```

Search for content:
```
You: Find information about neural networks in the document.
```

### Tool Functionality

The Document Analysis Agent includes several specialized tools:

#### fetch_document

Retrieves document content from a URL:

```python
fetch_document(url="https://example.com/article")
```

This tool:
- Checks if the document is already in memory
- If not, fetches it from the URL
- Stores it in document memory for future use
- Returns the document content

#### extract_info

Extracts specific types of information from text:

```python
extract_info(text="document content", info_type="dates")
```

Common info types:
- `dates`: Extracts dates and timestamps
- `names`: Extracts person names
- `organizations`: Extracts organization names
- `key points`: Extracts main ideas or arguments
- `statistics`: Extracts numerical data and statistics

#### search_document

Searches document content for relevant information:

```python
search_document(text="document content", query="neural networks")
```

This uses semantic search to find the most relevant paragraphs for the query.

### Document Memory

The Document Memory system provides persistent storage for documents:

```python
from document_memory import DocumentMemory

# Initialize memory
memory = DocumentMemory()

# Store a document
doc_id = memory.store_document(
    url="https://example.com/article",
    content="Document text goes here...",
    metadata={"author": "John Doe", "date": "2025-03-13"}
)

# Retrieve a document
doc = memory.get_document(doc_id)
print(doc["content"])

# List all documents
docs = memory.list_documents()
for doc in docs:
    print(f"URL: {doc['url']}")
```

Document memory is stored on disk and persists between sessions.

## Working with the Command-Line Interface

The Simulacra01 CLI provides a comprehensive interface for interacting with various agent types.

### Starting the CLI

```bash
# Start with interactive menu
python cli.py

# Start directly with a specific agent
python cli.py chat --agent document
python cli.py chat --agent research
```

### Global Commands

These commands work across all agent types:

- `exit`: End the current session
- `help`: Show available commands
- `clear`: Clear the conversation history
- `save [filename]`: Save the current conversation
- `load <filename>`: Load a saved conversation
- `list`: List saved conversations
- `tools`: List available tools

### Agent-Specific Commands

#### Document Agent

- `list docs`: List stored documents
- `analyze <url>`: Analyze a document at URL

#### Research Agent

- `search <topic>`: Research a topic
- `synthesize`: Summarize research findings
- `save research <filename>`: Save research data

#### Task Agent

- `add task <title>`: Add a new task
- `list tasks`: Show all tasks
- `update task <id>`: Update task status

### Configuration

Configure the CLI using:

```bash
python cli.py config
```

This allows you to customize:

- OpenAI and Ollama settings
- Model preferences
- Agent-specific parameters
- System prompts

Configuration is stored in `~/.simulacra/config.json`.

## Creating Custom Agents

Simulacra01 makes it easy to create custom agents tailored to specific use cases.

### Basic Agent Creation

```python
from agents import Agent, function_tool
from ollama_client import OllamaClient
from pydantic import BaseModel, Field

# Define the client
client = OllamaClient(model_name="mistral")

# Define tool schemas
class AddInput(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

class AddOutput(BaseModel):
    result: int = Field(..., description="Sum of the two numbers")

# Define the tool function
@function_tool
def add(a: int, b: int) -> dict:
    """Adds two numbers together."""
    return {"result": a + b}

# Create the agent
agent = Agent(
    name="MathAgent",
    instructions="You are a math assistant that helps users with calculations.",
    tools=[add],
    model=client,
)

# Use the agent
response = agent.run("What is 5 + 7?")
print(response.message)
```

### Tool Development Best Practices

1. **Clear Function Signatures**: Make parameter names intuitive
2. **Comprehensive Docstrings**: Explain what the tool does
3. **Error Handling**: Gracefully handle exceptions
4. **Type Annotations**: Use proper type hints
5. **Schema Definitions**: Use Pydantic for input/output validation

### Complex Agent Example

Here's a more complex example of a custom agent:

```python
from agents import Agent, function_tool
from ollama_client import OllamaClient
from pydantic import BaseModel, Field
import requests
import json
import re

class WeatherInput(BaseModel):
    location: str = Field(..., description="City or location name")

class WeatherOutput(BaseModel):
    temperature: float = Field(..., description="Current temperature in Celsius")
    conditions: str = Field(..., description="Weather conditions")
    humidity: float = Field(..., description="Humidity percentage")

class ForecastInput(BaseModel):
    location: str = Field(..., description="City or location name")
    days: int = Field(3, description="Number of days to forecast")

class ForecastOutput(BaseModel):
    forecast: list = Field(..., description="Daily forecast data")

@function_tool
def get_weather(location: str) -> dict:
    """Gets the current weather for a location."""
    try:
        # Example implementation (would use actual weather API)
        response = requests.get(f"https://weather-api.example.com/current?q={location}")
        data = response.json()
        return {
            "temperature": data["temp_c"],
            "conditions": data["condition"]["text"],
            "humidity": data["humidity"]
        }
    except Exception as e:
        return {"error": str(e)}

@function_tool
def get_forecast(location: str, days: int = 3) -> dict:
    """Gets a weather forecast for a location."""
    try:
        # Example implementation (would use actual weather API)
        response = requests.get(f"https://weather-api.example.com/forecast?q={location}&days={days}")
        data = response.json()
        forecast = []
        for day in data["forecast"]["forecastday"]:
            forecast.append({
                "date": day["date"],
                "max_temp": day["day"]["maxtemp_c"],
                "min_temp": day["day"]["mintemp_c"],
                "condition": day["day"]["condition"]["text"]
            })
        return {"forecast": forecast}
    except Exception as e:
        return {"error": str(e)}

def create_weather_agent():
    """Creates a weather information agent."""
    client = OllamaClient(model_name="mistral")
    
    agent = Agent(
        name="WeatherAgent",
        instructions="""
        You are a Weather Assistant that provides accurate weather information.
        
        When a user asks about the weather:
        1. Use get_weather to fetch current conditions
        2. Use get_forecast for multi-day forecasts
        
        Always specify temperature units (Celsius) in your responses.
        For forecasts, present the information in a clear, day-by-day format.
        If a user doesn't specify a location, ask them for clarification.
        """,
        tools=[get_weather, get_forecast],
        model=client,
    )
    
    return agent

# Usage
agent = create_weather_agent()
response = agent.run("What's the weather like in London?")
print(response.message)
```

## Advanced Customization

### Using Different Models

Ollama supports numerous open-source models. To use a different model:

1. Pull the model:

```bash
ollama pull llama3
```

2. Specify the model when creating the client:

```python
client = OllamaClient(model_name="llama3")
```

#### Model Recommendations

- **mistral**: Good balance of performance and speed
- **llama3**: High quality, larger context window
- **mixtral**: Strong multi-specialty model
- **gemma**: Efficient for simpler tasks
- **phi3**: Latest Microsoft model with strong capabilities

### Custom System Prompts

The system prompt (instructions) is crucial for agent behavior:

```python
agent = Agent(
    name="CustomAgent",
    instructions="""
    You are a specialized assistant that helps with [specific domain].
    
    Follow these guidelines:
    1. Always begin by [specific action]
    2. For complex queries, use [specific approach]
    3. When uncertain, [specific strategy]
    
    When using tools:
    - Use [tool1] for [specific scenario]
    - Use [tool2] when [specific condition]
    
    Response format:
    - Start with a [specific element]
    - Include [specific component]
    - Format using [specific style]
    
    Additional instructions:
    [any other specific behavioral guidance]
    """,
    tools=tools,
    model=client,
)
```

### Caching Implementation

Implement caching to improve performance and reduce redundant model calls:

```python
from caching_service import CachingService
from ollama_client import OllamaClient

# Create a caching layer
cache = CachingService(cache_dir="./cache")

# Create a cached client
class CachedOllamaClient(OllamaClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = cache
    
    def chat_completion(self, messages, **kwargs):
        cache_key = self._generate_cache_key(messages, kwargs)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        result = super().chat_completion(messages, **kwargs)
        self.cache.store(cache_key, result)
        
        return result
    
    def _generate_cache_key(self, messages, kwargs):
        # Create a deterministic key from messages and relevant kwargs
        key_components = [
            self.model_name,
            str(messages),
            str({k: v for k, v in kwargs.items() if k in ["temperature", "max_tokens"]})
        ]
        return hash("".join(key_components))

# Use the cached client
client = CachedOllamaClient(model_name="mistral")
```

### Custom Tool Categories

Organize tools into categories for more structured agents:

```python
from agents import Agent, function_tool
from ollama_client import OllamaClient

# Document tools
@function_tool(category="document")
def fetch_document(url: str) -> dict:
    """Fetches a document from URL."""
    # Implementation

@function_tool(category="document")
def analyze_document(text: str) -> dict:
    """Analyzes document content."""
    # Implementation

# Search tools
@function_tool(category="search")
def web_search(query: str) -> dict:
    """Searches the web for information."""
    # Implementation

@function_tool(category="search")
def image_search(query: str) -> dict:
    """Searches for images."""
    # Implementation

# Create an agent with tool categories
client = OllamaClient(model_name="mixtral")

agent = Agent(
    name="ResearchAssistant",
    instructions="""
    You are a Research Assistant with access to different tool categories:
    
    DOCUMENT TOOLS:
    - Use fetch_document to retrieve document content
    - Use analyze_document to extract insights from documents
    
    SEARCH TOOLS:
    - Use web_search to find information online
    - Use image_search to find relevant images
    
    Choose the appropriate tool category based on the user's request.
    """,
    tools=[fetch_document, analyze_document, web_search, image_search],
    model=client,
)
```

### Custom Response Formatting

Implement custom response formatting for specialized outputs:

```python
class CustomAgentResponse:
    def __init__(self, result):
        self.message = self._format_message(result.final_output)
        self.conversation_id = getattr(result, 'conversation_id', None)
        self.tool_calls = self._extract_tool_calls(result)
        self.formatted_output = self._generate_formatted_output()
        
    def _format_message(self, message):
        # Format the message (e.g., add markdown, structure sections)
        return message
        
    def _extract_tool_calls(self, result):
        # Extract tool call information
        tool_calls = []
        # Extraction logic
        return tool_calls
        
    def _generate_formatted_output(self):
        # Create a custom formatted output (e.g., HTML, JSON)
        return {
            "message": self.message,
            "tools_used": [t.name for t in self.tool_calls],
            "formatted_at": datetime.now().isoformat()
        }
        
    def to_json(self):
        """Convert the response to JSON."""
        return json.dumps(self.formatted_output)
        
    def to_html(self):
        """Convert the response to HTML."""
        html = f"<div class='agent-response'><p>{self.message}</p>"
        if self.tool_calls:
            html += "<ul class='tools-used'>"
            for tool in self.tool_calls:
                html += f"<li>{tool.name}</li>"
            html += "</ul>"
        html += "</div>"
        return html
```

## Debugging and Troubleshooting

### Enabling Debug Logging

```python
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulacra01.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create module-specific loggers
ollama_logger = logging.getLogger("ollama_client")
agent_logger = logging.getLogger("agent")
tools_logger = logging.getLogger("tools")

# Set specific log levels if needed
ollama_logger.setLevel(logging.DEBUG)
```

### Common Issues and Solutions

#### Model Not Found

**Problem**: `Error: model 'xyz' not found`

**Solution**:
1. Check available models: `ollama list`
2. Pull the missing model: `ollama pull xyz`
3. Verify model spelling in your code

#### Memory Issues

**Problem**: Out of memory errors when running large models

**Solution**:
1. Use a smaller model (mistral instead of mixtral)
2. Reduce batch size or context length
3. Add system swap space
4. Close other memory-intensive applications

#### API Compatibility

**Problem**: OpenAI SDK methods not working with Ollama

**Solution**:
1. Check the adapter implementation for the specific method
2. Add wrapper methods to OllamaClient class
3. Consult Ollama API documentation for endpoint limitations

#### Tool Call Issues

**Problem**: Model not using tools or using them incorrectly

**Solution**:
1. Simplify tool definitions and make their purpose more explicit
2. Check that tool schemas are properly defined
3. Add more specific instructions about tool usage in the agent prompt
4. Try a more capable model (llama3 or mixtral)

### Inspecting Raw Responses

To inspect raw model responses:

```python
from ollama_client import OllamaClient

client = OllamaClient(model_name="mistral")

# Get raw response
response = client.chat.completions.create(
    model="mistral",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    temperature=0.7,
)

# Print full response object
import json
print(json.dumps(response.model_dump(), indent=2))
```

## Performance Optimization

### Model Selection Strategies

Choose the right model for the task:

| Task Type | Recommended Model | Notes |
|-----------|-------------------|-------|
| Simple Q&A | gemma | Fastest, lower resource usage |
| General assistant | mistral | Good balance of quality/speed |
| Complex reasoning | llama3 | Higher quality, more resources |
| Specialized domains | mixtral | Multi-specialty, highest resources |

### Prompt Optimization

Optimize prompts for better efficiency:

1. **Be Specific**: Clear, concise instructions reduce token usage
2. **Provide Examples**: Few-shot examples improve response quality
3. **Structured Output**: Request specific formats to reduce parsing needs
4. **Limit Context**: Only include relevant information
5. **Use Separators**: Clearly delineate sections with markers

### Chunking Strategies

For large documents, implement effective chunking:

```python
def chunk_document(text, chunk_size=2000, overlap=200):
    """Split document into overlapping chunks."""
    chunks = []
    
    # Simple character-based chunking
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    
    return chunks

def chunk_document_by_section(text, chunk_size=2000):
    """Split document by natural section boundaries."""
    # Find section boundaries (e.g., headers, paragraph breaks)
    sections = re.split(r'(?:\n\s*){2,}|(?:#{1,6}\s+[^\n]+\n)', text)
    
    chunks = []
    current_chunk = ""
    
    for section in sections:
        # If adding this section would exceed chunk size, save current chunk
        if len(current_chunk) + len(section) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = section
        else:
            current_chunk += section
    
    # Add the final chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
```

### Caching Optimization

Implement multi-level caching for better performance:

```python
class MultiLevelCache:
    def __init__(self):
        self.memory_cache = {}  # Fast, in-memory cache
        self.disk_cache = DiskCache("./cache")  # Persistent disk cache
        
    def get(self, key):
        # First check memory cache (fastest)
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Then check disk cache
        disk_result = self.disk_cache.get(key)
        if disk_result:
            # Also store in memory for future fast access
            self.memory_cache[key] = disk_result
            return disk_result
        
        return None
        
    def store(self, key, value):
        # Store in both caches
        self.memory_cache[key] = value
        self.disk_cache.store(key, value)
        
    def clear_memory_cache(self):
        """Clear only memory cache (e.g., to free memory)."""
        self.memory_cache = {}
```

### Parallel Processing

Implement parallel processing for multiple operations:

```python
import concurrent.futures

def process_document_parallel(document, queries):
    """Process multiple queries against a document in parallel."""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Create a future for each query
        future_to_query = {
            executor.submit(search_document, document, query): query
            for query in queries
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                results[query] = future.result()
            except Exception as e:
                results[query] = {"error": str(e)}
                
    return results
```

## Contributing and Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/simulacra01.git
cd simulacra01

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ollama_client.py

# Run with coverage
coverage run -m pytest
coverage report
coverage html  # Generate HTML report
```

### Code Style and Linting

```bash
# Check code style
ruff check .

# Auto-fix issues
ruff check --fix .

# Run type checking
mypy .
```

### Documentation Generation

```bash
# Generate API documentation
python scripts/generate_docs.py

# Build documentation website
mkdocs build

# Serve documentation locally
mkdocs serve
```

### Branch Strategy

- `main`: Stable release branch
- `develop`: Main development branch
- `feature/*`: For new features
- `bugfix/*`: For bug fixes
- `release/*`: For release preparation

### Commit Guidelines

Use conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### Pull Request Process

1. Create a new branch from `develop`
2. Make your changes
3. Run tests and linting
4. Submit a pull request to `develop`
5. Ensure CI passes
6. Request review from maintainers
7. Address review feedback

---

This comprehensive guide covers all aspects of using, customizing, and extending Simulacra01. For additional information, refer to the specific documentation files in the docs/ directory.
