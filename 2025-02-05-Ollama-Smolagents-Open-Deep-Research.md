---
layout: post
title:  Ollama Smolagents Open Deep Research Integration
date:   2025-02-05 09:42:44 -0500
---
**Unlocking Open-Source AI Power: How to Run Your Own Deep Research Agent with Ollama and Smolagents**  

The AI landscape is rapidly evolving, but relying solely on proprietary models like OpenAI’s GPT-4 comes with limitations: cost, lack of transparency, and restricted customization. Enter **Ollama** and **smolagents**—a dynamic open-source duo that lets you build powerful, customizable AI agents for deep research, creative tasks, and more. In this guide, we’ll explore how to harness these tools to create your own AI research assistant, complete with web search, image generation, and advanced reasoning capabilities—all while maintaining full control over your stack.

---

### Why Open-Source AI Agents Matter

Before diving into the code, let’s address the *why*:  

1. **Transparency & Control**: Open-source models let you inspect, modify, and understand the AI’s decision-making process.  
2. **Cost Efficiency**: Avoid per-API-call pricing models.  
3. **Privacy**: Keep sensitive data in-house instead of sending it to third-party servers.  
4. **Customization**: Integrate domain-specific tools and workflows seamlessly.  

By combining Ollama (a lightweight framework for running local LLMs) with smolagents (a modular agent-building toolkit), you gain the flexibility to create AI solutions tailored to your needs—whether that’s academic research, content generation, or data analysis.

---

### The Architecture: Ollama + Smolagents + Tools

Our setup uses three core components:  

1. **Ollama**: Runs local language models (like Mistral) for text generation.  
2. **Smolagents**: Manages task planning, tool integration, and agent logic.  
3. **External Tools**: DuckDuckGo (web search) and text-to-image generation.  

Here’s how they interact:  
![Architecture diagram: User → Agent → Ollama → Tools → Output]  
*(Imagine a flowchart here showing the flow of prompts, model processing, and tool usage.)*

---

### Step-by-Step Setup Guide

#### 1. Prerequisites  
- Python 3.10+ installed  
- Basic terminal/command-line knowledge  
- Ollama installed ([Installation Guide](https://ollama.ai/download))  

#### 2. Install Dependencies  
```bash
pip install smolagents python-dotenv ollama
```

#### 3. Configure Environment  
Create a `.env` file for secrets (even if empty for now):  
```bash
touch .env
```

---

### Deep Dive: The Code Explained

Let’s break down the provided code into key sections:  

#### **1. Message Handling**  
```python
@dataclass
class Message:
    content: str
```
This simple class standardizes communication between the agent and tools, ensuring compatibility with smolagents’ expectations.

#### **2. Ollama Model Wrapper**  
```python
class OllamaModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = ollama.Client()

    def __call__(self, messages, **kwargs):
        # [Message formatting logic...]
        response = self.client.chat(...)
        return Message(content=response["message"]["content"])
```  
This class acts as a bridge between smolagents and Ollama’s API. Key features:  
- Handles multiple message types (strings, dictionaries)  
- Enforces role-based formatting (“user”, “assistant”, etc.)  
- Sets model parameters like temperature (0.7 for balanced creativity)  

#### **3. Tool Integration**  
```python
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
search_tool = DuckDuckGoSearchTool()
```  
- **DuckDuckGoSearchTool**: Enables real-time web searches for up-to-date information.  
- **Text-to-Image Tool**: Generates images from prompts using Hugging Face’s ecosystem.  

#### **4. Agent Initialization**  
```python
agent = CodeAgent(
    tools=[search_tool, image_generation_tool],
    model=ollama_model,
    planning_interval=3
)
```  
The `CodeAgent` is configured to:  
- Use Mistral 24B (a powerful open-source model) via Ollama  
- Re-plan actions every 3 steps to adapt to new information  
- Access both web search and image generation  

---

### Running Your Agent

Replace `"YOUR_PROMPT"` with a research question or task:  
```python
result = agent.run(
    "Explain quantum entanglement in simple terms, then generate a visualization."
)
```  
**Example Output Workflow:**  
1. Agent plans: “First search for quantum entanglement basics.”  
2. DuckDuckGo returns top 3 results.  
3. Ollama summarizes findings into layman’s terms.  
4. Image tool creates a conceptual diagram.  
5. Final response combines text and image URL.

---

### Why This Beats Proprietary Alternatives

1. **Full Control**: Adjust temperature, max tokens, and other parameters at will.  
2. **Tool Flexibility**: Swap DuckDuckGo for arXiv search, add Python execution, etc.  
3. **Cost**: Zero per-query fees after initial setup.  
4. **Privacy**: All data stays on your infrastructure.  

---

### Advanced Customization Ideas

1. **Domain-Specific Models**: Fine-tune Ollama with medical, legal, or technical datasets.  
2. **Multi-Agent Teams**: Create specialized agents (researcher, writer, fact-checker) that collaborate.  
3. **Custom Tools**: Integrate internal APIs or databases.  
4. **Human-in-the-Loop**: Add approval steps for sensitive tasks.  

---

### Troubleshooting Tips

- **Ollama Model Not Loading**: Ensure the model is downloaded via `ollama pull mistral-small:24b-instruct-2501-q8_0`  
- **Permission Issues**: Use `trust_remote_code=True` cautiously—only with trusted tools.  
- **Memory Constraints**: Smaller models like Mistral 7B work if 24B is too resource-heavy.  

---

### The Future of Open-Source AI Research

This setup is just the beginning. As the open-source ecosystem grows, expect:  
- Better multimodality (video processing, 3D generation)  
- Improved tool-learning frameworks  
- Lower hardware requirements via quantization  

By building with Ollama and smolagents today, you’re positioning yourself at the forefront of accessible, ethical AI development.

---


```python
from smolagents import load_tool, CodeAgent, DuckDuckGoSearchTool
from dotenv import load_dotenv
import ollama
from dataclasses import dataclass

# Load environment variables
load_dotenv()

@dataclass
class Message:
    content: str  # Required attribute for smolagents

class OllamaModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = ollama.Client()

    def __call__(self, messages, **kwargs):
        formatted_messages = []
        
        # Ensure messages are correctly formatted
        for msg in messages:
            if isinstance(msg, str):
                formatted_messages.append({
                    "role": "user",  # Default to 'user' for plain strings
                    "content": msg
                })
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(part.get("text", "") for part in content if isinstance(part, dict) and "text" in part)
                formatted_messages.append({
                    "role": role if role in ['user', 'assistant', 'system', 'tool'] else 'user',
                    "content": content
                })
            else:
                formatted_messages.append({
                    "role": "user",  # Default role for unexpected types
                    "content": str(msg)
                })

        response = self.client.chat(
            model=self.model_name,
            messages=formatted_messages,
            options={'temperature': 0.7, 'stream': False}
        )
        
        # Return a Message object with the 'content' attribute
        return Message(
            content=response.get("message", {}).get("content", "")
        )

# Define tools
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
search_tool = DuckDuckGoSearchTool()

# Define the custom Ollama model
ollama_model = OllamaModel("mistral-small:24b-instruct-2501-q8_0")

# Create the agent
agent = CodeAgent(
    tools=[search_tool, image_generation_tool],
    model=ollama_model,
    planning_interval=3
)

# Run the agent
result = agent.run(
    "YOUR_PROMPT"
)

# Output the result
print(result)
```
