---
layout: post
title:  PydanticAI-RAG
date:   2024-12-09 07:42:44 -0500
---

Below is a comprehensive, step-by-step guide designed for developers looking to combine persona-driven data modeling with Retrieval-Augmented Generation (RAG) using Pydantic AI’s Agent and Tools APIs. We’ll integrate concepts from the **PersonaGen07 repository**, the **RAG example from Pydantic AI**, and the **Agent** and **Tools** APIs into a cohesive system. By the end, you’ll have a working setup that allows you to define personas, retrieve relevant documents, and produce AI-generated responses customized to each persona’s style and preferences.

---

## 1. Introduction

Modern generative AI systems can be greatly enhanced by incorporating external data (for accuracy and recency) and persona-driven customization (for personalization and relevance to specific user profiles). **Retrieval-Augmented Generation (RAG)** ensures that the model’s output is grounded in reliable data sources, while persona-based logic tailors responses to different user archetypes, such as a student, a marketing professional, or a tech enthusiast.

**PersonaGen07** provides a structured way to define personas as JSON files, capturing attributes like communication style, domain interests, and preferred tone. **Pydantic AI** offers a typed, schema-driven approach to working with AI models, as well as the **Agent** and **Tools** APIs that streamline interaction with external data and services. Together, these tools create a system that:

- Retrieves context-relevant information dynamically.
- Adapts responses based on predefined persona traits.
- Maintains a clean, schema-based code structure for reliability and maintainability.

---

## 2. Prerequisites

Before we begin, ensure you have the following:

- **Python 3.9+** recommended.
- Access to the **OpenAI API** or another supported LLM provider (ensure you have an API key).
- **Pydantic AI** library installed.
- **PersonaGen07** repository cloned locally.

### Required Python Packages

- `pydantic[ai]` for Pydantic AI.
- `openai` for interacting with the OpenAI API.
- `requests` if needed for advanced retrieval scenarios.
- `json` (standard library) for handling persona files.

### Terminal Setup Commands

```bash
# Clone PersonaGen07 repository
git clone https://github.com/kliewerdaniel/PersonaGen07.git

# Navigate to your project directory
cd your-project-directory

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pydantic[ai] openai
```

You’ll also need to set your `OPENAI_API_KEY` as an environment variable or directly within your code. For example:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

---

## 3. Setup

### Cloning PersonaGen07

The PersonaGen07 repository provides a template for persona definitions. We’ll use its JSON format to structure our persona data.

```bash
git clone https://github.com/kliewerdaniel/PersonaGen07.git personas
```

This command clones the repo into a `personas` directory. Inside, you’ll find JSON schemas and example persona definitions. You may create your own persona files based on these examples.

### Installing Dependencies

We’ve already installed `pydantic[ai]` and `openai`. If you plan to use other retrieval methods or vector databases, install them here:

```bash
# Example for Pinecone or FAISS
pip install pinecone-client
```

### Setting Up API Keys

Make sure your environment is ready:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

If you use another LLM provider, refer to its documentation on key management.

---

## 4. Code Implementation

### Step 1: Define and Load Personas

First, create a persona JSON file. For example, `personas/student.json`:

```json
{
  "name": "Student",
  "attributes": {
    "communication_style": "friendly and explanatory",
    "interests": ["technology", "mathematics", "science"],
    "formality": "casual",
    "reading_level": "beginner"
  }
}
```

This file defines a “Student” persona who prefers casual, friendly explanations. You can create multiple personas—e.g., `personas/marketing_expert.json` with a more formal, sales-oriented style.

**Persona Loading Code (`persona_manager.py`):**

```python
import json
from pathlib import Path

class PersonaManager:
    def __init__(self, persona_path: str):
        persona_file = Path(persona_path)
        if not persona_file.exists():
            raise FileNotFoundError(f"Persona file not found: {persona_path}")
        with persona_file.open('r') as f:
            self.persona = json.load(f)
        self.name = self.persona.get("name", "Default")
        self.attributes = self.persona.get("attributes", {})

    def get_prompt_instructions(self) -> str:
        style = self.attributes.get("communication_style", "neutral")
        formality = self.attributes.get("formality", "neutral")
        return f"Please respond in a {formality}, {style} manner."
```

This simple class loads persona data and provides a method to generate persona-specific prompt instructions.

### Step 2: Set Up a Retriever Function with the Tools API

Pydantic AI’s **Tools API** allows you to define tools (functions) that can be called by the AI agent to perform certain tasks, such as retrieving documents. For simplicity, let’s implement a dummy retrieval tool. Later, you can integrate a vector database or other data sources.

**Tools Setup (`tools.py`):**

```python
from pydantic_ai import tool
from typing import List

@tool(name="retrieve_documents", description="Retrieve documents based on a query")
def retrieve_documents(query: str) -> List[str]:
    # In a production scenario, implement a semantic search here.
    # For now, we return static documents filtered by a keyword match.
    docs = [
        "Document: RAG integrates retrieval with generation.",
        "Document: Personas help tailor AI responses.",
        "Document: Using Agents and Tools can streamline RAG pipelines."
    ]
    return [doc for doc in docs if query.lower() in doc.lower()]
```

### Step 3: Use the Pydantic AI Agent API for Retrieval and Generation

The **Agent API** allows you to define an AI agent that can use tools and produce answers. The agent can call `retrieve_documents` to get content and then incorporate persona instructions into the prompt.

**Agent Setup (`agent.py`):**

```python
import os
from pydantic_ai import Agent, AISettings
from persona_manager import PersonaManager
from tools import retrieve_documents

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Persona
persona_manager = PersonaManager("personas/student.json")

# Create an agent with the RAG approach
# The agent can call the 'retrieve_documents' tool to gather context.
ai_settings = AISettings(
    model="gpt-4", 
    api_key=OPENAI_API_KEY,
    temperature=0.7
)

agent = Agent(
    settings=ai_settings,
    tools=[retrieve_documents]
)

def persona_aware_query(query: str) -> str:
    # Fetch persona-specific instructions
    persona_instructions = persona_manager.get_prompt_instructions()
    # Prompt structure includes instructions, user query, and a command to retrieve documents
    prompt = (
        f"{persona_instructions}\n"
        f"The user asked: {query}\n"
        f"Use the 'retrieve_documents' tool if needed. Then answer the user.\n"
    )
    # Agent reasoning: The agent can decide to call retrieve_documents(query) before answering.
    return agent.run(prompt, max_tokens=200)
```

**How This Works:**

- We define a prompt that instructs the agent on how to respond.
- The agent can invoke the `retrieve_documents` tool to ground its answer.
- The persona instructions set the communication style.
- The agent’s final answer will incorporate retrieved documents and persona-based style.

### Step 4: Customizing the Agent’s Behavior Based on Persona Attributes

You might want to influence not just the style but also the retrieval strategy. For instance, if a persona is interested in “technology,” you could filter documents with a tech focus. Modify the retrieval tool or the prompt generation logic to leverage persona attributes:

```python
def persona_aware_query(query: str) -> str:
    persona_instructions = persona_manager.get_prompt_instructions()
    interests = persona_manager.attributes.get("interests", [])
    # Incorporate interests into the prompt to guide retrieval
    interest_tags = ", ".join(interests) if interests else "general knowledge"

    prompt = (
        f"{persona_instructions}\n"
        f"Persona interests: {interest_tags}\n"
        f"The user asked: {query}\n"
        f"Carefully select documents that match persona interests.\n"
        f"Use the 'retrieve_documents' tool if needed. Then answer the user.\n"
    )

    return agent.run(prompt, max_tokens=200)
```

This improved prompt nudges the agent to consider persona interests during the retrieval step.

---

## 5. Testing the Integration

### Testing with Different Personas

1. **Switch Personas:**  
   Update the persona file in `agent.py`:
   ```python
   # For a different persona
   persona_manager = PersonaManager("personas/marketing_expert.json")
   ```
   
2. **Run a Query:**
   ```bash
   python agent.py
   ```
   If your `agent.py` includes a test block:
   ```python
   if __name__ == "__main__":
       response = persona_aware_query("Explain what RAG is and why it's useful.")
       print(response)
   ```

You should see a response that:
- Incorporates retrieved content from `retrieve_documents`.
- Matches the persona’s communication style (e.g., friendly, casual).

### Verifying Personalization

Try switching from a “Student” persona to a “Marketing Expert” persona and compare the responses. The “Student” persona might yield more explanatory, beginner-friendly language, while the “Marketing Expert” persona might use more persuasive or marketing-oriented phrasing.

---

## 6. Advanced Features

### Semantic Search / Vector Databases

For better retrieval results, integrate a vector database like Pinecone. After setting up an index, modify the `retrieve_documents` tool to query the index and return semantically matched documents:

```python
@tool(name="retrieve_documents", description="Retrieve documents based on a query")
def retrieve_documents(query: str) -> List[str]:
    # Example with Pinecone or FAISS
    # 1. Embed the query
    # 2. Search the index
    # 3. Return the top matches
    # Return doc strings.
    pass
```

### Extending Persona Attributes

Personas could include more attributes, like a preferred reading level or specific domains of interest. You might adjust the prompt to direct the agent to simplify language or focus on certain topics based on these attributes.

### Fine-Tuning and Prompt Engineering

- Experiment with different `temperature` settings for creativity.
- Add chain-of-thought or reasoning steps in the prompt to improve the agent’s performance.

---

## 7. Conclusion

By integrating PersonaGen07, Pydantic AI’s RAG example, and the Agent and Tools APIs, we’ve built a flexible system that:

- **Retrieves Relevant Data:** Ensures that responses are enriched with external, current information.
- **Adapts to Personas:** Adjusts tone, complexity, and style based on user or application-defined personas.
- **Is Maintainable and Extensible:** Uses Pydantic’s structured approach and JSON-based personas for easy maintenance and scaling.

**Next Steps:**

- **Scale Retrieval:** Incorporate more advanced retrieval techniques, multiple databases, or third-party APIs.
- **Persona Refinement:** Add new persona attributes and refine how they influence retrieval and generation.
- **Fine-Tune Models:** Consider fine-tuning or using custom models for domain-specific applications.

With this framework in place, you’re well-positioned to build personalized, dynamic, and contextually accurate AI systems that cater to diverse user needs.