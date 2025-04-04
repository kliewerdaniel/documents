---
layout: post
title:  Step-by-Step Guide to Running Open Deep Research with smolagents
date:   2025-02-05 07:42:44 -0500
---

# Step-by-Step Guide to Running Open Deep Research with `smolagents`

This guide walks you through setting up and using the Open Deep Research agent framework, inspired by OpenAI's Deep Research, leveraging Hugging Face's `smolagents` library. Follow these steps to reproduce agentic workflows for complex tasks like the GAIA benchmark.

---

## Prerequisites
- **Python 3.8+** installed
- **Git** installed
- **Hugging Face Account** (optional for some model access)
- Basic familiarity with CLI tools

---

## Step 1: Set Up a Virtual Environment

Create an isolated Python environment to avoid dependency conflicts:

```bash
python3 -m venv venv          # Create virtual environment
source venv/bin/activate      # Activate it (Linux/macOS)
# For Windows: venv\Scripts\activate
```

---

## Step 2: Install Dependencies

1. **Upgrade Pip**:
   ```bash
   pip install --upgrade pip
   ```

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/huggingface/smolagents.git
   cd smolagents/examples/open_deep_research
   ```

3. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Step 3: Configure the Agent

### Key Components:
- **Model**: Use `Qwen/Qwen2.5-Coder-32B-Instruct` (default) or choose from [supported models](#model-options).
- **Tools**: Built-in tools include `web_search`, `translation`, and file/text inspection.
- **Imports**: Add Python libraries (e.g., `pandas`, `numpy`) for code-based agent actions.

---

## Step 4: Run the Agent via CLI

Use the `smolagent` command to execute tasks:

```bash
smolagent "{PROMPT}" \
  --model-type "HfApiModel" \
  --model-id "Qwen/Qwen2.5-Coder-32B-Instruct" \
  --imports "pandas numpy" \
  --tools "web_search translation"
```

### Example: GAIA-Style Task
```bash
smolagent "Which fruits in the 2008 painting 'Embroidery from Uzbekistan' were served on the October 1949 breakfast menu of the ocean liner later used in 'The Last Voyage'? List them clockwise from 12 o'clock." \
  --tools "web_search text_inspector"
```

---

## Model Options

Customize the LLM backend:

| Model Type         | Example Command                                                                 |
|--------------------|---------------------------------------------------------------------------------|
| Hugging Face API   | `--model-type "HfApiModel" --model-id "deepseek-ai/DeepSeek-R1"`                |
| LiteLLM (100+ LLMs)| `--model-type "LiteLLMModel" --model-id "anthropic/claude-3-5-sonnet-latest"`   |
| Local Transformers | `--model-type "TransformersModel" --model-id "Qwen/Qwen2.5-Coder-32B-Instruct"` |

---

## Advanced Usage

### 1. Vision-Enabled Web Browser
For tasks requiring visual analysis (e.g., image-based GAIA questions):
```bash
webagent "Analyze the product images on example.com/sale and list prices" \
  --model "LiteLLMModel" \
  --model-id "gpt-4o"
```

### 2. Sandboxed Execution
Run untrusted code safely using [E2B](https://e2b.dev/):
```bash
smolagent "{PROMPT}" --sandbox
```

### 3. Custom Tools
Add tools from LangChain/Hugging Face Spaces:
```python
# In your Python script
from smolagents import Tool
custom_tool = Tool.from_hub("username/my-custom-tool")
```

---

## Troubleshooting

| Issue                          | Solution                                  |
|--------------------------------|-------------------------------------------|
| `ModuleNotFoundError`          | Ensure virtual env is activated           |
| API Key Errors                 | Set `HF_TOKEN`/`ANTHROPIC_API_KEY` env vars |
| Tool Execution Failures        | Check tool dependencies in `requirements.txt` |

---

## Performance Notes

- **Code vs. JSON Agents**: Code-based agents achieve **~55% accuracy** on GAIA validation set vs. 33% for JSON-based ([source](https://huggingface.co/blog/open-deep-research)).
- **Speed**: Typical response time ~2-5 minutes for complex tasks (varies by model).

---

## Community Contributions

To improve this project:
1. **Enhance Tools**: Add PDF/Excel support to `text_inspector`.
2. **Optimize Browser**: Implement vision-guided navigation.
3. **Benchmark**: Submit results to [GAIA Leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard).

---

By following this guide, you’ve replicated key components of OpenAI’s Deep Research using open-source tools. For updates, star the [smolagents repo](https://github.com/huggingface/smolagents) and join the Hugging Face community! 🚀