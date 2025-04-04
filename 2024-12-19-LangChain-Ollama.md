---
layout: post
title:  High-Level Architecture for the LangChain Application using Ollama
date:   2024-12-27 07:42:44 -0500
---


![Alt text](/static/images/langchain.png){:loading="lazy"}



**High-Level Architecture for the LangChain Application using Ollama:**

The application leverages a graph structure to manage and orchestrate interactions with a Language Model (LLM) using LangChain and Ollama. The key components and their interactions are:

1. **Graph Manager:**
   - *Purpose:* Manages a directed graph where each node represents an LLM prompt and its corresponding response.
   - *Implementation:* Utilizes a graph data structure (e.g., from the `networkx` library) to model nodes (prompts and responses) and edges (data flow between prompts).

2. **Persona Manager:**
   - *Purpose:* Handles different personas, each providing unique perspectives or areas of knowledge.
   - *Implementation:* Defines personas as configurations or templates that tailor prompts to reflect specific viewpoints.

3. **Context Manager:**
   - *Purpose:* Manages the context passed between LLM calls, ensuring each prompt is aware of relevant previous interactions.
   - *Implementation:* Accumulates and updates context based on the graph's edges, feeding necessary information to subsequent prompts.

4. **LLM Interface (via LangChain and Ollama):**
   - *Purpose:* Facilitates interactions with the LLM, generating responses to prompts with the given context and persona.
   - *Implementation:* Uses LangChain's `LLMChain` and `PromptTemplate`, with the `Ollama` LLM wrapper to construct and execute prompts.

5. **Markdown Logger:**
   - *Purpose:* Records all prompts, responses, and analyses in a structured markdown file for tracking and reviewing.
   - *Implementation:* Appends entries to a markdown file, formatting the content for readability and organization.

6. **Analysis Module:**
   - *Purpose:* Analyzes previous prompts and responses, potentially generating new insights or directing the flow of the conversation.
   - *Implementation:* Creates specialized nodes in the graph that process and reflect on prior interactions.

---

**Implementing the Application with Ollama:**

Below is a step-by-step guide to building the application using Ollama, including code snippets and explanations.

### **1. Set Up the Environment**

#### **Install the Necessary Python Libraries:**

Ensure you have Python installed (preferably 3.7 or higher), and then install the required packages:

```bash
pip install langchain networkx markdown
```

#### **Install Ollama:**

Ollama is a tool for running language models locally. Follow the installation instructions for your operating system:

- **macOS:**

  ```bash
  brew install ollama/tap/ollama
  ```

- **Linux and Windows:**

  Visit the [Ollama GitHub repository](https://github.com/jmorganca/ollama) for installation instructions specific to your platform.

#### **Download a Model for Ollama:**

Ollama can run various models. For this application, we'll use `llama2` or any compatible model.

```bash
ollama pull llama2
```

### **2. Import Required Modules**

```python
import os
import networkx as nx
from langchain import PromptTemplate, LLMChain
from langchain.llms import Ollama
```

### **3. Define the Node Class**

Create a class to encapsulate the properties of each node in the graph:

```python
class Node:
    def __init__(self, node_id, prompt_text, persona):
        self.id = node_id
        self.prompt_text = prompt_text
        self.response_text = None
        self.context = ""
        self.persona = persona
```

### **4. Initialize the Graph**

Initialize a directed graph using `networkx`:

```python
G = nx.DiGraph()
```

### **5. Define Personas**

Create a dictionary to hold different personas and their corresponding system prompts:

```python
personas = {
    "Historian": "You are a knowledgeable historian specializing in the industrial revolution.",
    "Scientist": "You are a scientist with expertise in technological advancements.",
    "Philosopher": "You are a philosopher pondering the societal impacts.",
    "Analyst": "You analyze information critically to provide insights.",
    # Add additional personas as needed
}
```

### **6. Implement the Graph Manager**

Add nodes and edges to construct the conversation flow:

```python
# Create initial prompt nodes with different personas
node1 = Node(1, prompt_text="Discuss the impacts of the industrial revolution.", persona="Historian")
G.add_node(node1.id, data=node1)

node2 = Node(2, prompt_text="Discuss the technological advancements during the industrial revolution.", persona="Scientist")
G.add_node(node2.id, data=node2)

# Add edges if node2 should consider node1's context
G.add_edge(node1.id, node2.id)

# Add an analysis node
node3 = Node(3, prompt_text="", persona="Analyst")
G.add_node(node3.id, data=node3)
G.add_edge(node1.id, node3.id)
G.add_edge(node2.id, node3.id)
```

### **7. Implement the Context Manager**

Define a function to collect context from predecessor nodes:

```python
def collect_context(node_id):
    predecessors = list(G.predecessors(node_id))
    context = ""
    for pred_id in predecessors:
        pred_node = G.nodes[pred_id]['data']
        if pred_node.response_text:
            context += f"From {pred_node.persona}:\n{pred_node.response_text}\n\n"
    return context
```

### **8. Implement the LLM Interface with Ollama**

Create a function to generate responses using LangChain and Ollama:

```python
def generate_response(node):
    system_prompt = personas[node.persona]
    # Build the complete prompt
    prompt_template = PromptTemplate(
        input_variables=["system_prompt", "context", "prompt"],
        template="{system_prompt}\n\n{context}\n\n{prompt}"
    )
    # Instantiate the Ollama LLM
    llm = Ollama(
        base_url="http://localhost:11434",  # Default Ollama server URL
        model="llama2",  # or specify the model you have downloaded
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(
        system_prompt=system_prompt,
        context=node.context,
        prompt=node.prompt_text
    )
    return response
```

#### **Note:** Ensure that the Ollama server is running before executing the script:

```bash
ollama serve
```

### **9. Implement the Markdown Logger**

Define a function to log interactions to a markdown file:

```python
def update_markdown(node):
    with open("conversation.md", "a", encoding="utf-8") as f:
        f.write(f"## Node {node.id}: {node.persona}\n\n")
        f.write(f"**Prompt:**\n\n{node.prompt_text}\n\n")
        f.write(f"**Response:**\n\n{node.response_text}\n\n---\n\n")
```

### **10. Implement the Analysis Module**

Create a function for nodes that perform analysis:

```python
def analyze_responses(node):
    # Collect responses from predecessor nodes
    predecessors = list(G.predecessors(node.id))
    analysis_input = ""
    for pred_id in predecessors:
        pred_node = G.nodes[pred_id]['data']
        analysis_input += f"{pred_node.persona}'s response:\n{pred_node.response_text}\n\n"

    node.prompt_text = f"Provide an analysis comparing the following perspectives:\n\n{analysis_input}"
    node.context = ""  # Analysis can be based solely on the provided responses
    node.response_text = generate_response(node)
    update_markdown(node)
```

### **11. Process the Nodes**

Iterate over the graph to process each node:

```python
for node_id in nx.topological_sort(G):
    node = G.nodes[node_id]['data']
    if node.persona != "Analyst":
        node.context = collect_context(node_id)
        node.response_text = generate_response(node)
        update_markdown(node)
    else:
        analyze_responses(node)
```

### **Detailed Explanation:**

- **Graph Processing Order:**
  - Use `nx.topological_sort(G)` to process nodes in an order that respects dependencies, ensuring predecessor nodes are processed before successors.

- **Context Collection:**
  - For each node, the `collect_context` function gathers responses from predecessor nodes, forming the context that will be included in the prompt.

- **Persona-Specific Prompts:**
  - The `system_prompt` variable injects persona characteristics into the prompt via LangChain's templating, guiding the LLM to respond from that perspective.

- **Response Generation with Ollama:**
  - The `generate_response` function constructs the prompt using the `PromptTemplate` and retrieves the LLM's response using LangChain's `LLMChain` with the `Ollama` LLM.

- **Logging Interactions:**
  - The `update_markdown` function appends each interaction to the `conversation.md` file, using markdown formatting for clarity and organization.

- **Analysis Nodes:**
  - Nodes with the persona "Analyst" execute the `analyze_responses` function, which compiles predecessor responses and generates an analytical output.

### **Example Output in Markdown:**

The `conversation.md` file will contain formatted entries like:

```
## Node 1: Historian

**Prompt:**

Discuss the impacts of the industrial revolution.

**Response:**

[Historian's response...]

---

## Node 2: Scientist

**Prompt:**

Discuss the technological advancements during the industrial revolution.

**Response:**

[Scientist's response...]

---

## Node 3: Analyst

**Prompt:**

Provide an analysis comparing the following perspectives:
...

**Response:**

[Analyst's comparative analysis...]

---
```

### **12. Expanding the Application**

To enhance the application further:

- **Dynamic Node Creation:**
  - Based on responses, new nodes can be added to explore emerging topics.

- **Advanced Personas:**
  - Enrich personas with more detailed backgrounds or expertise.

- **User Interaction:**
  - Introduce mechanisms for user input to guide the conversation.

- **Visualization:**
  - Generate visual representations of the graph to illustrate the conversation flow.

### **13. Considerations and Best Practices**

- **Ollama Model Selection:**
  - Ensure that the model used with Ollama is appropriate for the application's needs. Some models may require specific handling or have different capabilities.

- **Context Window Limitations:**
  - Be mindful of the token limit for the LLM's context window; if necessary, truncate or summarize context.

- **Error Handling:**
  - Implement robust error handling around LLM calls and file operations to handle exceptions gracefully.

- **Concurrency:**
  - For large graphs, consider asynchronous processing where dependencies allow.

- **Configuration Management:**
  - Use configuration files or environment variables to manage settings like the Ollama server URL and model name.

- **Privacy and Security:**
  - Ensure sensitive information is not exposed, especially when logging prompts and responses.

---

**Summary:**

By integrating these components with Ollama, the application can:

- **Orchestrate LLM Calls via a Graph:**
  - Manage complex conversational flows where prompts and responses are interconnected in non-linear ways.

- **Update Context Dynamically:**
  - Pass information between nodes, ensuring that each prompt is informed by relevant preceding interactions.

- **Utilize Multiple Personas:**
  - Simulate different perspectives by tailoring prompts to various personas, enriching the conversation.

- **Track Interaction History:**
  - Maintain a comprehensive record of the conversation, including analyses, in a markdown file for transparency and review.

- **Analyze and Reflect:**
  - Incorporate analysis steps that synthesize previous responses, potentially guiding future prompts.

**Implementation Steps Recap:**

1. **Set Up Environment and Libraries:**
   - Install `langchain`, `networkx`, `markdown`, and set up Ollama.

2. **Define Data Structures (Nodes and Edges):**
   - Create the `Node` class to represent each point in the conversation.

3. **Initialize the Directed Graph:**
   - Use `networkx` to manage the flow of conversations.

4. **Define Personas and Their Prompts:**
   - Establish different perspectives through personas.

5. **Build the Graph Manager Functions:**
   - Construct the conversation flow by adding nodes and edges.

6. **Implement Context Collection Mechanism:**
   - Gather context from predecessor nodes for each prompt.

7. **Create the LLM Interface with Ollama:**
   - Use LangChain's `Ollama` integration to interface with the local LLM.

8. **Set Up the Markdown Logger:**
   - Record the prompts and responses in a markdown file.

9. **Develop the Analysis Module:**
   - Analyze previous responses to generate insights.

10. **Process Nodes in Topological Order:**
    - Execute the conversation flow respecting dependencies.

By following this architecture and implementation plan with Ollama, you can create a robust application that leverages the power of LangChain and local LLMs to generate rich, context-aware conversations from multiple perspectives, all while maintaining a clear and organized record of the interaction history.

---

**Next Steps:**

- **Testing:**
  - Run the application with sample prompts and personas to verify functionality.

- **Refinement:**
  - Adjust personas, context management, and logging based on observed outcomes.

- **Scaling:**
  - Expand the graph to include more nodes and complex interactions, testing the application's scalability.

- **Documentation:**
  - Document the code thoroughly, explaining how each component works for future maintenance and updates.

- **Model Optimization:**
  - Experiment with different models available in Ollama to find the best fit for your application.

By iteratively refining the application, you can tailor it to specific use cases, such as educational tools, collaborative brainstorming platforms, or complex simulation environments, all powered locally using Ollama.

---

**Additional Resources:**

- **Ollama Documentation:**
  - Visit the [Ollama GitHub Repository](https://github.com/jmorganca/ollama) for more details on installation, models, and usage.

- **LangChain Documentation:**
  - Explore the [LangChain Documentation](https://langchain.readthedocs.io/) for in-depth guides on using LLMs, prompt templates, and chains.

- **Community Support:**
  - Engage with the communities around LangChain and Ollama for support, updates, and shared experiences.

---

**Troubleshooting Tips:**

- **Ollama Server Not Running:**
  - If you encounter connection errors, ensure the Ollama server is running with `ollama serve`.

- **Model Not Found:**
  - Verify that the model specified in the `Ollama` LLM instantiation is correctly downloaded and available.

- **Performance Issues:**
  - Running large models locally may require significant computational resources. Ensure your hardware meets the requirements.

- **Compatibility:**
  - Ensure all libraries are up-to-date to avoid compatibility issues. Use virtual environments to manage dependencies.

---



---

### **1. Create the Project Directory and File Structure**

Open your terminal and run the following commands to set up the project directory and files:

```bash
# Create the project directory and navigate into it
mkdir langchain_graph_app
cd langchain_graph_app

# Create the main Python script
touch main.py

# Create a requirements.txt file to list dependencies
touch requirements.txt
```

---

### **2. Write the Code for the Files**

I'll provide the code for each file. Please copy and paste the code into the respective files.

---

#### **File: `requirements.txt`**

First, specify the dependencies in a `requirements.txt` file. This will allow you to install all the necessary Python packages easily.

**Content of `requirements.txt`:**

```
langchain
networkx
markdown
langchain_community
```

---

#### **File: `main.py`**

This is the main script containing the code for the application.

**Content of `main.py`:**

```python
# main.py

import os
import networkx as nx
from langchain import PromptTemplate, LLMChain
from langchain.llms import Ollama

# Define the Node class
class Node:
    def __init__(self, node_id, prompt_text, persona):
        self.id = node_id
        self.prompt_text = prompt_text
        self.response_text = None
        self.context = ""
        self.persona = persona

# Initialize the graph
G = nx.DiGraph()

# Define personas
personas = {
    "Historian": "You are a knowledgeable historian specializing in the industrial revolution.",
    "Scientist": "You are a scientist with expertise in technological advancements.",
    "Philosopher": "You are a philosopher pondering societal impacts.",
    "Analyst": "You analyze information critically to provide insights.",
    # Add additional personas as needed
}

# Function to collect context from predecessor nodes
def collect_context(node_id):
    predecessors = list(G.predecessors(node_id))
    context = ""
    for pred_id in predecessors:
        pred_node = G.nodes[pred_id]['data']
        if pred_node.response_text:
            context += f"From {pred_node.persona}:\n{pred_node.response_text}\n\n"
    return context

# Function to generate responses using LangChain and Ollama
def generate_response(node):
    system_prompt = personas[node.persona]
    # Build the complete prompt
    prompt_template = PromptTemplate(
        input_variables=["system_prompt", "context", "prompt"],
        template="{system_prompt}\n\n{context}\n\n{prompt}"
    )
    # Instantiate the Ollama LLM
    llm = Ollama(
        base_url="http://localhost:11434",  # Default Ollama server URL
        model="llama2",  # Replace with the model you have downloaded
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(
        system_prompt=system_prompt,
        context=node.context,
        prompt=node.prompt_text
    )
    return response

# Function to log interactions to a markdown file
def update_markdown(node):
    with open("conversation.md", "a", encoding="utf-8") as f:
        f.write(f"## Node {node.id}: {node.persona}\n\n")
        f.write(f"**Prompt:**\n\n{node.prompt_text}\n\n")
        f.write(f"**Response:**\n\n{node.response_text}\n\n---\n\n")

# Function for nodes that perform analysis
def analyze_responses(node):
    # Collect responses from predecessor nodes
    predecessors = list(G.predecessors(node.id))
    analysis_input = ""
    for pred_id in predecessors:
        pred_node = G.nodes[pred_id]['data']
        analysis_input += f"{pred_node.persona}'s response:\n{pred_node.response_text}\n\n"

    node.prompt_text = f"Provide an analysis comparing the following perspectives:\n\n{analysis_input}"
    node.context = ""  # Analysis is based solely on the provided responses
    node.response_text = generate_response(node)
    update_markdown(node)

# Build the graph

# Create initial prompt nodes with different personas
node1 = Node(1, prompt_text="Discuss the impacts of the industrial revolution.", persona="Historian")
G.add_node(node1.id, data=node1)

node2 = Node(2, prompt_text="Discuss the technological advancements during the industrial revolution.", persona="Scientist")
G.add_node(node2.id, data=node2)

# Add edges if node2 should consider node1's context
G.add_edge(node1.id, node2.id)  # node2 considers node1's context

# Add an analysis node
node3 = Node(3, prompt_text="", persona="Analyst")
G.add_node(node3.id, data=node3)
G.add_edge(node1.id, node3.id)
G.add_edge(node2.id, node3.id)

# Process the nodes
for node_id in nx.topological_sort(G):
    node = G.nodes[node_id]['data']
    if node.persona != "Analyst":
        node.context = collect_context(node_id)
        node.response_text = generate_response(node)
        update_markdown(node)
    else:
        analyze_responses(node)

print("Conversation has been generated and logged to conversation.md")
```

---

### **3. Install Dependencies**

Make sure you have all the required Python packages installed.

In your terminal, from within the `langchain_graph_app` directory, run:

```bash
pip install -r requirements.txt
```

---

### **4. Install and Set Up Ollama**

#### **Install Ollama**

Follow the installation instructions specific to your operating system.

- **macOS (via Homebrew):**

  ```bash
  brew install ollama/tap/ollama
  ```

- **Other Platforms:**

  Visit the [Ollama GitHub repository](https://github.com/jmorganca/ollama) for installation instructions.

#### **Download a Model for Ollama**

Ollama runs models locally. You need to download a model compatible with your application.

For example, to download the `llama2` model:

```bash
ollama pull llama2
```

**Note:** Replace `llama2` with the name of the model you wish to use if different.

---

### **5. Run the Ollama Server**

Before running the application, ensure that the Ollama server is running.

In a separate terminal window, run:

```bash
ollama serve
```

This will start the Ollama server on the default port `11434`.

---

### **6. Run the Application**

Now you can run the application:

```bash
python main.py
```

This will execute the script, generate the conversation, and create (or update) the `conversation.md` file with the prompts and responses.

---

### **7. View the Output**

Open the `conversation.md` file to see the generated conversation:

```bash
cat conversation.md
```

Or open it in a text editor that supports Markdown to view it with proper formatting.

---

## **Explanation of the Files and Code**

### **`main.py` Overview**

- **Imports:**

  - `networkx`: For creating and managing the directed graph.
  - `langchain`: For interacting with the language model.
  - `Ollama`: LangChain's wrapper for the Ollama LLM.

- **Node Class:**

  - Represents each node in the graph.
  - Stores the node ID, prompt text, persona, response text, and context.

- **Graph Initialization:**

  - A directed graph `G` is created using `networkx.DiGraph()`.

- **Personas:**

  - A dictionary mapping persona names to their descriptions (system prompts).
  - Personas influence how the LLM generates responses.

- **Functions:**

  - `collect_context(node_id)`: Gathers responses from predecessor nodes to form the context.
  - `generate_response(node)`: Uses the `Ollama` LLM via LangChain to generate a response based on the persona, context, and prompt.
  - `update_markdown(node)`: Appends the prompt and response to `conversation.md` in a structured format.
  - `analyze_responses(node)`: Handles nodes designated for analysis, compiling predecessor responses and generating an analytical output.

- **Graph Construction:**

  - Nodes are added to the graph with unique IDs and associated data.
  - Edges define the flow of information (i.e., which nodes' contexts are considered in generating responses).

- **Processing Nodes:**

  - Nodes are processed in topological order to respect dependencies.
  - Responses are generated and logged for each node.

---

### **Understanding the Flow**

1. **Graph Creation:**

   - Three nodes are created: two with specific personas (`Historian` and `Scientist`) and one `Analyst`.
   - Edges are added to define how context flows between nodes.

2. **Processing:**

   - Nodes are processed so that predecessors are handled before successors.
   - For non-analyst nodes, the context is collected from predecessors, responses are generated, and the interaction is logged.
   - For the analyst node, it compiles the responses from its predecessors and generates an analysis.

3. **Logging:**

   - All prompts and responses are appended to `conversation.md` with markdown formatting for readability.

---

## **Sample Output in `conversation.md`**

The `conversation.md` file will contain entries like:

```
## Node 1: Historian

**Prompt:**

Discuss the impacts of the industrial revolution.

**Response:**

[Historian's response...]

---

## Node 2: Scientist

**Prompt:**

Discuss the technological advancements during the industrial revolution.

**Response:**

[Scientist's response...]

---

## Node 3: Analyst

**Prompt:**

Provide an analysis comparing the following perspectives:

Historian's response:
[Historian's response...]

Scientist's response:
[Scientist's response...]

**Response:**

[Analyst's comparative analysis...]

---
```

---

## **Additional Notes**

- **Model Configuration:**

  - Ensure the `model` parameter in the `Ollama` LLM instantiation matches the model you have downloaded.
  - Example: If you downloaded `llama2`, set `model="llama2"`.

- **Expanding the Graph:**

  - You can add more nodes and edges to create more complex conversations.
  - Be mindful of the context window limitations of the LLM.

- **Error Handling:**

  - The script does not include extensive error handling.
  - Consider adding try-except blocks around network calls, file operations, and LLM interactions.

- **Environment:**

  - It's recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate
    ```

- **Dependencies:**

  - Ensure all dependencies are properly installed and compatible with your Python interpreter.

---

## **Next Steps**

- **Customize Personas and Prompts:**

  - Modify the `personas` dictionary to add or adjust personas.
  - Change the `prompt_text` for each node to explore different topics.

- **Enhance Functionality:**

  - Add user input to create dynamic prompts or personas.
  - Implement a GUI or web interface for interactive usage.

- **Visualization:**

  - Use graph visualization libraries, like `matplotlib` or `pygraphviz`, to visualize the conversation flow.

- **Documentation and Testing:**

  - Document any changes or additions you make to the code.
  - Write unit tests for your functions to ensure reliability.

---

## **Recap of Commands**

### **Project Setup**

```bash
# Create project directory and navigate into it
mkdir langchain_graph_app
cd langchain_graph_app

# Create the main script and requirements file
touch main.py requirements.txt

# Open the files in a text editor to add the provided code
```

### **Installing Dependencies**

```bash
# Install Python packages
pip install -r requirements.txt
```

### **Installing and Running Ollama**

```bash
# Install Ollama (if not already installed)
# For macOS:
brew install ollama/tap/ollama

# Download the desired model
ollama pull llama2

# Start the Ollama server
ollama serve
```

### **Running the Application**

```bash
# Run the main script
python main.py

# View the generated conversation
cat conversation.md
```

---

## **Troubleshooting Tips**

- **Ollama Server Issues:**

  - Ensure the Ollama server is running (`ollama serve`) before running the script.
  - Check if the server is accessible at `http://localhost:11434`.

- **Model Not Found:**

  - Verify that the model name in the script matches the model you downloaded.
  - Run `ollama list` to see the available models.

- **Dependency Errors:**

  - Double-check that all dependencies are installed.
  - Use `pip list` to view installed packages.

- **Python Version:**

  - Ensure you're using a compatible Python version (3.7 or higher recommended).

- **Permission Issues:**

  - If you encounter permission errors when accessing files, ensure you have the necessary rights.

---

## **Additional Resources**

- **Ollama Documentation:**

  - [Ollama GitHub Repository](https://github.com/jmorganca/ollama)

- **LangChain Documentation:**

  - [LangChain Documentation](https://langchain.readthedocs.io/)

- **NetworkX Documentation:**

  - [NetworkX Documentation](https://networkx.org/documentation/stable/)

---

Let's extend the application to include detailed personas with stylistic attributes stored in JSON files. We'll modify the program to select a persona from a JSON file and adjust the LLM prompts to reflect the selected persona's writing style.

---

## **Overview of the Enhancement**

- **Personas with Stylistic Attributes:**
  - Store detailed personas in JSON format.
  - Each persona includes various stylistic and psychological attributes.

- **Integration with the Application:**
  - Load personas from JSON files.
  - Modify the prompt generation to include these attributes.
  - Adjust the LLM output to match the selected persona's style.

- **Implementation Steps:**
  1. **Update the File Structure:**
     - Create a `personas` directory to store JSON files.
  2. **Modify the Code:**
     - Load personas from JSON files.
     - Update the `generate_response` function to incorporate stylistic attributes.
  3. **Provide Instructions:**
     - Explain how to add new personas.
     - Show how to use the updated application.

---

## **1. Update the File Structure**

In your project directory (`langchain_graph_app`), create a new directory called `personas` to store persona JSON files.

```bash
mkdir personas
```

---

## **2. Create Persona JSON Files**

Each persona will be stored as an individual JSON file within the `personas` directory. Let's create a sample persona.

### **Example Persona: "Ernest Hemingway"**

Create a file named `ernest_hemingway.json` in the `personas` directory.

**Content of `personas/ernest_hemingway.json`:**

```json
{
  "name": "Ernest Hemingway",
  "vocabulary_complexity": 3,
  "sentence_structure": "simple",
  "paragraph_organization": "structured",
  "idiom_usage": 2,
  "metaphor_frequency": 4,
  "simile_frequency": 5,
  "tone": "formal",
  "punctuation_style": "minimal",
  "contraction_usage": 5,
  "pronoun_preference": "first-person",
  "passive_voice_frequency": 2,
  "rhetorical_question_usage": 3,
  "list_usage_tendency": 2,
  "personal_anecdote_inclusion": 7,
  "pop_culture_reference_frequency": 1,
  "technical_jargon_usage": 2,
  "parenthetical_aside_frequency": 1,
  "humor_sarcasm_usage": 3,
  "emotional_expressiveness": 6,
  "emphatic_device_usage": 4,
  "quotation_frequency": 3,
  "analogy_usage": 5,
  "sensory_detail_inclusion": 8,
  "onomatopoeia_usage": 2,
  "alliteration_frequency": 2,
  "word_length_preference": "short",
  "foreign_phrase_usage": 3,
  "rhetorical_device_usage": 4,
  "statistical_data_usage": 1,
  "personal_opinion_inclusion": 7,
  "transition_usage": 5,
  "reader_question_frequency": 2,
  "imperative_sentence_usage": 3,
  "dialogue_inclusion": 8,
  "regional_dialect_usage": 4,
  "hedging_language_frequency": 2,
  "language_abstraction": "concrete",
  "personal_belief_inclusion": 6,
  "repetition_usage": 5,
  "subordinate_clause_frequency": 3,
  "verb_type_preference": "active",
  "sensory_imagery_usage": 8,
  "symbolism_usage": 6,
  "digression_frequency": 2,
  "formality_level": 5,
  "reflection_inclusion": 7,
  "irony_usage": 3,
  "neologism_frequency": 1,
  "ellipsis_usage": 2,
  "cultural_reference_inclusion": 3,
  "stream_of_consciousness_usage": 2,
  "openness_to_experience": 8,
  "conscientiousness": 6,
  "extraversion": 5,
  "agreeableness": 7,
  "emotional_stability": 6,
  "dominant_motivations": "adventure",
  "core_values": "courage",
  "decision_making_style": "intuitive",
  "empathy_level": 7,
  "self_confidence": 8,
  "risk_taking_tendency": 9,
  "idealism_vs_realism": "realistic",
  "conflict_resolution_style": "assertive",
  "relationship_orientation": "independent",
  "emotional_response_tendency": "calm",
  "creativity_level": 8,
  "age": "Late 50s",
  "gender": "Male",
  "education_level": "High School",
  "professional_background": "Writer and Journalist",
  "cultural_background": "American",
  "primary_language": "English",
  "language_fluency": "Native"
}
```

---

## **3. Modify `main.py` to Load and Use Personas**

### **3.1. Import JSON and OS Modules**

Add imports at the top of `main.py`:

```python
import json
import os
```

### **3.2. Update the `Node` Class**

Modify the `Node` class to include a `persona_attributes` field:

```python
class Node:
    def __init__(self, node_id, prompt_text, persona_name):
        self.id = node_id
        self.prompt_text = prompt_text
        self.response_text = None
        self.context = ""
        self.persona_name = persona_name
        self.persona_attributes = {}
```

### **3.3. Load Personas from JSON Files**

Create a function to load personas from the `personas` directory:

```python
def load_personas(persona_dir):
    personas = {}
    for filename in os.listdir(persona_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(persona_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                persona_data = json.load(f)
                name = persona_data.get('name')
                if name:
                    personas[name] = persona_data
    return personas
```

Load the personas after defining the function:

```python
# Load personas
persona_dir = 'personas'  # Directory where persona JSON files are stored
personas = load_personas(persona_dir)
```

### **3.4. Update the `generate_response` Function**

Modify the `generate_response` function to include persona attributes in the system prompt:

```python
def generate_response(node):
    persona = personas.get(node.persona_name)
    if not persona:
        raise ValueError(f"Persona '{node.persona_name}' not found.")
    
    node.persona_attributes = persona

    # Build the system prompt based on persona attributes
    system_prompt = build_system_prompt(persona)

    # Build the complete prompt
    prompt_template = PromptTemplate(
        input_variables=["system_prompt", "context", "prompt"],
        template="{system_prompt}\n\n{context}\n\n{prompt}"
    )
    # Instantiate the Ollama LLM
    llm = Ollama(
        base_url="http://localhost:11434",  # Default Ollama server URL
        model="llama2",  # Replace with the model you have downloaded
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(
        system_prompt=system_prompt,
        context=node.context,
        prompt=node.prompt_text
    )
    return response
```

### **3.5. Create the `build_system_prompt` Function**

This function constructs the system prompt using the persona's attributes.

```python
def build_system_prompt(persona):
    # Construct descriptive sentences based on persona attributes
    # We'll focus on key attributes for brevity
    name = persona.get('name', 'The speaker')
    tone = persona.get('tone', 'neutral')
    sentence_structure = persona.get('sentence_structure', 'varied')
    vocabulary_complexity = persona.get('vocabulary_complexity', 5)
    formality_level = persona.get('formality_level', 5)
    pronoun_preference = persona.get('pronoun_preference', 'third-person')
    language_abstraction = persona.get('language_abstraction', 'mixed')

    # Create a description
    description = (
        f"You are {name}, writing in a {tone} tone using {sentence_structure} sentences. "
        f"Your vocabulary complexity is {vocabulary_complexity}/10, and your formality level is {formality_level}/10. "
        f"You prefer {pronoun_preference} narration and your language abstraction is {language_abstraction}."
    )

    # Include any other attributes as needed
    # ...

    return description
```

### **3.6. Update the Graph Definition**

Update the creation of nodes to use the new persona names.

For example, replace the old personas with the new ones:

```python
# Create initial prompt nodes with different personas
node1 = Node(1, prompt_text="Discuss the impacts of the industrial revolution.", persona_name="Ernest Hemingway")
G.add_node(node1.id, data=node1)

# You can create more nodes with different personas
node2 = Node(2, prompt_text="Explain quantum mechanics in simple terms.", persona_name="Albert Einstein")
G.add_node(node2.id, data=node2)

# Add edges between nodes if needed
# G.add_edge(node1.id, node2.id)

# Add an analysis node (you can define a generic analyst persona)
node3 = Node(3, prompt_text="", persona_name="Analyst")
G.add_node(node3.id, data=node3)
G.add_edge(node1.id, node3.id)
G.add_edge(node2.id, node3.id)
```

### **3.7. Create an Analyst Persona**

Create an analyst persona JSON file `analyst.json` or handle the analyst within the code.

**Option 1:** Create `personas/analyst.json`

```json
{
  "name": "Analyst",
  "tone": "formal",
  "sentence_structure": "complex",
  "vocabulary_complexity": 7,
  "formality_level": 8,
  "pronoun_preference": "third-person",
  "language_abstraction": "mixed"
}
```

**Option 2:** If the analyst persona is generic, handle the absence of specific attributes in `build_system_prompt` by providing defaults.

---

## **4. Run the Updated Application**

### **4.1. Ensure All Persona Files are Created**

Make sure you've created the necessary persona JSON files in the `personas` directory.

- `ernest_hemingway.json` (as above)
- `analyst.json` (if needed)
- Additional personas (e.g., `albert_einstein.json` if you use that persona)

### **4.2. Install Any New Dependencies**

If you haven't already imported the `json` module, ensure it's included in your code. No additional installations are required as `json` and `os` are part of the Python standard library.

### **4.3. Run the Application**

```bash
python main.py
```

This will generate the conversation with responses tailored to the selected personas.

---

## **5. View the Output**

Check the `conversation.md` file to see the prompts and responses with the new personas.

---

## **6. Additional Personas**

To add more personas, create new JSON files in the `personas` directory with the required attributes.

### **Example: "Albert Einstein" Persona**

Create `personas/albert_einstein.json`:

```json
{
  "name": "Albert Einstein",
  "vocabulary_complexity": 8,
  "sentence_structure": "complex",
  "paragraph_organization": "structured",
  "idiom_usage": 3,
  "metaphor_frequency": 6,
  "simile_frequency": 5,
  "tone": "academic",
  "punctuation_style": "conventional",
  "contraction_usage": 2,
  "pronoun_preference": "first-person",
  "passive_voice_frequency": 6,
  "rhetorical_question_usage": 4,
  "list_usage_tendency": 3,
  "personal_anecdote_inclusion": 5,
  "technical_jargon_usage": 9,
  "parenthetical_aside_frequency": 2,
  "humor_sarcasm_usage": 4,
  "emotional_expressiveness": 5,
  "emphatic_device_usage": 6,
  "quotation_frequency": 3,
  "analogy_usage": 7,
  "sensory_detail_inclusion": 4,
  "onomatopoeia_usage": 1,
  "alliteration_frequency": 2,
  "word_length_preference": "long",
  "foreign_phrase_usage": 5,
  "rhetorical_device_usage": 7,
  "statistical_data_usage": 8,
  "personal_opinion_inclusion": 6,
  "transition_usage": 7,
  "reader_question_frequency": 2,
  "imperative_sentence_usage": 2,
  "dialogue_inclusion": 3,
  "regional_dialect_usage": 1,
  "hedging_language_frequency": 5,
  "language_abstraction": "abstract",
  "personal_belief_inclusion": 6,
  "repetition_usage": 3,
  "subordinate_clause_frequency": 7,
  "verb_type_preference": "active",
  "sensory_imagery_usage": 3,
  "symbolism_usage": 5,
  "digression_frequency": 2,
  "formality_level": 8,
  "reflection_inclusion": 7,
  "irony_usage": 2,
  "neologism_frequency": 3,
  "ellipsis_usage": 2,
  "cultural_reference_inclusion": 3,
  "stream_of_consciousness_usage": 2,
  "openness_to_experience": 9,
  "conscientiousness": 7,
  "extraversion": 4,
  "agreeableness": 6,
  "emotional_stability": 7,
  "dominant_motivations": "knowledge",
  "core_values": "integrity",
  "decision_making_style": "analytical",
  "empathy_level": 7,
  "self_confidence": 8,
  "risk_taking_tendency": 6,
  "idealism_vs_realism": "idealistic",
  "conflict_resolution_style": "collaborative",
  "relationship_orientation": "independent",
  "emotional_response_tendency": "calm",
  "creativity_level": 9,
  "age": "Mid 40s",
  "gender": "Male",
  "education_level": "Doctorate",
  "professional_background": "Physicist",
  "cultural_background": "German-American",
  "primary_language": "German",
  "language_fluency": "Fluent in English"
}
```

---

## **7. Testing and Refinement**

### **7.1. Test the Application**

Run the application again and observe the differences in the responses based on the personas.

### **7.2. Refine the `build_system_prompt` Function**

You can enhance the `build_system_prompt` function to incorporate more attributes and create more nuanced system prompts.

For example:

```python
def build_system_prompt(persona):
    attributes = []

    # Add tone
    tone = persona.get('tone')
    if tone:
        attributes.append(f"Your tone is {tone}.")

    # Add vocabulary complexity
    vocab_complexity = persona.get('vocabulary_complexity')
    if vocab_complexity:
        attributes.append(f"Your vocabulary complexity is rated {vocab_complexity}/10.")

    # Add sentence structure
    sentence_structure = persona.get('sentence_structure')
    if sentence_structure:
        attributes.append(f"You use {sentence_structure} sentence structures.")

    # Add more attributes as needed
    # ...

    description = ' '.join(attributes)
    return f"You are {persona.get('name', 'a speaker')}. {description}"
```

### **7.3. Adjusting Prompts for LLM**

Ensure that the system prompt is concise but informative. Overly long prompts may exceed the LLM's context window or lead to less coherent responses.

---

## **8. Considerations and Best Practices**

- **LLM Limitations:**
  - The LLM's ability to mimic detailed stylistic attributes may vary.
  - Some attributes may have a more pronounced effect on the output than others.

- **Prompt Engineering:**
  - Experiment with how you convey persona attributes in the prompt.
  - You may need to adjust wording to achieve the desired effect.

- **Performance:**
  - Loading many large persona files might impact performance.
  - Consider optimizing persona file sizes if necessary.

- **Error Handling:**
  - Add error handling for missing attributes or files.
  - Validate persona data when loading.

---

## **9. Summary of Changes**

- **File Structure:**
  - Added a `personas` directory containing JSON files for each persona.

- **Code Modifications:**
  - Added functions to load personas from JSON files.
  - Updated the `Node` class to handle persona attributes.
  - Modified `generate_response` to include persona attributes in system prompts.
  - Created `build_system_prompt` to construct system prompts from persona attributes.
  - Adjusted node creation to reference personas by name from the JSON files.

- **Usage:**
  - Personas can now be added or modified by editing the JSON files.
  - The application uses these personas to tailor LLM outputs accordingly.

---

## **10. Next Steps**

- **Enhance Persona Attributes Handling:**
  - Implement more sophisticated mapping between persona attributes and system prompts.
  - Possibly use templates or mapping dictionaries to handle various attribute values.

- **Integrate with a Database (Optional):**
  - If you prefer using SQLite for storing personas, you can modify the code to load personas from a SQLite database instead of JSON files.

- **User Interface:**
  - Create a CLI or GUI to select personas and customize prompts.

- **Logging and Analysis:**
  - Enhance logging to include which attributes were applied.
  - Analyze how different attributes affect the generated responses.

---

## **11. Conclusion**

By incorporating detailed personas with stylistic attributes, the application can generate more personalized and varied responses from the LLM. This enhancement adds depth to the generated content and allows for experimentation with different writing styles and perspectives.

---

Debugged final main.py

```python
# main.py

import os
import networkx as nx
from langchain import PromptTemplate, LLMChain
from langchain.llms import Ollama
import json





# Define the Node class
class Node:
    def __init__(self, node_id, prompt_text, persona_name):
        self.id = node_id
        self.prompt_text = prompt_text
        self.response_text = None
        self.context = ""
        self.persona_name = persona_name
        self.persona_attributes = {}

# Initialize the graph
G = nx.DiGraph()

# Define personas
personas = {
    "Historian": "You are a knowledgeable historian specializing in the industrial revolution.",
    "Scientist": "You are a scientist with expertise in technological advancements.",
    "Philosopher": "You are a philosopher pondering societal impacts.",
    "Analyst": "You analyze information critically to provide insights.",
    # Add additional personas as needed
}

def load_personas(persona_dir):
    personas = {}
    for filename in os.listdir(persona_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(persona_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                persona_data = json.load(f)
                name = persona_data.get('name')
                if name:
                    personas[name] = persona_data
    return personas

# Load personas
persona_dir = 'personas'  # Directory where persona JSON files are stored
personas = load_personas(persona_dir)

# Function to collect context from predecessor nodes
def collect_context(node_id):
    predecessors = list(G.predecessors(node_id))
    context = ""
    for pred_id in predecessors:
        pred_node = G.nodes[pred_id]['data']
        if pred_node.response_text:
            context += f"From {pred_node.persona}:\n{pred_node.response_text}\n\n"
    return context

# Function to generate responses using LangChain and Ollama
def generate_response(node):
    persona = personas.get(node.persona_name)
    if not persona:
        raise ValueError(f"Persona '{node.persona_name}' not found.")
    
    node.persona_attributes = persona

    # Build the system prompt based on persona attributes
    system_prompt = build_system_prompt(persona)

    # Build the complete prompt
    prompt_template = PromptTemplate(
        input_variables=["system_prompt", "context", "prompt"],
        template="{system_prompt}\n\n{context}\n\n{prompt}"
    )
    # Instantiate the Ollama LLM
    llm = Ollama(
        base_url="http://localhost:11434",  # Default Ollama server URL
        model="qwq",  # Replace with the model you have downloaded
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(
        system_prompt=system_prompt,
        context=node.context,
        prompt=node.prompt_text
    )
    return response

def build_system_prompt(persona):
    # Construct descriptive sentences based on persona attributes
    # We'll focus on key attributes for brevity
    name = persona.get('name', 'The speaker')
    tone = persona.get('tone', 'neutral')
    sentence_structure = persona.get('sentence_structure', 'varied')
    vocabulary_complexity = persona.get('vocabulary_complexity', 5)
    formality_level = persona.get('formality_level', 5)
    pronoun_preference = persona.get('pronoun_preference', 'third-person')
    language_abstraction = persona.get('language_abstraction', 'mixed')

    # Create a description
    description = (
        f"You are {name}, writing in a {tone} tone using {sentence_structure} sentences. "
        f"Your vocabulary complexity is {vocabulary_complexity}/10, and your formality level is {formality_level}/10. "
        f"You prefer {pronoun_preference} narration and your language abstraction is {language_abstraction}."
    )

    # Include any other attributes as needed
    # ...

    return description


# Function to log interactions to a markdown file
def update_markdown(node):
    with open("conversation.md", "a", encoding="utf-8") as f:
        f.write(f"## Node {node.id}: {node.persona_name}\n\n")
        f.write(f"**Prompt:**\n\n{node.prompt_text}\n\n")
        f.write(f"**Response:**\n\n{node.response_text}\n\n---\n\n")

# Function for nodes that perform analysis
def analyze_responses(node):
    # Collect responses from predecessor nodes
    predecessors = list(G.predecessors(node.id))
    analysis_input = ""
    for pred_id in predecessors:
        pred_node = G.nodes[pred_id]['data']
        analysis_input += f"{pred_node.persona_name}'s response:\n{pred_node.response_text}\n\n"

    node.prompt_text = f"Provide an analysis comparing the following perspectives:\n\n{analysis_input}"
    node.context = ""  # Analysis is based solely on the provided responses
    node.response_text = generate_response(node)
    update_markdown(node)

# Build the graph

# Create initial prompt nodes with different personas
node1 = Node(1, prompt_text="Discuss the impacts of the industrial revolution.", persona_name="Ernest Hemingway")
G.add_node(node1.id, data=node1)

# You can create more nodes with different personas
node2 = Node(2, prompt_text="Explain quantum mechanics in simple terms.", persona_name="Albert Einstein")
G.add_node(node2.id, data=node2)

# Add edges between nodes if needed
# G.add_edge(node1.id, node2.id)

# Add an analysis node (you can define a generic analyst persona)
node3 = Node(3, prompt_text="", persona_name="Analyst")
G.add_node(node3.id, data=node3)
G.add_edge(node1.id, node3.id)
G.add_edge(node2.id, node3.id)

# Process the nodes
for node_id in nx.topological_sort(G):
    node = G.nodes[node_id]['data']
    if node.persona_name != "Analyst":
        node.context = collect_context(node_id)
        node.response_text = generate_response(node)
        update_markdown(node)
    else:
        analyze_responses(node)

print("Conversation has been generated and logged to conversation.md")
```


Given that a CLI is generally quicker to implement and can provide immediate utility, let's start with creating a CLI using the `argparse` or `click` library. After that, I'll provide guidance on how you can set up a GUI using **Streamlit**, which is a Python library that allows for quick and easy creation of web apps.

---

## **Option 1: Creating a Command Line Interface (CLI)**

We'll enhance your application to allow users to:

- List available personas.
- Select personas for nodes.
- Input custom prompts.
- Customize the graph structure if desired.

### **1. Install the Required Libraries**

We can use the `argparse` module from the standard library or the `click` library, which is more user-friendly for complex CLIs.

Let's use `click`:

```bash
pip install click
```

Add this to your `requirements.txt`:

```
click
```

### **2. Modify `main.py` to Include CLI Functionality**

#### **2.1. Import `click` Module**

At the top of your `main.py`, import `click`:

```python
import click
```

#### **2.2. Refactor the Code into Functions**

We'll encapsulate the main logic into functions that we can call from the CLI commands.

**Encapsulate Graph Building into a Function:**

```python
def build_graph(nodes_info, edges_info):
    G = nx.DiGraph()
    nodes = {}
    # Create nodes
    for node_info in nodes_info:
        node_id = node_info['id']
        prompt_text = node_info['prompt_text']
        persona_name = node_info['persona_name']
        node = Node(node_id, prompt_text, persona_name)
        G.add_node(node_id, data=node)
        nodes[node_id] = node

    # Add edges
    for edge in edges_info:
        G.add_edge(edge['from'], edge['to'])
    
    return G
```

**Encapsulate Node Processing into a Function:**

```python
def process_graph(G):
    for node_id in nx.topological_sort(G):
        node = G.nodes[node_id]['data']
        if node.persona_name != "Analyst":
            node.context = collect_context(node_id, G)
            node.response_text = generate_response(node)
            update_markdown(node)
        else:
            analyze_responses(node, G)
```

**Update the `collect_context` and `analyze_responses` functions to accept `G` as a parameter:**

```python
def collect_context(node_id, G):
    # Existing code...
```

```python
def analyze_responses(node, G):
    # Existing code...
```

#### **2.3. Create CLI Commands with `click`**

Below all the functions, add the CLI commands:

```python
@click.group()
def cli():
    pass
```

**Command to List Available Personas:**

```python
@cli.command()
def list_personas():
    """List all available personas."""
    for persona_name in personas.keys():
        print(persona_name)
```

**Command to Run the Application with Custom Inputs:**

```python
@cli.command()
@click.option('--nodes', '-n', default=2, help='Number of nodes (excluding the analyst node).')
def run(nodes):
    """Run the application with the specified number of nodes."""
    # Let the user select personas and input prompts for each node
    nodes_info = []
    for i in range(1, nodes + 1):
        print(f"\nConfiguring Node {i}")
        persona_name = click.prompt('Enter the persona name', type=str)
        while persona_name not in personas:
            print('Persona not found. Available personas:')
            for name in personas.keys():
                print(f" - {name}")
            persona_name = click.prompt('Enter the persona name', type=str)
        
        prompt_text = click.prompt('Enter the prompt text', type=str)
        node_info = {
            'id': i,
            'prompt_text': prompt_text,
            'persona_name': persona_name
        }
        nodes_info.append(node_info)
    
    # Add the analyst node
    analyst_node_id = nodes + 1
    analyst_node_info = {
        'id': analyst_node_id,
        'prompt_text': '',
        'persona_name': 'Analyst'
    }
    nodes_info.append(analyst_node_info)
    
    # Define edges (here we assume that the analyst node depends on all other nodes)
    edges_info = []
    for i in range(1, nodes + 1):
        edges_info.append({'from': i, 'to': analyst_node_id})
    
    # Build and process the graph
    G = build_graph(nodes_info, edges_info)
    process_graph(G)
    print("\nConversation has been generated and logged to conversation.md")
```

#### **2.4. Update the Main Execution Block**

Replace the existing execution code at the bottom of `main.py` with:

```python
if __name__ == '__main__':
    cli()
```

---

### **3. Running the Updated Application**

#### **3.1. List Available Personas**

```bash
python main.py list-personas
```

Output:

```
Ernest Hemingway
Albert Einstein
Analyst
```

#### **3.2. Run the Application with Custom Inputs**

```bash
python main.py run --nodes 2
```

The application will prompt you for inputs:

```
Configuring Node 1
Enter the persona name: Ernest Hemingway
Enter the prompt text: Discuss the impacts of the industrial revolution.

Configuring Node 2
Enter the persona name: Albert Einstein
Enter the prompt text: Explain quantum mechanics in simple terms.

Conversation has been generated and logged to conversation.md
```

---

## **Option 2: Creating a Graphical User Interface (GUI) with Streamlit**

**Streamlit** allows you to turn your Python scripts into interactive web apps with minimal effort.

### **1. Install Streamlit**

```bash
pip install streamlit
```

Add this to your `requirements.txt`:

```
streamlit
```

### **2. Create a New Streamlit App File**

Create a new file called `app.py` in your project directory.

### **3. Write the Streamlit App**

**Import Necessary Modules in `app.py`:**

```python
import streamlit as st
import os
import json
import networkx as nx
from main import Node, generate_response, collect_context, analyze_responses, build_system_prompt, load_personas, update_markdown, process_graph, build_graph
```

**Ensure `main.py` Functions are Importable**

- Modify your `main.py` functions to be importable without executing the CLI commands.
- Place the CLI commands under `if __name__ == '__main__':`.

**Load Personas**

```python
# Load personas
persona_dir = 'personas'
personas = load_personas(persona_dir)
```

**Streamlit Interface**

```python
def main():
    st.title("LangChain Graph App")
    
    st.header("Create a Conversation Graph")
    
    # Select number of nodes
    num_nodes = st.number_input('Number of nodes (excluding the analyst node):', min_value=1, value=2)
    
    nodes_info = []
    for i in range(1, int(num_nodes) + 1):
        st.subheader(f"Node {i} Configuration")
        persona_name = st.selectbox(f"Select persona for Node {i}:", options=list(personas.keys()), key=f"persona_{i}")
        prompt_text = st.text_area(f"Enter the prompt for Node {i}:", key=f"prompt_{i}")
        node_info = {
            'id': i,
            'prompt_text': prompt_text,
            'persona_name': persona_name
        }
        nodes_info.append(node_info)
    
    # Assume the analyst node
    analyst_node_id = int(num_nodes) + 1
    analyst_node_info = {
        'id': analyst_node_id,
        'prompt_text': '',
        'persona_name': 'Analyst'
    }
    nodes_info.append(analyst_node_info)
    
    # Define edges
    edges_info = []
    for i in range(1, int(num_nodes) + 1):
        edges_info.append({'from': i, 'to': analyst_node_id})
    
    if st.button("Generate Conversation"):
        G = build_graph(nodes_info, edges_info)
        process_graph(G)
        st.success("Conversation has been generated and logged to conversation.md")
        
        # Display the conversation
        with open("conversation.md", "r", encoding="utf-8") as f:
            content = f.read()
            st.markdown(content)
```

**Run the Streamlit App in `app.py`:**

```python
if __name__ == '__main__':
    main()
```

### **4. Running the Streamlit App**

In your terminal, run:

```bash
streamlit run app.py
```

A web browser will open displaying your app.

### **5. Interact with the GUI**

- Select the number of nodes.
- Choose personas from dropdown menus.
- Enter prompts for each node.
- Click "Generate Conversation" to run the application.
- The conversation will be displayed on the page.

---

## **Adjustments to `main.py`**

To make the functions importable for the GUI, ensure that your `main.py` does not execute any code upon import.

- Place the CLI commands and any execution code under:

```python
if __name__ == '__main__':
    cli()
```

---

## **Summary**

- **CLI Option:**
  - **Use `click` to create a user-friendly command-line interface.**
  - **Allows users to select personas and customize prompts interactively in the terminal.**

- **GUI Option:**
  - **Use Streamlit to build a simple web application.**
  - **Users can select personas and enter prompts in a web browser interface.**

---

## **Next Steps**

- **Add Error Handling:**
  - Validate user inputs for personas and prompts.
  - Handle exceptions that may occur during execution.

- **Enhance the GUI:**
  - Allow users to define custom personas through the interface.
  - Provide visualization of the conversation graph.

- **Improve the CLI:**
  - Add more options for advanced customization.
  - Save and load previous configurations.

- **Documentation:**
  - Update README files with instructions on how to use the CLI and GUI.
  - Provide examples and screenshots for clarity.

---

## **Testing**

- **CLI Testing:**
  - Run various configurations through the CLI to ensure functionality.
  - Test with different numbers of nodes and personas.

- **GUI Testing:**
  - Interact with the app in the browser.
  - Confirm that conversations are generated as expected.

---

## **Conclusion**

By implementing either the CLI or GUI, you've enhanced your application to be more user-friendly and interactive. Users can now select personas and customize prompts without modifying the code directly.

Final main.py:

```python

# main.py
import click
import os
import networkx as nx
from langchain import PromptTemplate, LLMChain
from langchain.llms import Ollama
import json





# Define the Node class
class Node:
    def __init__(self, node_id, prompt_text, persona_name):
        self.id = node_id
        self.prompt_text = prompt_text
        self.response_text = None
        self.context = ""
        self.persona_name = persona_name
        self.persona_attributes = {}

# Initialize the graph
G = nx.DiGraph()

def build_graph(nodes_info, edges_info):
    G = nx.DiGraph()
    nodes = {}
    # Create nodes
    for node_info in nodes_info:
        node_id = node_info['id']
        prompt_text = node_info['prompt_text']
        persona_name = node_info['persona_name']
        node = Node(node_id, prompt_text, persona_name)
        G.add_node(node_id, data=node)
        nodes[node_id] = node

    # Add edges
    for edge in edges_info:
        G.add_edge(edge['from'], edge['to'])
    
    return G

def process_graph(G):
    for node_id in nx.topological_sort(G):
        node = G.nodes[node_id]['data']
        if node.persona_name != "Analyst":
            node.context = collect_context(node_id, G)
            node.response_text = generate_response(node)
            update_markdown(node)
        else:
            analyze_responses(node, G)

def load_personas(persona_dir):
    personas = {}
    for filename in os.listdir(persona_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(persona_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                persona_data = json.load(f)
                name = persona_data.get('name')
                if name:
                    personas[name] = persona_data
    return personas

# Load personas
persona_dir = 'personas'  # Directory where persona JSON files are stored
personas = load_personas(persona_dir)

# Function to collect context from predecessor nodes
def collect_context(node_id, G):
    predecessors = list(G.predecessors(node_id))
    context = ""
    for pred_id in predecessors:
        pred_node = G.nodes[pred_id]['data']
        if pred_node.response_text:
            context += f"From {pred_node.persona}:\n{pred_node.response_text}\n\n"
    return context

# Function to generate responses using LangChain and Ollama
def generate_response(node):
    persona = personas.get(node.persona_name)
    if not persona:
        raise ValueError(f"Persona '{node.persona_name}' not found.")
    
    node.persona_attributes = persona

    # Build the system prompt based on persona attributes
    system_prompt = build_system_prompt(persona)

    # Build the complete prompt
    prompt_template = PromptTemplate(
        input_variables=["system_prompt", "context", "prompt"],
        template="{system_prompt}\n\n{context}\n\n{prompt}"
    )
    # Instantiate the Ollama LLM
    llm = Ollama(
        base_url="http://localhost:11434",  # Default Ollama server URL
        model="qwq",  # Replace with the model you have downloaded
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(
        system_prompt=system_prompt,
        context=node.context,
        prompt=node.prompt_text
    )
    return response

def build_system_prompt(persona):
    # Construct descriptive sentences based on persona attributes
    # We'll focus on key attributes for brevity
    name = persona.get('name', 'The speaker')
    tone = persona.get('tone', 'neutral')
    sentence_structure = persona.get('sentence_structure', 'varied')
    vocabulary_complexity = persona.get('vocabulary_complexity', 5)
    formality_level = persona.get('formality_level', 5)
    pronoun_preference = persona.get('pronoun_preference', 'third-person')
    language_abstraction = persona.get('language_abstraction', 'mixed')

    # Create a description
    description = (
        f"You are {name}, writing in a {tone} tone using {sentence_structure} sentences. "
        f"Your vocabulary complexity is {vocabulary_complexity}/10, and your formality level is {formality_level}/10. "
        f"You prefer {pronoun_preference} narration and your language abstraction is {language_abstraction}."
    )

    # Include any other attributes as needed
    # ...

    return description


# Function to log interactions to a markdown file
def update_markdown(node):
    with open("conversation.md", "a", encoding="utf-8") as f:
        f.write(f"## Node {node.id}: {node.persona_name}\n\n")
        f.write(f"**Prompt:**\n\n{node.prompt_text}\n\n")
        f.write(f"**Response:**\n\n{node.response_text}\n\n---\n\n")

# Function for nodes that perform analysis
def analyze_responses(node, G):
    # Collect responses from predecessor nodes
    predecessors = list(G.predecessors(node.id))
    analysis_input = ""
    for pred_id in predecessors:
        pred_node = G.nodes[pred_id]['data']
        analysis_input += f"{pred_node.persona_name}'s response:\n{pred_node.response_text}\n\n"

    node.prompt_text = f"Provide an analysis comparing the following perspectives:\n\n{analysis_input}"
    node.context = ""  # Analysis is based solely on the provided responses
    node.response_text = generate_response(node)
    update_markdown(node)


@click.group()
def cli():
    pass

@cli.command()
def list_personas():
    """List all available personas."""
    for persona_name in personas.keys():
        print(persona_name)
        
@cli.command()
@click.option('--nodes', '-n', default=2, help='Number of nodes (excluding the analyst node).')
def run(nodes):
    """Run the application with the specified number of nodes."""
    # Let the user select personas and input prompts for each node
    nodes_info = []
    for i in range(1, nodes + 1):
        print(f"\nConfiguring Node {i}")
        persona_name = click.prompt('Enter the persona name', type=str)
        while persona_name not in personas:
            print('Persona not found. Available personas:')
            for name in personas.keys():
                print(f" - {name}")
            persona_name = click.prompt('Enter the persona name', type=str)
        
        prompt_text = click.prompt('Enter the prompt text', type=str)
        node_info = {
            'id': i,
            'prompt_text': prompt_text,
            'persona_name': persona_name
        }
        nodes_info.append(node_info)
    
    # Add the analyst node
    analyst_node_id = nodes + 1
    analyst_node_info = {
        'id': analyst_node_id,
        'prompt_text': '',
        'persona_name': 'Analyst'
    }
    nodes_info.append(analyst_node_info)
    
    # Define edges (here we assume that the analyst node depends on all other nodes)
    edges_info = []
    for i in range(1, nodes + 1):
        edges_info.append({'from': i, 'to': analyst_node_id})
    
    # Build and process the graph
    G = build_graph(nodes_info, edges_info)
    process_graph(G)
    print("\nConversation has been generated and logged to conversation.md")

if __name__ == '__main__':
    cli()

```



## Then I added the ability to increase the number of iterations through the cli command --iterations X where X is number of iterations.

```python
# main.py

import click

import os

import networkx as nx

from langchain import PromptTemplate, LLMChain

from langchain.llms import Ollama

import json

  
  
  
  
  

# Define the Node class

class Node:

def __init__(self, node_id, prompt_text, persona_name):

self.id = node_id

self.prompt_text = prompt_text

self.response_text = None

self.context = ""

self.persona_name = persona_name

self.persona_attributes = {}

  

# Initialize the graph

G = nx.DiGraph()

  

def build_graph(nodes_info, edges_info):

G = nx.DiGraph()

nodes = {}

# Create nodes

for node_info in nodes_info:

node_id = node_info['id']

prompt_text = node_info['prompt_text']

persona_name = node_info['persona_name']

node = Node(node_id, prompt_text, persona_name)

G.add_node(node_id, data=node)

nodes[node_id] = node

  

# Add edges

for edge in edges_info:

G.add_edge(edge['from'], edge['to'])

return G

  

def process_graph(G, iterations):

"""Process the graph for the specified number of iterations."""

for iteration in range(iterations):

print(f"\nProcessing iteration {iteration + 1}/{iterations}")

# Store previous responses for context

previous_responses = {}

for node_id in G.nodes():

node = G.nodes[node_id]['data']

if node.response_text:

previous_responses[node_id] = node.response_text

  

# Process each node in topological order

for node_id in nx.topological_sort(G):

node = G.nodes[node_id]['data']

if node.persona_name != "Analyst":

node.context = collect_context(node_id, G, iteration, previous_responses)

node.response_text = generate_response(node, iteration)

update_markdown(node, iteration)

else:

analyze_responses(node, G, iteration)

  
  

def load_personas(persona_dir):

personas = {}

for filename in os.listdir(persona_dir):

if filename.endswith('.json'):

filepath = os.path.join(persona_dir, filename)

with open(filepath, 'r', encoding='utf-8') as f:

persona_data = json.load(f)

name = persona_data.get('name')

if name:

personas[name] = persona_data

return personas

  

# Load personas

persona_dir = 'personas' # Directory where persona JSON files are stored

personas = load_personas(persona_dir)

  

# Function to collect context from predecessor nodes

def collect_context(node_id, G, iteration, previous_responses):

"""Collect context including previous iterations."""

predecessors = list(G.predecessors(node_id))

context = ""

# Add context from previous iterations if they exist

if iteration > 0:

context += f"\nPrevious iteration responses:\n"

for pred_id in predecessors:

if pred_id in previous_responses:

pred_node = G.nodes[pred_id]['data']

context += f"From {pred_node.persona_name} (previous round):\n{previous_responses[pred_id]}\n\n"

# Add context from current iteration

context += f"\nCurrent iteration responses:\n"

for pred_id in predecessors:

pred_node = G.nodes[pred_id]['data']

if pred_node.response_text:

context += f"From {pred_node.persona_name}:\n{pred_node.response_text}\n\n"

return context

  

# Function to generate responses using LangChain and Ollama

def generate_response(node, iteration):

"""Generate response with awareness of the current iteration."""

persona = personas.get(node.persona_name)

if not persona:

raise ValueError(f"Persona '{node.persona_name}' not found.")

node.persona_attributes = persona

system_prompt = build_system_prompt(persona)

# Modify the prompt to include iteration information

iteration_prompt = f"This is round {iteration + 1} of the conversation. "

if iteration > 0:

iteration_prompt += "Please consider the previous responses in your reply. "

prompt_template = PromptTemplate(

input_variables=["system_prompt", "iteration_prompt", "context", "prompt"],

template="{system_prompt}\n\n{iteration_prompt}\n\n{context}\n\n{prompt}"

)

llm = Ollama(

base_url="http://localhost:11434",

model="qwq",

)

chain = LLMChain(llm=llm, prompt=prompt_template)

response = chain.run(

system_prompt=system_prompt,

iteration_prompt=iteration_prompt,

context=node.context,

prompt=node.prompt_text

)

return response

  

def build_system_prompt(persona):

# Construct descriptive sentences based on persona attributes

# We'll focus on key attributes for brevity

name = persona.get('name', 'The speaker')

tone = persona.get('tone', 'neutral')

sentence_structure = persona.get('sentence_structure', 'varied')

vocabulary_complexity = persona.get('vocabulary_complexity', 5)

formality_level = persona.get('formality_level', 5)

pronoun_preference = persona.get('pronoun_preference', 'third-person')

language_abstraction = persona.get('language_abstraction', 'mixed')

  

# Create a description

description = (

f"You are {name}, writing in a {tone} tone using {sentence_structure} sentences. "

f"Your vocabulary complexity is {vocabulary_complexity}/10, and your formality level is {formality_level}/10. "

f"You prefer {pronoun_preference} narration and your language abstraction is {language_abstraction}."

)

  

# Include any other attributes as needed

# ...

  

return description

  
  

# Function to log interactions to a markdown file

def update_markdown(node, iteration):

"""Update markdown file with iteration information."""

with open("conversation.md", "a", encoding="utf-8") as f:

f.write(f"## Iteration {iteration + 1} - Node {node.id}: {node.persona_name}\n\n")

f.write(f"**Prompt:**\n\n{node.prompt_text}\n\n")

f.write(f"**Response:**\n\n{node.response_text}\n\n---\n\n")

  

# Function for nodes that perform analysis

def analyze_responses(node, G, iteration):

"""Analyze responses with awareness of iteration context."""

predecessors = list(G.predecessors(node.id))

analysis_input = f"Analysis for Iteration {iteration + 1}:\n\n"

for pred_id in predecessors:

pred_node = G.nodes[pred_id]['data']

analysis_input += f"{pred_node.persona_name}'s response:\n{pred_node.response_text}\n\n"

  

node.prompt_text = (

f"Provide an analysis comparing the following perspectives from iteration {iteration + 1}:\n\n"

f"{analysis_input}\n"

f"Consider how the conversation has evolved across iterations."

)

node.context = ""

node.response_text = generate_response(node, iteration)

update_markdown(node, iteration)

  
  

@click.group()

def cli():

pass

  

@cli.command()

def list_personas():

"""List all available personas."""

for persona_name in personas.keys():

print(persona_name)

  

@cli.command()

@click.option('--nodes', '-n', default=2, help='Number of nodes (excluding the analyst node).')

@click.option('--iterations', '-i', default=1, help='Number of conversation iterations.')

def run(nodes, iterations):

"""Run the application with the specified number of nodes and iterations."""

# Clear previous conversation file

with open("conversation.md", "w", encoding="utf-8") as f:

f.write("# Conversation Log\n\n")

  

# Let the user select personas and input prompts for each node

nodes_info = []

for i in range(1, nodes + 1):

print(f"\nConfiguring Node {i}")

persona_name = click.prompt('Enter the persona name', type=str)

while persona_name not in personas:

print('Persona not found. Available personas:')

for name in personas.keys():

print(f" - {name}")

persona_name = click.prompt('Enter the persona name', type=str)

prompt_text = click.prompt('Enter the prompt text', type=str)

node_info = {

'id': i,

'prompt_text': prompt_text,

'persona_name': persona_name

}

nodes_info.append(node_info)

# Add the analyst node

analyst_node_id = nodes + 1

analyst_node_info = {

'id': analyst_node_id,

'prompt_text': '',

'persona_name': 'Analyst'

}

nodes_info.append(analyst_node_info)

# Define edges

edges_info = []

for i in range(1, nodes + 1):

edges_info.append({'from': i, 'to': analyst_node_id})

# Build and process the graph

G = build_graph(nodes_info, edges_info)

# Process the graph for the specified number of iterations

process_graph(G, iterations)

print(f"\nConversation with {iterations} iterations has been generated and logged to conversation.md")

  

if __name__ == '__main__':

cli()
```