---
layout: post
title:  Swarm Autogen
date:   2024-11-27 05:40:44 -0500
---

Enhancing your existing Python script by integrating [OpenAI Swarm](https://github.com/openai/swarm) and [Microsoft Autogen](https://github.com/microsoft/autogen) can significantly improve its capabilities, scalability, and maintainability. Below, I’ll guide you through understanding these tools, integrating them into your project, and adding new features to make your script more robust and feature-rich.

## Table of Contents

1. [Overview of OpenAI Swarm and Microsoft Autogen](#overview)
2. [Prerequisites](#prerequisites)
3. [Integrating OpenAI Swarm](#integrate-swarm)
4. [Integrating Microsoft Autogen](#integrate-autogen)
5. [Enhancing the Existing Script](#enhance-script)
6. [Adding New Features](#add-features)
7. [Security Improvements](#security)
8. [Final Thoughts](#final-thoughts)

---

<a name="overview"></a>
### 1. Overview of OpenAI Swarm and Microsoft Autogen

**OpenAI Swarm** is a framework designed to manage and coordinate multiple AI agents, enabling them to work collaboratively to solve complex tasks. It facilitates communication, task delegation, and aggregation of results from various agents.

**Microsoft Autogen** is a framework that simplifies the orchestration of large language models (LLMs) to build complex applications. It provides tools for chaining model calls, managing context, and integrating additional functionalities like data retrieval or transformation.

By integrating these frameworks, you can leverage multi-agent collaboration and advanced orchestration capabilities, making your persona generator and responder more powerful and flexible.

---

<a name="prerequisites"></a>
### 2. Prerequisites

Before proceeding, ensure you have the following:

1. **Python 3.8+** installed.
2. **Git** installed to clone repositories.
3. **Virtual Environment** set up to manage dependencies.
4. **API Keys** for OpenAI and any other services you intend to use.

---

<a name="integrate-swarm"></a>
### 3. Integrating OpenAI Swarm

**Step 1: Clone and Install OpenAI Swarm**

```bash
git clone https://github.com/openai/swarm.git
cd swarm
pip install -r requirements.txt
python setup.py install
```

**Step 2: Understanding Swarm Structure**

OpenAI Swarm allows you to define multiple agents that can perform specific tasks. For your application, you can create agents for:

- Persona Generation
- Response Generation
- Validation and Formatting
- Exporting

**Step 3: Define Swarm Agents**

Create separate modules for each agent. For example:

- `persona_agent.py`
- `response_agent.py`
- `validation_agent.py`
- `export_agent.py`

**Example: `persona_agent.py`**

```python
from swarm.agent import Agent
import json
import os
from openai import OpenAI

class PersonaAgent(Agent):
    def __init__(self, api_key, persona_file='persona.json'):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        self.persona_file = persona_file

    def generate_persona(self, sample_text: str) -> dict:
        prompt = (
            "Please analyze the writing style and personality of the given writing sample. "
            "You are a persona generation assistant. Analyze the following text and create a persona profile "
            "that captures the writing style and personality characteristics of the author. "
            "YOU MUST RESPOND WITH A VALID JSON OBJECT ONLY, no other text or analysis. "
            "The response must start with '{' and end with '}' and use the following exact structure:\n\n"
            "{...}"  # Truncated for brevity
            f"Sample Text:\n{sample_text}"
        )
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1
        }
        response = self.client.chat.completions.create(**payload)
        content = response.choices[0].message.content.strip()
        # Extract and parse JSON
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        json_str = content[start_idx:end_idx]
        persona = json.loads(json_str)
        return persona

    def save_persona(self, persona: dict) -> bool:
        try:
            if not persona:
                print("Error: Cannot save empty persona")
                return False
            os.makedirs(os.path.dirname(self.persona_file) if os.path.dirname(self.persona_file) else '.', exist_ok=True)
            with open(self.persona_file, 'w', encoding='utf-8') as f:
                json.dump(persona, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved persona to {self.persona_file}")
            return True
        except Exception as e:
            print(f"Error saving persona: {str(e)}")
            return False
```

**Step 4: Orchestrate Agents with Swarm**

Create a `main_swarm.py` to coordinate agents.

```python
from swarm import Swarm
from persona_agent import PersonaAgent
from response_agent import ResponseAgent
from validation_agent import ValidationAgent
from export_agent import ExportAgent

def main():
    swarm = Swarm()
    api_key = os.getenv("OPENAI_API_KEY")
    
    persona_agent = PersonaAgent(api_key)
    response_agent = ResponseAgent(api_key)
    validation_agent = ValidationAgent()
    export_agent = ExportAgent()
    
    swarm.add_agent(persona_agent)
    swarm.add_agent(response_agent)
    swarm.add_agent(validation_agent)
    swarm.add_agent(export_agent)
    
    # Example workflow
    sample_text = "Your sample text here..."
    persona = persona_agent.generate_persona(sample_text)
    if validation_agent.validate(persona):
        persona_agent.save_persona(persona)
        prompt = "Your prompt here..."
        response = response_agent.generate_response(persona, prompt)
        export_agent.export_to_markdown(response)
    else:
        print("Persona validation failed.")

if __name__ == "__main__":
    main()
```

---

<a name="integrate-autogen"></a>
### 4. Integrating Microsoft Autogen

**Step 1: Clone and Install Microsoft Autogen**

```bash
git clone https://github.com/microsoft/autogen.git
cd autogen
pip install -r requirements.txt
python setup.py install
```

**Step 2: Understanding Autogen Structure**

Microsoft Autogen allows you to create chains of model calls, manage context, and integrate additional functionalities seamlessly.

**Step 3: Define Autogen Chains**

You can create chains for tasks like persona generation, response generation, and exporting.

**Example: `autogen_chain.py`**

```python
from autogen import Chain, Step
from openai import OpenAI
import json

class PersonaGenerationChain(Chain):
    def __init__(self, api_key):
        super().__init__()
        self.client = OpenAI(api_key=api_key)

    @Step
    def generate_persona(self, sample_text: str) -> dict:
        prompt = (
            "Please analyze the writing style and personality of the given writing sample. "
            "You are a persona generation assistant. Analyze the following text and create a persona profile "
            "that captures the writing style and personality characteristics of the author. "
            "YOU MUST RESPOND WITH A VALID JSON OBJECT ONLY, no other text or analysis. "
            "The response must start with '{' and end with '}' and use the following exact structure:\n\n"
            "{...}"  # Truncated for brevity
            f"Sample Text:\n{sample_text}"
        )
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=1
        )
        content = response.choices[0].message.content.strip()
        # Extract and parse JSON
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        json_str = content[start_idx:end_idx]
        persona = json.loads(json_str)
        return persona
```

**Step 4: Orchestrate Chains with Autogen**

Create a `main_autogen.py` to manage chains.

```python
from autogen_chain import PersonaGenerationChain
from validation_agent import ValidationAgent
from export_agent import ExportAgent

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    persona_chain = PersonaGenerationChain(api_key)
    validation_agent = ValidationAgent()
    export_agent = ExportAgent()
    
    sample_text = "Your sample text here..."
    persona = persona_chain.generate_persona(sample_text)
    
    if validation_agent.validate(persona):
        # Save persona
        with open('persona.json', 'w', encoding='utf-8') as f:
            json.dump(persona, f, indent=4)
        # Generate response
        prompt = "Your prompt here..."
        # Define response generation logic, possibly another chain
        # Export response
        export_agent.export_to_markdown("Generated response")
    else:
        print("Persona validation failed.")

if __name__ == "__main__":
    main()
```

---

<a name="enhance-script"></a>
### 5. Enhancing the Existing Script

Now, let's enhance your existing script by integrating both OpenAI Swarm and Microsoft Autogen. Below are the key modifications and additions:

**Step 1: Refactor Code into Modular Components**

Organize your code into modules to separate concerns:

- `agents/`: Contains Swarm agents.
  - `persona_agent.py`
  - `response_agent.py`
  - `validation_agent.py`
  - `export_agent.py`
  
- `chains/`: Contains Autogen chains.
  - `persona_chain.py`
  - `response_chain.py`
  
- `utils/`: Utility functions.
  - `file_utils.py`
  - `input_utils.py`
  
- `main.py`: Main orchestrator.

**Step 2: Implement Agents and Chains**

As shown in the previous sections, implement agents and chains in their respective modules. Ensure each agent or chain has a single responsibility.

**Step 3: Update `main.py` to Use Swarm and Autogen**

Here's an example of how to integrate both frameworks into your main application.

```python
import os
from swarm import Swarm
from agents.persona_agent import PersonaAgent
from agents.response_agent import ResponseAgent
from agents.validation_agent import ValidationAgent
from agents.export_agent import ExportAgent
from utils.file_utils import load_sample_text, create_backup
from utils.input_utils import get_multiline_input

def main():
    print("\n=== Enhanced Persona Generator and Responder ===")
    swarm = Swarm()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize agents
    persona_agent = PersonaAgent(api_key)
    response_agent = ResponseAgent(api_key)
    validation_agent = ValidationAgent()
    export_agent = ExportAgent()
    
    # Add agents to swarm
    swarm.add_agent(persona_agent)
    swarm.add_agent(response_agent)
    swarm.add_agent(validation_agent)
    swarm.add_agent(export_agent)
    
    while True:
        print("\nOptions:")
        print("1. Use existing Persona")
        print("2. Generate new Persona from sample text")
        print("3. Load sample text from file")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '4':
            print("Exiting program...")
            break
        
        persona = {}
        
        if choice == '1':
            persona = swarm.get_agent('PersonaAgent').load_persona()
            if persona:
                print("\nCurrent Persona:")
                print(swarm.get_agent('PersonaAgent').format_persona_summary(persona))
            else:
                if input("\nNo persona loaded. Generate new one? (y/n): ").lower() == 'y':
                    choice = '2'
                else:
                    continue
        
        if choice == '2':
            sample_text = get_multiline_input("\nEnter sample text:")
            if not sample_text.strip():
                print("Error: Empty sample text provided")
                continue
            print("\nGenerating persona from sample text...")
            persona = swarm.get_agent('PersonaAgent').generate_persona(sample_text)
            if persona and swarm.get_agent('ValidationAgent').validate(persona):
                swarm.get_agent('PersonaAgent').create_backup()
                if swarm.get_agent('PersonaAgent').save_persona(persona):
                    print("\nGenerated Persona:")
                    print(swarm.get_agent('PersonaAgent').format_persona_summary(persona))
                else:
                    print("Warning: Persona generated but not saved")
            else:
                print("Error: Failed to generate valid persona")
                continue
        
        elif choice == '3':
            filename = input("\nEnter the path to the text file: ").strip()
            sample_text = load_sample_text(filename)
            if not sample_text:
                continue
            print("\nGenerating persona from file...")
            persona = swarm.get_agent('PersonaAgent').generate_persona(sample_text)
            if persona and swarm.get_agent('ValidationAgent').validate(persona):
                swarm.get_agent('PersonaAgent').create_backup()
                if swarm.get_agent('PersonaAgent').save_persona(persona):
                    print("\nGenerated Persona:")
                    print(swarm.get_agent('PersonaAgent').format_persona_summary(persona))
                else:
                    print("Warning: Persona generated but not saved")
            else:
                print("Error: Failed to generate valid persona")
                continue
        
        # Get prompt and generate response
        while True:
            prompt = get_multiline_input("\nEnter your prompt:")
            if not prompt.strip():
                print("Error: Empty prompt provided")
                if input("Try again? (y/n): ").lower() != 'y':
                    break
                continue
            print("\nGenerating response...")
            response = swarm.get_agent('ResponseAgent').generate_response(persona, prompt)
            print("\n=== Generated Response ===")
            print(response)
            if input("\nExport response to Markdown? (y/n): ").lower() == 'y':
                default_filename = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                custom_filename = input(f"Enter filename (default: {default_filename}): ").strip()
                filename = custom_filename if custom_filename else default_filename
                if swarm.get_agent('ExportAgent').export_to_markdown(response, filename):
                    print("Response exported successfully")
                else:
                    print("Error: Failed to export response")
            if input("\nGenerate another response with current persona? (y/n): ").lower() != 'y':
                break
        
        if input("\nStart over with a different persona? (y/n): ").lower() != 'y':
            print("Exiting program...")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"\nProgram terminated due to error: {str(e)}")
    finally:
        print("\nThank you for using the Enhanced Persona Generator and Responder!")
```

**Explanation:**

- **Swarm Initialization:** Initializes the swarm and adds the necessary agents.
- **User Interaction:** Maintains the existing user interface while leveraging the swarm for operations.
- **Agent Utilization:** Delegates tasks like persona generation, validation, and exporting to respective agents, promoting modularity and scalability.

---

<a name="add-features"></a>
### 6. Adding New Features

With OpenAI Swarm and Microsoft Autogen integrated, you can introduce several new features:

#### a. **Multi-Threaded Persona Generation**

Allow multiple personas to be generated simultaneously from different sample texts.

**Implementation:**

- Utilize Swarm’s multi-agent capabilities to handle concurrent persona generation requests.
- Modify the `PersonaAgent` to handle asynchronous tasks.

#### b. **Enhanced Validation and Error Handling**

Implement more robust validation using multiple validation agents.

**Implementation:**

- Create additional `ValidationAgent`s focusing on different aspects (e.g., schema validation, content quality).
- Aggregate validation results before proceeding.

#### c. **Persona Management Dashboard**

Develop a simple dashboard to manage personas, view summaries, and export options.

**Implementation:**

- Use a lightweight web framework like Flask or FastAPI.
- Integrate with Swarm to fetch and display persona data.

**Example: Flask Integration**

```python
from flask import Flask, jsonify, request
from swarm import Swarm

app = Flask(__name__)
swarm = Swarm()
# Initialize and add agents...

@app.route('/personas', methods=['GET'])
def get_personas():
    # Implement logic to list all saved personas
    pass

@app.route('/persona', methods=['POST'])
def create_persona():
    sample_text = request.json.get('sample_text')
    persona = swarm.get_agent('PersonaAgent').generate_persona(sample_text)
    if swarm.get_agent('ValidationAgent').validate(persona):
        swarm.get_agent('PersonaAgent').save_persona(persona)
        return jsonify(persona), 201
    else:
        return jsonify({'error': 'Invalid persona'}), 400

# Additional routes...

if __name__ == '__main__':
    app.run(debug=True)
```

#### d. **Integration with External APIs**

Enhance responses by integrating with APIs for data retrieval, sentiment analysis, or knowledge bases.

**Implementation:**

- Create new agents or steps in Autogen chains to handle API interactions.
- Example: An agent that fetches current events to make responses more relevant.

---

<a name="security"></a>
### 7. Security Improvements

Your current script includes the OpenAI API key hardcoded within the script. This is a significant security risk. Here's how to improve it:

#### a. **Use Environment Variables**

Store sensitive information like API keys in environment variables instead of hardcoding them.

**Implementation:**

1. **Set Environment Variable:**

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Access in Python:**

   ```python
   import os

   api_key = os.getenv("OPENAI_API_KEY")
   if not api_key:
       raise ValueError("OpenAI API key not found in environment variables.")
   ```

3. **Remove Hardcoded Keys:**

   Remove the hardcoded `api_key` from your script.

#### b. **Use `.env` Files with `python-dotenv`**

For easier management, especially during development, use a `.env` file.

**Implementation:**

1. **Install `python-dotenv`:**

   ```bash
   pip install python-dotenv
   ```

2. **Create a `.env` File:**

   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. **Load `.env` in Python:**

   ```python
   from dotenv import load_dotenv
   import os

   load_dotenv()
   api_key = os.getenv("OPENAI_API_KEY")
   ```

4. **Add `.env` to `.gitignore`:**

   ```gitignore
   # .gitignore
   .env
   ```

#### c. **Secure File Handling**

Ensure that sensitive files like `persona.json` are stored securely.

**Implementation:**

- **File Permissions:** Set appropriate file permissions to restrict access.

  ```python
  import os

  def save_persona(persona: dict, filename: str = PERSONA_FILE) -> bool:
      try:
          if not persona:
              print("Error: Cannot save empty persona")
              return False
          os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
          with open(filename, 'w', encoding='utf-8') as f:
              json.dump(persona, f, indent=4, ensure_ascii=False)
          os.chmod(filename, 0o600)  # Read and write permissions for the owner only
          print(f"Successfully saved persona to {filename}")
          return True
      except Exception as e:
          print(f"Error saving persona: {str(e)}")
          return False
  ```

- **Encryption:** For highly sensitive data, consider encrypting the JSON files.

---

<a name="final-thoughts"></a>
### 8. Final Thoughts

Integrating **OpenAI Swarm** and **Microsoft Autogen** into your existing persona generator and responder script can significantly enhance its capabilities by promoting modularity, scalability, and maintainability. Here's a summary of the steps and recommendations:

1. **Modular Architecture:** Break down your script into modular components (agents and chains) to separate concerns and facilitate easier maintenance.

2. **Leverage Swarm for Multi-Agent Collaboration:** Use Swarm to manage different agents responsible for specific tasks, enabling concurrent processing and better task management.

3. **Utilize Autogen for Advanced Orchestration:** Implement Autogen chains to handle complex workflows, context management, and integration with external services.

4. **Enhance User Experience:** Introduce new features like a management dashboard, multi-threaded processing, and integration with external APIs to provide a richer user experience.

5. **Prioritize Security:** Always handle sensitive information securely by using environment variables, securing file permissions, and avoiding hardcoded credentials.

6. **Continuous Improvement:** Regularly update dependencies, monitor performance, and seek user feedback to iteratively improve your application.

By following these guidelines and integrating the mentioned frameworks, your application will be better equipped to handle complex tasks, scale efficiently, and provide a more robust and feature-rich experience.

If you encounter specific challenges during the integration or need further assistance with particular components, feel free to ask!