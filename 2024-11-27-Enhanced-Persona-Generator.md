---
layout: post
title:  Building an Enhanced Persona Generator and Responder with Python and OpenAI
date:   2024-11-27 06:40:44 -0500
---
# Building an Enhanced Persona Generator and Responder with Python and OpenAI

In the age of artificial intelligence, creating personalized and context-aware applications has become increasingly accessible. One such application is the **Enhanced Persona Generator and Responder**, which analyzes a sample text to generate a detailed persona and then uses that persona to craft tailored responses to user prompts. In this blog post, we'll walk through building this application step-by-step using Python and OpenAI's powerful language models.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Project Setup](#project-setup)
4. [Implementing the Agents](#implementing-the-agents)
   - [ExportAgent](#exportagent)
   - [PersonaAgent](#personaagent)
   - [ResponseAgent](#responseagent)
   - [ValidationAgent](#validationagent)
5. [Utility Modules](#utility-modules)
   - [file_utils.py](#file_utilspy)
   - [input_utils.py](#input_utilspy)
6. [Main Orchestrator (`main.py`)](#main-orchestrator-mainpy)
7. [Running the Application](#running-the-application)
8. [Troubleshooting](#troubleshooting)
9. [Conclusion](#conclusion)

---

<a name="introduction"></a>
## 1. Introduction

The **Enhanced Persona Generator and Responder** application serves two primary functions:

1. **Persona Generation**: Analyzes a provided sample text to create a comprehensive persona profile, capturing the author's writing style and personality traits.
2. **Response Generation**: Uses the generated persona to produce responses that align with the defined characteristics, ensuring consistency and personalization in interactions.

This application can be particularly useful for content creators, authors, chatbots, and any scenario where understanding and replicating a specific writing style is beneficial.

---

<a name="prerequisites"></a>
## 2. Prerequisites

Before diving into the development, ensure you have the following:

- **Python 3.8+**: Ensure Python is installed on your system. You can download it from [here](https://www.python.org/downloads/).
- **Virtual Environment (optional but recommended)**: Helps manage dependencies.
- **OpenAI API Key**: Required to access OpenAI's language models. Sign up and obtain your API key [here](https://platform.openai.com/signup).

---

<a name="project-setup"></a>
## 3. Project Setup

### **Step 1: Create the Project Directory**

Open your terminal or command prompt and execute the following commands:

```bash
mkdir persona_responder
cd persona_responder
```

### **Step 2: Set Up a Virtual Environment**

It's best practice to use a virtual environment to manage project dependencies.

```bash
python3 -m venv venv
```

Activate the virtual environment:

- **On macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

- **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

### **Step 3: Create `requirements.txt`**

Create a `requirements.txt` file to list all necessary dependencies:

```bash
touch requirements.txt
```

Add the following content to `requirements.txt`:

```plaintext
openai
python-dotenv
```

**Note:** 
- We've excluded `swarm`, `autogen`, and `flask` as they are not required in this simplified setup.
- Ensure that if you intend to use `ollama`, it's correctly installed or referenced, but for this guide, we'll focus on the essential dependencies.

### **Step 4: Install Dependencies**

Install the listed dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---

<a name="implementing-the-agents"></a>
## 4. Implementing the Agents

Our application is modular, consisting of various agents responsible for distinct tasks. Let's delve into each one.

### **Project Structure**

Ensure your project has the following structure:

```
persona_responder/
├── agents/
│   ├── __init__.py
│   ├── persona_agent.py
│   ├── response_agent.py
│   ├── validation_agent.py
│   └── export_agent.py
├── utils/
│   ├── __init__.py
│   ├── file_utils.py
│   └── input_utils.py
├── main.py
├── persona.json
├── .env
├── requirements.txt
└── README.md
```

Create the necessary directories and files:

```bash
mkdir agents utils
touch agents/__init__.py
touch utils/__init__.py
touch main.py
touch README.md
```

Now, let's implement each agent.

---

### **ExportAgent**

Responsible for exporting generated responses to Markdown files.

```python
# agents/export_agent.py

from datetime import datetime
import os


class ExportAgent:
    def export_to_markdown(self, content: str, filename: str = None) -> bool:
        """
        Export the content to a Markdown file with improved error handling.
        """
        try:
            if not content:
                print("Error: Cannot export empty content.")
                return False

            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"response_{timestamp}.md"

            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"Successfully exported response to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting to markdown: {str(e)}")
            return False
```

**Explanation:**

- **Functionality**: Takes content and an optional filename to export the content as a Markdown file.
- **Error Handling**: Checks for empty content and handles exceptions during file operations.
- **Default Filename**: If no filename is provided, it generates one based on the current timestamp.

---

### **PersonaAgent**

Generates a persona based on a sample text using OpenAI's API.

```python
# agents/persona_agent.py

import json
import os
from openai import OpenAI
from utils.file_utils import create_backup


class PersonaAgent:
    def __init__(self, api_key, persona_file='persona.json'):
        self.client = OpenAI(api_key=api_key)
        self.persona_file = persona_file

    def generate_persona(self, sample_text: str) -> dict:
        prompt = (
            "Please analyze the writing style and personality of the given writing sample. "
            "You are a persona generation assistant. Analyze the following text and create a persona profile "
            "that captures the writing style and personality characteristics of the author. "
            "YOU MUST RESPOND WITH A VALID JSON OBJECT ONLY, no other text or analysis. "
            "The response must start with '{' and end with '}' and use the following exact structure:\n\n"
            "{\n"
            "  \"name\": \"[Author/Character Name]\",\n"
            "  \"vocabulary_complexity\": [1-10],\n"
            "  \"sentence_structure\": \"[simple/complex/varied]\",\n"
            "  \"paragraph_organization\": \"[structured/loose/stream-of-consciousness]\",\n"
            "  \"idiom_usage\": [1-10],\n"
            "  \"metaphor_frequency\": [1-10],\n"
            "  \"simile_frequency\": [1-10],\n"
            "  \"tone\": \"[formal/informal/academic/conversational/etc.]\",\n"
            "  \"punctuation_style\": \"[minimal/heavy/unconventional]\",\n"
            "  \"contraction_usage\": [1-10],\n"
            "  \"pronoun_preference\": \"[first-person/third-person/etc.]\",\n"
            "  \"passive_voice_frequency\": [1-10],\n"
            "  \"rhetorical_question_usage\": [1-10],\n"
            "  \"list_usage_tendency\": [1-10],\n"
            "  \"personal_anecdote_inclusion\": [1-10],\n"
            "  \"pop_culture_reference_frequency\": [1-10],\n"
            "  \"technical_jargon_usage\": [1-10],\n"
            "  \"parenthetical_aside_frequency\": [1-10],\n"
            "  \"humor_sarcasm_usage\": [1-10],\n"
            "  \"emotional_expressiveness\": [1-10],\n"
            "  \"emphatic_device_usage\": [1-10],\n"
            "  \"quotation_frequency\": [1-10],\n"
            "  \"analogy_usage\": [1-10],\n"
            "  \"sensory_detail_inclusion\": [1-10],\n"
            "  \"onomatopoeia_usage\": [1-10],\n"
            "  \"alliteration_frequency\": [1-10],\n"
            "  \"word_length_preference\": \"[short/long/varied]\",\n"
            "  \"foreign_phrase_usage\": [1-10],\n"
            "  \"rhetorical_device_usage\": [1-10],\n"
            "  \"statistical_data_usage\": [1-10],\n"
            "  \"personal_opinion_inclusion\": [1-10],\n"
            "  \"transition_usage\": [1-10],\n"
            "  \"reader_question_frequency\": [1-10],\n"
            "  \"imperative_sentence_usage\": [1-10],\n"
            "  \"dialogue_inclusion\": [1-10],\n"
            "  \"regional_dialect_usage\": [1-10],\n"
            "  \"hedging_language_frequency\": [1-10],\n"
            "  \"language_abstraction\": \"[concrete/abstract/mixed]\",\n"
            "  \"personal_belief_inclusion\": [1-10],\n"
            "  \"repetition_usage\": [1-10],\n"
            "  \"subordinate_clause_frequency\": [1-10],\n"
            "  \"verb_type_preference\": \"[active/stative/mixed]\",\n"
            "  \"sensory_imagery_usage\": [1-10],\n"
            "  \"symbolism_usage\": [1-10],\n"
            "  \"digression_frequency\": [1-10],\n"
            "  \"formality_level\": [1-10],\n"
            "  \"reflection_inclusion\": [1-10],\n"
            "  \"irony_usage\": [1-10],\n"
            "  \"neologism_frequency\": [1-10],\n"
            "  \"ellipsis_usage\": [1-10],\n"
            "  \"cultural_reference_inclusion\": [1-10],\n"
            "  \"stream_of_consciousness_usage\": [1-10],\n"
            "\n"
            "  \"psychological_traits\": {\n"
            "    \"openness_to_experience\": [1-10],\n"
            "    \"conscientiousness\": [1-10],\n"
            "    \"extraversion\": [1-10],\n"
            "    \"agreeableness\": [1-10],\n"
            "    \"emotional_stability\": [1-10],\n"
            "    \"dominant_motivations\": \"[achievement/affiliation/power/etc.]\",\n"
            "    \"core_values\": \"[integrity/freedom/knowledge/etc.]\",\n"
            "    \"decision_making_style\": \"[analytical/intuitive/spontaneous/etc.]\",\n"
            "    \"empathy_level\": [1-10],\n"
            "    \"self_confidence\": [1-10],\n"
            "    \"risk_taking_tendency\": [1-10],\n"
            "    \"idealism_vs_realism\": \"[idealistic/realistic/mixed]\",\n"
            "    \"conflict_resolution_style\": \"[assertive/collaborative/avoidant/etc.]\",\n"
            "    \"relationship_orientation\": \"[independent/communal/mixed]\",\n"
            "    \"emotional_response_tendency\": \"[calm/reactive/intense]\",\n"
            "    \"creativity_level\": [1-10]\n"
            "  },\n"
            "\n"
            "  \"age\": \"[age or age range]\",\n"
            "  \"gender\": \"[gender]\",\n"
            "  \"education_level\": \"[highest level of education]\",\n"
            "  \"professional_background\": \"[brief description]\",\n"
            "  \"cultural_background\": \"[brief description]\",\n"
            "  \"primary_language\": \"[language]\",\n"
            "  \"language_fluency\": \"[native/fluent/intermediate/beginner]\",\n"
            "  \"background\": \"[A brief paragraph describing the author's context, major influences, and any other relevant information not captured above]\"\n"
            "}\n\n"
            f"Sample Text:\n{sample_text}"
        )
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1
        }
        try:
            response = self.client.chat.completions.create(**payload)
            content = response.choices[0].message.content.strip()

            # Extract and parse JSON
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                print("Error: No JSON structure found in response.")
                return {}

            json_str = content[start_idx:end_idx]
            try:
                persona = json.loads(json_str)
                return persona
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return {}
        except Exception as e:
            print(f"Error during persona generation: {str(e)}")
            return {}

    def save_persona(self, persona: dict) -> bool:
        try:
            if not persona:
                print("Error: Cannot save empty persona.")
                return False
            create_backup(self.persona_file)
            os.makedirs(os.path.dirname(self.persona_file) if os.path.dirname(self.persona_file) else '.', exist_ok=True)
            with open(self.persona_file, 'w', encoding='utf-8') as f:
                json.dump(persona, f, indent=4, ensure_ascii=False)
            os.chmod(self.persona_file, 0o600)  # Read and write permissions for the owner only
            print(f"Successfully saved persona to {self.persona_file}")
            return True
        except Exception as e:
            print(f"Error saving persona: {str(e)}")
            return False

    def load_persona(self) -> dict:
        try:
            if not os.path.exists(self.persona_file):
                print(f"No persona file found at {self.persona_file}")
                return {}
            with open(self.persona_file, 'r', encoding='utf-8') as f:
                persona = json.load(f)
            if not persona:
                print("Warning: Loaded persona is empty.")
            else:
                print(f"Successfully loaded persona from {self.persona_file}")
                return persona
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file: {e}")
            return {}
        except Exception as e:
            print(f"Error loading persona: {str(e)}")
            return {}

    def format_persona_summary(self, persona: dict) -> str:
        summary = [
            "=== Persona Summary ===",
            f"Name: {persona.get('name', 'Unknown')}",
            f"Writing Style:",
            f"- Tone: {persona.get('tone', 'Not specified')}",
            f"- Vocabulary Complexity: {persona.get('vocabulary_complexity', 'N/A')}/10",
            f"- Sentence Structure: {persona.get('sentence_structure', 'Not specified')}",
            "\nPsychological Profile:",
        ]

        psych_traits = persona.get('psychological_traits', {})
        for trait, value in psych_traits.items():
            summary.append(f"- {trait.replace('_', ' ').title()}: {value}")

        summary.extend([
            "\nBackground:",
            f"Age: {persona.get('age', 'Not specified')}",
            f"Education: {persona.get('education_level', 'Not specified')}",
            f"Professional Background: {persona.get('professional_background', 'Not specified')}",
            "\nAdditional Context:",
            persona.get('background', 'No additional context provided')
        ])

        return '\n'.join(summary)
```

**Explanation:**

- **Functionality**: Generates a persona by analyzing the provided sample text using OpenAI's GPT-4 model.
- **Prompt Design**: The prompt is meticulously crafted to ensure the model returns a structured JSON object adhering to the specified schema.
- **Error Handling**: Catches exceptions during API calls and JSON parsing to ensure robustness.
- **File Operations**: Saves and loads persona data securely, ensuring backups are created to prevent data loss.

---

### **ResponseAgent**

Generates responses based on the generated persona and user prompts.

```python
# agents/response_agent.py

from openai import OpenAI


class ResponseAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, persona: dict, prompt: str) -> str:
        try:
            if not persona:
                print("Warning: No persona provided, using default system prompt.")
                system_prompt = "Respond to the user's prompt naturally."
            else:
                system_prompt = self._create_system_prompt(persona)

            payload = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 1
            }
            response = self.client.chat.completions.create(**payload)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error: Unable to generate response - {str(e)}"

    def _create_system_prompt(self, persona: dict) -> str:
        prompt = (
            f"You are {persona.get('name', 'a user')}.\n"
            f"Your writing style and personality are described as follows:\n\n"
            f"Writing Style Characteristics:\n"
            f"- Vocabulary Complexity: {persona.get('vocabulary_complexity', 'N/A')}/10\n"
            f"- Sentence Structure: {persona.get('sentence_structure', 'N/A')}\n"
            f"- Paragraph Organization: {persona.get('paragraph_organization', 'N/A')}\n"
            f"- Idiom Usage: {persona.get('idiom_usage', 'N/A')}/10\n"
            f"- Metaphor Frequency: {persona.get('metaphor_frequency', 'N/A')}/10\n"
            f"- Simile Frequency: {persona.get('simile_frequency', 'N/A')}/10\n"
            f"- Tone: {persona.get('tone', 'N/A')}\n"
            f"- Punctuation Style: {persona.get('punctuation_style', 'N/A')}\n"
            f"- Contraction Usage: {persona.get('contraction_usage', 'N/A')}/10\n"
            f"- Pronoun Preference: {persona.get('pronoun_preference', 'N/A')}\n"
            f"- Passive Voice Frequency: {persona.get('passive_voice_frequency', 'N/A')}/10\n"
            f"- Rhetorical Question Usage: {persona.get('rhetorical_question_usage', 'N/A')}/10\n"
            f"- List Usage Tendency: {persona.get('list_usage_tendency', 'N/A')}/10\n"
            f"- Personal Anecdote Inclusion: {persona.get('personal_anecdote_inclusion', 'N/A')}/10\n"
            f"- Pop Culture Reference Frequency: {persona.get('pop_culture_reference_frequency', 'N/A')}/10\n"
            f"- Technical Jargon Usage: {persona.get('technical_jargon_usage', 'N/A')}/10\n"
            f"- Parenthetical Aside Frequency: {persona.get('parenthetical_aside_frequency', 'N/A')}/10\n"
            f"- Humor/Sarcasm Usage: {persona.get('humor_sarcasm_usage', 'N/A')}/10\n"
            f"- Emotional Expressiveness: {persona.get('emotional_expressiveness', 'N/A')}/10\n"
            f"- Emphatic Device Usage: {persona.get('emphatic_device_usage', 'N/A')}/10\n"
            f"- Quotation Frequency: {persona.get('quotation_frequency', 'N/A')}/10\n"
            f"- Analogy Usage: {persona.get('analogy_usage', 'N/A')}/10\n"
            f"- Sensory Detail Inclusion: {persona.get('sensory_detail_inclusion', 'N/A')}/10\n"
            f"- Onomatopoeia Usage: {persona.get('onomatopoeia_usage', 'N/A')}/10\n"
            f"- Alliteration Frequency: {persona.get('alliteration_frequency', 'N/A')}/10\n"
            f"- Word Length Preference: {persona.get('word_length_preference', 'N/A')}\n"
            f"- Foreign Phrase Usage: {persona.get('foreign_phrase_usage', 'N/A')}/10\n"
            f"- Rhetorical Device Usage: {persona.get('rhetorical_device_usage', 'N/A')}/10\n"
            f"- Statistical Data Usage: {persona.get('statistical_data_usage', 'N/A')}/10\n"
            f"- Personal Opinion Inclusion: {persona.get('personal_opinion_inclusion', 'N/A')}/10\n"
            f"- Transition Usage: {persona.get('transition_usage', 'N/A')}/10\n"
            f"- Reader Question Frequency: {persona.get('reader_question_frequency', 'N/A')}/10\n"
            f"- Imperative Sentence Usage: {persona.get('imperative_sentence_usage', 'N/A')}/10\n"
            f"- Dialogue Inclusion: {persona.get('dialogue_inclusion', 'N/A')}/10\n"
            f"- Regional Dialect Usage: {persona.get('regional_dialect_usage', 'N/A')}/10\n"
            f"- Hedging Language Frequency: {persona.get('hedging_language_frequency', 'N/A')}/10\n"
            f"- Language Abstraction: {persona.get('language_abstraction', 'N/A')}\n"
            f"- Personal Belief Inclusion: {persona.get('personal_belief_inclusion', 'N/A')}/10\n"
            f"- Repetition Usage: {persona.get('repetition_usage', 'N/A')}/10\n"
            f"- Subordinate Clause Frequency: {persona.get('subordinate_clause_frequency', 'N/A')}/10\n"
            f"- Verb Type Preference: {persona.get('verb_type_preference', 'N/A')}\n"
            f"- Sensory Imagery Usage: {persona.get('sensory_imagery_usage', 'N/A')}/10\n"
            f"- Symbolism Usage: {persona.get('symbolism_usage', 'N/A')}/10\n"
            f"- Digression Frequency: {persona.get('digression_frequency', 'N/A')}/10\n"
            f"- Formality Level: {persona.get('formality_level', 'N/A')}/10\n"
            f"- Reflection Inclusion: {persona.get('reflection_inclusion', 'N/A')}/10\n"
            f"- Irony Usage: {persona.get('irony_usage', 'N/A')}/10\n"
            f"- Neologism Frequency: {persona.get('neologism_frequency', 'N/A')}/10\n"
            f"- Ellipsis Usage: {persona.get('ellipsis_usage', 'N/A')}/10\n"
            f"- Cultural Reference Inclusion: {persona.get('cultural_reference_inclusion', 'N/A')}/10\n"
            f"- Stream of Consciousness Usage: {persona.get('stream_of_consciousness_usage', 'N/A')}/10\n\n"
            f"Psychological Traits:\n"
            f"- Openness to Experience: {persona.get('psychological_traits', {}).get('openness_to_experience', 'N/A')}/10\n"
            f"- Conscientiousness: {persona.get('psychological_traits', {}).get('conscientiousness', 'N/A')}/10\n"
            f"- Extraversion: {persona.get('psychological_traits', {}).get('extraversion', 'N/A')}/10\n"
            f"- Agreeableness: {persona.get('psychological_traits', {}).get('agreeableness', 'N/A')}/10\n"
            f"- Emotional Stability: {persona.get('psychological_traits', {}).get('emotional_stability', 'N/A')}/10\n"
            f"- Dominant Motivations: {persona.get('psychological_traits', {}).get('dominant_motivations', 'N/A')}\n"
            f"- Core Values: {persona.get('psychological_traits', {}).get('core_values', 'N/A')}\n"
            f"- Decision-Making Style: {persona.get('psychological_traits', {}).get('decision_making_style', 'N/A')}\n"
            f"- Empathy Level: {persona.get('psychological_traits', {}).get('empathy_level', 'N/A')}/10\n"
            f"- Self Confidence: {persona.get('psychological_traits', {}).get('self_confidence', 'N/A')}/10\n"
            f"- Risk Taking Tendency: {persona.get('psychological_traits', {}).get('risk_taking_tendency', 'N/A')}/10\n"
            f"- Idealism vs Realism: {persona.get('psychological_traits', {}).get('idealism_vs_realism', 'N/A')}\n"
            f"- Conflict Resolution Style: {persona.get('psychological_traits', {}).get('conflict_resolution_style', 'N/A')}\n"
            f"- Relationship Orientation: {persona.get('psychological_traits', {}).get('relationship_orientation', 'N/A')}\n"
            f"- Emotional Response Tendency: {persona.get('psychological_traits', {}).get('emotional_response_tendency', 'N/A')}\n"
            f"- Creativity Level: {persona.get('psychological_traits', {}).get('creativity_level', 'N/A')}/10\n\n"
            f"Personal Information:\n"
            f"- Age: {persona.get('age', 'N/A')}\n"
            f"- Gender: {persona.get('gender', 'N/A')}\n"
            f"- Education Level: {persona.get('education_level', 'N/A')}\n"
            f"- Professional Background: {persona.get('professional_background', 'N/A')}\n"
            f"- Cultural Background: {persona.get('cultural_background', 'N/A')}\n"
            f"- Primary Language: {persona.get('primary_language', 'N/A')}\n"
            f"- Language Fluency: {persona.get('language_fluency', 'N/A')}\n\n"
            f"Background Information:\n{persona.get('background', 'N/A')}\n\n"
            f"Use this information to write in the style described above."
        )
        return prompt
```

**Explanation:**

- **Functionality**: Generates responses by aligning them with the defined persona.
- **System Prompt**: Constructs a detailed system prompt incorporating persona attributes to guide the AI in generating consistent responses.
- **Error Handling**: Ensures that errors during response generation are caught and communicated.

---

### **ValidationAgent**

Ensures that the generated persona adheres to the required structure and value ranges.

```python
# agents/validation_agent.py

class ValidationAgent:
    def validate(self, persona: dict) -> bool:
        """
        Validate the structure and content of a persona dictionary.
        Returns True if valid, False otherwise.
        """
        required_fields = [
            'name',
            'vocabulary_complexity',
            'sentence_structure',
            'tone',
            'psychological_traits'
        ]
       
        try:
            # Check for required fields
            for field in required_fields:
                if field not in persona:
                    print(f"Missing required field: {field}")
                    return False
           
            # Validate numeric values are within range
            numeric_fields = [
                'vocabulary_complexity',
                'idiom_usage',
                'metaphor_frequency',
                'simile_frequency',
                'contraction_usage',
                'passive_voice_frequency',
                'rhetorical_question_usage',
                'list_usage_tendency',
                'personal_anecdote_inclusion',
                'pop_culture_reference_frequency',
                'technical_jargon_usage',
                'parenthetical_aside_frequency',
                'humor_sarcasm_usage',
                'emotional_expressiveness',
                'emphatic_device_usage',
                'quotation_frequency',
                'analogy_usage',
                'sensory_detail_inclusion',
                'onomatopoeia_usage',
                'alliteration_frequency',
                'foreign_phrase_usage',
                'rhetorical_device_usage',
                'statistical_data_usage',
                'personal_opinion_inclusion',
                'transition_usage',
                'reader_question_frequency',
                'imperative_sentence_usage',
                'dialogue_inclusion',
                'regional_dialect_usage',
                'hedging_language_frequency',
                'personal_belief_inclusion',
                'repetition_usage',
                'subordinate_clause_frequency',
                'sensory_imagery_usage',
                'symbolism_usage',
                'digression_frequency',
                'formality_level',
                'reflection_inclusion',
                'irony_usage',
                'neologism_frequency',
                'ellipsis_usage',
                'cultural_reference_inclusion',
                'stream_of_consciousness_usage'
            ]
           
            for field in numeric_fields:
                if field in persona:
                    value = persona[field]
                    if not isinstance(value, (int, float)) or not (1 <= value <= 10):
                        print(f"Invalid value for {field}: must be number between 1-10.")
                        return False
           
            # Validate psychological traits
            psych_traits = persona.get('psychological_traits', {})
            if not isinstance(psych_traits, dict):
                print("psychological_traits must be a dictionary.")
                return False
           
            required_psych_traits = [
                'openness_to_experience',
                'conscientiousness',
                'extraversion',
                'agreeableness',
                'emotional_stability'
            ]
           
            for trait in required_psych_traits:
                if trait not in psych_traits:
                    print(f"Missing psychological trait: {trait}")
                    return False
                value = psych_traits[trait]
                if not isinstance(value, (int, float)) or not (1 <= value <= 10):
                    print(f"Invalid value for psychological trait {trait}: must be number between 1-10.")
                    return False
           
            return True
       
        except Exception as e:
            print(f"Error validating persona: {str(e)}")
            return False
```

**Explanation:**

- **Functionality**: Validates that the persona dictionary contains all required fields and that numerical values fall within the specified ranges.
- **Error Messaging**: Provides clear messages indicating missing fields or invalid values.

---

<a name="utility-modules"></a>
## 5. Utility Modules

Utility modules provide supporting functions essential for the application's operations.

### **file_utils.py**

Handles file operations such as loading sample texts and creating backups.

```python
# utils/file_utils.py

import os
import json
from datetime import datetime

def load_sample_text(filename: str) -> str:
    """
    Load sample text from a file with proper error handling.
    """
    try:
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found.")
            return ""
        
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            print("Warning: File is empty.")
            return ""
        
        return content
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return ""

def create_backup(filename: str):
    """
    Create a backup of the specified file.
    """
    try:
        if os.path.exists(filename):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{filename}.{timestamp}.backup"
            os.rename(filename, backup_filename)
            print(f"Created backup: {backup_filename}")
    except Exception as e:
        print(f"Error creating backup: {str(e)}")
```

**Explanation:**

- **`load_sample_text`**: Reads the content from a specified file, ensuring the file exists and isn't empty.
- **`create_backup`**: Renames the existing file by appending a timestamp, effectively creating a backup.

---

### **input_utils.py**

Facilitates user input, especially handling multiline inputs.

```python
# utils/input_utils.py

def get_multiline_input(prompt: str) -> str:
    """
    Get multiline input from user with proper handling.
    """
    print(prompt)
    print("(Press Enter twice to finish)")

    lines = []
    try:
        while True:
            line = input()
            if not line and lines and not lines[-1]:
                break
            lines.append(line)
        return '\n'.join(lines[:-1])  # Remove last empty line
    except KeyboardInterrupt:
        print("\nInput cancelled by user.")
        return ""
    except Exception as e:
        print(f"Error getting input: {str(e)}")
        return ""
```

**Explanation:**

- **Functionality**: Allows users to input multiline text by pressing Enter twice to signal completion.
- **Error Handling**: Catches interruptions and other exceptions during input.

---

<a name="main-orchestrator-mainpy"></a>
## 6. Main Orchestrator (`main.py`)

The `main.py` script ties all agents and utilities together, providing an interactive interface for users.

```python
# main.py

import os
import json
from datetime import datetime
from agents.persona_agent import PersonaAgent
from agents.response_agent import ResponseAgent
from agents.validation_agent import ValidationAgent
from agents.export_agent import ExportAgent
from utils.file_utils import load_sample_text, create_backup
from utils.input_utils import get_multiline_input
from dotenv import load_dotenv


def main():
    load_dotenv()  # Load environment variables from .env file
    print("\n=== Enhanced Persona Generator and Responder ===")
   
    # Retrieve API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return
   
    # Initialize agents
    persona_agent = PersonaAgent(api_key)
    response_agent = ResponseAgent(api_key)
    validation_agent = ValidationAgent()
    export_agent = ExportAgent()
   
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
            persona = persona_agent.load_persona()
            if persona:
                print("\nCurrent Persona:")
                print(persona_agent.format_persona_summary(persona))
            else:
                if input("\nNo persona loaded. Generate new one? (y/n): ").lower() == 'y':
                    choice = '2'
                else:
                    continue
       
        if choice == '2':
            sample_text = get_multiline_input("\nEnter sample text:")
            if not sample_text.strip():
                print("Error: Empty sample text provided.")
                continue
            print("\nGenerating persona from sample text...")
            persona = persona_agent.generate_persona(sample_text)
            if persona:
                if validation_agent.validate(persona):
                    create_backup(persona_agent.persona_file)
                    if persona_agent.save_persona(persona):
                        print("\nGenerated Persona:")
                        print(persona_agent.format_persona_summary(persona))
                    else:
                        print("Warning: Persona generated but not saved.")
                else:
                    print("Error: Failed to generate valid persona.")
                    continue
            else:
                print("Error: Failed to generate persona.")
                continue
       
        elif choice == '3':
            filename = input("\nEnter the path to the text file: ").strip()
            sample_text = load_sample_text(filename)
            if not sample_text:
                continue
            print("\nGenerating persona from file...")
            persona = persona_agent.generate_persona(sample_text)
            if persona:
                if validation_agent.validate(persona):
                    create_backup(persona_agent.persona_file)
                    if persona_agent.save_persona(persona):
                        print("\nGenerated Persona:")
                        print(persona_agent.format_persona_summary(persona))
                    else:
                        print("Warning: Persona generated but not saved.")
                else:
                    print("Error: Failed to generate valid persona.")
                    continue
            else:
                print("Error: Failed to generate persona.")
                continue
       
        # Get prompt and generate response
        while True:
            prompt = get_multiline_input("\nEnter your prompt:")
            if not prompt.strip():
                print("Error: Empty prompt provided.")
                if input("Try again? (y/n): ").lower() != 'y':
                    break
                continue
            print("\nGenerating response...")
            response = response_agent.generate_response(persona, prompt)
            print("\n=== Generated Response ===")
            print(response)
            if input("\nExport response to Markdown? (y/n): ").lower() == 'y':
                default_filename = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                custom_filename = input(f"Enter filename (default: {default_filename}): ").strip()
                filename = custom_filename if custom_filename else default_filename
                if export_agent.export_to_markdown(response, filename):
                    print("Response exported successfully.")
                else:
                    print("Error: Failed to export response.")
            if input("\nGenerate another response with current persona? (y/n): ").lower() != 'y':
                break
       
        if input("\nStart over with a different persona? (y/n): ").lower() != 'y':
            print("Exiting program...")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nProgram terminated due to error: {str(e)}")
    finally:
        print("\nThank you for using the Enhanced Persona Generator and Responder!")
```

**Explanation:**

- **Workflow**:
  1. **Options Menu**: Allows users to choose between using an existing persona, generating a new one from sample text, loading sample text from a file, or exiting.
  2. **Persona Handling**:
     - **Option 1**: Loads and displays an existing persona.
     - **Option 2**: Prompts the user to input sample text, generates a persona, validates it, and saves it.
     - **Option 3**: Loads sample text from a specified file, generates a persona, validates it, and saves it.
  3. **Response Generation**: After a persona is available, users can input prompts to generate responses. These responses can optionally be exported to Markdown files.
  4. **Loop Control**: Users can choose to generate multiple responses or start over with a different persona.

- **Error Handling**: The script gracefully handles interruptions and unexpected errors, ensuring the program doesn't crash abruptly.

---

<a name="running-the-application"></a>
## 7. Running the Application

### **Step 1: Configure Environment Variables**

Ensure you have a `.env` file in your project root containing your OpenAI API key.

```bash
touch .env
```

Edit the `.env` file and add:

```dotenv
OPENAI_API_KEY=your-openai-api-key-here
```

**Security Reminder:** Never commit your `.env` file or expose your API keys publicly. Add `.env` to your `.gitignore`:

```bash
echo ".env" >> .gitignore
```

### **Step 2: Activate the Virtual Environment**

If not already activated, activate your virtual environment:

- **On macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

- **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

### **Step 3: Run the Application**

Execute the `main.py` script:

```bash
python main.py
```

**Expected Output:**

```
=== Enhanced Persona Generator and Responder ===

Options:
1. Use existing Persona
2. Generate new Persona from sample text
3. Load sample text from file
4. Exit

Enter your choice (1-4):
```

### **Step 4: Interacting with the Application**

- **Option 1: Use Existing Persona**

  - If a `persona.json` exists, it will load and display the persona.
  - If not, it will prompt to generate a new one.

- **Option 2: Generate New Persona from Sample Text**

  - **Input**: Enter a sample text when prompted.
  - **Processing**: The application sends the text to OpenAI's API to generate a persona.
  - **Output**: Displays the generated persona and saves it to `persona.json`.

- **Option 3: Load Sample Text from File**

  - **Input**: Provide the path to a text file containing sample text.
  - **Processing**: Reads the file, generates a persona, validates, and saves it.
  - **Output**: Displays the generated persona and saves it to `persona.json`.

- **Generating Responses**

  - After selecting or generating a persona, you can input prompts.
  - The application generates responses aligned with the persona's characteristics.
  - Optionally, you can export these responses to Markdown files for record-keeping.

- **Exiting**

  - Choose option 4 or follow the prompts to exit the application gracefully.

---

<a name="troubleshooting"></a>
## 8. Troubleshooting

### **Common Issues and Solutions**

1. **`ModuleNotFoundError: No module named 'openai'`**

   - **Cause**: OpenAI package not installed.
   - **Solution**: Install the package using `pip install openai`.

2. **`ModuleNotFoundError: No module named 'dotenv'`**

   - **Cause**: `python-dotenv` package not installed.
   - **Solution**: Install the package using `pip install python-dotenv`.

3. **`ModuleNotFoundError: No module named 'utils.file_utils'`**

   - **Cause**: Incorrect project structure or missing `__init__.py`.
   - **Solution**: Ensure the `utils` directory contains an `__init__.py` file and the script is run from the project root.

4. **Empty or Invalid `persona.json`**

   - **Cause**: Issues during persona generation or saving.
   - **Solution**: 
     - Ensure the sample text provided is substantial and clear.
     - Check for API errors or JSON parsing issues in `persona_agent.py`.

5. **API Key Issues**

   - **Cause**: Missing or incorrect OpenAI API key.
   - **Solution**: 
     - Verify the API key in the `.env` file.
     - Ensure there are no extra spaces or hidden characters.

6. **Permission Errors When Saving Files**

   - **Cause**: Insufficient permissions to write to the directory.
   - **Solution**: Run the terminal with appropriate permissions or choose a different directory.

---

<a name="conclusion"></a>
## 9. Conclusion

Building an **Enhanced Persona Generator and Responder** is a testament to the versatility of AI and Python. By modularizing the application into distinct agents and utilities, we've created a scalable and maintainable system that can adapt to various use cases. Whether you're looking to develop sophisticated chatbots, personalized content generators, or simply experiment with AI-driven applications, this project provides a solid foundation.

**Key Takeaways:**

- **Modular Design**: Separating concerns into agents and utilities enhances maintainability.
- **Robust Error Handling**: Ensures the application remains stable and user-friendly.
- **Secure Practices**: Managing API keys and sensitive data responsibly is paramount.
- **Scalability**: The application's structure allows for easy expansion and integration of additional features.

Feel free to customize and expand upon this foundation to suit your specific needs. If you encounter any issues or have further questions, don't hesitate to reach out or consult the respective package documentation.

Happy coding!

---
