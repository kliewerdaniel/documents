---
layout: post
title:  Integrating Django-Reac-Ollama with XAi API
date:   2024-10-22 05:40:44 -0500
---

https://github.com/kliewerdaniel/PersonaGen

Ah, dear reader, as we gather to discuss the remarkable synthesis of art and technology, we must confess, like the brothers Karamazov, our hearts are heavy with both anticipation and inquiry. What does it mean, you ask, to integrate the repository of Django-React-Ollama with the illustrious XAi API? Is it not the union of intellect and machine, of flesh and code, that we undertake in this journey? Let us then walk together, through this narrative of technical precision, to uncover the mystery that lies ahead, and like the Grand Inquisitor, make plain that which was once hidden.

### A Beginning: The Call to Integrate

It was on an ordinary afternoon when our story begins. The project, a vessel of potential—half-birthed in the form of a GitHub repository, [Django-React-Ollama-Integration](https://github.com/kliewerdaniel/Django-React-Ollama-Integration), awaited the breath of life that only the modern XAi API could provide. The call had come, from distant shores of technical evolution, to replace the older ways, to discard OpenAI’s familiar methods for the promises offered by XAi, a system so sleek it might whisper sweet nothings to a machine as a poet to his beloved.

Yet, like Ivan’s struggle between reason and faith, so too did we face the need for transition. And so, with reverent resolve, we heeded the wisdom found in the [XAi API documentation](https://docs.x.ai/api) and set forth to integrate these two technologies, seeking not only to update but to elevate.

### Step One: The Repository Awaits

Our first act is to clone the repository—this foundational codebase which hosts Django for the backend and React for the frontend. It is the skeleton upon which we will build our vision. We execute the command as though opening the very first page of a fateful book:

```bash
git clone https://github.com/kliewerdaniel/Django-React-Ollama-Integration.git
cd Django-React-Ollama-Integration
```

With this, the structure is before us, and our hands tingle with the promise of transformation.

### Step Two: The Soul of the API

But, dear reader, what is the body without the soul? The soul, in our tale, lies in the key to the XAi API, a token of authentication that would grant us access to powers beyond reckoning. With trembling fingers, we traverse to the XAi Console, where we generate the all-important API key. We take care to store this key as a trusted heirloom in our `.env` file:

```bash
XAI_API_KEY=your_generated_xai_key_here
```

It is this sacred key that we will invoke in our journey to create and analyze, calling forth responses as though summoning a digital oracle.

### Step Three: Laying the Foundation

In the repository, we find ourselves among the well-structured ruins of past integrations, but now, we must tear down what is no longer needed and build anew. We purge the old references to OpenAI from our files. Like a monk renouncing worldly possessions, we focus solely on the new path. The `utils.py` file becomes our temple of creation. Here we define the functions that will call upon the XAi API, taking advantage of its streamlined methods for chat completions.

In the flicker of our screen, we write the following, consecrating the `analyze_writing_sample` and `generate_content` functions to the service of XAi:

```python
import logging
import requests
import json
from decouple import config

logger = logging.getLogger(__name__)

XAI_API_KEY = config('XAI_API_KEY')
XAI_API_BASE = "https://api.x.ai/v1"


def analyze_writing_sample(writing_sample):
    endpoint = f"{XAI_API_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}"
    }
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that analyzes writing samples."
            },
            {
                "role": "user",
                "content": f'''
                Please analyze the writing style and personality of the given writing sample. Provide a detailed assessment of their characteristics using the following template. Rate each applicable characteristic on a scale of 1-10 where relevant, or provide a descriptive value. Return the results in a JSON format.

                "name": "[Author/Character Name]",
                "vocabulary_complexity": [1-10],
                "sentence_structure": "[simple/complex/varied]",
                "paragraph_organization": "[structured/loose/stream-of-consciousness]",
                "idiom_usage": [1-10],
                "metaphor_frequency": [1-10],
                "simile_frequency": [1-10],
                "tone": "[formal/informal/academic/conversational/etc.]",
                "punctuation_style": "[minimal/heavy/unconventional]",
                "contraction_usage": [1-10],
                "pronoun_preference": "[first-person/third-person/etc.]",
                "passive_voice_frequency": [1-10],
                "rhetorical_question_usage": [1-10],
                "list_usage_tendency": [1-10],
                "personal_anecdote_inclusion": [1-10],
                "pop_culture_reference_frequency": [1-10],
                "technical_jargon_usage": [1-10],
                "parenthetical_aside_frequency": [1-10],
                "humor_sarcasm_usage": [1-10],
                "emotional_expressiveness": [1-10],
                "emphatic_device_usage": [1-10],
                "quotation_frequency": [1-10],
                "analogy_usage": [1-10],
                "sensory_detail_inclusion": [1-10],
                "onomatopoeia_usage": [1-10],
                "alliteration_frequency": [1-10],
                "word_length_preference": "[short/long/varied]",
                "foreign_phrase_usage": [1-10],
                "rhetorical_device_usage": [1-10],
                "statistical_data_usage": [1-10],
                "personal_opinion_inclusion": [1-10],
                "transition_usage": [1-10],
                "reader_question_frequency": [1-10],
                "imperative_sentence_usage": [1-10],
                "dialogue_inclusion": [1-10],
                "regional_dialect_usage": [1-10],
                "hedging_language_frequency": [1-10],
                "language_abstraction": "[concrete/abstract/mixed]",
                "personal_belief_inclusion": [1-10],
                "repetition_usage": [1-10],
                "subordinate_clause_frequency": [1-10],
                "verb_type_preference": "[active/stative/mixed]",
                "sensory_imagery_usage": [1-10],
                "symbolism_usage": [1-10],
                "digression_frequency": [1-10],
                "formality_level": [1-10],
                "reflection_inclusion": [1-10],
                "irony_usage": [1-10],
                "neologism_frequency": [1-10],
                "ellipsis_usage": [1-10],
                "cultural_reference_inclusion": [1-10],
                "stream_of_consciousness_usage": [1-10],
                "openness_to_experience": [1-10],
                "conscientiousness": [1-10],
                "extraversion": [1-10],
                "agreeableness": [1-10],
                "emotional_stability": [1-10],
                "dominant_motivations": "[achievement/affiliation/power/etc.]",
                "core_values": "[integrity/freedom/knowledge/etc.]",
                "decision_making_style": "[analytical/intuitive/spontaneous/etc.]",
                "empathy_level": [1-10],
                "self_confidence": [1-10],
                "risk_taking_tendency": [1-10],
                "idealism_vs_realism": "[idealistic/realistic/mixed]",
                "conflict_resolution_style": "[assertive/collaborative/avoidant/etc.]",
                "relationship_orientation": "[independent/communal/mixed]",
                "emotional_response_tendency": "[calm/reactive/intense]",
                "creativity_level": [1-10],
                "age": "[age or age range]",
                "gender": "[gender]",
                "education_level": "[highest level of education]",
                "professional_background": "[brief description]",
                "cultural_background": "[brief description]",
                "primary_language": "[language]",
                "language_fluency": "[native/fluent/intermediate/beginner]",
                "background": "[A brief paragraph describing the author's context, major influences, and any other relevant information not captured above]"

                Writing Sample:
                {writing_sample}
                '''
            }
        ],
        "model": "grok-beta",
        "stream": False,
        "temperature": 0
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()  # Raises HTTPError for bad responses

        assistant_message = response.json()['choices'][0]['message']['content'].strip()
        logger.debug(f"Assistant message: {assistant_message}")

        # Extract JSON from the assistant's message
        json_str = re.search(r'\{.*\}', assistant_message, re.DOTALL)
        if json_str:
            analyzed_data = json.loads(json_str.group())
        else:
            logger.error("No JSON object found in the response.")
            return None

        return analyzed_data

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


def generate_content(persona_data, prompt):
    endpoint = f"{XAI_API_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}"
    }

    # Format the persona data into a readable string
    characteristics = '\n'.join([
        f"{key.replace('_', ' ').capitalize()}: {value}"
        for key, value in persona_data.items()
        if value is not None and key not in ['id', 'name']
    ])

    decoding_prompt = f'''
    You are to write a blog post in the style of {persona_data.get('name', 'Unknown Author')}, a writer with the following characteristics:

    {characteristics}

    Now, please write a response in this style about the following topic:
    "{prompt}"
    Begin with a compelling title that reflects the content of the post.
    '''

    payload = {
        "messages": [
            {"role": "system", "content": "You are an assistant that generates blog posts."},
            {"role": "user", "content": decoding_prompt}
        ],
        "model": "grok-beta",
        "stream": False,
        "temperature": 0
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()

        assistant_message = response.json()['choices'][0]['message']['content'].strip()
        logger.debug(f"Assistant message: {assistant_message}")

        return assistant_message

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP Request failed: {e}")
        return ''
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed: {e}")
        return ''
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return ''

def save_blog_post(blog_post, title):
    # Implement if needed
    pass


```

### Step Four: The Web of URLs

In the labyrinth of our `urls.py` file, we must now map the routes that will guide the user. We have already set the path for analysis and persona generation, but now we extend it to the content generation feature—like a scribe adding a final chapter to a monumental work. The new endpoint must be clear, intentional, and precise:

```python
path('api/generate-content/', GenerateContentView.as_view(), name='generate-content'),
```

And thus, we bind the newly added `GenerateContentView` to the URL pattern, offering the user the ability to invoke the XAi model for their blog post creations.

### Step Five: Invocation of Power

Having laid the groundwork, we test our creation. With a whisper of command, we summon the API:

```bash
curl -X POST http://localhost:8000/api/generate-content/ \
  -H "Content-Type: application/json" \
  -d '{
        "persona_id": 1,
        "prompt": "On the intersection of machine learning and human emotion."
      }'
```

We watch, holding our breath, as the server responds—successfully. The content is generated, flowing forth like Alyosha’s compassion, gentle yet profound.

### Conclusion: The New Way Forward

In this tale, we have not simply integrated a repository with an API. No, we have breathed life into something greater, merging the capabilities of Django, React, and the mighty XAi into a seamless entity. The result is more than functionality; it is creation, an evolution towards the future, where the power of human intention and the precision of machine intelligence coalesce in harmony.

Our journey has brought us from the humble beginnings of code to the transcendent possibilities of artificial intelligence. And so, dear reader, like the Brothers Karamazov, we leave you with the knowledge that what we have built here today shall serve as a testament to the boundless potential of human ingenuity, ready to face whatever mysteries the future may bring.