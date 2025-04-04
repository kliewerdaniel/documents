---
layout: post
title: Ollama Chunking
description: Mastering Text Chunking with Ollama A Comprehensive Guide to Advanced Processing
date:   2025-03-28 02:42:44 -0500
---

# Mastering Text Chunking with Ollama: A Comprehensive Guide to Advanced Processing

In today's world of AI and large language models, one of the most common challenges developers face is handling text that exceeds a model's context window. Ollama, while powerful for running local language models, shares this limitation with other LLMs. This comprehensive guide will explore advanced chunking techniques to effectively process large documents with Ollama while maintaining coherence and context.

## Understanding Chunking in the Context of Ollama

Chunking is the process of dividing large text into smaller, manageable segments that fit within a model's token limit. Ollama, which provides access to models like Llama, Mistral, and others, has specific token limitations depending on the model you're using. Effective chunking isn't just about breaking text apart—it's about doing so intelligently to preserve meaning across segments.

## Why Advanced Chunking Matters for Ollama

When working with Ollama, proper chunking techniques become essential for several reasons:

1. **Context Window Constraints**: Most models accessible through Ollama have context windows ranging from 2K to 8K tokens, limiting how much text they can process at once.

2. **Memory Efficiency**: Even if a model technically supports larger contexts, processing smaller chunks can reduce RAM usage, allowing Ollama to run smoothly on machines with limited resources.

3. **Coherence Across Chunks**: Without proper chunking strategies, the model might lose the thread of thought between segments, resulting in disjointed or contradictory outputs.

4. **Processing Efficiency**: Well-designed chunking allows for parallel processing and can significantly reduce the time needed to handle large documents.

## Advanced Chunking Strategies for Ollama

Let's explore several sophisticated chunking approaches that go beyond basic text splitting:

### 1. Semantic Chunking

Rather than chunking based solely on character or token count, semantic chunking divides text based on meaning and context.

```python
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load SpaCy model for semantic understanding
nlp = spacy.load("en_core_web_md")

def semantic_chunking(text, max_tokens=1000, overlap=100):
    # Break into sentences first
    sentences = sent_tokenize(text)
    
    # Get sentence embeddings
    sentence_embeddings = [nlp(sentence).vector for sentence in sentences]
    
    # Track token count (approximate)
    token_counts = [len(sentence.split()) for sentence in sentences]
    
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for i, sentence in enumerate(sentences):
        # If adding this sentence would exceed our limit, start a new chunk
        if current_token_count + token_counts[i] > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # For overlap, find the most semantically similar sentences to include
            if overlap > 0 and len(current_chunk) > 0:
                # Get embeddings for current chunk sentences
                current_embs = sentence_embeddings[i-len(current_chunk):i]
                # Find sentences with highest similarity to include in overlap
                similarities = cosine_similarity([sentence_embeddings[i]], current_embs)[0]
                overlap_indices = np.argsort(similarities)[-int(overlap/10):]  # Heuristic for number of sentences
                
                # Add overlapping sentences to new chunk
                current_chunk = [sentences[i-len(current_chunk)+idx] for idx in overlap_indices]
                current_token_count = sum(token_counts[i-len(current_chunk)+idx] for idx in overlap_indices)
            else:
                current_chunk = []
                current_token_count = 0
        
        current_chunk.append(sentence)
        current_token_count += token_counts[i]
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
```

This approach ensures that semantically related content stays together, providing Ollama with more coherent chunks to process.

### 2. Hierarchical Chunking

Hierarchical chunking creates a tree-like structure where larger documents are first divided into major sections, then subsections, and finally into token-sized chunks.

```python
def hierarchical_chunking(document, max_tokens=1000):
    # First level: Split by major section headers
    sections = re.split(r'# [A-Za-z\s]+\n', document)
    
    # Second level: For each section, split by sub-headers
    subsections = []
    for section in sections:
        if not section.strip():
            continue
        subsecs = re.split(r'## [A-Za-z\s]+\n', section)
        subsections.extend([s for s in subsecs if s.strip()])
    
    # Final level: Split subsections into token-sized chunks
    final_chunks = []
    for subsection in subsections:
        words = subsection.split()
        for i in range(0, len(words), max_tokens):
            chunk = ' '.join(words[i:i+max_tokens])
            if chunk.strip():
                final_chunks.append(chunk)
    
    return final_chunks
```

This method is particularly useful for processing structured documents like academic papers or technical documentation with Ollama.

### 3. Sliding Window Chunking with Context Retention

This advanced technique maintains continuity by creating overlapping windows of text:

```python
def sliding_window_chunking(text, window_size=800, stride=600, context_size=200):
    """
    Process text using a sliding window approach that maintains context
    - window_size: The main processing window size in tokens
    - stride: How far to move the window for each chunk (smaller than window_size creates overlap)
    - context_size: How much previous context to include with each chunk
    """
    words = text.split()
    chunks = []
    
    # Initialize with first chunk having no previous context
    for i in range(0, len(words), stride):
        if i == 0:
            # First chunk has no previous context
            chunk = words[i:i+window_size]
        else:
            # Calculate how much previous context to include
            context_start = max(0, i-context_size)
            
            # Create a marker showing where previous context ends and new content begins
            context_part = words[context_start:i]
            new_part = words[i:i+window_size-len(context_part)]
            
            # Combine with a special separator
            chunk = (
                "--- PREVIOUS CONTEXT ---\n" + 
                " ".join(context_part) + 
                "\n--- NEW CONTENT ---\n" + 
                " ".join(new_part)
            )
        
        if chunk:
            chunks.append(chunk if isinstance(chunk, str) else " ".join(chunk))
        
        # If we've processed all words, break
        if i + window_size >= len(words):
            break
    
    return chunks
```

This approach is particularly effective for narrative text where continuity between chunks is critical for Ollama to maintain the flow of ideas.

## Implementing Advanced Chunking with Ollama

Now let's see how we can apply these chunking strategies with Ollama's API for practical use cases:

```python
import json
import requests

def process_with_ollama(chunks, model="llama2", system_prompt=None):
    """
    Process a list of text chunks with Ollama
    """
    responses = []
    
    # Base URL for Ollama API
    url = "http://localhost:11434/api/generate"
    
    for i, chunk in enumerate(chunks):
        # Create a metadata-rich prompt for context
        prompt = f"[Chunk {i+1} of {len(chunks)}]\n\n{chunk}"
        
        # Prepare the request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
            
        # Make the API call
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Check for HTTP errors
            
            # Extract and store the response
            result = response.json()
            responses.append(result["response"])
            
            print(f"Processed chunk {i+1}/{len(chunks)}")
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            responses.append(f"Error: {str(e)}")
    
    return responses
```

### Advanced Example: Document Analysis with Context Maintenance

Let's create a more complex workflow that uses semantic chunking for document analysis while maintaining context between chunks:

```python
def analyze_document_with_ollama(document_path, model="llama2:13b"):
    """
    Analyze a large document by:
    1. Reading the document
    2. Creating semantic chunks
    3. Processing each chunk while maintaining context
    4. Synthesizing a coherent analysis
    """
    # Read the document
    with open(document_path, 'r', encoding='utf-8') as f:
        document = f.read()
    
    # Create semantic chunks
    print("Creating semantic chunks...")
    chunks = semantic_chunking(document, max_tokens=1800, overlap=200)
    print(f"Document divided into {len(chunks)} semantic chunks")
    
    # Process each chunk with Ollama
    system_prompt = """
    You are analyzing a document that has been divided into chunks.
    For each chunk:
    1. Identify key points, arguments, and evidence
    2. Note how these connect to previous chunks if applicable
    3. Maintain a coherent understanding of the document as it progresses
    """
    
    print("Processing chunks with Ollama...")
    chunk_analyses = process_with_ollama(chunks, model=model, system_prompt=system_prompt)
    
    # Create a final synthesis prompt
    synthesis_prompt = "Below are analyses of different sections of a document:\n\n"
    for i, analysis in enumerate(chunk_analyses):
        synthesis_prompt += f"SECTION {i+1} ANALYSIS:\n{analysis}\n\n"
    
    synthesis_prompt += """
    Based on these section analyses, provide a comprehensive synthesis of the entire document.
    Include:
    1. The main thesis or argument
    2. Key supporting points and evidence
    3. Any significant counterarguments or limitations
    4. Overall evaluation of the document's effectiveness
    """
    
    # Process the synthesis with Ollama
    print("Creating final synthesis...")
    synthesis_payload = {
        "model": model,
        "prompt": synthesis_prompt,
        "stream": False
    }
    
    response = requests.post("http://localhost:11434/api/generate", json=synthesis_payload)
    synthesis = response.json()["response"]
    
    return {
        "num_chunks": len(chunks),
        "chunk_analyses": chunk_analyses,
        "synthesis": synthesis
    }
```

## Advanced Chunking Considerations for Ollama

### Token Estimation with Different Models

Different Ollama models have varying tokenization methods. Here's a simple utility to help estimate token counts across models:

```python
def estimate_tokens(text, model_type="llama2"):
    """
    Estimate token count for different Ollama models
    """
    # Average ratios of tokens to characters for different model families
    # These are approximations and will vary
    token_ratios = {
        "llama2": 0.25,   # ~4 characters per token
        "mistral": 0.23,  # ~4.3 characters per token
        "mpt": 0.22,      # ~4.5 characters per token
        "falcon": 0.26    # ~3.8 characters per token
    }
    
    ratio = token_ratios.get(model_type.lower(), 0.25)  # Default to llama2 ratio
    
    # Simple estimation based on character count
    return int(len(text) * ratio)
```

### Handling Code and Technical Content

Code and technical content require special chunking considerations:

```python
def chunk_code_document(document):
    """
    Specialized chunking for technical documents with code blocks
    """
    # Split document by Markdown code blocks
    parts = re.split(r'(```[\w]*\n[\s\S]*?\n```)', document)
    
    chunks = []
    current_chunk = ""
    current_token_est = 0
    
    for part in parts:
        # If this is a code block, try to keep it intact
        is_code_block = part.startswith('```') and part.endswith('```')
        part_token_est = estimate_tokens(part)
        
        # If adding this part would exceed our limit, start a new chunk
        if current_token_est + part_token_est > 1800 and current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""
            current_token_est = 0
            
        # If it's a code block that alone exceeds token limit, we need to split it
        if is_code_block and part_token_est > 1800:
            # Process the large code block separately
            if current_chunk:  # Save any accumulated content first
                chunks.append(current_chunk)
                current_chunk = ""
                current_token_est = 0
                
            # Split code by lines, preserving syntax highlighting info
            code_lang = re.match(r'```([\w]*)\n', part)
            code_lang = code_lang.group(1) if code_lang else ""
            
            code_content = part[3+len(code_lang):-3].strip()
            code_lines = code_content.split('\n')
            
            code_chunks = []
            current_code_chunk = f"```{code_lang}\n"
            current_code_tokens = estimate_tokens(current_code_chunk)
            
            for line in code_lines:
                line_tokens = estimate_tokens(line + '\n')
                if current_code_tokens + line_tokens > 1700:  # Leave room for the closing ```
                    current_code_chunk += "```"
                    code_chunks.append(current_code_chunk)
                    current_code_chunk = f"```{code_lang}\n{line}\n"
                    current_code_tokens = estimate_tokens(current_code_chunk)
                else:
                    current_code_chunk += line + '\n'
                    current_code_tokens += line_tokens
            
            # Add the last code chunk if not empty
            if current_code_chunk != f"```{code_lang}\n":
                current_code_chunk += "```"
                code_chunks.append(current_code_chunk)
                
            chunks.extend(code_chunks)
        else:
            # Regular text or small code block
            current_chunk += part
            current_token_est += part_token_est
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
```

## Parallel Processing with Ollama

For large documents, you can process multiple chunks in parallel to save time:

```python
import concurrent.futures

def process_chunks_in_parallel(chunks, model="llama2", max_workers=4):
    """
    Process multiple chunks in parallel with Ollama
    """
    def process_chunk(chunk_data):
        i, chunk = chunk_data
        url = "http://localhost:11434/api/generate"
        prompt = f"[Chunk {i+1} of {len(chunks)}]\n\n{chunk}"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error processing chunk {i+1}: {str(e)}"
    
    results = [None] * len(chunks)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks for processing
        future_to_index = {executor.submit(process_chunk, (i, chunk)): i 
                          for i, chunk in enumerate(chunks)}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                results[index] = f"Error: {str(e)}"
    
    return results
```

## Best Practices for Ollama Chunking

Based on extensive testing with various Ollama models, here are some best practices:

1. **Retain Document Structure**: When possible, align chunk boundaries with natural document divisions like paragraphs, sections, or sentences.

2. **Context Windows**: Use a smaller effective window size than the model's maximum to leave room for the model's response.

3. **Model-Specific Tuning**: 
   - Llama models generally perform better with slightly smaller chunks (1500-1800 tokens)
   - Mistral models can often handle larger coherent chunks (2000+ tokens)
   - Adjust based on your specific model

4. **Metadata Enhancement**: Include metadata in each chunk that indicates its position and relationship to other chunks.

5. **Adaptive Chunking**: Consider the content type—code, technical text, and narrative content may benefit from different chunking strategies.

6. **System Prompts**: Use clear system prompts to tell Ollama how to handle chunked content.

## Common Chunking Pitfalls with Ollama

When implementing chunking with Ollama, be aware of these common issues:

1. **Mid-Sentence Splitting**: Avoid splitting sentences between chunks when possible, as this can disrupt the model's understanding.

2. **Losing Key Context**: Critical information mentioned early in a document might be missing from later chunks if not properly carried forward.

3. **Tokenizer Mismatches**: Remember that character or word counts aren't perfect proxies for token counts, which can lead to chunks that exceed token limits.

4. **Neglecting Document Structure**: Splitting without respect to document structure (e.g., cutting across headers or code blocks) often produces poor results.

5. **Overloading Context Windows**: Very dense information-rich chunks may overwhelm the model even if they're within token limits.

## Conclusion: Mastering Ollama with Advanced Chunking

Advanced chunking techniques are essential for getting the most out of Ollama, especially when working with larger documents or complex content. By implementing semantic, hierarchical, or sliding window chunking approaches, you can process content that far exceeds the model's native context window while maintaining coherence and accuracy.

The techniques outlined in this guide will help you build more sophisticated applications with Ollama that can handle real-world document processing tasks efficiently. By understanding the nuances of different chunking strategies and how they interact with different Ollama models, you can create systems that make the most of local LLM capabilities without being constrained by context window limitations.

Remember that the ideal chunking strategy depends on your specific use case, content type, and chosen model. Experiment with the approaches outlined here and adapt them to your particular needs for optimal results.