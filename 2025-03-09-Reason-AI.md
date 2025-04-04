---
layout: post
title:  ReasonAI
date:   2025-03-09 10:42:44 -0500
---

# Building Intelligent AI Agents with Local Privacy
## A Deep Dive into the Ollama Reasoning Agent Framework

*A Developer's Guide to Core Concepts & Practical Implementation*

---

In today's AI landscape, most powerful agent frameworks require sending your data to cloud services, raising privacy concerns and dependency issues. This guide explores a revolutionary alternative: **building sophisticated reasoning agents that run entirely on your local machine** using the reasonai03 framework.

By combining Next.js with Ollama's local LLM capabilities, we'll create AI agents that:
- Process sensitive data without external API calls
- Provide transparent reasoning steps as they work
- Break complex goals into executable tasks—all while preserving privacy

Let's dive into the architecture, implementation patterns, and practical knowledge to build your own local-first AI agents.


## Why Local-First AI Agents Matter

Cloud-based AI services dominate the landscape, but this approach comes with inherent limitations:

1. **Privacy concerns** when handling sensitive data
2. **Network dependency** issues during outages
3. **Subscription costs** that scale with usage
4. **Black-box operation** with limited visibility into reasoning

The reasonai03 framework addresses these challenges by:
- Running models completely on your hardware
- Providing streaming insights into the agent's reasoning process
- Using task decomposition to tackle complex problems
- Maintaining full developer control over the execution environment

```bash
# The basic concept: run everything locally
ollama run llama2       # Local LLM server
npm run dev             # Next.js frontend
```

## Core Concept #1: Task Decomposition Architecture


### The Problem It Solves

When a user asks an AI to "Plan a European vacation," this seemingly simple request requires dozens of distinct reasoning steps. Traditional approaches either:
1. Attempt to solve everything in one massive prompt (leading to hallucinations)
2. Use rigid, pre-defined workflows (lacking flexibility)

### How It Works

The framework implements a recursive task decomposition pattern:

```javascript
// lib/agent/decompose.js
export async function decomposeTask(goal) {
  // 1. Ask LLM to identify necessary steps
  const decompositionPrompt = `
    Break this complex task into maximally parallelizable steps:
    GOAL: ${goal}
    
    Respond in JSON format:
    {
      "steps": [
        {
          "id": "step_1",
          "description": "...",
          "depends_on": [] // IDs of steps that must complete first
        }
      ]
    }
  `;
  
  // 2. Get step plan from model
  const stepPlan = await ollama.generate({
    model: 'llama2',
    prompt: decompositionPrompt
  });
  
  // 3. Parse and validate the plan
  const { steps } = JSON.parse(stepPlan);
  
  return steps;
}
```

### Implementation Insights

The framework employs several advanced patterns to make decomposition robust:

#### 1. Dependency Tracking

```javascript
// Example of how steps relate to each other
const steps = [
  {
    id: "find_flights",
    description: "Research flight options to Europe",
    depends_on: [] // Can start immediately
  },
  {
    id: "book_hotels",
    description: "Book accommodations in selected cities",
    depends_on: ["select_cities"] // Must wait for city selection
  },
  {
    id: "select_cities",
    description: "Choose which cities to visit based on interests",
    depends_on: [] // Can start immediately
  }
];
```

This dependency graph enables:
- **Parallel execution** of independent steps
- **Efficient sequencing** of dependent steps
- **Progress visualization** for the user

#### 2. Contextual Memory

As steps complete, their outputs become context for future steps:

```javascript
// lib/agent/execute.js
async function executeStep(step, context) {
  const relevantContext = step.depends_on.map(id => {
    return context[id]; // Look up results from previous steps
  });
  
  const executionPrompt = `
    GOAL: ${step.description}
    
    PREVIOUS RESULTS:
    ${relevantContext.join('\n')}
    
    Provide your solution:
  `;
  
  // ...run model and return result
}
```

## Core Concept #2: Real-Time Reasoning Streams


Traditional AI interactions are "black boxes" - you submit a request and wait for a complete response. The reasonai03 framework changes this by **streaming the agent's thought process in real-time**.

### Server Implementation

The framework leverages Server-Sent Events (SSE) to create a live stream from server to client:

```typescript
// app/api/agent/route.ts
export async function POST(req: Request) {
  const { goal } = await req.json();
  const encoder = new TextEncoder();
  
  const stream = new ReadableStream({
    async start(controller) {
      // Callback that sends tokens as they're generated
      const sendToken = (token: string) => {
        controller.enqueue(encoder.encode(token));
      };
      
      try {
        // Main agent execution with streaming callback
        await runAgent({ goal }, sendToken);
      } catch (error) {
        sendToken(`\nError: ${error.message}`);
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}
```

### Client Implementation

On the frontend, React components connect to this stream:

```tsx
// app/components/AgentConsole.tsx
import { useState, useEffect } from 'react';

export function AgentConsole({ goal }) {
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  
  async function startAgent() {
    setIsRunning(true);
    setOutput('');
    
    try {
      const response = await fetch('/api/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal }),
      });
      
      if (!response.body) throw new Error('No response body');
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const text = decoder.decode(value);
        setOutput(prev => prev + text);
      }
    } catch (error) {
      setOutput(prev => prev + '\nConnection error: ' + error.message);
    } finally {
      setIsRunning(false);
    }
  }
  
  return (
    <div className="agent-console">
      <button 
        onClick={startAgent} 
        disabled={isRunning}
      >
        {isRunning ? 'Running...' : 'Start Agent'}
      </button>
      
      <pre className="output-area">
        {output || 'Agent output will appear here...'}
      </pre>
    </div>
  );
}
```

### Key Benefits

This streaming approach provides several advantages:

1. **Transparency** - Users can see exactly how the agent approaches problems
2. **Early feedback** - Catch errors or misunderstandings before full execution
3. **Better UX** - No "waiting in the dark" for long-running operations

## Core Concept #3: Local-First AI with Ollama

At the heart of the framework is Ollama, an open-source tool for running LLMs locally:

```bash
# Install Ollama (Mac/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull models you need
ollama pull llama2        # General reasoning
ollama pull mistral       # Faster, smaller model
ollama pull codellama     # Code generation tasks

# Start the Ollama server (automatically runs in background)
ollama serve
```

### Privacy Architecture

The framework's privacy-preserving architecture has several layers:

1. **No data transmission** - All data processing happens on your machine
2. **Local model serving** - Ollama runs models fully on your hardware
3. **Isolation through workers** - Node.js worker threads separate execution contexts
4. **Optional at-rest encryption** - Data can be encrypted when stored

```javascript
// lib/ollama.js - Core interface to local models
export async function generate({ model, prompt, options = {} }) {
  try {
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: model || 'llama2',
        prompt,
        options,
      }),
    });
    
    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Ollama error: ${error}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Ollama generate error:', error);
    throw new Error(`Failed to communicate with Ollama: ${error.message}`);
  }
}
```

### Model Selection Guide

Different tasks require different models. Here's a practical guide:

| Task Type | Recommended Model | RAM Needs | Strengths |
|-----------|------------------|-----------|-----------|
| General reasoning | llama2:13b | 16GB+ | Good balance of speed/quality |
| Complex planning | llama2:70b | 32GB+ | Highest reasoning quality |
| Quick responses | mistral | 8GB+ | Fast with decent quality |
| Code generation | codellama | 16GB+ | Specialized for programming |
| Vision tasks | llava | 16GB+ | Can process images |

```typescript
// Example of model selection logic
function selectModelForTask(task) {
  if (task.type === 'code_generation') {
    return 'codellama';
  } else if (task.complexity === 'high') {
    return 'llama2:70b';
  } else if (task.priority === 'speed') {
    return 'mistral';
  } else {
    return 'llama2:13b'; // Default
  }
}
```

## Practical Implementation: Building Your First Agent

Now that we understand the core concepts, let's build a practical agent from the ground up.

### Step 1: Project Setup

```bash
# Create a Next.js project
npx create-next-app@latest reasonai-agent
cd reasonai-agent

# Install dependencies
npm install localforage uuid
```

### Step 2: Agent Core Implementation

Create the main agent execution file:

```typescript
import { v4 as uuid } from 'uuid';
import { decomposeTask } from './decompose';
import { executeStep } from './execute';

export async function runAgent(
  { goal }: { goal: string },
  streamCallback: (text: string) => void
) {
  const sessionId = uuid();
  
  try {
    // Start by streaming a confirmation
    streamCallback(`📋 Working on: "${goal}"\n\n`);
    
    // Step 1: Decompose the task
    streamCallback(`🔍 Analyzing task and creating plan...\n`);
    const steps = await decomposeTask(goal);
    
    streamCallback(`✅ Created plan with ${steps.length} steps:\n\n`);
    steps.forEach((step, i) => {
      streamCallback(`${i+1}. ${step.description}\n`);
    });
    streamCallback(`\n`);
    
    // Step 2: Execute each step with proper dependency handling
    const results = {};
    const completed = new Set();
    
    // Keep going until all steps are complete
    while (completed.size < steps.length) {
      // Find steps where all dependencies are satisfied
      const availableSteps = steps.filter(step => {
        if (completed.has(step.id)) return false;
        
        // Check if all dependencies are completed
        return step.depends_on.every(depId => completed.has(depId));
      });
      
      if (availableSteps.length === 0) {
        throw new Error("Deadlock detected - unable to proceed with execution plan");
      }
      
      // Execute available steps in parallel
      streamCallback(`🔄 Executing ${availableSteps.length} steps in parallel...\n\n`);
      
      await Promise.all(availableSteps.map(async (step) => {
        streamCallback(`⏳ Working on: ${step.description}...\n`);
        
        try {
          // Get context from dependencies
          const context = {};
          step.depends_on.forEach(depId => {
            context[depId] = results[depId];
          });
          
          // Execute the step
          const result = await executeStep(step, context, goal);
          results[step.id] = result;
          completed.add(step.id);
          
          streamCallback(`✅ Completed: ${step.description}\n`);
          streamCallback(`   Result: ${result.substring(0, 100)}${result.length > 100 ? '...' : ''}\n\n`);
        } catch (error) {
          streamCallback(`❌ Error in step "${step.description}": ${error.message}\n\n`);
          throw error;
        }
      }));
    }
    
    // Step 3: Generate final summary
    streamCallback(`🎉 All steps completed! Generating final report...\n\n`);
    
    const summaryPrompt = `
      You've completed all steps to: ${goal}
      
      Here are the results from each step:
      ${Object.entries(results).map(([id, result]) => {
        const step = steps.find(s => s.id === id);
        return `STEP: ${step.description}\nRESULT: ${result}\n\n`;
      }).join('')}
      
      Please provide a comprehensive final report that synthesizes all this information.
    `;
    
    const summary = await generateWithOllama('llama2', summaryPrompt);
    streamCallback(`📊 FINAL REPORT:\n\n${summary}\n\n`);
    
    return { sessionId, success: true };
  } catch (error) {
    streamCallback(`\n❌ Agent execution failed: ${error.message}\n`);
    return { sessionId, success: false, error: error.message };
  }
}

// Helper function for Ollama API calls
async function generateWithOllama(model, prompt) {
  const response = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, prompt, stream: false }),
  });
  
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Ollama error: ${error}`);
  }
  
  const data = await response.json();
  return data.response;
}
```

### Step 3: Task Decomposition

```typescript
import { generateWithOllama } from '../ollama';
import { v4 as uuid } from 'uuid';

export async function decomposeTask(goal: string) {
  const decompositionPrompt = `
    You are an expert project manager who breaks down complex tasks into well-organized steps.
    
    RULES:
    1. Create 3-7 logical steps to accomplish the goal
    2. Each step should be self-contained and focused
    3. Identify dependencies between steps (which steps must complete before others)
    4. Steps with no dependencies should be executable in parallel
    
    GOAL: ${goal}
    
    Respond in valid JSON format:
    {
      "steps": [
        {
          "id": "unique_id_1",
          "description": "Clear description of the step",
          "depends_on": [] // Array of step IDs this step depends on
        },
        ...
      ]
    }
  `;
  
  try {
    const result = await generateWithOllama('llama2', decompositionPrompt);
    let parsed;
    
    try {
      // Find the JSON part in the response (models sometimes add explanations)
      const jsonMatch = result.match(/\{[\s\S]*\}/);
      parsed = JSON.parse(jsonMatch ? jsonMatch[0] : result);
    } catch (e) {
      throw new Error(`Failed to parse task decomposition: ${e.message}`);
    }
    
    if (!parsed.steps || !Array.isArray(parsed.steps)) {
      throw new Error('Invalid decomposition format: missing steps array');
    }
    
    // Assign IDs if they're missing
    parsed.steps = parsed.steps.map(step => ({
      ...step,
      id: step.id || `step_${uuid().substring(0, 8)}`,
      depends_on: step.depends_on || []
    }));
    
    return parsed.steps;
  } catch (error) {
    console.error('Decomposition error:', error);
    throw new Error(`Failed to decompose task: ${error.message}`);
  }
}
```

### Step 4: Step Execution

```typescript
import { generateWithOllama } from '../ollama';

export async function executeStep(step, context, originalGoal) {
  // Build context from dependent steps
  const dependencyContext = step.depends_on.map(depId => {
    const result = context[depId];
    return `Previous Step Result (${depId}): ${result}`;
  }).join('\n\n');
  
  const executionPrompt = `
    You are an AI assistant working on a multi-step task.
    
    ORIGINAL GOAL: ${originalGoal}
    
    CURRENT STEP: ${step.description}
    
    ${dependencyContext ? `CONTEXT FROM PREVIOUS STEPS:\n${dependencyContext}\n\n` : ''}
    
    Your task is to complete ONLY this specific step thoroughly and accurately.
    Provide a complete solution for this step only.
  `;
  
  try {
    return await generateWithOllama('llama2', executionPrompt);
  } catch (error) {
    throw new Error(`Step execution failed: ${error.message}`);
  }
}
```

### Step 5: Creating the API Route

```typescript
import { NextRequest } from 'next/server';
import { runAgent } from '@/lib/agent';

export async function POST(req: NextRequest) {
  try {
    const { goal } = await req.json();
    
    if (!goal || typeof goal !== 'string') {
      return new Response(
        JSON.stringify({ error: 'Missing or invalid goal' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }
    
    // Create encoder for streaming
    const encoder = new TextEncoder();
    
    // Create a stream for the response
    const stream = new ReadableStream({
      async start(controller) {
        const sendToken = (token: string) => {
          controller.enqueue(encoder.encode(token));
        };
        
        try {
          await runAgent({ goal }, sendToken);
        } catch (error) {
          sendToken(`\nError: ${error.message}`);
          console.error('Agent execution error:', error);
        } finally {
          controller.close();
        }
      },
    });
    
    // Return the stream as a streamed response
    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
  } catch (error) {
    console.error('API route error:', error);
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
```

### Step 6: Creating the User Interface

```tsx
'use client';

import { useState, useRef, useEffect } from 'react';
import styles from './page.module.css';

export default function Home() {
  const [goal, setGoal] = useState('');
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const outputRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll output
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);
  
  async function runAgent() {
    if (!goal.trim()) return;
    
    setIsRunning(true);
    setOutput('');
    
    try {
      const response = await fetch('/api/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal }),
      });
      
      if (!response.body) {
        throw new Error('No response body');
      }
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      let done = false;
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        
        if (value) {
          const text = decoder.decode(value);
          setOutput(prev => prev + text);
        }
      }
    } catch (error) {
      setOutput(prev => prev + `\n\nError: ${error.message}`);
    } finally {
      setIsRunning(false);
    }
  }
  
  return (
    <main className={styles.main}>
      <h1 className={styles.title}>Local Reasoning Agent</h1>
      
      <div className={styles.inputContainer}>
        <input
          type="text"
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder="Enter your goal (e.g., 'Plan a weekend trip to New York')"
          className={styles.input}
          disabled={isRunning}
        />
        <button 
          onClick={runAgent}
          disabled={isRunning || !goal.trim()}
          className={styles.button}
        >
          {isRunning ? 'Running...' : 'Start Agent'}
        </button>
      </div>
      
      <div className={styles.outputContainer} ref={outputRef}>
        <pre className={styles.output}>
          {output || 'Agent output will appear here...'}
        </pre>
      </div>
    </main>
  );
}
```

```css
.main {
  display: flex;
  flex-direction: column;
  padding: 2rem;
  max-width: 1000px;
  margin: 0 auto;
  min-height: 100vh;
}

.title {
  margin-bottom: 2rem;
  text-align: center;
}

.inputContainer {
  display: flex;
  margin-bottom: 1.5rem;
  gap: 0.5rem;
}

.input {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 1rem;
}

.button {
  padding: 0.75rem 1.5rem;
  background-color: #0070f3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
}

.button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

.outputContainer {
  flex: 1;
  border: 1px solid #eaeaea;
  border-radius: 4px;
  padding: 1rem;
  background-color: #f7f7f7;
  overflow-y: auto;
  max-height: 70vh;
}

.output {
  white-space: pre-wrap;
  margin: 0;
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
  line-height: 1.5;
}
```

## Advanced Customization

Once you have the basic framework running, you can extend it in powerful ways:

### Adding Domain-Specific Knowledge

You can enhance your agent with specialized knowledge by customizing the prompts:


```typescript
export const PROMPTS = {
  // For travel planning domain
  travel: {
    decompose: `
      You are a professional travel planner. Break down the travel planning process into logical steps.
      
      Consider:
      - Transportation research
      - Accommodation booking
      - Activity planning
      - Budget management
      - Visa/document requirements
      
      GOAL: {goal}
      
      Create a comprehensive plan in JSON format:
      {
        "steps": [...]
      }
    `,
    
    // Other travel-specific prompts
  },
  
  // For research paper domain
  research: {
    decompose: `
      You are a research assistant helping to plan an academic paper. Break down the research process.
      
      Consider:
      - Literature review
      - Hypothesis formation
      - Methodology planning
      - Data collection strategy
      - Analysis approach
      
      GOAL: {goal}
      
      Create a comprehensive plan in JSON format:
      {
        "steps": [...]
      }
    `,
    
    // Other research-specific prompts
  }
};

export function getPrompt(domain, type, variables) {
  const template = PROMPTS[domain]?.[type] || PROMPTS.general[type];
  
  return Object.entries(variables).reduce((prompt, [key, value]) => {
    return prompt.replace(new RegExp(`{${key}}`, 'g'), value);
  }, template);
}
```

### Adding Tools and External Capabilities

For certain tasks, the agent may need to interact with external services or tools:


```typescript
export const tools = {
  // Get weather information
  weather: async (location) => {
    try {
      // Using a local-only weather API
      const res = await fetch(`http://localhost:3001/api/weather?location=${encodeURIComponent(location)}`);
      if (!res.ok) throw new Error('Weather service unavailable');
      return await res.json();
    } catch (error) {
      return { error: error.message };
    }
  },
  
  // Calculate distance between locations
  distance: (origin, destination) => {
    // Simple demo implementation (would use local libraries in real app)
    return { 
      distance: "1,234 km",
      duration: "Approximately 13 hours driving"
    };
  },
  
  // Current date/time (no external API needed)
  datetime: () => {
    const now = new Date();
    return {
      date: now.toLocaleDateString(),
      time: now.toLocaleTimeString(),
      timestamp: now.getTime()
    };
  }
};

// Tool usage function for the agent
export async function useTool(toolName, ...args) {
  if (!tools[toolName]) {
    return { error: `Tool '${toolName}' not found` };
  }
  
  try {
    return await tools[toolName](...args);
  } catch (error) {
    return { error: error.message };
  }
}
```

## Real-World Use Case: Trip Planner

Let's see the framework in action with a complete use case:

User Input: "Plan a 5-day trip to Japan with a $3000 budget"



### Agent Execution Flow:

1. **Decomposition**: The agent breaks this into:
   - Research flights to Japan
   - Find accommodations within budget
   - Determine visa requirements for Japan
   - Create daily itinerary options
   - Plan transportation between destinations
   - Allocate budget across categories

2. **Parallel Execution**: Starts with flights, visa research, and budget allocation in parallel

3. **Sequential Steps**: Once flights are chosen, proceeds to accommodation and transportation planning

4. **Interactive Output**: Throughout execution, the agent provides:
   ```
   🔍 Analyzing task and creating plan...
   ✅ Created plan with 6 steps:
   
   1. Research flights to Japan
   2. Find accommodations within budget
   3. Determine visa requirements for Japan
   4. Create daily itinerary options
   5. Plan transportation between destinations
   6. Allocate budget across categories
   
   🔄 Executing 3 steps in parallel...
   
   ⏳ Working on: Research flights to Japan...
   ⏳ Working on: Determine visa requirements for Japan...
   ⏳ Working on: Allocate budget across categories...
   
   ✅ Completed: Determine visa requirements for Japan
      Result: For US citizens, no visa is required for stays up to 90 days in Japan...
   ```

5. **Final Report**: Comprehensive trip plan with all details

## Best Practices & Troubleshooting

### Performance Optimization

For smooth operation on consumer hardware:

1. **Right-size your models**: Use smaller models for simpler tasks
   ```bash
   # For quick tasks, use mistral instead of llama2
   ollama pull mistral
   ```

2. **Implement caching**: Store results for expensive operations
   ```javascript
   // Simple cache implementation
   const cache = new Map();
   
   async function cachedGenerate(prompt, model) {
     const cacheKey = `${model}:${prompt}`;
     if (cache.has(cacheKey)) {
       return cache.get(cacheKey);
     }
     
     const result = await generateWithOllama(model, prompt);
     cache.set(cacheKey, result);
     return result;
   }
   ```

3. **Batch related operations**: Group small tasks when possible
   ```javascript
   // Instead of many small prompts
   const results = await Promise.all([
     generateWithOllama('mistral', 'Small task 1'),
     generateWithOllama('mistral', 'Small task 2'),
     generateWithOllama('mistral', 'Small task 3')
   ]);
   ```

### Error Handling

Robust error handling is crucial for agent reliability:


```typescript
export async function withErrorHandling(fn, options = {}) {
  const { 
    maxRetries = 3, 
    retryDelay = 1000,
    fallback = null
  } = options;
  
  let lastError;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      // Log the error with attempt count
      console.error(`Attempt ${attempt}/${maxRetries} failed:`, error);
      
      // Check if we should retry based on error type
      if (error.message.includes('context length exceeded')) {
        console.log('Context length error - trying with truncated input');
        // We could modify fn here to use less context
      } else if (error.message.includes('connection failed')) {
        console.log(`Connection error - retrying in ${retryDelay}ms`);
        await new Promise(r => setTimeout(r, retryDelay));
      } else if (attempt >= maxRetries) {
        // If this is the last attempt, don't retry
        break;
      }
    }
  }
  
  // All attempts failed
  if (fallback) {
    console.log('All attempts failed, using fallback');
    return fallback();
  }
  
  throw lastError;
}
```

### Common Issues and Solutions

| Problem | Possible Cause | Solution |
|---------|---------------|----------|
| "Ollama not responding" | Ollama service not running | Run `ollama serve` in terminal |
| "Model not found" | Model not downloaded | Run `ollama pull llama2` |
| Stream freezes or timeouts | Long-running operations | Implement heartbeat messages |
| JSON parsing errors | Model output not in valid JSON | Use structured output format with retries |
| High RAM usage | Large model loaded | Use smaller models or increase swap space |

## Conclusion

The reasonai03 framework demonstrates that powerful AI agents can run entirely on local hardware, preserving privacy while providing sophisticated reasoning capabilities. By combining Next.js with Ollama, we've created an architecture that:

1. Breaks complex goals into manageable steps through task decomposition
2. Provides transparency through real-time reasoning streams
3. Processes all data locally to maintain privacy
4. Can be extended with domain-specific knowledge and tools

This pattern represents a fundamental shift from cloud-dependent AI to sovereign, local-first artificial intelligence. As models continue to become more efficient, the capabilities of these local agents will only grow stronger.

Ready to build your own local reasoning agent? Clone the repository and start experimenting:

```bash
git clone https://github.com/kliewerdaniel/reasonai03.git
cd reasonai03
npm install && npm run dev
```

The future of AI isn't just in the cloud—it's running right on your own machine.

---

*Note: This implementation focuses on the core architecture. For production use, consider adding authentication, persistent storage, and enhanced error handling.*