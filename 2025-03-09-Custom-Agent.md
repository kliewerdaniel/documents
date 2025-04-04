---
layout: post
title:  Building a Custom AI Agent Framework with Next.js and Ollama
date:   2025-03-09 09:42:44 -0500
---
# Building a Custom AI Agent Framework with Next.js and Ollama

In today's rapidly evolving AI landscape, agent-based systems have emerged as powerful tools for task automation and complex problem-solving. This blog post will guide you through creating a sophisticated Next.js application with a custom AI agent framework powered by Ollama, an open-source local LLM runner.

## What We're Building

We'll develop an application where users can submit goals like "Create a content calendar for social media" and watch as an AI agent systematically works through the problem, documenting its reasoning and delivering high-quality results. The beauty of this approach is that everything runs locally on your machine using Ollama, providing privacy benefits and eliminating API costs.


## Key Concepts in Our Agent Framework

Before diving into the code, let's understand the core concepts that make our custom agent framework powerful:

### 1. Step-Based Task Decomposition

Complex tasks become manageable when broken down into smaller steps. Our agent takes a user's goal and automatically divides it into logical steps, similar to how a human would approach a complex problem:

```typescript
// Sample task decomposition
const steps = [
  "Analyze target audience and choose platforms",
  "Establish content themes and post types",
  "Create first half of weekly content calendar",
  "Create second half of weekly content calendar",
  "Add engagement strategies and hashtag recommendations"
];
```

### 2. Reasoning Before Action

For each step, our agent first explains its reasoning before taking action. This creates transparency and allows users to understand the agent's thought process:

```typescript
// Sample reasoning for a step
const reasoning = "Before creating content, I need to understand who we're targeting and which platforms would be most effective for a coffee shop. Typically, Instagram and Facebook work well for food/beverage businesses.";
```

### 3. Streaming Progress Updates

Users receive real-time updates as the agent works through each step, maintaining engagement and giving visibility into the process:

```typescript
// Sending a real-time update to the client
await sendUpdate({
  type: 'log',
  message: `📝 Step ${step.number}: ${step.description}`
});
```

### 4. Contextual Memory

Each step builds upon previous steps, maintaining context throughout the execution:

```typescript
const stepPrompt = `
  Task: "${this.goal}"
  Step ${stepNumber}/${Math.min(steps.length, this.maxSteps)}: ${stepDescription}
  Previous steps: ${this.steps.map(s => `Step ${s.number}: ${s.description} -> ${s.output?.substring(0, 100)}...`).join('\n')}
  Execute this step and provide the output. Be thorough but focused on just this step.
`;
```

## Setting Up the Project

Let's begin by creating a Next.js project and installing dependencies:

```bash
npx create-next-app@latest next-ollama-agent
cd next-ollama-agent
npm install dotenv react-markdown
```

Next, download and install [Ollama](https://ollama.com/), then pull the Mistral model:

```bash
ollama pull mistral
```

## Building the Custom Agent Class

The heart of our application is the `Agent` class, which handles the execution of tasks:

```typescript
// src/lib/agent.ts
export interface Step {
  number: number;
  description: string;
  reasoning?: string;
  output?: string;
}

export interface AgentResult {
  goal: string;
  steps: Step[];
  output: string;
}

export type StepCallback = (step: Step) => Promise<void> | void;

export class Agent {
  private goal: string;
  private maxSteps: number;
  private onStepComplete?: StepCallback;
  private steps: Step[] = [];

  constructor(options: {
    goal: string;
    maxSteps?: number;
    onStepComplete?: StepCallback;
  }) {
    this.goal = options.goal;
    this.maxSteps = options.maxSteps || 5;
    this.onStepComplete = options.onStepComplete;
  }

  async execute(): Promise<AgentResult> {
    // Step 1: Task analysis
    const taskAnalysis = await this.callOllama(
      `Analyze this task: "${this.goal}". Break it down into ${this.maxSteps} clear steps that would lead to a high-quality result. Return a JSON array of step descriptions only, no additional text.`
    );

    // Parse steps from the model response
    let steps: string[] = [];
    try {
      const parsed = JSON.parse(this.extractJSON(taskAnalysis));
      steps = Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      // Fallback extraction with regex if JSON parsing fails
      const stepRegex = /\d+\.\s*(.*?)(?=\d+\.|$)/gs;
      const matches = [...taskAnalysis.matchAll(stepRegex)];
      steps = matches.map(match => match[1].trim());
    }

    // Default steps if extraction fails
    if (steps.length === 0) {
      steps = ["Analyze the problem", "Generate solution", "Refine the output"];
    }

    // Execute each step
    for (let i = 0; i < Math.min(steps.length, this.maxSteps); i++) {
      const stepNumber = i + 1;
      const stepDescription = steps[i];

      // Generate reasoning for this step
      const reasoning = await this.callOllama(
        `For the task: "${this.goal}", I am on step ${stepNumber}: "${stepDescription}". Explain your reasoning for how you'll approach this step. Keep it clear and concise.`
      );

      // Execute the step with context from previous steps
      const stepPrompt = `
        Task: "${this.goal}"
        Step ${stepNumber}/${Math.min(steps.length, this.maxSteps)}: ${stepDescription}
        Previous steps: ${this.steps.map(s => `Step ${s.number}: ${s.description} -> ${s.output?.substring(0, 100)}...`).join('\n')}
        Execute this step and provide the output. Be thorough but focused on just this step.
      `;
      const stepOutput = await this.callOllama(stepPrompt);

      // Record the step
      const step: Step = {
        number: stepNumber,
        description: stepDescription,
        reasoning,
        output: stepOutput
      };
      this.steps.push(step);

      // Notify via callback if provided
      if (this.onStepComplete) {
        await this.onStepComplete(step);
      }
    }

    // Generate final comprehensive output
    const finalPrompt = `
      You've been working on: "${this.goal}"
      You've completed the following steps:
      ${this.steps.map(s => `Step ${s.number}: ${s.description}`).join('\n')}
      Now, compile all of your work into a comprehensive final output that achieves the original goal.
      Format your response using Markdown for readability.
    `;
    const finalOutput = await this.callOllama(finalPrompt);

    return {
      goal: this.goal,
      steps: this.steps,
      output: finalOutput
    };
  }

  private async callOllama(prompt: string): Promise<string> {
    try {
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'mistral',
          prompt: prompt,
          stream: false,
        }),
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.statusText}`);
      }

      const data = await response.json();
      return data.response;
    } catch (error) {
      console.error('Error calling Ollama:', error);
      return `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
    }
  }

  private extractJSON(text: string): string {
    // Try to extract JSON from the text
    const jsonRegex = /(\[.*\]|\{.*\})/s;
    const match = text.match(jsonRegex);
    return match ? match[0] : '[]';
  }
}
```

## Building the Frontend

Our frontend uses React and Next.js to create a clean, responsive interface:

```tsx
// src/app/page.tsx
"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

export default function Home() {
  const [goal, setGoal] = useState<string>("");
  const [logs, setLogs] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [result, setResult] = useState<string>("");
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the bottom of logs
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);

  const handleRunAgent = async () => {
    if (!goal.trim() || isRunning) return;
    setIsRunning(true);
    setLogs(["🤖 Initializing custom agent powered by Ollama..."]);
    setResult("");

    try {
      const response = await fetch("/api/run-agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to run agent");
      }

      // Use streaming for real-time updates
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      
      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const text = decoder.decode(value);
          try {
            // Handle multiple JSON objects in the same chunk
            const jsonObjects = text.split('\n').filter(line => line.trim());
            
            for (const jsonStr of jsonObjects) {
              if (!jsonStr.trim()) continue;
              const data = JSON.parse(jsonStr);
              
              if (data.type === "log") {
                setLogs(logs => [...logs, data.message]);
              } else if (data.type === "result") {
                setResult(data.content);
              }
            }
          } catch (error) {
            console.error("Error parsing stream data:", error);
          }
        }
      }
    } catch (error: any) {
      setLogs(logs => [...logs, `❌ Error: ${error.message}`]);
    } finally {
      setIsRunning(false);
      setLogs(logs => [...logs, "✅ Agent execution completed"]);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-8 max-w-5xl mx-auto">
      <h1 className="text-4xl font-bold mb-3">AI Agent Workspace</h1>
      <h2 className="text-xl text-gray-600 mb-8">Powered by Custom Agent Framework + Ollama</h2>
      
      <div className="w-full space-y-8">
        {/* Goal Input Section */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-lg font-semibold mb-3">What would you like the agent to accomplish?</h3>
          <div className="flex gap-3">
            <input
              type="text"
              placeholder="e.g., Create a marketing plan for a new product launch"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              className="flex-1 p-3 border rounded-md text-gray-800 focus:ring-2 focus:ring-blue-500"
              disabled={isRunning}
            />
            <button
              onClick={handleRunAgent}
              disabled={isRunning || !goal.trim()}
              className={`px-6 py-3 rounded-md font-medium transition ${
                isRunning ?
                "bg-gray-300 text-gray-600" :
                "bg-blue-600 text-white hover:bg-blue-700"
              }`}
            >
              {isRunning ? "Working..." : "Run Agent"}
            </button>
          </div>
        </div>

        {/* Agent Logs Section */}
        <div className="bg-gray-50 rounded-lg shadow-md">
          <div className="bg-gray-100 p-4 rounded-t-lg border-b">
            <h3 className="text-lg font-semibold">Agent Thinking Process</h3>
          </div>
          <div className="p-4 max-h-80 overflow-y-auto">
            {logs.length === 0 ? (
              <p className="text-gray-500 italic">Agent logs will appear here...</p>
            ) : (
              <div className="space-y-2">
                {logs.map((log, index) => (
                  <div key={index} className="p-3 bg-white rounded border">
                    {log}
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            )}
          </div>
        </div>

        {/* Result Section */}
        {result && (
          <div className="bg-white rounded-lg shadow-md">
            <div className="bg-green-100 p-4 rounded-t-lg border-b">
              <h3 className="text-lg font-semibold text-green-800">Agent Result</h3>
            </div>
            <div className="p-6 prose max-w-none">
              <ReactMarkdown>{result}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
```

## Creating the API Endpoint

Next, let's create the serverless API endpoint that will run our agent and stream results back to the client:

```typescript
// src/app/api/run-agent/route.ts
import { NextResponse } from 'next/server';
import { Agent, Step } from '@/lib/agent';

export const runtime = 'nodejs';

export async function POST(request: Request) {
  // Initialize the response encoder for streaming
  const encoder = new TextEncoder();
  const stream = new TransformStream();
  const writer = stream.writable.getWriter();

  // Function to send updates to the client
  const sendUpdate = async (data: any) => {
    await writer.write(encoder.encode(JSON.stringify(data) + '\n'));
  };

  // Process the request in the background while streaming updates
  const processRequest = async () => {
    try {
      // Parse the request body
      const { goal } = await request.json();
      
      if (!goal || typeof goal !== 'string') {
        await sendUpdate({
          type: 'log',
          message: '❌ Error: Please provide a valid goal'
        });
        writer.close();
        return;
      }

      // Initialize the custom agent
      const agent = new Agent({
        goal: goal,
        maxSteps: 5,
        onStepComplete: async (step: Step) => {
          await sendUpdate({
            type: 'log',
            message: `📝 Step ${step.number}: ${step.description}`
          });
          
          if (step.reasoning) {
            await sendUpdate({
              type: 'log',
              message: `🤔 Reasoning: ${step.reasoning}`
            });
          }
        }
      });

      // Log initialization
      await sendUpdate({
        type: 'log',
        message: `🧠 Analyzing task: "${goal}"`
      });

      // Execute the agent
      const result = await agent.execute();

      // Send the final result
      await sendUpdate({
        type: 'result',
        content: result.output
      });

      // Close the stream
      writer.close();
    } catch (error: any) {
      console.error('Agent execution error:', error);
      await sendUpdate({
        type: 'log',
        message: `❌ Error: ${error.message || 'Unknown error occurred'}`
      });
      writer.close();
    }
  };

  // Start processing in the background
  processRequest();

  // Return the stream response immediately
  return new NextResponse(stream.readable, {
    headers: {
      'Content-Type': 'application/json',
      'Transfer-Encoding': 'chunked',
    },
  });
}
```

## Advanced Enhancements

After getting the basic application working, we can enhance our agent framework with more advanced capabilities:

### 1. Adding Specialized Tools

Let's enhance our agent with tools that can perform specific functions:

```typescript
// src/lib/tools.ts
export interface Tool {
  name: string;
  description: string;
  execute: (input: string) => Promise<string>;
}

// Sample search tool
export const searchTool: Tool = {
  name: 'search',
  description: 'Search the web for information',
  async execute(query: string): Promise<string> {
    // This is a mock implementation - in a real app, you'd integrate with a search API
    return `Simulated search results for: ${query}\n\n1. First relevant result\n2. Second relevant result\n3. Third relevant result`;
  }
};
```

### 2. Multi-Agent Workflows

For complex tasks, we can create workflows with multiple specialized agents:

```typescript
// src/lib/workflow.ts
import { Agent, AgentResult } from './agent';

export async function runResearchAndSynthesisWorkflow(topic: string, updateCallback: (message: string) => Promise<void>) {
  await updateCallback(`Starting research workflow on: ${topic}`);
  
  // Research agent gathers information
  const researchAgent = new Agent({
    goal: `Research key facts about: ${topic}`,
    maxSteps: 3,
    onStepComplete: async (step) => {
      await updateCallback(`Research step ${step.number}: ${step.description}`);
    }
  });
  
  await updateCallback("Starting research phase...");
  const researchResult = await researchAgent.execute();
  
  // Analysis agent evaluates the research
  const analysisAgent = new Agent({
    goal: `Analyze these research findings and identify key insights: ${researchResult.output}`,
    maxSteps: 2,
    onStepComplete: async (step) => {
      await updateCallback(`Analysis step ${step.number}: ${step.description}`);
    }
  });
  
  await updateCallback("Starting analysis phase...");
  const analysisResult = await analysisAgent.execute();
  
  // Synthesis agent creates final output
  const synthesisAgent = new Agent({
    goal: `Create a comprehensive report on ${topic} using this research and analysis:
    Research: ${researchResult.output}
    Analysis: ${analysisResult.output}`,
    maxSteps: 3,
    onStepComplete: async (step) => {
      await updateCallback(`Synthesis step ${step.number}: ${step.description}`);
    }
  });
  
  await updateCallback("Starting synthesis phase...");
  const finalResult = await synthesisAgent.execute();
  
  await updateCallback("Workflow complete!");
  
  return {
    topic,
    research: researchResult.output,
    analysis: analysisResult.output,
    synthesis: finalResult.output
  };
}
```

### 3. Memory and Database Integration

For persistence between sessions, we can integrate a database:

```typescript
// Using Prisma with SQLite for simplicity
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function saveSession(goal: string, output: string, logs: string[]) {
  return await prisma.session.create({
    data: {
      goal,
      output,
      logs: JSON.stringify(logs),
    },
  });
}

export async function getSessions() {
  return await prisma.session.findMany({
    orderBy: {
      createdAt: 'desc',
    },
  });
}
```

## Testing and Evaluation

When testing your agent, try these diverse goals to evaluate its capabilities:

- **Business Tasks**: "Create a marketing strategy for a new fitness app"
- **Creative Tasks**: "Write a short story about time travel with a twist ending"
- **Analytical Problems**: "Analyze the pros and cons of remote work for a small business"

## Conclusion

We've built a sophisticated AI agent framework that leverages Next.js and Ollama to create a powerful task automation system. This combination offers several key advantages:

1. **Local Privacy**: By running models through Ollama, you maintain control of your data without sending it to external API services.
2. **Cost Efficiency**: Eliminate per-token or per-request charges by running inference locally.
3. **Architectural Flexibility**: Our custom agent implementation provides a structured framework that can be extended as needed.
4. **Realtime Feedback**: The streaming architecture keeps users informed of progress throughout the execution.

The step-based approach with explicit reasoning creates transparency that builds user trust in the AI system. By mastering these technologies, you're well-positioned to build intelligent applications that blend the best of human creativity with AI capabilities.

What tasks would you automate with your custom agent framework?

---

*Want to explore more advanced agent patterns and LLM applications? Follow our blog for more tutorials and insights into the world of AI development.*