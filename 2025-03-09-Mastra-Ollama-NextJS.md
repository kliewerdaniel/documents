---
layout: post
title:  Building an AI-Powered Next.js Application with Mastra and Ollama
date:   2025-03-09 07:42:44 -0500
description: "Learn to build a complete Next.js application that uses Mastra and Ollama to create AI agents for task automation with step-by-step code examples and best practices."
---
# Building an AI-Powered Next.js Application with Mastra and Ollama

## 1. Introduction

The world of AI is rapidly evolving, with agent-based systems emerging as powerful tools for task automation and complex problem-solving. In this comprehensive tutorial, we'll walk through building a sophisticated Next.js application that integrates **Mastra** (a production-ready AI agent framework) with **Ollama** (an open-source local LLM runner) to create an intelligent task automation system.

**What is Mastra?** Mastra is an enterprise-grade framework for creating autonomous AI agents with advanced reasoning capabilities, built-in workflow management, and production-ready features. It enables developers to build reliable, observable AI agents that can decompose complex tasks into manageable steps and execute them methodically.

**What is Ollama?** Ollama allows you to run large language models (LLMs) locally on your machine rather than relying on cloud APIs. This approach provides privacy benefits, reduces costs, and eliminates API latency issues—making it ideal for development and privacy-sensitive applications.

By the end of this tutorial, you'll have created a web application where users can submit goals like "Create a content calendar for social media" or "Analyze quarterly sales data," and watch as an AI agent systematically works through the problem, documenting its reasoning and producing high-quality results.

## 2. Setting Up the Project

### 2.1 Prerequisites

Before starting, ensure you have:
- Node.js 18+ installed
- Basic knowledge of React and Next.js
- Ollama installed (we'll cover this in detail)
- A Mastra account (we'll help you set this up)

### 2.2 Creating a Next.js Application

Let's begin by creating a fresh Next.js project:

```bash
npx create-next-app@latest mastra-ollama-app
cd mastra-ollama-app
```

During the setup, select the following options:
- Would you like to use TypeScript? → Yes (for type safety)
- Would you like to use ESLint? → Yes
- Would you like to use Tailwind CSS? → Yes (for styling)
- Would you like to use the src/ directory? → Yes (for organization)
- Would you like to use App Router? → Yes (for modern routing)
- Would you like to customize the default import alias? → No

### 2.3 Installing Dependencies

Install the Mastra client library and other necessary packages:

```bash
npm install @mastraai/client ollama-js dotenv react-markdown
```

### 2.4 Setting Up Ollama

1. Visit [Ollama's official website](https://ollama.com/) and download the installer for your operating system.
2. Install Ollama following the on-screen instructions.
3. Open a terminal and pull the Mistral model (a powerful open-source LLM):

```bash
ollama pull mistral
```

This will download the model, which may take several minutes depending on your internet connection.

### 2.5 Setting Up Mastra

1. Visit [Mastra's website](https://mastra.ai) and create an account
2. Generate an API key from your dashboard
3. Create a `.env.local` file in your project root with:

```
MASTRA_API_KEY=your_api_key_here
```

### 2.6 Verifying Your Setup

Let's ensure Ollama is working correctly:

```bash
ollama run mistral "What can you help me with today?"
```

You should see a coherent response from the model, confirming Ollama is properly installed.

## 3. Understanding the Frontend (React + Next.js)

Now, let's build a responsive, user-friendly interface for our agent application.

### 3.1 Creating the Home Page Component

Create or replace the file at `src/app/page.tsx` with:

```tsx
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
    setLogs(["🤖 Initializing Mastra agent powered by Ollama..."]);
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
          const data = JSON.parse(text);
          
          if (data.type === "log") {
            setLogs(logs => [...logs, data.message]);
          } else if (data.type === "result") {
            setResult(data.content);
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
      <h2 className="text-xl text-gray-600 mb-8">Powered by Mastra + Ollama</h2>
      
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

### 3.2 Understanding the Frontend Components

The frontend is built with several key features:

1. **State Management**:
   - `goal`: Stores the user's input task
   - `logs`: Maintains an array of execution logs
   - `isRunning`: Tracks the agent's execution state
   - `result`: Stores the final output from the agent

2. **Streaming Response Handling**:
   - Uses the Fetch API with a reader/decoder to process streamed updates
   - Separates log updates from final results

3. **UI Components**:
   - A clean input section for submitting tasks
   - A scrollable log window showing the agent's reasoning process
   - A formatted result section using React Markdown for rich text display

4. **User Experience Enhancements**:
   - Auto-scrolling logs to keep the latest updates visible
   - Disabled inputs during processing
   - Visual feedback for running state

## 4. Building the Backend (API Route with Mastra + Ollama)

Now, let's create the serverless API endpoint that will run our Mastra agent with Ollama.

### 4.1 Creating the API Route

Create a file at `src/app/api/run-agent/route.ts`:

```typescript
import { NextResponse } from 'next/server';
import { MastraClient, AgentSession } from '@mastraai/client';
import { OllamaProvider } from '@mastraai/client/providers';

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

      // Initialize Mastra with Ollama provider
      const ollamaProvider = new OllamaProvider({
        model: 'mistral',
        baseUrl: 'http://localhost:11434', // Default Ollama URL
      });
      
      const mastra = new MastraClient({
        apiKey: process.env.MASTRA_API_KEY,
        provider: ollamaProvider,
      });
      
      // Create and configure the agent session
      const session: AgentSession = await mastra.createSession({
        goal: goal,
        maxSteps: 7, // Limit steps for reasonable response times
        streamUpdates: true, // Enable streaming
      });
      
      // Log initialization
      await sendUpdate({
        type: 'log',
        message: `🧠 Analyzing task: "${goal}"`
      });
      
      // Execute the agent with a callback for progress updates
      session.onStepComplete(async (step) => {
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
      });
      
      const result = await session.execute();
      
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

### 4.2 Understanding the Backend Architecture

Our API route implements several advanced features:

1. **Streaming Response**:
   - Uses the Web Streams API to send real-time updates to the frontend
   - Maintains a single connection instead of polling

2. **Mastra Integration**:
   - Initializes the Mastra client with the Ollama provider
   - Creates an agent session with the user's goal
   - Configures step limits and streaming capabilities

3. **Progress Tracking**:
   - Uses the `onStepComplete` callback to report each step's progress
   - Separates reasoning logs from final results

4. **Error Handling**:
   - Robust error catching and reporting
   - Ensures the stream is properly closed even on errors

## 5. Running and Testing the Application

Now let's run our application and test its capabilities:

### 5.1 Starting the Development Server

Ensure Ollama is running, then start your Next.js application:

```bash
npm run dev
```

Open your browser to `http://localhost:3000`.

### 5.2 Testing with Different Goals

Try entering various goals to test the agent's capabilities:

**Business Planning Examples:**
- "Create a marketing strategy for a new fitness app"
- "Develop a 30-day content calendar for a tech startup"
- "Draft a project plan for website redesign"

**Creative Tasks:**
- "Write a short story about time travel with a twist ending"
- "Create a detailed character profile for a fantasy novel"
- "Develop three unique logo concepts for a sustainable fashion brand"

**Analytical Problems:**
- "Analyze the pros and cons of remote work for a small business"
- "Compare three different pricing strategies for a SaaS product"
- "Create a SWOT analysis for entering the electric vehicle market"

### 5.3 Sample Interaction

Here's an example of how the agent might process a goal to "Create a 7-day social media plan for a coffee shop":

**Agent Logs:**
```
🤖 Initializing Mastra agent powered by Ollama...
🧠 Analyzing task: "Create a 7-day social media plan for a coffee shop"
📝 Step 1: Define the target audience and social media platforms
🤔 Reasoning: Before creating content, I need to understand who we're targeting and which platforms would be most effective for a coffee shop. Typically, Instagram and Facebook work well for food/beverage businesses.
📝 Step 2: Establish content themes and post types
🤔 Reasoning: Coffee shops can benefit from diverse content including product highlights, behind-the-scenes, customer features, and educational content about coffee.
📝 Step 3: Create a content calendar for Monday through Wednesday
🤔 Reasoning: I'll start with the first half of the week, focusing on driving early-week traffic when coffee shops might be slower.
📝 Step 4: Create a content calendar for Thursday through Sunday
🤔 Reasoning: For the latter half of the week, I'll focus on weekend promotions and creating content that encourages longer visits and higher purchases.
📝 Step 5: Add engagement strategies and hashtag recommendations
🤔 Reasoning: Social media success requires engagement beyond just posting. I'll add strategies for responding to comments and effective hashtags.
📝 Step 6: Include measurement metrics and success indicators
🤔 Reasoning: The plan should include ways to track effectiveness so the coffee shop can refine their approach.
📝 Step 7: Finalize the 7-day plan with implementation tips
🤔 Reasoning: I'll compile everything into a cohesive plan and add practical tips for implementation.
✅ Agent execution completed
```

**Agent Result:**
The final output would be a comprehensive, day-by-day social media plan formatted in Markdown, including specific post ideas, optimal posting times, hashtag recommendations, and engagement strategies.

## 6. Expanding the Project

Once you have the basic application working, consider these enhancements to create a more powerful agent system:

### 6.1 Adding Specialized Agent Tools

Extend your Mastra agent with specialized tools for different tasks:

```typescript
const mastra = new MastraClient({
  apiKey: process.env.MASTRA_API_KEY,
  provider: ollamaProvider,
  tools: [
    {
      name: 'webSearch',
      description: 'Search the web for current information',
      async execute(query: string) {
        // Integration with a search API like Serper or SerpAPI
        const response = await fetch(`https://api.search.com?q=${encodeURIComponent(query)}`, {
          headers: { 'Authorization': `Bearer ${process.env.SEARCH_API_KEY}` }
        });
        return response.json();
      }
    },
    {
      name: 'imageGenerator',
      description: 'Generate an image based on a description',
      async execute(prompt: string) {
        // Integration with image generation API
        const response = await fetch('https://api.stability.ai/v1/generation', {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${process.env.STABILITY_API_KEY}` },
          body: JSON.stringify({ prompt })
        });
        return response.json();
      }
    }
  ]
});
```

### 6.2 Implementing a Database for Conversation History

Add persistence to your application with a database:

```typescript
// Using Prisma with PostgreSQL
import { PrismaClient } from '@prisma/client';
const prisma = new PrismaClient();

// In your API route
async function saveSession(userId: string, goal: string, result: string, logs: string[]) {
  await prisma.agentSession.create({
    data: {
      userId,
      goal,
      result,
      logs: JSON.stringify(logs),
      createdAt: new Date()
    }
  });
}

// Then retrieve past sessions
async function getUserSessions(userId: string) {
  return await prisma.agentSession.findMany({
    where: { userId },
    orderBy: { createdAt: 'desc' }
  });
}
```

### 6.3 Implementing Multi-Agent Workflows

Create complex workflows with multiple specialized agents:

```typescript
// Research agent gathers information
const researchAgent = await mastra.createSession({
  goal: "Research current coffee industry trends and customer preferences",
  agentType: "researcher"
});
const researchResult = await researchAgent.execute();

// Strategy agent creates a plan based on research
const strategyAgent = await mastra.createSession({
  goal: `Create a marketing strategy based on these industry insights: ${researchResult.output}`,
  agentType: "strategist",
  context: researchResult.output
});
const strategyResult = await strategyAgent.execute();
```

### 6.4 Adding Authentication and User Management

Implement authentication to enable personalized experiences:

```typescript
// Using NextAuth.js
import NextAuth from 'next-auth';
import GoogleProvider from 'next-auth/providers/google';

export default NextAuth({
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    }),
  ],
  callbacks: {
    async session({ session, token }) {
      // Add user data to the session
      session.userId = token.sub;
      return session;
    },
  },
});
```

### 6.5 Performance Optimization and Model Selection

Allow users to choose different models based on their needs:

```typescript
// In your API route
const modelOptions = {
  'mistral': { temperature: 0.7, maxTokens: 2048 },
  'llama3': { temperature: 0.5, maxTokens: 4096 },
  'wizard': { temperature: 0.8, maxTokens: 2048 }
};

const selectedModel = req.body.model || 'mistral';
const modelConfig = modelOptions[selectedModel];

const ollamaProvider = new OllamaProvider({
  model: selectedModel,
  baseUrl: 'http://localhost:11434',
  temperature: modelConfig.temperature,
  maxTokens: modelConfig.maxTokens
});
```

## 7. Conclusion

In this comprehensive tutorial, we've built a sophisticated Next.js application that leverages Mastra and Ollama to create an AI-powered task automation system. This combination offers several key advantages:

1. **Local Privacy**: By running models through Ollama, you maintain control of your data without sending it to external API services.

2. **Cost Efficiency**: Eliminate per-token or per-request charges by running inference locally.

3. **Enterprise-Ready Architecture**: Mastra provides a structured framework for building reliable agents with built-in workflow management and monitoring.

4. **Flexibility**: The system we've built can be easily extended with additional tools, models, and capabilities.

As AI agent technology continues to evolve, frameworks like Mastra provide the structure and reliability needed to build production-grade applications. Combined with the local inference capabilities of Ollama, you can create powerful, privacy-respecting AI systems that solve real-world problems.

### Additional Resources

- [Mastra Documentation](https://docs.mastra.ai)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Next.js Documentation](https://nextjs.org/docs)
- [LangChain.js (complementary framework)](https://js.langchain.com)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

By mastering these technologies, you're well-positioned to build the next generation of intelligent applications that blend the best of human creativity with AI capabilities.
