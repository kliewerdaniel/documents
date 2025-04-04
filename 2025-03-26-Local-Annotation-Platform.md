---
layout: post
title: Local Annotation Platform
description: The Ultimate Guide to Building a Fully Local Adaptive Persona-Based Data Annotation Platform
date:   2025-03-26 01:42:44 -0500
---
# The Ultimate Guide to Building a Fully Local Adaptive Persona-Based Data Annotation Platform

## Table of Contents

1. [Project Overview and Architecture](#1-project-overview-and-architecture)
2. [Core Technologies and Concepts](#2-core-technologies-and-concepts)
3. [Setting Up the Development Environment](#3-setting-up-the-development-environment)
4. [Developing the Persona-Based Annotation Engine](#4-developing-the-persona-based-annotation-engine)
5. [Implementing Reinforcement Learning with Human Feedback](#5-implementing-reinforcement-learning-with-human-feedback)
6. [Building a Scalable, Fully Local System](#6-building-a-scalable-fully-local-system)
7. [Advanced Topics and Future Enhancements](#7-advanced-topics-and-future-enhancements)
8. [Troubleshooting and Performance Optimization](#8-troubleshooting-and-performance-optimization)

---

## 1. Project Overview and Architecture

### 1.1 Understanding the Purpose and Goals

The Adaptive Persona-Based Data Annotation platform is designed to revolutionize how we annotate data by leveraging AI personas that can adapt to different annotation tasks. Traditional annotation systems often rely on rigid guidelines and produce inconsistent results as human annotators interpret instructions differently. Our system addresses this by:

1. Creating AI personas with specific traits, knowledge bases, and annotation styles
2. Using these personas to generate consistent annotations across diverse datasets
3. Adapting these personas over time through reinforcement learning with human feedback
4. Operating entirely locally to ensure data privacy and eliminate cloud dependencies

This approach significantly improves annotation consistency, reduces bias, accelerates annotation workflows, and makes high-quality data annotation accessible to individual researchers and organizations with strict privacy requirements.

### 1.2 System Architecture Overview

Our architecture follows a modern, modular approach that can run entirely on a local machine:

![System Architecture Diagram](https://i.imgur.com/WTCXzpU.png)
*(Conceptual architecture diagram showing the key components and data flow)*

The core components of our system include:

1. **Next.js Application**: Serves as both frontend UI and backend API, providing a unified codebase for the entire application.
   
2. **Local Database (SQLite/PostgreSQL)**: Stores project configurations, annotation tasks, user feedback, and system metadata.
   
3. **ChromaDB**: A local vector database that indexes and enables semantic search over persona embeddings and annotation examples.
   
4. **Ollama**: Runs local large language models (LLMs) to generate AI personas and power the annotation engine.
   
5. **Annotation Engine**: Orchestrates the interaction between personas and data to be annotated.
   
6. **RLHF Module**: Collects human feedback and refines persona behaviors accordingly.

### 1.3 Data Flow and Component Interaction

Let's walk through the typical data flow:

1. **Initialization**: Admin users configure project requirements and create initial personas.

2. **Data Ingestion**: Users upload datasets to be annotated through the Next.js frontend.

3. **Annotation Process**:
   - The system selects appropriate personas for the annotation task.
   - Ollama generates annotations based on persona characteristics.
   - Results are stored in the database and presented to users.

4. **Feedback Collection**:
   - Users provide feedback on annotation quality.
   - Feedback is stored in the database for reinforcement learning.

5. **Persona Refinement**:
   - The RLHF module analyzes feedback patterns.
   - Persona characteristics are adjusted to improve annotation quality.
   - Updated persona embeddings are stored in ChromaDB.

6. **Continuous Improvement**:
   - The system learns from ongoing feedback, becoming more accurate over time.

This architecture ensures a tightly integrated yet modular system where components can be improved independently while maintaining overall system coherence.

---

## 2. Core Technologies and Concepts

### 2.1 Next.js: The Foundation

Next.js serves as the perfect foundation for our application due to its unified approach to frontend and backend development.

#### 2.1.1 Next.js Application Structure

Our Next.js application will follow the App Router architecture (Next.js 13+):

```
/src
  /app
    /api             # Backend API routes
    /components      # Reusable UI components
    /contexts        # React contexts for state management
    /hooks           # Custom React hooks
    /(routes)        # App routes organized by feature
      /projects      # Project management pages
      /annotations   # Annotation interface
      /personas      # Persona management interface
      /feedback      # Feedback collection interface
  /lib
    /db              # Database utilities
    /ollama          # Ollama integration
    /chromadb        # ChromaDB integration
    /rlhf            # RLHF utilities
  /types             # TypeScript type definitions
  /utils             # Utility functions
```

#### 2.1.2 Key Next.js Features We'll Leverage

1. **Server Components**: For improved performance and SEO
2. **API Routes**: For building our backend services
3. **Route Handlers**: For handling API requests with modern patterns
4. **Data Fetching**: Using React's `use` hook and Suspense
5. **Server Actions**: For form submissions and data mutations

### 2.2 Local Database: SQLite vs PostgreSQL

Both SQLite and PostgreSQL can serve as our local database, but they have different strengths:

#### 2.2.1 SQLite: Lightweight and Zero-Configuration

SQLite is perfect for:
- Single-user deployments
- Simpler projects with moderate data volume
- Portable applications
- Development environments

#### 2.2.2 PostgreSQL: Robust and Scalable

PostgreSQL is ideal for:
- Multi-user systems
- Complex queries and data relationships
- Large datasets
- Projects that might scale beyond local deployment

For our implementation, we'll provide options for both, with PostgreSQL recommended for more advanced use cases.

#### 2.2.3 Database Schema Design

Here's our core schema design:

```sql
-- Users table
CREATE TABLE users (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Projects table
CREATE TABLE projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  created_by TEXT REFERENCES users(id),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Personas table
CREATE TABLE personas (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  traits JSON NOT NULL,
  embedding_id TEXT,  -- Reference to ChromaDB
  project_id TEXT REFERENCES projects(id),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Datasets table
CREATE TABLE datasets (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  project_id TEXT REFERENCES projects(id),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Items table (data to be annotated)
CREATE TABLE items (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  metadata JSON,
  dataset_id TEXT REFERENCES datasets(id),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Annotations table
CREATE TABLE annotations (
  id TEXT PRIMARY KEY,
  item_id TEXT REFERENCES items(id),
  persona_id TEXT REFERENCES personas(id),
  annotation TEXT NOT NULL,
  confidence REAL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feedback table
CREATE TABLE feedback (
  id TEXT PRIMARY KEY,
  annotation_id TEXT REFERENCES annotations(id),
  user_id TEXT REFERENCES users(id),
  rating INTEGER NOT NULL,  -- e.g., 1-5 scale
  comment TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2.3 ChromaDB: Vector Search for Persona Management

ChromaDB is an open-source embedding database that we'll use to store and retrieve our persona embeddings.

#### 2.3.1 Core ChromaDB Concepts

- **Collections**: Groups of related embeddings (we'll use a collection for each project)
- **Documents**: The text content associated with embeddings
- **Embeddings**: Vector representations of our personas and examples
- **Metadata**: Additional information stored alongside embeddings

#### 2.3.2 How We'll Use ChromaDB

1. **Persona Storage**: Store vector representations of personas' traits, behaviors, and examples
2. **Similarity Search**: Find the most relevant personas for specific annotation tasks
3. **Annotation Examples**: Store examples of high-quality annotations for reference
4. **Feedback Analysis**: Analyze patterns in feedback to identify improvement areas

### 2.4 Ollama: Local Large Language Models

Ollama allows us to run powerful language models locally without relying on cloud APIs.

#### 2.4.1 Why Ollama?

- **Privacy**: All data stays on your machine
- **No API costs**: Use models without usage fees
- **Customizability**: Fine-tune models for your specific needs
- **No internet dependence**: Works offline

#### 2.4.2 Models We'll Support

We'll focus on implementing support for these Ollama-compatible models:

1. **Llama 2 (7B)**: Good balance of quality and resource requirements
2. **Mistral 7B**: Excellent performance for its size
3. **Orca Mini**: Efficient for specific tasks
4. **Custom fine-tuned models**: Support for user-provided models

#### 2.4.3 Ollama Integration Architecture

Our Ollama integration will:
1. Manage model loading and unloading
2. Handle prompt engineering for persona creation
3. Generate annotations based on persona characteristics
4. Support batched inference for efficiency

### 2.5 Reinforcement Learning with Human Feedback (RLHF)

RLHF is the key to making our system truly adaptive. Here's how we'll implement it locally:

#### 2.5.1 RLHF Pipeline Components

1. **Feedback Collection**: UI components to gather user ratings and comments
2. **Reward Modeling**: Convert feedback into numeric rewards
3. **Policy Updates**: Adjust persona characteristics based on rewards
4. **Evaluation**: Measure improvement over time

#### 2.5.2 Simplified Local RLHF Approach

For a local implementation, we'll use a simplified RLHF approach:

1. Store examples of annotations with positive feedback
2. Use few-shot learning with these examples to guide future annotations
3. Adjust persona prompts based on feedback patterns
4. Implement lightweight preference models to rank annotation approaches

This provides the benefits of RLHF without the computational overhead of full model fine-tuning.

---

## 3. Setting Up the Development Environment

### 3.1 Prerequisites

Before getting started, ensure you have the following installed:

- **Node.js** (v18.17.0 or later)
- **npm** or **yarn** or **pnpm**
- **Python** (v3.9 or later) for ChromaDB and RLHF components
- **Git** for version control

Optional but recommended:
- **Docker** for containerized database deployment
- **Visual Studio Code** with relevant extensions

### 3.2 Creating the Next.js Project

Let's start by creating a new Next.js project:

```bash
npx create-next-app@latest persona-annotation-platform
cd persona-annotation-platform
```

When prompted, select the following options:
- TypeScript: Yes
- ESLint: Yes
- Tailwind CSS: Yes
- `src/` directory: Yes
- App Router: Yes
- Import alias: Yes (default @/*)

### 3.3 Setting Up the Database

We'll provide setup instructions for both SQLite and PostgreSQL:

#### 3.3.1 SQLite Setup with Prisma

Install Prisma:

```bash
npm install prisma @prisma/client
npx prisma init
```

Edit the `prisma/schema.prisma` file:

```prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "sqlite"
  url      = "file:./dev.db"
}

model User {
  id        String     @id @default(uuid())
  name      String
  createdAt DateTime   @default(now()) @map("created_at")
  feedback  Feedback[]

  @@map("users")
}

model Project {
  id          String    @id @default(uuid())
  name        String
  description String?
  createdBy   String    @map("created_by")
  createdAt   DateTime  @default(now()) @map("created_at")
  updatedAt   DateTime  @default(now()) @updatedAt @map("updated_at")
  personas    Persona[]
  datasets    Dataset[]

  @@map("projects")
}

model Persona {
  id          String       @id @default(uuid())
  name        String
  description String?
  traits      String       @default("{}")
  embeddingId String?      @map("embedding_id")
  projectId   String       @map("project_id")
  project     Project      @relation(fields: [projectId], references: [id])
  createdAt   DateTime     @default(now()) @map("created_at")
  updatedAt   DateTime     @default(now()) @updatedAt @map("updated_at")
  annotations Annotation[]

  @@map("personas")
}

model Dataset {
  id          String  @id @default(uuid())
  name        String
  description String?
  projectId   String  @map("project_id")
  project     Project @relation(fields: [projectId], references: [id])
  createdAt   DateTime @default(now()) @map("created_at")
  items       Item[]

  @@map("datasets")
}

model Item {
  id          String       @id @default(uuid())
  content     String
  metadata    String       @default("{}")
  datasetId   String       @map("dataset_id")
  dataset     Dataset      @relation(fields: [datasetId], references: [id])
  createdAt   DateTime     @default(now()) @map("created_at")
  annotations Annotation[]

  @@map("items")
}

model Annotation {
  id         String     @id @default(uuid())
  itemId     String     @map("item_id")
  item       Item       @relation(fields: [itemId], references: [id])
  personaId  String     @map("persona_id")
  persona    Persona    @relation(fields: [personaId], references: [id])
  annotation String
  confidence Float?
  createdAt  DateTime   @default(now()) @map("created_at")
  feedback   Feedback[]

  @@map("annotations")
}

model Feedback {
  id           String     @id @default(uuid())
  annotationId String     @map("annotation_id")
  annotation   Annotation @relation(fields: [annotationId], references: [id])
  userId       String     @map("user_id")
  user         User       @relation(fields: [userId], references: [id])
  rating       Int
  comment      String?
  createdAt    DateTime   @default(now()) @map("created_at")

  @@map("feedback")
}
```

Generate Prisma client and apply migrations:

```bash
npx prisma migrate dev --name init
```

#### 3.3.2 PostgreSQL Setup (Alternative)

If using PostgreSQL, modify the datasource in schema.prisma:

```prisma
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}
```

Create a `.env` file:

```
DATABASE_URL="postgresql://username:password@localhost:5432/annotation_platform?schema=public"
```

For a fully local PostgreSQL setup, you can use Docker:

```bash
docker run --name postgres-annotation -e POSTGRES_PASSWORD=mysecretpassword -e POSTGRES_USER=annotation_user -e POSTGRES_DB=annotation_platform -p 5432:5432 -d postgres
```

Then update your .env file:

```
DATABASE_URL="postgresql://annotation_user:mysecretpassword@localhost:5432/annotation_platform?schema=public"
```

### 3.4 Setting Up ChromaDB

Install the required Python packages:

```bash
pip install chromadb sentence-transformers
```

Create a `src/lib/chromadb/index.ts` file to interface with ChromaDB:

```typescript
// src/lib/chromadb/index.ts
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

// Define the ChromaDB client directory
const CHROMA_DIR = path.join(process.cwd(), 'chroma_db');

// Ensure the ChromaDB directory exists
if (!fs.existsSync(CHROMA_DIR)) {
  fs.mkdirSync(CHROMA_DIR, { recursive: true });
}

// Define interface for ChromaDB operations
export interface ChromaDBService {
  addPersona(personaId: string, text: string, metadata: Record<string, any>): Promise<void>;
  searchSimilarPersonas(query: string, limit?: number): Promise<Array<{ id: string; score: number; metadata: Record<string, any> }>>;
  deletePersona(personaId: string): Promise<void>;
}

// Python script runner for ChromaDB operations
class PythonChromaDBService implements ChromaDBService {
  private runPythonScript(scriptName: string, args: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
      const scriptPath = path.join(process.cwd(), 'scripts', 'chromadb', `${scriptName}.py`);
      
      const process = spawn('python', [scriptPath, ...args]);
      
      let output = '';
      process.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      let errorOutput = '';
      process.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });
      
      process.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Python script error: ${errorOutput}`));
        } else {
          resolve(output.trim());
        }
      });
    });
  }

  async addPersona(personaId: string, text: string, metadata: Record<string, any>): Promise<void> {
    await this.runPythonScript('add_persona', [
      personaId,
      text,
      JSON.stringify(metadata),
      CHROMA_DIR
    ]);
  }

  async searchSimilarPersonas(query: string, limit = 5): Promise<Array<{ id: string; score: number; metadata: Record<string, any> }>> {
    const result = await this.runPythonScript('search_personas', [
      query,
      limit.toString(),
      CHROMA_DIR
    ]);
    
    return JSON.parse(result);
  }

  async deletePersona(personaId: string): Promise<void> {
    await this.runPythonScript('delete_persona', [
      personaId,
      CHROMA_DIR
    ]);
  }
}

// Export an instance of the service
export const chromaDBService = new PythonChromaDBService();
```

Now create the Python scripts for ChromaDB operations:

```python
# scripts/chromadb/add_persona.py
import sys
import json
import chromadb
from chromadb.utils import embedding_functions

def main():
    if len(sys.argv) != 5:
        print("Usage: python add_persona.py <persona_id> <text> <metadata_json> <chroma_dir>")
        sys.exit(1)
    
    persona_id = sys.argv[1]
    text = sys.argv[2]
    metadata = json.loads(sys.argv[3])
    chroma_dir = sys.argv[4]
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Use sentence-transformers for embeddings
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name="personas",
        embedding_function=sentence_transformer_ef
    )
    
    # Add or update persona
    collection.upsert(
        ids=[persona_id],
        documents=[text],
        metadatas=[metadata]
    )
    
    print(f"Successfully added persona: {persona_id}")

if __name__ == "__main__":
    main()
```

```python
# scripts/chromadb/search_personas.py
import sys
import json
import chromadb
from chromadb.utils import embedding_functions

def main():
    if len(sys.argv) != 4:
        print("Usage: python search_personas.py <query> <limit> <chroma_dir>")
        sys.exit(1)
    
    query = sys.argv[1]
    limit = int(sys.argv[2])
    chroma_dir = sys.argv[3]
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Use sentence-transformers for embeddings
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Get collection
    try:
        collection = client.get_collection(
            name="personas",
            embedding_function=sentence_transformer_ef
        )
        
        # Search for similar personas
        results = collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'score': results['distances'][0][i] if 'distances' in results else 0,
                'metadata': results['metadatas'][0][i]
            })
        
        print(json.dumps(formatted_results))
        
    except Exception as e:
        print(json.dumps([]))
        sys.exit(0)

if __name__ == "__main__":
    main()
```

```python
# scripts/chromadb/delete_persona.py
import sys
import chromadb
from chromadb.utils import embedding_functions

def main():
    if len(sys.argv) != 3:
        print("Usage: python delete_persona.py <persona_id> <chroma_dir>")
        sys.exit(1)
    
    persona_id = sys.argv[1]
    chroma_dir = sys.argv[2]
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Use sentence-transformers for embeddings
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Get collection
    try:
        collection = client.get_collection(
            name="personas",
            embedding_function=sentence_transformer_ef
        )
        
        # Delete persona
        collection.delete(
            ids=[persona_id]
        )
        
        print(f"Successfully deleted persona: {persona_id}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 3.5 Setting Up Ollama Integration

Create a library to interface with Ollama:

```typescript
// src/lib/ollama/index.ts
export interface OllamaConfig {
  baseUrl: string;
  model: string;
}

export interface GenerationOptions {
  prompt: string;
  system?: string;
  temperature?: number;
  maxTokens?: number;
}

export interface GenerationResponse {
  text: string;
  model: string;
  promptTokens: number;
  generatedTokens: number;
}

export class OllamaService {
  private baseUrl: string;
  private model: string;

  constructor(config: OllamaConfig) {
    this.baseUrl = config.baseUrl;
    this.model = config.model;
  }

  async generate(options: GenerationOptions): Promise<GenerationResponse> {
    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.model,
        prompt: options.prompt,
        system: options.system,
        options: {
          temperature: options.temperature ?? 0.7,
          num_predict: options.maxTokens,
        },
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ollama API error: ${response.status} ${errorText}`);
    }

    const data = await response.json();
    return {
      text: data.response,
      model: data.model,
      promptTokens: data.prompt_eval_count,
      generatedTokens: data.eval_count,
    };
  }

  async getModels(): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/api/tags`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.status}`);
    }
    
    const data = await response.json();
    return data.models.map((model: any) => model.name);
  }

  setModel(model: string): void {
    this.model = model;
  }
}

// Default instance with localhost
export const ollamaService = new OllamaService({
  baseUrl: 'http://localhost:11434',
  model: 'llama2',
});
```

### 3.6 Environment Configuration

Create a `.env.local` file to store configuration:

```
# Database Configuration
DATABASE_URL="file:./dev.db"
# or for PostgreSQL:
# DATABASE_URL="postgresql://annotation_user:mysecretpassword@localhost:5432/annotation_platform?schema=public"

# Ollama Configuration
NEXT_PUBLIC_OLLAMA_BASE_URL="http://localhost:11434"
NEXT_PUBLIC_OLLAMA_DEFAULT_MODEL="llama2"

# ChromaDB Configuration
CHROMA_DB_DIR="./chroma_db"
```

Create a configuration utility to load these values:

```typescript
// src/lib/config.ts
export const config = {
  database: {
    url: process.env.DATABASE_URL || 'file:./dev.db',
  },
  ollama: {
    baseUrl: process.env.NEXT_PUBLIC_OLLAMA_BASE_URL || 'http://localhost:11434',
    defaultModel: process.env.NEXT_PUBLIC_OLLAMA_DEFAULT_MODEL || 'llama2',
  },
  chromaDb: {
    dir: process.env.CHROMA_DB_DIR || './chroma_db',
  },
};
```

---

## 4. Developing the Persona-Based Annotation Engine

### 4.1 Persona Creation and Management

First, let's create the core persona management interfaces and logic:

#### 4.1.1 Persona Types

```typescript
// src/types/persona.ts
export interface PersonaTrait {
  name: string;
  value: number; // 0-1 scale, representing trait intensity
  description?: string;
}

export interface PersonaExample {
  input: string;
  output: string;
  explanation?: string;
}

export interface PersonaData {
  id: string;
  name: string;
  description: string;
  traits: PersonaTrait[];
  examples: PersonaExample[];
  prompt?: string; // Generated system prompt
}
```

#### 4.1.2 Persona Service

Create a service to manage personas:

```typescript
// src/lib/services/personaService.ts
import { prisma } from '../db/prisma';
import { chromaDBService } from '../chromadb';
import { ollamaService } from '../ollama';
import { PersonaData, PersonaTrait, PersonaExample } from '@/types/persona';

export class PersonaService {
  async createPersona(projectId: string, personaData: Omit<PersonaData, 'id' | 'prompt'>): Promise<PersonaData> {
    // Generate system prompt from persona traits and examples
    const systemPrompt = await this.generateSystemPrompt(personaData.traits, personaData.examples);
    
    // Create persona in database
    const persona = await prisma.persona.create({
      data: {
        name: personaData.name,
        description: personaData.description,
        traits: JSON.stringify(personaData.traits),
        projectId,
      },
    });
    
    // Create the text representation for embedding
    const textRepresentation = this.createPersonaTextRepresentation(personaData);
    
    // Store in ChromaDB
    await chromaDBService.addPersona(
      persona.id,
      textRepresentation,
      {
        name: personaData.name,
        description: personaData.description,
        systemPrompt,
      }
    );
    
    return {
      id: persona.id,
      name: personaData.name,
      description: personaData.description,
      traits: personaData.traits,
      examples: personaData.examples,
      prompt: systemPrompt,
    };
  }
  
  async getPersona(personaId: string): Promise<PersonaData | null> {
    const persona = await prisma.persona.findUnique({
      where: { id: personaId },
    });
    
    if (!persona) return null;
    
    // Parse the traits from JSON string
    const traits = JSON.parse(persona.traits) as PersonaTrait[];
    
    // Retrieve examples from ChromaDB metadata
    const personaResults = await chromaDBService.searchSimilarPersonas(persona.id, 1);
    
    if (personaResults.length === 0) {
      throw new Error(`Persona ${personaId} not found in ChromaDB`);
    }
    
    const metadata = personaResults[0].metadata;
    
    return {
      id: persona.id,
      name: persona.name,
      description: persona.description || '',
      traits,
      examples: metadata.examples || [],
      prompt: metadata.systemPrompt,
    };
  }
  
  async updatePersona(personaId: string, updateData: Partial<PersonaData>): Promise<PersonaData> {
    // Get current persona
    const currentPersona = await this.getPersona(personaId);
    
    if (!currentPersona) {
      throw new Error(`Persona ${personaId} not found`);
    }
    
    // Merge current with updates
    const updatedPersona = {
      ...currentPersona,
      ...updateData,
    };
    
    // Generate new system prompt if traits or examples changed
    if (updateData.traits || updateData.examples) {
      updatedPersona.prompt = await this.generateSystemPrompt(
        updatedPersona.traits,
        updatedPersona.examples
      );
    }
    
    // Update in database
    await prisma.persona.update({
      where: { id: personaId },
      data: {
        name: updatedPersona.name,
        description: updatedPersona.description,
        traits: JSON.stringify(updatedPersona.traits),
        updatedAt: new Date(),
      },
    });
    
    // Update in ChromaDB
    const textRepresentation = this.createPersonaTextRepresentation(updatedPersona);
    
    await chromaDBService.addPersona(
      personaId,
      textRepresentation,
      {
        name: updatedPersona.name,
        description: updatedPersona.description,
        systemPrompt: updatedPersona.prompt,
        examples: updatedPersona.examples,
      }
    );
    
    return updatedPersona;
  }
  
  async deletePersona(personaId: string): Promise<void> {
    // Delete from database
    await prisma.persona.delete({
      where: { id: personaId },
    });
    
    // Delete from ChromaDB
    await chromaDBService.deletePersona(personaId);
  }
  
  private async generateSystemPrompt(traits: PersonaTrait[], examples: PersonaExample[]): Promise<string> {
    // Create a prompt for the LLM to generate a system prompt
    const traitsText = traits
      .map(trait => `- ${trait.name}: ${trait.value.toFixed(2)} (${trait.description || ''})`)
      .join('\n');
    
    const examplesText = examples
      .map(example => `Input: ${example.input}\nExpected Output: ${example.output}${example.explanation ? `\nExplanation: ${example.explanation}` : ''}`)
      .join('\n\n');
    
    const promptForLLM = `
You are an expert at creating detailed persona descriptions for AI assistants.
I need you to create a system prompt that will guide an AI to act according to the following personality traits:

${traitsText}

Here are examples of how this persona should respond:

${examplesText}

Create a detailed system prompt that captures this persona's essence and will guide an AI to produce similar outputs to the examples.
The system prompt should be clear, specific, and instructive.
`;

    // Generate the system prompt using Ollama
    const response = await ollamaService.generate({
      prompt: promptForLLM,
      temperature: 0.7,
    });
    
    return response.text.trim();
  }
  
  private createPersonaTextRepresentation(persona: Omit<PersonaData, 'id' | 'prompt'>): string {
    // Create a text representation for vector embedding
    const traitsText = persona.traits
      .map(trait => `${trait.name}: ${trait.value.toFixed(2)}`)
      .join(', ');
    
    return `Persona Name: ${persona.name}
Description: ${persona.description}
Traits: ${traitsText}
Examples: ${persona.examples.length} example(s) provided`;
  }
}

export const personaService = new PersonaService();
```

#### 4.1.3 Persona Management API Routes

Create API routes for persona management:

```typescript
// src/app/api/projects/[projectId]/personas/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { personaService } from '@/lib/services/personaService';

export async function GET(
  request: NextRequest,
  { params }: { params: { projectId: string } }
) {
  try {
    const projectId = params.projectId;
    
    // Get all personas for a project
    const personas = await prisma.persona.findMany({
      where: { projectId },
    });
    
    return NextResponse.json(personas);
  } catch (error) {
    console.error('Error fetching personas:', error);
    return NextResponse.json({ error: 'Failed to fetch personas' }, { status: 500 });
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: { projectId: string } }
) {
  try {
    const projectId = params.projectId;
    const data = await request.json();
    
    // Create new persona
    const persona = await personaService.createPersona(projectId, data);
    
    return NextResponse.json(persona, { status: 201 });
  } catch (error) {
    console.error('Error creating persona:', error);
    return NextResponse.json({ error: 'Failed to create persona' }, { status: 500 });
  }
}
```

```typescript
// src/app/api/personas/[personaId]/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { personaService } from '@/lib/services/personaService';

export async function GET(
  request: NextRequest,
  { params }: { params: { personaId: string } }
) {
  try {
    const personaId = params.personaId;
    
    // Get persona details
    const persona = await personaService.getPersona(personaId);
    
    if (!persona) {
      return NextResponse.json({ error: 'Persona not found' }, { status: 404 });
    }
    
    return NextResponse.json(persona);
  } catch (error) {
    console.error('Error fetching persona:', error);
    return NextResponse.json({ error: 'Failed to fetch persona' }, { status: 500 });
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { personaId: string } }
) {
  try {
    const personaId = params.personaId;
    const data = await request.json();
    
    // Update persona
    const persona = await personaService.updatePersona(personaId, data);
    
    return NextResponse.json(persona);
  } catch (error) {
    console.error('Error updating persona:', error);
    return NextResponse.json({ error: 'Failed to update persona' }, { status: 500 });
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { personaId: string } }
) {
  try {
    const personaId = params.personaId;
    
    // Delete persona
    await personaService.deletePersona(personaId);
    
    return new NextResponse(null, { status: 204 });
  } catch (error) {
    console.error('Error deleting persona:', error);
    return NextResponse.json({ error: 'Failed to delete persona' }, { status: 500 });
  }
}
```

### 4.2 Building the Annotation Engine

Now, let's create the core annotation engine that uses personas to generate annotations:

#### 4.2.1 Annotation Types

```typescript
// src/types/annotation.ts
export interface AnnotationRequest {
  itemId: string;
  personaId: string;
  content: string;
  metadata?: Record<string, any>;
}

export interface AnnotationResult {
  id: string;
  itemId: string;
  personaId: string;
  annotation: string;
  confidence?: number;
  createdAt: Date;
}
```

#### 4.2.2 Annotation Service

Create the service for generating annotations:

```typescript
// src/lib/services/annotationService.ts
import { prisma } from '../db/prisma';
import { personaService } from './personaService';
import { ollamaService } from '../ollama';
import { AnnotationRequest, AnnotationResult } from '@/types/annotation';

export class AnnotationService {
  async generateAnnotation(request: AnnotationRequest): Promise<AnnotationResult> {
    // Get the persona
    const persona = await personaService.getPersona(request.personaId);
    
    if (!persona) {
      throw new Error(`Persona ${request.personaId} not found`);
    }
    
    // Get item from database or create a temporary one if not provided
    let item;
    if (request.itemId) {
      item = await prisma.item.findUnique({
        where: { id: request.itemId },
      });
      
      if (!item) {
        throw new Error(`Item ${request.itemId} not found`);
      }
    }
    
    // Prepare the prompt for annotation
    const prompt = `Please analyze the following content and provide an annotation:

${request.content}`;

    // Generate annotation using Ollama
    const ollamaResponse = await ollamaService.generate({
      prompt,
      system: persona.prompt,
      temperature: 0.3, // Lower temperature for more focused annotations
    });
    
    // Calculate a simple confidence score
    const confidence = this.calculateConfidence(ollamaResponse.text);
    
    // Save annotation to database if we have an item
    let annotation;
    if (request.itemId) {
      annotation = await prisma.annotation.create({
        data: {
          itemId: request.itemId,
          personaId: request.personaId,
          annotation: ollamaResponse.text,
          confidence,
        },
      });
    } else {
      // Create an ephemeral annotation result
      annotation = {
        id: 'temp-' + Date.now(),
        itemId: 'temp-item',
        personaId: request.personaId,
        annotation: ollamaResponse.text,
        confidence,
        createdAt: new Date(),
      };
    }
    
    return annotation;
  }
  
  async getAnnotations(itemId: string): Promise<AnnotationResult[]> {
    const annotations = await prisma.annotation.findMany({
      where: { itemId },
      orderBy: { createdAt: 'desc' },
    });
    
    return annotations;
  }
  
  private calculateConfidence(text: string): number {
    // A simple heuristic for confidence based on response length and structure
    // In a real system, this would be more sophisticated
    const length = text.length;
    
    if (length < 10) return 0.1;
    if (length < 50) return 0.3;
    if (length < 200) return 0.5;
    if (length < 500) return 0.7;
    return 0.9;
  }
}

export const annotationService = new AnnotationService();
```

#### 4.2.3 Annotation API Routes

```typescript
// src/app/api/annotations/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { annotationService } from '@/lib/services/annotationService';

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();
    
    // Validate request
    if (!data.personaId || !data.content) {
      return NextResponse.json(
        { error: 'personaId and content are required' },
        { status: 400 }
      );
    }
    
    // Generate annotation
    const annotation = await annotationService.generateAnnotation(data);
    
    return NextResponse.json(annotation, { status: 201 });
  } catch (error) {
    console.error('Error generating annotation:', error);
    return NextResponse.json(
      { error: 'Failed to generate annotation' },
      { status: 500 }
    );
  }
}
```

```typescript
// src/app/api/items/[itemId]/annotations/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { annotationService } from '@/lib/services/annotationService';

export async function GET(
  request: NextRequest,
  { params }: { params: { itemId: string } }
) {
  try {
    const itemId = params.itemId;
    
    // Get annotations for item
    const annotations = await annotationService.getAnnotations(itemId);
    
    return NextResponse.json(annotations);
  } catch (error) {
    console.error('Error fetching annotations:', error);
    return NextResponse.json(
      { error: 'Failed to fetch annotations' },
      { status: 500 }
    );
  }
}
```

### 4.3 Building the Persona Selection Logic

An important aspect of the annotation system is selecting the right persona for a given annotation task. Let's create a service for this:

```typescript
// src/lib/services/personaSelectionService.ts
import { chromaDBService } from '../chromadb';

export class PersonaSelectionService {
  async findBestPersonas(
    content: string,
    taskDescription: string,
    projectId: string,
    limit = 3
  ): Promise<Array<{ id: string; score: number; name: string }>> {
    // Combine content and task description for better matching
    const query = `${taskDescription}\n\n${content}`;
    
    // Search for similar personas in ChromaDB
    const results = await chromaDBService.searchSimilarPersonas(query, limit);
    
    // Format results
    return results.map(result => ({
      id: result.id,
      score: 1 - result.score, // Convert distance to similarity score
      name: result.metadata.name,
    }));
  }
}

export const personaSelectionService = new PersonaSelectionService();
```

Create an API route for persona selection:

```typescript
// src/app/api/projects/[projectId]/select-personas/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { personaSelectionService } from '@/lib/services/personaSelectionService';

export async function POST(
  request: NextRequest,
  { params }: { params: { projectId: string } }
) {
  try {
    const projectId = params.projectId;
    const data = await request.json();
    
    // Validate request
    if (!data.content) {
      return NextResponse.json(
        { error: 'Content is required' },
        { status: 400 }
      );
    }
    
    // Find best matching personas
    const personas = await personaSelectionService.findBestPersonas(
      data.content,
      data.taskDescription || '',
      projectId,
      data.limit || 3
    );
    
    return NextResponse.json(personas);
  } catch (error) {
    console.error('Error selecting personas:', error);
    return NextResponse.json(
      { error: 'Failed to select personas' },
      { status: 500 }
    );
  }
}
```

### 4.4 Frontend UI for Annotation

Let's build the UI components for the annotation system. First, let's create a basic layout:

```tsx
// src/app/layout.tsx
import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import Sidebar from '@/components/Sidebar';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Persona-Based Annotation Platform',
  description: 'A local platform for AI-powered data annotation using personas',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="flex h-screen">
          <Sidebar />
          <main className="flex-1 overflow-y-auto p-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
```

Next, create a sidebar component:

```tsx
// src/components/Sidebar.tsx
'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { HomeIcon, FolderIcon, UsersIcon, PencilIcon, AdjustmentsHorizontalIcon } from '@heroicons/react/24/outline';

export default function Sidebar() {
  const pathname = usePathname();
  
  const navigation = [
    { name: 'Dashboard', href: '/', icon: HomeIcon },
    { name: 'Projects', href: '/projects', icon: FolderIcon },
    { name: 'Personas', href: '/personas', icon: UsersIcon },
    { name: 'Annotation', href: '/annotation', icon: PencilIcon },
    { name: 'Settings', href: '/settings', icon: AdjustmentsHorizontalIcon },
  ];
  
  return (
    <div className="w-64 bg-gray-800 text-white">
      <div className="p-4">
        <h1 className="text-xl font-bold">Annotation Platform</h1>
      </div>
      <nav className="mt-8">
        <ul>
          {navigation.map((item) => {
            const isActive = pathname === item.href;
            return (
              <li key={item.name} className="mb-2">
                <Link
                  href={item.href}
                  className={`flex items-center px-4 py-2 ${
                    isActive ? 'bg-gray-700' : 'hover:bg-gray-700'
                  }`}
                >
                  <item.icon className="h-5 w-5 mr-3" />
                  {item.name}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </div>
  );
}
```

Now, create the annotation page:

```tsx
// src/app/annotation/page.tsx
'use client';

import { useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { PersonaData } from '@/types/persona';

export default function AnnotationPage() {
  const searchParams = useSearchParams();
  const itemId = searchParams.get('itemId');
  
  const [content, setContent] = useState('');
  const [selectedPersonaId, setSelectedPersonaId] = useState('');
  const [personas, setPersonas] = useState<Array<{ id: string; name: string; score?: number }>>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [annotation, setAnnotation] = useState('');
  
  // Fetch available personas
  useEffect(() => {
    const fetchPersonas = async () => {
      try {
        // In a real app, you'd get the current project ID from context/state
        const projectId = 'default-project';
        const response = await fetch(`/api/projects/${projectId}/personas`);
        
        if (response.ok) {
          const data = await response.json();
          setPersonas(data.map((p: any) => ({ id: p.id, name: p.name })));
        }
      } catch (error) {
        console.error('Error fetching personas:', error);
      }
    };
    
    fetchPersonas();
  }, []);
  
  // If item ID is provided, fetch the item content
  useEffect(() => {
    if (itemId) {
      const fetchItem = async () => {
        try {
          const response = await fetch(`/api/items/${itemId}`);
          
          if (response.ok) {
            const data = await response.json();
            setContent(data.content);
          }
        } catch (error) {
          console.error('Error fetching item:', error);
        }
      };
      
      fetchItem();
    }
  }, [itemId]);
  
  // Function to select best personas
  const findBestPersonas = async () => {
    if (!content) return;
    
    setIsLoading(true);
    
    try {
      // In a real app, you'd get the current project ID from context/state
      const projectId = 'default-project';
      const response = await fetch(`/api/projects/${projectId}/select-personas`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content,
          taskDescription: 'Annotate the following content',
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setPersonas(data);
        
        // Auto-select the top persona
        if (data.length > 0) {
          setSelectedPersonaId(data[0].id);
        }
      }
    } catch (error) {
      console.error('Error finding best personas:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Function to generate annotation
  const generateAnnotation = async () => {
    if (!selectedPersonaId || !content) return;
    
    setIsLoading(true);
    setAnnotation('');
    
    try {
      const response = await fetch('/api/annotations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          personaId: selectedPersonaId,
          content,
          itemId, // Include if we have an item ID
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setAnnotation(data.annotation);
      }
    } catch (error) {
      console.error('Error generating annotation:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Annotation Interface</h1>
      
      <div className="mb-6">
        <label className="block mb-2 font-medium">Content to Annotate</label>
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          className="w-full h-40 p-3 border rounded"
          placeholder="Enter or paste content to annotate..."
        />
      </div>
      
      <div className="flex gap-4 mb-6">
        <button
          onClick={findBestPersonas}
          disabled={!content || isLoading}
          className="px-4 py-2 bg-blue-600 text-white rounded disabled:bg-blue-300"
        >
          Find Best Personas
        </button>
      </div>
      
      {personas.length > 0 && (
        <div className="mb-6">
          <label className="block mb-2 font-medium">Select Persona</label>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {personas.map((persona) => (
              <div
                key={persona.id}
                onClick={() => setSelectedPersonaId(persona.id)}
                className={`p-3 border rounded cursor-pointer ${
                  selectedPersonaId === persona.id ? 'border-blue-500 bg-blue-50' : ''
                }`}
              >
                <div className="font-medium">{persona.name}</div>
                {persona.score !== undefined && (
                  <div className="text-sm text-gray-500">
                    Match score: {(persona.score * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      
      {selectedPersonaId && (
        <div className="mb-6">
          <button
            onClick={generateAnnotation}
            disabled={isLoading}
            className="px-4 py-2 bg-green-600 text-white rounded disabled:bg-green-300"
          >
            {isLoading ? 'Generating...' : 'Generate Annotation'}
          </button>
        </div>
      )}
      
      {annotation && (
        <div className="mt-8">
          <h2 className="text-xl font-bold mb-3">Annotation Result</h2>
          <div className="p-4 bg-gray-50 border rounded">
            <pre className="whitespace-pre-wrap">{annotation}</pre>
          </div>
          
          {/* Feedback buttons would go here */}
          <div className="mt-4 flex gap-2">
            <button className="px-3 py-1 bg-green-100 text-green-800 rounded border border-green-300">
              Approve
            </button>
            <button className="px-3 py-1 bg-red-100 text-red-800 rounded border border-red-300">
              Reject
            </button>
            <button className="px-3 py-1 bg-gray-100 text-gray-800 rounded border border-gray-300">
              Edit
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
```

Let's also create a persona management page:

```tsx
// src/app/personas/page.tsx
'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { PlusIcon } from '@heroicons/react/24/outline';
import { PersonaData } from '@/types/persona';

export default function PersonasPage() {
  const [personas, setPersonas] = useState<PersonaData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    const fetchPersonas = async () => {
      try {
        // In a real app, you'd get the current project ID from context/state
        const projectId = 'default-project';
        const response = await fetch(`/api/projects/${projectId}/personas`);
        
        if (response.ok) {
          const data = await response.json();
          setPersonas(data);
        }
      } catch (error) {
        console.error('Error fetching personas:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchPersonas();
  }, []);
  
  const deletePersona = async (personaId: string) => {
    if (!confirm('Are you sure you want to delete this persona?')) return;
    
    try {
      const response = await fetch(`/api/personas/${personaId}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        setPersonas(personas.filter(p => p.id !== personaId));
      }
    } catch (error) {
      console.error('Error deleting persona:', error);
    }
  };
  
  if (isLoading) {
    return <div className="text-center py-8">Loading personas...</div>;
  }
  
  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Personas</h1>
        <Link
          href="/personas/new"
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded"
        >
          <PlusIcon className="h-5 w-5 mr-2" />
          Create Persona
        </Link>
      </div>
      
      {personas.length === 0 ? (
        <div className="text-center py-8 bg-gray-50 rounded border">
          <p className="text-gray-500">No personas created yet.</p>
          <Link
            href="/personas/new"
            className="inline-block mt-3 px-4 py-2 bg-blue-600 text-white rounded"
          >
            Create your first persona
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {personas.map((persona) => (
            <div key={persona.id} className="border rounded overflow-hidden">
              <div className="p-4">
                <h2 className="text-lg font-bold mb-2">{persona.name}</h2>
                <p className="text-gray-600 text-sm mb-3">{persona.description}</p>
                
                <div className="mb-4">
                  <h3 className="text-sm font-medium text-gray-500 mb-1">Traits:</h3>
                  <div className="flex flex-wrap gap-2">
                    {persona.traits.map((trait, index) => (
                      <span
                        key={index}
                        className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded"
                      >
                        {trait.name}: {trait.value.toFixed(1)}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="border-t px-4 py-3 bg-gray-50 flex justify-between">
                <Link
                  href={`/personas/${persona.id}`}
                  className="text-blue-600 hover:underline text-sm"
                >
                  View & Edit
                </Link>
                <button
                  onClick={() => deletePersona(persona.id)}
                  className="text-red-600 hover:underline text-sm"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

And finally, create a form to create new personas:

```tsx
// src/app/personas/new/page.tsx
'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { PersonaTrait, PersonaExample } from '@/types/persona';

export default function NewPersonaPage() {
  const router = useRouter();
  
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [traits, setTraits] = useState<PersonaTrait[]>([
    { name: 'Clarity', value: 0.5, description: 'How clear and concise the annotations should be' },
    { name: 'Detail', value: 0.5, description: 'Level of detail in annotations' },
    { name: 'Formality', value: 0.5, description: 'How formal the language should be' },
  ]);
  const [examples, setExamples] = useState<PersonaExample[]>([
    { input: '', output: '', explanation: '' },
  ]);
  
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  
  const handleTraitChange = (index: number, field: keyof PersonaTrait, value: string | number) => {
    const updatedTraits = [...traits];
    updatedTraits[index] = { ...updatedTraits[index], [field]: value };
    setTraits(updatedTraits);
  };
  
  const addTrait = () => {
    setTraits([...traits, { name: '', value: 0.5, description: '' }]);
  };
  
  const removeTrait = (index: number) => {
    setTraits(traits.filter((_, i) => i !== index));
  };
  
  const handleExampleChange = (index: number, field: keyof PersonaExample, value: string) => {
    const updatedExamples = [...examples];
    updatedExamples[index] = { ...updatedExamples[index], [field]: value };
    setExamples(updatedExamples);
  };
  
  const addExample = () => {
    setExamples([...examples, { input: '', output: '', explanation: '' }]);
  };
  
  const removeExample = (index: number) => {
    setExamples(examples.filter((_, i) => i !== index));
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!name) {
      setError('Name is required');
      return;
    }
    
    setIsSubmitting(true);
    setError('');
    
    try {
      // In a real app, you'd get the current project ID from context/state
      const projectId = 'default-project';
      const response = await fetch(`/api/projects/${projectId}/personas`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name,
          description,
          traits,
          examples,
        }),
      });
      
      if (response.ok) {
        router.push('/personas');
      } else {
        const data = await response.json();
        setError(data.error || 'Failed to create persona');
      }
    } catch (error) {
      console.error('Error creating persona:', error);
      setError('An unexpected error occurred');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Create New Persona</h1>
      
      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-800 rounded">
          {error}
        </div>
      )}
      
      <form onSubmit={handleSubmit}>
        <div className="mb-6">
          <label className="block mb-2 font-medium">Name *</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full p-2 border rounded"
            placeholder="e.g., Technical Reviewer"
            required
          />
        </div>
        
        <div className="mb-6">
          <label className="block mb-2 font-medium">Description</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="w-full p-2 border rounded h-24"
            placeholder="Describe this persona's purpose and characteristics..."
          />
        </div>
        
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <label className="font-medium">Persona Traits</label>
            <button
              type="button"
              onClick={addTrait}
              className="text-sm text-blue-600"
            >
              + Add Trait
            </button>
          </div>
          
          {traits.map((trait, index) => (
            <div key={index} className="mb-3 p-3 border rounded">
              <div className="flex gap-4 mb-3">
                <div className="flex-1">
                  <label className="text-sm text-gray-600 mb-1 block">Trait Name</label>
                  <input
                    type="text"
                    value={trait.name}
                    onChange={(e) => handleTraitChange(index, 'name', e.target.value)}
                    className="w-full p-2 border rounded"
                    placeholder="e.g., Clarity"
                  />
                </div>
                
                <div className="w-24">
                  <label className="text-sm text-gray-600 mb-1 block">Value (0-1)</label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={trait.value}
                    onChange={(e) => handleTraitChange(index, 'value', parseFloat(e.target.value))}
                    className="w-full p-2 border rounded"
                  />
                </div>
              </div>
              
              <div className="mb-2">
                <label className="text-sm text-gray-600 mb-1 block">Description</label>
                <input
                  type="text"
                  value={trait.description || ''}
                  onChange={(e) => handleTraitChange(index, 'description', e.target.value)}
                  className="w-full p-2 border rounded"
                  placeholder="What this trait means..."
                />
              </div>
              
              {traits.length > 1 && (
                <button
                  type="button"
                  onClick={() => removeTrait(index)}
                  className="text-sm text-red-600"
                >
                  Remove
                </button>
              )}
            </div>
          ))}
        </div>
        
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <label className="font-medium">Example Annotations</label>
            <button
              type="button"
              onClick={addExample}
              className="text-sm text-blue-600"
            >
              + Add Example
            </button>
          </div>
          
          {examples.map((example, index) => (
            <div key={index} className="mb-3 p-3 border rounded">
              <div className="mb-3">
                <label className="text-sm text-gray-600 mb-1 block">Input Content</label>
                <textarea
                  value={example.input}
                  onChange={(e) => handleExampleChange(index, 'input', e.target.value)}
                  className="w-full p-2 border rounded h-24"
                  placeholder="Example content to be annotated..."
                />
              </div>
              
              <div className="mb-3">
                <label className="text-sm text-gray-600 mb-1 block">Expected Output</label>
                <textarea
                  value={example.output}
                  onChange={(e) => handleExampleChange(index, 'output', e.target.value)}
                  className="w-full p-2 border rounded h-24"
                  placeholder="How this persona should annotate the input..."
                />
              </div>
              
              <div className="mb-2">
                <label className="text-sm text-gray-600 mb-1 block">Explanation (Optional)</label>
                <textarea
                  value={example.explanation || ''}
                  onChange={(e) => handleExampleChange(index, 'explanation', e.target.value)}
                  className="w-full p-2 border rounded h-16"
                  placeholder="Why this is a good annotation..."
                />
              </div>
              
              {examples.length > 1 && (
                <button
                  type="button"
                  onClick={() => removeExample(index)}
                  className="text-sm text-red-600"
                >
                  Remove
                </button>
              )}
            </div>
          ))}
        </div>
        
        <div className="flex gap-3">
          <button
            type="submit"
            disabled={isSubmitting}
            className="px-4 py-2 bg-blue-600 text-white rounded disabled:bg-blue-300"
          >
            {isSubmitting ? 'Creating...' : 'Create Persona'}
          </button>
          
          <button
            type="button"
            onClick={() => router.back()}
            className="px-4 py-2 border rounded text-gray-700"
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
}
```

---

## 5. Implementing Reinforcement Learning with Human Feedback

Now, let's implement the RLHF system to improve our annotations over time.

### 5.1 Feedback Collection System

First, we'll create types and interfaces for the feedback system:

```typescript
// src/types/feedback.ts
export interface FeedbackData {
  annotationId: string;
  userId: string;
  rating: number; // 1-5 scale
  comment?: string;
}

export interface FeedbackResult {
  id: string;
  annotationId: string;
  userId: string;
  rating: number;
  comment?: string;
  createdAt: Date;
}
```

Now, let's create the feedback service:

```typescript
// src/lib/services/feedbackService.ts
import { prisma } from '../db/prisma';
import { FeedbackData, FeedbackResult } from '@/types/feedback';

export class FeedbackService {
  async submitFeedback(data: FeedbackData): Promise<FeedbackResult> {
    const feedback = await prisma.feedback.create({
      data: {
        annotationId: data.annotationId,
        userId: data.userId,
        rating: data.rating,
        comment: data.comment,
      },
    });
    
    return feedback;
  }
  
  async getFeedbackForAnnotation(annotationId: string): Promise<FeedbackResult[]> {
    const feedback = await prisma.feedback.findMany({
      where: { annotationId },
      orderBy: { createdAt: 'desc' },
    });
    
    return feedback;
  }
  
  async getFeedbackForPersona(personaId: string, limit = 50): Promise<FeedbackResult[]> {
    const feedback = await prisma.feedback.findMany({
      where: {
        annotation: {
          personaId,
        },
      },
      include: {
        annotation: true,
      },
      orderBy: { createdAt: 'desc' },
      take: limit,
    });
    
    return feedback;
  }
  
  async getAverageFeedbackRating(personaId: string): Promise<number | null> {
    const result = await prisma.feedback.aggregate({
      where: {
        annotation: {
          personaId,
        },
      },
      _avg: {
        rating: true,
      },
    });
    
    return result._avg.rating;
  }
}

export const feedbackService = new FeedbackService();
```

Let's create the API routes for feedback:

```typescript
// src/app/api/feedback/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { feedbackService } from '@/lib/services/feedbackService';

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();
    
    // Validate request
    if (!data.annotationId || !data.userId || data.rating === undefined) {
      return NextResponse.json(
        { error: 'annotationId, userId, and rating are required' },
        { status: 400 }
      );
    }
    
    // Submit feedback
    const feedback = await feedbackService.submitFeedback(data);
    
    return NextResponse.json(feedback, { status: 201 });
  } catch (error) {
    console.error('Error submitting feedback:', error);
    return NextResponse.json(
      { error: 'Failed to submit feedback' },
      { status: 500 }
    );
  }
}
```

```typescript
// src/app/api/annotations/[annotationId]/feedback/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { feedbackService } from '@/lib/services/feedbackService';

export async function GET(
  request: NextRequest,
  { params }: { params: { annotationId: string } }
) {
  try {
    const annotationId = params.annotationId;
    
    // Get feedback for annotation
    const feedback = await feedbackService.getFeedbackForAnnotation(annotationId);
    
    return NextResponse.json(feedback);
  } catch (error) {
    console.error('Error fetching feedback:', error);
    return NextResponse.json(
      { error: 'Failed to fetch feedback' },
      { status: 500 }
    );
  }
}
```

### 5.2 RLHF Model for Persona Refinement

Now, let's implement the RLHF model to refine personas based on feedback:

```typescript
// src/lib/rlhf/personaRefinement.ts
import { prisma } from '../db/prisma';
import { personaService } from '../services/personaService';
import { feedbackService } from '../services/feedbackService';
import { ollamaService } from '../ollama';
import { PersonaData, PersonaTrait } from '@/types/persona';

export class PersonaRefinementService {
  /**
   * Analyzes feedback and suggests improvements to a persona
   */
  async analyzeAndRefinePersona(personaId: string): Promise<{
    originalPersona: PersonaData;
    refinedPersona: PersonaData;
    changes: string[];
  }> {
    // Get the current persona
    const originalPersona = await personaService.getPersona(personaId);
    
    if (!originalPersona) {
      throw new Error(`Persona ${personaId} not found`);
    }
    
    // Get feedback for this persona
    const feedback = await feedbackService.getFeedbackForPersona(personaId);
    
    if (feedback.length === 0) {
      throw new Error('No feedback available for refinement');
    }
    
    // Get annotations with positive and negative feedback
    const annotations = await prisma.annotation.findMany({
      where: { personaId },
      include: {
        feedback: true,
        item: true,
      },
    });
    
    // Organize into positive and negative examples
    const positiveExamples = annotations
      .filter(a => {
        const avgRating = a.feedback.reduce((sum, f) => sum + f.rating, 0) / a.feedback.length;
        return avgRating >= 4 && a.feedback.length > 0;
      })
      .map(a => ({
        input: a.item.content,
        output: a.annotation,
      }));
    
    const negativeExamples = annotations
      .filter(a => {
        const avgRating = a.feedback.reduce((sum, f) => sum + f.rating, 0) / a.feedback.length;
        return avgRating <= 2 && a.feedback.length > 0;
      })
      .map(a => ({
        input: a.item.content,
        output: a.annotation,
        feedback: a.feedback.map(f => f.comment).filter(Boolean).join('. '),
      }));
    
    // If we have no examples, we can't refine
    if (positiveExamples.length === 0 && negativeExamples.length === 0) {
      throw new Error('Not enough quality feedback to perform refinement');
    }
    
    // Use Ollama to suggest improvements
    const refinementPrompt = this.createRefinementPrompt(
      originalPersona,
      positiveExamples,
      negativeExamples,
      feedback
    );
    
    const response = await ollamaService.generate({
      prompt: refinementPrompt,
      temperature: 0.7,
    });
    
    // Parse the response to extract suggested changes
    const { refinedTraits, newExamples, changes } = this.parseRefinementResponse(
      response.text,
      originalPersona
    );
    
    // Create the refined persona
    const refinedPersona: PersonaData = {
      ...originalPersona,
      traits: refinedTraits,
      examples: [...originalPersona.examples, ...newExamples],
    };
    
    // Return both the original and refined personas, plus changes
    return {
      originalPersona,
      refinedPersona,
      changes,
    };
  }
  
  /**
   * Apply the suggested refinements to the persona
   */
  async applyRefinement(personaId: string, refinedPersona: PersonaData): Promise<PersonaData> {
    // Update the persona with the refinements
    const updatedPersona = await personaService.updatePersona(personaId, refinedPersona);
    return updatedPersona;
  }
  
  /**
   * Create a prompt for refining the persona
   */
  private createRefinementPrompt(
    persona: PersonaData,
    positiveExamples: Array<{ input: string; output: string }>,
    negativeExamples: Array<{ input: string; output: string; feedback?: string }>,
    feedback: any[]
  ): string {
    // Format the traits
    const traitsText = persona.traits
      .map(trait => `- ${trait.name}: ${trait.value.toFixed(2)} (${trait.description || ''})`)
      .join('\n');
    
    // Format the existing examples
    const existingExamplesText = persona.examples
      .map(example => `Input: ${example.input}\nOutput: ${example.output}${example.explanation ? `\nExplanation: ${example.explanation}` : ''}`)
      .join('\n\n');
    
    // Format the positive examples
    const positiveExamplesText = positiveExamples
      .map(example => `Input: ${example.input}\nOutput: ${example.output}\nFeedback: Positive (high rating)`)
      .join('\n\n');
    
    // Format the negative examples
    const negativeExamplesText = negativeExamples
      .map(example => `Input: ${example.input}\nOutput: ${example.output}\nFeedback: Negative (low rating)${example.feedback ? `\nFeedback comments: ${example.feedback}` : ''}`)
      .join('\n\n');
    
    // Extract common feedback themes
    const feedbackComments = feedback
      .filter(f => f.comment)
      .map(f => f.comment);
    
    // Create the prompt
    return `
You are an expert at refining AI personas based on human feedback. I need your help to improve an existing annotation persona.

CURRENT PERSONA:
Name: ${persona.name}
Description: ${persona.description}

Traits:
${traitsText}

Current system prompt: ${persona.prompt}

Existing examples:
${existingExamplesText}

FEEDBACK DATA:
${feedback.length} pieces of feedback received

Examples with positive feedback:
${positiveExamplesText || "None available"}

Examples with negative feedback:
${negativeExamplesText || "None available"}

Common feedback themes:
${feedbackComments.length > 0 ? feedbackComments.join('\n') : "No textual feedback provided"}

INSTRUCTIONS:
Based on the feedback data, please provide the following:

1. REFINED_TRAITS: Suggest adjustments to the trait values to improve performance. Return the full list of traits with adjusted values.
2. NEW_EXAMPLES: Suggest 1-2 new examples that would help the persona improve based on the feedback.
3. CHANGES_SUMMARY: Summarize the key changes you're recommending and why.

Format your response using these exact section headers:
REFINED_TRAITS:
[trait adjustments here]

NEW_EXAMPLES:
[new examples here]

CHANGES_SUMMARY:
[summary of changes here]
`;
  }
  
  /**
   * Parse the LLM response to extract refined traits, new examples, and changes
   */
  private parseRefinementResponse(
    response: string,
    originalPersona: PersonaData
  ): {
    refinedTraits: PersonaTrait[];
    newExamples: Array<{ input: string; output: string; explanation?: string }>;
    changes: string[];
  } {
    // Initialize with defaults
    let refinedTraits = [...originalPersona.traits];
    let newExamples: Array<{ input: string; output: string; explanation?: string }> = [];
    let changes: string[] = [];
    
    // Extract refined traits
    const traitsMatch = response.match(/REFINED_TRAITS:\s*([\s\S]*?)(?=NEW_EXAMPLES:|$)/);
    if (traitsMatch && traitsMatch[1]) {
      const traitsText = traitsMatch[1].trim();
      
      // Try to parse the traits
      try {
        // Look for patterns like "- Name: 0.8 (Description)" or "Name: 0.8"
        const traitRegex = /[-\s]*([^:]+):\s*([\d.]+)(?:\s*\(([^)]+)\))?/g;
        let match;
        
        const extractedTraits: PersonaTrait[] = [];
        
        while ((match = traitRegex.exec(traitsText)) !== null) {
          const name = match[1].trim();
          const value = parseFloat(match[2]);
          const description = match[3] ? match[3].trim() : undefined;
          
          // Find the original trait to preserve its description if not provided
          const originalTrait = originalPersona.traits.find(t => t.name.toLowerCase() === name.toLowerCase());
          
          extractedTraits.push({
            name,
            value: isNaN(value) ? 0.5 : Math.max(0, Math.min(1, value)), // Ensure value is between 0 and 1
            description: description || (originalTrait ? originalTrait.description : undefined),
          });
        }
        
        if (extractedTraits.length > 0) {
          refinedTraits = extractedTraits;
        }
      } catch (error) {
        console.error('Error parsing refined traits:', error);
      }
    }
    
    // Extract new examples
    const examplesMatch = response.match(/NEW_EXAMPLES:\s*([\s\S]*?)(?=CHANGES_SUMMARY:|$)/);
    if (examplesMatch && examplesMatch[1]) {
      const examplesText = examplesMatch[1].trim();
      
      // Try to parse the examples
      try {
        // Split by obvious example boundaries
        const exampleBlocks = examplesText.split(/(?:Example \d+:|Input:)/).filter(Boolean);
        
        for (const block of exampleBlocks) {
          const inputMatch = block.match(/Input:\s*([\s\S]*?)(?=Output:|$)/i) || 
                           block.match(/([\s\S]*?)(?=Output:|$)/i);
          const outputMatch = block.match(/Output:\s*([\s\S]*?)(?=Explanation:|$)/i);
          const explanationMatch = block.match(/Explanation:\s*([\s\S]*?)(?=$)/i);
          
          if (inputMatch && outputMatch) {
            newExamples.push({
              input: inputMatch[1].trim(),
              output: outputMatch[1].trim(),
              explanation: explanationMatch ? explanationMatch[1].trim() : undefined,
            });
          }
        }
      } catch (error) {
        console.error('Error parsing new examples:', error);
      }
    }
    
    // Extract changes summary
    const changesMatch = response.match(/CHANGES_SUMMARY:\s*([\s\S]*?)(?=$)/);
    if (changesMatch && changesMatch[1]) {
      const changesText = changesMatch[1].trim();
      
      // Split by bullet points or paragraphs
      changes = changesText
        .split(/\n+/)
        .map(line => line.replace(/^[-*•]\s*/, '').trim())
        .filter(Boolean);
    }
    
    return {
      refinedTraits,
      newExamples,
      changes,
    };
  }
}

export const personaRefinementService = new PersonaRefinementService();
```

Now, let's create the API routes for persona refinement:

```typescript
// src/app/api/personas/[personaId]/refine/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { personaRefinementService } from '@/lib/rlhf/personaRefinement';

export async function POST(
  request: NextRequest,
  { params }: { params: { personaId: string } }
) {
  try {
    const personaId = params.personaId;
    
    // Analyze feedback and refine persona
    const refinementResult = await personaRefinementService.analyzeAndRefinePersona(personaId);
    
    return NextResponse.json(refinementResult);
  } catch (error) {
    console.error('Error refining persona:', error);
    
    if (error instanceof Error && error.message.includes('No feedback available')) {
      return NextResponse.json(
        { error: error.message },
        { status: 400 }
      );
    }
    
    return NextResponse.json(
      { error: 'Failed to refine persona' },
      { status: 500 }
    );
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { personaId: string } }
) {
  try {
    const personaId = params.personaId;
    const data = await request.json();
    
    // Apply the refinements
    const updatedPersona = await personaRefinementService.applyRefinement(
      personaId,
      data.refinedPersona
    );
    
    return NextResponse.json(updatedPersona);
  } catch (error) {
    console.error('Error applying refinement:', error);
    return NextResponse.json(
      { error: 'Failed to apply refinement' },
      { status: 500 }
    );
  }
}
```

### 5.3 Feedback UI Components

Let's create UI components for collecting feedback:

```tsx
// src/components/FeedbackForm.tsx
'use client';

import { useState } from 'react';
import { StarIcon } from '@heroicons/react/24/solid';
import { StarIcon as StarOutlineIcon } from '@heroicons/react/24/outline';

interface FeedbackFormProps {
  annotationId: string;
  userId: string;
  onSubmitSuccess?: () => void;
  className?: string;
}

export default function FeedbackForm({
  annotationId,
  userId,
  onSubmitSuccess,
  className = '',
}: FeedbackFormProps) {
  const [rating, setRating] = useState<number | null>(null);
  const [comment, setComment] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (rating === null) {
      setError('Please provide a rating');
      return;
    }
    
    setIsSubmitting(true);
    setError('');
    
    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          annotationId,
          userId,
          rating,
          comment: comment.trim() || undefined,
        }),
      });
      
      if (response.ok) {
        setSuccess(true);
        setRating(null);
        setComment('');
        
        if (onSubmitSuccess) {
          onSubmitSuccess();
        }
      } else {
        const data = await response.json();
        setError(data.error || 'Failed to submit feedback');
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      setError('An unexpected error occurred');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <div className={`border rounded p-4 ${className}`}>
      <h3 className="font-medium mb-3">Provide Feedback</h3>
      
      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-800 rounded text-sm">
          {error}
        </div>
      )}
      
      {success && (
        <div className="mb-4 p-3 bg-green-100 text-green-800 rounded text-sm">
          Thank you for your feedback!
        </div>
      )}
      
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label className="block mb-2 text-sm">Rating</label>
          <div className="flex gap-1">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                type="button"
                onClick={() => setRating(star)}
                className="text-yellow-400 focus:outline-none"
              >
                {rating !== null && star <= rating ? (
                  <StarIcon className="h-6 w-6" />
                ) : (
                  <StarOutlineIcon className="h-6 w-6" />
                )}
              </button>
            ))}
          </div>
        </div>
        
        <div className="mb-4">
          <label className="block mb-2 text-sm">Comments (Optional)</label>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            className="w-full p-2 border rounded h-24 text-sm"
            placeholder="What did you like or dislike about this annotation?"
          />
        </div>
        
        <button
          type="submit"
          disabled={isSubmitting || rating === null}
          className="px-4 py-2 bg-blue-600 text-white rounded text-sm disabled:bg-blue-300"
        >
          {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
        </button>
      </form>
    </div>
  );
}
```

Now, let's update our annotation page to include the feedback form:

```tsx
// src/app/annotation/[annotationId]/page.tsx
'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import FeedbackForm from '@/components/FeedbackForm';

export default function AnnotationDetailPage() {
  const params = useParams();
  const annotationId = params.annotationId as string;
  
  const [annotation, setAnnotation] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  
  useEffect(() => {
    const fetchAnnotation = async () => {
      try {
        const response = await fetch(`/api/annotations/${annotationId}`);
        
        if (response.ok) {
          const data = await response.json();
          setAnnotation(data);
        } else {
          setError('Failed to fetch annotation');
        }
      } catch (error) {
        console.error('Error fetching annotation:', error);
        setError('An unexpected error occurred');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchAnnotation();
  }, [annotationId]);
  
  if (isLoading) {
    return <div className="text-center py-8">Loading annotation...</div>;
  }
  
  if (error) {
    return (
      <div className="max-w-3xl mx-auto">
        <div className="p-4 bg-red-100 text-red-800 rounded">
          {error}
        </div>
      </div>
    );
  }
  
  if (!annotation) {
    return (
      <div className="max-w-3xl mx-auto">
        <div className="p-4 bg-yellow-100 text-yellow-800 rounded">
          Annotation not found
        </div>
      </div>
    );
  }
  
  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Annotation Details</h1>
      
      <div className="mb-6 p-4 border rounded">
        <h2 className="text-lg font-medium mb-3">Original Content</h2>
        <div className="p-3 bg-gray-50 rounded">
          <p>{annotation.item?.content || 'Content not available'}</p>
        </div>
      </div>
      
      <div className="mb-8 p-4 border rounded">
        <h2 className="text-lg font-medium mb-3">Annotation</h2>
        <div className="p-3 bg-blue-50 rounded">
          <p>{annotation.annotation}</p>
        </div>
        <div className="mt-2 text-sm text-gray-600">
          <p>Confidence: {(annotation.confidence * 100).toFixed(1)}%</p>
          <p>Created: {new Date(annotation.createdAt).toLocaleString()}</p>
        </div>
      </div>
      
      <FeedbackForm
        annotationId={annotationId}
        userId="current-user" // In a real app, you'd get this from auth
        className="mb-8"
      />
    </div>
  );
}
```

### 5.4 Persona Refinement UI

Let's create a UI for refining personas based on feedback:

```tsx
// src/app/personas/[personaId]/refine/page.tsx
'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';

export default function PersonaRefinementPage() {
  const params = useParams();
  const router = useRouter();
  const personaId = params.personaId as string;
  
  const [originalPersona, setOriginalPersona] = useState<any>(null);
  const [refinedPersona, setRefinedPersona] = useState<any>(null);
  const [changes, setChanges] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isApplying, setIsApplying] = useState(false);
  const [error, setError] = useState('');
  
  const startRefinement = async () => {
    setIsLoading(true);
    setError('');
    
    try {
      const response = await fetch(`/api/personas/${personaId}/refine`, {
        method: 'POST',
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setOriginalPersona(data.originalPersona);
        setRefinedPersona(data.refinedPersona);
        setChanges(data.changes);
      } else {
        setError(data.error || 'Failed to refine persona');
      }
    } catch (error) {
      console.error('Error refining persona:', error);
      setError('An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  };
  
  const applyRefinement = async () => {
    if (!refinedPersona) return;
    
    setIsApplying(true);
    setError('');
    
    try {
      const response = await fetch(`/api/personas/${personaId}/refine`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          refinedPersona,
        }),
      });
      
      if (response.ok) {
        // Redirect to persona detail page
        router.push(`/personas/${personaId}`);
      } else {
        const data = await response.json();
        setError(data.error || 'Failed to apply refinement');
      }
    } catch (error) {
      console.error('Error applying refinement:', error);
      setError('An unexpected error occurred');
    } finally {
      setIsApplying(false);
    }
  };
  
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Persona Refinement</h1>
      
      {error && (
        <div className="mb-6 p-4 bg-red-100 text-red-800 rounded">
          {error}
        </div>
      )}
      
      {!refinedPersona && !isLoading && (
        <div className="mb-6 p-6 border rounded bg-gray-50 text-center">
          <p className="mb-4">
            This tool will analyze feedback data for this persona and suggest improvements
            based on user ratings and comments.
          </p>
          <button
            onClick={startRefinement}
            className="px-4 py-2 bg-blue-600 text-white rounded"
          >
            Start Refinement Process
          </button>
        </div>
      )}
      
      {isLoading && (
        <div className="mb-6 p-6 border rounded bg-gray-50 text-center">
          <p>Analyzing feedback and generating refinements...</p>
          <div className="mt-4 flex justify-center">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600"></div>
          </div>
        </div>
      )}
      
      {refinedPersona && (
        <div>
          <div className="mb-6 p-4 border rounded bg-blue-50">
            <h2 className="text-lg font-bold mb-3">Suggested Changes</h2>
            <ul className="list-disc pl-5 space-y-2">
              {changes.map((change, index) => (
                <li key={index}>{change}</li>
              ))}
            </ul>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div className="border rounded p-4">
              <h2 className="text-lg font-bold mb-3">Original Persona</h2>
              <div className="mb-3">
                <h3 className="font-medium text-sm text-gray-600">Traits</h3>
                <div className="mt-2 space-y-2">
                  {originalPersona.traits.map((trait: any, index: number) => (
                    <div key={index} className="flex justify-between">
                      <span>{trait.name}</span>
                      <span className="font-mono">{trait.value.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <h3 className="font-medium text-sm text-gray-600">Examples</h3>
                <p className="text-sm text-gray-500">
                  {originalPersona.examples.length} examples
                </p>
              </div>
            </div>
            
            <div className="border rounded p-4 bg-green-50">
              <h2 className="text-lg font-bold mb-3">Refined Persona</h2>
              <div className="mb-3">
                <h3 className="font-medium text-sm text-gray-600">Traits</h3>
                <div className="mt-2 space-y-2">
                  {refinedPersona.traits.map((trait: any, index: number) => {
                    const originalTrait = originalPersona.traits.find(
                      (t: any) => t.name === trait.name
                    );
                    const changed = originalTrait && originalTrait.value !== trait.value;
                    
                    return (
                      <div key={index} className="flex justify-between">
                        <span>{trait.name}</span>
                        <span className={`font-mono ${changed ? 'text-green-700 font-bold' : ''}`}>
                          {trait.value.toFixed(2)}
                          {changed && ` (was ${originalTrait.value.toFixed(2)})`}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
              
              <div>
                <h3 className="font-medium text-sm text-gray-600">Examples</h3>
                <p className="text-sm text-gray-500">
                  {refinedPersona.examples.length} examples 
                  {refinedPersona.examples.length > originalPersona.examples.length && 
                    ` (+${refinedPersona.examples.length - originalPersona.examples.length} new)`}
                </p>
              </div>
            </div>
          </div>
          
          <div className="flex gap-3">
            <button
              onClick={applyRefinement}
              disabled={isApplying}
              className="px-4 py-2 bg-green-600 text-white rounded disabled:bg-green-300"
            >
              {isApplying ? 'Applying Changes...' : 'Apply Refinements'}
            </button>
            
            <Link href={`/personas/${personaId}`} className="px-4 py-2 border rounded text-gray-700">
              Cancel
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## 6. Building a Scalable, Fully Local System

Now that we have implemented the core components, let's focus on configuring and running the complete system locally, optimizing for performance, and ensuring scalability.

### 6.1 Project Deployment Configuration

Let's create a configuration file for the full local deployment:

```typescript
// src/lib/config/deployment.ts
import path from 'path';
import os from 'os';

interface DeploymentConfig {
  database: {
    type: 'sqlite' | 'postgres';
    sqlitePath?: string;
    postgresConfig?: {
      host: string;
      port: number;
      user: string;
      password: string;
      database: string;
    };
  };
  ollama: {
    url: string;
    defaultModel: string;
    maxConcurrentRequests: number;
  };
  chromadb: {
    directory: string;
    pythonPath: string;
  };
  system: {
    tempDir: string;
    cacheDir: string;
    dataDir: string;
    maxConcurrency: number;
  };
}

// Default configuration for local development
const defaultConfig: DeploymentConfig = {
  database: {
    type: 'sqlite',
    sqlitePath: path.join(process.cwd(), 'prisma', 'dev.db'),
  },
  ollama: {
    url: 'http://localhost:11434',
    defaultModel: 'llama2',
    maxConcurrentRequests: 2, // Prevent overloading the local machine
  },
  chromadb: {
    directory: path.join(process.cwd(), 'chroma_db'),
    pythonPath: 'python', // Assumes Python is in PATH
  },
  system: {
    tempDir: path.join(os.tmpdir(), 'annotation-platform'),
    cacheDir: path.join(process.cwd(), '.cache'),
    dataDir: path.join(process.cwd(), 'data'),
    maxConcurrency: Math.max(1, Math.floor(os.cpus().length / 2)), // Use half of available CPU cores
  },
};

// Load configuration from environment variables if available
export const loadDeploymentConfig = (): DeploymentConfig => {
  const config = { ...defaultConfig };
  
  // Database configuration
  if (process.env.DATABASE_TYPE === 'postgres') {
    config.database = {
      type: 'postgres',
      postgresConfig: {
        host: process.env.POSTGRES_HOST || 'localhost',
        port: parseInt(process.env.POSTGRES_PORT || '5432', 10),
        user: process.env.POSTGRES_USER || 'postgres',
        password: process.env.POSTGRES_PASSWORD || 'postgres',
        database: process.env.POSTGRES_DB || 'annotation_platform',
      },
    };
  } else if (process.env.SQLITE_PATH) {
    config.database = {
      type: 'sqlite',
      sqlitePath: process.env.SQLITE_PATH,
    };
  }
  
  // Ollama configuration
  if (process.env.OLLAMA_URL) {
    config.ollama.url = process.env.OLLAMA_URL;
  }
  if (process.env.OLLAMA_DEFAULT_MODEL) {
    config.ollama.defaultModel = process.env.OLLAMA_DEFAULT_MODEL;
  }
  if (process.env.OLLAMA_MAX_CONCURRENT) {
    config.ollama.maxConcurrentRequests = parseInt(process.env.OLLAMA_MAX_CONCURRENT, 10);
  }
  
  // ChromaDB configuration
  if (process.env.CHROMADB_DIR) {
    config.chromadb.directory = process.env.CHROMADB_DIR;
  }
  if (process.env.PYTHON_PATH) {
    config.chromadb.pythonPath = process.env.PYTHON_PATH;
  }
  
  // System configuration
  if (process.env.TEMP_DIR) {
    config.system.tempDir = process.env.TEMP_DIR;
  }
  if (process.env.CACHE_DIR) {
    config.system.cacheDir = process.env.CACHE_DIR;
  }
  if (process.env.DATA_DIR) {
    config.system.dataDir = process.env.DATA_DIR;
  }
  if (process.env.MAX_CONCURRENCY) {
    config.system.maxConcurrency = parseInt(process.env.MAX_CONCURRENCY, 10);
  }
  
  return config;
};

export const deploymentConfig = loadDeploymentConfig();
```

### 6.2 System Health Monitoring Service

Let's create a service to monitor the health of our local dependencies:

```typescript
// src/lib/services/systemHealthService.ts
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import fetch from 'node-fetch';
import { deploymentConfig } from '../config/deployment';

export interface SystemStatus {
  database: {
    connected: boolean;
    type: string;
    error?: string;
  };
  ollama: {
    connected: boolean;
    models?: string[];
    error?: string;
  };
  chromadb: {
    connected: boolean;
    error?: string;
  };
  system: {
    cpuUsage: number; // percentage
    memoryUsage: number; // percentage
    diskSpace: {
      total: number; // bytes
      free: number; // bytes
      used: number; // percentage
    };
  };
}

export class SystemHealthService {
  async checkSystemHealth(): Promise<SystemStatus> {
    const status: SystemStatus = {
      database: await this.checkDatabaseStatus(),
      ollama: await this.checkOllamaStatus(),
      chromadb: await this.checkChromaDBStatus(),
      system: await this.getSystemResourceUsage(),
    };
    
    return status;
  }
  
  private async checkDatabaseStatus(): Promise<SystemStatus['database']> {
    try {
      const { prisma } = await import('../db/prisma');
      
      // Execute a simple query to check connection
      await prisma.$queryRaw`SELECT 1`;
      
      return {
        connected: true,
        type: deploymentConfig.database.type,
      };
    } catch (error) {
      console.error('Database connection error:', error);
      return {
        connected: false,
        type: deploymentConfig.database.type,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
  
  private async checkOllamaStatus(): Promise<SystemStatus['ollama']> {
    try {
      const response = await fetch(`${deploymentConfig.ollama.url}/api/tags`);
      
      if (!response.ok) {
        throw new Error(`Ollama server returned ${response.status}`);
      }
      
      const data = await response.json() as any;
      const models = data.models ? data.models.map((model: any) => model.name) : [];
      
      return {
        connected: true,
        models,
      };
    } catch (error) {
      console.error('Ollama connection error:', error);
      return {
        connected: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
  
  private async checkChromaDBStatus(): Promise<SystemStatus['chromadb']> {
    try {
      // Run a Python script to check ChromaDB
      const scriptPath = path.join(process.cwd(), 'scripts', 'chromadb', 'check_status.py');
      
      if (!fs.existsSync(scriptPath)) {
        // Create the script if it doesn't exist
        this.createChromaDBCheckScript(scriptPath);
      }
      
      const chromaDir = deploymentConfig.chromadb.directory;
      const pythonPath = deploymentConfig.chromadb.pythonPath;
      
      const result = await this.runPythonScript(pythonPath, [scriptPath, chromaDir]);
      const status = JSON.parse(result);
      
      return {
        connected: status.connected,
        error: status.error,
      };
    } catch (error) {
      console.error('ChromaDB check error:', error);
      return {
        connected: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
  
  private async getSystemResourceUsage(): Promise<SystemStatus['system']> {
    try {
      const cpuUsage = await this.getCpuUsage();
      const memoryUsage = this.getMemoryUsage();
      const diskSpace = this.getDiskSpace(process.cwd());
      
      return {
        cpuUsage,
        memoryUsage,
        diskSpace,
      };
    } catch (error) {
      console.error('Error getting system resource usage:', error);
      return {
        cpuUsage: 0,
        memoryUsage: 0,
        diskSpace: {
          total: 0,
          free: 0,
          used: 0,
        },
      };
    }
  }
  
  private async getCpuUsage(): Promise<number> {
    // A simple method to estimate CPU usage on different platforms
    return new Promise((resolve) => {
      let scriptPath = '';
      
      if (process.platform === 'win32') {
        scriptPath = path.join(process.cwd(), 'scripts', 'system', 'cpu_usage_windows.ps1');
        this.createWindowsCpuScript(scriptPath);
        
        const process = spawn('powershell', ['-ExecutionPolicy', 'Bypass', '-File', scriptPath]);
        let output = '';
        
        process.stdout.on('data', (data) => {
          output += data.toString();
        });
        
        process.on('close', () => {
          const usage = parseFloat(output.trim());
          resolve(isNaN(usage) ? 0 : usage);
        });
      } else {
        // Linux/macOS
        const process = spawn('sh', ['-c', 'top -bn1 | grep "%Cpu(s)" | sed "s/.*, *\\([0-9.]*\\)%* id.*/\\1/" | awk \'{print 100 - $1}\'']);
        let output = '';
        
        process.stdout.on('data', (data) => {
          output += data.toString();
        });
        
        process.on('close', () => {
          const usage = parseFloat(output.trim());
          resolve(isNaN(usage) ? 0 : usage);
        });
      }
    });
  }
  
  private getMemoryUsage(): number {
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    return Math.round(((totalMem - freeMem) / totalMem) * 100);
  }
  
  private getDiskSpace(directory: string): SystemStatus['system']['diskSpace'] {
    try {
      if (process.platform === 'win32') {
        // Windows implementation is more complex, simplified version here
        return {
          total: 100e9, // 100 GB placeholder
          free: 50e9,   // 50 GB placeholder
          used: 50,     // 50% placeholder
        };
      } else {
        // Linux/macOS
        const { stdout } = require('child_process').execSync(`df -k "${directory}"`);
        const lines = stdout.toString().trim().split('\n');
        const data = lines[1].split(/\s+/);
        
        const total = parseInt(data[1]) * 1024;
        const used = parseInt(data[2]) * 1024;
        const free = parseInt(data[3]) * 1024;
        const usedPercentage = Math.round((used / total) * 100);
        
        return {
          total,
          free,
          used: usedPercentage,
        };
      }
    } catch (error) {
      console.error('Error getting disk space:', error);
      return {
        total: 0,
        free: 0,
        used: 0,
      };
    }
  }
  
  private runPythonScript(pythonPath: string, args: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
      const process = spawn(pythonPath, args);
      
      let output = '';
      let errorOutput = '';
      
      process.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      process.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });
      
      process.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Python script error: ${errorOutput}`));
        } else {
          resolve(output.trim());
        }
      });
    });
  }
  
  private createChromaDBCheckScript(scriptPath: string) {
    const scriptDir = path.dirname(scriptPath);
    
    if (!fs.existsSync(scriptDir)) {
      fs.mkdirSync(scriptDir, { recursive: true });
    }
    
    const scriptContent = `
import sys
import json
import os

def check_chromadb(chroma_dir):
    try:
        import chromadb
        
        # Check if ChromaDB directory exists
        if not os.path.exists(chroma_dir):
            return {
                "connected": False,
                "error": f"ChromaDB directory {chroma_dir} does not exist"
            }
        
        # Try to initialize a client
        client = chromadb.PersistentClient(path=chroma_dir)
        
        # Try to list collections to verify connection
        collections = client.list_collections()
        
        return {
            "connected": True
        }
    except ImportError:
        return {
            "connected": False,
            "error": "ChromaDB Python package is not installed"
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "connected": False,
            "error": "ChromaDB directory path not provided"
        }))
        sys.exit(1)
    
    chroma_dir = sys.argv[1]
    status = check_chromadb(chroma_dir)
    print(json.dumps(status))
`;
    
    fs.writeFileSync(scriptPath, scriptContent);
  }
  
  private createWindowsCpuScript(scriptPath: string) {
    const scriptDir = path.dirname(scriptPath);
    
    if (!fs.existsSync(scriptDir)) {
      fs.mkdirSync(scriptDir, { recursive: true });
    }
    
    const scriptContent = `
$CpuUsage = (Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples.CookedValue
Write-Output $CpuUsage
`;
    
    fs.writeFileSync(scriptPath, scriptContent);
  }
}

export const systemHealthService = new SystemHealthService();
```

Let's create an API route for system health:

```typescript
// src/app/api/system/health/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { systemHealthService } from '@/lib/services/systemHealthService';

export async function GET() {
  try {
    const health = await systemHealthService.checkSystemHealth();
    return NextResponse.json(health);
  } catch (error) {
    console.error('Error checking system health:', error);
    return NextResponse.json(
      { error: 'Failed to check system health' },
      { status: 500 }
    );
  }
}
```

### 6.3 Request Queue Management

To prevent overloading the local system, let's implement a request queue for LLM calls:

```typescript
// src/lib/queue/requestQueue.ts
import pLimit from 'p-limit';
import { deploymentConfig } from '../config/deployment';

// Function type for tasks that can be queued
type QueueableFunction<T> = () => Promise<T>;

// Interface for the result of a queued task
interface QueueResult<T> {
  data?: T;
  error?: Error;
  queueTime: number; // milliseconds spent in queue
  processingTime: number; // milliseconds spent processing
  totalTime: number; // total milliseconds
}

export class RequestQueue {
  private queue: pLimit.Limit;
  private activeRequests = 0;
  private maxConcurrent: number;
  
  constructor(maxConcurrent = deploymentConfig.ollama.maxConcurrentRequests) {
    this.maxConcurrent = maxConcurrent;
    this.queue = pLimit(maxConcurrent);
  }
  
  /**
   * Add a task to the queue
   */
  async enqueue<T>(task: QueueableFunction<T>): Promise<QueueResult<T>> {
    const queueStart = Date.now();
    
    try {
      this.activeRequests++;
      
      const processingStart = Date.now();
      const queueTime = processingStart - queueStart;
      
      const data = await this.queue(task);
      
      const processingEnd = Date.now();
      const processingTime = processingEnd - processingStart;
      
      return {
        data,
        queueTime,
        processingTime,
        totalTime: queueTime + processingTime,
      };
    } catch (error) {
      const processingEnd = Date.now();
      const processingTime = processingEnd - queueStart;
      
      return {
        error: error instanceof Error ? error : new Error(String(error)),
        queueTime: 0, // Unknown actual queue time in case of error
        processingTime,
        totalTime: processingTime,
      };
    } finally {
      this.activeRequests--;
    }
  }
  
  /**
   * Get queue statistics
   */
  getStats() {
    return {
      activeRequests: this.activeRequests,
      pendingRequests: this.queue.pendingCount,
      maxConcurrent: this.maxConcurrent,
    };
  }
}

// Create a singleton instance for the Ollama LLM requests
export const ollamaQueue = new RequestQueue();
```

Update the Ollama service to use the queue:

```typescript
// src/lib/ollama/index.ts
import { ollamaQueue } from '../queue/requestQueue';
// ... existing imports ...

export class OllamaService {
  // ... existing properties ...

  async generate(options: GenerationOptions): Promise<GenerationResponse> {
    const result = await ollamaQueue.enqueue(async () => {
      const response = await fetch(`${this.baseUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.model,
          prompt: options.prompt,
          system: options.system,
          options: {
            temperature: options.temperature ?? 0.7,
            num_predict: options.maxTokens,
          },
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Ollama API error: ${response.status} ${errorText}`);
      }

      const data = await response.json();
      return {
        text: data.response,
        model: data.model,
        promptTokens: data.prompt_eval_count,
        generatedTokens: data.eval_count,
      };
    });
    
    if (result.error) {
      throw result.error;
    }
    
    return result.data!;
  }
  
  // ... existing methods ...
}
```

### 6.4 System-wide Caching

Let's implement a caching system to improve performance:

```typescript
// src/lib/cache/index.ts
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { deploymentConfig } from '../config/deployment';

interface CacheOptions {
  ttl?: number; // Time to live in seconds
  namespace?: string; // For organizing cached items
}

interface CacheItem<T> {
  value: T;
  expires: number; // Unix timestamp in milliseconds
}

export class CacheService {
  private cacheDir: string;
  
  constructor(cacheDir = deploymentConfig.system.cacheDir) {
    this.cacheDir = cacheDir;
    this.ensureCacheDir();
  }
  
  /**
   * Get a value from cache
   */
  async get<T>(key: string, options: CacheOptions = {}): Promise<T | null> {
    const cacheKey = this.getCacheKey(key, options.namespace);
    const cacheFile = this.getCacheFilePath(cacheKey);
    
    try {
      if (!fs.existsSync(cacheFile)) {
        return null;
      }
      
      const cacheData = JSON.parse(fs.readFileSync(cacheFile, 'utf8')) as CacheItem<T>;
      
      // Check if cache has expired
      if (cacheData.expires && cacheData.expires < Date.now()) {
        await this.delete(key, options);
        return null;
      }
      
      return cacheData.value;
    } catch (error) {
      console.error('Cache read error:', error);
      return null;
    }
  }
  
  /**
   * Set a value in cache
   */
  async set<T>(key: string, value: T, options: CacheOptions = {}): Promise<void> {
    const cacheKey = this.getCacheKey(key, options.namespace);
    const cacheFile = this.getCacheFilePath(cacheKey);
    
    try {
      const expires = options.ttl ? Date.now() + options.ttl * 1000 : 0;
      
      const cacheData: CacheItem<T> = {
        value,
        expires,
      };
      
      fs.writeFileSync(cacheFile, JSON.stringify(cacheData), 'utf8');
    } catch (error) {
      console.error('Cache write error:', error);
    }
  }
  
  /**
   * Delete a value from cache
   */
  async delete(key: string, options: CacheOptions = {}): Promise<void> {
    const cacheKey = this.getCacheKey(key, options.namespace);
    const cacheFile = this.getCacheFilePath(cacheKey);
    
    try {
      if (fs.existsSync(cacheFile)) {
        fs.unlinkSync(cacheFile);
      }
    } catch (error) {
      console.error('Cache delete error:', error);
    }
  }
  
  /**
   * Clear all cache or specific namespace
   */
  async clear(namespace?: string): Promise<void> {
    try {
      if (namespace) {
        // Clear specific namespace
        const namespaceDir = path.join(this.cacheDir, namespace);
        
        if (fs.existsSync(namespaceDir)) {
          fs.readdirSync(namespaceDir).forEach((file) => {
            fs.unlinkSync(path.join(namespaceDir, file));
          });
        }
      } else {
        // Clear all cache
        fs.readdirSync(this.cacheDir).forEach((dir) => {
          const dirPath = path.join(this.cacheDir, dir);
          
          if (fs.statSync(dirPath).isDirectory()) {
            fs.readdirSync(dirPath).forEach((file) => {
              fs.unlinkSync(path.join(dirPath, file));
            });
          }
        });
      }
    } catch (error) {
      console.error('Cache clear error:', error);
    }
  }
  
  /**
   * Create cache key
   */
  private getCacheKey(key: string, namespace = 'default'): string {
    return `${namespace}:${crypto.createHash('md5').update(key).digest('hex')}`;
  }
  
  /**
   * Get cache file path
   */
  private getCacheFilePath(cacheKey: string): string {
    const [namespace, hash] = cacheKey.split(':');
    const namespaceDir = path.join(this.cacheDir, namespace);
    
    if (!fs.existsSync(namespaceDir)) {
      fs.mkdirSync(namespaceDir, { recursive: true });
    }
    
    return path.join(namespaceDir, `${hash}.json`);
  }
  
  /**
   * Ensure cache directory exists
   */
  private ensureCacheDir(): void {
    if (!fs.existsSync(this.cacheDir)) {
      fs.mkdirSync(this.cacheDir, { recursive: true });
    }
  }
}

// Create a singleton instance
export const cacheService = new CacheService();
```

Update the annotation service to use caching:

```typescript
// src/lib/services/annotationService.ts
import { cacheService } from '../cache';
// ... existing imports ...

export class AnnotationService {
  async generateAnnotation(request: AnnotationRequest): Promise<AnnotationResult> {
    // Check cache first
    const cacheKey = `annotation:${request.personaId}:${Buffer.from(request.content).toString('base64')}`;
    const cachedResult = await cacheService.get<AnnotationResult>(cacheKey, {
      namespace: 'annotations',
      ttl: 3600, // 1 hour cache
    });
    
    if (cachedResult) {
      return cachedResult;
    }
    
    // Get the persona
    const persona = await personaService.getPersona(request.personaId);
    
    if (!persona) {
      throw new Error(`Persona ${request.personaId} not found`);
    }
    
    // Get item from database or create a temporary one if not provided
    let item;
    if (request.itemId) {
      item = await prisma.item.findUnique({
        where: { id: request.itemId },
      });
      
      if (!item) {
        throw new Error(`Item ${request.itemId} not found`);
      }
    }
    
    // Prepare the prompt for annotation
    const prompt = `Please analyze the following content and provide an annotation:

${request.content}`;

    // Generate annotation using Ollama
    const ollamaResponse = await ollamaService.generate({
      prompt,
      system: persona.prompt,
      temperature: 0.3, // Lower temperature for more focused annotations
    });
    
    // Calculate a simple confidence score
    const confidence = this.calculateConfidence(ollamaResponse.text);
    
    // Save annotation to database if we have an item
    let annotation;
    if (request.itemId) {
      annotation = await prisma.annotation.create({
        data: {
          itemId: request.itemId,
          personaId: request.personaId,
          annotation: ollamaResponse.text,
          confidence,
        },
      });
    } else {
      // Create an ephemeral annotation result
      annotation = {
        id: 'temp-' + Date.now(),
        itemId: 'temp-item',
        personaId: request.personaId,
        annotation: ollamaResponse.text,
        confidence,
        createdAt: new Date(),
      };
    }
    
    // Cache the result
    await cacheService.set(cacheKey, annotation, {
      namespace: 'annotations',
      ttl: 3600, // 1 hour cache
    });
    
    return annotation;
  }
  
  // ... rest of the class ...
}
```

### 6.5 Running and Configuring the Complete System

Let's create a script to help users run and configure the complete system locally:

```typescript
// scripts/setup-local-environment.js
#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const readline = require('readline');
const os = require('os');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const ask = (question) => new Promise(resolve => rl.question(question, resolve));

const configFile = path.join(process.cwd(), '.env.local');

async function main() {
  console.log('=== Local Annotation Platform Setup ===');
  console.log('This script will help you set up a local development environment.\n');
  
  // Check for required software
  checkRequirements();
  
  // Database configuration
  const dbType = await ask('Which database would you like to use? (sqlite/postgres) [sqlite]: ');
  
  let databaseConfig = '';
  
  if (dbType.toLowerCase() === 'postgres') {
    console.log('\nConfiguring PostgreSQL:');
    
    const pgHost = await ask('PostgreSQL host [localhost]: ');
    const pgPort = await ask('PostgreSQL port [5432]: ');
    const pgUser = await ask('PostgreSQL username [postgres]: ');
    const pgPassword = await ask('PostgreSQL password: ');
    const pgDatabase = await ask('PostgreSQL database name [annotation_platform]: ');
    
    databaseConfig = `
# Database Configuration
DATABASE_TYPE=postgres
POSTGRES_HOST=${pgHost || 'localhost'}
POSTGRES_PORT=${pgPort || '5432'}
POSTGRES_USER=${pgUser || 'postgres'}
POSTGRES_PASSWORD=${pgPassword}
POSTGRES_DB=${pgDatabase || 'annotation_platform'}
DATABASE_URL="postgresql://${pgUser || 'postgres'}:${pgPassword}@${pgHost || 'localhost'}:${pgPort || '5432'}/${pgDatabase || 'annotation_platform'}?schema=public"
`;
  } else {
    databaseConfig = `
# Database Configuration
DATABASE_TYPE=sqlite
SQLITE_PATH="./prisma/dev.db"
DATABASE_URL="file:./dev.db"
`;
  }
  
  // Ollama configuration
  console.log('\nConfiguring Ollama:');
  
  const ollamaUrl = await ask('Ollama API URL [http://localhost:11434]: ');
  const ollamaModel = await ask('Default Ollama model [llama2]: ');
  const ollmaMaxConcurrent = await ask('Maximum concurrent Ollama requests [2]: ');
  
  const ollamaConfig = `
# Ollama Configuration
OLLAMA_URL=${ollamaUrl || 'http://localhost:11434'}
OLLAMA_DEFAULT_MODEL=${ollamaModel || 'llama2'}
OLLAMA_MAX_CONCURRENT=${ollmaMaxConcurrent || '2'}
NEXT_PUBLIC_OLLAMA_BASE_URL=${ollamaUrl || 'http://localhost:11434'}
NEXT_PUBLIC_OLLAMA_DEFAULT_MODEL=${ollamaModel || 'llama2'}
`;
  
  // ChromaDB configuration
  console.log('\nConfiguring ChromaDB:');
  
  const chromaDir = await ask(`ChromaDB directory [${path.join(process.cwd(), 'chroma_db')}]: `);
  const pythonPath = await ask('Python executable path [python]: ');
  
  const chromaConfig = `
# ChromaDB Configuration
CHROMADB_DIR=${chromaDir || path.join(process.cwd(), 'chroma_db')}
PYTHON_PATH=${pythonPath || 'python'}
`;
  
  // System configuration
  console.log('\nConfiguring system parameters:');
  
  const tempDir = await ask(`Temporary directory [${os.tmpdir()}]: `);
  const cacheDir = await ask(`Cache directory [${path.join(process.cwd(), '.cache')}]: `);
  const dataDir = await ask(`Data directory [${path.join(process.cwd(), 'data')}]: `);
  const maxConcurrency = await ask(`Maximum concurrency [${Math.max(1, Math.floor(os.cpus().length / 2))}]: `);
  
  const systemConfig = `
# System Configuration
TEMP_DIR=${tempDir || os.tmpdir()}
CACHE_DIR=${cacheDir || path.join(process.cwd(), '.cache')}
DATA_DIR=${dataDir || path.join(process.cwd(), 'data')}
MAX_CONCURRENCY=${maxConcurrency || Math.max(1, Math.floor(os.cpus().length / 2))}
`;
  
  // Write configuration to .env.local
  fs.writeFileSync(configFile, `${databaseConfig}${ollamaConfig}${chromaConfig}${systemConfig}`);
  
  console.log('\nConfiguration saved to .env.local');
  
  // Setup database
  if (dbType.toLowerCase() === 'postgres') {
    console.log('\nSetting up PostgreSQL database...');
    try {
      execSync('npx prisma migrate dev --name init');
      console.log('Database migrations applied successfully.');
    } catch (error) {
      console.error('\nError setting up PostgreSQL database:');
      console.error(error.message);
      console.log('\nMake sure your PostgreSQL server is running and accessible.');
    }
  } else {
    console.log('\nSetting up SQLite database...');
    try {
      execSync('npx prisma migrate dev --name init');
      console.log('Database migrations applied successfully.');
    } catch (error) {
      console.error('\nError setting up SQLite database:');
      console.error(error.message);
    }
  }
  
  // Create required directories
  createDirectories([
    chromaDir || path.join(process.cwd(), 'chroma_db'),
    cacheDir || path.join(process.cwd(), '.cache'),
    dataDir || path.join(process.cwd(), 'data'),
    path.join(process.cwd(), 'scripts', 'chromadb'),
    path.join(process.cwd(), 'scripts', 'system'),
  ]);
  
  // Check Ollama installation
  checkOllama(ollamaUrl || 'http://localhost:11434', ollamaModel || 'llama2');
  
  // Check ChromaDB installation
  checkChromaDB(pythonPath || 'python');
  
  console.log('\n=== Setup Complete ===');
  console.log('You can now start the application with:');
  console.log('  npm run dev');
  console.log('\nMake sure Ollama is running with your preferred model.');
  
  rl.close();
}

function checkRequirements() {
  console.log('Checking system requirements...');
  
  // Check Node.js
  try {
    const nodeVersion = execSync('node --version').toString().trim();
    console.log(`✅ Node.js: ${nodeVersion}`);
  } catch (error) {
    console.error('❌ Node.js: Not found or not in PATH');
    console.error('Please install Node.js v18.17.0 or later');
    process.exit(1);
  }
  
  // Check npm
  try {
    const npmVersion = execSync('npm --version').toString().trim();
    console.log(`✅ npm: ${npmVersion}`);
  } catch (error) {
    console.error('❌ npm: Not found or not in PATH');
    console.error('Please ensure npm is installed with Node.js');
    process.exit(1);
  }
  
  // Check Python
  try {
    const pythonVersion = execSync('python --version').toString().trim();
    console.log(`✅ Python: ${pythonVersion}`);
  } catch (error) {
    try {
      const python3Version = execSync('python3 --version').toString().trim();
      console.log(`✅ Python: ${python3Version}`);
    } catch (error) {
      console.error('❌ Python: Not found or not in PATH');
      console.error('Please install Python 3.9 or later');
      process.exit(1);
    }
  }
  
  console.log('All system requirements met.\n');
}

function createDirectories(dirs) {
  console.log('\nCreating required directories...');
  
  for (const dir of dirs) {
    try {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
        console.log(`✅ Created directory: ${dir}`);
      } else {
        console.log(`✅ Directory already exists: ${dir}`);
      }
    } catch (error) {
      console.error(`❌ Failed to create directory ${dir}: ${error.message}`);
    }
  }
}

async function checkOllama(ollamaUrl, defaultModel) {
  console.log('\nChecking Ollama installation...');
  
  try {
    const response = await fetch(`${ollamaUrl}/api/tags`);
    
    if (response.ok) {
      console.log('✅ Ollama is running');
      
      const data = await response.json();
      const models = data.models ? data.models.map(model => model.name) : [];
      
      if (models.length > 0) {
        console.log(`Available models: ${models.join(', ')}`);
        
        if (models.includes(defaultModel)) {
          console.log(`✅ Default model "${defaultModel}" is available`);
        } else {
          console.log(`❌ Default model "${defaultModel}" is not available`);
          console.log(`You can pull it with: ollama pull ${defaultModel}`);
        }
      } else {
        console.log('No models found. You need to pull a model:');
        console.log(`  ollama pull ${defaultModel}`);
      }
    } else {
      console.log('❌ Ollama is not running or not accessible at the specified URL');
      console.log('Please start Ollama before running the application');
    }
  } catch (error) {
    console.error('❌ Ollama check failed:', error.message);
    console.log('Please ensure Ollama is installed and running:');
    console.log('  https://ollama.ai/download');
  }
}

async function checkChromaDB(pythonPath) {
  console.log('\nChecking ChromaDB installation...');
  
  try {
    execSync(`${pythonPath} -c "import chromadb"`);
    console.log('✅ ChromaDB Python package is installed');
  } catch (error) {
    console.error('❌ ChromaDB Python package is not installed');
    console.log('Installing ChromaDB and sentence-transformers...');
    
    try {
      execSync(`${pythonPath} -m pip install chromadb sentence-transformers`);
      console.log('✅ ChromaDB and sentence-transformers installed successfully');
    } catch (installError) {
      console.error('❌ Failed to install ChromaDB:', installError.message);
      console.log('Please install manually with:');
      console.log('  pip install chromadb sentence-transformers');
    }
  }
}

main().catch(error => {
  console.error('Setup failed:', error);
  rl.close();
  process.exit(1);
});
```

Add this script to package.json:

```json
{
  "scripts": {
    "setup": "node scripts/setup-local-environment.js",
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  }
}
```

### 6.6 Dashboard for System Monitoring

Let's create a dashboard to monitor the system:

```tsx
// src/app/settings/page.tsx
'use client';

import { useState, useEffect } from 'react';

interface SystemHealth {
  database: {
    connected: boolean;
    type: string;
    error?: string;
  };
  ollama: {
    connected: boolean;
    models?: string[];
    error?: string;
  };
  chromadb: {
    connected: boolean;
    error?: string;
  };
  system: {
    cpuUsage: number;
    memoryUsage: number;
    diskSpace: {
      total: number;
      free: number;
      used: number;
    };
  };
}

export default function SettingsPage() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [refreshCounter, setRefreshCounter] = useState(0);
  
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        setIsLoading(true);
        const response = await fetch('/api/system/health');
        
        if (response.ok) {
          const data = await response.json();
          setHealth(data);
        } else {
          setError('Failed to fetch system health');
        }
      } catch (error) {
        console.error('Error fetching system health:', error);
        setError('An unexpected error occurred');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchHealth();
    
    // Refresh every 30 seconds
    const interval = setInterval(() => {
      setRefreshCounter(prev => prev + 1);
    }, 30000);
    
    return () => clearInterval(interval);
  }, [refreshCounter]);
  
  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };
  
  const getStatusColor = (isConnected: boolean) => {
    return isConnected ? 'bg-green-500' : 'bg-red-500';
  };
  
  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">System Settings & Monitoring</h1>
      
      {error && (
        <div className="mb-6 p-4 bg-red-100 text-red-800 rounded">
          {error}
        </div>
      )}
      
      {isLoading && !health ? (
        <div className="text-center py-8">Loading system status...</div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {/* Database Status */}
            <div className="border rounded p-4">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-bold">Database</h2>
                <div className={`w-3 h-3 rounded-full ${health ? getStatusColor(health.database.connected) : 'bg-gray-300'}`}></div>
              </div>
              
              {health && (
                <>
                  <p className="mb-2">Type: {health.database.type}</p>
                  <p className="mb-2">Status: {health.database.connected ? 'Connected' : 'Disconnected'}</p>
                  {health.database.error && (
                    <p className="text-red-600 text-sm">{health.database.error}</p>
                  )}
                </>
              )}
            </div>
            
            {/* Ollama Status */}
            <div className="border rounded p-4">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-bold">Ollama</h2>
                <div className={`w-3 h-3 rounded-full ${health ? getStatusColor(health.ollama.connected) : 'bg-gray-300'}`}></div>
              </div>
              
              {health && (
                <>
                  <p className="mb-2">Status: {health.ollama.connected ? 'Connected' : 'Disconnected'}</p>
                  
                  {health.ollama.models && health.ollama.models.length > 0 && (
                    <div className="mb-2">
                      <p className="font-medium">Available Models:</p>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {health.ollama.models.map((model, index) => (
                          <span key={index} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                            {model}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {health.ollama.error && (
                    <p className="text-red-600 text-sm">{health.ollama.error}</p>
                  )}
                </>
              )}
            </div>
            
            {/* ChromaDB Status */}
            <div className="border rounded p-4">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-bold">ChromaDB</h2>
                <div className={`w-3 h-3 rounded-full ${health ? getStatusColor(health.chromadb.connected) : 'bg-gray-300'}`}></div>
              </div>
              
              {health && (
                <>
                  <p className="mb-2">Status: {health.chromadb.connected ? 'Connected' : 'Disconnected'}</p>
                  
                  {health.chromadb.error && (
                    <p className="text-red-600 text-sm">{health.chromadb.error}</p>
                  )}
                </>
              )}
            </div>
          </div>
          
          {/* System Resources */}
          {health && (
            <div className="border rounded p-4 mb-8">
              <h2 className="text-lg font-bold mb-4">System Resources</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* CPU Usage */}
                <div>
                  <h3 className="font-medium mb-2">CPU Usage</h3>
                  <div className="w-full bg-gray-200 rounded-full h-4">
                    <div
                      className={`h-4 rounded-full ${health.system.cpuUsage > 80 ? 'bg-red-500' : health.system.cpuUsage > 60 ? 'bg-yellow-500' : 'bg-green-500'}`}
                      style={{ width: `${health.system.cpuUsage}%` }}
                    ></div>
                  </div>
                  <p className="text-right text-sm mt-1">{health.system.cpuUsage.toFixed(1)}%</p>
                </div>
                
                {/* Memory Usage */}
                <div>
                  <h3 className="font-medium mb-2">Memory Usage</h3>
                  <div className="w-full bg-gray-200 rounded-full h-4">
                    <div
                      className={`h-4 rounded-full ${health.system.memoryUsage > 80 ? 'bg-red-500' : health.system.memoryUsage > 60 ? 'bg-yellow-500' : 'bg-green-500'}`}
                      style={{ width: `${health.system.memoryUsage}%` }}
                    ></div>
                  </div>
                  <p className="text-right text-sm mt-1">{health.system.memoryUsage.toFixed(1)}%</p>
                </div>
                
                {/* Disk Usage */}
                <div>
                  <h3 className="font-medium mb-2">Disk Usage</h3>
                  <div className="w-full bg-gray-200 rounded-full h-4">
                    <div
                      className={`h-4 rounded-full ${health.system.diskSpace.used > 80 ? 'bg-red-500' : health.system.diskSpace.used > 60 ? 'bg-yellow-500' : 'bg-green-500'}`}
                      style={{ width: `${health.system.diskSpace.used}%` }}
                    ></div>
                  </div>
                  <div className="flex justify-between text-sm mt-1">
                    <span>Free: {formatBytes(health.system.diskSpace.free)}</span>
                    <span>{health.system.diskSpace.used.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Admin Actions */}
          <div className="border rounded p-4">
            <h2 className="text-lg font-bold mb-4">Administration</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <button
                onClick={() => setRefreshCounter(prev => prev + 1)}
                className="px-4 py-2 bg-blue-600 text-white rounded"
              >
                Refresh System Status
              </button>
              
              <button
                onClick={() => {
                  if (confirm('Are you sure you want to clear the system cache?')) {
                    fetch('/api/system/cache/clear', { method: 'POST' })
                      .then(response => {
                        if (response.ok) {
                          alert('Cache cleared successfully');
                        } else {
                          alert('Failed to clear cache');
                        }
                      })
                      .catch(error => {
                        console.error('Error clearing cache:', error);
                        alert('Error clearing cache');
                      });
                  }
                }}
                className="px-4 py-2 bg-yellow-600 text-white rounded"
              >
                Clear System Cache
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
```

Create the API endpoint for clearing the cache:

```typescript
// src/app/api/system/cache/clear/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { cacheService } from '@/lib/cache';

export async function POST() {
  try {
    await cacheService.clear();
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error clearing cache:', error);
    return NextResponse.json(
      { error: 'Failed to clear cache' },
      { status: 500 }
    );
  }
}
```

---

## 7. Advanced Topics and Future Enhancements

### 7.1 Multi-User Collaboration

To enhance the platform with multi-user collaboration capabilities, we can extend our system with the following components:

#### 7.1.1 Real-time Updates with WebSockets

First, let's implement WebSocket support for real-time collaboration:

```typescript
// src/lib/websocket/index.ts
import { Server as HTTPServer } from 'http';
import { Server as WebSocketServer } from 'ws';
import { parse } from 'url';

// Define message types
export enum MessageType {
  ANNOTATION_CREATED = 'annotation_created',
  ANNOTATION_UPDATED = 'annotation_updated',
  FEEDBACK_SUBMITTED = 'feedback_submitted',
  PERSONA_UPDATED = 'persona_updated',
  USER_JOINED = 'user_joined',
  USER_LEFT = 'user_left',
}

// Define message interface
export interface WebSocketMessage {
  type: MessageType;
  payload: any;
  sender?: string;
  timestamp?: number;
}

export class WebSocketService {
  private wss: WebSocketServer | null = null;
  private clients = new Map<string, any>();
  
  initialize(server: HTTPServer) {
    this.wss = new WebSocketServer({ noServer: true });
    
    // Handle WebSocket connections
    server.on('upgrade', (request, socket, head) => {
      const { pathname } = parse(request.url || '', true);
      
      if (pathname === '/api/ws') {
        this.wss!.handleUpgrade(request, socket, head, (ws) => {
          this.wss!.emit('connection', ws, request);
        });
      }
    });
    
    // Set up connection handler
    this.wss.on('connection', (ws, request) => {
      // Generate client ID
      const clientId = Math.random().toString(36).substring(2, 15);
      
      // Store client
      this.clients.set(clientId, {
        ws,
        joinTime: Date.now(),
      });
      
      // Send welcome message
      this.sendToClient(clientId, {
        type: MessageType.USER_JOINED,
        payload: {
          id: clientId,
          message: 'Connected to annotation platform',
        },
        timestamp: Date.now(),
      });
      
      // Announce to other clients
      this.broadcast({
        type: MessageType.USER_JOINED,
        payload: {
          id: clientId,
        },
        timestamp: Date.now(),
      }, clientId);
      
      // Handle incoming messages
      ws.on('message', (message) => {
        try {
          const parsedMessage = JSON.parse(message.toString()) as WebSocketMessage;
          
          // Add sender and timestamp if not present
          parsedMessage.sender = parsedMessage.sender || clientId;
          parsedMessage.timestamp = parsedMessage.timestamp || Date.now();
          
          // Broadcast message to other clients
          this.broadcast(parsedMessage, clientId);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      });
      
      // Handle disconnection
      ws.on('close', () => {
        // Remove client
        this.clients.delete(clientId);
        
        // Announce to other clients
        this.broadcast({
          type: MessageType.USER_LEFT,
          payload: {
            id: clientId,
          },
          timestamp: Date.now(),
        });
      });
    });
  }
  
  /**
   * Send a message to all connected clients
   */
  broadcast(message: WebSocketMessage, excludeClientId?: string) {
    this.clients.forEach((client, clientId) => {
      if (excludeClientId && clientId === excludeClientId) {
        return;
      }
      
      this.sendToClient(clientId, message);
    });
  }
  
  /**
   * Send a message to a specific client
   */
  sendToClient(clientId: string, message: WebSocketMessage) {
    const client = this.clients.get(clientId);
    
    if (client && client.ws.readyState === 1) {
      client.ws.send(JSON.stringify(message));
    }
  }
  
  /**
   * Get number of connected clients
   */
  getConnectedCount() {
    return this.clients.size;
  }
}

// Create a singleton instance
export const webSocketService = new WebSocketService();
```

Next, update our server initialization to include WebSockets:

```typescript
// src/lib/websocket/init.ts
import { Server as HTTPServer } from 'http';
import { webSocketService } from './index';

export const initializeWebSockets = (server: HTTPServer) => {
  webSocketService.initialize(server);
  
  console.log('WebSocket server initialized');
};
```

Modify Next.js server start in a custom server file:

```javascript
// server.js
const { createServer } = require('http');
const { parse } = require('url');
const next = require('next');
const { initializeWebSockets } = require('./dist/lib/websocket/init');

const dev = process.env.NODE_ENV !== 'production';
const app = next({ dev });
const handle = app.getRequestHandler();

app.prepare().then(() => {
  const server = createServer((req, res) => {
    const parsedUrl = parse(req.url, true);
    handle(req, res, parsedUrl);
  });
  
  // Initialize WebSockets
  initializeWebSockets(server);
  
  server.listen(3000, (err) => {
    if (err) throw err;
    console.log('> Ready on http://localhost:3000');
  });
});
```

Update package.json to use the custom server:

```json
{
  "scripts": {
    "dev": "node server.js",
    "build": "next build && tsc --project tsconfig.server.json",
    "start": "NODE_ENV=production node server.js"
  }
}
```

Create a tsconfig.server.json for the server code:

```json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "module": "commonjs",
    "outDir": "dist",
    "noEmit": false
  },
  "include": ["src/lib/websocket"]
}
```

#### 7.1.2 User Authentication System

Let's create a basic user authentication system:

```typescript
// src/lib/auth/types.ts
export interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'annotator' | 'reviewer';
  createdAt: Date;
}

export interface AuthRequest {
  email: string;
  password: string;
}

export interface AuthResponse {
  user: User;
  token: string;
}
```

Create a simple auth service:

```typescript
// src/lib/auth/authService.ts
import { prisma } from '../db/prisma';
import { User, AuthRequest, AuthResponse } from './types';
import crypto from 'crypto';
import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'local-development-secret';

export class AuthService {
  async login(request: AuthRequest): Promise<AuthResponse> {
    // In a production environment, use a proper authentication system
    // This is a simplified version for local development
    
    const user = await prisma.user.findFirst({
      where: {
        email: request.email,
      },
    });
    
    if (!user) {
      throw new Error('User not found');
    }
    
    // In a real system, verify the password with a proper hashing algorithm
    // Here we're just using a placeholder
    const passwordValid = this.verifyPassword(request.password, user.passwordHash);
    
    if (!passwordValid) {
      throw new Error('Invalid password');
    }
    
    // Generate JWT token
    const token = jwt.sign(
      { 
        userId: user.id,
        email: user.email,
        role: user.role,
      },
      JWT_SECRET,
      { expiresIn: '1d' }
    );
    
    return {
      user: {
        id: user.id,
        name: user.name,
        email: user.email,
        role: user.role as 'admin' | 'annotator' | 'reviewer',
        createdAt: user.createdAt,
      },
      token,
    };
  }
  
  async register(user: Omit<User, 'id' | 'createdAt'> & { password: string }): Promise<User> {
    // Check if user already exists
    const existingUser = await prisma.user.findFirst({
      where: {
        email: user.email,
      },
    });
    
    if (existingUser) {
      throw new Error('User already exists');
    }
    
    // Hash password
    const passwordHash = this.hashPassword(user.password);
    
    // Create user
    const newUser = await prisma.user.create({
      data: {
        name: user.name,
        email: user.email,
        role: user.role,
        passwordHash,
      },
    });
    
    return {
      id: newUser.id,
      name: newUser.name,
      email: newUser.email,
      role: newUser.role as 'admin' | 'annotator' | 'reviewer',
      createdAt: newUser.createdAt,
    };
  }
  
  async validateToken(token: string): Promise<User | null> {
    try {
      const decoded = jwt.verify(token, JWT_SECRET) as { userId: string };
      
      const user = await prisma.user.findUnique({
        where: {
          id: decoded.userId,
        },
      });
      
      if (!user) {
        return null;
      }
      
      return {
        id: user.id,
        name: user.name,
        email: user.email,
        role: user.role as 'admin' | 'annotator' | 'reviewer',
        createdAt: user.createdAt,
      };
    } catch (error) {
      return null;
    }
  }
  
  private hashPassword(password: string): string {
    // In a production environment, use a proper password hashing library
    // This is a simplified version for local development
    return crypto.createHash('sha256').update(password).digest('hex');
  }
  
  private verifyPassword(password: string, hash: string): boolean {
    const passwordHash = this.hashPassword(password);
    return passwordHash === hash;
  }
}

export const authService = new AuthService();
```

Create authentication middleware:

```typescript
// src/middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { authService } from './lib/auth/authService';

export async function middleware(request: NextRequest) {
  // Skip authentication for public routes
  const publicPaths = ['/api/auth/login', '/api/auth/register', '/login', '/register'];
  const path = request.nextUrl.pathname;
  
  if (publicPaths.includes(path)) {
    return NextResponse.next();
  }
  
  // Check if it's an API route
  if (path.startsWith('/api/')) {
    // Get token from Authorization header
    const token = request.headers.get('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }
    
    // Validate token
    const user = await authService.validateToken(token);
    
    if (!user) {
      return NextResponse.json(
        { error: 'Invalid or expired token' },
        { status: 401 }
      );
    }
    
    // Add user to request
    const requestHeaders = new Headers(request.headers);
    requestHeaders.set('x-user-id', user.id);
    requestHeaders.set('x-user-role', user.role);
    
    return NextResponse.next({
      request: {
        headers: requestHeaders,
      },
    });
  }
  
  // For non-API routes, check for token in cookie
  const token = request.cookies.get('token')?.value;
  
  if (!token) {
    return NextResponse.redirect(new URL('/login', request.url));
  }
  
  // Validate token
  const user = await authService.validateToken(token);
  
  if (!user) {
    // Clear invalid token
    const response = NextResponse.redirect(new URL('/login', request.url));
    response.cookies.delete('token');
    return response;
  }
  
  return NextResponse.next();
}

export const config = {
  matcher: [
    '/api/:path*',
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
};
```

#### 7.1.3 Collaborative Annotation Interface

Now, let's create a WebSocket hook for the frontend:

```typescript
// src/hooks/useWebSocket.ts
'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { MessageType, WebSocketMessage } from '@/lib/websocket';

interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const {
    onMessage,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
  } = options;
  
  const connect = useCallback(() => {
    // Use a relative URL so it works in any environment
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/ws`;
    
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      setIsConnected(true);
      reconnectAttemptsRef.current = 0;
      
      if (reconnectIntervalRef.current) {
        clearInterval(reconnectIntervalRef.current);
        reconnectIntervalRef.current = null;
      }
    };
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WebSocketMessage;
        setLastMessage(message);
        
        if (onMessage) {
          onMessage(message);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    ws.onclose = () => {
      setIsConnected(false);
      
      // Attempt to reconnect
      if (reconnectAttemptsRef.current < maxReconnectAttempts) {
        reconnectAttemptsRef.current++;
        
        reconnectIntervalRef.current = setTimeout(() => {
          console.log(`Attempting to reconnect (${reconnectAttemptsRef.current}/${maxReconnectAttempts})...`);
          connect();
        }, reconnectInterval);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      ws.close();
    };
    
    wsRef.current = ws;
  }, [onMessage, reconnectInterval, maxReconnectAttempts]);
  
  const sendMessage = useCallback((message: Omit<WebSocketMessage, 'timestamp'>) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        ...message,
        timestamp: Date.now(),
      }));
      return true;
    }
    return false;
  }, []);
  
  // Connect when component mounts
  useEffect(() => {
    connect();
    
    // Cleanup
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      
      if (reconnectIntervalRef.current) {
        clearTimeout(reconnectIntervalRef.current);
      }
    };
  }, [connect]);
  
  return {
    isConnected,
    sendMessage,
    lastMessage,
  };
}
```

Now, let's create a collaborative annotation page:

```tsx
// src/app/projects/[projectId]/collaborative/page.tsx
'use client';

import { useState, useEffect, useCallback } from 'react';
import { useParams } from 'next/navigation';
import { useWebSocket } from '@/hooks/useWebSocket';
import { MessageType } from '@/lib/websocket';
import FeedbackForm from '@/components/FeedbackForm';

interface Annotation {
  id: string;
  annotation: string;
  personaId: string;
  personaName?: string;
  confidence: number;
  createdAt: string;
}

interface CollaboratorActivity {
  userId: string;
  name: string;
  action: string;
  timestamp: number;
}

export default function CollaborativeAnnotationPage() {
  const params = useParams();
  const projectId = params.projectId as string;
  
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [content, setContent] = useState('');
  const [itemId, setItemId] = useState<string | null>(null);
  const [collaborators, setCollaborators] = useState<Record<string, { name: string; isActive: boolean }>>({});
  const [activities, setActivities] = useState<CollaboratorActivity[]>([]);
  
  // Setup WebSocket
  const { isConnected, sendMessage } = useWebSocket({
    onMessage: (message) => {
      switch (message.type) {
        case MessageType.ANNOTATION_CREATED:
          // Add new annotation to the list
          setAnnotations(prev => [message.payload, ...prev]);
          
          // Add activity
          addActivity(message.sender!, 'created an annotation', message.timestamp!);
          break;
          
        case MessageType.FEEDBACK_SUBMITTED:
          // Add activity
          addActivity(message.sender!, 'provided feedback', message.timestamp!);
          break;
          
        case MessageType.USER_JOINED:
          // Add user to collaborators
          setCollaborators(prev => ({
            ...prev,
            [message.payload.id]: {
              name: message.payload.name || `User ${message.payload.id.substring(0, 5)}`,
              isActive: true,
            },
          }));
          
          // Add activity
          addActivity(message.payload.id, 'joined', message.timestamp!);
          break;
          
        case MessageType.USER_LEFT:
          // Mark user as inactive
          setCollaborators(prev => ({
            ...prev,
            [message.payload.id]: {
              ...prev[message.payload.id],
              isActive: false,
            },
          }));
          
          // Add activity
          addActivity(message.payload.id, 'left', message.timestamp!);
          break;
      }
    },
  });
  
  // Helper to add activity
  const addActivity = useCallback((userId: string, action: string, timestamp: number) => {
    const name = collaborators[userId]?.name || `User ${userId.substring(0, 5)}`;
    
    setActivities(prev => [
      { userId, name, action, timestamp },
      ...prev.slice(0, 19), // Keep only the last 20 activities
    ]);
  }, [collaborators]);
  
  // Fetch initial annotations
  useEffect(() => {
    if (!itemId) return;
    
    const fetchAnnotations = async () => {
      try {
        const response = await fetch(`/api/items/${itemId}/annotations`);
        
        if (response.ok) {
          const data = await response.json();
          setAnnotations(data);
        }
      } catch (error) {
        console.error('Error fetching annotations:', error);
      }
    };
    
    fetchAnnotations();
  }, [itemId]);
  
  // Generate annotation
  const generateAnnotation = async (personaId: string) => {
    if (!content) return;
    
    try {
      const response = await fetch('/api/annotations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          personaId,
          content,
          itemId,
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        
        // Add to local state
        setAnnotations(prev => [data, ...prev]);
        
        // Notify collaborators
        sendMessage({
          type: MessageType.ANNOTATION_CREATED,
          payload: data,
        });
      }
    } catch (error) {
      console.error('Error generating annotation:', error);
    }
  };
  
  // Format timestamp
  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString();
  };
  
  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Collaborative Annotation</h1>
        <div className={`px-3 py-1 rounded-full text-sm ${isConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Main annotation area */}
        <div className="md:col-span-2 space-y-6">
          {/* Content to annotate */}
          <div className="border rounded p-4">
            <h2 className="text-lg font-bold mb-3">Content</h2>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              className="w-full h-40 p-3 border rounded"
              placeholder="Enter or paste content to annotate..."
            />
            
            <div className="mt-3 flex gap-2">
              <button
                onClick={() => generateAnnotation('persona1')} // Use a real persona ID
                className="px-3 py-1 bg-blue-600 text-white rounded"
              >
                Annotate as Persona 1
              </button>
              <button
                onClick={() => generateAnnotation('persona2')} // Use a real persona ID
                className="px-3 py-1 bg-green-600 text-white rounded"
              >
                Annotate as Persona 2
              </button>
            </div>
          </div>
          
          {/* Annotations */}
          <div className="border rounded p-4">
            <h2 className="text-lg font-bold mb-3">Annotations</h2>
            
            {annotations.length === 0 ? (
              <p className="text-gray-500">No annotations yet</p>
            ) : (
              <div className="space-y-4">
                {annotations.map((annotation) => (
                  <div key={annotation.id} className="border rounded p-3">
                    <div className="flex justify-between text-sm text-gray-600 mb-2">
                      <span>Persona: {annotation.personaName || annotation.personaId}</span>
                      <span>Confidence: {(annotation.confidence * 100).toFixed(1)}%</span>
                    </div>
                    
                    <div className="p-3 bg-gray-50 rounded">
                      <p>{annotation.annotation}</p>
                    </div>
                    
                    <div className="mt-3">
                      <FeedbackForm
                        annotationId={annotation.id}
                        userId="current-user" // In a real app, use actual user ID
                        onSubmitSuccess={() => {
                          // Notify collaborators about feedback
                          sendMessage({
                            type: MessageType.FEEDBACK_SUBMITTED,
                            payload: {
                              annotationId: annotation.id,
                            },
                          });
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
        
        {/* Collaboration sidebar */}
        <div className="space-y-6">
          {/* Active collaborators */}
          <div className="border rounded p-4">
            <h2 className="text-lg font-bold mb-3">Collaborators</h2>
            
            {Object.keys(collaborators).length === 0 ? (
              <p className="text-gray-500">No one else is here</p>
            ) : (
              <ul className="space-y-2">
                {Object.entries(collaborators).map(([id, { name, isActive }]) => (
                  <li key={id} className="flex items-center">
                    <span className={`w-2 h-2 rounded-full mr-2 ${isActive ? 'bg-green-500' : 'bg-gray-300'}`}></span>
                    <span className={isActive ? '' : 'text-gray-400'}>{name}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
          
          {/* Activity feed */}
          <div className="border rounded p-4">
            <h2 className="text-lg font-bold mb-3">Activity</h2>
            
            {activities.length === 0 ? (
              <p className="text-gray-500">No activity yet</p>
            ) : (
              <ul className="space-y-2">
                {activities.map((activity, index) => (
                  <li key={index} className="text-sm">
                    <span className="text-gray-400">{formatTime(activity.timestamp)}</span>
                    <span className="mx-1">-</span>
                    <span className="font-medium">{activity.name}</span>
                    <span className="ml-1">{activity.action}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
```

### 7.2 Enhancing the Annotation Pipeline with Additional AI Models

As the system evolves, you might want to incorporate additional AI models for specialized tasks. Here's how we can extend the system:

#### 7.2.1 Model Registry and Factory

First, let's create a model registry to manage different AI models:

```typescript
// src/lib/models/registry.ts
import { ollamaService } from '../ollama';

export interface ModelProvider {
  id: string;
  name: string;
  type: 'text' | 'image' | 'audio' | 'multimodal';
  available: boolean;
}

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  description: string;
  capabilities: string[];
  contextLength: number;
  recommended: boolean;
}

export class ModelRegistry {
  private providers: Record<string, ModelProvider> = {};
  private models: Record<string, ModelInfo> = {};
  
  constructor() {
    // Register default providers
    this.registerProvider({
      id: 'ollama',
      name: 'Ollama',
      type: 'text',
      available: true,
    });
    
    // Register default models
    this.registerModel({
      id: 'ollama/llama2',
      name: 'Llama 2',
      provider: 'ollama',
      description: 'A powerful open-source LLM by Meta',
      capabilities: ['text-generation', 'summarization', 'classification'],
      contextLength: 4096,
      recommended: true,
    });
    
    this.registerModel({
      id: 'ollama/mistral',
      name: 'Mistral 7B',
      provider: 'ollama',
      description: 'High-performance small language model',
      capabilities: ['text-generation', 'question-answering'],
      contextLength: 8192,
      recommended: true,
    });
  }
  
  registerProvider(provider: ModelProvider): void {
    this.providers[provider.id] = provider;
  }
  
  registerModel(model: ModelInfo): void {
    this.models[model.id] = model;
  }
  
  async getAvailableModels(): Promise<ModelInfo[]> {
    // Fetch available Ollama models
    const ollamaModels = await this.getOllamaModels();
    
    // Filter registered models to only those available
    return Object.values(this.models).filter(model => {
      if (model.provider === 'ollama') {
        return ollamaModels.includes(model.name.toLowerCase());
      }
      return true;
    });
  }
  
  getModelById(id: string): ModelInfo | null {
    return this.models[id] || null;
  }
  
  getProviderById(id: string): ModelProvider | null {
    return this.providers[id] || null;
  }
  
  private async getOllamaModels(): Promise<string[]> {
    try {
      const models = await ollamaService.getModels();
      return models;
    } catch (error) {
      console.error('Error fetching Ollama models:', error);
      return [];
    }
  }
}

export const modelRegistry = new ModelRegistry();
```

Now, let's create a model factory to generate model instances:

```typescript
// src/lib/models/factory.ts
import { modelRegistry } from './registry';
import { ollamaService } from '../ollama';

export interface ModelOptions {
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
}

export interface ModelResponse {
  text: string;
  model: string;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export abstract class AIModel {
  protected options: ModelOptions;
  
  constructor(options: ModelOptions = {}) {
    this.options = {
      temperature: 0.7,
      maxTokens: 1024,
      topP: 1,
      frequencyPenalty: 0,
      presencePenalty: 0,
      ...options,
    };
  }
  
  abstract generate(prompt: string, systemPrompt?: string): Promise<ModelResponse>;
}

export class OllamaModel extends AIModel {
  private modelName: string;
  
  constructor(modelName: string, options: ModelOptions = {}) {
    super(options);
    this.modelName = modelName;
  }
  
  async generate(prompt: string, systemPrompt?: string): Promise<ModelResponse> {
    // Set the model in the Ollama service
    ollamaService.setModel(this.modelName);
    
    // Generate text
    const response = await ollamaService.generate({
      prompt,
      system: systemPrompt,
      temperature: this.options.temperature,
      maxTokens: this.options.maxTokens,
    });
    
    return {
      text: response.text,
      model: response.model,
      promptTokens: response.promptTokens,
      completionTokens: response.generatedTokens,
      totalTokens: response.promptTokens + response.generatedTokens,
    };
  }
}

export class ModelFactory {
  static createModel(modelId: string, options: ModelOptions = {}): AIModel | null {
    const model = modelRegistry.getModelById(modelId);
    
    if (!model) {
      return null;
    }
    
    if (model.provider === 'ollama') {
      // Extract actual model name (after the provider prefix)
      const modelName = model.id.split('/')[1];
      return new OllamaModel(modelName, options);
    }
    
    // Add other model providers as needed
    
    return null;
  }
}
```

Finally, let's update our annotation service to use the model factory:

```typescript
// src/lib/services/annotationService.ts
import { ModelFactory } from '../models/factory';
// ... other imports

export class AnnotationService {
  async generateAnnotation(request: AnnotationRequest): Promise<AnnotationResult> {
    // ... existing code
    
    // Get the persona
    const persona = await personaService.getPersona(request.personaId);
    
    if (!persona) {
      throw new Error(`Persona ${request.personaId} not found`);
    }
    
    // Get the model information from the persona
    const modelId = persona.modelId || 'ollama/llama2'; // Default model
    
    // Create the model instance
    const model = ModelFactory.createModel(modelId, {
      temperature: 0.3, // Lower temperature for more focused annotations
    });
    
    if (!model) {
      throw new Error(`Model ${modelId} not found or not available`);
    }
    
    // Prepare the prompt for annotation
    const prompt = `Please analyze the following content and provide an annotation:

${request.content}`;

    // Generate annotation using the model
    const modelResponse = await model.generate(prompt, persona.prompt);
    
    // Calculate a simple confidence score
    const confidence = this.calculateConfidence(modelResponse.text);
    
    // ... rest of the method
  }
  
  // ... rest of the class
}
```

#### 7.2.2 Adding Image Annotation Capabilities

Let's extend our system to support image annotations:

```typescript
// src/types/annotation.ts
export interface AnnotationRequest {
  itemId: string;
  personaId: string;
  content: string;
  mediaType?: 'text' | 'image' | 'audio';
  mediaUrl?: string;
  metadata?: Record<string, any>;
}

// ... other types
```

Now, let's create a specialized image annotation service:

```typescript
// src/lib/services/imageAnnotationService.ts
import { prisma } from '../db/prisma';
import { personaService } from './personaService';
import { ModelFactory } from '../models/factory';
import { AnnotationRequest, AnnotationResult } from '@/types/annotation';

export class ImageAnnotationService {
  async generateImageAnnotation(request: AnnotationRequest): Promise<AnnotationResult> {
    if (!request.mediaUrl) {
      throw new Error('Media URL is required for image annotation');
    }
    
    // Get the persona
    const persona = await personaService.getPersona(request.personaId);
    
    if (!persona) {
      throw new Error(`Persona ${request.personaId} not found`);
    }
    
    // For this example, we'll use a text model and include the image URL
    // In a more advanced implementation, you would use a multimodal model
    // or integrate with computer vision APIs
    
    const modelId = persona.modelId || 'ollama/llama2';
    const model = ModelFactory.createModel(modelId, {
      temperature: 0.3,
    });
    
    if (!model) {
      throw new Error(`Model ${modelId} not found or not available`);
    }
    
    // Prepare the prompt for image annotation
    const prompt = `Please analyze the image at URL: ${request.mediaUrl}
    
${request.content ? `Additional context: ${request.content}` : ''}

Provide a detailed annotation of what you see in the image.`;

    // Generate annotation
    const modelResponse = await model.generate(prompt, persona.prompt);
    
    // Calculate confidence
    const confidence = 0.7; // Placeholder value
    
    // Save annotation to database if we have an item
    let annotation;
    if (request.itemId) {
      annotation = await prisma.annotation.create({
        data: {
          itemId: request.itemId,
          personaId: request.personaId,
          annotation: modelResponse.text,
          confidence,
          metadata: JSON.stringify({
            mediaType: 'image',
            mediaUrl: request.mediaUrl,
          }),
        },
      });
    } else {
      // Create an ephemeral annotation result
      annotation = {
        id: 'temp-' + Date.now(),
        itemId: 'temp-item',
        personaId: request.personaId,
        annotation: modelResponse.text,
        confidence,
        createdAt: new Date(),
      };
    }
    
    return annotation;
  }
}

export const imageAnnotationService = new ImageAnnotationService();
```

Create an API route for image annotation:

```typescript
// src/app/api/annotations/image/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { imageAnnotationService } from '@/lib/services/imageAnnotationService';

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();
    
    // Validate request
    if (!data.personaId || !data.mediaUrl) {
      return NextResponse.json(
        { error: 'personaId and mediaUrl are required' },
        { status: 400 }
      );
    }
    
    // Ensure mediaType is image
    data.mediaType = 'image';
    
    // Generate image annotation
    const annotation = await imageAnnotationService.generateImageAnnotation(data);
    
    return NextResponse.json(annotation, { status: 201 });
  } catch (error) {
    console.error('Error generating image annotation:', error);
    return NextResponse.json(
      { error: 'Failed to generate image annotation' },
      { status: 500 }
    );
  }
}
```

### 7.3 Scaling Strategies for Local to Production

While our system is designed to run locally, you might eventually want to scale it for larger deployments. Here are some scaling strategies:

#### 7.3.1 Docker Containerization

Create a `Dockerfile` for containerization:

```dockerfile
# Dockerfile
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
WORKDIR /app

# Install Python and required packages
RUN apk add --no-cache python3 py3-pip
RUN pip3 install chromadb sentence-transformers

# Copy package.json and install dependencies
COPY package.json package-lock.json* ./
RUN npm ci

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Next.js collects anonymous telemetry data about usage
ENV NEXT_TELEMETRY_DISABLED 1

RUN npm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

# Install Python and required packages in the runner
RUN apk add --no-cache python3 py3-pip
RUN pip3 install chromadb sentence-transformers

# Create necessary directories
RUN mkdir -p /app/data /app/chroma_db /app/.cache

# Copy built files
COPY --from=builder /app/next.config.js ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/prisma ./prisma
COPY --from=builder /app/scripts ./scripts

# Copy custom server
COPY --from=builder /app/server.js ./
COPY --from=builder /app/dist ./dist

# Make data directories accessible
RUN chmod -R 777 /app/data /app/chroma_db /app/.cache

EXPOSE 3000

ENV PORT 3000

CMD ["node", "server.js"]
```

Create a `docker-compose.yml` file to run the application and its dependencies:

```yaml
version: '3'

services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_TYPE=postgres
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_USER=annotation_user
      - POSTGRES_PASSWORD=annotation_password
      - POSTGRES_DB=annotation_platform
      - DATABASE_URL=postgresql://annotation_user:annotation_password@postgres:5432/annotation_platform?schema=public
      - OLLAMA_URL=http://ollama:11434
      - OLLAMA_DEFAULT_MODEL=llama2
      - NEXT_PUBLIC_OLLAMA_BASE_URL=http://localhost:11434
      - NEXT_PUBLIC_OLLAMA_DEFAULT_MODEL=llama2
      - CHROMADB_DIR=/app/chroma_db
      - PYTHON_PATH=python3
    volumes:
      - ./data:/app/data
      - ./chroma_db:/app/chroma_db
      - ./.cache:/app/.cache
    depends_on:
      - postgres
      - ollama

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_USER=annotation_user
      - POSTGRES_PASSWORD=annotation_password
      - POSTGRES_DB=annotation_platform
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_models:/root/.ollama
    ports:
      - "11434:11434"

volumes:
  postgres_data:
  ollama_models:
```

#### 7.3.2 Distributed Annotation Service

Create a job queue for distributed annotation tasks:

```typescript
// src/lib/queue/annotationQueue.ts
import { Queue, Worker, QueueEvents } from 'bullmq';
import { AnnotationRequest, AnnotationResult } from '@/types/annotation';
import { annotationService } from '../services/annotationService';
import { imageAnnotationService } from '../services/imageAnnotationService';
import { prisma } from '../db/prisma';
import { deploymentConfig } from '../config/deployment';

interface AnnotationJob {
  request: AnnotationRequest;
  callbackUrl?: string;
}

// Redis connection config (for distributed mode)
const connection = deploymentConfig.redis ? {
  host: deploymentConfig.redis.host,
  port: deploymentConfig.redis.port,
  password: deploymentConfig.redis.password,
} : undefined;

// Create job queue
const annotationQueue = new Queue<AnnotationJob, AnnotationResult>('annotation', {
  connection,
  defaultJobOptions: {
    attempts: 3,
    backoff: {
      type: 'exponential',
      delay: 1000,
    },
  },
});

// Create queue events listener
const queueEvents = new QueueEvents('annotation', { connection });

// Create worker
const worker = new Worker<AnnotationJob, AnnotationResult>(
  'annotation',
  async (job) => {
    const { request } = job.data;
    
    // Process annotation based on media type
    if (request.mediaType === 'image' && request.mediaUrl) {
      return await imageAnnotationService.generateImageAnnotation(request);
    } else {
      return await annotationService.generateAnnotation(request);
    }
  },
  { connection, concurrency: deploymentConfig.system.maxConcurrency }
);

// Handle completed jobs
worker.on('completed', async (job, result) => {
  console.log(`Job ${job.id} completed`, result.id);
  
  // Notify via callback URL if provided
  if (job.data.callbackUrl) {
    try {
      await fetch(job.data.callbackUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          jobId: job.id,
          annotationId: result.id,
          status: 'completed',
        }),
      });
    } catch (error) {
      console.error(`Error calling callback URL ${job.data.callbackUrl}:`, error);
    }
  }
});

// Handle failed jobs
worker.on('failed', async (job, error) => {
  console.error(`Job ${job?.id} failed:`, error);
  
  // Store failure information
  if (job) {
    await prisma.annotationJob.update({
      where: { id: job.id as string },
      data: {
        status: 'failed',
        error: error.message,
      },
    });
    
    // Notify via callback URL if provided
    if (job.data.callbackUrl) {
      try {
        await fetch(job.data.callbackUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            jobId: job.id,
            status: 'failed',
            error: error.message,
          }),
        });
      } catch (callbackError) {
        console.error(`Error calling callback URL ${job.data.callbackUrl}:`, callbackError);
      }
    }
  }
});

export { annotationQueue };
```

Create an API for submitting annotation jobs:

```typescript
// src/app/api/annotations/job/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { annotationQueue } from '@/lib/queue/annotationQueue';
import { prisma } from '@/lib/db/prisma';

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();
    
    // Validate request
    if (!data.personaId || !data.content) {
      return NextResponse.json(
        { error: 'personaId and content are required' },
        { status: 400 }
      );
    }
    
    // Create job record in database
    const job = await prisma.annotationJob.create({
      data: {
        status: 'pending',
        request: JSON.stringify(data),
      },
    });
    
    // Add job to queue
    const queuedJob = await annotationQueue.add(
      'annotation',
      {
        request: data,
        callbackUrl: data.callbackUrl,
      },
      {
        jobId: job.id,
      }
    );
    
    return NextResponse.json(
      {
        jobId: job.id,
        status: 'pending',
      },
      { status: 202 }
    );
  } catch (error) {
    console.error('Error creating annotation job:', error);
    return NextResponse.json(
      { error: 'Failed to create annotation job' },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const jobId = searchParams.get('jobId');
    
    if (!jobId) {
      return NextResponse.json(
        { error: 'jobId parameter is required' },
        { status: 400 }
      );
    }
    
    // Get job status from database
    const job = await prisma.annotationJob.findUnique({
      where: { id: jobId },
    });
    
    if (!job) {
      return NextResponse.json(
        { error: 'Job not found' },
        { status: 404 }
      );
    }
    
    // Get annotation if job is completed
    let annotation = null;
    if (job.status === 'completed' && job.annotationId) {
      annotation = await prisma.annotation.findUnique({
        where: { id: job.annotationId },
      });
    }
    
    return NextResponse.json({
      jobId: job.id,
      status: job.status,
      error: job.error,
      createdAt: job.createdAt,
      updatedAt: job.updatedAt,
      annotation,
    });
  } catch (error) {
    console.error('Error fetching annotation job:', error);
    return NextResponse.json(
      { error: 'Failed to fetch annotation job' },
      { status: 500 }
    );
  }
}
```

---

## 8. Troubleshooting and Performance Optimization

### 8.1 Common Issues and Solutions

Let's address some common issues that users might encounter:

#### 8.1.1 Troubleshooting Guide

Create a comprehensive troubleshooting guide:

```tsx
// src/app/help/troubleshooting/page.tsx
export default function TroubleshootingPage() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Troubleshooting Guide</h1>
      
      <div className="space-y-8">
        <section>
          <h2 className="text-2xl font-bold mb-4">Ollama Issues</h2>
          
          <div className="space-y-4">
            <div className="border rounded p-4">
              <h3 className="text-lg font-bold mb-2">Cannot connect to Ollama</h3>
              <p className="mb-3">If you see error messages about failing to connect to Ollama, try the following:</p>
              
              <ol className="list-decimal pl-5 space-y-2">
                <li>Ensure Ollama is installed and running on your machine.</li>
                <li>Check if Ollama is accessible at <code>http://localhost:11434</code>.</li>
                <li>Verify that the URL in your <code>.env.local</code> file matches your Ollama installation.</li>
                <li>Restart both Ollama and the annotation platform.</li>
              </ol>
              
              <div className="mt-3 p-3 bg-gray-100 rounded">
                <p className="font-bold">Command to check Ollama:</p>
                <pre className="whitespace-pre-wrap overflow-x-auto">
                  curl http://localhost:11434/api/tags
                </pre>
              </div>
            </div>
            
            <div className="border rounded p-4">
              <h3 className="text-lg font-bold mb-2">Model not found</h3>
              <p className="mb-3">If you get "model not found" errors, follow these steps:</p>
              
              <ol className="list-decimal pl-5 space-y-2">
                <li>Pull the required model with Ollama CLI: <code>ollama pull llama2</code> (replace llama2 with your model).</li>
                <li>Verify available models: <code>ollama list</code></li>
                <li>Ensure the model name in your configuration matches exactly (case-sensitive).</li>
              </ol>
            </div>
            
            <div className="border rounded p-4">
              <h3 className="text-lg font-bold mb-2">Slow Model Responses</h3>
              <p className="mb-3">If model responses are very slow, consider:</p>
              
              <ul className="list-disc pl-5 space-y-2">
                <li>Using a smaller model (e.g., Mistral 7B instead of larger models).</li>
                <li>Reducing the maximum token length in your requests.</li>
                <li>Checking if your machine has sufficient resources (CPU/RAM/GPU).</li>
                <li>Closing other resource-intensive applications.</li>
              </ul>
            </div>
          </div>
        </section>
        
        <section>
          <h2 className="text-2xl font-bold mb-4">Database Issues</h2>
          
          <div className="space-y-4">
            <div className="border rounded p-4">
              <h3 className="text-lg font-bold mb-2">Failed Database Connection</h3>
              <p className="mb-3">If the application cannot connect to the database:</p>
              
              <ol className="list-decimal pl-5 space-y-2">
                <li>Verify that your database configuration in <code>.env.local</code> is correct.</li>
                <li>For SQLite, ensure the directory is writable.</li>
                <li>For PostgreSQL, ensure the server is running and accessible.</li>
                <li>Try regenerating the Prisma client: <code>npx prisma generate</code></li>
              </ol>
            </div>
            
            <div className="border rounded p-4">
              <h3 className="text-lg font-bold mb-2">Migration Issues</h3>
              <p className="mb-3">If database migrations fail:</p>
              
              <ol className="list-decimal pl-5 space-y-2">
                <li>Reset the database if it's in development: <code>npx prisma migrate reset</code></li>
                <li>Check for syntax errors in your schema.prisma file.</li>
                <li>Ensure you have proper permissions to create/alter tables.</li>
              </ol>
              
              <div className="mt-3 p-3 bg-gray-100 rounded">
                <p className="font-bold">Command to apply migrations:</p>
                <pre className="whitespace-pre-wrap overflow-x-auto">
                  npx prisma migrate dev --name update
                </pre>
              </div>
            </div>
          </div>
        </section>
        
        <section>
          <h2 className="text-2xl font-bold mb-4">ChromaDB Issues</h2>
          
          <div className="space-y-4">
            <div className="border rounded p-4">
              <h3 className="text-lg font-bold mb-2">ChromaDB Not Found</h3>
              <p className="mb-3">If the application cannot connect to ChromaDB:</p>
              
              <ol className="list-decimal pl-5 space-y-2">
                <li>Ensure Python and required packages are installed: <code>pip install chromadb sentence-transformers</code></li>
                <li>Verify that the ChromaDB directory exists and is writable.</li>
                <li>Check if the Python path in your configuration is correct.</li>
              </ol>
            </div>
            
            <div className="border rounded p-4">
              <h3 className="text-lg font-bold mb-2">Import Error or Module Not Found</h3>
              <p className="mb-3">If you see Python import errors:</p>
              
              <ol className="list-decimal pl-5 space-y-2">
                <li>Ensure you've installed all required packages.</li>
                <li>Try creating a dedicated virtual environment for Python.</li>
                <li>For Windows, ensure Python is in your PATH environment variable.</li>
              </ol>
              
              <div className="mt-3 p-3 bg-gray-100 rounded">
                <p className="font-bold">Command to check Python packages:</p>
                <pre className="whitespace-pre-wrap overflow-x-auto">
                  python -c "import sys; print(sys.executable); import chromadb; print('ChromaDB imported successfully')"
                </pre>
              </div>
            </div>
          </div>
        </section>
        
        <section>
          <h2 className="text-2xl font-bold mb-4">Performance Issues</h2>
          
          <div className="space-y-4">
            <div className="border rounded p-4">
              <h3 className="text-lg font-bold mb-2">High Memory Usage</h3>
              <p className="mb-3">If the application is using excessive memory:</p>
              
              <ul className="list-disc pl-5 space-y-2">
                <li>Reduce the number of concurrent requests in <code>.env.local</code> (MAX_CONCURRENCY=1).</li>
                <li>Use smaller AI models.</li>
                <li>Implement regular cache clearing.</li>
                <li>Close other memory-intensive applications.</li>
              </ul>
            </div>
            
            <div className="border rounded p-4">
              <h3 className="text-lg font-bold mb-2">Slow Application Startup</h3>
              <p className="mb-3">If the application takes a long time to start:</p>
              
              <ul className="list-disc pl-5 space-y-2">
                <li>Check the database size and consider optimizing or cleaning unused data.</li>
                <li>Ensure ChromaDB data directory isn't excessively large.</li>
                <li>Consider using the development build for faster startup during development.</li>
              </ul>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
```

### 8.2 Performance Optimization Techniques

Let's implement some performance optimization techniques:

#### 8.2.1 Optimized Database Queries

Create a service to optimize database operations:

```typescript
// src/lib/db/optimization.ts
import { prisma } from './prisma';

export class DatabaseOptimization {
  /**
   * Creates indices for frequently queried columns
   */
  async createIndices(): Promise<void> {
    // Execute raw SQL to create indices
    // Note: Prisma doesn't directly support index creation via its API
    
    // For SQLite
    if (process.env.DATABASE_TYPE !== 'postgres') {
      await prisma.$executeRaw`
        CREATE INDEX IF NOT EXISTS idx_annotations_persona_id ON annotations(persona_id);
        CREATE INDEX IF NOT EXISTS idx_annotations_item_id ON annotations(item_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_annotation_id ON feedback(annotation_id);
        CREATE INDEX IF NOT EXISTS idx_personas_project_id ON personas(project_id);
      `;
    } 
    // For PostgreSQL
    else {
      await prisma.$executeRaw`
        CREATE INDEX IF NOT EXISTS idx_annotations_persona_id ON annotations(persona_id);
        CREATE INDEX IF NOT EXISTS idx_annotations_item_id ON annotations(item_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_annotation_id ON feedback(annotation_id);
        CREATE INDEX IF NOT EXISTS idx_personas_project_id ON personas(project_id);
      `;
    }
  }
  
  /**
   * Vacuum/optimize the database
   */
  async optimizeDatabase(): Promise<void> {
    // For SQLite
    if (process.env.DATABASE_TYPE !== 'postgres') {
      await prisma.$executeRaw`VACUUM;`;
    } 
    // For PostgreSQL
    else {
      // Analyze tables for better query planning
      await prisma.$executeRaw`ANALYZE;`;
      // Vacuum to reclaim space
      await prisma.$executeRaw`VACUUM;`;
    }
  }
  
  /**
   * Clean up old data to improve performance
   */
  async cleanupOldData(daysToKeep = 30): Promise<{ deletedAnnotations: number; deletedFeedback: number }> {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - daysToKeep);
    
    // Delete old feedback first (due to foreign key constraints)
    const deletedFeedback = await prisma.feedback.deleteMany({
      where: {
        createdAt: {
          lt: cutoffDate,
        },
      },
    });
    
    // Delete old annotations
    const deletedAnnotations = await prisma.annotation.deleteMany({
      where: {
        createdAt: {
          lt: cutoffDate,
        },
        // Only delete annotations without recent feedback
        feedback: {
          none: {
            createdAt: {
              gte: cutoffDate,
            },
          },
        },
      },
    });
    
    return {
      deletedAnnotations: deletedAnnotations.count,
      deletedFeedback: deletedFeedback.count,
    };
  }
}

export const databaseOptimization = new DatabaseOptimization();
```

Create a maintenance API route:

```typescript
// src/app/api/system/maintenance/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { databaseOptimization } from '@/lib/db/optimization';

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();
    const operation = data.operation;
    
    let result;
    
    switch (operation) {
      case 'create-indices':
        await databaseOptimization.createIndices();
        result = { success: true, message: 'Indices created successfully' };
        break;
      
      case 'optimize-database':
        await databaseOptimization.optimizeDatabase();
        result = { success: true, message: 'Database optimized successfully' };
        break;
      
      case 'cleanup-old-data':
        const daysToKeep = data.daysToKeep || 30;
        const cleanup = await databaseOptimization.cleanupOldData(daysToKeep);
        result = { 
          success: true, 
          message: `Cleanup completed successfully`,
          deletedAnnotations: cleanup.deletedAnnotations,
          deletedFeedback: cleanup.deletedFeedback,
        };
        break;
      
      default:
        return NextResponse.json(
          { error: 'Invalid operation' },
          { status: 400 }
        );
    }
    
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error performing maintenance:', error);
    return NextResponse.json(
      { error: 'Maintenance operation failed' },
      { status: 500 }
    );
  }
}
```

#### 8.2.2 ChromaDB Performance Optimization

Create a service to optimize ChromaDB:

```typescript
// src/lib/chromadb/optimization.ts
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { deploymentConfig } from '../config/deployment';

export class ChromaDBOptimization {
  async compactDatabase(): Promise<{ success: boolean; message: string }> {
    const scriptPath = path.join(process.cwd(), 'scripts', 'chromadb', 'compact.py');
    
    // Create the script if it doesn't exist
    if (!fs.existsSync(scriptPath)) {
      this.createCompactScript(scriptPath);
    }
    
    try {
      const output = await this.runPythonScript(
        deploymentConfig.chromadb.pythonPath,
        [scriptPath, deploymentConfig.chromadb.directory]
      );
      
      return JSON.parse(output);
    } catch (error) {
      console.error('Error compacting ChromaDB:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
  
  async reindexDatabase(): Promise<{ success: boolean; message: string }> {
    const scriptPath = path.join(process.cwd(), 'scripts', 'chromadb', 'reindex.py');
    
    // Create the script if it doesn't exist
    if (!fs.existsSync(scriptPath)) {
      this.createReindexScript(scriptPath);
    }
    
    try {
      const output = await this.runPythonScript(
        deploymentConfig.chromadb.pythonPath,
        [scriptPath, deploymentConfig.chromadb.directory]
      );
      
      return JSON.parse(output);
    } catch (error) {
      console.error('Error reindexing ChromaDB:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
  
  private runPythonScript(pythonPath: string, args: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
      const process = spawn(pythonPath, args);
      
      let output = '';
      let errorOutput = '';
      
      process.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      process.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });
      
      process.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Python script error: ${errorOutput}`));
        } else {
          resolve(output.trim());
        }
      });
    });
  }
  
  private createCompactScript(scriptPath: string) {
    const scriptDir = path.dirname(scriptPath);
    
    if (!fs.existsSync(scriptDir)) {
      fs.mkdirSync(scriptDir, { recursive: true });
    }
    
    const scriptContent = `
import sys
import json
import os
import shutil
import tempfile

def compact_chromadb(chroma_dir):
    try:
        import chromadb
        
        # Check if ChromaDB directory exists
        if not os.path.exists(chroma_dir):
            return {
                "success": False,
                "message": f"ChromaDB directory {chroma_dir} does not exist"
            }
        
        # Create a temporary directory for the new database
        temp_dir = tempfile.mkdtemp()
        
        # Create new client for the temporary directory
        temp_client = chromadb.PersistentClient(path=temp_dir)
        
        # Create original client
        original_client = chromadb.PersistentClient(path=chroma_dir)
        
        # Get all collections
        collections = original_client.list_collections()
        
        # Copy each collection to the temporary client
        for collection_info in collections:
            collection_name = collection_info.name
            
            # Get original collection
            original_collection = original_client.get_collection(name=collection_name)
            
            # Create new collection
            new_collection = temp_client.create_collection(name=collection_name)
            
            # Get all items
            items = original_collection.get(include=["documents", "metadatas", "embeddings"])
            
            # Skip if no items
            if not items["ids"]:
                continue
                
            # Add items to new collection
            new_collection.add(
                ids=items["ids"],
                documents=items["documents"],
                metadatas=items["metadatas"],
                embeddings=items["embeddings"]
            )
        
        # Close clients
        del original_client
        del temp_client
        
        # Backup original directory
        backup_dir = f"{chroma_dir}_backup"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        
        shutil.move(chroma_dir, backup_dir)
        
        # Move temp directory to original location
        shutil.move(temp_dir, chroma_dir)
        
        return {
            "success": True,
            "message": f"ChromaDB compacted successfully. Original data backed up to {backup_dir}"
        }
    except ImportError:
        return {
            "success": False,
            "message": "ChromaDB Python package is not installed"
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "success": False,
            "message": "ChromaDB directory path not provided"
        }))
        sys.exit(1)
    
    chroma_dir = sys.argv[1]
    result = compact_chromadb(chroma_dir)
    print(json.dumps(result))
`;
    
    fs.writeFileSync(scriptPath, scriptContent);
  }
  
  private createReindexScript(scriptPath: string) {
    const scriptDir = path.dirname(scriptPath);
    
    if (!fs.existsSync(scriptDir)) {
      fs.mkdirSync(scriptDir, { recursive: true });
    }
    
    const scriptContent = `
import sys
import json
import os

def reindex_chromadb(chroma_dir):
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        
        # Check if ChromaDB directory exists
        if not os.path.exists(chroma_dir):
            return {
                "success": False,
                "message": f"ChromaDB directory {chroma_dir} does not exist"
            }
        
        # Initialize client
        client = chromadb.PersistentClient(path=chroma_dir)
        
        # Use sentence-transformers for embeddings
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get all collections
        collections = client.list_collections()
        
        reindexed_count = 0
        
        # Reindex each collection
        for collection_info in collections:
            collection_name = collection_info.name
            
            # Get collection
            collection = client.get_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef
            )
            
            # Get all items
            items = collection.get(include=["documents", "metadatas"])
            
            # Skip if no items
            if not items["ids"]:
                continue
                
            # Recompute embeddings and update
            collection.update(
                ids=items["ids"],
                documents=items["documents"],
                metadatas=items["metadatas"]
            )
            
            reindexed_count += len(items["ids"])
        
        return {
            "success": True,
            "message": f"ChromaDB reindexed successfully. {reindexed_count} items reindexed across {len(collections)} collections."
        }
    except ImportError:
        return {
            "success": False,
            "message": "ChromaDB Python package is not installed"
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "success": False,
            "message": "ChromaDB directory path not provided"
        }))
        sys.exit(1)
    
    chroma_dir = sys.argv[1]
    result = reindex_chromadb(chroma_dir)
    print(json.dumps(result))
`;
    
    fs.writeFileSync(scriptPath, scriptContent);
  }
}

export const chromaDBOptimization = new ChromaDBOptimization();
```

Create an API route for ChromaDB maintenance:

```typescript
// src/app/api/system/chromadb/maintenance/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { chromaDBOptimization } from '@/lib/chromadb/optimization';

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();
    const operation = data.operation;
    
    let result;
    
    switch (operation) {
      case 'compact':
        result = await chromaDBOptimization.compactDatabase();
        break;
      
      case 'reindex':
        result = await chromaDBOptimization.reindexDatabase();
        break;
      
      default:
        return NextResponse.json(
          { error: 'Invalid operation' },
          { status: 400 }
        );
    }
    
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error performing ChromaDB maintenance:', error);
    return NextResponse.json(
      { error: 'ChromaDB maintenance operation failed' },
      { status: 500 }
    );
  }
}
```

Add maintenance options to the system settings page:

```tsx
// src/app/settings/maintenance/page.tsx
'use client';

import { useState } from 'react';

export default function MaintenancePage() {
  const [isWorking, setIsWorking] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');
  
  const performMaintenance = async (type: string, operation: string, data = {}) => {
    setIsWorking(true);
    setResult(null);
    setError('');
    
    try {
      const response = await fetch(`/api/system/${type}/maintenance`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          operation,
          ...data,
        }),
      });
      
      const responseData = await response.json();
      
      if (response.ok) {
        setResult(responseData);
      } else {
        setError(responseData.error || 'Operation failed');
      }
    } catch (err) {
      setError('An unexpected error occurred');
      console.error(err);
    } finally {
      setIsWorking(false);
    }
  };
  
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">System Maintenance</h1>
      
      {error && (
        <div className="mb-6 p-4 bg-red-100 text-red-800 rounded">
          {error}
        </div>
      )}
      
      {result && (
        <div className="mb-6 p-4 bg-green-100 text-green-800 rounded">
          <p className="font-bold">{result.message || 'Operation completed successfully'}</p>
          {result.deletedAnnotations !== undefined && (
            <p>Deleted annotations: {result.deletedAnnotations}</p>
          )}
          {result.deletedFeedback !== undefined && (
            <p>Deleted feedback: {result.deletedFeedback}</p>
          )}
        </div>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Database Maintenance */}
        <div className="border rounded p-4">
          <h2 className="text-lg font-bold mb-3">Database Maintenance</h2>
          
          <div className="space-y-3">
            <button
              onClick={() => performMaintenance('system', 'create-indices')}
              disabled={isWorking}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded disabled:bg-blue-300"
            >
              Create/Update Indices
            </button>
            
            <button
              onClick={() => performMaintenance('system', 'optimize-database')}
              disabled={isWorking}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded disabled:bg-blue-300"
            >
              Optimize Database
            </button>
            
            <div>
              <button
                onClick={() => {
                  const days = prompt('How many days of data to keep?', '30');
                  if (days !== null) {
                    const daysNumber = parseInt(days, 10);
                    if (!isNaN(daysNumber) && daysNumber > 0) {
                      performMaintenance('system', 'cleanup-old-data', { daysToKeep: daysNumber });
                    } else {
                      alert('Please enter a valid number of days');
                    }
                  }
                }}
                disabled={isWorking}
                className="w-full px-4 py-2 bg-yellow-600 text-white rounded disabled:bg-yellow-300"
              >
                Clean Up Old Data
              </button>
              <p className="text-sm text-gray-500 mt-1">
                Removes old annotations and feedback to improve performance.
              </p>
            </div>
          </div>
        </div>
        
        {/* ChromaDB Maintenance */}
        <div className="border rounded p-4">
          <h2 className="text-lg font-bold mb-3">ChromaDB Maintenance</h2>
          
          <div className="space-y-3">
            <button
              onClick={() => {
                if (confirm('This will compact the ChromaDB database. A backup will be created before compacting. Continue?')) {
                  performMaintenance('chromadb', 'compact');
                }
              }}
              disabled={isWorking}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded disabled:bg-blue-300"
            >
              Compact ChromaDB
            </button>
            <p className="text-sm text-gray-500 mt-1">
              Reduces database size and improves query performance.
            </p>
            
            <button
              onClick={() => {
                if (confirm('This will reindex all embeddings in ChromaDB. This may take some time for large databases. Continue?')) {
                  performMaintenance('chromadb', 'reindex');
                }
              }}
              disabled={isWorking}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded disabled:bg-blue-300"
            >
              Reindex ChromaDB
            </button>
            <p className="text-sm text-gray-500 mt-1">
              Rebuilds all embeddings to ensure consistency and optimal performance.
            </p>
          </div>
        </div>
        
        {/* Cache Management */}
        <div className="border rounded p-4">
          <h2 className="text-lg font-bold mb-3">Cache Management</h2>
          
          <div className="space-y-3">
            <button
              onClick={() => {
                if (confirm('This will clear all cached results. Continue?')) {
                  fetch('/api/system/cache/clear', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                      if (data.success) {
                        setResult({ message: 'Cache cleared successfully' });
                      } else {
                        setError('Failed to clear cache');
                      }
                    })
                    .catch(err => {
                      console.error('Error clearing cache:', err);
                      setError('Error clearing cache');
                    });
                }
              }}
              disabled={isWorking}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded disabled:bg-blue-300"
            >
              Clear All Cache
            </button>
            <p className="text-sm text-gray-500 mt-1">
              Removes all cached results to free up disk space.
            </p>
          </div>
        </div>
      </div>
      
      {isWorking && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-lg text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-lg font-medium">Operation in progress...</p>
            <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## Conclusion

You've now built a fully local, privacy-first Adaptive Persona-Based Data Annotation platform. This system allows you to:

1. **Create and manage AI personas** with specific traits and behaviors
2. **Generate consistent, high-quality annotations** using these personas
3. **Collect feedback** and use it to refine personas over time through reinforcement learning
4. **Run everything locally** without any cloud dependencies
5. **Scale and optimize** the system as your needs grow


The platform combines the power of:
- **Next.js** for a seamless frontend and backend experience
- **Local databases** (SQLite/PostgreSQL) for data persistence
- **ChromaDB** for efficient vector search and semantic retrieval
- **Ollama** for running powerful language models locally
- **RLHF** for continuous improvement based on human feedback

By running everything locally, you maintain complete control over your data while still leveraging the power of modern AI for annotation tasks. The system is modular and extensible, allowing you to add new features, support additional model types, or scale to production environments as needed.

This guide has provided a comprehensive foundation, but there are many ways you could extend and enhance this platform:

1. **Specialized annotation interfaces** for different content types (code, medical data, legal documents)
2. **More sophisticated RLHF algorithms** for faster persona adaptation
3. **Integration with domain-specific models** for specialized annotation tasks
4. **Advanced visualization tools** for tracking annotation quality and consistency
5. **Batch processing capabilities** for handling large datasets efficiently

Remember that the key advantage of this system is its adaptivity - as you collect more feedback, the personas will continuously improve, leading to increasingly accurate and consistent annotations over time.

For optimal performance, regularly run the maintenance operations we've implemented, especially as your data volume grows. And don't hesitate to adjust the configuration parameters based on your specific hardware capabilities and requirements.

Thank you for following this guide. You now have a powerful, local-first annotation platform that respects privacy while delivering sophisticated AI-powered annotations through adaptive personas.






