---
layout: post
title: Building a Personalized AI Learning System with Local LLMs
description: In this guide, we will build an interactive learning platform that leverages AI to generate dynamic lessons, quizzes, and coding challenges from user-uploaded Markdown files.
date:   2025-03-30 02:42:44 -0500
---
# Building a Personalized AI Learning System with Local LLMs

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. System Architecture](#2-system-architecture)
- [3. Tech Stack & Tools](#3-tech-stack--tools)
- [4. Step-by-Step Implementation](#4-step-by-step-implementation)
- [5. Optimization & Expansion](#5-optimization--expansion)
- [6. Deployment & Hosting](#6-deployment--hosting)
- [7. Next Steps](#7-next-steps)

## 1. Introduction

### Why Build a Personalized AI Learning System?

Traditional e-learning platforms often rely on static content that doesn't adapt to individual learners. This guide presents a **fully AI-driven personalized learning system** that generates **entirely new lessons** for each interaction, making every session unique and context-aware.

The system dynamically adjusts content using **a knowledge graph and a local LLM**, ensuring learners receive increasingly relevant and challenging material based on their progress. This adaptive approach maximizes engagement and retention in ways traditional courses cannot.

### Key Features

✅ **Self-Hosted & Private:** Everything runs locally without reliance on cloud APIs  
✅ **Dynamic Lesson Generation:** Each lesson is uniquely tailored to the user's progress  
✅ **Knowledge Graph-Driven:** Lessons structured on connected concept maps, not linear modules  
✅ **Retrieval-Augmented Generation (RAG):** AI enhances lessons with relevant context  
✅ **Scalable & Modular:** Built with modern tech for flexibility and growth

## 2. System Architecture

The system uses a modular three-layer architecture:

### Frontend – Next.js + React

This provides the interface where users engage with AI-generated lessons:

- **User Dashboard:** Displays progress, completed lessons, and recommendations
- **Lesson UI:** Renders AI-generated content in an engaging format
- **Interactive Exercises:** Supports quizzes and challenges with real-time AI feedback
- **Progress Visualization:** Shows topic mastery through knowledge graph visualizations
- **AI Chat:** Provides on-demand explanations for concepts

### Backend – FastAPI

Manages user data, lesson requests, and AI interactions:

- **Content Processing:** Handles markdown files and processes them for the AI
- **Progress Tracking:** Stores learning history to adapt future lessons
- **Knowledge Graph Management:** Maintains concept relationships
- **API Endpoints:** Connects frontend and AI layer

### AI Layer – Local LLM + Knowledge Graph

The brain of the system:

- **Knowledge Graph:** Maps concepts and their relationships
- **RAG Implementation:** Enhances lesson quality with relevant context
- **Adaptive Generation:** Creates lessons based on user progress
- **Local Execution:** All AI runs on your hardware for privacy and control

### Data Flow

1. User requests a lesson from the frontend
2. Backend queries knowledge graph and past progress
3. AI layer generates a personalized, non-repetitive lesson
4. Frontend displays the lesson with interactive elements
5. User interactions update the knowledge graph and progress data

## 3. Tech Stack & Tools

### Frontend

- **Next.js (React):** For a responsive, server-rendered interface
- **TailwindCSS:** For utility-first styling
- **ShadCN UI:** For pre-built, customizable components
- **React-Flow:** For visualizing knowledge graphs

### Backend

- **FastAPI:** Python-based API with async support
- **SQLAlchemy:** ORM for database interactions
- **Pydantic:** For data validation

### Databases

- **PostgreSQL:** Stores structured data (user progress, lesson history)
- **ChromaDB:** Vector database for semantic search

### AI Components

- **Ollama:** Framework for running local LLMs
- **Mistral or Llama 3:** High-quality open-source LLM
- **NetworkX:** Python library for knowledge graph implementation
- **Sentence-Transformers:** For generating text embeddings

## 4. Step-by-Step Implementation

### Step 1: Environment Setup

First, let's set up our project structure and install dependencies:

```bash
# Create project directory
mkdir ai-learning-system
cd ai-learning-system

# Create subdirectories
mkdir -p frontend backend
```

#### Backend Setup:

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn pydantic sqlalchemy psycopg2-binary chromadb sentence-transformers networkx python-multipart

# Create basic directory structure
mkdir -p app/api app/db app/models app/services
```

#### Frontend Setup:

```bash
cd ../frontend

# Initialize Next.js project
npx create-next-app@latest . --typescript --tailwind --eslint --app

# Install additional dependencies
npm install react-flow-renderer react-markdown react-dropzone
```

### Step 2: Database Setup

#### PostgreSQL Setup

Let's create our database models for user progress and lesson history:

```python
# backend/app/models/database.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    progress = relationship("UserProgress", back_populates="user")
    
class Concept(Base):
    __tablename__ = "concepts"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text)
    difficulty = Column(Integer)  # 1-10 scale
    
    prerequisites = relationship(
        "ConceptRelationship",
        primaryjoin="Concept.id==ConceptRelationship.target_id",
        back_populates="target"
    )
    followups = relationship(
        "ConceptRelationship",
        primaryjoin="Concept.id==ConceptRelationship.source_id",
        back_populates="source"
    )

class ConceptRelationship(Base):
    __tablename__ = "concept_relationships"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("concepts.id"))
    target_id = Column(Integer, ForeignKey("concepts.id"))
    relationship_type = Column(String)  # e.g., "prerequisite", "related"
    strength = Column(Float)  # 0-1 representing relationship strength
    
    source = relationship("Concept", foreign_keys=[source_id], back_populates="followups")
    target = relationship("Concept", foreign_keys=[target_id], back_populates="prerequisites")

class UserProgress(Base):
    __tablename__ = "user_progress"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    concept_id = Column(Integer, ForeignKey("concepts.id"))
    mastery_level = Column(Float)  # 0-1 scale
    last_studied = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="progress")
    concept = relationship("Concept")

class Lesson(Base):
    __tablename__ = "lessons"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    concept_id = Column(Integer, ForeignKey("concepts.id"))
    content = Column(Text)
    generated_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    exercises = relationship("Exercise", back_populates="lesson")
    
class Exercise(Base):
    __tablename__ = "exercises"
    
    id = Column(Integer, primary_key=True, index=True)
    lesson_id = Column(Integer, ForeignKey("lessons.id"))
    question = Column(Text)
    answer = Column(Text)
    
    lesson = relationship("Lesson", back_populates="exercises")
```

Now, let's set up the database connection:

```python
# backend/app/db/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost/ai_learning")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

#### ChromaDB Setup

```python
# backend/app/db/vector_store.py
import chromadb
from chromadb.config import Settings
import os

class VectorStore:
    def __init__(self, persist_directory="./chroma_data"):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        
        # Create collections if they don't exist
        self.lesson_collection = self.client.get_or_create_collection("lessons")
        self.concept_collection = self.client.get_or_create_collection("concepts")
    
    def add_concept(self, concept_id, concept_name, concept_description, embedding):
        """Add a concept to the vector store"""
        self.concept_collection.add(
            ids=[str(concept_id)],
            embeddings=[embedding],
            metadatas=[{"name": concept_name}],
            documents=[concept_description]
        )
    
    def add_lesson_chunk(self, chunk_id, lesson_id, concept_id, content, embedding):
        """Add a lesson chunk to the vector store"""
        self.lesson_collection.add(
            ids=[str(chunk_id)],
            embeddings=[embedding],
            metadatas=[{"lesson_id": str(lesson_id), "concept_id": str(concept_id)}],
            documents=[content]
        )
    
    def search_similar_concepts(self, query_embedding, n_results=5):
        """Find similar concepts based on embedding"""
        results = self.concept_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    
    def search_relevant_content(self, query_embedding, n_results=10):
        """Find relevant lesson content based on embedding"""
        results = self.lesson_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

# Singleton instance to be used throughout the app
vector_store = VectorStore()
```

### Step 3: Knowledge Graph Implementation

Let's implement the knowledge graph using NetworkX:

```python
# backend/app/services/knowledge_graph.py
import networkx as nx
from app.db.database import get_db
from app.models.database import Concept, ConceptRelationship, UserProgress
import json

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.load_from_database()
    
    def load_from_database(self):
        """Load concept relationships from database into NetworkX graph"""
        db = next(get_db())
        
        # Get all concepts
        concepts = db.query(Concept).all()
        for concept in concepts:
            self.graph.add_node(
                concept.id, 
                name=concept.name, 
                description=concept.description,
                difficulty=concept.difficulty
            )
        
        # Get all relationships
        relationships = db.query(ConceptRelationship).all()
        for rel in relationships:
            self.graph.add_edge(
                rel.source_id, 
                rel.target_id, 
                type=rel.relationship_type,
                strength=rel.strength
            )
    
    def get_prerequisites(self, concept_id):
        """Get prerequisites for a given concept"""
        if not self.graph.has_node(concept_id):
            return []
        
        prerequisites = []
        for pred in self.graph.predecessors(concept_id):
            if self.graph[pred][concept_id].get('type') == 'prerequisite':
                prerequisites.append(pred)
        
        return prerequisites
    
    def get_next_concepts(self, concept_id):
        """Get concepts that follow the current one"""
        if not self.graph.has_node(concept_id):
            return []
        
        next_concepts = []
        for succ in self.graph.successors(concept_id):
            next_concepts.append(succ)
        
        return next_concepts
    
    def get_learning_path(self, start_concept, target_concept):
        """Find shortest path between concepts"""
        if not (self.graph.has_node(start_concept) and self.graph.has_node(target_concept)):
            return []
        
        try:
            path = nx.shortest_path(self.graph, start_concept, target_concept)
            return path
        except nx.NetworkXNoPath:
            return []
    
    def recommend_next_concept(self, user_id):
        """Recommend next concept for user based on progress"""
        db = next(get_db())
        
        # Get user's current progress
        progress_records = db.query(UserProgress).filter(
            UserProgress.user_id == user_id
        ).all()
        
        # Create a dict of concept_id -> mastery_level
        mastery = {p.concept_id: p.mastery_level for p in progress_records}
        
        # Find concepts user has started but not mastered
        in_progress = [cid for cid, level in mastery.items() if level < 0.8]
        
        if in_progress:
            # Return the concept with lowest mastery
            return min(in_progress, key=lambda x: mastery.get(x, 0))
        
        # If no concepts in progress, find new concepts where prerequisites are met
        mastered = [cid for cid, level in mastery.items() if level >= 0.8]
        
        candidate_concepts = []
        for concept_id in self.graph.nodes:
            if concept_id in mastery:
                continue  # Skip concepts user has already started
                
            prereqs = self.get_prerequisites(concept_id)
            if not prereqs or all(p in mastered for p in prereqs):
                # All prerequisites met
                candidate_concepts.append(concept_id)
        
        if not candidate_concepts:
            # If no obvious next concepts, recommend any starter concept
            starter_concepts = [n for n in self.graph.nodes if not list(self.graph.predecessors(n))]
            return starter_concepts[0] if starter_concepts else list(self.graph.nodes)[0]
        
        # Return easiest candidate concept (by difficulty)
        return min(candidate_concepts, key=lambda x: self.graph.nodes[x].get('difficulty', 5))

# Create singleton instance
knowledge_graph = KnowledgeGraph()

# Ensure graph is updated when DB changes
def refresh_knowledge_graph():
    knowledge_graph.load_from_database()
```

### Step 4: Embedding Service

Let's create a service for generating embeddings:

```python
# backend/app/services/embedding_service.py
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def get_embedding(self, text):
        """Generate embedding for text"""
        return self.model.encode(text).tolist()
    
    def get_embeddings(self, texts):
        """Generate embeddings for multiple texts"""
        return self.model.encode(texts).tolist()

# Create singleton instance
embedding_service = EmbeddingService()
```

### Step 5: LLM Service with Ollama

```python
# backend/app/services/llm_service.py
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("LLM_MODEL", "mistral")

class LLMService:
    def __init__(self, base_url=OLLAMA_BASE_URL, model=MODEL_NAME):
        self.base_url = base_url
        self.model = model
        self.generate_url = f"{self.base_url}/api/generate"
        
    def generate_text(self, prompt, max_tokens=2000, temperature=0.7):
        """Generate text from prompt using Ollama API"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }
        
        try:
            response = requests.post(self.generate_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return f"Error: {str(e)}"
    
    def generate_lesson(self, concept_name, concept_description, user_level="beginner", 
                        previous_knowledge=None, related_content=None):
        """Generate a complete lesson with RAG enhancement"""
        # Build context from related content
        context = ""
        if related_content:
            context = "Related information:\n" + "\n".join(related_content)
        
        # Include previous knowledge if available
        previous = ""
        if previous_knowledge:
            previous = "The user has previously learned:\n" + "\n".join(previous_knowledge)
        
        prompt = f"""
        You are an expert tutor creating a lesson about "{concept_name}".
        
        Basic description of the concept: {concept_description}
        
        User knowledge level: {user_level}
        
        {previous}
        
        {context}
        
        Create a comprehensive lesson that includes:
        1. A clear explanation of {concept_name}
        2. Key points to understand
        3. 2-3 concrete examples that demonstrate the concept
        4. 3 practice exercises with answer explanations
        5. A summary of what was covered
        
        Format the lesson using markdown with proper headings, lists, and code blocks if needed.
        Tailor the difficulty to {user_level} level while ensuring the content is engaging and not repetitive.
        """
        
        return self.generate_text(prompt)
    
    def generate_exercise_feedback(self, exercise, user_answer, correct_answer):
        """Generate feedback on a user's exercise answer"""
        prompt = f"""
        Exercise: {exercise}
        
        User's answer: {user_answer}
        
        Correct answer: {correct_answer}
        
        Provide helpful feedback on the user's answer. Include:
        1. Whether the answer is correct, partially correct, or incorrect
        2. Explanation of any mistakes or misconceptions
        3. Guidance on how to improve their understanding
        4. Positive reinforcement for what they did correctly
        
        Keep your tone encouraging and constructive.
        """
        
        return self.generate_text(prompt, max_tokens=800, temperature=0.5)

# Create singleton instance
llm_service = LLMService()
```

### Step 6: Backend API

Let's implement the main FastAPI application:

```python
# backend/app/main.py
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os
import json
from typing import List, Optional
from pydantic import BaseModel

from app.db.database import get_db, engine
from app.models.database import Base, User, Concept, UserProgress, Lesson, Exercise
from app.db.vector_store import vector_store
from app.services.embedding_service import embedding_service
from app.services.knowledge_graph import knowledge_graph, refresh_knowledge_graph
from app.services.llm_service import llm_service

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Learning System API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ConceptBase(BaseModel):
    name: str
    description: str
    difficulty: int
    
class ConceptCreate(ConceptBase):
    pass

class ConceptRead(ConceptBase):
    id: int
    
    class Config:
        orm_mode = True

class LessonRequest(BaseModel):
    concept_id: Optional[int] = None
    user_id: int

class LessonResponse(BaseModel):
    id: int
    content: str
    concept: ConceptRead
    exercises: List[dict]
    
    class Config:
        orm_mode = True

# Routes
@app.get("/")
def read_root():
    return {"message": "AI Learning System API"}

@app.post("/concepts/", response_model=ConceptRead)
def create_concept(concept: ConceptCreate, db: Session = Depends(get_db)):
    db_concept = Concept(
        name=concept.name,
        description=concept.description,
        difficulty=concept.difficulty
    )
    db.add(db_concept)
    db.commit()
    db.refresh(db_concept)
    
    # Add to vector store
    embedding = embedding_service.get_embedding(f"{concept.name} {concept.description}")
    vector_store.add_concept(
        db_concept.id, 
        db_concept.name, 
        db_concept.description, 
        embedding
    )
    
    # Refresh knowledge graph
    refresh_knowledge_graph()
    
    return db_concept

@app.get("/concepts/", response_model=List[ConceptRead])
def get_concepts(db: Session = Depends(get_db)):
    concepts = db.query(Concept).all()
    return concepts

@app.post("/lessons/generate/", response_model=dict)
def generate_lesson(request: LessonRequest, db: Session = Depends(get_db)):
    # Check if user exists
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Determine which concept to teach
    concept_id = request.concept_id
    if not concept_id:
        # Use knowledge graph to recommend next concept
        concept_id = knowledge_graph.recommend_next_concept(request.user_id)
    
    concept = db.query(Concept).filter(Concept.id == concept_id).first()
    if not concept:
        raise HTTPException(status_code=404, detail="Concept not found")
    
    # Get user's level based on progress
    progress = db.query(UserProgress).filter(
        UserProgress.user_id == request.user_id,
        UserProgress.concept_id == concept_id
    ).first()
    
    user_level = "beginner"
    if progress:
        if progress.mastery_level > 0.8:
            user_level = "advanced"
        elif progress.mastery_level > 0.4:
            user_level = "intermediate"
    
    # Get previous knowledge (mastered concepts)
    mastered_concepts = db.query(Concept).join(UserProgress).filter(
        UserProgress.user_id == request.user_id,
        UserProgress.mastery_level >= 0.8
    ).all()
    
    previous_knowledge = [f"{c.name}: {c.description}" for c in mastered_concepts]
    
    # Find related content using RAG
    concept_embedding = embedding_service.get_embedding(
        f"{concept.name} {concept.description}"
    )
    
    related_results = vector_store.search_relevant_content(concept_embedding)
    related_content = related_results.get("documents", [])
    
    # Generate lesson with LLM
    lesson_content = llm_service.generate_lesson(
        concept.name,
        concept.description,
        user_level,
        previous_knowledge,
        related_content
    )
    
    # Parse exercises from the lesson (simplified)
    # In a real implementation, you'd use a more robust method to extract exercises
    exercises = []
    
    # Create lesson record
    db_lesson = Lesson(
        user_id=request.user_id,
        concept_id=concept_id,
        content=lesson_content
    )
    db.add(db_lesson)
    db.commit()
    db.refresh(db_lesson)
    
    # Update or create user progress
    if not progress:
        progress = UserProgress(
            user_id=request.user_id,
            concept_id=concept_id,
            mastery_level=0.1  # Initial mastery
        )
        db.add(progress)
    else:
        # Increment slightly just for viewing the lesson
        progress.mastery_level = min(progress.mastery_level + 0.05, 1.0)
    
    db.commit()
    
    # Return lesson data
    return {
        "id": db_lesson.id,
        "content": lesson_content,
        "concept": {
            "id": concept.id,
            "name": concept.name,
            "description": concept.description,
            "difficulty": concept.difficulty
        },
        "exercises": exercises
    }

@app.post("/upload/")
async def upload_markdown(
    file: UploadFile = File(...),
    user_id: int = Form(...)
):
    # Read file contents
    contents = await file.read()
    text = contents.decode("utf-8")
    
    # Here you would implement markdown parsing to extract concepts
    # For simplicity, we'll just assume the file contains concept data
    
    # Example implementation:
    import re
    
    # Extract headings as concepts
    headings = re.findall(r'## (.*?)\n', text)
    
    # Process each heading as a concept
    for heading in headings:
        # Extract paragraph after heading as description
        description_match = re.search(f'## {re.escape(heading)}\n\n(.*?)\n\n', text, re.DOTALL)
        description = description_match.group(1) if description_match else "No description available"
        
        # Create concept
        db = next(get_db())
        concept = Concept(
            name=heading,
            description=description,
            difficulty=5  # Default difficulty
        )
        db.add(concept)
        db.commit()
        db.refresh(concept)
        
        # Add to vector store
        embedding = embedding_service.get_embedding(f"{heading} {description}")
        vector_store.add_concept(
            concept.id,
            concept.name,
            concept.description,
            embedding
        )
    
    # Refresh knowledge graph
    refresh_knowledge_graph()
    
    return {"message": f"Processed {len(headings)} concepts from {file.filename}"}

@app.post("/progress/update/")
def update_progress(
    user_id: int,
    concept_id: int,
    mastery_level: float,
    db: Session = Depends(get_db)
):
    progress = db.query(UserProgress).filter(
        UserProgress.user_id == user_id,
        UserProgress.concept_id == concept_id
    ).first()
    
    if not progress:
        progress = UserProgress(
            user_id=user_id,
            concept_id=concept_id,
            mastery_level=mastery_level
        )
        db.add(progress)
    else:
        progress.mastery_level = mastery_level
    
    db.commit()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

### Step 7: Frontend Implementation

Let's create the key components for our Next.js frontend:

#### App Layout

```tsx
// frontend/app/layout.tsx
import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import Sidebar from '@/components/Sidebar'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AI Learning System',
  description: 'Personalized learning powered by AI',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="flex h-screen">
          <Sidebar />
          <main className="flex-1 p-6 overflow-auto">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
```

#### Sidebar Component

```tsx
// frontend/components/Sidebar.tsx
import Link from 'next/link'
import { usePathname } from 'next/navigation'

const Sidebar = () => {
  const pathname = usePathname()
  
  const links = [
    { href: '/', label: 'Dashboard' },
    { href: '/learn', label: 'Learn' },
    { href: '/progress', label: 'Progress' },
    { href: '/upload', label: 'Upload Content' },
  ]
  
  return (
    <aside className="w-64 bg-gray-800 text-white p-4">
      <h1 className="text-xl font-bold mb-6">AI Learning System</h1>
      <nav>
        <ul>
          {links.map((link) => (
            <li key={link.href} className="mb-2">
              <Link href={link.href} 
                className={`block p-2 rounded hover:bg-gray-700 ${
                  pathname === link.href ? 'bg-gray-700' : ''
                }`}
              >
                {link.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  )
}

export default Sidebar
```

#### Dashboard Page

```tsx
// frontend/app/page.tsx
'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'

export default function Dashboard() {
  const [recentLessons, setRecentLessons] = useState([])
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(true)
  
  // In a real app, you'd fetch this data from your API
  useEffect(() => {
    // Mock data for demonstration
    setRecentLessons([
      { id: 1, title: 'Introduction to Neural Networks', date: '2023-05-15' },
      { id: 2, title: 'Python Basics', date: '2023-05-12' },
    ])
    
    setRecommendations([
      { id: 3, title: 'Reinforcement Learning', difficulty: 'Intermediate' },
      { id: 4, title: 'Data Preprocessing', difficulty: 'Beginner' },
    ])
    
    setLoading(false)
  }, [])
  
  if (loading) {
    return <div className="flex justify-center items-center h-full">Loading...</div>
  }
  
  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Learning Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Recent Lessons</h2>
          {recentLessons.length > 0 ? (
            <ul className="space-y-2">
              {recentLessons.map((lesson) => (
                <li key={lesson.id} className="border-b pb-2">
                  <Link href={`/learn/${lesson.id}`} className="text-blue-600 hover:underline">
                    {lesson.title}
                  </Link>
                  <p className="text-sm text-gray-500">{lesson.date}</p>
                </li>
              ))}
            </ul>
          ) : (
            <p>No recent lessons found.</p>
          )}
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Recommended For You</h2>
          {recommendations.length > 0 ? (
            <ul className="space-y-2">
              {recommendations.map((rec) => (
                <li key={rec.id} className="border-b pb-2">
                  <Link href={`/learn/start?concept=${rec.id}`} className="text-blue-600 hover:underline">
                    {rec.title}
                  </Link>
                  <p className="text-sm text-gray-500">Difficulty: {rec.difficulty}</p>
                </li>
              ))}
            </ul>
          ) : (
            <p>No recommendations available.</p>
          )}
        </div>
      </div>
      
      <div className="mt-6 bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Your Learning Progress</h2>
        <div className="h-64 flex items-center justify-center border border-gray-200 rounded">
          <p className="text-gray-500">Progress visualization will appear here</p>
        </div>
      </div>
    </div>
  )
}
```

#### Learn Page

```tsx
// frontend/app/learn/page.tsx
'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

export default function LearnPage() {
  const router = useRouter()
  const [loading, setLoading] = useState(false)
  
  // Mock data - in a real app you'd fetch these from your API
  const topics = [
    { id: 1, name: 'Python Basics', difficulty: 'Beginner' },
    { id: 2, name: 'Data Structures', difficulty: 'Intermediate' },
    { id: 3, name: 'Machine Learning Fundamentals', difficulty: 'Intermediate' },
    { id: 4, name: 'Neural Networks', difficulty: 'Advanced' },
  ]
  
  const startLesson = async (topicId: number) => {
    setLoading(true)
    
    try {
      // In a real app, you'd make an API call to generate a lesson
      // const response = await fetch('/api/lessons/generate', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ concept_id: topicId, user_id: 1 }),
      // })
      // const data = await response.json()
      
      // For demo purposes, we'll just navigate to a mock lesson
      router.push(`/learn/${topicId}`)
    } catch (error) {
      console.error('Error starting lesson:', error)
    } finally {
      setLoading(false)
    }
  }
  
  const getRecommendedLesson = async () => {
    setLoading(true)
    
    try {
      // In a real app, you'd make an API call to get a recommended lesson
      // const response = await fetch('/api/lessons/recommend', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ user_id: 1 }),
      // })
      // const data = await response.json()
      
      // For demo, we'll randomly select a topic
      const randomTopic = topics[Math.floor(Math.random() * topics.length)]
      router.push(`/learn/${randomTopic.id}`)
    } catch (error) {
      console.error('Error getting recommendation:', error)
    } finally {
      setLoading(false)
    }
  }
  
  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Start Learning</h1>
      
      <div className="mb-6">
        <button
          onClick={getRecommendedLesson}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Loading...' : 'Generate Recommended Lesson'}
        </button>
      </div>
      
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Available Topics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {topics.map((topic) => (
            <div key={topic.id} className="border rounded p-4">
              <h3 className="font-medium">{topic.name}</h3>
              <p className="text-sm text-gray-500 mb-3">Difficulty: {topic.difficulty}</p>
              <button
                onClick={() => startLesson(topic.id)}
                disabled={loading}
                className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
              >
                Start Lesson
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
```

#### Lesson Display Component

```tsx
// frontend/app/learn/[id]/page.tsx
'use client'

import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

export default function LessonPage({ params }: { params: { id: string } }) {
  const [lesson, setLesson] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('lesson')
  const [userAnswers, setUserAnswers] = useState<Record<number, string>>({})
  const [feedback, setFeedback] = useState<Record<number, string>>({})
  
  useEffect(() => {
    // In a real app, fetch from your API
    // For demo, we'll use mock data
    const mockLesson = {
      id: parseInt(params.id),
      title: 'Introduction to Neural Networks',
      content: `
# Introduction to Neural Networks

Neural networks are computational models inspired by the human brain. They consist of layers of interconnected nodes or "neurons" that process information.

## Key Concepts

- **Neurons**: Basic units that receive inputs, apply weights, and output signals
- **Layers**: Groups of neurons that process information sequentially
- **Activation Functions**: Functions that determine the output of a neuron
- **Weights and Biases**: Parameters that are adjusted during training

## How Neural Networks Work

A neural network processes data through layers:

1. Input layer receives the initial data
2. Hidden layers process the data
3. Output layer produces the final result

## Examples

### Example 1: Image Recognition

A neural network can be trained to recognize images:
- Input: Pixel values from an image
- Process: Multiple layers extract features (edges, shapes, etc.)
- Output: Classification (e.g., "cat", "dog", "car")

### Example 2: Natural Language Processing

Neural networks power modern language models:
- Input: Text converted to numerical representations
- Process: Layers extract meaning and context
- Output: Generated text, translations, or classifications
      `,
      exercises: [
        {
          id: 1,
          question: "What is the main inspiration for neural networks?",
          answer: "The human brain"
        },
        {
          id: 2,
          question: "Name the three main types of layers in a neural network.",
          answer: "Input layer, hidden layers, and output layer"
        },
        {
          id: 3,
          question: "What parameters are adjusted during the training process?",
          answer: "Weights and biases"
        }
      ]
    }
    
    setLesson(mockLesson)
    setLoading(false)
  }, [params.id])
  
  const submitAnswer = async (exerciseId: number) => {
    const userAnswer = userAnswers[exerciseId] || ''
    
    // In a real app, you'd send this to your API for evaluation
    // For demo, we'll just compare with the expected answer
    const exercise = lesson.exercises.find((ex: any) => ex.id === exerciseId)
    
    if (exercise) {
      const correctAnswer = exercise.answer
      
      // Simple check - in a real app, use the LLM to evaluate
      let feedbackText
      if (userAnswer.toLowerCase() === correctAnswer.toLowerCase()) {
        feedbackText = "Correct! Well done."
      } else {
        feedbackText = `Not quite. The correct answer is: ${correctAnswer}`
      }
      
      setFeedback({
        ...feedback,
        [exerciseId]: feedbackText
      })
    }
  }
  
  if (loading) {
    return <div className="flex justify-center items-center h-full">Loading lesson...</div>
  }
  
  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">{lesson.title}</h1>
      
      <div className="mb-6">
        <nav className="flex border-b">
          <button
            className={`py-2 px-4 ${activeTab === 'lesson' ? 'border-b-2 border-blue-500 font-medium' : ''}`}
            onClick={() => setActiveTab('lesson')}
          >
            Lesson
          </button>
          <button
            className={`py-2 px-4 ${activeTab === 'exercises' ? 'border-b-2 border-blue-500 font-medium' : ''}`}
            onClick={() => setActiveTab('exercises')}
          >
            Exercises
          </button>
        </nav>
      </div>
      
      <div className="bg-white p-6 rounded-lg shadow">
        {activeTab === 'lesson' ? (
          <div className="prose max-w-none">
            <ReactMarkdown>{lesson.content}</ReactMarkdown>
          </div>
        ) : (
          <div>
            <h2 className="text-xl font-semibold mb-4">Practice Exercises</h2>
            <div className="space-y-6">
              {lesson.exercises.map((exercise: any) => (
                <div key={exercise.id} className="border p-4 rounded">
                  <p className="font-medium mb-2">{exercise.question}</p>
                  <textarea
                    className="w-full border rounded p-2 mb-2"
                    placeholder="Your answer..."
                    rows={3}
                    value={userAnswers[exercise.id] || ''}
                    onChange={(e) => setUserAnswers({
                      ...userAnswers,
                      [exercise.id]: e.target.value
                    })}
                  />
                  <button
                    className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
                    onClick={() => submitAnswer(exercise.id)}
                  >
                    Submit Answer
                  </button>
                  
                  {feedback[exercise.id] && (
                    <div className={`mt-2 p-2 rounded ${
                      feedback[exercise.id].startsWith('Correct') 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {feedback[exercise.id]}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      
      <div className="mt-6 flex justify-between">
        <button className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700">
          Previous Lesson
        </button>
        <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
          Mark as Complete
        </button>
      </div>
    </div>
  )
}
```

#### File Upload Component

```tsx
// frontend/app/upload/page.tsx
'use client'

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

export default function UploadPage() {
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<null | { success: boolean; message: string }>(null)
  
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return
    
    setUploading(true)
    setUploadStatus(null)
    
    try {
      const file = acceptedFiles[0]
      
      // Create form data for file upload
      const formData = new FormData()
      formData.append('file', file)
      formData.append('user_id', '1')  // In a real app, get from auth context
      
      // In a real app, send to your API
      // const response = await fetch('http://localhost:8000/upload/', {
      //   method: 'POST',
      //   body: formData,
      // })
      
      // For demo purposes, simulate success
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      setUploadStatus({
        success: true,
        message: `Successfully processed ${file.name}`
      })
    } catch (error) {
      console.error('Upload error:', error)
      setUploadStatus({
        success: false,
        message: 'Error uploading file. Please try again.'
      })
    } finally {
      setUploading(false)
    }
  }, [])
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/markdown': ['.md'],
      'text/plain': ['.txt']
    },
    maxFiles: 1
  })
  
  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Upload Learning Material</h1>
      
      <div className="bg-white p-6 rounded-lg shadow">
        <p className="mb-4">
          Upload markdown files containing learning materials. The system will process the content
          and extract concepts, examples, and exercises.
        </p>
        
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer ${
            isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
          }`}
        >
          <input {...getInputProps()} />
          {uploading ? (
            <p>Uploading...</p>
          ) : isDragActive ? (
            <p>Drop the file here...</p>
          ) : (
            <div>
              <p>Drag and drop a markdown file here, or click to select a file</p>
              <p className="text-sm text-gray-500 mt-2">Supports .md and .txt files</p>
            </div>
          )}
        </div>
        
        {uploadStatus && (
          <div
            className={`mt-4 p-3 rounded ${
              uploadStatus.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}
          >
            {uploadStatus.message}
          </div>
        )}
        
        <div className="mt-6">
          <h3 className="font-medium mb-2">How it works:</h3>
          <ol className="list-decimal list-inside space-y-1 text-sm">
            <li>Upload a markdown file with headings and content</li>
            <li>The system extracts concepts and their relationships</li>
            <li>Content is processed into the knowledge graph</li>
            <li>AI generates personalized lessons based on this content</li>
          </ol>
        </div>
      </div>
    </div>
  )
}
```

### Step 8: Bringing It Together

Now you need to make both systems work together:

1. Start the backend service:

```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. In a separate terminal, start the frontend:

```bash
cd frontend
npm run dev
```

3. Ensure Ollama is running with your chosen model:

```bash
ollama run mistral
```

## 5. Optimization & Expansion

### Explain-Back Challenges

One of the most effective ways to enhance learning is to have students explain concepts back in their own words. Let's implement this:

```python
# backend/app/services/llm_service.py
# Add this method to the LLMService class

def evaluate_explanation(self, concept_name, concept_description, user_explanation):
    """Evaluate a user's explanation of a concept"""
    prompt = f"""
    Concept: {concept_name}
    
    Expert explanation: {concept_description}
    
    User explanation: {user_explanation}
    
    As an AI tutor, evaluate how well the user has understood and explained the concept.
    
    Provide feedback in the following format:
    - Accuracy (0-10): [score]
    - Completeness (0-10): [score]
    - Areas of strength: [what they explained well]
    - Areas for improvement: [what they missed or misunderstood]
    - Suggestions: [specific advice to improve understanding]
    
    Keep your tone encouraging and constructive.
    """
    
    return self.generate_text(prompt, max_tokens=800, temperature=0.3)
```

### Adaptive Scaling

To implement adaptive difficulty scaling:

```python
# backend/app/services/knowledge_graph.py
# Add this method to KnowledgeGraph class

def get_appropriate_difficulty(self, user_id, max_difficulty=10):
    """Determine appropriate difficulty level based on user mastery"""
    db = next(get_db())
    
    # Get average mastery level across all concepts
    progress = db.query(UserProgress).filter(
        UserProgress.user_id == user_id
    ).all()
    
    if not progress:
        return 2  # Start with easy concepts for new users
    
    avg_mastery = sum(p.mastery_level for p in progress) / len(progress)
    
    # Scale difficulty based on mastery (higher mastery = higher difficulty)
    # This is a simple linear scaling approach
    recommended_difficulty = int(avg_mastery * 10) + 1
    
    # Cap at max_difficulty
    return min(recommended_difficulty, max_difficulty)
```

### Custom Learning Paths

Allow users to define their own learning goals:

```python
# backend/app/main.py
# Add this endpoint

@app.post("/learning-path/create/")
def create_learning_path(
    user_id: int,
    goal_concept_id: int,
    db: Session = Depends(get_db)
):
    # Verify user and concept exist
    user = db.query(User).filter(User.id == user_id).first()
    goal_concept = db.query(Concept).filter(Concept.id == goal_concept_id).first()
    
    if not user or not goal_concept:
        raise HTTPException(status_code=404, detail="User or concept not found")
    
    # Get user's current knowledge state
    mastered_concepts = db.query(Concept).join(UserProgress).filter(
        UserProgress.user_id == user_id,
        UserProgress.mastery_level >= 0.8
    ).all()
    
    # If user has no mastered concepts, find starter concepts
    if not mastered_concepts:
        starter_concepts = []
        for node in knowledge_graph.graph.nodes:
            if not list(knowledge_graph.graph.predecessors(node)):
                # This is a starter node with no prerequisites
                starter_concepts.append(node)
        
        return {
            "starter_concepts": [
                {"id": c, "name": knowledge_graph.graph.nodes[c].get('name')}
                for c in starter_concepts
            ],
            "path": []
        }
    
    # Find path from user's most relevant mastered concept to goal
    most_relevant_mastered = None
    shortest_path = None
    
    for mc in mastered_concepts:
        try:
            path = nx.shortest_path(knowledge_graph.graph, mc.id, goal_concept_id)
            if shortest_path is None or len(path) < len(shortest_path):
                shortest_path = path
                most_relevant_mastered = mc
        except nx.NetworkXNoPath:
            continue
    
    # If no path found, return next best concepts to learn
    if not shortest_path:
        # Find concepts that user hasn't mastered that have all prerequisites met
        candidates = []
        for node in knowledge_graph.graph.nodes:
            if node in [c.id for c in mastered_concepts]:
                continue  # Skip already mastered concepts
                
            prereqs = knowledge_graph.get_prerequisites(node)
            if not prereqs or all(p in [c.id for c in mastered_concepts] for p in prereqs):
                candidates.append(node)
        
        return {
            "message": "No direct path to goal. Consider these intermediate concepts:",
            "recommended_concepts": [
                {"id": c, "name": knowledge_graph.graph.nodes[c].get('name')}
                for c in candidates
            ]
        }
    
    # Return the learning path
    path_concepts = []
    for concept_id in shortest_path:
        if concept_id not in [c.id for c in mastered_concepts]:
            node = knowledge_graph.graph.nodes[concept_id]
            path_concepts.append({
                "id": concept_id,
                "name": node.get('name'),
                "difficulty": node.get('difficulty', 5)
            })
    
    return {
        "starting_from": most_relevant_mastered.name,
        "goal": goal_concept.name,
        "learning_path": path_concepts
    }
```

## 6. Deployment & Hosting

### Local Deployment with Docker

Create a `docker-compose.yml` file in the project root:

```yaml
version: '3'

services:
  frontend:
    build:
      context: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    
  backend:
    build:
      context: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - ollama
    environment:
      - DATABASE_URL=postgresql://ailearning:password@postgres/ailearning
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ./backend:/app
      - chroma_data:/app/chroma_data

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=ailearning
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=ailearning
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
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  postgres_data:
  ollama_models:
  chroma_data:
```

Create a Dockerfile for the backend:

```dockerfile
# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create a Dockerfile for the frontend:

```dockerfile
# frontend/Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build

CMD ["npm", "start"]
```

### VPS Deployment

For deployment to a VPS:

1. Set up a server with Ubuntu
2. Install Docker and Docker Compose
3. Clone your repository
4. Start the services:

```bash
docker-compose up -d
```

5. Set up Nginx as a reverse proxy:

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api {
        rewrite ^/api/(.*) /$1 break;
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

6. Set up SSL with Certbot:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

## 7. Next Steps

After implementing the core system, consider these next steps:

1. **Define a Comprehensive API Schema**
   - Document all endpoints with OpenAPI
   - Implement strong validation rules
   - Create API clients for frontend use

2. **Enhance the Feedback Loop**
   - Add metrics for lesson effectiveness
   - Implement user ratings for lessons
   - Collect and analyze exercise performance

3. **Fine-Tune RAG Prompts**
   - Experiment with different context retrieval methods
   - Optimize prompt templates for different learning styles
   - Implement prompt versioning to compare effectiveness

4. **Add Comprehensive Logging**
   - Track user interactions for debugging
   - Monitor system performance
   - Set up alerts for critical errors

5. **Implement User Authentication**
   - Add secure login and registration
   - Support multiple user roles (learner, instructor)
   - Add profile management

---

This guide provides a comprehensive framework for building your personalized AI learning system with local LLMs. By following these steps, you'll create a system that can generate unique, adaptive lessons, build a knowledge graph of concepts, and provide an engaging learning experience.

The modular nature of this implementation allows you to start with core functionality and progressively enhance the system with additional features as you go. Happy building!