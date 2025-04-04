---
layout: post
title:  Persona Chat
date:   2024-12-02 07:42:44 -0500
---
# Building a Personalized AI Assistant with LangChain: A Starting Point for Your Next NLP Project

Artificial Intelligence has revolutionized the way we interact with technology, and Natural Language Processing (NLP) is at the forefront of this transformation. With the rise of language models like OpenAI's GPT series, developers now have the tools to create sophisticated AI assistants that can understand and generate human-like text. In this blog post, we'll explore a Python script that serves as a foundation for building such an AI assistant. We'll delve into how you can use this code as a starting point and discuss several exciting applications, complete with follow-up prompts to inspire your next project.

## Overview of the Repository

The provided Python script leverages the power of LangChain, OpenAI's GPT models, and vector databases to create an AI assistant that imitates the writing style of a specific persona based on provided writing samples. Here's what the script does:

1. **Loads Writing Samples**: Reads text and PDF files from a specified folder to gather writing samples.
2. **Processes and Embeds Text**: Splits the text into manageable chunks and creates embeddings using OpenAI's API.
3. **Creates a Vector Store**: Stores the embeddings in a Chroma vector store for efficient retrieval.
4. **Sets Up a Retrieval QA Chain**: Uses LangChain's RetrievalQA to build an interactive question-answering system.
5. **Interacts with the User**: Provides a conversational interface where the AI assistant responds in the persona's writing style.
6. **Saves Conversations**: Logs the conversation history into a Markdown file for future reference.

## Getting Started

Before diving into applications, let's understand how to set up the environment.

### Prerequisites

- **Python 3.7+**
- **OpenAI API Key**: Obtain one from the [OpenAI dashboard](https://beta.openai.com/account/api-keys).
- **Required Libraries**: Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Directory Structure

- **`writing_samples/`**: Place your text (`.txt`) and PDF (`.pdf`) files here.
- **`persona_vectorstore/`**: Directory where the vector store will be persisted.
- **`conversation.md`**: File where the conversation history is saved.

### Running the Script

1. **Set Up Environment Variables**: Create a `.env` file with your OpenAI API key.

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. **Execute the Script**:

   ```bash
   python script_name.py
   ```

   Replace `script_name.py` with the actual name of the Python file.

## Step-by-Step Explanation

Let's break down the main components of the script.

### 1. Loading and Processing Writing Samples

The script recursively scans the `writing_samples/` directory for `.txt` and `.pdf` files.

```python
folder_path = './writing_samples'
documents = []
for filepath in glob.glob(os.path.join(folder_path, '**/*.*'), recursive=True):
    # Load text and PDF files
```

It uses `TextLoader` for text files and `PyPDFLoader` for PDFs. The loaded documents are then split into chunks using `RecursiveCharacterTextSplitter` to ensure the embeddings are manageable.

### 2. Creating Embeddings and Vector Store

Embeddings are generated using `OpenAIEmbeddings`, which converts text chunks into high-dimensional vectors.

```python
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = Chroma.from_documents(texts, embeddings, persist_directory="./persona_vectorstore")
```

These embeddings are stored in a Chroma vector store, allowing for efficient similarity searches during retrieval.

### 3. Setting Up the Retrieval QA Chain

A retriever is created to fetch relevant chunks based on the user's query.

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

A `PromptTemplate` is defined to instruct the AI assistant to answer in the persona's writing style.

```python
persona_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant imitating the writing style of a specific persona based on provided writing samples.

Context:
{context}

Question:
{question}

Answer in the persona's writing style.
"""
)
```

The `RetrievalQA` chain ties everything together.

### 4. User Interaction and Conversation Logging

The script enters an interactive loop where it prompts the user for input and generates responses using the QA chain.

```python
while True:
    user_input = input("You: ")
    if user_input.lower() in ('exit', 'quit'):
        break

    response = qa_chain.run(user_input)
    print(f"Persona: {response}\n")
```

Each conversation turn is appended to a Markdown file for record-keeping.

## Potential Applications and Follow-Up Prompts

This repository serves as a versatile starting point for various NLP applications. Let's explore some ideas and provide follow-up prompts to guide your development.

### 1. Personal Writing Assistant

**Description**: Create an AI assistant that helps you write emails, articles, or stories in your own writing style.

**Implementation Tips**:

- Use your own writing samples to train the model.
- Modify the prompt to focus on assisting with specific writing tasks.
- Incorporate additional memory components to retain context over longer interactions.

**Follow-Up Prompts**:

- *"Can you help me draft an email to my team about the upcoming project deadline?"*
- *"Write a blog post introduction about the importance of mental health in the workplace."*
- *"How would I explain the concept of blockchain in my own writing style?"*

### 2. Chatbot in the Style of a Famous Author

**Description**: Build a chatbot that responds in the writing style of a renowned author like Shakespeare, Jane Austen, or Mark Twain.

**Implementation Tips**:

- Collect public domain works of the author as writing samples.
- Adjust the prompt to encourage creative and stylistic responses.
- Consider adding constraints to match the historical context or language.

**Follow-Up Prompts**:

- *"Tell me a story about a modern-day adventure in the style of Mark Twain."*
- *"Compose a sonnet about technology as Shakespeare would."*
- *"Discuss the themes of love and society in today's world like Jane Austen."*

### 3. Customer Service Bot Trained on Company Documents

**Description**: Develop a customer service assistant that provides support using information from company manuals, FAQs, and policy documents.

**Implementation Tips**:

- Load internal documents into the `writing_samples/` directory.
- Ensure sensitive information is handled appropriately.
- Fine-tune the prompt to maintain a professional tone.

**Follow-Up Prompts**:

- *"How can I reset my account password?"*
- *"What is the return policy for defective products?"*
- *"Explain the warranty terms for my new purchase."*

### 4. Educational Tutor Imitating a Teaching Style

**Description**: Create an AI tutor that teaches subjects using a specific educator's style, making learning more personalized.

**Implementation Tips**:

- Use transcripts or written materials from the educator.
- Adjust the prompt to include educational objectives.
- Incorporate interactive elements like quizzes or prompts for student reflection.

**Follow-Up Prompts**:

- *"Help me understand the Pythagorean theorem in your teaching style."*
- *"Explain the causes of World War II as Mr. Smith would in his history class."*
- *"Provide a chemistry lesson on the periodic table in an engaging way."*

### 5. Content Generator for Marketing Teams

**Description**: Assist marketing teams in generating content that aligns with the brand's voice and style guidelines.

**Implementation Tips**:

- Include brand guidelines and previous marketing materials as writing samples.
- Modify the prompt to focus on content creation objectives.
- Ensure compliance with brand messaging and tone.

**Follow-Up Prompts**:

- *"Draft a social media post announcing our new product launch."*
- *"Write an engaging headline for our upcoming email newsletter."*
- *"Create ad copy that highlights the benefits of our service."*

## Conclusion

The provided Python script offers a solid foundation for building AI assistants that can mimic specific writing styles and serve various purposes. By customizing the writing samples, prompts, and chain configurations, you can adapt this code to fit numerous applications, from personal assistants to educational tools.

As you embark on your NLP project, consider how you can extend and refine this script to meet your goals. The possibilities are vast, and with powerful libraries like LangChain and OpenAI's APIs at your disposal, you're well-equipped to innovate in the field of natural language processing.

---

*Happy coding! If you have any questions or need further guidance, feel free to reach out or leave a comment below.*