---
layout: post
title:  Basic Swarm Chatbot
date:   2024-11-28 07:40:44 -0500
---
# Guide to Building an AI-Powered Customer Support Chatbot Using Swarm

This guide will help you create an AI-powered customer support chatbot that utilizes OpenAI's Swarm to coordinate multiple specialized agents. Each agent will handle specific types of customer queries, such as billing issues, technical support, or general inquiries.

---

## Prerequisites

- **Python 3.10+** installed on your machine.
- **OpenAI API Key**: Obtain one from [OpenAI](https://platform.openai.com/account/api-keys).
- **Terminal Access**: Ability to run commands in your operating system's terminal.
- **Git** (optional): For version control.

---

## Step 1: Set Up the Project Environment

### 1.1 Create a Project Directory and Navigate Into It

```bash
mkdir ai_customer_support_chatbot
cd ai_customer_support_chatbot
```

### 1.2 Initialize a Git Repository (Optional)

```bash
git init
```

### 1.3 Create a Virtual Environment

```bash
python3 -m venv venv
```

### 1.4 Activate the Virtual Environment

- On **Linux/macOS**:

  ```bash
  source venv/bin/activate
  ```

- On **Windows**:

  ```bash
  venv\Scripts\activate
  ```

---

## Step 2: Install Required Dependencies

### 2.1 Upgrade pip

```bash
pip install --upgrade pip
```

### 2.2 Install Swarm and Other Required Packages

```bash
pip install git+https://github.com/openai/swarm.git
pip install python-dotenv
```

---

## Step 3: Securely Store Your OpenAI API Key

### 3.1 Create a `.env` File to Store Environment Variables

```bash
touch .env
```

### 3.2 Add `.env` to `.gitignore`

```bash
echo ".env" >> .gitignore
```

### 3.3 Add Your API Key to `.env`

Open `.env` in a text editor and add:

```ini
OPENAI_API_KEY=your_openai_api_key_here
```

**Note:** Replace `your_openai_api_key_here` with your actual API key.

---

## Step 4: Create the Main Script

### 4.1 Create `main.py`

```bash
touch main.py
```

### 4.2 Add the Following Code to `main.py`

```python
# main.py

import os
from dotenv import load_dotenv
from swarm import Swarm, Agent

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Swarm client
client = Swarm(openai_api_key=openai_api_key)

# Define specialized agents

# Billing Support Agent
billing_agent = Agent(
    name="Billing Support Agent",
    instructions="""
You are a helpful customer support agent specializing in billing issues.
Assist the user with their billing inquiries, such as charges, refunds, and payment methods.
If the query is not related to billing, politely inform the user and suggest contacting the appropriate department.
""",
)

# Technical Support Agent
technical_agent = Agent(
    name="Technical Support Agent",
    instructions="""
You are a helpful customer support agent specializing in technical issues.
Assist the user with technical problems, such as troubleshooting errors, connectivity issues, and software bugs.
If the query is not related to technical support, politely inform the user and suggest contacting the appropriate department.
""",
)

# General Inquiry Agent
general_agent = Agent(
    name="General Inquiry Agent",
    instructions="""
You are a helpful customer support agent handling general inquiries.
Assist the user with questions about account information, product details, and other general topics.
If the query is specialized (billing or technical), politely inform the user and suggest contacting the appropriate department.
""",
)

# Define a function to triage the user's query
def triage_query(context_variables, query: str):
    """
    Analyze the user's query and determine the appropriate agent to handle it.
    """
    if any(keyword in query.lower() for keyword in ["bill", "charge", "payment", "invoice", "refund"]):
        return billing_agent
    elif any(keyword in query.lower() for keyword in ["error", "issue", "bug", "technical", "problem", "troubleshoot"]):
        return technical_agent
    else:
        return general_agent

# Initial Agent (Triage Agent)
triage_agent = Agent(
    name="Triage Agent",
    instructions="""
You are an AI assistant that routes customer inquiries to the appropriate department.
Analyze the user's message and determine which specialized agent should handle it.
Call the function 'triage_query' to perform the routing.
""",
    functions=[triage_query],
)

def main():
    # Start the conversation
    user_message = input("User: ")

    # Prepare the initial messages
    messages = [
        {"role": "user", "content": user_message}
    ]

    # Run the Swarm client with the triage agent
    response = client.run(
        agent=triage_agent,
        messages=messages,
        context_variables={},
        max_turns=5,
        debug=False
    )

    # Get the final response
    final_agent = response.agent
    final_message = response.messages[-1]["content"]

    print(f"{final_agent.name}: {final_message}")

if __name__ == "__main__":
    main()
```

---

## Step 5: Run the Application

### 5.1 Execute `main.py`

```bash
python main.py
```

### 5.2 Interact with the Chatbot

After running the script, you will be prompted to enter a user message:

```
User: I need help with a charge on my account.
```

The chatbot will process your input and route it to the appropriate agent.

**Example Output:**

```
Billing Support Agent: I'm sorry to hear you're experiencing issues with a charge on your account. Could you please provide more details so I can assist you further?
```

---

## Additional Notes

- **Extending Functionality**: You can add more specialized agents for other departments like Sales, Account Management, etc.
- **Improving Triage**: Enhance the `triage_query` function to handle more complex routing logic.
- **Conversation Loop**: Modify the script to allow multiple turns in the conversation by placing the interaction inside a loop.

---

## Example: Extended Conversation Loop

To allow continuous interaction, update the `main()` function as follows:

```python
def main():
    # Initialize context variables
    context_variables = {}

    # Prepare initial messages
    messages = []

    # Conversation loop
    while True:
        user_message = input("User: ")
        if user_message.lower() in ["exit", "quit"]:
            print("Chatbot: Thank you for contacting support. Goodbye!")
            break

        messages.append({"role": "user", "content": user_message})

        # Run the Swarm client
        response = client.run(
            agent=triage_agent,
            messages=messages,
            context_variables=context_variables,
            max_turns=5,
            debug=False
        )

        # Get the latest agent and message
        final_agent = response.agent
        final_message = response.messages[-1]["content"]

        print(f"{final_agent.name}: {final_message}")

        # Update messages and context variables for the next turn
        messages = response.messages
        context_variables = response.context_variables
```

---

## Step 6: Test the Extended Chatbot

### 6.1 Run the Application

```bash
python main.py
```

### 6.2 Sample Interaction

```
User: I'm having trouble logging into my account.
Technical Support Agent: I'm sorry to hear you're having trouble logging in. Could you please describe the issue you're experiencing, and any error messages you might have received?
User: It says my password is incorrect, but I'm sure it's right.
Technical Support Agent: Understood. It's possible that your password needs to be reset. Would you like me to guide you through the password reset process?
User: Yes, please.
Technical Support Agent: Certainly! To reset your password, please click on the "Forgot Password" link on the login page. You'll be prompted to enter your registered email address, and we'll send you instructions to create a new password.
User: Thank you.
Technical Support Agent: You're welcome! If you have any more questions or need further assistance, feel free to ask.
User: exit
Chatbot: Thank you for contacting support. Goodbye!
```

---

## Conclusion

You've successfully built an AI-powered customer support chatbot using OpenAI's Swarm. The chatbot intelligently routes user queries to specialized agents based on the content of the message, providing a tailored support experience.

---

**Happy Coding!**

---

**Note:** This project uses OpenAI's Swarm, focusing on agent coordination and execution. By utilizing multiple agents with specific expertise, you can create a more dynamic and responsive chatbot that handles various customer needs efficiently.