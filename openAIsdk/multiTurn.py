from openai import OpenAI
from dotenv import load_dotenv
import os 
from rich import print  # reemplaza el print normal


load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),  # http://localhost:11434/v1
    api_key=os.getenv("API_KEY", "ollama"),
)

# Start a conversation
messages = [
    {"role": "user", "content": "Hi"}
]

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    max_tokens=100,
    messages=messages
)
# First agent response
agent_response = response.choices[0].message.content

print("[bold green] Agent Response[/] ", agent_response)

# Add the assistant's response to the conversation
messages.append({"role": "assistant", "content": agent_response})


#Second user message
messages.append({"role": "user", "content": "whats 2+2?"})
print("[bold blue] User Response [/] ", "What's 2+2?")

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    max_tokens=100,
    messages=messages
)
# Second agent response
agent_response = response.choices[0].message.content

print("[bold green] Agent Response[/] ", agent_response)