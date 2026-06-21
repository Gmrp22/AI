from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),  # http://localhost:11434/v1
    api_key=os.getenv("API_KEY", "ollama"),
)

response = client.chat.completions.create(
    model=os.getenv("MODEL"),  # llama3.2:3b
    max_tokens=50,
    messages=[
        {"role": "system", "content": "You are a transfer agent. Say hello, introduce yourself, and ask only for the user's name. Once you have it, tell them they will be transferred and call transfer_agent."},
        {
            "role": "user",
            "content": "Hello"
        }
    ]
)

agent_response = response.choices[0].message.content
print(response)
print(agent_response)