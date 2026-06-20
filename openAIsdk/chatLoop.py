from rich import print

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY", "ollama"),
)

messages = []

print("Chat con Ollama (escribr 'salir' para terminar)\n")

while True:
    user_input = input("\n[bold blue]Tú[/]: ")

    if user_input.lower() == 'salir':
        print("\n[yellow]Adiós :wave:[/]")
        break
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
        temperature=0.7,
        max_tokens=150,
        top_p=0.9
    )

    agent_response = response.choices[0].message.content
    print("\n[bold green]Agente[/]: ", agent_response)

    messages.append({"role": "assistant", "content": agent_response})
    