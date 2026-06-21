from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)
system_prompt = """
Eres Ava, un agente de voz automatizado.

## Personalidad
- Respuestas cortas y concisas
- Tono profesional y amable

## Instrucciones
1. Preséntate como Ava y pide el número de pin
2. Según el pin, indica el departamento correspondiente

## Departamentos
- Pin 11: Recursos Humanos
- Pin 22: Contabilidad  
- Pin 33: Soporte Técnico
- Pin 44: Dirección General
"""

messages = [ ]
messages.append({"role": "user", "content": system_prompt})



while True:
    user_input = input("Tú: ").strip()
    if user_input.lower() == "salir":
        break
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        max_tokens=2048,
        messages=messages
    )

    reply = response.choices[0].message.content
    print(f"Ava: {reply}\n")

    messages.append({"role": "assistant", "content": reply})