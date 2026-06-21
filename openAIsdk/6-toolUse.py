from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import requests

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)
system_prompt = """
Eres Ava, asistente de clima. Responde SIEMPRE en español. Respuestas cortas.

Cuando el usuario inicie la conversacion, preséntate usando la siguietne frase:  "Hola soy Ava" y di que puedes consultar el clima de cualquier ciudad.

Cuando el usuario mencione una ciudad, llama la tool `get_clima` con esa ciudad.
Si el usuario quiere hablar con un humano o agente, llama la tool `transfer_call`.
"""



tools = [
    {
        "type": "function",
        "function": {
            "name": "transfer_call",
            "description": "Transferir la llamada a un humano o agente cuand el usuario lo solicite",
            "parameters": {
                "type": "object",
                "properties": {
                    "transfer_reason": {
                        "type": "string",
                        "description": "Motivo por el cual se transfiere la llamada"
                    }
                },
                "required": ["transfer_reason"]
            }
        }
    },
     {
        "type": "function",
        "function": {
            "name": "get_clima",
            "description": "Obtiene el clima actual de una ciudad.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ciudad": {"type": "string"}
                },
                "required": ["ciudad"]
            }
        }
    }
]

def get_clima(ciudad: str) -> str:
    # Geocoding: ciudad → coordenadas
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": ciudad, "count": 1, "language": "es"}
    ).json()

    if not geo.get("results"):
        return json.dumps({"error": f"Ciudad '{ciudad}' no encontrada"})

    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]

    # Clima actual
    clima = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weathercode,windspeed_10m",
            "timezone": "auto"
        }
    ).json()

    current = clima["current"]
    return json.dumps({
        "ciudad": ciudad,
        "temperatura": f"{current['temperature_2m']}°C",
        "viento": f"{current['windspeed_10m']} km/h",
    })


messages = [{"role": "system", "content": system_prompt}]

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
        tools=tools,
        messages=messages
    )

    choice = response.choices[0]
    messages.append(choice.message)

    if choice.finish_reason == "tool_calls":
        for tool_call in choice.message.tool_calls:
            nombre = tool_call.function.name
            print(f"[tool] llamando: {nombre}")

            args = json.loads(tool_call.function.arguments)
            print(f"[args] {args}")
            if nombre == "get_clima":
                resultado = get_clima(args["ciudad"])
            elif nombre == "transfer_call":
                resultado = json.dumps({"status": "transferido", "motivo": args["transfer_reason"]})
            else:
                resultado = json.dumps({"error": f"tool {nombre} no implementada"})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": resultado
            })

        # Segunda llamada para que el modelo responda con el resultado
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            max_tokens=2048,
            tools=tools,
            messages=messages
        )
        reply = response.choices[0].message.content
    else:
        reply = choice.message.content

    print(f"Ava: {reply}\n")
    messages.append({"role": "assistant", "content": reply})