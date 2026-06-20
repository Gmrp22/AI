# Llamar una API de AI con Python — De Cero a Avanzado

Esta guía usa **OpenAI SDK + Ollama** (gratis, local) para aprender los conceptos.
Cuando tengas créditos de Claude, al final de cada sección hay una nota con los cambios necesarios.

---

## Tabla de Contenido

1. [¿Qué es una API?](#1-qué-es-una-api)
2. [Setup del Ambiente](#2-setup-del-ambiente)
3. [Tu Primera Llamada](#3-tu-primera-llamada)
4. [Entendiendo la Respuesta](#4-entendiendo-la-respuesta)
5. [Conversaciones — Multi-Turn](#5-conversaciones--multi-turn)
6. [Streaming — Palabra por Palabra](#6-streaming--palabra-por-palabra)
7. [System Prompts — Darle un Rol al Modelo](#7-system-prompts--darle-un-rol-al-modelo)
8. [Tool Use — Darle Habilidades al Modelo](#8-tool-use--darle-habilidades-al-modelo)
9. [Structured Outputs — JSON Confiable](#9-structured-outputs--json-confiable)
10. [Vision — Mandar Imágenes](#10-vision--mandar-imágenes)
11. [Error Handling](#11-error-handling)
12. [OpenAI vs Claude — Diferencias Clave](#12-openai-vs-claude--diferencias-clave)

---

## 1. ¿Qué es una API?

Una API es una forma de que tu código hable con otro servicio por internet. Como ordenar en un restaurante:

- Tú (tu código) mandas un **pedido** (request) con parámetros específicos
- La cocina (Ollama/Claude) lo procesa
- El mesero trae tu **plato** (la respuesta)

Cada llamada manda datos a una URL y recibe datos de vuelta — normalmente en formato JSON (texto estructurado, como un diccionario).

**En nuestro caso:**
```
tu código → OpenAI SDK → Ollama (en tu Mac) → llama3.2 responde
```

Todo local, gratis, sin internet.

---

## 2. Setup del Ambiente

### Lo que necesitas tener instalado

```bash
# Verificar Python
python3 --version   # 3.9+

# Instalar paquetes en tu venv
.venv/bin/python3 -m pip install openai python-dotenv
```

### Estructura de tu proyecto

```
mi-proyecto/
├── .venv/
├── .env          ← credenciales (nunca en git)
└── API/
    └── basicCall.py
```

### Tu `.env`

```
BASE_URL=http://localhost:11434/v1
MODEL=llama3.2:3b
API_KEY=ollama
```

### Tener Ollama corriendo

```bash
ollama pull llama3.2:3b   # descargar el modelo (una vez)
ollama serve               # correr el servidor (o ya corre en background)
```

---

## 3. Tu Primera Llamada

```python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hola, ¿cómo estás?"}
    ]
)

print(response.choices[0].message.content)
```

Correr:
```bash
.venv/bin/python3 API/basicCall.py
```

### Qué hace cada parámetro

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `model` | string | Qué modelo usar |
| `max_tokens` | int | Máximo de tokens (palabras aprox.) en la respuesta |
| `messages` | list | La conversación. Cada item tiene `role` y `content` |

> **Con Claude:** `from anthropic import Anthropic` → `client.messages.create()` → `response.content[0].text`

---

## 4. Entendiendo la Respuesta

El objeto completo que devuelve la API:

```python
response = client.chat.completions.create(...)

# El texto que te importa
print(response.choices[0].message.content)

# Por qué paró el modelo
# "stop"       = terminó naturalmente
# "length"     = llegó al límite de max_tokens
# "tool_calls" = quiere llamar una función tuya
print(response.choices[0].finish_reason)

# Tokens usados (útil para calcular costo)
print(response.usage.prompt_tokens)      # tokens que mandaste
print(response.usage.completion_tokens)  # tokens que respondió
print(response.usage.total_tokens)       # total

# Qué modelo respondió
print(response.model)
```

### La estructura completa

```
ChatCompletion
  ├── id                    → ID único de este request
  ├── model                 → modelo que respondió
  ├── created               → timestamp
  ├── choices
  │     └── [0]
  │           ├── finish_reason  → por qué paró
  │           └── message
  │                 ├── role     → "assistant"
  │                 └── content  → 👈 EL TEXTO
  └── usage
        ├── prompt_tokens
        ├── completion_tokens
        └── total_tokens
```

> **Con Claude:** `response.stop_reason` en vez de `finish_reason`, `response.usage.input_tokens` y `output_tokens`

---

## 5. Conversaciones — Multi-Turn

El modelo no recuerda nada entre llamadas. Tú mandas toda la conversación cada vez.

```python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

# Iniciar conversación
messages = [
    {"role": "user", "content": "Me llamo Sofia."}
]

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    max_tokens=1024,
    messages=messages
)

reply = response.choices[0].message.content
print("Modelo:", reply)

# Agregar respuesta del modelo al historial
messages.append({"role": "assistant", "content": reply})

# Pregunta de seguimiento
messages.append({"role": "user", "content": "¿Cómo me llamo?"})

response2 = client.chat.completions.create(
    model=os.getenv("MODEL"),
    max_tokens=1024,
    messages=messages
)

print("Modelo:", response2.choices[0].message.content)
```

### Chat Loop — Conversación interactiva

```python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

messages = []
print("Chat con Ollama (escribe 'salir' para terminar)\n")

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
    print(f"Modelo: {reply}\n")

    messages.append({"role": "assistant", "content": reply})
```

> **Con Claude:** Idéntico, solo cambia cómo extraes el texto: `response.content[0].text`

---

## 6. Streaming — Palabra por Palabra

Sin streaming tu programa espera hasta que el modelo termina. Con streaming el texto aparece palabra por palabra.

```python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

stream = client.chat.completions.create(
    model=os.getenv("MODEL"),
    max_tokens=1024,
    stream=True,   # 👈 esto activa streaming
    messages=[
        {"role": "user", "content": "Escribe un poema corto sobre el mar."}
    ]
)

for chunk in stream:
    text = chunk.choices[0].delta.content
    if text:
        print(text, end="", flush=True)

print()  # salto de línea al final
```

### ¿Qué es `delta`?

En streaming no recibes la respuesta completa — recibes pedacitos (`chunks`). Cada chunk tiene un `delta` con el fragmento de texto nuevo:

```
chunk 1 → delta.content = "El"
chunk 2 → delta.content = " mar"
chunk 3 → delta.content = " es"
chunk 4 → delta.content = " azul"
...
```

> **Con Claude:** `with client.messages.stream(...) as stream: for text in stream.text_stream:`

---

## 7. System Prompts — Darle un Rol al Modelo

El system prompt le dice al modelo quién es y cómo debe comportarse. Va antes de los mensajes del usuario.

```python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    max_tokens=1024,
    messages=[
        {
            "role": "system",   # 👈 el system prompt va aquí
            "content": "Eres un chef mexicano llamado Don Tacho. Solo hablas de comida mexicana y usas dichos mexicanos en tus respuestas."
        },
        {
            "role": "user",
            "content": "¿Qué me recomiendas para cenar?"
        }
    ]
)

print(response.choices[0].message.content)
```

### Sistema de roles

```
"system"     → instrucciones para el modelo (no lo ve como "usuario")
"user"       → mensajes del usuario
"assistant"  → respuestas anteriores del modelo
```

### Ejemplo práctico — Asistente de soporte

```python
SYSTEM_PROMPT = """Eres un agente de soporte de TechCorp.
Sé útil, conciso y profesional.
Si no sabes algo, dilo — nunca inventes información.
No menciones precios ni hagas promesas."""

def preguntar_soporte(pregunta: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pregunta}
        ]
    )
    return response.choices[0].message.content

print(preguntar_soporte("¿Cómo reseteo mi contraseña?"))
```

> **Con Claude:** El system prompt va en un parámetro separado: `client.messages.create(system="...", messages=[...])`

---

## 8. Tool Use — Darle Habilidades al Modelo

Por defecto el modelo solo puede escribir texto. Tool use le permite llamar funciones que tú defines — buscar en internet, consultar una base de datos, hacer cálculos, etc.

### ¿Cómo funciona?

```
1. Le dices al modelo qué herramientas existen
2. El modelo decide usar una → devuelve tool_calls en vez de texto
3. Tú ejecutas la función real
4. Mandas el resultado de vuelta
5. El modelo usa el resultado para responder
```

### Ejemplo — Calculadora

```python
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

# Paso 1: Definir las herramientas
tools = [
    {
        "type": "function",
        "function": {
            "name": "calcular",
            "description": "Evalúa una expresión matemática y devuelve el resultado.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expresion": {
                        "type": "string",
                        "description": "La expresión a calcular, ej: '2 + 2' o '(10 * 3) / 2'"
                    }
                },
                "required": ["expresion"]
            }
        }
    }
]

# Paso 2: Tu función real
def calcular(expresion: str) -> float:
    return eval(expresion)

# Paso 3: Primera llamada
messages = [
    {"role": "user", "content": "¿Cuánto es (15 * 4) + (100 / 5)?"}
]

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    max_tokens=1024,
    tools=tools,
    messages=messages
)

# Paso 4: Loop hasta que el modelo termine
while response.choices[0].finish_reason == "tool_calls":
    tool_calls = response.choices[0].message.tool_calls
    
    # Agregar respuesta del modelo al historial
    messages.append(response.choices[0].message)
    
    # Ejecutar cada tool call
    for tool_call in tool_calls:
        nombre = tool_call.function.name
        argumentos = json.loads(tool_call.function.arguments)
        
        print(f"Modelo quiere llamar: {nombre}({argumentos})")
        
        if nombre == "calcular":
            resultado = calcular(argumentos["expresion"])
            print(f"Resultado: {resultado}")
        
        # Agregar resultado al historial
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(resultado)
        })
    
    # Nueva llamada con el resultado
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

print("\nModelo:", response.choices[0].message.content)
```

### Ejemplo más real — Clima y hora

```python
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

tools = [
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_hora",
            "description": "Devuelve la hora actual.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

def get_clima(ciudad: str) -> dict:
    # En producción llamarías una API real de clima
    return {"ciudad": ciudad, "temperatura": "22°C", "condicion": "Nublado"}

def get_hora() -> str:
    return datetime.now().strftime("%H:%M:%S")

def ejecutar_tool(nombre: str, argumentos: dict):
    if nombre == "get_clima":
        return json.dumps(get_clima(**argumentos))
    elif nombre == "get_hora":
        return get_hora()

def chat_con_tools(pregunta: str) -> str:
    messages = [{"role": "user", "content": pregunta}]
    
    while True:
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        if response.choices[0].finish_reason != "tool_calls":
            return response.choices[0].message.content
        
        messages.append(response.choices[0].message)
        
        for tool_call in response.choices[0].message.tool_calls:
            argumentos = json.loads(tool_call.function.arguments)
            resultado = ejecutar_tool(tool_call.function.name, argumentos)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": resultado
            })

print(chat_con_tools("¿Qué hora es y cómo está el clima en Ciudad de México?"))
```

> **Con Claude:** Los tools se definen sin el wrapper `"type": "function"`. El resultado se manda con `"role": "user"` y `"type": "tool_result"`

---

## 9. Structured Outputs — JSON Confiable

Cuando necesitas que el modelo devuelva datos en un formato específico (para una base de datos, un componente de UI, etc.).

```python
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

class ResenaLibro(BaseModel):
    titulo: str
    autor: str
    calificacion: int      # 1-5
    resumen: str
    recomendado: bool

# Pedir JSON estructurado
response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    max_tokens=1024,
    messages=[
        {
            "role": "system",
            "content": "Responde siempre en formato JSON válido siguiendo exactamente el esquema que se te pide."
        },
        {
            "role": "user",
            "content": f"Escribe una reseña del libro 'El Principito'. Devuelve un JSON con estos campos: {ResenaLibro.model_json_schema()}"
        }
    ]
)

import json
data = json.loads(response.choices[0].message.content)
resena = ResenaLibro(**data)

print(f"Título: {resena.titulo}")
print(f"Autor: {resena.autor}")
print(f"Calificación: {resena.calificacion}/5")
print(f"Recomendado: {resena.recomendado}")
```

> **Nota:** Con Ollama hay que pedirlo en el prompt explícitamente. Con la API oficial de OpenAI existe `response_format={"type": "json_object"}`. Con Claude existe `client.messages.parse()` que valida automáticamente.

---

## 10. Vision — Mandar Imágenes

Algunos modelos pueden entender imágenes. Con Ollama usa `llava` para esto.

```bash
ollama pull llava
```

### Imagen desde URL

```python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

response = client.chat.completions.create(
    model="llava",   # modelo con visión
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
                    }
                },
                {
                    "type": "text",
                    "text": "¿Qué ves en esta imagen?"
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

### Imagen desde archivo local

```python
import base64

def codificar_imagen(ruta: str) -> str:
    with open(ruta, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

imagen_base64 = codificar_imagen("./captura.png")

response = client.chat.completions.create(
    model="llava",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{imagen_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": "¿Hay texto en esta imagen? Si es así, transcríbelo."
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

---

## 11. Error Handling

Siempre envuelve tus llamadas en try/except para manejar errores de red, límites de rate, etc.

```python
from openai import OpenAI, RateLimitError, AuthenticationError, APIConnectionError
from dotenv import load_dotenv
import os
import time

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

def llamar_modelo(prompt: str, max_intentos: int = 3) -> str | None:
    for intento in range(max_intentos):
        try:
            response = client.chat.completions.create(
                model=os.getenv("MODEL"),
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        except RateLimitError:
            espera = 2 ** intento   # 1s, 2s, 4s
            print(f"Rate limit. Esperando {espera}s... (intento {intento + 1})")
            time.sleep(espera)

        except AuthenticationError:
            print("API key inválida.")
            return None

        except APIConnectionError:
            print(f"Error de conexión. Reintentando... ({intento + 1}/{max_intentos})")
            time.sleep(2)

        except Exception as e:
            print(f"Error inesperado: {e}")
            return None

    print("Máximo de intentos alcanzado.")
    return None

resultado = llamar_modelo("¿Cuál es la capital de Francia?")
if resultado:
    print(resultado)
```

### Errores comunes

| Error | Causa | Solución |
|-------|-------|----------|
| `AuthenticationError` | API key incorrecta | Verificar `.env` |
| `RateLimitError` | Demasiadas requests | Exponential backoff |
| `APIConnectionError` | Sin conexión a Ollama | Verificar `ollama serve` |
| `NotFoundError` | Modelo no existe | Verificar `ollama list` |

---

## 12. OpenAI vs Claude — Diferencias Clave

Cuando tengas créditos de Claude, estos son los cambios:

```python
# OPENAI / OLLAMA
from openai import OpenAI
client = OpenAI(base_url="...", api_key="ollama")

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[
        {"role": "system", "content": "Eres un asistente."},
        {"role": "user", "content": "Hola"}
    ]
)
texto = response.choices[0].message.content
razon = response.choices[0].finish_reason


# ANTHROPIC / CLAUDE
from anthropic import Anthropic
client = Anthropic()   # lee ANTHROPIC_API_KEY automáticamente

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    system="Eres un asistente.",   # 👈 parámetro separado
    messages=[
        {"role": "user", "content": "Hola"}
    ]
)
texto = response.content[0].text   # 👈 diferente
razon = response.stop_reason       # 👈 diferente
```

### Tabla de diferencias

| Concepto | OpenAI/Ollama | Anthropic/Claude |
|----------|---------------|------------------|
| Import | `from openai import OpenAI` | `from anthropic import Anthropic` |
| Client | `OpenAI(base_url, api_key)` | `Anthropic()` |
| Llamada | `chat.completions.create()` | `messages.create()` |
| System | dentro de `messages` con `role: "system"` | parámetro `system=""` separado |
| Texto | `choices[0].message.content` | `content[0].text` |
| Stop | `finish_reason` | `stop_reason` |
| Tools | `tool_calls` | `tool_use` |
| Tool result | `role: "tool"` | `role: "user"` con `type: "tool_result"` |
| Streaming | `stream=True` + `delta.content` | `client.messages.stream()` |
| Tokens | `prompt_tokens` / `completion_tokens` | `input_tokens` / `output_tokens` |
| Features exclusivas | — | Caching, Thinking, Batch |

---

## Cheat Sheet

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))
MODEL = os.getenv("MODEL")

# Llamada básica
r = client.chat.completions.create(model=MODEL, max_tokens=1024,
    messages=[{"role": "user", "content": "Hola"}])
print(r.choices[0].message.content)

# Con system prompt
r = client.chat.completions.create(model=MODEL, max_tokens=1024,
    messages=[
        {"role": "system", "content": "Eres un asistente."},
        {"role": "user", "content": "Hola"}
    ])

# Streaming
stream = client.chat.completions.create(model=MODEL, max_tokens=1024,
    stream=True, messages=[{"role": "user", "content": "Cuenta hasta 5"}])
for chunk in stream:
    text = chunk.choices[0].delta.content
    if text:
        print(text, end="", flush=True)

# Multi-turn
messages = []
messages.append({"role": "user", "content": "Hola"})
r = client.chat.completions.create(model=MODEL, max_tokens=1024, messages=messages)
messages.append({"role": "assistant", "content": r.choices[0].message.content})
```

---

## Próximos Pasos

Una vez que domines todo esto:

1. **Consigue créditos de Claude** — aplica los cambios de la sección 12
2. **Aprende LangChain** — todo lo que aprendiste aquí aplica
3. **Embeddings** — el siguiente concepto importante para RAG y búsqueda semántica
4. **Agentes** — combinar tool use + multi-turn para tareas complejas
