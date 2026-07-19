# IDLinens RAG Assistant

## Descripción

IDLinens RAG Assistant es un asistente inteligente basado en Retrieval-Augmented Generation (RAG) desarrollado como proyecto final del programa Oracle Next Education (ONE) + Alura.

El asistente responde preguntas sobre la plataforma hospitalaria **IDLinens HA** utilizando únicamente la documentación oficial del sistema, reduciendo las alucinaciones del modelo y proporcionando referencias de las fuentes consultadas.

---

# Objetivo

Desarrollar un agente de inteligencia artificial capaz de comprender preguntas en lenguaje natural y responder utilizando exclusivamente la documentación técnica y de usuario de IDLinens HA.

---

# Características

- Búsqueda semántica mediante embeddings.
- Base vectorial FAISS.
- Generación de respuestas con Gemma 3 ejecutándose localmente mediante Ollama.
- Interfaz web desarrollada con Streamlit.
- Citas automáticas de la documentación utilizada.
- Compatible con documentación en español.
- Funcionamiento sin depender de APIs comerciales.

---

# Arquitectura

Documentos PDF

↓

Carga y procesamiento

↓

Fragmentación (Chunking)

↓

Embeddings (Sentence Transformers)

↓

Base Vectorial FAISS

↓

Recuperación de información

↓

Gemma 3 (Ollama)

↓

Respuesta al usuario

---

# Base documental

El asistente utiliza la siguiente documentación oficial:

- Manual Operativo IDLinens HA
- Manual de Usuario App Android
- Manual de Usuario Dashboard

Total procesado:

- 3 documentos PDF
- 137 páginas
- 288 fragmentos indexados

---

# Tecnologías

- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- Sentence Transformers
- Ollama
- Gemma 3
- Streamlit

---

# Instalación

Clonar el repositorio

```bash
git clone https://github.com/ivonneR5/idlinens-rag-assistant.git
```

Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Crear la base vectorial

```bash
python src/chunk_documents.py

python src/create_vector_store.py
```

---

# Ejecutar la aplicación

```bash
streamlit run src/app.py
```

---

# Ejemplos de preguntas

- ¿Cómo registro una nueva prenda?
- ¿Cómo envío prendas a lavandería?
- ¿Cómo cambio la potencia de la lectora?
- ¿Cómo creo un usuario en el Dashboard?
- ¿Cómo consulto el historial de un activo?

---

# Estructura del proyecto

```
docs/
src/
vector_store/
tests/
requirements.txt
README.md
LICENSE
```

---

# Limitaciones

El tiempo de respuesta depende de los recursos de hardware disponibles.

Cuando el asistente se ejecuta en servidores con CPU y memoria limitadas (por ejemplo Oracle Cloud Free Tier), la generación de respuestas puede tardar algunos segundos adicionales debido a la ejecución local del modelo Gemma 3 mediante Ollama.

---

# Mejoras futuras

- Re-ranking de resultados.
- Actualización incremental de la base vectorial.
- Búsqueda híbrida (semántica + palabras clave).
- OCR para documentos escaneados.
- Soporte para múltiples hospitales.

---

# Autor

Ivonne Negrete

Proyecto desarrollado para Oracle Next Education (ONE) + Alura.