from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = BASE_DIR / "vector_store"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

OLLAMA_MODEL = "gemma3:4b"



def load_embeddings():
    """Carga el mismo modelo utilizado para crear el índice vectorial."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vector_store(embeddings):
    """Carga la base vectorial FAISS generada previamente."""

    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            "No se encontró la base vectorial. "
            "Ejecuta primero: python src/create_vector_store.py"
        )

    return FAISS.load_local(
        folder_path=str(VECTOR_STORE_PATH),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def load_language_model():
    """Inicializa Gemma 3 mediante Ollama."""

    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )


def retrieve_documents(vector_store, question):
    """Recupera fragmentos relevantes y evita resultados repetitivos."""

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 12,
            "lambda_mult": 0.7,
        },
    )

    return retriever.invoke(question)


def build_context(documents):
    """Construye el contexto que será enviado al modelo."""

    context_parts = []

    for index, document in enumerate(documents, start=1):
        file_name = document.metadata.get(
            "file_name",
            "Documento desconocido",
        )

        page = document.metadata.get("page")

        if isinstance(page, int):
            page += 1

        source_header = (
            f"[Fuente {index}: {file_name}, "
            f"página {page if page is not None else 'no disponible'}]"
        )

        context_parts.append(
            f"{source_header}\n{document.page_content.strip()}"
        )

    return "\n\n".join(context_parts)


def build_prompt(question, context):
    """Construye instrucciones claras para responder solo con el contexto."""

    return f"""
Eres un asistente especializado en el sistema IDLinens HA.

Debes responder la pregunta utilizando exclusivamente la información incluida
en el contexto recuperado de los manuales.

INSTRUCCIONES OBLIGATORIAS:

1. Responde siempre en español.
2. Analiza todos los fragmentos del contexto antes de responder.
3. Si al menos un fragmento contiene la respuesta, utiliza esa información.
4. Cuando la pregunta solicite un procedimiento, presenta los pasos en orden.
5. No sustituyas el procedimiento solicitado por otro parecido.
6. No inventes funciones, botones, campos, requisitos ni resultados.
7. Si el contexto no contiene información útil, responde únicamente:
   "No encontré esa información en la documentación disponible."
8. Nunca combines el mensaje anterior con una respuesta parcial.
9. No escribas ni inventes una sección de fuentes.

CONTEXTO RECUPERADO:

{context}

PREGUNTA:

{question}

RESPUESTA:
""".strip()


def generate_answer(model, prompt):
    """Envía el contexto y la pregunta a Gemma."""

    response = model.invoke(prompt)

    return response.content.strip()

def format_sources(documents):
    """Genera una lista de fuentes reales sin duplicados."""

    sources = []
    seen = set()

    for document in documents:
        file_name = document.metadata.get(
            "file_name",
            "Documento desconocido",
        )

        page = document.metadata.get("page")

        if isinstance(page, int):
            page += 1

        source_key = (file_name, page)

        if source_key in seen:
            continue

        seen.add(source_key)

        page_text = (
            str(page)
            if page is not None
            else "no disponible"
        )

        sources.append(
            f"- {file_name}, página {page_text}"
        )

    if not sources:
        return ""

    return "\n\nFuentes consultadas:\n" + "\n".join(sources)


def show_retrieved_sources(documents):
    """Muestra las fuentes recuperadas por FAISS."""

    print("\nFragmentos recuperados:")

    for index, document in enumerate(documents, start=1):
        file_name = document.metadata.get(
            "file_name",
            "Documento desconocido",
        )

        page = document.metadata.get("page")

        if isinstance(page, int):
            page += 1

        print(
            f"{index}. {file_name} - "
            f"página {page if page is not None else 'no disponible'}"
        )


def answer_question(vector_store, model, question):
    """Ejecuta el flujo completo de recuperación y generación."""

    documents = retrieve_documents(
        vector_store=vector_store,
        question=question,
    )

    if not documents:
        return (
            "No encontré información relacionada con la pregunta "
            "en la documentación disponible. Por favor pongase en contacto con nosostrso via correo electronico"
        )

    context = build_context(documents)
    prompt = build_prompt(question, context)

    answer = generate_answer(model, prompt)
    sources = format_sources(documents)

    show_retrieved_sources(documents)

    no_information_message = (
        "No encontré esa información en la documentación disponible, por favor pongase en contacto con nosostrso via correo electronico."
    )

    if answer.strip() == no_information_message:
        return no_information_message

    return answer + sources


def main():
    try:
        print("\n" + "=" * 70)
        print("IDLINENS RAG ASSISTANT")
        print("=" * 70)
        print("Escribe una pregunta sobre IDLinens HA.")
        print("Escribe 'salir' para finalizar.\n")

        print("Cargando modelo de embeddings...")
        embeddings = load_embeddings()

        print("Cargando base vectorial...")
        vector_store = load_vector_store(embeddings)

        print("Conectando con Gemma 3 mediante Ollama...")
        model = load_language_model()

        print("\nAgente listo.\n")

        while True:
            question = input("Tu pregunta:\n> ").strip()

            if not question:
                print("\nDebes escribir una pregunta.\n")
                continue

            if question.lower() in {"salir", "exit", "quit"}:
                print("\nAgente finalizado.")
                break

            print("\nBuscando información y generando respuesta...\n")

            answer = answer_question(
                vector_store=vector_store,
                model=model,
                question=question,
            )

            print("\n" + "=" * 70)
            print("RESPUESTA")
            print("=" * 70)
            print(answer)
            print("=" * 70 + "\n")

    except FileNotFoundError as error:
        print(f"\nError: {error}")

    except ConnectionError:
        print(
            "\nNo fue posible conectarse con Ollama. "
            "Verifica que Ollama se encuentre abierto."
        )

    except KeyboardInterrupt:
        print("\n\nAgente finalizado.")

    except Exception as error:
        print("\nOcurrió un error al ejecutar el agente.")
        print(f"Detalle: {error}")


if __name__ == "__main__":
    main()