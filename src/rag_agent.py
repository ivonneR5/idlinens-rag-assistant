
from pathlib import Path
from collections import defaultdict
import math
import re
import time
import unicodedata

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# ============================================================
# Configuración general
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = BASE_DIR / "vector_store"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Gemma 3 4B ofrece mejor capacidad para comparar fragmentos,
# distinguir procesos similares y reconstruir procedimientos.
OLLAMA_MODEL = "gemma3:4b"

# Recuperación híbrida.
SEMANTIC_CANDIDATES = 24
LEXICAL_CANDIDATES = 24
MAX_RETRIEVED_DOCUMENTS = 6
MAX_CONTEXT_CHARACTERS = 8000
MAX_CHARS_PER_DOCUMENT = 1600

# Reciprocal Rank Fusion.
RRF_K = 60

# Diagnóstico.
MAX_SOURCES_TO_DISPLAY = 4
SHOW_RETRIEVED_SOURCES_IN_TERMINAL = True
SHOW_PERFORMANCE_TIMES = True

# Caché simple en memoria.
ENABLE_ANSWER_CACHE = True
MAX_CACHE_ENTRIES = 100
ANSWER_CACHE = {}

NO_INFORMATION_BASE = (
    "No encontré esa información en la documentación disponible."
)

NO_INFORMATION_MESSAGE = (
    f"{NO_INFORMATION_BASE} "
    "Por favor, ponte en contacto con el equipo de soporte "
    "por correo electrónico."
)

STOPWORDS = {
    "a", "al", "algo", "como", "con", "cual", "cuando", "de", "del",
    "desde", "donde", "el", "ella", "en", "es", "esta", "este", "hacer",
    "hago", "hay", "la", "las", "lo", "los", "me", "mi", "mis", "para",
    "por", "que", "se", "si", "sin", "su", "sus", "un", "una", "unas",
    "unos", "ver", "y",
}


# ============================================================
# Utilidades de texto
# ============================================================

def normalize_text(text):
    """Normaliza texto para búsquedas y comparaciones."""

    normalized = unicodedata.normalize(
        "NFKD",
        str(text).strip().lower(),
    )

    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )

    normalized = re.sub(
        r"[^a-z0-9\s_-]",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def tokenize(text):
    """Obtiene términos informativos para la búsqueda léxica."""

    return [
        token
        for token in normalize_text(text).split()
        if len(token) > 2 and token not in STOPWORDS
    ]


def clean_document_text(text):
    """Limpia espacios y saltos sin destruir listas o títulos."""

    clean_text = re.sub(
        r"\r\n?",
        "\n",
        str(text),
    ).strip()

    clean_text = re.sub(
        r"[ \t]+",
        " ",
        clean_text,
    )

    clean_text = re.sub(
        r"\n{3,}",
        "\n\n",
        clean_text,
    )

    return clean_text


def answer_indicates_no_information(answer):
    """Detecta respuestas equivalentes a falta de información."""

    normalized_answer = normalize_text(answer)

    variants = [
        "no encontre esa informacion en la documentacion disponible",
        "no encontre informacion en la documentacion disponible",
        "no se encontro esa informacion en la documentacion disponible",
        "la documentacion disponible no contiene esa informacion",
        "no hay informacion suficiente en el contexto",
        "el contexto no contiene informacion suficiente",
        "no encontre informacion relevante",
        "no se proporciona informacion suficiente",
    ]

    return any(
        variant in normalized_answer
        for variant in variants
    )


# ============================================================
# Carga de recursos
# ============================================================

def load_embeddings():
    """Carga el modelo de embeddings usado al crear FAISS."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vector_store(embeddings):
    """Carga la base vectorial generada previamente."""

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
    """Inicializa Gemma 3 4B mediante Ollama."""

    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        num_predict=420,
        num_ctx=8192,
        num_thread=4,
        keep_alive="60m",
    )


# ============================================================
# Acceso a documentos y metadatos
# ============================================================

def get_all_documents(vector_store):
    """Devuelve todos los documentos almacenados en FAISS."""

    docstore = getattr(vector_store, "docstore", None)
    document_dict = getattr(docstore, "_dict", {})

    if not isinstance(document_dict, dict):
        return []

    return list(document_dict.values())


def document_key(document):
    """Genera una clave estable para deduplicar fragmentos."""

    file_name = document.metadata.get("file_name", "")
    page = document.metadata.get("page")
    chunk_index = document.metadata.get("chunk_index")
    content = normalize_text(document.page_content)[:400]

    return (
        file_name,
        page,
        chunk_index,
        content,
    )


def get_page_number(document):
    """Obtiene el número de página visible para el usuario."""

    page = document.metadata.get("page")

    if isinstance(page, int):
        return page + 1

    return None


# ============================================================
# Búsqueda léxica
# ============================================================

def lexical_score(question, document):
    """
    Calcula relevancia por coincidencia de términos.

    Premia:
    - cobertura de palabras de la pregunta;
    - frases consecutivas;
    - apariciones en las primeras líneas, donde suelen estar
      títulos y nombres de módulos.
    """

    query_tokens = tokenize(question)

    if not query_tokens:
        return 0.0

    normalized_content = normalize_text(document.page_content)
    content_tokens = set(normalized_content.split())

    matched = [
        token
        for token in query_tokens
        if token in content_tokens
    ]

    if not matched:
        return 0.0

    coverage = len(set(matched)) / len(set(query_tokens))

    term_frequency = sum(
        min(normalized_content.count(token), 4)
        for token in set(matched)
    ) / max(len(set(query_tokens)), 1)

    normalized_question = normalize_text(question)
    phrase_bonus = 0.0

    if normalized_question and normalized_question in normalized_content:
        phrase_bonus = 2.0

    first_part = normalized_content[:500]
    heading_bonus = sum(
        1.0
        for token in set(matched)
        if token in first_part
    ) / max(len(set(query_tokens)), 1)

    return (
        coverage * 5.0
        + term_frequency * 0.7
        + heading_bonus * 1.5
        + phrase_bonus
    )


def lexical_search(vector_store, question, k):
    """Busca los mejores fragmentos por coincidencia textual."""

    scored = []

    for document in get_all_documents(vector_store):
        score = lexical_score(question, document)

        if score > 0:
            scored.append((document, score))

    scored.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored[:k]


# ============================================================
# Recuperación híbrida y reranking
# ============================================================

def semantic_search(vector_store, question, k):
    """Realiza la búsqueda semántica en FAISS."""

    results = vector_store.similarity_search_with_score(
        query=question,
        k=k,
    )

    return [
        (document, float(distance))
        for document, distance in results
    ]


def reciprocal_rank_fusion(semantic_results, lexical_results):
    """
    Fusiona los rankings semántico y léxico con RRF.

    No depende de escalas incompatibles entre distancia FAISS
    y puntuación léxica.
    """

    fused = {}

    for rank, (document, distance) in enumerate(
        semantic_results,
        start=1,
    ):
        key = document_key(document)

        if key not in fused:
            fused[key] = {
                "document": document,
                "rrf_score": 0.0,
                "semantic_distance": distance,
                "lexical_score": 0.0,
                "semantic_rank": None,
                "lexical_rank": None,
            }

        fused[key]["rrf_score"] += 1.0 / (RRF_K + rank)
        fused[key]["semantic_rank"] = rank
        fused[key]["semantic_distance"] = distance

    for rank, (document, score) in enumerate(
        lexical_results,
        start=1,
    ):
        key = document_key(document)

        if key not in fused:
            fused[key] = {
                "document": document,
                "rrf_score": 0.0,
                "semantic_distance": None,
                "lexical_score": score,
                "semantic_rank": None,
                "lexical_rank": None,
            }

        fused[key]["rrf_score"] += 1.0 / (RRF_K + rank)
        fused[key]["lexical_rank"] = rank
        fused[key]["lexical_score"] = score

    return list(fused.values())


def rerank_candidates(question, candidates):
    """
    Aplica un reranking final general.

    Combina RRF, cobertura de términos y señales de procedimiento.
    No contiene respuestas ni reglas para preguntas específicas.
    """

    query_tokens = set(tokenize(question))
    asks_for_procedure = bool(
        re.search(
            r"\b(como|pasos|procedimiento|crear|registrar|cambiar|"
            r"configurar|consultar|localizar|enviar|regresar|recibir|"
            r"agregar|editar|eliminar)\b",
            normalize_text(question),
        )
    )

    ranked = []

    for candidate in candidates:
        document = candidate["document"]
        normalized_content = normalize_text(
            document.page_content
        )
        content_tokens = set(
            normalized_content.split()
        )

        overlap = (
            len(query_tokens & content_tokens)
            / max(len(query_tokens), 1)
        )

        procedure_bonus = 0.0

        if asks_for_procedure:
            procedure_markers = [
                "paso",
                "procedimiento",
                "seleccione",
                "presione",
                "ingrese",
                "abra",
                "ajuste",
                "confirme",
                "guardar",
                "iniciar",
                "terminar",
            ]

            marker_hits = sum(
                1
                for marker in procedure_markers
                if marker in normalized_content
            )

            numbered_steps = len(
                re.findall(
                    r"(?:^|\n)\s*\d+[\.\)]\s+",
                    str(document.page_content),
                )
            )

            procedure_bonus = (
                min(marker_hits, 5) * 0.002
                + min(numbered_steps, 5) * 0.003
            )

        final_score = (
            candidate["rrf_score"]
            + overlap * 0.025
            + procedure_bonus
        )

        document.metadata["hybrid_score"] = final_score
        document.metadata["semantic_rank"] = (
            candidate["semantic_rank"]
        )
        document.metadata["lexical_rank"] = (
            candidate["lexical_rank"]
        )
        document.metadata["semantic_distance"] = (
            candidate["semantic_distance"]
        )
        document.metadata["lexical_score"] = (
            candidate["lexical_score"]
        )

        ranked.append(
            (document, final_score)
        )

    ranked.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked


def select_diverse_documents(ranked_documents):
    """
    Selecciona resultados relevantes sin llenar el contexto con
    fragmentos casi idénticos.

    Permite varias páginas del mismo manual cuando aportan
    información diferente.
    """

    selected = []
    seen_keys = set()
    per_page = defaultdict(int)

    for document, _score in ranked_documents:
        key = document_key(document)

        if key in seen_keys:
            continue

        file_name = document.metadata.get(
            "file_name",
            "",
        )
        page = document.metadata.get("page")
        page_key = (file_name, page)

        if per_page[page_key] >= 2:
            continue

        selected.append(document)
        seen_keys.add(key)
        per_page[page_key] += 1

        if len(selected) >= MAX_RETRIEVED_DOCUMENTS:
            break

    return selected


def retrieve_documents(vector_store, question):
    """Ejecuta búsqueda semántica, léxica, fusión y reranking."""

    semantic_results = semantic_search(
        vector_store=vector_store,
        question=question,
        k=SEMANTIC_CANDIDATES,
    )

    lexical_results = lexical_search(
        vector_store=vector_store,
        question=question,
        k=LEXICAL_CANDIDATES,
    )

    fused_candidates = reciprocal_rank_fusion(
        semantic_results=semantic_results,
        lexical_results=lexical_results,
    )

    ranked_documents = rerank_candidates(
        question=question,
        candidates=fused_candidates,
    )

    return select_diverse_documents(
        ranked_documents
    )


# ============================================================
# Construcción del contexto
# ============================================================

def build_context(documents):
    """
    Construye un contexto ordenado por relevancia.

    Conserva el orden del reranking; no reordena por página.
    """

    context_parts = []
    total_characters = 0

    for index, document in enumerate(
        documents,
        start=1,
    ):
        file_name = document.metadata.get(
            "file_name",
            "Documento desconocido",
        )

        page = get_page_number(document)

        page_text = (
            str(page)
            if page is not None
            else "no disponible"
        )

        content = clean_document_text(
            document.page_content
        )[:MAX_CHARS_PER_DOCUMENT]

        source_block = (
            f"[FRAGMENTO {index} | "
            f"Documento: {file_name} | "
            f"Página: {page_text}]\n"
            f"{content}"
        )

        if (
            total_characters
            + len(source_block)
            > MAX_CONTEXT_CHARACTERS
        ):
            remaining = (
                MAX_CONTEXT_CHARACTERS
                - total_characters
            )

            if remaining > 300:
                context_parts.append(
                    source_block[:remaining]
                )

            break

        context_parts.append(source_block)
        total_characters += len(source_block)

    return "\n\n".join(context_parts)


# ============================================================
# Prompt
# ============================================================

def build_prompt(question, context):
    """Construye un prompt orientado a exactitud documental."""

    return f"""
Eres un asistente empresarial especializado en IDLinens HA.

Tu única fuente de verdad es el CONTEXTO recuperado de los
manuales oficiales. No utilices conocimientos externos.

TAREA:
Responde exactamente la pregunta del usuario usando la evidencia
más relevante del contexto.

REGLAS OBLIGATORIAS:

1. Responde siempre en español.
2. Primero identifica qué módulo, función u operación pregunta
   el usuario.
3. Distingue cuidadosamente operaciones con nombres parecidos.
4. No mezcles pasos de módulos diferentes.
5. Si el usuario pregunta cómo realizar una operación:
   - busca en todos los fragmentos acciones, botones y pasos;
   - integra la información complementaria;
   - presenta una lista numerada completa;
   - conserva exactamente los nombres de botones, módulos,
     pestañas y campos que aparezcan en el contexto.
6. No respondas solo con el objetivo, descripción, recomendación
   o momento de uso cuando existan instrucciones operativas.
7. Si el contexto únicamente contiene una descripción y no
   contiene pasos verificables, explica solo lo que sí está
   documentado. No inventes el procedimiento.
8. No inventes botones, campos, requisitos, permisos, resultados
   ni pasos.
9. No agregues fuentes; el sistema las añadirá automáticamente.
10. Si ningún fragmento contiene información suficiente para
    responder, escribe exactamente:

"{NO_INFORMATION_BASE}"

CONTEXTO:

{context}

PREGUNTA DEL USUARIO:

{question}

RESPUESTA:
""".strip()


def build_retry_prompt(question, context):
    """Segundo intento, más estricto, cuando Gemma rechaza contexto útil."""

    return f"""
Analiza nuevamente el contexto y responde únicamente con hechos
que aparezcan literalmente o puedan integrarse directamente de
los fragmentos.

Pregunta:
{question}

Reglas:
- Responde en español.
- No mezcles módulos distintos.
- Para procedimientos, usa una lista numerada.
- Conserva los nombres exactos del contexto.
- No inventes información.
- No agregues fuentes.

Si no existe información suficiente, responde exactamente:
"{NO_INFORMATION_BASE}"

Contexto:
{context}

Respuesta:
""".strip()


# ============================================================
# Generación de respuestas
# ============================================================

def generate_answer(model, prompt):
    """Envía el prompt a Gemma y devuelve texto limpio."""

    response = model.invoke(prompt)
    content = getattr(response, "content", "")
    return str(content).strip()


# ============================================================
# Fuentes y diagnóstico
# ============================================================

def format_sources(documents):
    """Genera la lista de fuentes realmente enviadas al modelo."""

    sources = []
    seen = set()

    for document in documents:
        file_name = document.metadata.get(
            "file_name",
            "Documento desconocido",
        )
        page = get_page_number(document)
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

        if len(sources) >= MAX_SOURCES_TO_DISPLAY:
            break

    if not sources:
        return ""

    return (
        "\n\n### Fuentes consultadas\n"
        + "\n".join(sources)
    )


def show_retrieved_sources(documents):
    """Muestra el resultado del reranking en la terminal."""

    if not SHOW_RETRIEVED_SOURCES_IN_TERMINAL:
        return

    print("\nFragmentos recuperados y reordenados:")

    for index, document in enumerate(
        documents,
        start=1,
    ):
        file_name = document.metadata.get(
            "file_name",
            "Documento desconocido",
        )
        page = get_page_number(document)
        hybrid_score = document.metadata.get(
            "hybrid_score"
        )
        semantic_rank = document.metadata.get(
            "semantic_rank"
        )
        lexical_rank = document.metadata.get(
            "lexical_rank"
        )

        page_text = (
            str(page)
            if page is not None
            else "no disponible"
        )

        score_text = (
            f"{hybrid_score:.4f}"
            if isinstance(hybrid_score, float)
            else "n/d"
        )

        print(
            f"{index}. {file_name} - página {page_text} "
            f"| híbrido: {score_text} "
            f"| semántico: {semantic_rank} "
            f"| léxico: {lexical_rank}"
        )


# ============================================================
# Caché
# ============================================================

def get_cached_answer(question):
    """Obtiene una respuesta guardada para una pregunta repetida."""

    if not ENABLE_ANSWER_CACHE:
        return None

    return ANSWER_CACHE.get(
        normalize_text(question)
    )


def save_cached_answer(question, answer):
    """Guarda una respuesta y limita el tamaño de la caché."""

    if not ENABLE_ANSWER_CACHE:
        return

    cache_key = normalize_text(question)

    if len(ANSWER_CACHE) >= MAX_CACHE_ENTRIES:
        oldest_key = next(iter(ANSWER_CACHE))
        ANSWER_CACHE.pop(oldest_key, None)

    ANSWER_CACHE[cache_key] = answer


# ============================================================
# Flujo principal del agente
# ============================================================

def answer_question(
    vector_store,
    model,
    question,
):
    """Ejecuta recuperación híbrida, reranking y generación."""

    total_start = time.perf_counter()
    clean_question = str(question).strip()

    if not clean_question:
        return "Debes escribir una pregunta."

    cached_answer = get_cached_answer(
        clean_question
    )

    if cached_answer is not None:
        if SHOW_PERFORMANCE_TIMES:
            print(
                "\nRespuesta recuperada desde caché."
            )

        return cached_answer

    retrieval_start = time.perf_counter()

    documents = retrieve_documents(
        vector_store=vector_store,
        question=clean_question,
    )

    retrieval_seconds = (
        time.perf_counter() - retrieval_start
    )

    if not documents:
        return NO_INFORMATION_MESSAGE

    show_retrieved_sources(documents)

    context_start = time.perf_counter()
    context = build_context(documents)
    context_seconds = (
        time.perf_counter() - context_start
    )

    prompt = build_prompt(
        question=clean_question,
        context=context,
    )

    generation_start = time.perf_counter()

    answer = generate_answer(
        model=model,
        prompt=prompt,
    )

    if answer_indicates_no_information(answer):
        retry_documents = documents[:4]
        retry_context = build_context(
            retry_documents
        )
        retry_prompt = build_retry_prompt(
            question=clean_question,
            context=retry_context,
        )
        retry_answer = generate_answer(
            model=model,
            prompt=retry_prompt,
        )

        if not answer_indicates_no_information(
            retry_answer
        ):
            answer = retry_answer
            documents = retry_documents

    generation_seconds = (
        time.perf_counter() - generation_start
    )

    if answer_indicates_no_information(answer):
        final_answer = NO_INFORMATION_MESSAGE
    else:
        final_answer = (
            answer.strip()
            + format_sources(documents)
        )

    save_cached_answer(
        question=clean_question,
        answer=final_answer,
    )

    if SHOW_PERFORMANCE_TIMES:
        total_seconds = (
            time.perf_counter() - total_start
        )

        print("\nTiempos de ejecución:")
        print(
            f"- Recuperación híbrida: "
            f"{retrieval_seconds:.2f} s"
        )
        print(
            f"- Preparación del contexto: "
            f"{context_seconds:.2f} s"
        )
        print(
            f"- Generación con Gemma: "
            f"{generation_seconds:.2f} s"
        )
        print(
            f"- Tiempo total: "
            f"{total_seconds:.2f} s"
        )
        print(
            f"- Caracteres del contexto: "
            f"{len(context)}"
        )

    return final_answer


# ============================================================
# Ejecución desde terminal
# ============================================================

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
        vector_store = load_vector_store(
            embeddings
        )

        print(
            "Conectando con Gemma 3 "
            "mediante Ollama..."
        )
        model = load_language_model()

        print("\nAgente listo.\n")

        while True:
            question = input(
                "Tu pregunta:\n> "
            ).strip()

            if not question:
                print(
                    "\nDebes escribir una pregunta.\n"
                )
                continue

            if normalize_text(question) in {
                "salir",
                "exit",
                "quit",
            }:
                print("\nAgente finalizado.")
                break

            print(
                "\nBuscando información "
                "y generando respuesta...\n"
            )

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

    except KeyboardInterrupt:
        print("\n\nAgente finalizado.")

    except Exception as error:
        print(
            "\nOcurrió un error "
            "al ejecutar el agente."
        )
        print(f"Detalle: {error}")


if __name__ == "__main__":
    main()