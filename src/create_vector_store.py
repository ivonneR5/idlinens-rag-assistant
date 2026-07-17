from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PATH = BASE_DIR / "docs"
VECTOR_STORE_PATH = BASE_DIR / "vector_store"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def load_documents():
    """Carga todos los PDF y conserva el nombre del archivo en los metadatos."""

    pdf_files = sorted(DOCS_PATH.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No se encontraron archivos PDF en: {DOCS_PATH}"
        )

    documents = []

    print("\nCargando documentos...\n")

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        for page in pages:
            page.metadata["file_name"] = pdf_file.name

        documents.extend(pages)

        print(f"Documento: {pdf_file.name}")
        print(f"Páginas cargadas: {len(pages)}")
        print("-" * 60)

    return documents


def split_documents(documents):
    """Divide el contenido en fragmentos pequeños con superposición."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    print(f"\nFragmentos generados: {len(chunks)}")

    return chunks


def create_embeddings():
    """Inicializa el modelo local de embeddings."""

    print("\nCargando modelo de embeddings...")
    print(f"Modelo: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    return embeddings


def build_vector_store(chunks, embeddings):
    """Crea el índice FAISS con los fragmentos procesados."""

    print("\nGenerando embeddings y creando el índice FAISS...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


def save_vector_store(vector_store):
    """Guarda el índice FAISS dentro del proyecto."""

    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(VECTOR_STORE_PATH))

    print("\nÍndice vectorial guardado correctamente.")
    print(f"Ubicación: {VECTOR_STORE_PATH}")


def main():
    try:
        print("\n" + "=" * 70)
        print("CREACIÓN DE LA BASE VECTORIAL")
        print("=" * 70)

        documents = load_documents()
        chunks = split_documents(documents)
        embeddings = create_embeddings()
        vector_store = build_vector_store(chunks, embeddings)
        save_vector_store(vector_store)

        print("\n" + "=" * 70)
        print("PROCESO COMPLETADO")
        print("=" * 70)
        print(f"Páginas procesadas: {len(documents)}")
        print(f"Fragmentos indexados: {len(chunks)}")
        print(f"Modelo utilizado: {EMBEDDING_MODEL}")
        print("=" * 70)

    except FileNotFoundError as error:
        print(f"\nError: {error}")

    except Exception as error:
        print("\nOcurrió un error durante la indexación.")
        print(f"Detalle: {error}")


if __name__ == "__main__":
    main()