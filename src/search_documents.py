from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH = BASE_DIR / "vector_store"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def load_embeddings():
    """Carga el mismo modelo utilizado durante la indexación."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vector_store(embeddings):
    """Carga el índice FAISS previamente generado."""

    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            "No se encontró la base vectorial. "
            "Ejecuta primero create_vector_store.py."
        )

    return FAISS.load_local(
        folder_path=str(VECTOR_STORE_PATH),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def search_documents(vector_store, question, number_of_results=4):
    """Busca los fragmentos más relacionados con la pregunta."""

    return vector_store.similarity_search(
        query=question,
        k=number_of_results,
    )


def show_results(question, results):
    """Muestra los fragmentos recuperados y su fuente."""

    print("\n" + "=" * 70)
    print("PREGUNTA")
    print("=" * 70)
    print(question)

    if not results:
        print("\nNo se encontraron resultados.")
        return

    for index, document in enumerate(results, start=1):
        file_name = document.metadata.get("file_name", "Archivo desconocido")
        page = document.metadata.get("page")

        if isinstance(page, int):
            page += 1

        print("\n" + "-" * 70)
        print(f"RESULTADO {index}")
        print(f"Archivo: {file_name}")
        print(f"Página: {page if page is not None else 'No disponible'}")
        print("-" * 70)
        print(document.page_content[:900])


def main():
    try:
        embeddings = load_embeddings()
        vector_store = load_vector_store(embeddings)

        question = input(
            "\nEscribe una pregunta sobre IDLinens :\n> "
        ).strip()

        if not question:
            print("Debes escribir una pregunta.")
            return

        results = search_documents(
            vector_store=vector_store,
            question=question,
            number_of_results=4,
        )

        show_results(question, results)

    except FileNotFoundError as error:
        print(f"\nError: {error}")

    except Exception as error:
        print("\nOcurrió un error durante la búsqueda.")
        print(f"Detalle: {error}")


if __name__ == "__main__":
    main()