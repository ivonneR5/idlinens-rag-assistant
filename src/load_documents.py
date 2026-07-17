from pathlib import Path
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PATH = BASE_DIR / "docs"

def load_pdfs():
    """Carga todos los archivos PDF de la carpeta docs."""

    pdf_files = list(DOCS_PATH.glob("*.pdf"))

    if not pdf_files:
        print("No se encontraron archivos PDF.")
        return

    print(f"\nSe encontraron {len(pdf_files)} documentos.\n")

    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)

            print("=" * 60)
            print(f"Documento: {pdf.name}")
            print(f"Páginas: {len(reader.pages)}")

        except Exception as e:
            print(f"Error leyendo {pdf.name}: {e}")


if __name__ == "__main__":
    load_pdfs()