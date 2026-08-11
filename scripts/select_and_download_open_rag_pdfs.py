import json
from pathlib import Path
from urllib.request import urlretrieve

TARGET_CHARS = 700_000

BASE_DIR = Path("open_rag_data/open_ragbench/pdf/arxiv")
CORPUS_DIR = BASE_DIR / "corpus"

QRELS_PATH = BASE_DIR / "qrels.json"
QUERIES_PATH = BASE_DIR / "queries.json"
PDF_URLS_PATH = BASE_DIR / "pdf_urls.json"

PDF_DIR = Path("open_rag_data/selected_pdfs")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def count_document_chars(document: dict) -> int:
    total = 0

    for section in document.get("sections", []):
        text = section.get("text", "")
        total += len(text)

    return total


def main():
    qrels = load_json(QRELS_PATH)
    queries = load_json(QUERIES_PATH)
    pdf_urls = load_json(PDF_URLS_PATH)

    # Pobranie unikalnych dokumentów występujących w qrels.
    relevant_doc_ids = []
    seen = set()

    for relation in qrels.values():
        doc_id = relation["doc_id"]

        if doc_id not in seen:
            seen.add(doc_id)
            relevant_doc_ids.append(doc_id)

    # Wybór dokumentów do limitu około 700 000 znaków.
    selected_documents = []
    total_chars = 0

    for doc_id in relevant_doc_ids:
        document_path = CORPUS_DIR / f"{doc_id}.json"

        if not document_path.exists():
            continue

        document = load_json(document_path)
        document_chars = count_document_chars(document)

        if total_chars + document_chars > TARGET_CHARS:
            continue

        selected_documents.append(
            {
                "doc_id": doc_id,
                "chars": document_chars,
            }
        )

        total_chars += document_chars

    # Liczenie pytań z pominięciem pytań opartych na obrazkach.
    question_counts = {}

    for query_id, relation in qrels.items():
        query_data = queries.get(query_id)

        if query_data is None:
            continue

        if query_data.get("source") == "text-image":
            continue

        doc_id = relation["doc_id"]

        question_counts[doc_id] = question_counts.get(doc_id, 0) + 1

    # Podsumowanie wybranego korpusu.
    print(f"Selected documents: {len(selected_documents)}")
    print(f"Total characters: {total_chars:,}")
    print()

    total_questions = 0

    for document in selected_documents:
        doc_id = document["doc_id"]
        questions = question_counts.get(doc_id, 0)

        total_questions += questions

        print(
            f"{doc_id}: "
            f'{document["chars"]:,} chars, '
            f"{questions} questions"
        )

    print()
    print(f"Total questions (excluding text-image): {total_questions}")

    # Linki do wybranych PDF-ów.
    print("\nSelected PDF URLs:")

    for document in selected_documents:
        doc_id = document["doc_id"]
        url = pdf_urls.get(doc_id)

        if url:
            print(url)

    # Pobranie PDF-ów.
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    print("\nDownloading PDFs...")

    for document in selected_documents:
        doc_id = document["doc_id"]
        url = pdf_urls.get(doc_id)

        if url is None:
            print(f"{doc_id}: URL not found")
            continue

        output_path = PDF_DIR / f"{doc_id}.pdf"

        if output_path.exists():
            print(f"{doc_id}: already downloaded")
            continue

        print(f"{doc_id}: downloading...")
        urlretrieve(url, output_path)


if __name__ == "__main__":
    main()