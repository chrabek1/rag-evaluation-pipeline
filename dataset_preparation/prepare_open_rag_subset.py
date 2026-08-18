import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve


PREPARATION_ROOT = Path(__file__).resolve().parent
TARGET_CHARS = 700_000

BASE_DIR = PREPARATION_ROOT / "open_rag_data/open_ragbench/pdf/arxiv"
CORPUS_DIR = BASE_DIR / "corpus"
QRELS_PATH = BASE_DIR / "qrels.json"
QUERIES_PATH = BASE_DIR / "queries.json"
ANSWERS_PATH = BASE_DIR / "answers.json"
PDF_URLS_PATH = BASE_DIR / "pdf_urls.json"

PDF_DIR = PREPARATION_ROOT / "open_rag_data/selected_pdfs"
DOCUMENTS_PATH = PREPARATION_ROOT / "open_rag_data/selected_documents.json"
QUESTIONS_OUTPUT_PATH = PREPARATION_ROOT / "open_rag_data/selected_questions.json"

EXCLUDED_SOURCES = {"text-image", "text-table-image"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the deterministic Open RAGBench subset."
    )
    parser.add_argument(
        "--skip-pdf-download",
        action="store_true",
        help="Create the manifests without downloading the selected PDFs.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def count_document_chars(document: dict) -> int:
    return sum(
        len(section.get("text", ""))
        for section in document.get("sections", [])
    )


def ground_truth_text(document: dict, section_id: int) -> str:
    for section in document["sections"]:
        if section["section_id"] == section_id:
            return section["text"]
    raise ValueError(
        f"Section {section_id} not found in document {document['id']}"
    )


def main() -> None:
    args = parse_args()
    qrels = load_json(QRELS_PATH)
    queries = load_json(QUERIES_PATH)
    answers = load_json(ANSWERS_PATH)
    pdf_urls = load_json(PDF_URLS_PATH)

    relevant_doc_ids = list(
        dict.fromkeys(
            relation["doc_id"]
            for relation in qrels.values()
        )
    )

    selected_documents = []
    documents_by_id = {}
    total_chars = 0

    for doc_id in relevant_doc_ids:
        document_path = CORPUS_DIR / f"{doc_id}.json"
        if not document_path.is_file():
            raise FileNotFoundError(
                f"Missing Open RAGBench document: {document_path}"
            )

        document = load_json(document_path)
        document_chars = count_document_chars(document)
        if total_chars + document_chars > TARGET_CHARS:
            continue

        selected_documents.append(
            {
                "doc_id": doc_id,
                "filename": f"{doc_id}.pdf",
                "chars": document_chars,
                "pdf_url": pdf_urls.get(doc_id),
            }
        )
        documents_by_id[doc_id] = document
        total_chars += document_chars

    selected_doc_ids = {
        document["doc_id"]
        for document in selected_documents
    }
    selected_questions = []

    for query_id, relation in qrels.items():
        doc_id = relation["doc_id"]
        if doc_id not in selected_doc_ids:
            continue

        query = queries.get(query_id)
        if query is None or query.get("source") in EXCLUDED_SOURCES:
            continue

        expected_answer = answers.get(query_id)
        if not isinstance(expected_answer, str) or not expected_answer.strip():
            raise ValueError(
                f"Missing or empty answer for query {query_id}"
            )

        section_id = relation["section_id"]
        selected_questions.append(
            {
                "query_id": query_id,
                "question": query["query"],
                "expected_answer": expected_answer,
                "type": query["type"],
                "source": query.get("source"),
                "doc_id": doc_id,
                "filename": f"{doc_id}.pdf",
                "section_id": section_id,
                "ground_truth_text": ground_truth_text(
                    documents_by_id[doc_id],
                    section_id,
                ),
            }
        )

    save_json(DOCUMENTS_PATH, selected_documents)
    save_json(QUESTIONS_OUTPUT_PATH, selected_questions)

    downloaded = 0
    reused = 0
    if not args.skip_pdf_download:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        for document in selected_documents:
            url = document["pdf_url"]
            if not url:
                raise ValueError(
                    f"PDF URL missing for document {document['doc_id']}"
                )

            output_path = PDF_DIR / document["filename"]
            if output_path.is_file() and output_path.stat().st_size > 0:
                reused += 1
                continue

            temporary_path = output_path.with_suffix(".pdf.part")
            urlretrieve(url, temporary_path)
            temporary_path.replace(output_path)
            downloaded += 1

    print(f"Selected documents: {len(selected_documents)}")
    print(f"Selected questions: {len(selected_questions)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Downloaded PDFs: {downloaded}")
    print(f"Reused PDFs: {reused}")
    print(f"Manifest: {DOCUMENTS_PATH.relative_to(PREPARATION_ROOT)}")
    print(f"Questions: {QUESTIONS_OUTPUT_PATH.relative_to(PREPARATION_ROOT)}")


if __name__ == "__main__":
    main()
