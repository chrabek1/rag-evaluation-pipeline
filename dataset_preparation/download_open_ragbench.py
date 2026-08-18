import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PREPARATION_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PREPARATION_ROOT / "open_rag_data/open_ragbench/pdf/arxiv"

DEFAULT_REPOSITORY = "deepmatics/open_ragbench"
DEFAULT_REVISION = "main"
SOURCE_DIR = "official/pdf/arxiv"
METADATA_FILES = (
    "qrels.json",
    "queries.json",
    "pdf_urls.json",
    "answers.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Open RAGBench arXiv metadata and the corpus "
            "documents referenced by qrels from Hugging Face."
        )
    )
    parser.add_argument(
        "--repository",
        default=DEFAULT_REPOSITORY,
        help="Hugging Face dataset repository.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Repository revision, branch, tag, or commit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download files again even when they already exist.",
    )
    return parser.parse_args()


def source_url(
    repository: str,
    revision: str,
    relative_path: str,
) -> str:
    encoded_path = quote(relative_path, safe="/")
    encoded_revision = quote(revision, safe="")
    return (
        f"https://huggingface.co/datasets/{repository}/resolve/"
        f"{encoded_revision}/{encoded_path}?download=true"
    )


def download_file(
    url: str,
    destination: Path,
    force: bool,
    attempts: int = 3,
) -> bool:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "rag-evaluation-pipeline/1.0"})

    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=120) as response:
                with temporary_path.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)

            temporary_path.replace(destination)
            return True
        except (HTTPError, URLError, TimeoutError, OSError):
            temporary_path.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(2 ** (attempt - 1))

    return False


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()
    downloaded = 0
    reused = 0

    for filename in METADATA_FILES:
        relative_path = f"{SOURCE_DIR}/{filename}"
        destination = OUTPUT_DIR / filename
        changed = download_file(
            source_url(
                args.repository,
                args.revision,
                relative_path,
            ),
            destination,
            args.force,
        )
        downloaded += changed
        reused += not changed

    qrels = load_json(OUTPUT_DIR / "qrels.json")
    document_ids = list(
        dict.fromkeys(
            relation["doc_id"]
            for relation in qrels.values()
        )
    )

    print(f"Referenced corpus documents: {len(document_ids)}")

    for index, document_id in enumerate(document_ids, start=1):
        filename = f"{document_id}.json"
        relative_path = f"{SOURCE_DIR}/corpus/{filename}"
        destination = OUTPUT_DIR / "corpus" / filename
        changed = download_file(
            source_url(
                args.repository,
                args.revision,
                relative_path,
            ),
            destination,
            args.force,
        )
        downloaded += changed
        reused += not changed

        if changed and (index % 25 == 0 or index == len(document_ids)):
            print(f"Downloaded corpus documents: {index}/{len(document_ids)}")

    print(f"Downloaded files: {downloaded}")
    print(f"Reused existing files: {reused}")
    print(f"Saved to: {OUTPUT_DIR.relative_to(PREPARATION_ROOT)}")


if __name__ == "__main__":
    main()
