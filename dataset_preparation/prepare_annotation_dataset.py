import argparse
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PREPARATION_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PREPARATION_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Open RAGBench and prepare the selected documents "
            "and questions for manual evidence annotation."
        )
    )
    parser.add_argument(
        "--skip-open-ragbench-download",
        action="store_true",
        help="Use the Open RAGBench files already present on disk.",
    )
    parser.add_argument(
        "--skip-pdf-download",
        action="store_true",
        help="Create the manifests without downloading the selected PDFs.",
    )
    parser.add_argument(
        "--force-open-ragbench-download",
        action="store_true",
        help="Redownload Open RAGBench source files.",
    )
    return parser.parse_args()


def run_script(script_name: str, *arguments: str) -> None:
    command = [
        sys.executable,
        str(SCRIPTS_DIR / script_name),
        *arguments,
    ]
    print(f"\n=== {script_name} ===", flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def main() -> None:
    args = parse_args()

    if not args.skip_open_ragbench_download:
        download_arguments = (
            ("--force",)
            if args.force_open_ragbench_download
            else ()
        )
        run_script("download_open_ragbench.py", *download_arguments)

    subset_arguments = (
        ("--skip-pdf-download",)
        if args.skip_pdf_download
        else ()
    )
    run_script("prepare_open_rag_subset.py", *subset_arguments)

    print("\nAnnotation dataset preparation completed successfully.")
    print("Outputs:")
    print("  - dataset_preparation/open_rag_data/selected_documents.json")
    print("  - dataset_preparation/open_rag_data/selected_questions.json")
    if not args.skip_pdf_download:
        print("  - dataset_preparation/open_rag_data/selected_pdfs/")
    print("\nNext manual step:")
    print(
        "  Create or update "
        "dataset_preparation/open_rag_data/evidence_annotations.json "
        "using only the ground_truth_text assigned to each question."
    )
    print("\nAfter annotation, build the golden dataset with:")
    print("  python dataset_preparation/build_golden_dataset.py")


if __name__ == "__main__":
    main()
