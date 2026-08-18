# Golden Dataset Preparation

This directory contains the offline workflow used to prepare the project's
golden dataset from Open RAGBench. It supports the evaluation project, but it
is deliberately separated from the runtime indexing, retrieval, and evaluation
pipeline.

The boundary between the two parts of the project is the generated
`golden_dataset.json` file:

```text
Open RAGBench + manual evidence annotation
                    ↓
        dataset_preparation scripts
                    ↓
        ../golden_dataset.json
                    ↓
          evaluation pipeline input
```

The pipeline does not download Open RAGBench, select documents, create evidence
annotations, or map evidence to chunks. It only consumes the validated output.

## Inputs and outputs

The preparation workflow uses:

- `../dane.csv` — the fixed corpus of chunks evaluated by the pipeline;
- Open RAGBench metadata and source documents downloaded by the scripts;
- manually reviewed `open_rag_data/evidence_annotations.json`.

Intermediate and final preparation artifacts are stored as follows:

```text
dataset_preparation/
├── open_rag_data/
│   ├── open_ragbench/             # downloaded source data; not versioned
│   ├── selected_pdfs/             # downloaded PDFs; not versioned
│   ├── selected_documents.json    # reproducible selection manifest
│   ├── selected_questions.json    # questions, answers and source sections
│   └── evidence_annotations.json  # manually reviewed evidence
└── ...

../golden_dataset.json             # validated artifact for the pipeline
```

`dane.csv` must remain fixed while building the supplied golden dataset.
Changing chunk boundaries changes the relevant chunk IDs and requires evidence
to be mapped again.

## Requirements

- Python 3.12 or newer;
- internet access when downloading Open RAGBench metadata and selected PDFs;
- the fixed `dane.csv` corpus in the repository root.

The scripts currently use only the Python standard library, so they do not
require the backend or embedding-service environment. Run all commands from the
repository root.

## 1. Prepare the annotation dataset

Download Open RAGBench, select a deterministic subset near the configured
character target, download its PDFs, and create the selection manifests:

```bash
python dataset_preparation/prepare_annotation_dataset.py
```

Useful options:

```bash
# Reuse Open RAGBench files already downloaded to open_rag_data.
python dataset_preparation/prepare_annotation_dataset.py \
  --skip-open-ragbench-download

# Build manifests without downloading the selected PDFs.
python dataset_preparation/prepare_annotation_dataset.py \
  --skip-pdf-download

# Download the Open RAGBench source files again.
python dataset_preparation/prepare_annotation_dataset.py \
  --force-open-ragbench-download
```

The preparation command produces:

- `open_rag_data/selected_documents.json`;
- `open_rag_data/selected_questions.json`;
- `open_rag_data/selected_pdfs/`, unless PDF download was skipped.

Paths above are relative to `dataset_preparation/`.

## 2. Annotate evidence manually

`evidence_annotations.json` is a human-created input. None of the preparation
scripts generates its contents automatically. Create or update:

```text
dataset_preparation/open_rag_data/evidence_annotations.json
```

Prepare the annotations manually using all three artifacts produced in the
previous step:

- `selected_documents.json` identifies the selected documents and preserves
  their source metadata;
- `selected_questions.json` assigns each question to a document and section
  and contains the `ground_truth_text` from which evidence must be copied;
- `selected_pdfs/` contains the source documents used to read the surrounding
  context and manually verify that the selected evidence is sufficient and
  faithful to the source.

For every record in `selected_questions.json`, identify all text fragments
required to answer the question. Each `evidence_texts` entry must be copied
exactly from that question's `ground_truth_text`. Use the corresponding PDF to
understand and verify the context, but do not copy evidence from the extracted
PDF text because its whitespace, formulas, or punctuation may differ from
`ground_truth_text` and fail exact validation.

Do not paraphrase, correct, or supplement the source text. Preserve complete
sentences and include conditions, limitations, and exceptions needed for the
answer. Review every annotation manually before building the final artifact.

Expected record shape:

```json
{
  "query_id": "question-id",
  "evidence_texts": [
    "An exact fragment copied from ground_truth_text."
  ]
}
```

## 3. Inspect text-size distributions

The optional comparison script reports size statistics for chunks, source
sections, and evidence annotations:

```bash
python dataset_preparation/compare_text_sizes.py
```

These statistics help assess whether source sections are too broad to be used
directly as chunk-level relevance labels.

## 4. Build and validate the golden dataset

Map the reviewed evidence to chunks from `dane.csv`, validate the records, and
publish the final artifact:

```bash
python dataset_preparation/build_golden_dataset.py
```

The command writes `golden_dataset.json` to the repository root. It validates,
among other things:

- unique and matching question IDs;
- a non-empty Open RAGBench answer for every selected question;
- non-empty evidence annotations;
- exact evidence occurrence in the assigned source section;
- chunk/document consistency;
- relevant chunk coverage against the configured threshold;
- absence of questions without relevant chunks.

The command finishes with `Validation: PASS` only after all checks succeed.

## How evidence coverage is calculated

Coverage measures how much of the manually annotated evidence is represented
in a corpus chunk. Before comparison, both evidence and chunk text are
normalized to reduce irrelevant differences introduced by Markdown and PDF
extraction. Normalization includes lowercasing, Unicode normalization,
whitespace and punctuation handling, repairing line-break hyphenation, and
normalizing common mathematical symbols and LaTeX commands.

For a single chunk, the builder calculates:

```text
chunk evidence coverage =
    matched normalized evidence characters
    / total normalized evidence characters
```

An exact occurrence of an evidence fragment counts as full coverage for that
fragment. Otherwise, the matcher counts common text blocks containing at least
15 characters. Mathematical evidence additionally uses a local token-based
comparison so that harmless PDF extraction changes, such as reordered
subscripts or superscripts, do not prevent a match.

A chunk is added to `relevant_chunks` when its coverage is at least `0.25`,
meaning that it contains at least 25% of the evidence required for that
question. Its score is stored as `evidence_coverage` and rounded to four decimal
places.

The builder also calculates `evidence_coverage_percentage` for all selected
relevant chunks together. It merges the evidence ranges covered by individual
chunks before counting them, so the same fragment retrieved in multiple chunks
is not counted more than once. The combined value is converted to a percentage
and rounded to two decimal places.

## Golden dataset contract

The runtime evaluation pipeline needs only these fields:

```json
{
  "query_id": "question-id",
  "question": "Question sent to the retriever",
  "expected_answer": "Reference answer from Open RAGBench",
  "relevant_chunks": [
    {
      "chunk_id": "document.pdf_0001"
    }
  ]
}
```

`expected_answer` can be used later by generation-quality metrics. The
generated file also retains provenance and audit fields such as
`ground_truth_text`, `evidence_text`, evidence coverage, document identifiers,
and per-chunk coverage. These fields document how the relevance labels were
created, but retrieval metrics should not depend on the Open RAGBench-specific
preparation process.

## Rebuilding safely

Building the dataset overwrites the repository-root `golden_dataset.json`.
Before accepting a rebuilt artifact:

1. confirm that `dane.csv` is the intended fixed corpus;
2. review changes to annotations and selection manifests;
3. run the builder and require `Validation: PASS`;
4. inspect the resulting Git diff, especially changed relevant chunk IDs and
   coverage values;
5. run the pipeline's dataset-contract tests once they are available.
