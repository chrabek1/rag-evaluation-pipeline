# Golden Dataset Preparation

The workflow includes a required evidence annotation stage.
`evidence_annotations.json` may be prepared manually or with LLM assistance,
but every annotation must be reviewed and approved by a person. No script in
this repository generates its contents automatically.

This directory contains the offline workflow used to prepare the project's
golden dataset from Open RAGBench. It supports the evaluation project, but it
is deliberately separated from the runtime indexing, retrieval, and evaluation
pipeline.

The boundary between the two parts of the project is the generated
`golden_dataset.json` file:

```text
Open RAGBench + human-reviewed evidence annotation
                    ↓
        dataset_preparation scripts
                    ↓
        ../golden_dataset.json
                    ↓
          evaluation pipeline input
```

The runtime indexing, retrieval, and evaluation pipeline does not perform these
preparation steps; it only consumes the validated `golden_dataset.json`.
The scripts in this directory run the offline preparation workflow:

- `download_open_ragbench.py` downloads the required Open RAGBench source
  files;
- `prepare_open_rag_subset.py` selects the source subset;
- `prepare_annotation_dataset.py` orchestrates source preparation, selection,
  and PDF download;
- `build_golden_dataset.py` maps reviewed evidence annotations to the
  fixed chunks and writes the validated golden dataset.

The evidence annotations may be selected manually or proposed with help from
an LLM. In both cases they must be reviewed manually as described below.

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
python3 dataset_preparation/prepare_annotation_dataset.py
```

Useful options:

```bash
# Reuse Open RAGBench files already downloaded to open_rag_data.
python3 dataset_preparation/prepare_annotation_dataset.py \
  --skip-open-ragbench-download

# Build manifests without downloading the selected PDFs.
python3 dataset_preparation/prepare_annotation_dataset.py \
  --skip-pdf-download

# Download the Open RAGBench source files again.
python3 dataset_preparation/prepare_annotation_dataset.py \
  --force-open-ragbench-download
```

The preparation command produces:

- `open_rag_data/selected_documents.json`;
- `open_rag_data/selected_questions.json`;
- `open_rag_data/selected_pdfs/`, unless PDF download was skipped.

Paths above are relative to `dataset_preparation/`.

## 2. Prepare and review evidence annotations

`evidence_annotations.json` is a reviewed preparation input. Its contents may
be created manually or proposed with help from an LLM. The preparation scripts
do not generate or approve the annotations. Before running
`build_golden_dataset.py`, a person must verify every annotation and approve it
for the selected question. Create or update:

```text
dataset_preparation/open_rag_data/evidence_annotations.json
```

Prepare and verify the annotations using all three artifacts produced in the
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
python3 dataset_preparation/build_golden_dataset.py
```

The command writes `golden_dataset.json` to the repository root. It validates,
among other things:

- unique and matching question IDs;
- a non-empty Open RAGBench answer for every selected question;
- non-empty evidence annotations;
- exact evidence occurrence in the assigned source section;
- chunk/document consistency;
- evidence-to-chunk matches against the configured relevance rules;
- absence of questions without relevant chunks.

The command finishes with `Validation: PASS` only after all checks succeed.

## How evidence coverage is calculated

Coverage measures how much of the reviewed evidence is represented
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
fragment. Otherwise, the matcher searches for local, ordered groups of common
text blocks containing at least 15 characters. Blocks may be joined only when
the gaps between them are small in both the evidence and the chunk. This
prevents unrelated phrases found in distant parts of a chunk from being added
together.

Chunk relevance and chunk coverage are related but separate concepts:

- `relevant_chunks` answers whether a chunk contains a meaningful part of at
  least one reviewed evidence fragment;
- `evidence_coverage` measures how much of all evidence assigned to the
  question is covered by that chunk.

A chunk is relevant when a match contains at least 30 normalized characters
(or the complete fragment when it is shorter) and one of these conditions is
met:

- the local match covers at least 80% of an evidence fragment;
- one contiguous block covers at least 40% of a fragment;
- the evidence is split at a chunk boundary, the match touches the beginning
  or end of the fragment, and it covers at least 35% of that fragment.

These rules are evaluated independently for every entry in `evidence_texts`.
Weak matches from multiple evidence fragments are not added together to decide
whether a chunk is relevant. Consequently, a relevant chunk may have a global
`evidence_coverage` below `0.25` when it contains a meaningful part of one
short fragment that represents only a small part of all evidence for the
question.

Mathematical notation is handled separately from surrounding prose. Large
LaTeX spans are compared locally using normalized tokens while allowing benign
PDF extraction differences, such as reordered subscripts and superscripts. A
mathematical match requires at least 85% token coverage, matching numeric
values, and matching anchor tokens. This prevents a formula with a different
numeric result, or a related equation using similar symbols, from being marked
as the annotated evidence. Text surrounding a formula can still establish
relevance independently.

The resulting per-chunk score is stored as `evidence_coverage` and rounded to
four decimal places.

The builder also calculates `evidence_coverage_percentage` for all selected
relevant chunks together. It merges the evidence ranges covered by individual
chunks before counting them, so the same fragment retrieved in multiple chunks
is not counted more than once. Small gaps of at most three normalized
characters are treated as PDF formatting differences when they occur inside a
confirmed local match. Short prefixes or suffixes below the 15-character block
limit are included only when they occur next to an already confirmed match in
the same chunk. The combined value is converted to a percentage and rounded to
two decimal places.

## Golden dataset contract

The runtime evaluation pipeline needs only these fields:

```json
{
  "query_id": "question-id",
  "question": "Question sent to the retriever",
  "expected_answer": "Reference answer from Open RAGBench",
  "relevant_chunks": [
    {
      "chunk_id": "document.pdf_0001",
      "evidence_coverage": 0.75
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
