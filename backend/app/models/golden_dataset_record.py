from dataclasses import dataclass

from app.models.golden_evidence import GoldenEvidence
from app.models.golden_relevant_chunk import GoldenRelevantChunk


@dataclass(frozen=True, slots=True)
class GoldenDatasetRecord:
    query_id: str
    question: str
    expected_answer: str
    evidence: tuple[GoldenEvidence, ...]
    relevant_chunks: tuple[GoldenRelevantChunk, ...]
    
    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id must not be empty")
        
        if not self.question.strip():
            raise ValueError("question must not be empty")
        
        if not self.expected_answer.strip():
            raise ValueError("expected_answer must not be empty")
        
        if not self.evidence:
            raise ValueError("evidence must not be empty")
        
        if not self.relevant_chunks:
            raise ValueError("relevant_chunks must not be empty")
        
        chunk_ids = [
            chunk.chunk_id
            for chunk in self.relevant_chunks
        ]
        
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "relevant_chunks must have unique chunk_id values"
            )
        
        for chunk in self.relevant_chunks:
            for group in chunk.evidence_intervals:
                if group.evidence_index >= len(self.evidence):
                    raise ValueError(
                        "evidence_index is outside evidence"
                    )
                
                evidence_length = self.evidence[
                    group.evidence_index
                ].normalized_length
                
                if any(
                    interval.end > evidence_length
                    for interval in group.intervals
                ):
                    raise ValueError(
                        "evidence interval exceeds normalized evidence length"
                    )