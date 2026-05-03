import os
import re
import json
import time
import uuid
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sentence_transformers import SentenceTransformer

@dataclass
class QuestionItem:
    question_id: str
    question: str
    reference_answer: Optional[str] = None


@dataclass
class DocumentItem:
    doc_id: str
    title: str
    text: str


@dataclass
class ChunkItem:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    start_word: int
    end_word: int


@dataclass
class RetrievedChunk:
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    text: str
    similarity: float


@dataclass
class ClaimResult:
    question_id: str
    claim_id: int
    claim_text: str
    verifier_label: str
    verifier_confidence: float
    evidence_chunk_ids: List[str]
    explanation: str
    retrieval_issue_flag: int
    generation_issue_flag: int
    human_label: Optional[str] = None


@dataclass
class AnswerResult:
    question_id: str
    question: str
    generated_answer: str
    num_claims: int
    num_supported: int
    num_partially_supported: int
    num_unsupported: int
    num_contradicted: int
    unsupported_rate: float
    final_system_label: str
    final_human_label: Optional[str]
    retrieved_chunk_ids: List[str]
    retrieved_similarities: List[float]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalize_label(label: str) -> str:
    if not label:
        return "Unsupported"

    label = label.strip().lower()
    mapping = {
        "supported": "Supported",
        "partially supported": "Partially Supported",
        "partial": "Partially Supported",
        "partially_supported": "Partially Supported",
        "unsupported": "Unsupported",
        "hallucinated": "Unsupported",
        "unsupported/hallucinated": "Unsupported",
        "contradicted": "Contradicted",
        "insufficient evidence": "Partially Supported",
        "insufficient": "Partially Supported",
    }
    return mapping.get(label, "Unsupported")


def answer_label_to_binary(label: str) -> int:
    return 1 if str(label).strip().lower() == "hallucinated" else 0


def claim_label_to_binary(label: str) -> int:
    label = normalize_label(label)
    return 1 if label in {"Unsupported", "Contradicted"} else 0


#loading the file

def load_questions(path: str) -> List[QuestionItem]:
    rows = load_jsonl(path)
    items = []
    for r in rows:
        items.append(
            QuestionItem(
                question_id=str(r["question_id"]),
                question=normalize_whitespace(r["question"]),
                reference_answer=normalize_whitespace(r.get("reference_answer", "")),
            )
        )
    return items


def load_documents(path: str) -> List[DocumentItem]:
    rows = load_jsonl(path)
    items = []
    for r in rows:
        items.append(
            DocumentItem(
                doc_id=str(r["doc_id"]),
                title=normalize_whitespace(r.get("title", "")),
                text=normalize_whitespace(r["text"]),
            )
        )
    return items


def chunk_text_by_words(text: str, chunk_size: int = 256, chunk_overlap: int = 50) -> List[Tuple[str, int, int]]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append((chunk_text, start, end))
        if end == len(words):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def build_chunks(documents: List[DocumentItem], chunk_size: int, chunk_overlap: int) -> List[ChunkItem]:
    all_chunks = []
    for doc in documents:
        pieces = chunk_text_by_words(doc.text, chunk_size, chunk_overlap)
        for idx, (chunk_text, start_word, end_word) in enumerate(pieces, start=1):
            all_chunks.append(
                ChunkItem(
                    chunk_id=f"{doc.doc_id}_chunk_{idx}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=chunk_text,
                    start_word=start_word,
                    end_word=end_word,
                )
            )
    return all_chunks


class LocalEmbeddingClient:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings


class SimpleVectorRetriever:
    def __init__(self, embedding_client: LocalEmbeddingClient):
        self.embedding_client = embedding_client
        self.chunks: List[ChunkItem] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self, chunks: List[ChunkItem], batch_size: int = 64) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embeddings = self.embedding_client.embed_texts(texts, batch_size=batch_size)

    def retrieve(self, question: str, top_k: int = 3) -> List[RetrievedChunk]:
        if self.embeddings is None or not self.chunks:
            raise RuntimeError("Index is empty. Build the index first.")

        q_vec = self.embedding_client.embed_texts([question])[0]
        sims = np.dot(self.embeddings, q_vec)

        top_indices = np.argsort(-sims)[:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[int(idx)]
            sim = float(sims[int(idx)])
            results.append(
                RetrievedChunk(
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    text=chunk.text,
                    similarity=sim,
                )
            )
        return results


def split_answer_into_claims(answer_text: str) -> List[str]:
    answer_text = normalize_whitespace(answer_text)
    if not answer_text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", answer_text)
    claims = [normalize_whitespace(p) for p in parts if normalize_whitespace(p)]

    if not claims:
        claims = [answer_text]

    return claims


def verify_claim_locally(
    claim_text: str,
    retrieved_chunks: List[RetrievedChunk],
    embedding_client: LocalEmbeddingClient,
    supported_threshold: float,
    partial_threshold: float,
) -> Dict[str, Any]:
    
    claim_vec = embedding_client.embed_texts([claim_text])[0]
    chunk_texts = [c.text for c in retrieved_chunks]
    chunk_vecs = embedding_client.embed_texts(chunk_texts)

    sims = np.dot(chunk_vecs, claim_vec)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_chunk = retrieved_chunks[best_idx]

    if best_score >= supported_threshold:
        label = "Supported"
        evidence_chunk_ids = [best_chunk.chunk_id]
        explanation = f"Highest semantic similarity with retrieved evidence is {best_score:.4f}, above supported threshold."
    elif best_score >= partial_threshold:
        label = "Partially Supported"
        evidence_chunk_ids = [best_chunk.chunk_id]
        explanation = f"Highest semantic similarity with retrieved evidence is {best_score:.4f}, between partial and supported thresholds."
    else:
        label = "Unsupported"
        evidence_chunk_ids = []
        explanation = f"Highest semantic similarity with retrieved evidence is {best_score:.4f}, below partial threshold."

    return {
        "label": label,
        "confidence": max(0.0, min(1.0, best_score)),
        "evidence_chunk_ids": evidence_chunk_ids,
        "explanation": explanation,
        "best_similarity": best_score,
    }


def derive_issue_flags(verifier_label: str, evidence_chunk_ids: List[str]) -> Tuple[int, int]:
    retrieval_issue_flag = 0
    generation_issue_flag = 0

    if verifier_label in {"Unsupported", "Partially Supported"} and len(evidence_chunk_ids) == 0:
        retrieval_issue_flag = 1

    if verifier_label == "Contradicted":
        generation_issue_flag = 1

    return retrieval_issue_flag, generation_issue_flag


def aggregate_claims_to_answer_label(
    claim_results: List[ClaimResult],
    hallucination_threshold: float = 0.5,
) -> Tuple[str, float]:
    if not claim_results:
        return "Hallucinated", 1.0

    unsupported_like = 0
    contradicted_count = 0

    for c in claim_results:
        if c.verifier_label == "Contradicted":
            contradicted_count += 1
        if c.verifier_label in {"Unsupported", "Contradicted"}:
            unsupported_like += 1

    unsupported_rate = unsupported_like / len(claim_results)

    if contradicted_count > 0 or unsupported_rate >= hallucination_threshold:
        return "Hallucinated", unsupported_rate
    return "Grounded", unsupported_rate


def load_human_claim_labels(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def derive_human_answer_labels_from_claim_labels(human_claim_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid, group in human_claim_df.groupby("question_id"):
        labels = [normalize_label(x) for x in group["human_label"].tolist()]
        if any(lbl in {"Unsupported", "Contradicted"} for lbl in labels):
            final_label = "Hallucinated"
        else:
            final_label = "Grounded"
        rows.append({"question_id": qid, "final_human_label": final_label})
    return pd.DataFrame(rows)


def evaluate_claim_level(claim_df: pd.DataFrame) -> Dict[str, Any]:
    if claim_df.empty or "human_label" not in claim_df.columns:
        return {}

    eval_df = claim_df.dropna(subset=["human_label"]).copy()
    eval_df = eval_df[eval_df["human_label"].astype(str).str.strip() != ""].copy()
    if eval_df.empty:
        return {}

    y_true = eval_df["human_label"].apply(claim_label_to_binary).tolist()
    y_pred = eval_df["verifier_label"].apply(claim_label_to_binary).tolist()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    return {
        "claim_precision": float(precision),
        "claim_recall": float(recall),
        "claim_f1": float(f1),
        "claim_accuracy": float(acc),
        "claim_n": int(len(eval_df)),
    }


def evaluate_answer_level(answer_df: pd.DataFrame) -> Dict[str, Any]:
    if answer_df.empty or "final_human_label" not in answer_df.columns:
        return {}

    eval_df = answer_df.dropna(subset=["final_human_label"]).copy()
    eval_df = eval_df[eval_df["final_human_label"].astype(str).str.strip() != ""].copy()
    if eval_df.empty:
        return {}

    y_true = eval_df["final_human_label"].apply(answer_label_to_binary).tolist()
    y_pred = eval_df["final_system_label"].apply(answer_label_to_binary).tolist()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    return {
        "answer_precision": float(precision),
        "answer_recall": float(recall),
        "answer_f1": float(f1),
        "answer_accuracy": float(acc),
        "answer_n": int(len(eval_df)),
    }


def run_pipeline(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    questions = load_questions(args.questions)
    documents = load_documents(args.documents)

    print(f"Loaded {len(questions)} questions.")
    print(f"Loaded {len(documents)} documents.")

    embedding_client = LocalEmbeddingClient(model_name=args.embedding_model)

    chunks = build_chunks(
        documents=documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Built {len(chunks)} chunks.")

    retriever = SimpleVectorRetriever(embedding_client)
    print("Building local embedding index...")
    retriever.build_index(chunks=chunks, batch_size=args.embedding_batch_size)

    claim_rows: List[Dict[str, Any]] = []
    answer_rows: List[Dict[str, Any]] = []
    raw_trace_rows: List[Dict[str, Any]] = []

    for q in tqdm(questions, desc="Running local no-API pipeline"):
        retrieved = retriever.retrieve(q.question, top_k=args.top_k)

        generated_answer = q.reference_answer if q.reference_answer else "No reference answer available."

        claims = split_answer_into_claims(generated_answer)
        if not claims:
            claims = [generated_answer]

        question_claim_results: List[ClaimResult] = []

        for claim_idx, claim_text in enumerate(claims, start=1):
            verification = verify_claim_locally(
                claim_text=claim_text,
                retrieved_chunks=retrieved,
                embedding_client=embedding_client,
                supported_threshold=args.supported_threshold,
                partial_threshold=args.partial_threshold,
            )

            retrieval_issue_flag, generation_issue_flag = derive_issue_flags(
                verifier_label=verification["label"],
                evidence_chunk_ids=verification["evidence_chunk_ids"],
            )

            claim_result = ClaimResult(
                question_id=q.question_id,
                claim_id=claim_idx,
                claim_text=claim_text,
                verifier_label=verification["label"],
                verifier_confidence=float(verification["confidence"]),
                evidence_chunk_ids=verification["evidence_chunk_ids"],
                explanation=verification["explanation"],
                retrieval_issue_flag=retrieval_issue_flag,
                generation_issue_flag=generation_issue_flag,
                human_label=None,
            )
            question_claim_results.append(claim_result)
            claim_rows.append(asdict(claim_result))

        final_system_label, unsupported_rate = aggregate_claims_to_answer_label(
            question_claim_results,
            hallucination_threshold=args.hallucination_threshold,
        )

        num_supported = sum(1 for c in question_claim_results if c.verifier_label == "Supported")
        num_partial = sum(1 for c in question_claim_results if c.verifier_label == "Partially Supported")
        num_unsupported = sum(1 for c in question_claim_results if c.verifier_label == "Unsupported")
        num_contradicted = sum(1 for c in question_claim_results if c.verifier_label == "Contradicted")

        answer_result = AnswerResult(
            question_id=q.question_id,
            question=q.question,
            generated_answer=generated_answer,
            num_claims=len(question_claim_results),
            num_supported=num_supported,
            num_partially_supported=num_partial,
            num_unsupported=num_unsupported,
            num_contradicted=num_contradicted,
            unsupported_rate=unsupported_rate,
            final_system_label=final_system_label,
            final_human_label=None,
            retrieved_chunk_ids=[r.chunk_id for r in retrieved],
            retrieved_similarities=[r.similarity for r in retrieved],
        )
        answer_rows.append(asdict(answer_result))

        raw_trace_rows.append(
            {
                "question_id": q.question_id,
                "question": q.question,
                "reference_answer": q.reference_answer,
                "retrieved_chunks": [asdict(r) for r in retrieved],
                "generated_answer": generated_answer,
                "claims": [asdict(c) for c in question_claim_results],
                "final_system_label": final_system_label,
                "unsupported_rate": unsupported_rate,
            }
        )

    claim_df = pd.DataFrame(claim_rows)
    answer_df = pd.DataFrame(answer_rows)

    if args.human_claim_labels and os.path.exists(args.human_claim_labels):
        human_claim_df = load_human_claim_labels(args.human_claim_labels)

        claim_df = claim_df.merge(
            human_claim_df[["question_id", "claim_id", "human_label"]],
            on=["question_id", "claim_id"],
            how="left",
            suffixes=("", "_human"),
        )

        if "human_label_human" in claim_df.columns:
            claim_df["human_label"] = claim_df["human_label_human"]
            claim_df.drop(columns=["human_label_human"], inplace=True)

        human_answer_df = derive_human_answer_labels_from_claim_labels(human_claim_df)
        answer_df = answer_df.merge(human_answer_df, on="question_id", how="left")

    claim_csv_path = os.path.join(args.output_dir, "claim_level_results.csv")
    answer_csv_path = os.path.join(args.output_dir, "answer_level_results.csv")
    trace_jsonl_path = os.path.join(args.output_dir, "rag_outputs.jsonl")
    summary_path = os.path.join(args.output_dir, "summary_metrics.json")
    experiment_log_path = os.path.join(args.output_dir, "experiment_log.csv")
    annotation_template_path = os.path.join(args.output_dir, "manual_claim_annotation_template.csv")
    error_analysis_path = os.path.join(args.output_dir, "error_analysis.csv")

    claim_df.to_csv(claim_csv_path, index=False, encoding="utf-8")
    answer_df.to_csv(answer_csv_path, index=False, encoding="utf-8")
    save_jsonl(trace_jsonl_path, raw_trace_rows)

    annotation_template = claim_df[["question_id", "claim_id", "claim_text"]].copy()
    annotation_template["human_label"] = claim_df["human_label"] if "human_label" in claim_df.columns else ""
    annotation_template.to_csv(annotation_template_path, index=False, encoding="utf-8")

    if "human_label" in claim_df.columns:
        error_df = claim_df.dropna(subset=["human_label"]).copy()
        error_df = error_df[error_df["human_label"].astype(str).str.strip() != ""].copy()
        if not error_df.empty:
            error_df["pred_bin"] = error_df["verifier_label"].apply(claim_label_to_binary)
            error_df["true_bin"] = error_df["human_label"].apply(claim_label_to_binary)
            error_df = error_df[error_df["pred_bin"] != error_df["true_bin"]].copy()

            def classify_error(row: pd.Series) -> str:
                if row["retrieval_issue_flag"] == 1:
                    return "retrieval_miss"
                if row["generation_issue_flag"] == 1:
                    return "generation_issue"
                if row["verifier_label"] == "Partially Supported":
                    return "partial_evidence"
                return "similarity_threshold_error"

            if not error_df.empty:
                error_df["error_type"] = error_df.apply(classify_error, axis=1)
                error_df = error_df[
                    [
                        "question_id",
                        "claim_id",
                        "claim_text",
                        "verifier_label",
                        "human_label",
                        "error_type",
                        "explanation",
                    ]
                ]
        if error_df.empty:
            error_df = pd.DataFrame(
                columns=[
                    "question_id",
                    "claim_id",
                    "claim_text",
                    "verifier_label",
                    "human_label",
                    "error_type",
                    "explanation",
                ]
            )
    else:
        error_df = pd.DataFrame(
            columns=[
                "question_id",
                "claim_id",
                "claim_text",
                "verifier_label",
                "human_label",
                "error_type",
                "explanation",
            ]
        )

    error_df.to_csv(error_analysis_path, index=False, encoding="utf-8")

    claim_metrics = evaluate_claim_level(claim_df)
    answer_metrics = evaluate_answer_level(answer_df)

    summary = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "questions_n": int(len(questions)),
        "documents_n": int(len(documents)),
        "chunks_n": int(len(chunks)),
        "embedding_model": args.embedding_model,
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "top_k": int(args.top_k),
        "supported_threshold": float(args.supported_threshold),
        "partial_threshold": float(args.partial_threshold),
        "hallucination_threshold": float(args.hallucination_threshold),
        "avg_claims_per_answer": float(answer_df["num_claims"].mean()) if not answer_df.empty else 0.0,
        "avg_unsupported_rate": float(answer_df["unsupported_rate"].mean()) if not answer_df.empty else 0.0,
    }
    summary.update(claim_metrics)
    summary.update(answer_metrics)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    exp_row = pd.DataFrame([summary])
    if os.path.exists(experiment_log_path):
        old_df = pd.read_csv(experiment_log_path)
        all_df = pd.concat([old_df, exp_row], ignore_index=True)
        all_df.to_csv(experiment_log_path, index=False, encoding="utf-8")
    else:
        exp_row.to_csv(experiment_log_path, index=False, encoding="utf-8")

    print("\nRun completed.")
    print(f"Claim results: {claim_csv_path}")
    print(f"Answer results: {answer_csv_path}")
    print(f"Trace outputs : {trace_jsonl_path}")
    print(f"Annotation CSV: {annotation_template_path}")
    print(f"Error analysis: {error_analysis_path}")
    print(f"Summary file  : {summary_path}")

    print("\nSummary metrics:")
    for k, v in summary.items():
        print(f"{k}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local no-API claim-level hallucination detection for RAG"
    )
    parser.add_argument("--questions", type=str, required=True, help="Path to questions.jsonl")
    parser.add_argument("--documents", type=str, required=True, help="Path to documents.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--embedding_batch_size", type=int, default=64)

    parser.add_argument("--supported_threshold", type=float, default=0.75)
    parser.add_argument("--partial_threshold", type=float, default=0.55)
    parser.add_argument("--hallucination_threshold", type=float, default=0.5)

    parser.add_argument(
        "--human_claim_labels",
        type=str,
        default="",
        help="Optional CSV with columns: question_id, claim_id, human_label",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)