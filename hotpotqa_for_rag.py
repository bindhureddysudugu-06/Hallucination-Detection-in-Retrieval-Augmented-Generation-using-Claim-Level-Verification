import os
import json
import argparse
from typing import List, Dict, Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def build_question_item(example: Dict[str, Any]) -> Dict[str, Any]:
    question_id = str(example.get("_id", ""))
    question = normalize_text(example.get("question", ""))
    answer = normalize_text(example.get("answer", ""))

    return {
        "question_id": question_id,
        "question": question,
        "reference_answer": answer
    }


def build_documents_from_context(example: Dict[str, Any], question_id: str) -> List[Dict[str, Any]]:
    """
    HotpotQA context format is usually:
    "context": [
        ["Title 1", ["sentence1", "sentence2"]],
        ["Title 2", ["sentence1", "sentence2"]]
    ]
    """
    documents = []
    context = example.get("context", [])

    for idx, item in enumerate(context, start=1):
        if not isinstance(item, list) or len(item) != 2:
            continue

        title = normalize_text(item[0])
        sentences = item[1]

        if isinstance(sentences, list):
            text = " ".join([normalize_text(s) for s in sentences if str(s).strip()])
        else:
            text = normalize_text(sentences)

        if not text:
            continue

        doc_id = f"{question_id}_doc_{idx}"

        documents.append({
            "doc_id": doc_id,
            "title": title,
            "text": text
        })

    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HotpotQA JSON to questions.jsonl and documents.jsonl")
    parser.add_argument("--input_json", type=str, required=True, help="Path to HotpotQA JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save converted files")
    parser.add_argument("--max_examples", type=int, default=100, help="Maximum number of examples to convert")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    data = load_json(args.input_json)

    if not isinstance(data, list):
        raise ValueError("Expected the HotpotQA file to contain a JSON list.")

    questions = []
    all_documents = []

    count = 0
    for example in data:
        if count >= args.max_examples:
            break

        question_item = build_question_item(example)
        if not question_item["question_id"] or not question_item["question"]:
            continue

        questions.append(question_item)

        docs = build_documents_from_context(example, question_item["question_id"])
        all_documents.extend(docs)

        count += 1

    questions_path = os.path.join(args.output_dir, "questions.jsonl")
    documents_path = os.path.join(args.output_dir, "documents.jsonl")

    save_jsonl(questions_path, questions)
    save_jsonl(documents_path, all_documents)

    print("Conversion completed.")
    print(f"Questions saved to: {questions_path}")
    print(f"Documents saved to: {documents_path}")
    print(f"Total questions: {len(questions)}")
    print(f"Total documents: {len(all_documents)}")


if __name__ == "__main__":
    main()