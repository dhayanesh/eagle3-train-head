import json
from pathlib import Path

INPUT_PATH = Path("data/raw_summaries.jsonl")
OUTPUT_PATH = Path("data/summarization_train.jsonl")

SOURCE_FIELD = "article"
SUMMARY_FIELD = "summary"

SYSTEM_PROMPT = "You are a precise summarization assistant."
INSTRUCTION = "Summarize this article in 1-3 concise sentences."


def build_record(source: str, summary: str) -> dict:
    user_content = f"{INSTRUCTION}\n\n{source}".strip()

    conversations = []
    if SYSTEM_PROMPT.strip():
        conversations.append({"role": "system", "content": SYSTEM_PROMPT})
    conversations.append({"role": "user", "content": user_content})
    conversations.append({"role": "assistant", "content": summary})

    return {"conversations": conversations}


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

written = 0
skipped = 0

with INPUT_PATH.open("r", encoding="utf-8") as src, OUTPUT_PATH.open("w", encoding="utf-8") as dst:
    for line in src:
        line = line.strip()
        if not line:
            skipped += 1
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        source = str(row.get(SOURCE_FIELD, "")).strip()
        summary = str(row.get(SUMMARY_FIELD, "")).strip()

        if not source or not summary:
            skipped += 1
            continue

        dst.write(json.dumps(build_record(source, summary), ensure_ascii=False) + "\n")
        written += 1

print(f"Wrote {written} records to {OUTPUT_PATH}")
print(f"Skipped {skipped} records")
