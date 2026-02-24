import json
from datasets import load_dataset

out = "data/raw_summaries.jsonl"
ds = load_dataset("EdinburghNLP/xsum", split="train")
with open(out, "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps({"article": row["document"], "summary": row["summary"]}, ensure_ascii=False) + "\n")
print("wrote", len(ds), "rows to", out)
