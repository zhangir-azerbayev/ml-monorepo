import ndjson
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

ds = load_dataset("openai/summarize_from_feedback", name="axis")["validation"]

ds = [
    {
        "rating": int(x["summary"]["axes"]["overall"]),
        "body": "\n\n".join([x["info"][k] for k in ['title', 'post', 'article'] if x["info"][k] is not None]).strip(),
        "summary": x["summary"]["text"].strip(),
        "feedback": x["summary"]["note"].strip(),
    }
    for x in tqdm(ds) if x ["summary"]["note"] is not None and x["summary"]["axes"]["overall"] is not None 

]

Path('data/').mkdir(exist_ok=True, parents=True)

with open('data/summarize_from_feedback.jsonl', 'w') as f: 
    ndjson.dump(ds, f)
