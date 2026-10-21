"""Run comparable Lab 3 audits and select traceable Vietnamese news examples."""
import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # both CLI and imported-test modes
from scripts.compare_vietnews_audits import compare_audits, save_results

def hashfile(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()

def heldout_rows(df, label_col, seed, max_rows):
    frame=df.copy()
    frame["_source_row"]=range(2,len(frame)+2)
    labels=frame[label_col].astype(str)
    counts=labels.value_counts()
    if max_rows is not None and max_rows<len(frame):
        frame,_=train_test_split(frame,train_size=max_rows,random_state=seed,stratify=labels if len(counts)>1 and counts.min()>1 else None)
        frame=frame.reset_index(drop=True)
    labels=frame[label_col].astype(str)
    counts=labels.value_counts()
    _,test=train_test_split(frame,test_size=0.2,random_state=seed,stratify=labels if len(counts)>1 and counts.min()>1 else None)
    return test.reset_index(drop=True)

def select_articles(test, text_col, label_col, tokenize):
    rows=[]
    for _,r in test.iterrows():
        text=str(r[text_col]) if pd.notna(r[text_col]) else ""
        n=len(tokenize(text))
        if not n:continue
        rows.append(dict(source_csv_row=int(r["_source_row"]),article_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                         category=str(r[label_col]),n_tokens=n,removed_at_128=max(n-128,0),removed_at_256=max(n-256,0)))
    rows.sort(key=lambda r:(-r["n_tokens"],r["source_csv_row"]))
    return rows[:3]

def main():
    p=argparse.ArgumentParser(description="VietNewsSense Part B: comparable 128/256 audits")
    p.add_argument("--lab3_repo",type=Path,required=True)
    p.add_argument("--data_path",type=Path,required=True)
    p.add_argument("--text_col",default="content")
    p.add_argument("--label_col",default="category")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max_rows",type=int,default=1000)
    p.add_argument("--vocab_size",type=int,default=20000)
    p.add_argument("--output_dir",type=Path,default=Path("outputs/vietnews_lab4"))
    args=p.parse_args()
    root=args.lab3_repo.resolve()
    data=args.data_path.resolve()
    if not (root/"run_vietnews_sequence_audit.py").is_file() or not data.is_file():
        p.error("Provide existing Lab 3 repo and real CSV")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("Output directory is not empty; choose a fresh directory")
    df=pd.read_csv(data)
    for c in (args.text_col,args.label_col):
        if c not in df.columns:p.error("Missing column: "+c)
    sample=heldout_rows(df,args.label_col,args.seed,args.max_rows)
    spec=importlib.util.spec_from_file_location("lab3_viet",root/"src"/"vietnamese_text.py")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    articles=select_articles(sample,args.text_col,args.label_col,mod.get_tokenizer("underthesea"))
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False if not args.output_dir.exists() else True)
    fingerprint=hashlib.sha256(",".join(map(str,sample["_source_row"].tolist())).encode()).hexdigest()
    manifest={"dataset_sha256":hashfile(data),"dataset_basename":data.name,"n_rows":len(df),"test_rows_sha256":fingerprint,
              "seed":args.seed,"max_rows":args.max_rows,"vocab_size":args.vocab_size,
              "text_col":args.text_col,"label_col":args.label_col,"lengths":[128,256],"tokenizer":"underthesea"}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    with (out/"article_candidates.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["source_csv_row","article_sha256","category","n_tokens","removed_at_128","removed_at_256"])
        writer.writeheader()
        writer.writerows(articles)
    for length in [128,256]:
        dest=out/("len"+str(length))
        cmd=[sys.executable,str(root/"run_vietnews_sequence_audit.py"),"--data_path",str(data),
             "--text_col",args.text_col,"--label_col",args.label_col,"--seed",str(args.seed),
             "--max_rows",str(args.max_rows),"--vocab_size",str(args.vocab_size),
             "--max_len",str(length),"--modes","whitespace","underthesea","--output_dir",str(dest)]
        subprocess.run(cmd,cwd=root,check=True)
        (dest/"token_examples.csv").unlink(missing_ok=True)
    save_results(compare_audits(out/"len128"/"transfer_comparison.csv",out/"len256"/"transfer_comparison.csv"),out)
    if len(articles)<3 or articles[-1]["n_tokens"]<=256:
        print("WARNING: fewer than 3 documents longer than 256 tokens; request more real data")
    print("Done:",out)

if __name__=="__main__":
    main()
