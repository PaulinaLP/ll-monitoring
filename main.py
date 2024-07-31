from ingest import ingest_data
import pandas as pd
from sentence_transformers import SentenceTransformer

INGEST=0

if INGEST==1:
    ingest_data()
df=pd.read_csv("top300.csv")
print(df.columns)
model_name="multi-qa-mpnet-base-dot-v1"
embedding_model = SentenceTransformer(model_name)
answer_llm = df.iloc[0].answer_llm
answer_emb = embedding_model.encode(answer_llm)
print(answer_emb[0])



