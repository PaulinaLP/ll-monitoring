from ingest import ingest_data
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

INGEST=1

if INGEST==1:
    ingest_data()
df=pd.read_csv("top300.csv")
model_name="multi-qa-mpnet-base-dot-v1"
embedding_model = SentenceTransformer(model_name)
answer_llm = df.iloc[0].answer_llm
answer_emb = embedding_model.encode(answer_llm)
print(answer_emb[0])

df['answer_llm_emb']=df['answer_llm'].apply(embedding_model.encode)
df['answer_orig_emb']=df['answer_orig'].apply(embedding_model.encode)

df['dot']=df.apply(lambda row: np.dot(row['answer_llm_emb'],row['answer_orig_emb'] ), axis=1  )
quantile_75= df['dot'].quantile(0.75)
print(f'the quantile is {quantile_75}')

def calculate_norm(v):
    norm = np.sqrt((v * v).sum())    
    v_norm = v / norm
    return v_norm
df['answer_llm_norm']=df['answer_llm_emb'].apply(calculate_norm)
df['answer_orig_norm']=df['answer_orig_emb'].apply(calculate_norm)

df['dot_norm']=df.apply(lambda row: np.dot(row['answer_llm_norm'],row['answer_orig_norm'] ), axis=1  )
cosine_quantile_75= df['dot_norm'].quantile(0.75)
print(f'the cosine quantile is {cosine_quantile_75}')




