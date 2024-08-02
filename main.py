from ingest import ingest_data
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from rouge import Rouge

INGEST=0
EMBEDDING=0
ROUGE=1


def calculate_norm(v):
        norm = np.sqrt((v * v).sum())    
        v_norm = v / norm
        return v_norm


def embedding (df):
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

    
    df['answer_llm_norm']=df['answer_llm_emb'].apply(calculate_norm)
    df['answer_orig_norm']=df['answer_orig_emb'].apply(calculate_norm)

    df['dot_norm']=df.apply(lambda row: np.dot(row['answer_llm_norm'],row['answer_orig_norm'] ), axis=1  )
    cosine_quantile_75= df['dot_norm'].quantile(0.75)
    print(f'the cosine quantile is {cosine_quantile_75}')


def rouge_calculation(df):
    rouge_scorer = Rouge()
    df['scores'] = rouge_scorer.get_scores(df['answer_llm'], df['answer_orig'])
    scores_10=df.iloc[10,df.columns.get_loc("scores")]
    f_rouge_1=scores_10['rouge-1']['f']
    print(f_rouge_1)
    f_rouge_2=scores_10['rouge-2']['f']
    f_rouge_l=scores_10['rouge-l']['f']
    print((f_rouge_1+f_rouge_2+f_rouge_l)/3)
    df['f_rouge_2']=df['scores'].apply(lambda x:x['rouge-2']['f'])
    print(df['f_rouge_2'].mean())


if INGEST==1:
    ingest_data()

df=pd.read_csv("top300.csv")

if EMBEDDING==1:
    embedding(df)

if ROUGE==1:
     rouge_calculation(df)





