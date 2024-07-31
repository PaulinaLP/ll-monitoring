import pandas as pd

def ingest_data ():
    base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
    relative_url = '04-monitoring/data/results-gpt4o-mini.csv'
    url = f'{base_url}/{relative_url}?raw=1'
    df = pd.read_csv(url)
    df = df.iloc[:300]
    df.to_csv("top300.csv")
    print("done")