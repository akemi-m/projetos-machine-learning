import pandas as pd

df = pd.read_csv('./docs/knn/credit_score_classification.csv')

print(df.iloc[8414:8417].to_markdown(index=False))

