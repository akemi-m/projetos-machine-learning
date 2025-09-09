import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv('./docs/arvore-decisao/credit_score_classification.csv')

df['Payment_Behaviour'] = df['Payment_Behaviour'].str.replace('_', '\n')

ct = pd.crosstab(df["Payment_Behaviour"], df["Credit_Score"])

plt.figure(figsize=(12,8))
ct.plot(kind="barh", figsize=(12,8), alpha=0.8, legend=True)

plt.ylabel("Payment_Behaviour")
plt.xlabel("Quantidade")
plt.title("Distribuição de Credit_Score em relação a Payment_Behaviour")
plt.legend(title="Credit_Score")

plt.tight_layout()
plt.yticks(fontsize=10)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()