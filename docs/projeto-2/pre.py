import pandas as pd


df = pd.read_csv('dados/dadosprojeto.csv', low_memory=True)

d = df.copy()

d['elemento'] = d['elemento'].astype('string')
d['foil'] = d['foil'].astype('string')
d['tipoCarta'] = d['tipoCarta'].astype('string')
d['colecao'] = d['colecao'].astype('string')

d['elemento'] = d['elemento'].fillna('None')



print(d.sample(n=10).to_markdown(index=False))