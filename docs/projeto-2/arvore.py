import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# =========================
# Configurações

def pre(d):


    d['elemento'] = d['elemento'].astype('string')
    d['foil'] = d['foil'].astype('string')
    d['tipoCarta'] = d['tipoCarta'].astype('string')
    d['colecao'] = d['colecao'].astype('string')

    d['elemento'] = d['elemento'].fillna('None')
    return d

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CSV_PATH = "dados/dadosprojeto.csv"   # <-- ajuste aqui
TARGET_COLUMN = "valor"       # <-- ajuste aqui

# =========================
# Leitura dos dados
# =========================
df = pd.read_csv(CSV_PATH)
df = pre(df)

y = df[TARGET_COLUMN].values
X = df.drop(columns=[TARGET_COLUMN])

# =========================
# One-Hot Encoding
# =========================
ohe = OneHotEncoder(handle_unknown="ignore")
X_ohe = ohe.fit_transform(X)

# =========================
# PCA para 2 componentes
# =========================
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_ohe.toarray())

# =========================
# Treinar modelo no espaço PCA
# =========================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE
)
model.fit(X_pca, y)

y_pred = model.predict(X_pca)

# =========================
# Plot – Espaço PCA colorido pelo valor previsto
# =========================
plt.figure(figsize=(7, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Random Forest – Espaço PCA (cor = valor previsto)")
plt.colorbar(scatter, label="Valor previsto")
plt.grid(alpha=0.3)
plt.tight_layout()

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())