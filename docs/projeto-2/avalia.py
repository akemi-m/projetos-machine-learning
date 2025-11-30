import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# Configurações iniciais
# =========================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =========================
# Configurações
# =========================

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

# y = variável numérica (alvo)
y = df[TARGET_COLUMN]

# X = todas as demais colunas (assumidas categóricas)
X = df.drop(columns=[TARGET_COLUMN])

# =========================
# Train/Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

# =========================
# One-Hot Encoding para todas as features
# =========================
categorical_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# =========================
# Definição dos modelos (pipelines)
# =========================
models = {
    "LinearRegression": Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression())
    ]),
    
    "RandomForest": Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE
        ))
    ]),
    
    "KNN": Pipeline([
        ("preprocess", preprocessor),
        ("model", KNeighborsRegressor(
            n_neighbors=5
        ))
    ])
}

# =========================
# Treinamento dos modelos
# =========================
fitted_models = {}
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    fitted_models[name] = mdl

# =========================
# Função de métricas
# =========================
def compute_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

# =========================
# Avaliar modelos e montar DataFrame de métricas
# =========================
results = {}

for name, mdl in fitted_models.items():
    y_pred = mdl.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)
    results[name] = metrics

metrics_df = pd.DataFrame(results).T  # modelos nas linhas

# =========================
# Gráficos de comparação (R² e RMSE)
# =========================
# Gráfico 1 – R² por modelo
plt.figure(figsize=(8, 5))
plt.bar(metrics_df.index, metrics_df["R2"])
plt.title("R² por modelo")
plt.ylabel("R²")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)

buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())

# Gráfico 2 – RMSE por modelo
plt.figure(figsize=(8, 5))
plt.bar(metrics_df.index, metrics_df["RMSE"])
plt.title("RMSE por modelo")
plt.ylabel("RMSE")
plt.grid(axis="y", alpha=0.3)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
