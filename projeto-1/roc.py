import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc

# ---- Configuração
SAMPLE_SIZE = 20000  # número máximo de linhas para análise/plot

# Carregar dados
df = pd.read_csv(
    'https://raw.githubusercontent.com/akemi-m/projetos-machine-learning/refs/heads/main/docs/projeto-1/covid_data.csv',
    low_memory=True
)
d = df.copy()

# Preencher valores nulos
fill_values = {col: d[col].mode()[0] for col in [
    'USMER', 'SEX', 'PATIENT_TYPE', 'MEDICAL_UNIT', 'DATE_DIED', 'INTUBED', 'PNEUMONIA',
    'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE',
    'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU'
]}
d.fillna(fill_values, inplace=True)
d['AGE'] = d['AGE'].fillna(d['AGE'].median())

# Agrupar categorias raras
for col in ['MEDICAL_UNIT', 'PATIENT_TYPE']:
    counts = d[col].value_counts()
    rare_labels = counts[counts < 100].index
    d[col] = d[col].replace(rare_labels, 'OUTRO')

# Variável alvo
d['TARGET'] = (d['CLASIFFICATION_FINAL'] >= 4).astype(int)

# Variáveis binárias já como int
bin_cols = ['DATE_DIED','INTUBED','PNEUMONIA','PREGNANT','DIABETES','COPD','ASTHMA',
            'INMSUPR','HIPERTENSION','OTHER_DISEASE','CARDIOVASCULAR','OBESITY',
            'RENAL_CHRONIC','TOBACCO','ICU']
for col in bin_cols:
    d[col] = (d[col] != 0) & (d[col] != '9999-99-99')
    d[col] = d[col].astype(int)

d['DIED_FLAG'] = (d['DATE_DIED'] != '9999-99-99').astype(int)

# Features
num_features = ['AGE']
cat_features = ['USMER','SEX','PATIENT_TYPE','MEDICAL_UNIT','DIED_FLAG']
X = d[num_features + cat_features + bin_cols]
y = d['TARGET']

# ---- Amostragem para acelerar
if len(X) > SAMPLE_SIZE:
    X_sample = X.sample(SAMPLE_SIZE, random_state=42)
    y_sample = y.loc[X_sample.index]
else:
    X_sample = X
    y_sample = y

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample
)

# ---- Modelo KNN (com pipeline de preprocessamento)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

knn_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=3, n_jobs=-1))
])

knn_pipeline.fit(X_train, y_train)
y_proba_knn = knn_pipeline.predict_proba(X_test)[:, 1]

# ---- Modelo Árvore de Decisão
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_proba_tree = tree_clf.predict_proba(X_test)[:, 1]

# ---- Curvas ROC
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_proba_tree)
roc_auc_tree = auc(fpr_tree, tpr_tree)

# ---- Plot Comparativo
plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})', lw=2)
plt.plot(fpr_tree, tpr_tree, label=f'Árvore de Decisão (AUC = {roc_auc_tree:.2f})', lw=2, color='green')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC - Comparação KNN vs Árvore de Decisão')
plt.legend(loc="lower right")
plt.grid(True)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
