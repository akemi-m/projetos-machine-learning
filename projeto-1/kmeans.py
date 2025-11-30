import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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

# Features e Target
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

# Pré-processamento
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

# Pipeline com KMeans
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KMeans(n_clusters=2, init='k-means++', max_iter=100, random_state=42))
])

# Ajustar pipeline na amostra
pipeline.fit(X_sample)

# Predição para a amostra
y_pred_sample = pipeline.predict(X_sample)


print("Accuracy:", accuracy_score(y_sample, y_pred_sample))
print("<h3>Relatório de Classificação:</h3>")
report_df = pd.DataFrame(classification_report(y_sample, y_pred_sample, output_dict=True)).transpose()
print(report_df.to_html(classes="table table-bordered table-striped", border=0))

# ---- Matriz de confusão
print("<h3> Matriz de Confusão:</h3>")
cm = confusion_matrix(y_sample, y_pred_sample)
cm_df = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
print(cm_df.to_html(classes="table table-bordered table-striped", border=0))

# ---- PCA para visualização
X_sample_transformed = pipeline.named_steps['preprocessor'].transform(X_sample)
pca = PCA(n_components=2)
X_sample_pca = pca.fit_transform(X_sample_transformed)

# Centróides no espaço PCA
kmeans_step = pipeline.named_steps['classifier']
centroids = kmeans_step.cluster_centers_
centroids_pca = pca.transform(centroids)

# Plot rápido
plt.figure(figsize=(10, 8))
plt.scatter(X_sample_pca[:, 0], X_sample_pca[:, 1], c=y_pred_sample, cmap='viridis', s=8)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           c='red', marker='*', s=200, label='Centroids')
plt.title('K-Means Clustering (PCA 2D) - Amostragem')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())