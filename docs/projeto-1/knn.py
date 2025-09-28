import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

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
cat_features = ['USMER','SEX','PATIENT_TYPE','MEDICAL_UNIT','DIED_FLAG']  # só OneHot necessário aqui
X = d[num_features + cat_features + bin_cols]
y = d['TARGET']

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=3, n_jobs=-1))
])

pipeline.fit(X_train, y_train)

# Previsão
y_pred = pipeline.predict(X_test)

# Avaliação
print("Accuracy:", accuracy_score(y_test, y_pred))
print("<h3>Relatório de Classificação:</h3>")
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
print(report_df.to_html(classes="table table-bordered table-striped", border=0))

print("<h3> Matriz de Confusão:</h3>")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Leve', 'Grave'], columns=['Leve', 'Grave'])
print(cm_df.to_html(classes="table table-bordered table-striped", border=0))

