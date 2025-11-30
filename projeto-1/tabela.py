import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# Variáveis binárias como int
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

# ---- Amostragem
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

# Preprocessamento comum
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

# ---- Modelos
models = {
    "KNN": Pipeline([('preprocessor', preprocessor),
                     ('classifier', KNeighborsClassifier(n_neighbors=3, n_jobs=-1))]),
    
    "Árvore de Decisão": Pipeline([('preprocessor', preprocessor),
                                   ('classifier', DecisionTreeClassifier(random_state=42))]),
    
    "KMeans": Pipeline([('preprocessor', preprocessor),
                        ('classifier', KMeans(n_clusters=2, random_state=42, n_init=10))])
}

# ---- Resultados
results = []

for name, model in models.items():
    if name == "KMeans":
        model.fit(X_sample)  # não supervisionado
        y_pred = model.predict(X_sample)
        y_true = y_sample
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_true = y_test

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results.append({"Modelo": name, "Acurácia": acc, "Precisão": prec,
                    "Recall": rec, "F1-score": f1})

# ---- Tabela comparativa
results_df = pd.DataFrame(results)
print(results_df.to_html(classes="table table-bordered table-striped", border=0))