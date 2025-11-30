import matplotlib.pyplot as plt
import pandas as pd

from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv('https://raw.githubusercontent.com/akemi-m/projetos-machine-learning/refs/heads/main/docs/projeto-1/covid_data.csv', low_memory=True)

d = df.copy()

fill_values = {
    'USMER': d['USMER'].mode()[0],
    'SEX': d['SEX'].mode()[0],
    'PATIENT_TYPE': d['PATIENT_TYPE'].mode()[0],
    'MEDICAL_UNIT': d['MEDICAL_UNIT'].mode()[0],
    'DATE_DIED': d['DATE_DIED'].mode()[0],
    'INTUBED': d['INTUBED'].mode()[0],
    'PNEUMONIA': d['PNEUMONIA'].mode()[0],
    'PREGNANT': d['PREGNANT'].mode()[0],
    'DIABETES': d['DIABETES'].mode()[0],
    'COPD': d['COPD'].mode()[0],
    'ASTHMA': d['ASTHMA'].mode()[0],
    'INMSUPR': d['INMSUPR'].mode()[0],
    'HIPERTENSION': d['HIPERTENSION'].mode()[0],
    'OTHER_DISEASE': d['OTHER_DISEASE'].mode()[0],
    'CARDIOVASCULAR': d['CARDIOVASCULAR'].mode()[0],
    'OBESITY': d['OBESITY'].mode()[0],
    'RENAL_CHRONIC': d['RENAL_CHRONIC'].mode()[0],
    'TOBACCO': d['TOBACCO'].mode()[0],
    'CLASIFFICATION_FINAL': d['CLASIFFICATION_FINAL'].mode()[0],
    'ICU': d['ICU'].mode()[0]
}

d.fillna(fill_values, inplace=True)

d['DIED_FLAG'] = (d['DATE_DIED'] != '9999-99-99').astype(int)
d['AGE'] = d['AGE'].fillna(d['AGE'].median())

age_min = d['AGE'].min()
age_max = d['AGE'].max()
d['AGE'] = (d['AGE'] - age_min) / (age_max - age_min)

plt.figure(figsize=(12, 10))

# Carregar o conjunto de dados
x = d[['USMER', 
    'SEX', 
    'PATIENT_TYPE', 
    'MEDICAL_UNIT', 
    'DIED_FLAG', 
    'INTUBED', 
    'PNEUMONIA', 
    'AGE', 
    'PREGNANT', 
    'DIABETES', 
    'COPD', 
    'ASTHMA', 
    'INMSUPR', 
    'HIPERTENSION', 
    'OTHER_DISEASE', 
    'CARDIOVASCULAR', 
    'OBESITY', 
    'RENAL_CHRONIC', 
    'TOBACCO', 
    'ICU']]
y = (df['CLASIFFICATION_FINAL'] >= 4).map({False: 1, True: 0})

# Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
labels = classifier.classes_
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Avaliar o modelo
accuracy = classifier.score(x_test, y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("<h3>Relatório de Classificação:</h3>")
print(report_df.to_html(classes="table table-bordered table-striped", border=0))

print("<h3>Matriz de Confusão:</h3>")
print(cm_df.to_html(classes="table table-bordered table-striped", border=0))

tree.plot_tree(classifier, max_depth=3, fontsize=15)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())