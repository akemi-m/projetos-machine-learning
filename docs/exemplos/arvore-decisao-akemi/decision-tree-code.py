import matplotlib.pyplot as plt
import pandas as pd

from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

df = pd.read_csv('credit_score_classification.csv', low_memory=False)

# ajustes de data type e limpezas primárias
df['ID'] = df['ID'].astype('string')
df['Customer_ID'] = df['Customer_ID'].astype('string')
df['Month'] = df['Month'].astype('string')
df['Name'] = df['Name'].astype('string')
df['SSN'] = df['SSN'].astype('string')
df['Occupation'] = df['Occupation'].astype('string')
df['Type_of_Loan'] = df['Type_of_Loan'].astype('string')
df['Payment_Behaviour'] = df['Payment_Behaviour'].astype('string')
df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].astype('string')
df['Credit_Utilization_Ratio'] = df['Credit_Utilization_Ratio'].astype('float')

df['Age'] = df['Age'].str.replace('_', '').astype('int')
df['Annual_Income'] = df['Annual_Income'].str.replace('_', '').astype('float')
df['Num_of_Loan'] = df['Num_of_Loan'].str.replace('_', '').astype('int')
df['Outstanding_Debt'] = df['Outstanding_Debt'].str.replace('_', '').astype('float')
df['Amount_invested_monthly'] = df['Amount_invested_monthly'].str.replace('_', '').astype('float')
df['Monthly_Balance'] = df['Monthly_Balance'].str.replace('_', '').astype('float')
df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].str.replace('_', '').astype('float').fillna(0).astype('int')
df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].str.replace('_', '0').astype('float').fillna(0)
df['Credit_Mix'] = df['Credit_Mix'].str.replace('_', 'Não informado').astype('string')
df['Credit_History_Age'] = df['Credit_History_Age'].fillna('Não informado').astype('string')

df['Credit_Score'] = df['Credit_Score'].astype('category')

# feature engineering e limpeza em base de range de números
bins = [0 * 12, 2500 * 12, 4000 * 12, 6000 * 12, 10000 * 12, float('inf')]
labels = ['low', 'lower_middle', 'middle', 'upper_middle', 'high']
df['Annual_Income_Ajustado'] = pd.cut(df['Annual_Income'], bins=bins, labels=labels)

bins = [-1, 10, 30, 50, 75, 100, float('inf')]
labels = ['muito_baixo', 'baixo', 'moderado', 'alto', 'muito_alto', 'estourado']
df['Credit_Utilization_Ratio_Ajustado'] = pd.cut(df['Credit_Utilization_Ratio'], bins=bins, labels=labels)

bins = [0, 2500, 4000, 6000, 10000, float('inf')]
labels = ['low', 'lower_middle', 'middle', 'upper_middle', 'high']
df['Monthly_Inhand_Salary_Ajustado'] = pd.cut(df['Monthly_Inhand_Salary'], bins=bins, labels=labels)

df = df.loc[(df['Num_Bank_Accounts'] > 0) & (df['Num_Bank_Accounts'] <= 10)]

df = df.loc[(df['Age'] >= 0) & (df['Age'] <= 120)]
bins = [0, 12, 19, 35, 55, 75, float('inf')]
labels = ['crianca', 'adolescente', 'jovem_adulto', 'adulto', 'meia_idade', 'idoso']
df['Age_Ajustado'] = pd.cut(df['Age'], bins=bins, labels=labels)

bins = [0, 500, 1500, 3000, 5000, float('inf')]
labels = ['muito_baixo', 'baixo', 'medio', 'alto', 'muito_alto']
df['Total_EMI_per_month_Ajustado'] = pd.cut(df['Total_EMI_per_month'], bins=bins, labels=labels)

df = df.loc[(df['Num_Credit_Card'] > 0) & (df['Num_Credit_Card'] <= 10)]

df = df.loc[df['Interest_Rate'] <= 30]

df = df.loc[(df['Num_of_Loan'] >= 0) & (df['Num_of_Loan'] <= 10)]

df = df.loc[df['Delay_from_due_date'] >= 0]

df = df.loc[(df['Num_of_Delayed_Payment'] >= 0) & (df['Num_of_Delayed_Payment'] <= 25)]

df = df.loc[df['Num_Credit_Inquiries'] <= 12]

df['years'] = df['Credit_History_Age'].str.extract(r'(\d+)\s*Years')
df['months'] = df['Credit_History_Age'].str.extract(r'(\d+)\s*Months')
df['years'] = pd.to_numeric(df['years'], errors='coerce').fillna(0).astype(int)
df['months'] = pd.to_numeric(df['months'], errors='coerce').fillna(0).astype(int)
df['Credit_History_Age_Ajustado'] = df['years'] * 12 + df['months']

df = df.loc[df['Total_EMI_per_month'] <= 200]

bins = [0, 1000, 10000, 50000, 250000, float('inf')]
labels = ['very_low', 'low', 'medium', 'high', 'very_high']
df['Amount_invested_monthly_Ajustado'] = pd.cut(df['Amount_invested_monthly'], bins=bins, labels=labels)

bins = [-float('inf'), 0, 1000, 5000, 20000, float('inf')]
labels = ['negative', 'low', 'moderate', 'high', 'very_high']
df['Monthly_Balance_Ajustado'] = pd.cut(df['Monthly_Balance'], bins=bins, labels=labels)

# remover colunas que não entram em modelos
df = df.drop(['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Type_of_Loan', 'Credit_History_Age', 
              'Monthly_Balance', 'Amount_invested_monthly', 'Age', 'Monthly_Inhand_Salary', 'Annual_Income', 'months', 'years', 'Credit_Utilization_Ratio', 'Total_EMI_per_month'], axis=1)

# limpeza de colunas categóricas e mudança de tipo
indices_para_remover = df[df["Occupation"].str.contains("_", na=False)].index
df.drop(index=indices_para_remover, inplace=True)

indices_para_remover = df[df["Credit_Mix"].str.contains("Não informado", na=False)].index
df.drop(index=indices_para_remover, inplace=True) 

indices_para_remover = df[df["Payment_of_Min_Amount"].str.contains("NM", na=False)].index
df.drop(index=indices_para_remover, inplace=True)

indices_para_remover = df[df["Payment_Behaviour"].str.contains("!@9#%8", na=False)].index
df.drop(index=indices_para_remover, inplace=True)

df['Occupation'] = df['Occupation'].astype('category')
df['Credit_Mix'] = df['Credit_Mix'].astype('category')
df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].astype('category')
df['Payment_Behaviour'] = df['Payment_Behaviour'].astype('category')

df['Annual_Income_Ajustado'] = df['Annual_Income_Ajustado'].astype('category')
df['Age_Ajustado'] = df['Age_Ajustado'].astype('category')
df['Monthly_Inhand_Salary_Ajustado'] = df['Monthly_Inhand_Salary_Ajustado'].astype('category')
df['Credit_History_Age_Ajustado'] = df['Credit_History_Age_Ajustado'].astype('category')
df['Amount_invested_monthly_Ajustado'] = df['Amount_invested_monthly_Ajustado'].astype('category')
df['Monthly_Balance_Ajustado'] = df['Monthly_Balance_Ajustado'].astype('category')
df['Credit_Utilization_Ratio_Ajustado'] = df['Credit_Utilization_Ratio_Ajustado'].astype('category')
df['Total_EMI_per_month_Ajustado'] = df['Total_EMI_per_month_Ajustado'].astype('category')

# remoção de linhas com valores nulos
df = df.dropna()

# resetar index
df = df.reset_index(drop=True)

# Modelo de Árvore de Decisão

label_encoder = LabelEncoder()
# aplicar label encoder na target
df['Credit_Score'] = label_encoder.fit_transform(df['Credit_Score'])

# todas as colunas menos o target
x = df.drop("Credit_Score", axis=1)

# aplicar label encoder nas categóricas
x['Occupation'] = label_encoder.fit_transform(x['Occupation'])
x['Credit_Mix'] = label_encoder.fit_transform(x['Credit_Mix'])
x['Payment_of_Min_Amount'] = label_encoder.fit_transform(x['Payment_of_Min_Amount'])
x['Payment_Behaviour'] = label_encoder.fit_transform(x['Payment_Behaviour'])
x['Age_Ajustado'] = label_encoder.fit_transform(x['Age_Ajustado'])
x['Annual_Income_Ajustado'] = label_encoder.fit_transform(x['Annual_Income_Ajustado'])
x['Monthly_Inhand_Salary_Ajustado'] = label_encoder.fit_transform(x['Monthly_Inhand_Salary_Ajustado'])
x['Credit_History_Age_Ajustado'] = label_encoder.fit_transform(x['Credit_History_Age_Ajustado'])
x['Amount_invested_monthly_Ajustado'] = label_encoder.fit_transform(x['Amount_invested_monthly_Ajustado'])
x['Monthly_Balance_Ajustado'] = label_encoder.fit_transform(x['Monthly_Balance_Ajustado'])
x['Credit_Utilization_Ratio_Ajustado'] = label_encoder.fit_transform(x['Credit_Utilization_Ratio_Ajustado'])
x['Total_EMI_per_month_Ajustado'] = label_encoder.fit_transform(x['Total_EMI_per_month_Ajustado'])

# apenas a target
y = df["Credit_Score"]

# Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

cm = metrics.confusion_matrix(y_test, y_pred)

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2])
disp.plot(cmap="Blues")
plt.show()

# Avaliar o modelo
accuracy = classifier.score(x_test, y_test)
print(metrics.classification_report(y_test, y_pred, digits=4))
print(f"Accuracy: {accuracy:.2f}")
tree.plot_tree(classifier)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())