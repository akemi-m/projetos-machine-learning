import pandas as pd


df = pd.read_csv('https://raw.githubusercontent.com/akemi-m/projetos-machine-learning/refs/heads/main/docs/projeto-1/covid_data.csv')

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

print(d.to_markdown(index=False))
