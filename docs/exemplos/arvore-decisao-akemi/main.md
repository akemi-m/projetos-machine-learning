As **árvores de decisão** são uma técnica popular de **aprendizado de máquina supervisionado** usada para classificação e regressão.

Neste projeto, foi utilizada a base de dados **Credit Score Classification**, que reúne informações bancárias essenciais e uma grande quantidade de registros relacionados a crédito.

O objetivo principal é **aplicar árvores de decisão** para segmentar indivíduos em diferentes **faixas de score de crédito**, contribuindo para a identificação de perfis de risco e para o apoio em processos de decisão financeira.

## Exploração dos Dados

A **análise inicial** do conjunto de dados inclui a **descrição da natureza das variáveis, estatísticas descritivas e visualizações** para compreender a distribuição e relevância das informações.

### Descrição e estatísticas descritivas das colunas

A base contém **28 colunas**, resumidas abaixo:

| Coluna                         | Descrição                                       | Observações / Relevância                                                                     |
| ------------------------------ | ----------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `ID`                           | Identificação única de um registro              | **Irrelevante**, não agrega valor estatístico                                                |
| `Customer_ID`                  | Identificação única de um cliente               | **Irrelevante**, mesma justificativa da anterior                                             |
| `Month`                        | Mês do ano                                      | Bem distribuído, cada mês ≈ **12,5%**                                                        |
| `Name`                         | Nome do cliente                                 | **Irrelevante**, não agrega valor estatístico                                                |
| `Age`                          | Idade do cliente                                | Pode ser transformada em faixas etárias (_feature engineering_)                              |
| `SSN`                          | Número de seguridade social                     | **Irrelevante**, valor único por cliente                                                     |
| `Occupation`                   | Ocupação do cliente                             | Bem distribuída, média ≈ **6,6%**                                                            |
| `Annual_Income`                | Renda anual                                     | Valores quase únicos; útil se agrupada em faixas ou escalas (_feature engineering_)          |
| `Monthly_Inhand_Salary`        | Salário mensal líquido                          | Similar à renda anual; pode ser útil após agregação ou categorização (_feature engineering_) |
| `Num_Bank_Accounts`            | Número de contas bancárias                      | Equilibrado, maior frequência ≈ **13%**                                                      |
| `Num_Credit_Card`              | Número de cartões de crédito                    | Bem distribuído, maior representatividade ≈ **18%**                                          |
| `Interest_Rate`                | Taxa de juros do cartão                         | Distribuição uniforme, cada valor ≈ **5%**                                                   |
| `Num_of_Loan`                  | Número de empréstimos obtidos                   | Equilibrado, maior concentração ≈ **15%**                                                    |
| `Type_of_Loan`                 | Tipo de empréstimo                              | Diversidade de valores, cada tipo ≈ **1,44%**                                                |
| `Delay_from_due_date`          | Média de dias de atraso                         | Bem distribuído, cada valor ≈ **3,6%**                                                       |
| `Num_of_Delayed_Payment`       | Quantidade de pagamentos em atraso              | Maior representatividade ≈ **8,6%**                                                          |
| `Changed_Credit_Limit`         | Alteração percentual do limite de crédito       | **Pouco relevante isoladamente**, útil em _feature engineering_                              |
| `Num_Credit_Inquiries`         | Número de consultas de crédito                  | Equilibrado, maior frequência ≈ **11,5%**                                                    |
| `Credit_Mix`                   | Composição de créditos                          | Balanceado, maior representatividade ≈ **36,5%**                                             |
| `Outstanding_Debt`             | Dívida pendente (USD)                           | **Pouco útil isoladamente**, mas explorável via _feature engineering_                        |
| `Credit_Utilization_Ratio`     | Taxa de utilização do crédito                   | Útil se agrupada em faixas ou transformada em variáveis derivadas (_feature engineering_)    |
| `Credit_History_Age`           | Tempo de histórico de crédito                   | **Dados inconsistentes**, mas pode ser transformado em meses/anos (_feature engineering_)    |
| `Payment_of_Min_Amount`        | Indica se paga apenas o mínimo                  | Distribuição clara, maior representatividade ≈ **52,3%**                                     |
| `Total_EMI_per_month_Ajustado` | Valor total de pagamentos mensais de EMI (USD)  | Útil após normalização ou criação de faixas (_feature engineering_)                          |
| `Amount_invested_monthly`      | Valor investido mensalmente (USD)               | Pode ser agregado em categorias ou usado em índices relativos (_feature engineering_)        |
| `Payment_Behaviour`            | Comportamento de pagamento                      | Destaque para **25,5%** dos registros                                                        |
| `Monthly_Balance`              | Saldo mensal (USD)                              | Pode ser transformado em faixas ou indicadores derivados (_feature engineering_)             |
| `Credit_Score`                 | Faixa de pontuação (_Poor_, _Standard_, _Good_) | Distribuição equilibrada, maior representatividade ≈ **53,2%**                               |

<!-- ### Descrição e estatísticas descritivas das colunas

A base contém **28 colunas**, descritas a seguir:

- `ID`: Identificação única de um registro. **Irrelevante**, pois não se repete e não agrega valor estatístico.
- `Customer_ID`: Identificação única de um cliente. **Irrelevante**, mesma justificativa da coluna anterior.
- `Month`: Mês do ano. Valores bem distribuídos, cada mês representando aproximadamente **12,5%** da base.
- `Name`: Nome do cliente. **Irrelevante**, pois não se repete e não agrega valor estatístico.
- `Age`: Idade do cliente. Distribuição equilibrada, com a maior concentração em apenas **3%** dos registros.
- `SSN`: Número de seguridade social. **Irrelevante**, pois é único por cliente.
- `Occupation`: Ocupação do cliente. Valores bem distribuídos, com cada ocupação representando em média **6,6%** da base.
- `Annual_Income`: Renda anual. A distribuição sugere comportamento similar a um identificador, devido ao excesso de casas decimais, tornando cada valor praticamente único.
- `Monthly_Inhand_Salary`: Salário mensal líquido. Mesma situação da renda anual, funcionando como um identificador devido à ausência de repetição.
- `Num_Bank_Accounts`: Número de contas bancárias do cliente. Distribuição equilibrada, com a maior frequência em **13%** dos registros.
- `Num_Credit_Card`: Número de cartões de crédito. Bem distribuído, com a maior representatividade em **18%** da base.
- `Interest_Rate`: Taxa de juros do cartão. Distribuição uniforme, cada valor correspondendo a aproximadamente **5%**.
- `Num_of_Loan`: Número de empréstimos obtidos. Distribuição equilibrada, com a maior concentração em **15%**.
- `Type_of_Loan`: Tipo de empréstimo. Diversidade de valores, cada tipo representando cerca de **1,44%** da base.
- `Delay_from_due_date`: Média de dias de atraso no pagamento. Bem distribuído, com valores uniformes em torno de **3,6%** cada.
- `Num_of_Delayed_Payment`: Média de pagamentos em atraso. Maior representatividade em **8,6%**.
- `Changed_Credit_Limit`: Percentual de alteração no limite de crédito. **Pouco relevante isoladamente**, mas com potencial para _feature engineering_.
- `Num_Credit_Inquiries`: Número de consultas de crédito. Distribuição equilibrada, com a maior frequência em **11,5%**.
- `Credit_Mix`: Classificação da composição de créditos. Distribuição balanceada, com maior representatividade em **36,5%** dos registros.
- `Outstanding_Debt`: Dívida pendente (USD). **Irrelevante isoladamente**, pois cada valor é praticamente único. Contudo, pode ser explorado via _feature engineering_.
- `Credit_Utilization_Ratio`: Taxa de utilização do crédito. **Irrelevante isoladamente**, pois cada valor é praticamente único. Contudo, pode ser explorado via _feature engineering_.
- `Credit_History_Age`: Tempo de histórico de crédito. **Dados inconsistentes**, exigindo pré-processamento. Pode ser útil após _feature engineering_.
- `Payment_of_Min_Amount`: Indica se o cliente paga apenas o valor mínimo. Distribuição clara, com maior representatividade em **52,3%**.
- `Total_EMI_per_month`: Valor total de pagamentos mensais de EMI (USD). **Irrelevante isoladamente**, mas útil após _feature engineering_.
- `Amount_invested_monthly`: Valor investido mensalmente (USD). **Irrelevante isoladamente**, mas útil após _feature engineering_.
- `Payment_Behaviour`: Comportamento de pagamento do cliente. Distribuição com destaque para **25,5%** dos registros.
- `Monthly_Balance`: Saldo mensal do cliente (USD). **Irrelevante isoladamente**, mas útil após _feature engineering_.
- `Credit_Score`: Faixa de pontuação de crédito (_Poor_, _Standard_, _Good_). Distribuição equilibrada, com maior representatividade em **53,2%**. -->

### Tipagem das colunas

O dataset é composto por apenas três tipos de dados: `Object`, `Float64` e `Int64`:

- **Object(20):** `ID`, `Customer_ID`, `Month`, `Name`, `Age`, `SSN`, `Occupation`, `Annual_Income`, `Num_of_Loan`, `Type_of_Loan`, `Num_of_Delayed_Payment`, `Changed_Credit_Limit`, `Outstanding_Debt`, `Credit_History_Age`, `Payment_of_Min_Amount`, `Amount_invested_monthly`, `Monthly_Balance`, `Payment_Behaviour`, `Credit_Score`, `Credit_Mix`.
- **Float64(4):** `Monthly_Inhand_Salary`, `Num_Credit_Inquiries`, `Credit_Utilization_Ratio`, `Total_EMI_per_month`.
- **Int64(4):** `Num_Bank_Accounts`, `Num_Credit_Card`, `Interest_Rate`, `Delay_from_due_date`.

### Variável Alvo

O modelo tem como **variável alvo (target)** a coluna **`Credit_Score`**, que representa a **classificação de crédito** dos clientes.

A distribuição da variável é a seguinte:

- **Standard:** 53.174 (53,2%)
- **Poor:** 28.998 (29,0%)
- **Good:** 17.828 (17,8%)

Essa distribuição evidencia um **leve desbalanceamento de classes**, que deve ser considerado no pré-processamento ou na avaliação do modelo.

### Amostra dos Dados

A seguir, um exemplo com 3 linhas de um total de **100.000 registros** e um gráfico de barras da distribuição de `Credit_Score` em relação a `Payment_Behavior`:

=== "data sample (3/100.000)"

    ```python exec="1"
    --8<-- "./docs/arvore-decisao/dataset.py"
    ```

=== "plot"

    ```python exec="1" html="1"
    --8<-- "./docs/arvore-decisao/decision-tree-plot.py"
    ```

## Pré-processamento

Nesta etapa foi realizada a **limpeza dos dados** e o **tratamento de valores ausentes**, além da **adequação da tipagem das colunas** para que estivessem compatíveis com o modelo.  
Esse processo foi fundamental para garantir consistência e qualidade antes do treinamento da árvore de decisão.

### Alteração de Tipagem dos Dados

Grande parte das colunas do dataset estavam no formato `object`.  
O primeiro passo foi padronizar as tipagens, ajustando para `float64`, `int64` e `category`, conforme a natureza de cada variável.

---

#### Apenas alteração de tipo

Variáveis que puderam ser convertidas diretamente com `astype()` no **Pandas**:

- `ID`
- `Customer_ID`
- `Month`
- `Name`
- `SSN`
- `Occupation`
- `Type_of_Loan`
- `Payment_Behaviour`
- `Payment_of_Min_Amount`
- `Credit_Score`

---

#### Necessidade de limpeza antes da conversão

Colunas que apresentavam valores inválidos (ex.: `'_'`) e precisaram de tratamento prévio:

- `Num_of_Loan`
- `Outstanding_Debt`
- `Num_of_Delayed_Payment`
- `Changed_Credit_Limit`
- `Credit_Mix`

---

#### Feature engineering (transformação em categorias)

Variáveis com grande dispersão foram **agrupadas em faixas de valores** para reduzir ruído e melhorar a identificação de padrões:

- `Age`
- `Annual_Income`
- `Monthly_Inhand_Salary`
- `Credit_History_Age`
- `Total_EMI_per_month_Ajustado`
- `Credit_Utilization_Ratio`
- `Amount_invested_monthly`
- `Monthly_Balance`

> Essas variáveis, na forma original, apresentavam valores praticamente únicos, funcionando como identificadores.  
> A categorização permitiu simplificar a análise e aumentar a relevância para o modelo.

### Colunas Retiradas do Modelo

Durante o processo de preparação dos dados, algumas variáveis foram consideradas **irrelevantes** ou **inadequadas** para a análise de risco. A exclusão foi feita para reduzir **ruído**, melhorar a **qualidade das features** e aumentar a **robustez preditiva** do modelo.

As remoções ocorreram pelos seguintes motivos:

- **`ID`, `Customer_ID`, `Name`, `SSN`**  
  Identificadores únicos que não possuem valor preditivo, apenas acrescentando variabilidade desnecessária.

- **`Month`**  
  A variável não apresentou relação consistente com a pontuação de crédito, sendo descartada por falta de interpretabilidade.

- **`Type_of_Loan`**  
  Apresentou registros inconsistentes, com baixa padronização dos valores, dificultando a extração de informação relevante.

- **`Credit_History_Age`, `Monthly_Balance`, `Amount_invested_monthly`, `Age`, `Monthly_Inhand_Salary`, `Annual_Income`**  
  Apesar de importantes, essas variáveis possuíam **valores ruidosos ou dispersos** em sua forma original.  
  Foram reaproveitadas por meio de **_feature engineering_**, originando versões ajustadas e mais informativas.

---

### Colunas Utilizadas no Modelo

Após os ajustes, foram mantidas **21 colunas** efetivas para a modelagem. A variável alvo (`Credit_Score`) foi separada do conjunto de preditores.

- **Numéricas Contínuas – Float64 (5):**  
  `Num_Credit_Inquiries`, `Credit_Utilization_Ratio`, `Total_EMI_per_month`, `Changed_Credit_Limit`, `Outstanding_Debt`.

- **Numéricas Discretas – Int64 (6):**  
  `Num_Bank_Accounts`, `Num_Credit_Card`, `Interest_Rate`, `Delay_from_due_date`, `Num_of_Loan`, `Num_of_Delayed_Payment`.

- **Categóricas – Category (10):**  
  `Credit_History_Age_Ajustado`, `Monthly_Balance_Ajustado`, `Amount_invested_monthly_Ajustado`, `Age_Ajustado`,  
  `Monthly_Inhand_Salary`, `Annual_Income`, `Occupation`, `Credit_Mix`, `Payment_of_Min_Amount`, `Payment_Behaviour`.

### Linhas Retiradas do Modelo

Foram removidas **71.635 linhas** que continham valores nulos ou nulos lógicos, resultando em **28.365 registros** disponíveis para o treinamento do modelo.

## Divisão dos Dados

Os dados foram divididos em conjuntos de treino e teste utilizando a função `train_test_split` do **scikit-learn**:

- **70%** da base para treino
- **30%** da base para teste

## Treinamento do Modelo

O modelo utilizado foi a **Decision Tree (Árvore de Decisão)**, considerando a variável alvo `Credit_Score`.

=== "output"

    ```python exec="1" html="1"
    --8<-- "./docs/arvore-decisao/decision-tree-results.py"
    ```

=== "code"

    ```python exec="0"
    --8<-- "./docs/arvore-decisao/decision-tree-code.py"
    ```

=== "data sample (6/28.365)"

    ```python exec="1"
    --8<-- "./docs/arvore-decisao/decision-tree-dataset.py"
    ```

## Avaliação do Modelo

A avaliação do desempenho do modelo foi realizada utilizando métricas apropriadas para classificação.

**Acurácia:** o modelo apresentou **≈70%** de acertos, indicando um desempenho satisfatório considerando o conjunto de teste.

### Importância das Features

Analisando o impacto de cada variável na decisão da árvore de classificação, as mais relevantes foram:

- `Outstanding_Debt` – 0,208
- `Credit_Mix` – 0,133
- `Credit_History_Age_Ajustado` – 0,095
- `Changed_Credit_Limit` – 0,088
- `Delay_from_due_date` – 0,086

Essas informações são úteis para **interpretabilidade do modelo**, permitindo compreender quais atributos têm maior influência na classificação do `Credit_Score`.

## Relatório Final

O projeto teve como objetivo aplicar **árvores de decisão** para classificar clientes em faixas de **pontuação de crédito**, utilizando um conjunto de dados com informações financeiras e bancárias.

Durante o processo, foram realizadas etapas importantes de **exploração e pré-processamento dos dados**, incluindo:

- Limpeza e padronização das variáveis;
- Conversão de tipos de dados (`float64`, `int64`, `category`) para adequação ao modelo;
- Remoção de colunas irrelevantes e tratamento de valores nulos ou inconsistentes;
- Aplicação de **feature engineering**, transformando variáveis dispersas em categorias significativas.

Após essas etapas, o modelo foi treinado com **28.365 registros** e **21 variáveis preditivas**. A base foi dividida em **70% para treino** e **30% para teste**, utilizando uma árvore de decisão.
O modelo apresentou **≈70% de acurácia** no conjunto de teste, com destaque para as features mais relevantes na classificação:

- `Outstanding_Debt`
- `Credit_Mix`
- `Credit_History_Age_Ajustado`
- `Changed_Credit_Limit`
- `Delay_from_due_date`

Esses resultados demonstram que o modelo consegue identificar padrões significativos no histórico financeiro dos clientes, oferecendo **insights relevantes para análise de risco**.

Como próximos passos, é recomendada a exploração de **hiperparâmetros**, técnicas de **balanceamento de classes** e análise de **interações entre variáveis**, a fim de melhorar ainda mais a performance e a robustez do modelo.
