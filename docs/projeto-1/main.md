<style>
    table {
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 14px;
    text-align: center;
    }
    table td, table th {
    padding: 8px 12px;
    }
</style>

Este Projeto I tem como objetivo explorar diferentes técnicas de aprendizado supervisionado e não supervisionado, aplicadas a um conjunto de dados real.

Serão utilizados algoritmos como:

- **Decision Tree**: modelo interpretável e estruturado em forma de árvore de decisão, adequado para classificação e regressão.  
- **K-Nearest Neighbors (KNN)**: algoritmo baseado em instâncias, que realiza previsões considerando a proximidade entre amostras.  
- **K-Means**: técnica de aprendizado não supervisionado voltada para agrupar dados em clusters de acordo com sua similaridade.  

A partir da aplicação desses métodos, o projeto busca compreender os pontos fortes e limitações de cada abordagem, bem como analisar seu desempenho em diferentes cenários.

## Exploração de dados

A **análise inicial** do conjunto de [dados](https://www.kaggle.com/datasets/meirnizri/covid19-dataset) inclui a **descrição da natureza das variáveis, estatísticas descritivas e visualizações** para compreender a distribuição e relevância das informações.

### Contexto da Base de Dados

O dataset utilizado neste projeto contém um **grande volume de informações anonimizadas de pacientes**, abrangendo tanto características gerais quanto condições pré-existentes.  

No total, são disponibilizadas **21 variáveis (colunas)** que representam atributos clínicos e demográficos, e **1.048.575 registros (linhas)**, correspondentes a pacientes únicos.  

As variáveis são mistas, incluindo **atributos numéricos e categóricos**, além de um conjunto expressivo de **features Booleanas**, nas quais:  
- O valor **1 indica "sim"**  
- O valor **2 indica "não"**  
- Valores como **97** e **99** representam **dados ausentes ou não informados**  

Para os experimentos com **modelos supervisionados**, a variável **target** será **`classification`**, representando o resultado do teste de COVID-19 dos pacientes.  

### Descrição e Estatísticas Descritivas das colunas

| Variável                  | Descrição                                                                                             | Estatísticas Descritivas (%)                                                                     |
| ------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **USMER**                 | Indica se o paciente foi tratado em unidades médicas de primeiro, segundo ou terceiro nível.          | 1º nível: 36,8%<br>2º nível: 63,2%                                                               |
| **MEDICAL\_UNIT**         | Tipo de instituição do Sistema Nacional de Saúde que prestou o atendimento.                           | Unidade 12: 57,5%<br>Unidade 4: 30,0%<br>Demais unidades: 12,5%                                  |
| **SEX**                   | 1 para feminino e 2 para masculino.                                                                   | Feminino: 50,1%<br>Masculino: 49,9%                                                              |
| **PATIENT\_TYPE**         | Tipo de atendimento recebido na unidade. 1 para alta domiciliar e 2 para hospitalização.              | Alta domiciliar: 80,9%<br>Hospitalização: 19,1%                                                  |
| **DATE\_DIED**            | Se o paciente faleceu, indica a data da morte; caso contrário, 9999-99-99.                            | Vivos: 92,7%<br>Óbitos registrados: 7,3%                                                         |
| **INTUBED**               | Se o paciente foi conectado ao respirador.                                                            | Desconhecido (97): 80,9%<br>Não intubado (2): 15,2%<br>Intubado (1): 3,2%<br>Ignorado (99): 0,7% |
| **PNEUMONIA**             | Se o paciente já apresentava inflamação nos alvéolos pulmonares.                                      | Não: 85,1%<br>Sim: 13,4%<br>Ignorado: 1,5%                                                       |
| **AGE**                   | Idade do paciente.                                                                                    | Moda: 30 anos (2,6%)<br>Faixa 28–34: \~7,3%                                                      |
| **PREGNANT**              | Se o paciente está grávida.                                                                           | Não aplicável: 50,0%<br>Não grávida: 48,9%<br>Grávida: 0,8%<br>Ignorado: 0,3%  |
| **DIABETES**              | Se o paciente tem diabetes.                                                                           | Não: 87,7%<br>Sim: 11,9%<br>Ignorado: 0,3%                                                       |
| **COPD**                  | Se o paciente tem DPOC.                                                                               | Não: 98,3%<br>Sim: 1,4%<br>Ignorado: 0,3%                                                        |
| **ASTHMA**                | Se o paciente tem asma.                                                                               | Não: 96,7%<br>Sim: 3,0%<br>Ignorado: 0,3%                                                        |
| **INMSUPR**               | Se o paciente é imunossuprimido.                                                                      | Não: 98,3%<br>Sim: 1,4%<br>Ignorado: 0,3%                                                        |
| **HIPERTENSION**          | Se o paciente tem hipertensão.                                                                        | Não: 84,1%<br>Sim: 15,5%<br>Ignorado: 0,3%                                                       |
| **OTHER\_DISEASE**        | Se o paciente tem outra doença.                                                                       | Não: 96,8%<br>Sim: 2,7%<br>Ignorado: 0,5%                                                        |
| **CARDIOVASCULAR**        | Se o paciente tem doença cardíaca ou vascular.                                                        | Não: 97,7%<br>Sim: 2,0%<br>Ignorado: 0,3%                                                        |
| **OBESITY**               | Se o paciente é obeso.                                                                                | Não: 84,4%<br>Sim: 15,2%<br>Ignorado: 0,3%                                                       |
| **RENAL\_CHRONIC**        | Se o paciente tem doença renal crônica.                                                               | Não: 97,9%<br>Sim: 1,8%<br>Ignorado: 0,3%                                                        |
| **TOBACCO**               | Se o paciente faz uso de tabaco.                                                                      | Não: 91,7%<br>Sim: 8,0%<br>Ignorado: 0,3%                                                        |
| **CLASIFFICATION\_FINAL** | Resultados do teste de covid. Valores 1-3 = positivo em diferentes graus; ≥4 = negativo/inconclusivo. | Positivo (1–3): 37,4%<br>Negativo/Inconclusivo (≥4): 62,6%                                       |
| **ICU**                   | Indica se o paciente foi internado em UTI.                                                            | Desconhecido: 80,9%<br>Não: 16,8%<br>Sim: 1,6%<br>Ignorado: 0,7%               |


## Pré Processamento 

Primeiramente, verificamos a presença de valores nulos em todas as colunas. Na coluna `AGE` havia poucos valores ausentes em relação ao total, e por isso optamos por substituí-los pela mediana. Nas colunas categóricas, os valores ausentes foram substituídos pela moda.
Em algumas variáveis, como `DATE_DIED`, identificamos a presença de “nulos lógicos” (valores ausentes que representam situações específicas, como “paciente não faleceu”). Para esses casos, foi feita uma análise contextual antes de qualquer imputação para evitar distorcer a informação.
Por fim, aplicamos a normalização Min–Max à coluna `AGE` para colocá-la na mesma escala das variáveis categóricas codificadas, favorecendo o desempenho de algoritmos de machine learning baseados em distância ou gradientes.

=== "Result"

    ```python exec="1" html="0"
    --8<-- "docs/projeto-1/pre.py"
    ```
=== "Prep Code"

    ```python
    --8<-- "docs/projeto-1/pre.py"
    ```



## Divisão dos Dados

O conjunto de dados foi dividido em 70% para treino e 30% para validação, garantindo que os modelos fossem treinados em partes significativa das observações, mas ainda avaliados em dados não vistos. O uso do conjunto de validação tem como objetivo detectar e reduzir o risco de overfitting.



## Treinamento dos modelos

### Árvore de Decisão

=== "Result"

    ```python exec="1" html="1"
    --8<-- "docs/projeto-1/arvore.py"
    ```
=== "Prep Code"

    ```python
    --8<-- "docs/projeto-1/arvore.py"
    ```

### KNN

=== "Result"

    ```python exec="1" html="1"
    --8<-- "docs/projeto-1/knn.py"
    ```
=== "Prep Code"

    ```python
    --8<-- "docs/projeto-1/knn.py"
    ```