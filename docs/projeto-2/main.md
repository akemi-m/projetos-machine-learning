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

# **Introdução**

Este projeto tem como objetivo **explorar diferentes técnicas de aprendizado supervisionado e não supervisionado**, aplicadas a um conjunto de dados real relacionado a cartas colecionáveis.

Serão utilizados algoritmos como:  
- **Random Forest**,  
- **KNN**,  
- **Regressão Linear**,  
além de outros métodos complementares conforme a necessidade analítica.

Com a aplicação dessas abordagens, busca-se compreender **os pontos fortes e limitações de cada técnica**, bem como avaliar seu desempenho em diferentes cenários preditivos e exploratórios.

---

## **Exploração de Dados**

A análise inicial do conjunto de dados envolve a **descrição detalhada das variáveis**, cálculos de **estatísticas descritivas** e produção de **visualizações** que permitam entender o comportamento, distribuição e relevância das informações contidas no dataset.

---

### **Contexto da Base de Dados**

O dataset utilizado neste projeto é composto por um **grande volume de registros de cartas colecionáveis**, incluindo atributos de preço, tipo, acabamento e coleção.

No total, são disponibilizadas **82.423 observações (linhas)** distribuídas em **5 variáveis (colunas)**.

Essas variáveis são do tipo **numérico** (como valores monetários) e **categórico**, contendo múltiplas classificações de cartas, elementos e versões foil.

Entre os tipos de atributos presentes:

- **Atributos numéricos** — representando valores monetários das cartas.  
- **Atributos categóricos** — como **tipo de carta**, **elemento**, **coleção** e **tipo de acabamento (foil)**.  
- O dataset não possui variáveis booleanas codificadas numericamente, mas apresenta **categorias bem definidas**, adequadas para análises supervisionadas e não supervisionadas.

Como o conjunto possui estrutura consistente e volume expressivo, **não houve necessidade de criação de dados sintéticos**. O dataset real foi integralmente utilizado para a fase de exploração.

Para experimentos envolvendo técnicas preditivas — caso necessário — a variável **`valor`** pode ser tratada como **target** em problemas de regressão, enquanto variáveis como **elemento**, **coleção** ou **foil** podem ser utilizadas como alvos em cenários de classificação.


### Descrição e Estatísticas Descritivas das colunas

| Variável      | Descrição | Estatísticas Descritivas (%) |
|---------------|-----------|-------------------------------|
| **valor** | Valor monetário da carta. | Média: 43,24<br>Mediana: 8,40<br>Mínimo: 0,07<br>Máximo: 12.276,10 |
| **elemento** | Tipo elemental associado ao Pokémon ou carta. | Grama: 11,94%<br>Normal: 11,94%<br>Psíquica: 10,70%<br>Água: 10,61%<br>Luta: 10,50%<br>Fogo: 7,47%<br>Escuridão: 7,37%<br>Raio: 5,96%<br>Metal: 5,11%<br>Dragão: 2,28% |
| **foil** | Tipo de acabamento da carta (variações de brilho). | Normal: 33,06%<br>Foil: 25,94%<br>Reverse Foil: 24,67%<br>Pokeball Foil: 16,27%<br>Masterball Foil: 0,06% |
| **tipoCarta** | Categoria da carta dentro do jogo. | Pokémon – Basic: 47,91%<br>Pokémon – Stage 1: 26,51%<br>Pokémon – Stage 2: 7,41% |
| **colecao** | Coleção/expansão à qual a carta pertence. | Megaevolução: 21,83%<br>Amigos da Jornada: 21,04%<br>Evoluções Prismáticas: 19,13%<br>Raio Preto: 13,17%<br>Fogo Branco: 13,02%<br>151: 11,82% |



## Pré Processamento 

Como os dados foram obtidos a partir de scripts de raspagem desenvolvidos pelo grupo, já existe um bom nível de padronização nas colunas. Durante a etapa de inspeção inicial, verificou-se que apenas a variável elemento necessitava de pré-processamento. Isso ocorre porque nem todas as cartas possuem um tipo elemental — por exemplo, cartas de Treinador, Itens e algumas cartas especiais não apresentam elemento associado.

Para tratar essa inconsistência, a coluna foi convertida para o tipo string e os valores ausentes foram substituídos pela categoria "None", representando corretamente a ausência de elemento. Dessa forma, evita-se perda de informação e garante-se a consistência sem introduzir distorções nos modelos a serem aplicados.

=== "Result"

    ```python exec="1" html="0"
    --8<-- "docs/projeto-2/pre.py"
    ```
=== "Prep Code"

    ```python
    --8<-- "docs/projeto-2/pre.py"
    ```



## Divisão dos Dados

O conjunto de dados foi dividido em 70% para treino e 30% para validação, garantindo que os modelos fossem treinados em partes significativa das observações, mas ainda avaliados em dados não vistos. O uso do conjunto de validação tem como objetivo detectar e reduzir o risco de overfitting.



## Treinamento dos modelos

### Random Forest

=== "Result"

    ```python exec="1" html="1"
    --8<-- "docs/projeto-2/arvore.py"
    ```
=== "Prep Code"

    ```python
    --8<-- "docs/projeto-2/arvore.py"
    ```

### KNN

=== "Result"

    ```python exec="1" html="1"
    --8<-- "docs/projeto-2/knn.py"
    ```
=== "Prep Code"

    ```python
    --8<-- "docs/projeto-2/knn.py"
    ```

### Linear

=== "Result"

    ```python exec="1" html="1"
    --8<-- "docs/projeto-2/linear.py"
    ```
=== "Prep Code"

    ```python
    --8<-- "docs/projeto-2/linear.py"
    ```

## Avaliação

=== "Result"

    ```python exec="1" html="1"
    --8<-- "docs/projeto-2/avalia.py"
    ```
=== "Prep Code"

    ```python
    --8<-- "docs/projeto-2/avalia.py"
    ```

## Avaliação dos Modelos

A avaliação dos modelos foi realizada combinando a visualização das predições no espaço PCA e as métricas quantitativas de desempenho (R² e RMSE). Nos gráficos de PCA, observamos que tanto o **Random Forest** quanto o **KNN** apresentam baixa variação nas cores, indicando que ambos tendem a prever valores muito próximos entre si, mesmo em regiões distintas do espaço reduzido — um sinal claro de **subajuste (underfitting)**. O desempenho numérico confirma essa limitação: o **KNN** apresentou o menor R² do grupo e o maior RMSE, indicando dificuldade em capturar relações relevantes no dataset. 

O **Random Forest**, apesar de também apresentar um padrão visual homogêneo no PCA, foi o modelo que obteve **melhor desempenho relativo**, alcançando o maior R² (ainda baixo) e o menor RMSE, mostrando ligeira superioridade na modelagem da variabilidade dos preços. 

Já a **Regressão Linear** apresentou um comportamento diferente: no PCA, suas predições exibem gradientes de cor bem definidos, refletindo maior sensibilidade às mudanças nas componentes principais. No entanto, seu R² reduzido demonstra que, apesar de capturar alguma estrutura linear nos dados, o modelo não explica adequadamente a variabilidade da variável resposta. 

Em conjunto, os resultados mostram que **nenhum dos modelos conseguiu generalizar bem**, sugerindo necessidade de engenharia de features mais robusta, uso de técnicas avançadas ou ajustes adicionais para melhorar a capacidade preditiva.
