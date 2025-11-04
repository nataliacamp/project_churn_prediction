# Análise Preditiva do Churn em uma empresa de telecomunicações

Este projeto utiliza dados de uma empresa do setor de telecomunicações dos Estados Unidos, que representa um caso realista do mercado competitivo de provedores de telefonia, internet, TV por assinatura e streaming. Operar em um ambiente tão competitivo torna a retenção de clientes crucial para a sustentabilidade do negócio. Sendo assim, a análise preditiva de cancelamento não se mostra apenas uma ferramenta analítica, mas uma estratégia de negócio essencial para a sobrevivência e crescimento em um mercado saturado e com baixa fidelidade do consumidor. 

Os objetivos desse projeto foram, portanto:
- Identificar os **principais fatores** que influenciam o cancelamento do serviço;
- Desenvolver **modelos preditivos** para avaliação do churn;
- Fornecer insights acionáveis para **estratégias de retenção**;
- Comparar **desempenho de dois diferentes algoritmos** de ML.

## Tecnologias e Metodologias utilizadas
Como linguagem base, foi utilizado **Python**, a fim de garantir manutenibilidade e escalabilidade do projeto. As bibliotecas **Pandas** e **NumPy** foram utilizadas para manipulação eficiente de dados em larga escala. O **Scikit-learn** foi escolhido como framework principal para o machine learning e as ferramentas **Matplotlib** e **Seaborn** foram empregadas para criação dos gráficos.

Em relação à **Metodologia** utilizada, podem ser sumarizados em quatro etapas principais
1. **Análise exploratória (EDA)**  
Investigação inicial dos dados para compreender distribuições, identificar valores ausentes e detectar correlações iniciais entre variáveis e o cancelamento.
2. **Pré-processamento**  
Etapa de limpeza e preparação dos dados, em que valores nulos foram tratados, variáveis referentes às categorias foram convertidas em base numérica e valores foram padronizados a partir da normalização de z-score.
3. **Modelagem preditiva com Machine Learning**  
Desenvolvimento e treinamento de algoritmos preditivos - Regressão Logísticar e Random Forest.
4. **Avaliação de desempenho**  
Análise  do desempenho utilizando métricas como matriz de confusão, curva ROC e feature importance, validando a eficácia dos modelos e identificando quais variáveis mais impactam nas previsões de cancelamento.

## Resultados
Os modelos de machine learning desenvolvidos demonstraram desempenho notável na previsão de cancelamento de clientes, com a Regressão Logística alcançando uma acurácia de 80.1% e uma área sob a curva ROC (AUC) de 0.820, superando ligeiramente o modelo de Random Forest que obteve 79.8% de acurácia e AUC de 0.815. A análise de importância de variáveis revelou que o valor total gasto pelo cliente (TotalCharges), o tempo de relacionamento (tenure) e o valor da mensalidade (MonthlyCharges) foram os fatores mais determinantes para a previsão de churn.

Além disso, observou-se que clientes com contratos de longa duração (especialmente bienais) apresentaram um risco significativamente menor de cancelamento quando comparados com clientes com planos. Um insight particularmente relevante foi que clientes que utilizam serviços de streaming demonstraram maior probabilidade de cancelamento, possivelmente por serem mais sensíveis à qualidade do serviço e terem maior acesso a alternativas concorrentes. Esses resultados fornecem uma base quantitativa sólida para o desenvolvimento de estratégias segmentadas de retenção de clientes.

## Estrutura e execução do projeto
   ```
   projec_churn_prediction/
   │
   ├── data/                # dataset original
   ├── notebooks/           # notebooks de análise e modelagem
   ├── images/              # gráficos e resultados
   ├── README.md            # documentação
   └── requirements.txt     # dependências
   ```
Para execução do projeto:
   ```bash
   !git clone https://github.com/nataliacamp/project_churn_prediction.git
   %cd project_churn_prediction

   !pip install -r requirements.txt

   exec(open('notebooks/01_data_loading_and_initial_exploration.py').read())
   exec(open('notebooks/02_exploratory_data_analysis.py').read())
   exec(open('notebooks/03_data_cleaning.py').read())
   exec(open('notebooks/04_machine_learning_models.py').read())
   exec(open('notebooks/05_results_analysis.py').read())
   ```

