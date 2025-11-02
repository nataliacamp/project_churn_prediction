# 🧠 Telco Customer Churn Prediction

Projeto desenvolvido como parte de um estudo de Data Analytics e Machine Learning, com o objetivo de prever o **churn (evasão de clientes)** em uma empresa de telecomunicações.

## 📊 Objetivos
- Analisar o comportamento dos clientes
- Identificar fatores que influenciam o churn
- Construir modelos preditivos (Logistic Regression, Random Forest)
- Avaliar desempenho com métricas e curva ROC

## 🧰 Tecnologias utilizadas
- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## 🧠 Etapas do projeto
1. **Análise exploratória (EDA)**  
   Exploração das variáveis e visualização de padrões.
2. **Pré-processamento**  
   Limpeza de dados, encoding e normalização.
3. **Modelagem preditiva**  
   Aplicação de algoritmos de classificação.
4. **Avaliação de desempenho**  
   Acurácia, precisão, recall, ROC AUC e importância das features.

## 📈 Resultados
- Melhor modelo: Random Forest
- AUC: 0.84
- Principais fatores: tipo de contrato, mensalidade e tempo de permanência

## 📂 Estrutura do projeto
```
telco-churn-prediction/
│
├── data/                # dataset original
├── notebooks/           # notebooks de análise e modelagem
├── images/              # gráficos e resultados
├── README.md            # documentação
└── requirements.txt     # dependências
```

## 🚀 Como executar
1. Clone o repositório:
   ```bash
   git clone https://github.com/SEU_USUARIO/telco-churn-prediction.git
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Abra o notebook:
   ```bash
   jupyter notebook notebooks/Projeto_Prev_Churn_Telco.ipynb
   ```

## 👩‍💻 Autor
**Seu nome**  
[LinkedIn](https://linkedin.com/in/seuusuario) | [GitHub](https://github.com/seuusuario)
