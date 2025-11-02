# Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Carregamento do banco de dados
url = 'https://raw.githubusercontent.com/nataliacamp/project_churn_prediction/main/data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(url)

print("ANÁLISE INICIAL DOS DADOS")
print(f"Formato do dataset: {df.shape}")
print(f"Número de linhas: {df.shape[0]}")
print(f"Número de colunas: {df.shape[1]}")
print("\nTipos de dados:")
print(df.dtypes.value_counts())
print("\nPrimeiras 5 linhas:")
display(df.head())
print("\nInformações sobre valores nulos:")
print(df.isnull().sum().sum()) 
