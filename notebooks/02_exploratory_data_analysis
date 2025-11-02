plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("INFORMAÇÕES GERAIS DA BASE DADOS")
print(df.info())

print("\nESTATÍSTICAS DESCRITIVAS")
print(df.describe())

print("\nDISTRIBUIÇÃO DO CHURN")
churn_dist = df['Churn'].value_counts(normalize=True)
print(f"Clientes que permanecem: {churn_dist['No']:.2%}")
print(f"Clientes que cancelaram: {churn_dist['Yes']:.2%}")

# Gráfico 1 - Distribuição de Churn
plt.figure(figsize=(8, 6))
df_churn_plot = df.copy()
df_churn_plot['Churn'] = df_churn_plot['Churn'].map({'Yes': 'Cancelamentos', 'No': 'Clientes Ativos'})

ax = sns.countplot(x='Churn', data=df_churn_plot, 
                  hue='Churn',  # Adicionado hue
                  palette=['#34a853', '#ea4335'],
                  legend=False)

plt.title('Distribuição de Cancelamento (Churn)', fontsize=14, fontweight='bold')
plt.xlabel('Status do Cliente', fontsize=12)
plt.ylabel('Número de Clientes', fontsize=12)

total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    ax.annotate(percentage, 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Gráfico 2 - Histogramas das variáveis numéricas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

variaveis_numericas = ['tenure', 'MonthlyCharges', 'TotalCharges']
titulos = ['Tempo de Contrato (meses)', 'Cobrança Mensal (R$)', 'Cobrança Total (R$)']
cores = ['#4682B4', '#32CD32', '#FF6347']

for i, (col, titulo, cor) in enumerate(zip(variaveis_numericas, titulos, cores)):
    sns.histplot(df[col].dropna(), kde=True, color=cor, ax=axes[i], bins=20)
    axes[i].set_title(f'Distribuição de {titulo}', fontweight='bold')
    axes[i].set_xlabel(titulo)
    axes[i].set_ylabel('Frequência')
    
    media = df[col].mean()
    axes[i].axvline(media, color='red', linestyle='--', linewidth=2, 
                   label=f'Média: {media:.1f}')
    axes[i].legend()

plt.tight_layout()
plt.show()

# Gráfico 3 - Churn rate por tipo de contrato
plt.figure(figsize=(10, 6))
df_contract_plot = df.copy()
df_contract_plot['Contract'] = df_contract_plot['Contract'].map({
    'Month-to-month': 'Mensal',
    'One year': 'Anual', 
    'Two year': 'Bienal'
})

contract_churn = df_contract_plot.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()

ordem_contratos = ['Mensal', 'Anual', 'Bienal']

ax = sns.barplot(x='Contract', y='Churn', data=contract_churn, 
                hue='Contract', 
                order=ordem_contratos,
                palette=['#34a853', '#4682B4', '#ea4335'],
                legend=False)    

plt.title('Taxa de Cancelamento por Tipo de Contrato', fontsize=14, fontweight='bold')
plt.xlabel('Tipo de Contrato', fontsize=12)
plt.ylabel('Taxa de Cancelamento (%)', fontsize=12)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', 
               (p.get_x() + p.get_width() / 2., p.get_height()),
               ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
