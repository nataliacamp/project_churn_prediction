print("Regressão Logística")
print(classification_report(y_test, y_pred_log))
print("AUC:", roc_auc_score(y_test, log_model.predict_proba(X_test)[:,1]))

print("\nRandom Forest")
print(classification_report(y_test, y_pred_rf))
print("AUC:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Matriz de Confusão - Regressão Logística
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
axes[0,0].set_title('Matriz de Confusão - Regressão Logística')
axes[0,0].set_xlabel('Predito')
axes[0,0].set_ylabel('Real')

# Matriz de Confusão - Random Forest
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
axes[0,1].set_title('Matriz de Confusão - Random Forest')
axes[0,1].set_xlabel('Predito')
axes[0,1].set_ylabel('Real')

# Curva ROC - Regressão Logística
fpr_log, tpr_log, _ = roc_curve(y_test, log_model.predict_proba(X_test)[:,1])
axes[1,0].plot(fpr_log, tpr_log, label=f'ROC (AUC = {roc_auc_score(y_test, log_model.predict_proba(X_test)[:,1]):.3f})', color='red')
axes[1,0].plot([0,1],[0,1],'--', color='gray')
axes[1,0].legend()
axes[1,0].set_title('Curva ROC - Regressão Logística')
axes[1,0].set_xlabel('False Positive Rate')
axes[1,0].set_ylabel('True Positive Rate')

# Curva ROC - Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])
axes[1,1].plot(fpr_rf, tpr_rf, label=f'ROC (AUC = {roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]):.3f})', color='blue')
axes[1,1].plot([0,1],[0,1],'--', color='gray')
axes[1,1].legend()
axes[1,1].set_title('Curva ROC - Random Forest')
axes[1,1].set_xlabel('False Positive Rate')
axes[1,1].set_ylabel('True Positive Rate')

plt.tight_layout()
plt.show()

# Importância de cada categoria - Random Forest
plt.figure(figsize=(10, 6))
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
importances[:10].plot(kind='barh')
plt.title('Top 10 Variáveis mais Importantes - Random Forest')
plt.gca().invert_yaxis()
plt.show()

# Coeficientes - Regressão Logística
plt.figure(figsize=(10, 6))
coeficientes = pd.DataFrame({
    'feature': X.columns,
    'coef': log_model.coef_[0]
}).sort_values('coef', key=abs, ascending=False)

coeficientes[:10].plot(kind='barh', x='feature', y='coef')
plt.title('Top 10 Coeficientes - Regressão Logística')
plt.xlabel('Coeficiente')
plt.gca().invert_yaxis()
plt.show()
