# 🎯 PLANO DE OTIMIZAÇÃO DE MÉTRICAS - SANKOFA ENTERPRISE PRO

**Data**: 08 de Novembro de 2025  
**Objetivo**: Alcançar métricas ideais de detecção de fraude  
**Meta**: F1-Score > 85%, Precision > 80%, Recall > 75%  

---

## 📊 SITUAÇÃO ATUAL vs META

| Métrica | Atual | Meta | Gap |
|---------|-------|------|-----|
| **Accuracy** | 48% | 85%+ | -37 pontos |
| **Precision** | 48% | 80%+ | -32 pontos |
| **Recall** | 100% | 75%+ | +25 pontos (precisa reduzir) |
| **F1-Score** | 64.88% | 85%+ | -20.12 pontos |
| **False Positive Rate** | 100% | <10% | -90 pontos |

---

## 🔍 DIAGNÓSTICO DO PROBLEMA

### Problema Principal: Threshold Muito Baixo

O sistema está marcando **TODAS as transações como fraude** porque o **threshold de decisão está muito baixo** (provavelmente 0.3 ou menos).

```python
# Código atual (production_fraud_engine.py)
threshold_high_risk = 0.35
threshold_medium_risk = 0.2
detection_threshold = 0.3  # ← MUITO BAIXO!
```

**Consequência**: Qualquer transação com probabilidade > 0.3 é marcada como fraude, resultando em 100% de falsos positivos.

---

## 🛠️ SOLUÇÕES PRÁTICAS

### Solução 1: Ajustar o Threshold de Decisão (CRÍTICO)

#### Passo 1: Encontrar o Threshold Ótimo

**Método**: Usar a curva ROC e Precision-Recall para encontrar o ponto ótimo.

```python
# backend/ml_engine/threshold_optimizer.py
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score

class ThresholdOptimizer:
    """
    Otimiza o threshold de decisão para maximizar F1-Score.
    """
    
    def __init__(self, target_precision=0.80, target_recall=0.75):
        """
        Args:
            target_precision: Precision mínima desejada (0.80 = 80%)
            target_recall: Recall mínimo desejado (0.75 = 75%)
        """
        self.target_precision = target_precision
        self.target_recall = target_recall
    
    def find_optimal_threshold(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray
    ) -> dict:
        """
        Encontra o threshold ótimo que maximiza F1-Score.
        
        Args:
            y_true: Labels verdadeiros (0 ou 1)
            y_proba: Probabilidades preditas (0.0 a 1.0)
        
        Returns:
            Dict com threshold ótimo e métricas
        """
        # Calcular precision e recall para diferentes thresholds
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Calcular F1-Score para cada threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        # Encontrar threshold que maximiza F1-Score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]
        
        # Encontrar threshold que atende aos requisitos mínimos
        valid_idx = np.where(
            (precisions >= self.target_precision) & 
            (recalls >= self.target_recall)
        )[0]
        
        if len(valid_idx) > 0:
            # Usar o threshold que maximiza F1 entre os válidos
            valid_f1 = f1_scores[valid_idx]
            best_valid_idx = valid_idx[np.argmax(valid_f1)]
            recommended_threshold = thresholds[best_valid_idx]
            recommended_f1 = f1_scores[best_valid_idx]
            recommended_precision = precisions[best_valid_idx]
            recommended_recall = recalls[best_valid_idx]
        else:
            # Nenhum threshold atende aos requisitos, usar o melhor F1
            recommended_threshold = best_threshold
            recommended_f1 = best_f1
            recommended_precision = best_precision
            recommended_recall = best_recall
        
        return {
            'optimal_threshold': float(recommended_threshold),
            'f1_score': float(recommended_f1),
            'precision': float(recommended_precision),
            'recall': float(recommended_recall),
            'meets_requirements': (
                recommended_precision >= self.target_precision and 
                recommended_recall >= self.target_recall
            )
        }
    
    def plot_threshold_analysis(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray,
        save_path: str = None
    ):
        """
        Plota análise de threshold (Precision-Recall e ROC).
        """
        import matplotlib.pyplot as plt
        
        # Precision-Recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Precision, Recall, F1 vs Threshold
        axes[0].plot(thresholds, precisions[:-1], label='Precision', linewidth=2)
        axes[0].plot(thresholds, recalls[:-1], label='Recall', linewidth=2)
        axes[0].plot(thresholds, f1_scores[:-1], label='F1-Score', linewidth=2, linestyle='--')
        axes[0].axhline(y=self.target_precision, color='r', linestyle=':', label=f'Target Precision ({self.target_precision})')
        axes[0].axhline(y=self.target_recall, color='g', linestyle=':', label=f'Target Recall ({self.target_recall})')
        axes[0].set_xlabel('Threshold', fontsize=12)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('Métricas vs Threshold', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Precision-Recall curve
        axes[1].plot(recalls, precisions, linewidth=2)
        axes[1].set_xlabel('Recall', fontsize=12)
        axes[1].set_ylabel('Precision', fontsize=12)
        axes[1].set_title('Curva Precision-Recall', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo em: {save_path}")
        else:
            plt.show()
```

#### Passo 2: Executar Otimização

```python
# backend/scripts/optimize_threshold.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from ml_engine.production_fraud_engine import get_fraud_engine
from ml_engine.threshold_optimizer import ThresholdOptimizer

# Carregar dados de validação
df_val = pd.read_csv('data/validation_data.csv')
X_val = df_val.drop('isFraud', axis=1)
y_val = df_val['isFraud']

# Carregar engine e fazer predições
engine = get_fraud_engine()
predictions = engine.predict(X_val)
y_proba = np.array([p.fraud_probability for p in predictions])

# Otimizar threshold
optimizer = ThresholdOptimizer(target_precision=0.80, target_recall=0.75)
result = optimizer.find_optimal_threshold(y_val.values, y_proba)

print("=" * 80)
print("OTIMIZAÇÃO DE THRESHOLD")
print("=" * 80)
print(f"Threshold Ótimo: {result['optimal_threshold']:.4f}")
print(f"F1-Score: {result['f1_score']:.4f}")
print(f"Precision: {result['precision']:.4f}")
print(f"Recall: {result['recall']:.4f}")
print(f"Atende Requisitos: {result['meets_requirements']}")
print("=" * 80)

# Salvar gráfico
optimizer.plot_threshold_analysis(
    y_val.values, 
    y_proba, 
    save_path='reports/threshold_analysis.png'
)

# Atualizar configuração
with open('backend/config/optimal_threshold.txt', 'w') as f:
    f.write(str(result['optimal_threshold']))
```

#### Passo 3: Atualizar o Motor com o Threshold Ótimo

```python
# backend/ml_engine/production_fraud_engine.py (ATUALIZAR)

class ProductionFraudEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # ... código existente ...
        
        # Carregar threshold otimizado
        threshold_path = Path(__file__).parent.parent / 'config' / 'optimal_threshold.txt'
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                self.confidence_threshold = float(f.read().strip())
            logger.info(f"Loaded optimized threshold: {self.confidence_threshold}")
        else:
            # Usar threshold padrão mais conservador
            self.confidence_threshold = 0.65  # ← AUMENTADO de 0.3 para 0.65
            logger.warning(f"Using default threshold: {self.confidence_threshold}")
```

---

### Solução 2: Melhorar a Engenharia de Features

#### Problema: Features Fracas

O modelo atual pode não ter features discriminativas suficientes.

#### Novas Features a Adicionar

```python
# backend/ml_engine/feature_engineering.py

class AdvancedFeatureEngineering:
    """
    Engenharia de features avançada para detecção de fraude.
    """
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features avançadas."""
        df = df.copy()
        
        # 1. Features Temporais
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].between(22, 6).astype(int)
        df['is_business_hours'] = df['hour'].between(9, 18).astype(int)
        
        # 2. Features de Valor
        df['log_value'] = np.log1p(df['value'])
        df['value_rounded'] = (df['value'] % 1 == 0).astype(int)  # Valores redondos são suspeitos
        
        # 3. Features de Comportamento do Cliente
        # Calcular estatísticas históricas por CPF
        client_stats = df.groupby('client_cpf').agg({
            'value': ['mean', 'std', 'count'],
            'transaction_type': lambda x: x.mode()[0] if len(x) > 0 else 'UNKNOWN'
        }).reset_index()
        
        client_stats.columns = ['client_cpf', 'avg_value', 'std_value', 'num_transactions', 'preferred_type']
        df = df.merge(client_stats, on='client_cpf', how='left')
        
        # Desvio do comportamento normal
        df['value_deviation'] = (df['value'] - df['avg_value']) / (df['std_value'] + 1e-10)
        df['is_new_client'] = (df['num_transactions'] < 5).astype(int)
        
        # 4. Features de Dispositivo
        device_stats = df.groupby('device_id').agg({
            'client_cpf': 'nunique',  # Quantos clientes usam este dispositivo
            'value': 'sum'
        }).reset_index()
        
        device_stats.columns = ['device_id', 'num_clients_per_device', 'total_value_device']
        df = df.merge(device_stats, on='device_id', how='left')
        
        # Dispositivo compartilhado é suspeito
        df['is_shared_device'] = (df['num_clients_per_device'] > 1).astype(int)
        
        # 5. Features de Localização
        df['is_high_risk_state'] = df['state'].isin(['SP', 'RJ']).astype(int)
        
        # 6. Features de Canal
        df['is_mobile'] = (df['channel'] == 'MOBILE').astype(int)
        df['is_pix'] = (df['transaction_type'] == 'PIX').astype(int)
        
        # 7. Features de Velocidade
        # Ordenar por cliente e timestamp
        df = df.sort_values(['client_cpf', 'timestamp'])
        df['time_since_last_transaction'] = (
            df.groupby('client_cpf')['timestamp']
            .diff()
            .dt.total_seconds()
            .fillna(999999)
        )
        
        # Transações muito rápidas são suspeitas
        df['is_rapid_transaction'] = (df['time_since_last_transaction'] < 60).astype(int)
        
        return df
```

---

### Solução 3: Balancear o Dataset de Treinamento

#### Problema: Dataset Desbalanceado

Se o dataset tem muito mais transações legítimas que fraudes, o modelo pode ter dificuldade em aprender.

#### Técnicas de Balanceamento

```python
# backend/ml_engine/data_balancing.py

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

class DataBalancer:
    """
    Balanceia dataset de fraude usando técnicas de resampling.
    """
    
    def __init__(self, method='smote'):
        """
        Args:
            method: 'smote', 'undersample', 'smotetomek'
        """
        self.method = method
    
    def balance(self, X, y):
        """
        Balanceia o dataset.
        
        Args:
            X: Features
            y: Labels (0=legítimo, 1=fraude)
        
        Returns:
            X_balanced, y_balanced
        """
        print(f"Dataset original: {len(y)} samples")
        print(f"  - Legítimas: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
        print(f"  - Fraudes: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
        
        if self.method == 'smote':
            # SMOTE: Synthetic Minority Over-sampling Technique
            sampler = SMOTE(random_state=42, k_neighbors=5)
        elif self.method == 'undersample':
            # Random Under-sampling
            sampler = RandomUnderSampler(random_state=42)
        elif self.method == 'smotetomek':
            # SMOTE + Tomek Links (remove amostras ambíguas)
            sampler = SMOTETomek(random_state=42)
        else:
            raise ValueError(f"Método inválido: {self.method}")
        
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        
        print(f"\nDataset balanceado: {len(y_balanced)} samples")
        print(f"  - Legítimas: {(y_balanced==0).sum()} ({(y_balanced==0).sum()/len(y_balanced)*100:.1f}%)")
        print(f"  - Fraudes: {(y_balanced==1).sum()} ({(y_balanced==1).sum()/len(y_balanced)*100:.1f}%)")
        
        return X_balanced, y_balanced
```

---

### Solução 4: Ajustar Pesos das Classes

#### Alternativa ao Balanceamento

Se não quiser modificar o dataset, ajuste os pesos das classes no modelo.

```python
# backend/ml_engine/production_fraud_engine.py (ATUALIZAR)

from sklearn.utils.class_weight import compute_class_weight

class ProductionFraudEngine:
    def train(self, X_train, y_train):
        """Treina o modelo com pesos de classe ajustados."""
        
        # Calcular pesos das classes
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )
        class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
        
        print(f"Class weights: {class_weight_dict}")
        
        # Treinar modelos com pesos
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=class_weight_dict,  # ← ADICIONAR
            random_state=42
        )
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # ... resto do código ...
```

---

### Solução 5: Usar Ensemble com Votação Ponderada

#### Problema: Ensemble Não Otimizado

O ensemble atual pode estar dando peso igual a todos os modelos.

#### Solução: Votação Ponderada

```python
# backend/ml_engine/production_fraud_engine.py (ATUALIZAR)

from sklearn.ensemble import VotingClassifier

class ProductionFraudEngine:
    def train(self, X_train, y_train):
        """Treina ensemble com votação ponderada."""
        
        # Treinar modelos individuais
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        
        # Avaliar performance individual em validação
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        rf.fit(X_train_split, y_train_split)
        gb.fit(X_train_split, y_train_split)
        lr.fit(X_train_split, y_train_split)
        
        # Calcular F1-Score de cada modelo
        rf_f1 = f1_score(y_val_split, rf.predict(X_val_split))
        gb_f1 = f1_score(y_val_split, gb.predict(X_val_split))
        lr_f1 = f1_score(y_val_split, lr.predict(X_val_split))
        
        print(f"RF F1: {rf_f1:.4f}")
        print(f"GB F1: {gb_f1:.4f}")
        print(f"LR F1: {lr_f1:.4f}")
        
        # Normalizar pesos
        total_f1 = rf_f1 + gb_f1 + lr_f1
        rf_weight = rf_f1 / total_f1
        gb_weight = gb_f1 / total_f1
        lr_weight = lr_f1 / total_f1
        
        print(f"Pesos: RF={rf_weight:.3f}, GB={gb_weight:.3f}, LR={lr_weight:.3f}")
        
        # Criar ensemble com votação ponderada
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('lr', lr)
            ],
            voting='soft',  # Usar probabilidades
            weights=[rf_weight, gb_weight, lr_weight]  # ← PESOS OTIMIZADOS
        )
        
        # Treinar ensemble no dataset completo
        self.ensemble.fit(X_train, y_train)
```

---

## 📋 ROADMAP DE IMPLEMENTAÇÃO

### Semana 1: Ajuste de Threshold (PRIORIDADE MÁXIMA)
- [ ] Implementar `ThresholdOptimizer`
- [ ] Executar otimização em dados de validação
- [ ] Atualizar `production_fraud_engine.py` com threshold ótimo
- [ ] Validar métricas após ajuste

**Meta**: F1-Score > 75%

### Semana 2: Engenharia de Features
- [ ] Implementar `AdvancedFeatureEngineering`
- [ ] Adicionar 15+ novas features
- [ ] Re-treinar modelo com novas features
- [ ] Validar impacto nas métricas

**Meta**: F1-Score > 80%

### Semana 3: Balanceamento e Pesos
- [ ] Implementar `DataBalancer`
- [ ] Testar SMOTE, undersample e SMOTETomek
- [ ] Ajustar pesos das classes
- [ ] Escolher melhor abordagem

**Meta**: F1-Score > 85%

### Semana 4: Otimização de Ensemble
- [ ] Implementar votação ponderada
- [ ] Testar diferentes combinações de modelos
- [ ] Calibrar probabilidades
- [ ] Validação final

**Meta**: F1-Score > 90%

---

## ✅ CHECKLIST DE VALIDAÇÃO

- [ ] Threshold otimizado e documentado
- [ ] Features avançadas implementadas
- [ ] Dataset balanceado ou pesos ajustados
- [ ] Ensemble otimizado com votação ponderada
- [ ] Métricas validadas em dados de teste separados
- [ ] F1-Score > 85%
- [ ] Precision > 80%
- [ ] Recall > 75%
- [ ] False Positive Rate < 10%
- [ ] Documentação atualizada

---

**Documento preparado por**: Análise Automatizada  
**Data**: 08 de Novembro de 2025  
**Versão**: 1.0  
