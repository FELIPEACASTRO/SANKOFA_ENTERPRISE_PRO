"""
Script de Teste e Validação do Motor Otimizado
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import time

from ml_engine.optimized_production_fraud_engine import get_optimized_fraud_engine
from data.brazilian_synthetic_data_generator import BrazilianSyntheticDataGenerator


def main():
    print("=" * 80)
    print("TESTE DO MOTOR OTIMIZADO - SANKOFA ENTERPRISE PRO v2.0")
    print("=" * 80)
    
    # 1. Gerar dados sintéticos
    print("\n[1/4] Gerando dados sintéticos...")
    generator = BrazilianSyntheticDataGenerator(
        num_transactions=50000,
        fraud_rate=0.05  # 5% de fraudes (mais realista)
    )
    df = generator.generate_data()
    
    print(f"   Total de transações: {len(df)}")
    print(f"   Fraudes: {df['isFraud'].sum()} ({df['isFraud'].sum()/len(df)*100:.2f}%)")
    print(f"   Legítimas: {(df['isFraud']==0).sum()} ({(df['isFraud']==0).sum()/len(df)*100:.2f}%)")
    
    # 2. Dividir em treino e teste
    print("\n[2/4] Dividindo em treino (80%) e teste (20%)...")
    train_size = int(0.8 * len(df))
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()
    
    print(f"   Treino: {len(df_train)} transações")
    print(f"   Teste: {len(df_test)} transações")
    
    # 3. Treinar motor otimizado
    print("\n[3/4] Treinando motor otimizado...")
    engine = get_optimized_fraud_engine()
    
    start_train = time.time()
    engine.train(df_train, optimize_threshold=True)
    train_time = time.time() - start_train
    
    print(f"\n   Tempo de treinamento: {train_time:.2f}s")
    
    # 4. Testar no conjunto de teste
    print("\n[4/4] Testando no conjunto de teste...")
    
    start_pred = time.time()
    predictions = engine.predict(df_test)
    pred_time = time.time() - start_pred
    
    # Extrair predições
    y_true = df_test['isFraud'].values
    y_pred = np.array([p.is_fraud for p in predictions])
    y_proba = np.array([p.fraud_probability for p in predictions])
    
    # Calcular métricas
    print("\n" + "=" * 80)
    print("RESULTADOS FINAIS")
    print("=" * 80)
    
    print(f"\nTempo de predição: {pred_time:.2f}s")
    print(f"Throughput: {len(df_test)/pred_time:.2f} TPS")
    print(f"Latência média: {(pred_time/len(df_test))*1000:.2f} ms")
    
    print("\n### MÉTRICAS DE QUALIDADE")
    print("-" * 80)
    print(classification_report(y_true, y_pred, target_names=['Legítima', 'Fraude']))
    
    print("\n### MATRIZ DE CONFUSÃO")
    print("-" * 80)
    cm = confusion_matrix(y_true, y_pred)
    print(f"True Negatives:  {cm[0,0]:>6d}  (legítimas corretamente identificadas)")
    print(f"False Positives: {cm[0,1]:>6d}  (legítimas marcadas como fraude)")
    print(f"False Negatives: {cm[1,0]:>6d}  (fraudes que passaram)")
    print(f"True Positives:  {cm[1,1]:>6d}  (fraudes corretamente identificadas)")
    
    # Calcular métricas adicionais
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print("\n### MÉTRICAS CONSOLIDADAS")
    print("-" * 80)
    print(f"Accuracy:             {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:            {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:               {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:             {f1:.4f} ({f1*100:.2f}%)")
    print(f"Specificity:          {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"False Positive Rate:  {fpr:.4f} ({fpr*100:.2f}%)")
    
    # Verificar se atende aos requisitos
    print("\n### VERIFICAÇÃO DE REQUISITOS")
    print("-" * 80)
    
    requirements = {
        'F1-Score > 85%': f1 > 0.85,
        'Precision > 80%': precision > 0.80,
        'Recall > 75%': recall > 0.75,
        'False Positive Rate < 10%': fpr < 0.10
    }
    
    all_met = all(requirements.values())
    
    for req, met in requirements.items():
        status = "✅ ATENDIDO" if met else "❌ NÃO ATENDIDO"
        print(f"{req:30s} {status}")
    
    print("\n" + "=" * 80)
    if all_met:
        print("🎉 TODOS OS REQUISITOS FORAM ATENDIDOS!")
        print("Sistema pronto para produção.")
    else:
        print("⚠️  ALGUNS REQUISITOS NÃO FORAM ATENDIDOS")
        print("Ajustes adicionais podem ser necessários.")
    print("=" * 80)
    
    # Salvar métricas
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'false_positive_rate': fpr,
        'threshold': engine.confidence_threshold,
        'training_time_s': train_time,
        'prediction_time_s': pred_time,
        'throughput_tps': len(df_test)/pred_time,
        'latency_ms': (pred_time/len(df_test))*1000
    }
    
    # Salvar em arquivo
    import json
    with open('/home/ubuntu/SANKOFA_ENTERPRISE_PRO/optimized_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"\nMétricas salvas em: /home/ubuntu/SANKOFA_ENTERPRISE_PRO/optimized_metrics.json")


if __name__ == '__main__':
    main()
