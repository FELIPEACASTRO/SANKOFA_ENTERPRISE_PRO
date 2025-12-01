# URLs Verificadas - Dataset Research
## Status: ATIVO | Última Verificação: 01/12/2025

---

## KAGGLE DATASETS

| Nome | URL | Status | Tamanho |
|------|-----|--------|---------|
| MLG-ULB Credit Card Fraud | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | ATIVO | 284K |
| IEEE-CIS Fraud Detection | https://www.kaggle.com/c/ieee-fraud-detection | ATIVO | 590K |
| PaySim Mobile Money | https://www.kaggle.com/datasets/ealaxi/paysim1 | ATIVO | 6.3M |
| Feedzai BAF NeurIPS 2022 | https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 | ATIVO | 6M |
| Sparkov CC Transactions | https://www.kaggle.com/datasets/kartik2112/fraud-detection | ATIVO | 1.3M |
| Fraud E-commerce | https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce | ATIVO | 150K |

---

## GITHUB REPOSITORIES

| Nome | URL | Status | Stars |
|------|-----|--------|-------|
| AI4Risk AntiFraud | https://github.com/AI4Risk/antifraud | ATIVO | 298+ |
| NVIDIA Financial Fraud | https://github.com/NVIDIA-AI-Blueprints/financial-fraud-detection | ATIVO | 25+ |
| Amazon FDB Benchmark | https://github.com/amazon-science/fraud-dataset-benchmark | ATIVO | 226+ |
| Feedzai BAF Suite | https://github.com/feedzai/bank-account-fraud | ATIVO | N/A |
| PaySim Simulator | https://github.com/EdgarLopezPhD/PaySim | ATIVO | N/A |

---

## HUGGING FACE DATASETS

| Nome | URL | Status | Tamanho |
|------|-----|--------|---------|
| Nigerian Financial Transactions | https://huggingface.co/datasets/electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset | ATIVO | 5M |

---

## RECURSOS GOVERNAMENTAIS

| Nome | URL | Status | Notas |
|------|-----|--------|-------|
| FCA-UK APP Fraud Data | https://www.fca.org.uk/firms/digital-sandbox/authorised-push-payment-synthetic-data | ATIVO | Requer registro |
| FCA Digital Sandbox | https://digitalsandbox.fcainnovation.co.uk/datasets/627/description | ATIVO | Acesso controlado |
| FCA Evaluation Report | https://www.fca.org.uk/publication/external-research/app-fraud-dataset-evaluation-report.pdf | ATIVO | PDF |

---

## PAPERS ACADÊMICOS

| Título | URL | Status | Ano |
|--------|-----|--------|-----|
| PIX Fraud Taxonomy Brazil | https://arxiv.org/abs/2511.20902 | ATIVO | 2025 |
| Feedzai BAF NeurIPS | https://arxiv.org/abs/2211.13358 | ATIVO | 2022 |
| Amazon FDB Paper | https://arxiv.org/abs/2208.14417 | ATIVO | 2022 |

---

## DOWNLOADS DIRETOS

### Kaggle CLI Commands
```bash
# Configurar Kaggle
pip install kaggle
mkdir -p ~/.kaggle
# Colocar kaggle.json em ~/.kaggle/

# Download datasets
kaggle datasets download -d mlg-ulb/creditcardfraud
kaggle datasets download -d ealaxi/paysim1
kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022
kaggle competitions download -c ieee-fraud-detection
```

### Hugging Face CLI
```bash
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset')"
```

### Git Clones
```bash
git clone https://github.com/AI4Risk/antifraud.git
git clone https://github.com/amazon-science/fraud-dataset-benchmark.git
git clone https://github.com/feedzai/bank-account-fraud.git
```

---

*Verificação automática: 01/12/2025 - Todos os links testados e funcionais*
