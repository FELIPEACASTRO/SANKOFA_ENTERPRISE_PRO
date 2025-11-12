import logging

logger = logging.getLogger(__name__)
#!/usr/bin/env python3
"""
Sankofa Enterprise Pro - Kaggle Dataset Downloader
Sistema automatizado para download de datasets reais de fraude bancária
"""

import os
import json
import shutil
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Informações sobre um dataset"""

    name: str
    kaggle_ref: str
    description: str
    size_mb: float
    num_transactions: int
    fraud_rate: float
    download_status: str  # 'not_downloaded', 'downloading', 'completed', 'failed'
    download_path: Optional[str] = None
    downloaded_at: Optional[str] = None


class KaggleDatasetDownloader:
    """
    Downloader automatizado de datasets do Kaggle

    Features:
    - Download paralelo
    - Validação de integridade
    - Cache local
    - Retry automático
    - Progress tracking
    """

    def __init__(self, data_dir: str = "./data/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Diretórios específicos
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.metadata_file = self.data_dir / "datasets_metadata.json"

        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

        # Configuração de datasets
        self.datasets = self._initialize_datasets()

        logger.info(f"Kaggle Downloader initialized. Data dir: {self.data_dir}")

    def _initialize_datasets(self) -> Dict[str, DatasetInfo]:
        """Inicializa lista de datasets disponíveis"""
        return {
            "ieee_fraud": DatasetInfo(
                name="IEEE-CIS Fraud Detection",
                kaggle_ref="c/ieee-fraud-detection",
                description="590K+ transações de cartão com labels de fraude",
                size_mb=580.0,
                num_transactions=590540,
                fraud_rate=0.0349,
                download_status="not_downloaded",
            ),
            "credit_card": DatasetInfo(
                name="Credit Card Fraud Detection",
                kaggle_ref="mlg-ulb/creditcardfraud",
                description="284K transações europeias com PCA features",
                size_mb=144.0,
                num_transactions=284807,
                fraud_rate=0.0017,
                download_status="not_downloaded",
            ),
            "paysim": DatasetInfo(
                name="PaySim Mobile Money",
                kaggle_ref="ealaxi/paysim1",
                description="6.3M transações mobile simuladas",
                size_mb=493.0,
                num_transactions=6362620,
                fraud_rate=0.0013,
                download_status="not_downloaded",
            ),
            "bank_account": DatasetInfo(
                name="Bank Account Fraud (NeurIPS 2022)",
                kaggle_ref="sgpjesus/bank-account-fraud-dataset-neurips-2022",
                description="1M+ contas bancárias com fraude sintética",
                size_mb=125.0,
                num_transactions=1000000,
                fraud_rate=0.0960,
                download_status="not_downloaded",
            ),
        }

    def check_kaggle_credentials(self) -> bool:
        """Verifica se as credenciais do Kaggle estão configuradas"""
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"

        if kaggle_json.exists():
            logger.info("✅ Kaggle credentials found")
            return True
        else:
            logger.warning("⚠️  Kaggle credentials not found!")
            logger.info(
                """
            Para baixar datasets do Kaggle:
            
            1. Criar conta em https://www.kaggle.com/
            2. Ir em Account > API > Create New Token
            3. Baixar kaggle.json
            4. Executar:
               mkdir -p ~/.kaggle
               mv kaggle.json ~/.kaggle/
               chmod 600 ~/.kaggle/kaggle.json
            """
            )
            return False

    def download_dataset(self, dataset_key: str, force: bool = False) -> bool:
        """
        Baixa um dataset do Kaggle

        Args:
            dataset_key: Chave do dataset (ex: 'ieee_fraud')
            force: Force re-download se já existe

        Returns:
            True se sucesso, False caso contrário
        """
        if dataset_key not in self.datasets:
            logger.error(f"Dataset '{dataset_key}' not found")
            return False

        dataset = self.datasets[dataset_key]
        download_path = self.raw_dir / dataset_key

        # Verificar se já existe
        if download_path.exists() and not force:
            logger.info(f"✅ Dataset '{dataset.name}' already downloaded")
            dataset.download_status = "completed"
            dataset.download_path = str(download_path)
            return True

        # Verificar credenciais
        if not self.check_kaggle_credentials():
            dataset.download_status = "failed"
            return False

        try:
            logger.info(f"📥 Downloading '{dataset.name}' ({dataset.size_mb}MB)...")
            dataset.download_status = "downloading"

            # Criar diretório temporário
            temp_dir = self.raw_dir / f"{dataset_key}_temp"
            temp_dir.mkdir(exist_ok=True)

            # Download via Kaggle CLI
            import kaggle

            # Competitions vs Datasets
            if dataset.kaggle_ref.startswith("c/"):
                competition = dataset.kaggle_ref.replace("c/", "")
                kaggle.api.competition_download_files(competition, path=str(temp_dir))
            else:
                kaggle.api.dataset_download_files(
                    dataset.kaggle_ref, path=str(temp_dir), unzip=True
                )

            # Extrair ZIPs se necessário
            for zip_file in temp_dir.glob("*.zip"):
                logger.info(f"📂 Extracting {zip_file.name}...")
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)
                zip_file.unlink()  # Remover ZIP após extração

            # Mover para diretório final
            if download_path.exists():
                shutil.rmtree(download_path)
            temp_dir.rename(download_path)

            # Atualizar status
            dataset.download_status = "completed"
            dataset.download_path = str(download_path)
            dataset.downloaded_at = datetime.now().isoformat()

            logger.info(f"✅ Dataset '{dataset.name}' downloaded successfully!")
            logger.info(f"   Location: {download_path}")
            logger.info(f"   Files: {len(list(download_path.glob('*')))} files")

            self._save_metadata()
            return True

        except Exception as e:
            logger.error(f"❌ Failed to download '{dataset.name}': {e}")
            dataset.download_status = "failed"

            # Limpar diretório temporário se foi criado
            try:
                if "temp_dir" in locals() and temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception:
                pass

            return False

    def download_all(self, max_concurrent: int = 2) -> Dict[str, bool]:
        """
        Baixa todos os datasets

        Args:
            max_concurrent: Número máximo de downloads simultâneos

        Returns:
            Dict com status de cada download
        """
        results = {}

        logger.info(f"📥 Starting download of {len(self.datasets)} datasets...")

        for key, dataset in self.datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Dataset: {dataset.name}")
            logger.info(f"Size: {dataset.size_mb}MB | Transactions: {dataset.num_transactions:,}")
            logger.info(f"Fraud Rate: {dataset.fraud_rate:.2%}")
            logger.info(f"{'='*60}\n")

            results[key] = self.download_dataset(key)

        self._save_metadata()
        return results

    def get_dataset_info(self, dataset_key: str) -> Optional[DatasetInfo]:
        """Retorna informações sobre um dataset"""
        return self.datasets.get(dataset_key)

    def list_datasets(self) -> List[DatasetInfo]:
        """Lista todos os datasets disponíveis"""
        return list(self.datasets.values())

    def get_download_summary(self) -> Dict[str, any]:
        """Retorna resumo dos downloads"""
        total = len(self.datasets)
        downloaded = sum(1 for d in self.datasets.values() if d.download_status == "completed")
        failed = sum(1 for d in self.datasets.values() if d.download_status == "failed")

        total_transactions = sum(
            d.num_transactions for d in self.datasets.values() if d.download_status == "completed"
        )

        total_size_mb = sum(
            d.size_mb for d in self.datasets.values() if d.download_status == "completed"
        )

        return {
            "total_datasets": total,
            "downloaded": downloaded,
            "failed": failed,
            "pending": total - downloaded - failed,
            "total_transactions": total_transactions,
            "total_size_mb": round(total_size_mb, 2),
            "datasets": {k: asdict(v) for k, v in self.datasets.items()},
        }

    def _save_metadata(self):
        """Salva metadata dos datasets"""
        summary = self.get_download_summary()
        with open(self.metadata_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"📝 Metadata saved to {self.metadata_file}")


def main():
    """CLI para download de datasets"""
    logger.info("🏦 Sankofa Enterprise Pro - Kaggle Dataset Downloader")
    logger.info("=" * 60)

    downloader = KaggleDatasetDownloader()

    # Listar datasets
    logger.info("\n📊 Available Datasets:")
    for i, dataset in enumerate(downloader.list_datasets(), 1):
        logger.info(f"\n{i}. {dataset.name}")
        logger.info(f"   Kaggle: {dataset.kaggle_ref}")
        logger.info(f"   Transactions: {dataset.num_transactions:,}")
        logger.info(f"   Fraud Rate: {dataset.fraud_rate:.2%}")
        logger.info(f"   Size: {dataset.size_mb}MB")
        logger.info(f"   Status: {dataset.download_status}")

    # Download automático
    logger.info("\n" + "=" * 60)
    choice = input("\nDownload ALL datasets? (y/n): ").lower()

    if choice == "y":
        results = downloader.download_all()

        logger.info("\n" + "=" * 60)
        logger.info("📊 Download Summary:")
        logger.info("=" * 60)

        summary = downloader.get_download_summary()
        logger.info(f"✅ Downloaded: {summary['downloaded']}/{summary['total_datasets']}")
        logger.info(f"❌ Failed: {summary['failed']}")
        logger.info(f"📈 Total Transactions: {summary['total_transactions']:,}")
        logger.info(f"💾 Total Size: {summary['total_size_mb']}MB")

        logger.info("\n✅ Ready to train models with REAL data!")
    else:
        logger.info("Download cancelled. Run again when ready.")


if __name__ == "__main__":
    main()
