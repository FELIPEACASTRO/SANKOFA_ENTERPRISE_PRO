"""
Sankofa Enterprise Pro - Experiment Tracker
Sistema de tracking de experimentos ML (MLflow-like)
Conformidade com MLOps best practices
"""

import json
import uuid
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Métrica individual de um experimento"""
    name: str
    value: float
    step: int = 0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


@dataclass
class Artifact:
    """Artefato de um experimento (modelo, dados, etc)"""
    name: str
    artifact_type: str
    path: str
    size_bytes: int = 0
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"


@dataclass
class ExperimentRun:
    """Uma execução de experimento"""
    run_id: str
    experiment_name: str
    run_name: str
    status: str
    start_time: str
    end_time: Optional[str]
    parameters: Dict[str, Any]
    metrics: List[Metric]
    artifacts: List[Artifact]
    tags: Dict[str, str]
    git_commit: Optional[str]
    user: str
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['metrics'] = [asdict(m) for m in self.metrics]
        data['artifacts'] = [asdict(a) for a in self.artifacts]
        return data


class ExperimentTracker:
    """
    Sistema de Tracking de Experimentos ML
    
    Features:
    - Registro de parâmetros, métricas e artefatos
    - Versionamento de modelos
    - Comparação de experimentos
    - Histórico completo de treinamentos
    - Integração com Git para reproducibilidade
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, tracking_dir: str = "mlops/experiments"):
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_run: Optional[ExperimentRun] = None
        self._experiments_file = self.tracking_dir / "experiments.json"
        self._runs_dir = self.tracking_dir / "runs"
        self._runs_dir.mkdir(exist_ok=True)
        
        self._load_experiments()
        
        logger.info(f"ExperimentTracker initialized v{self.VERSION}")
    
    def _load_experiments(self):
        """Carrega experimentos existentes"""
        self.experiments: Dict[str, Dict[str, Any]] = {}
        
        if self._experiments_file.exists():
            try:
                with open(self._experiments_file, 'r') as f:
                    self.experiments = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load experiments: {e}")
                self.experiments = {}
    
    def _save_experiments(self):
        """Salva índice de experimentos"""
        with open(self._experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
    
    def _get_git_commit(self) -> Optional[str]:
        """Obtém commit atual do Git"""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return None
    
    def create_experiment(self, name: str, description: str = "") -> str:
        """Cria novo experimento"""
        experiment_id = str(uuid.uuid4())[:8]
        
        self.experiments[name] = {
            'experiment_id': experiment_id,
            'name': name,
            'description': description,
            'created_at': datetime.utcnow().isoformat() + "Z",
            'runs': []
        }
        
        self._save_experiments()
        logger.info(f"Created experiment: {name} ({experiment_id})")
        
        return experiment_id
    
    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = ""
    ) -> str:
        """Inicia nova execução de experimento"""
        
        if experiment_name not in self.experiments:
            self.create_experiment(experiment_name)
        
        run_id = str(uuid.uuid4())[:12]
        
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._current_run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            run_name=run_name,
            status="RUNNING",
            start_time=datetime.utcnow().isoformat() + "Z",
            end_time=None,
            parameters=parameters or {},
            metrics=[],
            artifacts=[],
            tags=tags or {},
            git_commit=self._get_git_commit(),
            user=os.environ.get('USER', 'unknown'),
            description=description
        )
        
        self.experiments[experiment_name]['runs'].append(run_id)
        self._save_experiments()
        
        logger.info(f"Started run: {run_name} ({run_id})")
        return run_id
    
    def log_param(self, key: str, value: Any):
        """Registra parâmetro do experimento"""
        if self._current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        self._current_run.parameters[key] = value
        logger.debug(f"Logged param: {key}={value}")
    
    def log_params(self, params: Dict[str, Any]):
        """Registra múltiplos parâmetros"""
        for key, value in params.items():
            self.log_param(key, value)
    
    def log_metric(self, key: str, value: float, step: int = 0):
        """Registra métrica do experimento"""
        if self._current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        metric = Metric(name=key, value=value, step=step)
        self._current_run.metrics.append(metric)
        logger.debug(f"Logged metric: {key}={value} (step={step})")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        """Registra múltiplas métricas"""
        for key, value in metrics.items():
            self.log_metric(key, value, step)
    
    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registra artefato (modelo, dados, etc)"""
        if self._current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        file_path = Path(path)
        size_bytes = file_path.stat().st_size if file_path.exists() else 0
        
        checksum = ""
        if file_path.exists() and size_bytes < 100_000_000:
            try:
                with open(file_path, 'rb') as f:
                    checksum = hashlib.md5(f.read()).hexdigest()[:16]
            except Exception:
                pass
        
        artifact = Artifact(
            name=name,
            artifact_type=artifact_type,
            path=str(path),
            size_bytes=size_bytes,
            checksum=checksum,
            metadata=metadata or {}
        )
        
        self._current_run.artifacts.append(artifact)
        logger.debug(f"Logged artifact: {name} ({artifact_type})")
    
    def set_tag(self, key: str, value: str):
        """Define tag no experimento"""
        if self._current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        self._current_run.tags[key] = value
    
    def end_run(self, status: str = "COMPLETED") -> Dict[str, Any]:
        """Finaliza execução do experimento"""
        if self._current_run is None:
            raise ValueError("No active run.")
        
        self._current_run.status = status
        self._current_run.end_time = datetime.utcnow().isoformat() + "Z"
        
        run_file = self._runs_dir / f"{self._current_run.run_id}.json"
        with open(run_file, 'w') as f:
            json.dump(self._current_run.to_dict(), f, indent=2, default=str)
        
        result = self._current_run.to_dict()
        
        logger.info(
            f"Ended run: {self._current_run.run_name} ({self._current_run.run_id}) - {status}"
        )
        
        self._current_run = None
        return result
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Obtém detalhes de uma execução"""
        run_file = self._runs_dir / f"{run_id}.json"
        
        if run_file.exists():
            with open(run_file, 'r') as f:
                return json.load(f)
        return None
    
    def list_runs(self, experiment_name: str) -> List[Dict[str, Any]]:
        """Lista execuções de um experimento"""
        if experiment_name not in self.experiments:
            return []
        
        runs = []
        for run_id in self.experiments[experiment_name].get('runs', []):
            run_data = self.get_run(run_id)
            if run_data:
                runs.append(run_data)
        
        runs.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return runs
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compara múltiplas execuções"""
        comparison = {
            'runs': [],
            'metrics_comparison': {},
            'parameters_comparison': {}
        }
        
        all_metrics: Dict[str, List[float]] = {}
        all_params: Dict[str, List[Any]] = {}
        
        for run_id in run_ids:
            run_data = self.get_run(run_id)
            if run_data:
                comparison['runs'].append({
                    'run_id': run_id,
                    'run_name': run_data.get('run_name'),
                    'status': run_data.get('status'),
                    'start_time': run_data.get('start_time')
                })
                
                for metric in run_data.get('metrics', []):
                    name = metric.get('name')
                    value = metric.get('value')
                    if name not in all_metrics:
                        all_metrics[name] = []
                    all_metrics[name].append(value)
                
                for key, value in run_data.get('parameters', {}).items():
                    if key not in all_params:
                        all_params[key] = []
                    all_params[key].append(value)
        
        for name, values in all_metrics.items():
            comparison['metrics_comparison'][name] = {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'values': values
            }
        
        comparison['parameters_comparison'] = all_params
        
        return comparison
    
    def get_best_run(
        self,
        experiment_name: str,
        metric_name: str,
        maximize: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Obtém melhor execução baseada em métrica"""
        runs = self.list_runs(experiment_name)
        
        if not runs:
            return None
        
        best_run = None
        best_value = float('-inf') if maximize else float('inf')
        
        for run in runs:
            for metric in run.get('metrics', []):
                if metric.get('name') == metric_name:
                    value = metric.get('value', 0)
                    if maximize and value > best_value:
                        best_value = value
                        best_run = run
                    elif not maximize and value < best_value:
                        best_value = value
                        best_run = run
        
        return best_run
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo de todos os experimentos"""
        summary = {
            'version': self.VERSION,
            'total_experiments': len(self.experiments),
            'total_runs': 0,
            'experiments': []
        }
        
        for name, exp in self.experiments.items():
            run_count = len(exp.get('runs', []))
            summary['total_runs'] += run_count
            summary['experiments'].append({
                'name': name,
                'run_count': run_count,
                'created_at': exp.get('created_at')
            })
        
        return summary


_tracker_instance: Optional[ExperimentTracker] = None


def get_experiment_tracker() -> ExperimentTracker:
    """Singleton para ExperimentTracker"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ExperimentTracker()
    return _tracker_instance


def log_training_run(
    experiment_name: str,
    model_name: str,
    parameters: Dict[str, Any],
    metrics: Dict[str, float],
    model_path: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> str:
    """Função utilitária para logar treinamento completo"""
    tracker = get_experiment_tracker()
    
    run_id = tracker.start_run(
        experiment_name=experiment_name,
        run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        parameters=parameters,
        tags=tags,
        description=f"Training run for {model_name}"
    )
    
    tracker.log_metrics(metrics)
    
    if model_path:
        tracker.log_artifact(
            name=model_name,
            artifact_type="model",
            path=model_path
        )
    
    tracker.end_run()
    
    return run_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    tracker = ExperimentTracker(tracking_dir="test_experiments")
    
    tracker.start_run(
        experiment_name="fraud_detection_v1",
        run_name="baseline_rf",
        parameters={
            'n_estimators': 100,
            'max_depth': 10,
            'learning_rate': 0.1
        },
        tags={'model': 'random_forest'}
    )
    
    tracker.log_metrics({
        'accuracy': 0.95,
        'precision': 0.92,
        'recall': 0.88,
        'f1_score': 0.90,
        'auc_roc': 0.96
    })
    
    result = tracker.end_run()
    
    print(f"Run completed: {result['run_id']}")
    print(f"Summary: {json.dumps(tracker.get_summary(), indent=2)}")
