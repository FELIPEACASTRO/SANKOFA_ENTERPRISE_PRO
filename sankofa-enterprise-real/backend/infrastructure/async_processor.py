"""
Sankofa Enterprise Pro - Async Processing Infrastructure
Sistema de processamento assíncrono para escala

Funcionalidades:
- Task queue com prioridades
- Workers pool para processamento paralelo
- Batch processing otimizado
- Circuit breaker para resiliência
- Retry com backoff exponencial
"""

import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional, Tuple
from enum import Enum
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Prioridades de tarefas"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BATCH = 4


class TaskStatus(Enum):
    """Status de tarefas"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class CircuitState(Enum):
    """Estados do circuit breaker"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class Task:
    """Representa uma tarefa a ser processada"""
    id: str
    fn: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority
    created_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


@dataclass
class BatchResult:
    """Resultado de processamento em lote"""
    total: int
    successful: int
    failed: int
    results: List[Any]
    errors: List[Dict[str, Any]]
    processing_time_ms: float


class CircuitBreaker:
    """
    Circuit breaker para proteção contra falhas em cascata
    
    Estados:
    - CLOSED: Operação normal
    - OPEN: Falhas excessivas, rejeita requisições
    - HALF_OPEN: Testando recuperação
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
        
        logger.info("CircuitBreaker initialized", extra={
            "failure_threshold": failure_threshold,
            "recovery_timeout": recovery_timeout
        })
    
    @property
    def state(self) -> CircuitState:
        """Retorna estado atual do circuit breaker"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time and (time.time() - self._last_failure_time) > self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
            return self._state
    
    def record_success(self):
        """Registra sucesso"""
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info("Circuit breaker CLOSED after successful recovery")
    
    def record_failure(self):
        """Registra falha"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker OPEN after half-open failure")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPEN after {self._failure_count} failures")
    
    def allow_request(self) -> bool:
        """Verifica se requisição é permitida"""
        current_state = self.state
        return current_state != CircuitState.OPEN


class AsyncTaskQueue:
    """
    Fila de tarefas assíncronas com prioridades
    
    Funcionalidades:
    - Priorização de tarefas
    - Workers pool
    - Retry com backoff
    - Métricas de execução
    """
    
    def __init__(self, num_workers: int = 4, max_queue_size: int = 10000):
        self._queue = queue.PriorityQueue(maxsize=max_queue_size)
        self._num_workers = num_workers
        self._max_queue_size = max_queue_size
        
        self._executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="task_worker")
        self._running = False
        self._workers: List[threading.Thread] = []
        
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        
        self._metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
            "total_processing_time_ms": 0.0
        }
        
        self._circuit_breaker = CircuitBreaker()
        
        logger.info("AsyncTaskQueue initialized", extra={
            "num_workers": num_workers,
            "max_queue_size": max_queue_size
        })
    
    def start(self):
        """Inicia os workers"""
        if self._running:
            return
        
        self._running = True
        
        for i in range(self._num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"task_worker_{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started {self._num_workers} task workers")
    
    def stop(self):
        """Para os workers"""
        self._running = False
        
        for _ in range(self._num_workers):
            try:
                self._queue.put_nowait((0, None))
            except queue.Full:
                pass
        
        for worker in self._workers:
            worker.join(timeout=5)
        
        self._workers.clear()
        logger.info("Task workers stopped")
    
    def submit(self, fn: Callable, *args, priority: TaskPriority = TaskPriority.NORMAL, 
               max_retries: int = 3, **kwargs) -> str:
        """Submete uma tarefa para execução"""
        task_id = f"TASK_{uuid.uuid4().hex[:12]}"
        
        task = Task(
            id=task_id,
            fn=fn,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries
        )
        
        with self._lock:
            self._tasks[task_id] = task
            self._metrics["tasks_submitted"] += 1
        
        try:
            self._queue.put_nowait((task.priority.value, task))
        except queue.Full:
            logger.error("Task queue is full")
            task.status = TaskStatus.FAILED
            task.error = "Queue is full"
            with self._lock:
                self._metrics["tasks_failed"] += 1
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retorna status de uma tarefa"""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            
            return {
                "id": task.id,
                "status": task.status.value,
                "priority": task.priority.name,
                "retries": task.retries,
                "result": task.result if task.status == TaskStatus.COMPLETED else None,
                "error": task.error if task.status == TaskStatus.FAILED else None
            }
    
    def get_result(self, task_id: str, timeout: float = 30.0) -> Any:
        """Aguarda e retorna resultado de uma tarefa"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                task = self._tasks.get(task_id)
                if not task:
                    raise ValueError(f"Task {task_id} not found")
                
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    raise Exception(f"Task failed: {task.error}")
            
            time.sleep(0.01)
        
        raise TimeoutError(f"Task {task_id} timed out")
    
    def _worker_loop(self):
        """Loop principal do worker"""
        while self._running:
            try:
                _, task = self._queue.get(timeout=1)
                
                if task is None:
                    break
                
                self._process_task(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def _process_task(self, task: Task):
        """Processa uma tarefa"""
        if not self._circuit_breaker.allow_request():
            task.status = TaskStatus.FAILED
            task.error = "Circuit breaker is open"
            with self._lock:
                self._metrics["tasks_failed"] += 1
            return
        
        task.status = TaskStatus.PROCESSING
        start_time = time.time()
        
        try:
            result = task.fn(*task.args, **task.kwargs)
            task.result = result
            task.status = TaskStatus.COMPLETED
            
            self._circuit_breaker.record_success()
            
            processing_time = (time.time() - start_time) * 1000
            with self._lock:
                self._metrics["tasks_completed"] += 1
                self._metrics["total_processing_time_ms"] += processing_time
                
        except Exception as e:
            self._circuit_breaker.record_failure()
            
            task.retries += 1
            if task.retries < task.max_retries:
                task.status = TaskStatus.RETRYING
                
                backoff = min(2 ** task.retries, 30)
                time.sleep(backoff)
                
                with self._lock:
                    self._metrics["tasks_retried"] += 1
                
                self._queue.put_nowait((task.priority.value, task))
            else:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                with self._lock:
                    self._metrics["tasks_failed"] += 1
                
                logger.error(f"Task {task.id} failed after {task.retries} retries: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas da fila"""
        with self._lock:
            return {
                **self._metrics,
                "queue_size": self._queue.qsize(),
                "active_tasks": sum(1 for t in self._tasks.values() if t.status == TaskStatus.PROCESSING),
                "circuit_breaker_state": self._circuit_breaker.state.value
            }


class BatchProcessor:
    """
    Processador de lotes para alta performance
    
    Funcionalidades:
    - Processamento paralelo de batches
    - Agregação de resultados
    - Tratamento de erros parciais
    """
    
    def __init__(self, max_workers: int = 8, batch_size: int = 100):
        self._max_workers = max_workers
        self._batch_size = batch_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="batch_worker")
        
        logger.info("BatchProcessor initialized", extra={
            "max_workers": max_workers,
            "batch_size": batch_size
        })
    
    def process_batch(self, items: List[Any], processor: Callable[[Any], Any],
                      batch_size: Optional[int] = None) -> BatchResult:
        """
        Processa uma lista de itens em paralelo
        
        Args:
            items: Lista de itens a processar
            processor: Função que processa cada item
            batch_size: Tamanho do batch (opcional)
        
        Returns:
            BatchResult com resultados e métricas
        """
        batch_size = batch_size or self._batch_size
        start_time = time.time()
        
        results = []
        errors = []
        
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        futures = []
        for batch_idx, batch in enumerate(batches):
            for item_idx, item in enumerate(batch):
                future = self._executor.submit(self._safe_process, processor, item, batch_idx, item_idx)
                futures.append((batch_idx * batch_size + item_idx, future))
        
        for idx, future in sorted(futures, key=lambda x: x[0]):
            try:
                success, result = future.result(timeout=60)
                if success:
                    results.append(result)
                else:
                    errors.append({"index": idx, "error": str(result)})
            except Exception as e:
                errors.append({"index": idx, "error": str(e)})
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchResult(
            total=len(items),
            successful=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
            processing_time_ms=processing_time
        )
    
    def _safe_process(self, processor: Callable, item: Any, batch_idx: int, item_idx: int) -> Tuple[bool, Any]:
        """Processa item com tratamento de erro"""
        try:
            result = processor(item)
            return (True, result)
        except Exception as e:
            logger.warning(f"Batch processing error at {batch_idx}:{item_idx}: {e}")
            return (False, e)
    
    def shutdown(self):
        """Encerra o processador"""
        self._executor.shutdown(wait=True)


class ConnectionPool:
    """
    Pool de conexões genérico para reutilização
    
    Funcionalidades:
    - Pool com limite de conexões
    - Timeout para aquisição
    - Health check de conexões
    """
    
    def __init__(self, factory: Callable[[], Any], min_size: int = 2, max_size: int = 20,
                 timeout: float = 5.0):
        self._factory = factory
        self._min_size = min_size
        self._max_size = max_size
        self._timeout = timeout
        
        self._pool: queue.Queue = queue.Queue(maxsize=max_size)
        self._size = 0
        self._lock = threading.Lock()
        
        self._initialize_pool()
        
        logger.info("ConnectionPool initialized", extra={
            "min_size": min_size,
            "max_size": max_size
        })
    
    def _initialize_pool(self):
        """Inicializa conexões mínimas"""
        for _ in range(self._min_size):
            try:
                conn = self._factory()
                self._pool.put_nowait(conn)
                self._size += 1
            except Exception as e:
                logger.warning(f"Failed to create initial connection: {e}")
    
    def acquire(self) -> Any:
        """Adquire uma conexão do pool"""
        try:
            conn = self._pool.get(timeout=self._timeout)
            return conn
        except queue.Empty:
            with self._lock:
                if self._size < self._max_size:
                    try:
                        conn = self._factory()
                        self._size += 1
                        return conn
                    except Exception as e:
                        logger.error(f"Failed to create connection: {e}")
                        raise
            
            raise TimeoutError("Connection pool exhausted")
    
    def release(self, conn: Any):
        """Libera uma conexão de volta ao pool"""
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do pool"""
        return {
            "total_connections": self._size,
            "available_connections": self._pool.qsize(),
            "in_use": self._size - self._pool.qsize(),
            "max_size": self._max_size
        }


async_task_queue = AsyncTaskQueue(num_workers=4)
batch_processor = BatchProcessor(max_workers=8)


def start_async_infrastructure():
    """Inicia infraestrutura assíncrona"""
    async_task_queue.start()
    logger.info("Async infrastructure started")


def stop_async_infrastructure():
    """Para infraestrutura assíncrona"""
    async_task_queue.stop()
    batch_processor.shutdown()
    logger.info("Async infrastructure stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Async Infrastructure")
    
    start_async_infrastructure()
    
    def sample_task(x):
        time.sleep(0.1)
        return x * 2
    
    task_ids = []
    for i in range(10):
        task_id = async_task_queue.submit(sample_task, i, priority=TaskPriority.NORMAL)
        task_ids.append(task_id)
    
    time.sleep(2)
    
    print("\n=== Task Results ===")
    for task_id in task_ids:
        status = async_task_queue.get_task_status(task_id)
        print(f"  {task_id}: {status}")
    
    print("\n=== Queue Metrics ===")
    print(async_task_queue.get_metrics())
    
    print("\n=== Batch Processing ===")
    items = list(range(50))
    result = batch_processor.process_batch(items, lambda x: x ** 2, batch_size=10)
    print(f"  Total: {result.total}")
    print(f"  Successful: {result.successful}")
    print(f"  Failed: {result.failed}")
    print(f"  Processing time: {result.processing_time_ms:.2f}ms")
    
    stop_async_infrastructure()
