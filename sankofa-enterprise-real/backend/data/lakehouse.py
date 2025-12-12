"""
Lakehouse Architecture - Delta Lake for ACID transactions and time travel
Unified data platform for batch and streaming analytics
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import json

try:
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType, BooleanType
    DELTA_AVAILABLE = True
except ImportError:
    DELTA_AVAILABLE = False
    logging.warning("Delta Lake not available. Install with: pip install delta-spark pyspark")

logger = logging.getLogger(__name__)


class DeltaLakeManager:
    """
    Delta Lake manager for fraud detection data

    Features:
    - ACID transactions
    - Time travel (data versioning)
    - Schema evolution
    - Automatic compaction
    - Upserts (merge operations)
    - CDC (Change Data Capture)
    """

    def __init__(
        self,
        lakehouse_path: str = "./lakehouse",
        warehouse_path: str = "./spark-warehouse"
    ):
        """
        Args:
            lakehouse_path: Root path for Delta tables
            warehouse_path: Spark warehouse path
        """
        if not DELTA_AVAILABLE:
            raise ImportError("Delta Lake not installed")

        self.lakehouse_path = Path(lakehouse_path)
        self.warehouse_path = Path(warehouse_path)

        # Create directories
        self.lakehouse_path.mkdir(parents=True, exist_ok=True)
        self.warehouse_path.mkdir(parents=True, exist_ok=True)

        # Initialize Spark with Delta
        self.spark = self._init_spark()

        # Table registry
        self.tables: Dict[str, str] = {}

        logger.info(f"Delta Lake Manager initialized: path={lakehouse_path}")

    def _init_spark(self) -> SparkSession:
        """Initialize Spark session with Delta Lake"""
        builder = (
            SparkSession.builder
            .appName("SankoFraudLakehouse")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.sql.warehouse.dir", str(self.warehouse_path))
            .config("spark.driver.memory", "4g")
        )

        spark = configure_spark_with_delta_pip(builder).getOrCreate()

        logger.info("Spark session initialized with Delta Lake")
        return spark

    def create_table(
        self,
        table_name: str,
        schema: Optional[StructType] = None,
        partition_cols: Optional[List[str]] = None,
        mode: str = 'overwrite'
    ) -> str:
        """
        Create Delta table

        Args:
            table_name: Table name
            schema: Table schema
            partition_cols: Partition columns
            mode: Write mode ('overwrite', 'append', 'error')

        Returns:
            Table path
        """
        table_path = str(self.lakehouse_path / table_name)

        # Default fraud transaction schema
        if schema is None:
            schema = self._get_transaction_schema()

        # Create empty dataframe with schema
        df = self.spark.createDataFrame([], schema)

        # Write as Delta table
        writer = df.write.format("delta").mode(mode)

        if partition_cols:
            writer = writer.partitionBy(*partition_cols)

        writer.save(table_path)

        self.tables[table_name] = table_path

        logger.info(f"Delta table created: {table_name} at {table_path}")

        return table_path

    def _get_transaction_schema(self) -> StructType:
        """Get default transaction schema"""
        return StructType([
            StructField("transaction_id", StringType(), False),
            StructField("customer_id", StringType(), True),
            StructField("merchant_id", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("currency", StringType(), True),
            StructField("channel", StringType(), True),
            StructField("device_id", StringType(), True),
            StructField("ip_address", StringType(), True),
            StructField("location", StringType(), True),
            StructField("is_fraud", BooleanType(), True),
            StructField("fraud_score", DoubleType(), True),
            StructField("status", StringType(), True),
            StructField("created_at", TimestampType(), True),
            StructField("updated_at", TimestampType(), True)
        ])

    async def write_transactions(
        self,
        table_name: str,
        transactions: Union[pd.DataFrame, List[Dict[str, Any]]],
        mode: str = 'append'
    ) -> int:
        """
        Write transactions to Delta table

        Args:
            table_name: Table name
            transactions: Transaction data
            mode: Write mode

        Returns:
            Number of rows written
        """
        # Convert to Spark DataFrame
        if isinstance(transactions, list):
            transactions = pd.DataFrame(transactions)

        spark_df = self.spark.createDataFrame(transactions)

        # Get table path
        table_path = self.tables.get(table_name)
        if not table_path:
            table_path = self.create_table(table_name)

        # Write
        spark_df.write.format("delta").mode(mode).save(table_path)

        row_count = len(transactions)
        logger.info(f"Wrote {row_count} transactions to {table_name}")

        return row_count

    async def read_transactions(
        self,
        table_name: str,
        filters: Optional[str] = None,
        columns: Optional[List[str]] = None,
        version: Optional[int] = None,
        timestamp: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Read transactions from Delta table

        Args:
            table_name: Table name
            filters: SQL WHERE clause
            columns: Columns to select
            version: Table version (for time travel)
            timestamp: Timestamp (for time travel)

        Returns:
            Pandas DataFrame
        """
        table_path = self.tables.get(table_name)
        if not table_path:
            raise ValueError(f"Table {table_name} not found")

        # Read Delta table
        reader = self.spark.read.format("delta")

        # Time travel
        if version is not None:
            reader = reader.option("versionAsOf", version)
        elif timestamp is not None:
            reader = reader.option("timestampAsOf", timestamp)

        df = reader.load(table_path)

        # Select columns
        if columns:
            df = df.select(*columns)

        # Apply filters
        if filters:
            df = df.filter(filters)

        # Convert to Pandas
        pandas_df = df.toPandas()

        logger.info(f"Read {len(pandas_df)} transactions from {table_name}")

        return pandas_df

    async def upsert_transactions(
        self,
        table_name: str,
        updates: Union[pd.DataFrame, List[Dict[str, Any]]],
        merge_key: str = 'transaction_id'
    ) -> Dict[str, int]:
        """
        Upsert (merge) transactions using Delta Lake MERGE

        Args:
            table_name: Table name
            updates: Update data
            merge_key: Column to merge on

        Returns:
            Merge statistics
        """
        from delta.tables import DeltaTable

        # Convert to Spark DataFrame
        if isinstance(updates, list):
            updates = pd.DataFrame(updates)

        updates_df = self.spark.createDataFrame(updates)

        # Get Delta table
        table_path = self.tables.get(table_name)
        if not table_path:
            raise ValueError(f"Table {table_name} not found")

        delta_table = DeltaTable.forPath(self.spark, table_path)

        # Merge
        merge_result = (
            delta_table.alias("target")
            .merge(
                updates_df.alias("source"),
                f"target.{merge_key} = source.{merge_key}"
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

        logger.info(f"Upserted {len(updates)} transactions to {table_name}")

        return {
            'rows_updated': len(updates),
            'merge_key': merge_key
        }

    async def delete_transactions(
        self,
        table_name: str,
        condition: str
    ) -> int:
        """
        Delete transactions (ACID delete)

        Args:
            table_name: Table name
            condition: Delete condition (SQL WHERE clause)

        Returns:
            Number of rows deleted
        """
        from delta.tables import DeltaTable

        table_path = self.tables.get(table_name)
        if not table_path:
            raise ValueError(f"Table {table_name} not found")

        delta_table = DeltaTable.forPath(self.spark, table_path)

        # Count before delete
        count_before = delta_table.toDF().count()

        # Delete
        delta_table.delete(condition)

        # Count after delete
        count_after = delta_table.toDF().count()

        deleted = count_before - count_after

        logger.info(f"Deleted {deleted} transactions from {table_name}")

        return deleted

    async def get_table_history(
        self,
        table_name: str,
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Get table version history

        Args:
            table_name: Table name
            limit: Max versions to return

        Returns:
            History dataframe
        """
        from delta.tables import DeltaTable

        table_path = self.tables.get(table_name)
        if not table_path:
            raise ValueError(f"Table {table_name} not found")

        delta_table = DeltaTable.forPath(self.spark, table_path)

        history = delta_table.history(limit).toPandas()

        return history

    async def vacuum_table(
        self,
        table_name: str,
        retention_hours: int = 168  # 7 days
    ) -> None:
        """
        Vacuum old versions (cleanup)

        Args:
            table_name: Table name
            retention_hours: Retention period in hours
        """
        from delta.tables import DeltaTable

        table_path = self.tables.get(table_name)
        if not table_path:
            raise ValueError(f"Table {table_name} not found")

        delta_table = DeltaTable.forPath(self.spark, table_path)

        # Vacuum
        delta_table.vacuum(retention_hours)

        logger.info(f"Vacuumed {table_name}: retention={retention_hours}h")

    async def optimize_table(
        self,
        table_name: str,
        z_order_cols: Optional[List[str]] = None
    ) -> None:
        """
        Optimize table (compaction + Z-ordering)

        Args:
            table_name: Table name
            z_order_cols: Columns for Z-ordering
        """
        from delta.tables import DeltaTable

        table_path = self.tables.get(table_name)
        if not table_path:
            raise ValueError(f"Table {table_name} not found")

        delta_table = DeltaTable.forPath(self.spark, table_path)

        # Optimize
        optimize_builder = delta_table.optimize()

        if z_order_cols:
            optimize_builder = optimize_builder.executeZOrderBy(*z_order_cols)
        else:
            optimize_builder.executeCompaction()

        logger.info(f"Optimized {table_name}")

    def create_fraud_analytics_views(self) -> None:
        """Create materialized views for analytics"""

        # Daily fraud stats view
        self.spark.sql("""
            CREATE OR REPLACE TEMP VIEW daily_fraud_stats AS
            SELECT
                DATE(created_at) as date,
                COUNT(*) as total_transactions,
                SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
                SUM(CASE WHEN is_fraud THEN amount ELSE 0 END) as fraud_amount,
                AVG(fraud_score) as avg_fraud_score
            FROM transactions
            GROUP BY DATE(created_at)
        """)

        # Merchant risk view
        self.spark.sql("""
            CREATE OR REPLACE TEMP VIEW merchant_risk AS
            SELECT
                merchant_id,
                COUNT(*) as transaction_count,
                SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
                SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as fraud_rate,
                AVG(amount) as avg_amount
            FROM transactions
            GROUP BY merchant_id
            HAVING COUNT(*) >= 10
        """)

        logger.info("Analytics views created")

    def close(self) -> None:
        """Close Spark session"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


class LakehouseQueryEngine:
    """
    Query engine for lakehouse analytics

    Supports:
    - SQL queries
    - Spark DataFrame operations
    - Time-series analysis
    - Aggregations
    """

    def __init__(self, lakehouse: DeltaLakeManager):
        """
        Args:
            lakehouse: Delta Lake manager
        """
        self.lakehouse = lakehouse
        self.spark = lakehouse.spark

        logger.info("Lakehouse Query Engine initialized")

    async def execute_sql(
        self,
        query: str
    ) -> pd.DataFrame:
        """
        Execute SQL query

        Args:
            query: SQL query

        Returns:
            Result dataframe
        """
        df = self.spark.sql(query)
        return df.toPandas()

    async def get_fraud_trends(
        self,
        table_name: str,
        start_date: str,
        end_date: str,
        granularity: str = 'day'
    ) -> pd.DataFrame:
        """
        Get fraud trends over time

        Args:
            table_name: Table name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date
            granularity: 'hour', 'day', 'week', 'month'

        Returns:
            Trends dataframe
        """
        table_path = self.lakehouse.tables.get(table_name)
        if not table_path:
            raise ValueError(f"Table {table_name} not found")

        df = self.spark.read.format("delta").load(table_path)

        # Time grouping
        if granularity == 'hour':
            time_col = F.date_trunc('hour', 'created_at')
        elif granularity == 'day':
            time_col = F.date_trunc('day', 'created_at')
        elif granularity == 'week':
            time_col = F.date_trunc('week', 'created_at')
        else:  # month
            time_col = F.date_trunc('month', 'created_at')

        # Aggregate
        trends = (
            df
            .filter(f"created_at BETWEEN '{start_date}' AND '{end_date}'")
            .groupBy(time_col.alias('period'))
            .agg(
                F.count('*').alias('total_transactions'),
                F.sum(F.when(F.col('is_fraud'), 1).otherwise(0)).alias('fraud_count'),
                F.avg('fraud_score').alias('avg_fraud_score'),
                F.sum('amount').alias('total_amount')
            )
            .orderBy('period')
        )

        return trends.toPandas()

    async def get_high_risk_entities(
        self,
        table_name: str,
        entity_type: str = 'customer',
        min_transactions: int = 10,
        top_n: int = 100
    ) -> pd.DataFrame:
        """
        Get high-risk entities (customers, merchants, devices)

        Args:
            table_name: Table name
            entity_type: 'customer', 'merchant', 'device'
            min_transactions: Minimum transactions
            top_n: Top N entities

        Returns:
            High-risk entities
        """
        table_path = self.lakehouse.tables.get(table_name)
        if not table_path:
            raise ValueError(f"Table {table_name} not found")

        df = self.spark.read.format("delta").load(table_path)

        entity_col = f"{entity_type}_id"

        high_risk = (
            df
            .groupBy(entity_col)
            .agg(
                F.count('*').alias('transaction_count'),
                F.sum(F.when(F.col('is_fraud'), 1).otherwise(0)).alias('fraud_count'),
                F.avg('fraud_score').alias('avg_fraud_score'),
                F.sum('amount').alias('total_amount')
            )
            .filter(f"transaction_count >= {min_transactions}")
            .withColumn('fraud_rate', F.col('fraud_count') / F.col('transaction_count'))
            .orderBy(F.desc('fraud_rate'))
            .limit(top_n)
        )

        return high_risk.toPandas()


# Example usage
async def example_lakehouse():
    """Example: Delta Lake lakehouse"""

    if not DELTA_AVAILABLE:
        print("Delta Lake not installed. Install with: pip install delta-spark pyspark")
        return

    # Initialize lakehouse
    lakehouse = DeltaLakeManager(lakehouse_path="./lakehouse_demo")

    # Create transactions table
    lakehouse.create_table(
        table_name="transactions",
        partition_cols=["created_at"]
    )

    # Generate synthetic transactions
    np.random.seed(42)
    n = 1000

    transactions = pd.DataFrame({
        'transaction_id': [f'TXN_{i:06d}' for i in range(n)],
        'customer_id': [f'CUST_{i % 100}' for i in range(n)],
        'merchant_id': [f'MERCH_{i % 50}' for i in range(n)],
        'amount': np.random.exponential(500, n),
        'currency': 'BRL',
        'channel': np.random.choice(['PIX', 'credit_card', 'debit_card'], n),
        'device_id': [f'DEV_{i % 200}' for i in range(n)],
        'ip_address': [f'192.168.1.{i % 255}' for i in range(n)],
        'location': np.random.choice(['São Paulo', 'Rio de Janeiro', 'Brasília'], n),
        'is_fraud': np.random.choice([False, True], n, p=[0.98, 0.02]),
        'fraud_score': np.random.beta(2, 5, n),
        'status': 'approved',
        'created_at': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'updated_at': pd.date_range('2024-01-01', periods=n, freq='5min')
    })

    # Write transactions
    await lakehouse.write_transactions('transactions', transactions)

    # Read transactions
    read_df = await lakehouse.read_transactions(
        'transactions',
        filters="is_fraud = True",
        columns=['transaction_id', 'amount', 'fraud_score']
    )

    print(f"\nFraud transactions: {len(read_df)}")
    print(read_df.head())

    # Get table history
    history = await lakehouse.get_table_history('transactions')
    print(f"\nTable versions: {len(history)}")
    print(history[['version', 'timestamp', 'operation']])

    # Query engine
    query_engine = LakehouseQueryEngine(lakehouse)

    # Fraud trends
    trends = await query_engine.get_fraud_trends(
        'transactions',
        start_date='2024-01-01',
        end_date='2024-01-02',
        granularity='hour'
    )

    print(f"\nFraud trends:")
    print(trends.head())

    # Cleanup
    lakehouse.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_lakehouse())
