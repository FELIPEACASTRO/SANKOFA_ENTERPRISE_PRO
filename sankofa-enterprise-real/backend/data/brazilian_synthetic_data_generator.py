import logging

logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
import datetime
import random
import uuid


class BrazilianSyntheticDataGenerator:
    def __init__(self, num_transactions=100000, fraud_rate=0.01):
        self.num_transactions = num_transactions
        self.fraud_rate = fraud_rate
        self.start_date = datetime.datetime(2024, 1, 1)
        self.end_date = datetime.datetime(2025, 1, 1)
        self.transaction_types = ["DEBITO", "CREDITO", "PIX", "BOLETO"]
        self.channels = ["POS", "ATM", "WEB", "MOBILE"]
        self.brazilian_states = ["SP", "RJ", "MG", "BA", "RS", "PR", "PE", "CE", "AM", "DF"]
        self.cities_per_state = {
            state: [f"{state}_City_{i}" for i in range(5)] for state in self.brazilian_states
        }
        self.device_ids = [f"device_{i}" for i in range(50)]
        self.num_clients = 10000
        self.client_cpfs = [f"{i:011d}" for i in range(self.num_clients)]

    def generate_transaction(self, client_cpf):
        timestamp = self.start_date + (self.end_date - self.start_date) * random.random()
        value = round(random.uniform(10.0, 5000.0), 2)
        transaction_type = random.choice(self.transaction_types)
        channel = random.choice(self.channels)
        state = random.choice(self.brazilian_states)
        city = random.choice(self.cities_per_state[state])
        ip_address = f"192.168.{random.randint(0, 255)}.{random.randint(0, 255)}"
        device_id = random.choice(self.device_ids)
        receiver_account = f"merchant_{random.randint(1000, 9999)}"

        return {
            "id": str(uuid.uuid4()),
            "timestamp": timestamp.isoformat(),
            "value": value,
            "transaction_type": transaction_type,
            "channel": channel,
            "city": city,
            "state": state,
            "country": "BR",
            "ip_address": ip_address,
            "device_id": device_id,
            "receiver_account": receiver_account,
            "client_cpf": client_cpf,
            "isFraud": 0,
        }

    def inject_fraud(self, transaction):
        # Simple fraud injection logic for demonstration
        fraud_type = random.choice(["high_value_night", "multiple_small_pix", "fake_boleto"])

        # SEMPRE marcar como fraude quando esta função é chamada
        transaction["isFraud"] = 1

        if fraud_type == "high_value_night":
            # Transação de alto valor à noite
            ts = datetime.datetime.fromisoformat(transaction["timestamp"])
            hour = ts.hour
            if hour < 6 or hour >= 22:
                transaction["value"] = round(random.uniform(5000.0, 20000.0), 2)
            else:
                # Forçar horário noturno
                new_hour = random.choice([0, 1, 2, 3, 4, 5, 22, 23])
                ts = ts.replace(hour=new_hour)
                transaction["timestamp"] = ts.isoformat()
                transaction["value"] = round(random.uniform(5000.0, 20000.0), 2)

        elif fraud_type == "multiple_small_pix":
            # Múltiplas transações PIX pequenas de dispositivo novo
            transaction["transaction_type"] = "PIX"
            transaction["value"] = round(random.uniform(10.0, 99.0), 2)
            transaction["device_id"] = "new_unregistered_device"

        elif fraud_type == "fake_boleto":
            # Boleto suspeito
            transaction["transaction_type"] = "BOLETO"
            transaction["receiver_account"] = "suspicious_boleto_account"
            transaction["value"] = round(random.uniform(500.0, 5000.0), 2)

        return transaction

    def generate_data(self):
        data = []
        for _ in range(self.num_transactions):
            client_cpf = random.choice(self.client_cpfs)
            transaction = self.generate_transaction(client_cpf)

            if random.random() < self.fraud_rate:
                transaction = self.inject_fraud(transaction)
            data.append(transaction)

        return pd.DataFrame(data)


if __name__ == "__main__":
    import uuid

    generator = BrazilianSyntheticDataGenerator(num_transactions=100000)
    df_synthetic = generator.generate_data()
    logger.info("\nBrazilian Synthetic Data Info:")
    df_synthetic.info()
    logger.info(f'\nTotal fraudulent transactions: {df_synthetic["isFraud"].sum()}')
    df_synthetic.to_csv(
        "/home/ubuntu/sankofa-enterprise-real/data/external_datasets/brazilian_synthetic_fraud_data.csv",
        index=False,
    )
    logger.info("Brazilian synthetic fraud data saved to brazilian_synthetic_fraud_data.csv")
