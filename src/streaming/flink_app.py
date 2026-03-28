import json
import os

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.functions import RuntimeContext, MapFunction
from pyflink.common.typeinfo import Types

from pyflink.datastream.connectors.kafka import (
    KafkaSource,
    KafkaSink,
    KafkaRecordSerializationSchema
)
from pyflink.common.watermark_strategy import WatermarkStrategy

from src.streaming.consumer import FraudDetector


class FraudMapFunction(MapFunction):
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.detector = None

    def open(self, runtime_context: RuntimeContext):
        self.detector = FraudDetector(
            model_path=self.model_path,
            config_path=self.config_path
        ).load()

    def map(self, value):
        try:
            # 🔥 Fix 1: handle bytes properly
            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8")

            transaction = json.loads(value)
        except Exception:
            return None

        result = self.detector.predict(transaction)

        enriched = transaction.copy()
        enriched["fraud_score"] = result["fraud_score"]
        enriched["is_anomaly"] = result["is_anomaly"]

        # 🔥 Fix 2: always return STRING (not bytes, not weird object)
        return json.dumps(enriched)


def main():
    model_path = os.environ.get("MODEL_PATH", "models/fraud_model.joblib")
    config_path = os.environ.get("CONFIG_PATH", "models/feature_config.json")
    kafka_bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    input_topic = "transactions"
    anomaly_topic = "anomaly_transactions"
    normal_topic = "normal_transactions"

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    # 🔥 Fix 3: make sure Kafka connector is loaded BEFORE anything
    env.add_jars("file:///C:/flink-jars/flink-sql-connector-kafka-3.4.0-1.20.jar")

    # ✅ SOURCE
    source = KafkaSource.builder() \
        .set_bootstrap_servers(kafka_bootstrap) \
        .set_topics(input_topic) \
        .set_group_id("fraud-flink") \
        .set_value_only_deserializer(SimpleStringSchema()) \
        .build()

    stream = env.from_source(
        source,
        WatermarkStrategy.no_watermarks(),
        "Kafka Source"
    )

    # 🔥 Fix 4: enforce string type early
    stream = stream.map(
        lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x,
        output_type=Types.STRING()
    )

    # ✅ Processing
    enriched_stream = stream.map(
        FraudMapFunction(model_path, config_path),
        output_type=Types.STRING()   # 🔥 Fix 5: VERY IMPORTANT
    ).filter(lambda x: x is not None)

    anomalies = enriched_stream.filter(
        lambda x: json.loads(x)["is_anomaly"]
    )

    normals = enriched_stream.filter(
        lambda x: not json.loads(x)["is_anomaly"]
    )

    # ✅ SINKS
    anomaly_sink = KafkaSink.builder() \
        .set_bootstrap_servers(kafka_bootstrap) \
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic(anomaly_topic)
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        ).build()

    normal_sink = KafkaSink.builder() \
        .set_bootstrap_servers(kafka_bootstrap) \
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic(normal_topic)
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        ).build()

    anomalies.sink_to(anomaly_sink)
    normals.sink_to(normal_sink)

    env.execute("Fraud Detection Flink Job")


if __name__ == "__main__":
    main()