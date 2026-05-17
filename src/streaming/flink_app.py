import glob
import json
import os
import sys
from pathlib import Path

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
            # Fix 1: handle bytes properly
            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8")

            transaction = json.loads(value)
        except Exception:
            return None

        result = self.detector.predict(transaction)

        enriched = transaction.copy()
        enriched["fraud_score"] = result["fraud_score"]
        enriched["is_anomaly"] = result["is_anomaly"]

        # Fix 2: always return STRING (not bytes, not weird object)
        return json.dumps(enriched)

def _find_kafka_jar() -> str:
    """
    Locate the Flink Kafka connector JAR and return it as a file:// URI.

    Search order (first match wins):
      1. FLINK_KAFKA_JAR env var  — explicit override, highest priority
      2. Sibling jars/ directory next to this file  — project-local copy
      3. PyFlink's own bundled jars  — installed alongside apache-flink
      4. Common system install paths  — /opt/flink, /usr/local/flink, etc.
      5. Anywhere on PATH that flink binary knows about

    Raises RuntimeError with clear instructions if nothing is found.
    """
    JAR_GLOB = "flink-sql-connector-kafka*.jar"

    def to_uri(path: str) -> str:
        # Normalise to forward slashes and prefix with file:///
        p = Path(path).resolve()
        return p.as_uri()          # pathlib gives correct file:// on all OSes

    # 1. Explicit env var
    env_jar = os.environ.get("FLINK_KAFKA_JAR", "").strip()
    if env_jar:
        if not Path(env_jar).is_file():
            raise FileNotFoundError(
                f"FLINK_KAFKA_JAR is set to '{env_jar}' but the file does not exist."
            )
        return to_uri(env_jar)

    # 2. jars/ folder next to this script
    script_dir = Path(__file__).resolve().parent
    for candidate in sorted((script_dir / "jars").glob(JAR_GLOB)):
        return to_uri(str(candidate))

    # 3. PyFlink bundled jars  (site-packages/pyflink/lib/)
    try:
        import pyflink
        pyflink_lib = Path(pyflink.__file__).parent / "lib"
        for candidate in sorted(pyflink_lib.glob(JAR_GLOB)):
            return to_uri(str(candidate))
    except ImportError:
        pass

    # 4. Common system install paths
    system_roots = [
        "/opt/flink/lib",
        "/opt/flink/plugins/kafka",
        "/usr/local/flink/lib",
        "/usr/lib/flink/lib",
        # Docker image convention used in confluentinc/cp-flink
        "/opt/bitnami/flink/lib",
    ]
    for root in system_roots:
        for candidate in sorted(glob.glob(os.path.join(root, JAR_GLOB))):
            return to_uri(candidate)

    # 5. Walk FLINK_HOME if set
    flink_home = os.environ.get("FLINK_HOME", "").strip()
    if flink_home:
        for candidate in sorted(
            glob.glob(os.path.join(flink_home, "**", JAR_GLOB), recursive=True)
        ):
            return to_uri(candidate)

    # Nothing found — give the user a clear fix
    raise RuntimeError(
        "\n"
        "Could not locate the Flink Kafka connector JAR.\n"
        "Fix any ONE of the following:\n\n"
        "  Option A (recommended) — drop the JAR next to flink_app.py:\n"
        "    mkdir -p src/streaming/jars\n"
        "    # copy flink-sql-connector-kafka-*.jar into that folder\n\n"
        "  Option B — set an env var (docker-compose or shell):\n"
        "    FLINK_KAFKA_JAR=/absolute/path/to/flink-sql-connector-kafka-3.4.0-1.20.jar\n\n"
        "  Option C — put it in FLINK_HOME/lib:\n"
        "    export FLINK_HOME=/opt/flink\n"
        "    cp flink-sql-connector-kafka-*.jar $FLINK_HOME/lib/\n\n"
        "Download the JAR from:\n"
        "  https://repo1.maven.org/maven2/org/apache/flink/"
        "flink-sql-connector-kafka/3.4.0-1.20/\n"
    )


def main():
    model_path = os.environ.get("MODEL_PATH", "models/fraud_model.joblib")
    config_path = os.environ.get("CONFIG_PATH", "models/feature_config.json")
    kafka_bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    input_topic = "transactions"
    anomaly_topic = "anomaly_transactions"
    normal_topic = "normal_transactions"

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    # Locate the Kafka connector JAR — works on Windows, Linux, macOS, and Docker.
    env.add_jars(_find_kafka_jar())

    #  SOURCE
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

    #  Fix 4: enforce string type early
    stream = stream.map(
        lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x,
        output_type=Types.STRING()
    )

    #  Processing
    enriched_stream = stream.map(
        FraudMapFunction(model_path, config_path),
        output_type=Types.STRING()   #  Fix 5: VERY IMPORTANT
    ).filter(lambda x: x is not None)

    anomalies = enriched_stream.filter(
        lambda x: json.loads(x)["is_anomaly"]
    )

    normals = enriched_stream.filter(
        lambda x: not json.loads(x)["is_anomaly"]
    )

    #  SINKS
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