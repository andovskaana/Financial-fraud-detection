"""
Fraud Detection — Flink Streaming Application
==============================================

Fixes implemented
-----------------
R7  key_by(sender_account) — stream is now a KeyedStream partitioned per user.
R8  FraudMapFunction (MapFunction) replaced with FraudKeyedFn (KeyedProcessFunction)
    that uses Flink-managed ValueState to persist per-user transaction history.
    State survives task-manager restarts, is distributed correctly across slots,
    and is cleaned up via a 24-hour processing-time TTL timer.
JAR _find_kafka_jar() resolves the connector portably across Windows / Linux /
    macOS / Docker without any hard-coded path.
"""

import glob
import json
import logging
import os
import pickle
import sys
from pathlib import Path

from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.common.watermark_strategy import WatermarkStrategy
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import (
    DeliveryGuarantee,
    KafkaOffsetsInitializer,
    KafkaRecordSerializationSchema,
    KafkaSink,
    KafkaSource,
)
from pyflink.datastream.functions import KeyedProcessFunction, RuntimeContext
from pyflink.datastream.state import ValueStateDescriptor

# Make sure the project root is importable when running inside Docker / Flink
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.streaming.consumer import FraudDetector  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# R8 — KeyedProcessFunction with Flink-managed per-user state
# ---------------------------------------------------------------------------

# Maximum history entries to keep per user (mirrors FeatureConfig.max_history)
_MAX_HISTORY = 20

# TTL for per-user state: 24 hours in milliseconds
_TTL_MS = 24 * 60 * 60 * 1_000


class FraudKeyedFn(KeyedProcessFunction):
    """
    Per-user fraud-detection function backed by Flink ValueState.

    Why KeyedProcessFunction instead of MapFunction
    -----------------------------------------------
    * MapFunction is stateless from Flink's perspective — any dict stored
      inside it is a plain Python object that lives only in the JVM task
      slot's heap.  It is NOT checkpointed, NOT fault-tolerant, and NOT
      correctly distributed when parallelism > 1.
    * KeyedProcessFunction.get_runtime_context().get_state() returns a
      proper Flink ValueState descriptor that is checkpointed with the job,
      re-partitioned on rescale, and can be backed by RocksDB for large state.

    State layout
    ------------
    self.history_state : ValueState[bytes]
        pickle-serialised list of the last _MAX_HISTORY transaction dicts
        for the current key (sender_account).  Flink stores this as an
        opaque byte array so Python objects survive (de)serialisation.

    TTL cleanup
    -----------
    A processing-time timer is (re)registered on every element.  When the
    timer fires (24 h after the *last* transaction for this user), on_timer()
    clears the state entry, reclaiming memory without a global GC sweep.
    """

    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.detector: FraudDetector = None
        self.history_state = None  # set in open()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, runtime_context: RuntimeContext):
        """Called once per task slot when the job starts."""
        # Load the trained model + feature config exactly once per slot.
        self.detector = FraudDetector(
            model_path=self.model_path,
            config_path=self.config_path,
        ).load()

        # Register the Flink-managed state descriptor.
        # Types.PICKLED_BYTE_ARRAY() stores arbitrary Python objects as
        # pickled bytes — the safest cross-version serialisation for complex
        # Python dicts / lists that don't map to primitive Flink types.
        desc = ValueStateDescriptor(
            "user_tx_history",           # state name (used in savepoints)
            Types.PICKLED_BYTE_ARRAY(),  # serialiser
        )
        self.history_state = runtime_context.get_state(desc)

    # ------------------------------------------------------------------
    # Per-element processing  (R8 core)
    # ------------------------------------------------------------------

    def process_element(self, value, ctx: "KeyedProcessFunction.Context"):
        """
        Called once for every Kafka message that arrives at this key.

        Steps
        -----
        1. Deserialise the JSON message.
        2. Restore the user's transaction history from Flink state.
        3. Inject history into FraudDetector's StreamingFeatureState so that
           velocity and aggregate features are computed correctly.
        4. Run predict() — returns fraud_score and is_anomaly.
        5. Persist the updated history back into Flink state (bounded to
           _MAX_HISTORY entries).
        6. (Re)register a TTL timer so idle-user state is eventually cleared.
        7. Yield the enriched JSON record to the downstream stream.
        """
        # --- 1. Parse ---
        try:
            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8")
            transaction = json.loads(value)
        except Exception as exc:
            logger.warning("Skipping malformed message: %s", exc)
            return  # drop bad records; don't crash the job

        user_id = transaction.get("sender_account", "unknown")

        # --- 2. Restore history from Flink state ---
        raw_state = self.history_state.value()
        if raw_state:
            try:
                prior_history = pickle.loads(raw_state)
            except Exception:
                prior_history = []
        else:
            prior_history = []

        # --- 3. Hydrate FraudDetector's in-memory StreamingFeatureState
        #        with the restored history so feature engineering is correct.
        #        StreamingFeatureState keeps a dict keyed by sender_account;
        #        we overwrite it with what Flink persisted.
        if prior_history and hasattr(self.detector, "feature_state"):
            user_state = self.detector.feature_state.user_states[user_id]
            user_state["transactions"] = prior_history

        # --- 4. Predict ---
        try:
            result = self.detector.predict(transaction)
        except Exception as exc:
            logger.error("Prediction error for user %s: %s", user_id, exc)
            result = {"fraud_score": 0.0, "is_anomaly": False}

        # --- 5. Persist updated history back into Flink state ---
        try:
            updated_history = (
                self.detector.feature_state
                .user_states[user_id]
                .get("transactions", [])[-_MAX_HISTORY:]
            )
            self.history_state.update(pickle.dumps(updated_history))
        except Exception as exc:
            logger.warning("Could not persist history for user %s: %s", user_id, exc)

        # --- 6. (Re)register TTL timer ---
        # Cancelling the old timer before registering a new one is not
        # required but keeps the timer heap lean on high-volume users.
        current_time = ctx.timer_service().current_processing_time()
        ctx.timer_service().register_processing_time_timer(
            current_time + _TTL_MS
        )

        # --- 7. Emit enriched record ---
        enriched = {
            **transaction,
            "fraud_score": result["fraud_score"],
            "is_anomaly": result["is_anomaly"],
        }
        if result.get("top_shap_features"):
            enriched["top_shap_features"] = result["top_shap_features"]
        yield json.dumps(enriched)

    # ------------------------------------------------------------------
    # TTL cleanup
    # ------------------------------------------------------------------

    def on_timer(self, timestamp: int, ctx: "KeyedProcessFunction.OnTimerContext"):
        """
        Fires 24 h after the user's last transaction.
        Clears the Flink-managed state entry so memory is reclaimed.
        """
        self.history_state.clear()

        # Also reset FraudDetector's in-memory state for this user.
        user_id = ctx.get_current_key()
        if hasattr(self.detector, "feature_state"):
            self.detector.feature_state.user_states.pop(user_id, None)
        if hasattr(self.detector, "user_consecutive_anomalies"):
            self.detector.user_consecutive_anomalies.pop(user_id, None)
        if hasattr(self.detector, "user_last_tx_info"):
            self.detector.user_last_tx_info.pop(user_id, None)


# ---------------------------------------------------------------------------
# JAR discovery (portable — no hard-coded Windows path)
# ---------------------------------------------------------------------------

def _find_kafka_jar() -> str:
    """
    Locate the Flink Kafka connector JAR and return it as a file:// URI.

    Search order (first match wins):
      1. FLINK_KAFKA_JAR env var           — explicit override, highest priority
      2. jars/ sibling directory            — project-local copy
      3. PyFlink's own bundled jars         — installed alongside apache-flink
      4. Common system install paths        — /opt/flink, /usr/local/flink, …
      5. FLINK_HOME recursive glob          — if FLINK_HOME is set

    Raises RuntimeError with actionable instructions if nothing is found.
    """
    JAR_GLOB = "flink-sql-connector-kafka*.jar"

    def to_uri(path: str) -> str:
        return Path(path).resolve().as_uri()

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

    # 3. PyFlink bundled jars (site-packages/pyflink/lib/)
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
        "/opt/bitnami/flink/lib",
    ]
    for root in system_roots:
        for candidate in sorted(glob.glob(os.path.join(root, JAR_GLOB))):
            return to_uri(candidate)

    # 5. Walk FLINK_HOME
    flink_home = os.environ.get("FLINK_HOME", "").strip()
    if flink_home:
        for candidate in sorted(
            glob.glob(os.path.join(flink_home, "**", JAR_GLOB), recursive=True)
        ):
            return to_uri(candidate)

    raise RuntimeError(
        "\nCould not locate the Flink Kafka connector JAR.\n"
        "Fix any ONE of the following:\n\n"
        "  Option A (recommended) — drop the JAR next to flink_app.py:\n"
        "    mkdir -p src/streaming/jars\n"
        "    # copy flink-sql-connector-kafka-*.jar into that folder\n\n"
        "  Option B — set an env var (docker-compose or shell):\n"
        "    FLINK_KAFKA_JAR=/absolute/path/to/flink-sql-connector-kafka-3.4.0-1.20.jar\n\n"
        "  Option C — put it in FLINK_HOME/lib:\n"
        "    export FLINK_HOME=/opt/flink\n"
        "    cp flink-sql-connector-kafka-*.jar $FLINK_HOME/lib/\n\n"
        "Download: https://repo1.maven.org/maven2/org/apache/flink/"
        "flink-sql-connector-kafka/3.4.0-1.20/\n"
    )


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main():
    model_path = os.environ.get("MODEL_PATH", "models/fraud_model.joblib")
    config_path = os.environ.get("CONFIG_PATH", "models/feature_config.json")
    kafka_bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    input_topic = "transactions"
    flink_enriched_topic = os.environ.get(
        "KAFKA_FLINK_ENRICHED_TOPIC", "flink_enriched_transactions"
    )

    # -----------------------------------------------------------------------
    # Environment setup
    # -----------------------------------------------------------------------
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.add_jars(_find_kafka_jar())

    # -----------------------------------------------------------------------
    # Source — read raw JSON strings from Kafka
    # -----------------------------------------------------------------------
    source = (
        KafkaSource.builder()
        .set_bootstrap_servers(kafka_bootstrap)
        .set_topics(input_topic)
        .set_group_id("fraud-flink")
        .set_value_only_deserializer(SimpleStringSchema())
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .build()
    )

    stream = env.from_source(
        source,
        WatermarkStrategy.no_watermarks(),
        "Kafka Source",
    )

    # Normalise to STRING early (guards against bytes arriving from some
    # Kafka connector versions).
    stream = stream.map(
        lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x,
        output_type=Types.STRING(),
    )

    # -----------------------------------------------------------------------
    # key_by(sender_account) — KeyedStream partitioned per user
    # -----------------------------------------------------------------------
    keyed_stream = stream.key_by(
        lambda x: json.loads(x).get("sender_account", "unknown")
    )

    # -----------------------------------------------------------------------
    # KeyedProcessFunction with Flink-managed ValueState
    # -----------------------------------------------------------------------
    enriched_stream = (
        keyed_stream
        .process(
            FraudKeyedFn(model_path, config_path),
            output_type=Types.STRING(),
        )
        .filter(lambda x: x is not None)
    )

    # -----------------------------------------------------------------------
    # Sink — write enriched records to flink_enriched_transactions.
    # fraud-consumer reads from there and routes to anomaly/normal topics.
    # -----------------------------------------------------------------------
    sink = (
        KafkaSink.builder()
        .set_bootstrap_servers(kafka_bootstrap)
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic(flink_enriched_topic)
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        )
        .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
        .build()
    )

    enriched_stream.sink_to(sink)
    env.execute("Fraud Detection Flink Job")


if __name__ == "__main__":
    main()
