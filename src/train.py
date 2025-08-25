# src/train.py
from __future__ import annotations
import argparse, json, os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone"]

def spark_session(app="titanic-train"):
    """Creates a SparkSession configured for a local environment."""
    return (
        SparkSession.builder.appName(app)
        .config("spark.driver.host", "127.0.0.1")  # Fix for Spark RPC errors
        .getOrCreate()
    )

def f1_from(pred, label="label", predcol="prediction") -> float:
    """Calculates F1 score from a predictions DataFrame."""
    agg = pred.groupBy().agg(
        F.sum(F.when((F.col(label)==1) & (F.col(predcol)==1),1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col(label)==0) & (F.col(predcol)==1),1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col(label)==1) & (F.col(predcol)==0),1).otherwise(0)).alias("fn"),
    ).collect()[0]
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    return (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0

def get_current_production_auc(model_name: str) -> float | None:
    """Looks up the current Production model in MLflow and returns its val_auc."""
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        if not prod_versions: return None
        prod_versions.sort(key=lambda v: int(v.creation_timestamp), reverse=True)
        run = client.get_run(prod_versions[0].run_id)
        return run.data.metrics.get("val_auc")
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--label-col", default="Survived")
    args = ap.parse_args()

    spark = spark_session()
    df = spark.read.csv(args.in_path, header=True, inferSchema=True).cache()
    if args.label_col != "label" and args.label_col in df.columns:
        df = df.withColumnRenamed(args.label_col, "label")

    train, valid = df.randomSplit([0.8, 0.2], seed=42)
    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)

    candidates = {
        "LogReg": LogisticRegression(featuresCol="features", labelCol="label", maxIter=200),
        "RandomForest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=200, maxDepth=8),
        "GBT": GBTClassifier(featuresCol="features", labelCol="label", maxIter=100, maxDepth=5),
    }
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    best = {"name": None, "auc": -1.0, "f1": -1.0, "model": None, "run_id": None}

    mlflow.set_experiment(args.model_name)
    for name, est in candidates.items():
        with mlflow.start_run(run_name=f"DEV_{name}") as run:
            model = Pipeline(stages=[assembler, scaler, est]).fit(train)
            pred = model.transform(valid)
            auc = float(evaluator.evaluate(pred))
            f1v = float(f1_from(pred))

            mlflow.log_param("algorithm", name)
            mlflow.log_metrics({"val_auc": auc, "val_f1": f1v})
            mlflow.spark.log_model(model, artifact_path="spark_model")

            print(f"[DEV {name}] AUC={auc:.4f} F1={f1v:.4f}")
            if auc > best["auc"] or (auc == best["auc"] and f1v > best["f1"]):
                best.update({"name": name, "auc": auc, "f1": f1v, "model": model, "run_id": run.info.run_id})

    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    best["model"].write().overwrite().save(args.export)
    with open(args.metrics, "w") as f:
        json.dump({"best_model": best["name"], "val_auc": best["auc"], "val_f1": best["f1"]}, f, indent=2)
    print(f"[BEST] {best['name']} -> {args.export} (AUC={best['auc']:.4f})")

    client = MlflowClient()
    reg = mlflow.register_model(f"runs:/{best['run_id']}/spark_model", args.model_name)
    client.transition_model_version_stage(name=args.model_name, version=reg.version, stage="Staging")

    prod_auc = get_current_production_auc(args.model_name)
    if (prod_auc is None) or (best["auc"] > prod_auc):
        client.transition_model_version_stage(
            name=args.model_name, version=reg.version, stage="Production", archive_existing_versions=True
        )
        print(f"[PROMOTION] Model v{reg.version} promoted to Production.")
    else:
        print(f"[STAGING ONLY] Model v{reg.version} kept in Staging.")

    spark.stop()

if __name__ == "__main__":
    main()