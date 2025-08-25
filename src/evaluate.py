# src/evaluate.py
from __future__ import annotations
import argparse, json, os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow

def spark_session(app="titanic-eval"):
    """Creates a SparkSession configured for a local environment."""
    return (
        SparkSession.builder.appName(app)
        .config("spark.driver.host", "127.0.0.1")  # Fix for Spark RPC errors
        .getOrCreate()
    )

def f1_from(pred, label="label", predcol="prediction") -> float:
    """Calculates F1 score from a predictions DataFrame."""
    agg = pred.groupBy().agg(
        F.sum(F.when((F.col(label)==1) & (F.col(predcol)==1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col(label)==0) & (F.col(predcol)==1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col(label)==1) & (F.col(predcol)==0), 1).otherwise(0)).alias("fn"),
    ).collect()[0]
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    return (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--predictions", default="models/predictions_csv")
    ap.add_argument("--label-col", default="Survived")
    ap.add_argument("--experiment", default="TitanicClassifier")
    args = ap.parse_args()

    spark = spark_session()
    df = spark.read.csv(args.in_path, header=True, inferSchema=True)

    has_label = args.label_col in df.columns
    if has_label and args.label_col != "label":
        df = df.withColumnRenamed(args.label_col, "label")

    model = PipelineModel.load(args.model_dir)

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="EVALUATE_ON_TEST_DATA") as run:
        pred = model.transform(df)

        metrics = {}
        if has_label:
            evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
            auc = float(evaluator.evaluate(pred))
            f1v = float(f1_from(pred))
            metrics = {"test_auc": auc, "test_f1": f1v}
        else:
            metrics = {"n_predictions": pred.count()}
        
        mlflow.log_metrics(metrics)

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)

        pred_out = pred.withColumn("probability_str", F.col("probability").cast("string"))
        cols = ["PassengerId", "prediction", "probability_str"]
        (pred_out.select(*[c for c in cols if c in pred_out.columns])
                .coalesce(1).write.mode("overwrite")
                .option("header", True).csv(args.predictions))

    print(f"[EVAL] Metrics: {metrics}")
    spark.stop()

if __name__ == "__main__":
    main()