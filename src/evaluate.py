# src/evaluate.py
from __future__ import annotations
import argparse, json, os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def spark_session(app="titanic-eval"):
    return SparkSession.builder.appName(app).getOrCreate()

def f1_from(pred, label="label", predcol="prediction") -> float:
    agg = pred.groupBy().agg(
        F.sum(F.when((F.col(label)==1) & (F.col(predcol)==1),1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col(label)==0) & (F.col(predcol)==1),1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col(label)==1) & (F.col(predcol)==0),1).otherwise(0)).alias("fn"),
    ).collect()[0]
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    return (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)     # data/processed/test_csv
    ap.add_argument("--model-dir", required=True)              # models/model_spark
    ap.add_argument("--out", required=True)                    # models/eval_metrics.json
    ap.add_argument("--predictions", default="models/predictions_csv")
    ap.add_argument("--label-col", default="Survived")
    args = ap.parse_args()

    spark = spark_session()
    df = spark.read.csv(args.in_path, header=True, inferSchema=True)

    has_label = args.label_col in df.columns
    if has_label and args.label_col != "label":
        df = df.withColumnRenamed(args.label_col, "label")

    model = PipelineModel.load(args.model_dir)
    pred = model.transform(df)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if has_label:
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = float(evaluator.evaluate(pred))
        f1v = float(f1_from(pred))
        metrics = {"test_auc": auc, "test_f1": f1v}
    else:
        metrics = {"n_predictions": pred.count()}

    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions as CSV (Spark folder)
    (pred.select("PassengerId","prediction","probability")
         .coalesce(1)
         .write.mode("overwrite").option("header", True).csv(args.predictions))

    print(f"[EVAL] {metrics}")
    spark.stop()

if __name__ == "__main__":
    main()
