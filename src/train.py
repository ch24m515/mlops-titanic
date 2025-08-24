from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

def run_experiment(model, params, train_df, test_df):
    with mlflow.start_run() as run:
        fitted = model.fit(train_df)
        preds = fitted.transform(test_df)

        evaluator = BinaryClassificationEvaluator(labelCol="label")
        auc = evaluator.evaluate(preds)

        # Log params + metrics
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("AUC", auc)

        # Log model artifact
        mlflow.spark.log_model(fitted, "titanic_model")

        print(f"✅ {params['model_type']} | AUC = {auc:.4f}")
        return auc, run.info.run_id

def train_and_register():
    spark = SparkSession.builder.appName("TitanicExperiments").getOrCreate()
    df = spark.read.parquet("data/processed/titanic_cleaned")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Run experiments
    auc_lr, run_lr = run_experiment(
        LogisticRegression(featuresCol="features", labelCol="label"),
        {"model_type": "LogisticRegression"},
        train_df, test_df
    )

    auc_rf, run_rf = run_experiment(
        RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50),
        {"model_type": "RandomForest", "numTrees": 50},
        train_df, test_df
    )

    # Pick best model
    if auc_lr >= auc_rf:
        best_auc, best_run, best_model = auc_lr, run_lr, "LogisticRegression"
    else:
        best_auc, best_run, best_model = auc_rf, run_rf, "RandomForest"

    print(f"🎯 Best Model: {best_model} | AUC={best_auc:.4f}")

    # Register best model in MLflow Model Registry
    model_uri = f"runs:/{best_run}/titanic_model"
    model_name = "TitanicClassifier"
    result = mlflow.register_model(model_uri, model_name)

    print(f"📌 Registered model {model_name}, version {result.version}")

    # Auto-transition to Staging
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging"
    )
    print(f"🚀 Model {model_name} v{result.version} moved to Staging")

    # If better than previous Production, promote
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production"
    )
    print(f"✅ Model {model_name} v{result.version} promoted to Production")

if __name__ == "__main__":
    train_and_register()
