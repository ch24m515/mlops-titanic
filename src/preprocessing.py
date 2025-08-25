# src/preprocessing.py
from __future__ import annotations
import argparse
from typing import List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

def spark_session(app="titanic-preprocess"):
    return (
        SparkSession.builder.appName(app)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

def featurize(df: DataFrame, is_train: bool, label_col: str = "Survived") -> DataFrame:
    # Basic encodings
    df = df.withColumn("Sex", F.when(F.col("Sex") == "male", 1).otherwise(0))
    df = df.withColumn(
        "Embarked",
        F.when(F.col("Embarked") == "S", 0)
         .when(F.col("Embarked") == "C", 1)
         .when(F.col("Embarked") == "Q", 2)
         .otherwise(0),
    )

    # Fill NA
    df = df.fillna({
        "Age": 28.0, "Fare": 14.0,
        "SibSp": 0, "Parch": 0,
        "Pclass": 3, "Embarked": 0
    })

    # Engineered features
    df = df.withColumn("FamilySize", F.col("SibSp") + F.col("Parch") + F.lit(1))
    df = df.withColumn("IsAlone", F.when(F.col("FamilySize") == 1, 1).otherwise(0))

    keep: List[str] = [
        "PassengerId","Pclass","Sex","Age","SibSp","Parch",
        "Fare","Embarked","FamilySize","IsAlone"
    ]
    if is_train and label_col in df.columns:
        keep.append(label_col)
    return df.select(*[c for c in keep if c in df.columns])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-in", required=True)
    ap.add_argument("--test-in", required=True)
    ap.add_argument("--train-out", required=True)  # e.g. data/processed/train_csv
    ap.add_argument("--test-out", required=True)   # e.g. data/processed/test_csv
    ap.add_argument("--label-col", default="Survived")
    args = ap.parse_args()

    spark = spark_session()
    train = spark.read.csv(args.train_in, header=True, inferSchema=True)
    test  = spark.read.csv(args.test_in,  header=True, inferSchema=True)

    train_p = featurize(train, is_train=True,  label_col=args.label_col)
    test_p  = featurize(test,  is_train=False, label_col=args.label_col)

    # Write single-file CSV directories that match dvc.yaml
    (train_p.coalesce(1)
        .write.mode("overwrite").option("header", True).csv(args.train_out))
    (test_p.coalesce(1)
        .write.mode("overwrite").option("header", True).csv(args.test_out))

    spark.stop()

if __name__ == "__main__":
    main()
