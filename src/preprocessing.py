# src/preprocessing.py
from __future__ import annotations
from typing import List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


def spark_session(app: str = "titanic-preprocess") -> SparkSession:
    """
    Create (or get) a SparkSession with a deterministic timezone.
    """
    return (
        SparkSession.builder.appName(app)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def featurize(df: DataFrame, is_train: bool, label_col: str = "Survived") -> DataFrame:
    """
    Minimal Titanic feature engineering with sane defaults.
    Keeps only the columns needed by the training pipeline.
    """
    # Encode Sex, Embarked
    df = df.withColumn("Sex", F.when(F.col("Sex") == "male", 1).otherwise(0))
    df = df.withColumn(
        "Embarked",
        F.when(F.col("Embarked") == "S", 0)
         .when(F.col("Embarked") == "C", 1)
         .when(F.col("Embarked") == "Q", 2)
         .otherwise(0),
    )

    # Fill NA with simple statistics/defaults
    df = df.fillna(
        {
            "Age": 28.0,
            "Fare": 14.0,
            "SibSp": 0,
            "Parch": 0,
            "Pclass": 3,
            "Embarked": 0,
        }
    )

    # Engineered features
    df = df.withColumn("FamilySize", F.col("SibSp") + F.col("Parch") + F.lit(1))
    df = df.withColumn("IsAlone", F.when(F.col("FamilySize") == 1, 1).otherwise(0))

    # Keep a consistent feature set
    keep: List[str] = [
        "PassengerId",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
    ]
    if is_train and (label_col in df.columns):
        keep.append(label_col)

    return df.select(*[c for c in keep if c in df.columns])


def run(train_in: str, test_in: str, train_out: str, test_out: str, label_col: str = "Survived") -> None:
    spark = spark_session()

    # Read raw CSVs
    train = spark.read.csv(train_in, header=True, inferSchema=True)
    test  = spark.read.csv(test_in,  header=True, inferSchema=True)

    # Transform
    train_p = featurize(train, is_train=True,  label_col=label_col)
    test_p  = featurize(test,  is_train=False, label_col=label_col)

    # Write as single-file CSV directories that DVC tracks as outs
    (train_p.coalesce(1)
           .write.mode("overwrite")
           .option("header", True)
           .csv(train_out))

    (test_p.coalesce(1)
          .write.mode("overwrite")
          .option("header", True)
          .csv(test_out))

    spark.stop()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Preprocess Titanic CSVs with Spark.")
    p.add_argument("--train-in", required=True, help="Path to raw train.csv")
    p.add_argument("--test-in",  required=True, help="Path to raw test.csv")
    p.add_argument("--train-out", required=True, help="Output dir for processed train CSV (Spark folder)")
    p.add_argument("--test-out",  required=True, help="Output dir for processed test CSV (Spark folder)")
    p.add_argument("--label-col", default="Survived", help="Label column name in training data")
    args = p.parse_args()

    run(
        train_in=args.train_in,
        test_in=args.test_in,
        train_out=args.train_out,
        test_out=args.test_out,
        label_col=args.label_col,
    )
