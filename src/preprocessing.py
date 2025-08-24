from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

def preprocess(input_path: str, output_path: str):
    spark = SparkSession.builder.appName("TitanicPreprocessing").getOrCreate()
    
    # Load data
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # Drop high-cardinality / leakage columns
    df = df.drop("Name", "Ticket", "Cabin")
    
    # Handle missing Age with median
    median_age = df.approxQuantile("Age", [0.5], 0.0)[0]
    df = df.withColumn("Age", when(col("Age").isNull(), median_age).otherwise(col("Age")))
    
    # Handle missing Fare with median
    median_fare = df.approxQuantile("Fare", [0.5], 0.0)[0]
    df = df.withColumn("Fare", when(col("Fare").isNull(), median_fare).otherwise(col("Fare")))
    
    # Fill missing Embarked with most frequent
    most_common_embarked = df.groupBy("Embarked").count().orderBy(col("count").desc()).first()["Embarked"]
    df = df.withColumn("Embarked", when(col("Embarked").isNull(), most_common_embarked).otherwise(col("Embarked")))
    
    # Encode categorical features
    indexers = [
        StringIndexer(inputCol="Sex", outputCol="Sex_indexed"),
        StringIndexer(inputCol="Embarked", outputCol="Embarked_indexed"),
        StringIndexer(inputCol="Pclass", outputCol="Pclass_indexed")
    ]
    for indexer in indexers:
        df = indexer.fit(df).transform(df)
    
    # Assemble features
    assembler = VectorAssembler(
        inputCols=["Pclass_indexed", "Sex_indexed", "Embarked_indexed", "Age", "SibSp", "Parch", "Fare"],
        outputCol="features_raw"
    )
    df = assembler.transform(df)
    
    # Scale features
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    df = scaler.fit(df).transform(df)
    
    # Keep only label + features
    df = df.select(col("Survived").alias("label"), "features")
    
    # Save processed dataset as Parquet (scalable format)
    df.coalesce(1).write.mode("overwrite").parquet(output_path)
    print(f"✅ Preprocessed data saved at {output_path}")

if __name__ == "__main__":
    preprocess("data/raw/titanic.csv", "data/processed/titanic_cleaned")
