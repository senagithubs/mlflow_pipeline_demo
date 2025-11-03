from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import mlflow
import mlflow.pyfunc
import pickle
import os


tracking_path = "file:///C:/mlflow_test/mlruns"  # Dilersen C:/mlflow_demo da olabilir
os.makedirs("C:/mlflow_test/mlruns", exist_ok=True)

mlflow.set_tracking_uri(tracking_path)
mlflow.set_experiment("Spark_MLflow_Test")


spark = SparkSession.builder \
    .appName("TUBITAK_MLflow_Pipeline") \
    .master("local[*]") \
    .config("spark.ui.showConsoleProgress", "false") \
    .getOrCreate()

print("✅ Spark started with version:", spark.version)


csv_path = "datasets/TUBITAK_data_280925__041025.csv"
df = spark.read.csv(csv_path, header=True, inferSchema=True)
print("📈 Veri başarıyla yüklendi! Satır sayısı:", df.count())


numeric_cols = [c for c, t in df.dtypes if t in ("int", "double", "float")]
feature_cols = numeric_cols[:-1]
label_col = numeric_cols[-1]

vec = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_vec = vec.transform(df).select("features", label_col)


lr = LinearRegression(featuresCol="features", labelCol=label_col)
model = lr.fit(df_vec)

print("✅ Model başarıyla eğitildi.")


with mlflow.start_run(run_name="spark_pipeline_fixed"):

    mlflow.log_param("app_name", "TUBITAK_MLflow_Pipeline")
    mlflow.log_metric("rmse", model.summary.rootMeanSquaredError)
    mlflow.log_metric("r2", model.summary.r2)

    # Spark modelini pickle olarak kaydet ve MLflow’a yükle
    model_path = "spark_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    mlflow.log_artifact(model_path)
    os.remove(model_path)

    print("✅ Model ve metrikler MLflow’a başarıyla kaydedildi!")


spark.stop()
print("✅ Spark oturumu kapatıldı.")

print("\n🎯 Tüm işlem başarıyla tamamlandı!")
print("💻 MLflow UI'ı görmek için terminalde şu komutu çalıştır:")
print("mlflow ui --backend-store-uri file:///C:/mlflow_test/mlruns")
print("ve ardından tarayıcıda http://127.0.0.1:5000 adresine git.")
