from datetime import datetime
from pyspark.context import SparkContext
import pyspark.sql.functions as f
import sys

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job

# Initialize Spark and Glue contexts
try:
    print("Initializing Spark and Glue contexts...")
    spark_context = SparkContext.getOrCreate()
    glue_context = GlueContext(spark_context)
    spark = glue_context.spark_session
    print("Contexts initialized successfully.")
except Exception as e:
    print(f"Error initializing Spark/Glue contexts: {e}")
    raise

# Job parameters
try:
    args = getResolvedOptions(sys.argv, ["JOB_NAME"])
    job_name = args["JOB_NAME"]
except Exception as e:
    print(f"Error resolving job parameters: {e}")
    raise

# Initialize the Glue job
try:
    print(f"Starting job: {job_name}")
    job = Job(glue_context)
    job.init(job_name, args)
except Exception as e:
    print(f"Error initializing Glue job: {e}")
    raise

# ETL Process
try:
    # Step 1: Read data from Glue Data Catalog
    print("Reading data from Glue Data Catalog...")
    glue_db = "sales-vrpn2n"
    glue_tbl = "sales-inputsales_raw_bucket_vrpn2n"
    dynamic_frame_read = glue_context.create_dynamic_frame.from_catalog(database=glue_db, table_name=glue_tbl)
    print("Data read successfully.")

    # Step 2: Transformations
    print("Starting data transformations...")
    # Convert to Spark DataFrame
    data_frame = dynamic_frame_read.toDF()

    # Add a 'decade' column
    data_frame = data_frame.withColumn("decade", (f.floor(f.col("ad_date") / 10) * 10).cast("int"))
    data_frame = data_frame.withColumn("intkilometer", (f.floor(f.col("kilometer").getField("DECIMAL")).cast("int")))
    # Group by 'decade' and aggregate
    data_frame_aggregated = data_frame.groupby("decade").agg(
        f.count(f.col("model")).alias("model_count"),
        f.mean(f.col("intkilometer")).alias("kilometer_mean")
    )

    # Sort by 'movie_count' in descending order
    data_frame_aggregated = data_frame_aggregated.orderBy(f.desc("model_count"))
    print("Data transformations completed successfully.")

    # Step 3: Repartition for single file output
    print("Repartitioning data for single file output...")
    data_frame_aggregated = data_frame_aggregated.repartition(1)

    # Convert back to DynamicFrame
    dynamic_frame_write = DynamicFrame.fromDF(data_frame_aggregated, glue_context, "dynamic_frame_write")

    # Step 4: Write to S3
    print("Writing data to S3...")
    s3_write_path = "s3://sales-raw-bucket-vrpn2n/"
    glue_context.write_dynamic_frame.from_options(
        frame=dynamic_frame_write,
        connection_type="s3",
        connection_options={"path": s3_write_path},
        format="csv"
    )
    print("Data written to S3 successfully.")

except Exception as e:
    print(f"Error during ETL process: {e}")
    raise

# Finalize the job
try:
    print("Finalizing the Glue job...")
    job.commit()
    print("Glue job completed successfully.")
except Exception as e:
    print(f"Error committing Glue job: {e}")
    raise