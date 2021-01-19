# bazel run galvasr2:spark_forced_aligner

from io import BytesIO
import json
import subprocess
import shlex
import wave

from ftfy import fix_text, guess_bytes
import langid
import numpy as np
import pandas as pd
import srt

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
import pyspark.sql.functions as F
from pyspark.sql.functions import array, array_contains, count, explode, lit, sum
from pyspark.sql.types import ArrayType, BinaryType, DoubleType, FloatType, ShortType, StructType, StructField, StringType, IntegerType, LongType

from lingvo.tools.audio_lib import DecodeToWav

from galvasr2.align.audio import AudioFormat, vad_split

# Caused by: java.lang.IllegalArgumentException: the requested size must be non-negative
#         at org.apache.arrow.util.Preconditions.checkArgument(Preconditions.java:136)
#         at org.apache.arrow.memory.BaseAllocator.buffer(BaseAllocator.java:288)
#         at org.apache.arrow.memory.BaseAllocator.buffer(BaseAllocator.java:277)
#         at org.apache.arrow.vector.ipc.message.MessageSerializer.readMessageBody(MessageSerializer.java:695)
#         at org.apache.arrow.vector.ipc.message.MessageChannelReader.readNext(MessageChannelReader.java:68)
#         at org.apache.arrow.vector.ipc.ArrowStreamReader.loadNextBatch(ArrowStreamReader.java:106)
#         at org.apache.spark.sql.execution.python.PythonArrowOutput$$anon$1.read(PythonArrowOutput.scala:74)
#         at org.apache.spark.sql.execution.python.PythonArrowOutput$$anon$1.read(PythonArrowOutput.scala:94)
#         at org.apache.spark.sql.execution.python.PythonArrowOutput$$anon$1.read(PythonArrowOutput.scala:49)
#         at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:456)
#         at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
#         at scala.collection.Iterator$$anon$11.hasNext(Iterator.scala:489)
#         at scala.collection.Iterator$$anon$10.hasNext(Iterator.scala:458)
#         at org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.agg_doAggregateWithoutKey_0$(Unknown Source)
#         at org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.processNext(Unknown Source)

def DecodeToWavPipe(input_bytes, fmt):
  cmd = f'sox -t {fmt} - -t wav --channels 1 --rate 16000 --encoding signed --bits 16 -'
  p = subprocess.Popen(shlex.split(cmd),
                       stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
  out, err = p.communicate(input=input_bytes)
  assert p.returncode == 0, err
  return out

                            # .config("spark.driver.memory", "4g")\
                            # .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
#                            .config("spark.executor.memory", "4g")\
spark = SparkSession.builder \
                            .master("local[8]") \
                            .appName("Forced Aligner") \
                            .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
                            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1")\
                            .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
                            .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
                            .config("spark.driver.memory", "100g")\
                            .config("spark.executor.memory", "100g")\
                            .getOrCreate()
                            # .config("spark.memory.offHeap.enabled", "true")\
                            # .config("spark.memory.offHeap.size", "20g")\
 # -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps")\
#  -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/development/lingvo-source/executor-mem-dump.hlog -Xlog:gc
#  -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/development/lingvo-source/driver-mem-dump.hlog -Xlog:gc
spark.sparkContext.setLogLevel("INFO") # "ALL"

@pandas_udf(StringType())
def srt_to_text(srt_file_contents: pd.Series) -> pd.Series:
  def helper(content: str) -> str:
    try:
      return " ".join(line.content.replace("\n", " ") for line in srt.parse(content))
    except (srt.SRTParseError, srt.TimestampParseError) as exc:
      # Is this really the best way to log in a pandas UDF?
      print("WARNING: trouble parsing content")
      print(exc)
      return ""
  return srt_file_contents.apply(helper)

@pandas_udf(StringType())
def infer_language_func(text_column: pd.Series) -> pd.Series:
  return text_column.apply(lambda string: langid.classify(string)[0] if string else "")


# bytes, length, sampling_frequency, number_channels
def load_audio_files(spark, base_path: str):
    import itertools
    with open("/development/lingvo-source/mp3_files.txt") as fh:
      paths = (path.rstrip() for path in fh.readlines() if "[" not in path and "]" not in path)
      # paths = list(itertools.islice(paths, 10))
      paths = list(paths)
    raw_audio_df = (spark.read.format("binaryFile")
                    # .option("pathGlobFilter", "*.mp3")
                    # .option("recursiveFileLookup", "true")
                    .load(paths))
                    #.load("gs://the-peoples-speech-west-europe/archive_org/small_dataset/Highway_and_Hedges_Outreach_Ministries_Show_-_Show_49/Highway_and_Hedges_Outreach_Ministries_Show_-_Show_49.mp3"))
                    #.load("gs://the-peoples-speech-west-europe/archive_org/small_dataset/hrs02APR2359_090603/hrs02APR2359_090603.mp3"))

    print("GALVEZ:", raw_audio_df.rdd.getNumPartitions())
    
    return raw_audio_df.select('content',
                               F.reverse(F.split(raw_audio_df.path, "[.]"))[0].alias("format"),
                               # We will have repeats with this form of ID... It does not fulfill the purpose of an primary key...
                               # 44635        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/01-Ml.Z.Ragi-JinnandJadoo18.05.05.asr.srt
                               # 53884        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/02-Ml.Z.Ragi-JinnandJadoo25.05.05.asr.srt
                               # 55971        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/03-Ml.Z.Ragi-JinnandJadoo01.06.05.asr.srt
                               # 48287        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/04-Ml.Z.Ragi-JinnandJadoo08.06.05.asr.srt
                               # 44184        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/05-Ml.Z.Ragi-JinnandJadoo22.06.05.asr.srt
                               # 29040        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/06-Ml.Z.Ragi-JinnandJadoo29.06.05.asr.srt
                               # 53849        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/07-Ml.Z.Ragi-JinnandJadoo20.07.05.asr.srt
                               # 54745        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/08-Ml.Z.Ragi-JinnandJadoo27.07.05.asr.srt
                               # 44990        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/09-Ml.Z.Ragi-JinnandJadoo03.08.05.asr.srt
                               # 47756        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/10-Ml.Z.Ragi-JinnandJadoo10.08.05.asr.srt
                               # 46275        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/11-Ml.Z.Ragi-JinnandJadoo07.09.05.asr.srt
                               # 35660        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/12-Ml.Z.Ragi-JinnandJadoo14.09.05.asr.srt
                               # 50201        gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/07Ml.Z.RagiJinnandJadoo20.07.05/13-Ml.Z.Ragi-JinnandJadoo21.09.05.asr.srt
                               F.reverse(F.split(raw_audio_df.path, "/"))[1].alias("id"))

@pandas_udf(StringType())
def fix_text_udf(binary_column: pd.Series) -> pd.Series:
  return binary_column.apply(lambda b: fix_text(guess_bytes(b)[0]))

def load_transcripts(spark, base_path: str):
    srt_df = (spark.read.format("binaryFile")
              .option("pathGlobFilter", "*.srt") # Will change to "*.{srt,vtt}" at some point... cc5 file format as well...
              .option("recursiveFileLookup", "true")
              .load(base_path))
    # Note the duplication with load_audio_files
    return srt_df.select(srt_to_text(fix_text_udf(srt_df.content)).alias('transcript'),
                         F.reverse(F.split(srt_df.path, "/"))[1].alias("id"))

# https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html#setting-arrow-batch-size

# Caused by: java.lang.IllegalArgumentException: the requested size must be non-negative
#         at org.apache.arrow.util.Preconditions.checkArgument(Preconditions.java:136)
#         at org.apache.arrow.memory.BaseAllocator.buffer(BaseAllocator.java:288)
#         at org.apache.arrow.memory.BaseAllocator.buffer(BaseAllocator.java:277)
#         at org.apache.arrow.vector.ipc.message.MessageSerializer.readMessageBody(MessageSerializer.java:695)
#         at org.apache.arrow.vector.ipc.message.MessageChannelReader.readNext(MessageChannelReader.java:68)
#         at org.apache.arrow.vector.ipc.ArrowStreamReader.loadNextBatch(ArrowStreamReader.java:106)
#         at org.apache.spark.sql.execution.python.PythonArrowOutput$$anon$1.read(PythonArrowOutput.scala:74)
#         at org.apache.spark.sql.execution.python.PythonArrowOutput$$anon$1.read(PythonArrowOutput.scala:94)
#         at org.apache.spark.sql.execution.python.PythonArrowOutput$$anon$1.read(PythonArrowOutput.scala:49)
#         at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:456)
#         at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
#         at scala.collection.Iterator$$anon$11.hasNext(Iterator.scala:489)
#         at scala.collection.Iterator$$anon$10.hasNext(Iterator.scala:458)
#         at org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.processNext(Unknown Source)
#         at org.apache.spark.sql.execution.BufferedRowIterator.hasNext(BufferedRowIterator.java:43)
#         at org.apache.spark.sql.execution.WholeStageCodegenExec$$anon$1.hasNext(WholeStageCodegenExec.scala:729)
#         at scala.collection.Iterator$$anon$11.hasNext(Iterator.scala:489)
#         at scala.collection.Iterator$ConcatIterator.hasNext(Iterator.scala:222)
#         at scala.collection.Iterator$$anon$10.hasNext(Iterator.scala:458)
#         at org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage2.processNext(Unknown Source)
#         at org.apache.spark.sql.execution.BufferedRowIterator.hasNext(BufferedRowIterator.java:43)
#         at org.apache.spark.sql.execution.WholeStageCodegenExec$$anon$1.hasNext(WholeStageCodegenExec.scala:729)
#         at org.apache.spark.sql.execution.datasources.FileFormatWriter$.executeTask(FileFormatWriter.scala:260)
#         at org.apache.spark.sql.execution.datasources.FileFormatWriter$.$anonfun$write$15(FileFormatWriter.scala:205)
#         at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:90)
#         at org.apache.spark.scheduler.Task.run(Task.scala:127)
#         at org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$3(Executor.scala:444)
#         at org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:1377)
#         at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:447)
#         at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1128)
#         at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:628)
#         ... 1 more
def prepare_vad_udf(num_padding_frames, threshold, aggressiveness, frame_duration_ms):
  # Each audio file returns multiple voiced fragments. I need an Array, don't I?
  return_type = StructType(
    [
      StructField("start_ms", ArrayType(IntegerType())),
      StructField("end_ms", ArrayType(IntegerType())),
      StructField("voiced_buffer", ArrayType(ArrayType(ShortType()))),
    ]
  )
  AUDIO_FORMAT = AudioFormat(sample_rate=16_000, channels=1, sample_byte_width=2)
  FRAME_DURATION_SAMPLES = (AUDIO_FORMAT.sample_rate * frame_duration_ms) // 1000
  FRAME_DURATION_BYTES = (FRAME_DURATION_SAMPLES * AUDIO_FORMAT.channels * 
                          AUDIO_FORMAT.sample_byte_width)
  @pandas_udf(return_type)
  def vad(audio_series: pd.Series, audio_types_series: pd.Series) -> pd.DataFrame:
    df_rows = []
    for audio_buffer, audio_type in zip(audio_series, audio_types_series):
      wav_bytes_buffer = BytesIO(DecodeToWavPipe(audio_buffer, audio_type))
      with wave.open(wav_bytes_buffer, "rb") as fh:
        num_frames = fh.getnframes()
        assert fh.getframerate() == AUDIO_FORMAT.sample_rate
        assert fh.getnchannels() == AUDIO_FORMAT.channels
        assert fh.getsampwidth() == AUDIO_FORMAT.sample_byte_width
        pcm_buffer = fh.readframes(num_frames)
        del wav_bytes_buffer
        num_frames = len(pcm_buffer) // FRAME_DURATION_BYTES
        # Can we lazily generate this? Yes.
        buffers = [pcm_buffer[FRAME_DURATION_BYTES * i: FRAME_DURATION_BYTES * (i + 1)] for i in range(num_frames)]
        del pcm_buffer
        generator = vad_split(buffers, AUDIO_FORMAT, num_padding_frames, 
                              threshold, aggressiveness)
        
        voiced_buffer_list, start_ms_list, end_ms_list = [], [], []
        for voiced_buffer, start_ms, end_ms in generator:
          voiced_buffer_list.append(np.frombuffer(voiced_buffer, dtype=np.int16))  #.astype(np.float32) / 32768.0)
          start_ms_list.append(start_ms)
          end_ms_list.append(end_ms)
        del buffers
        # float_voiced_buffer_arrays = []
        # for voiced_buffer in voiced_buffer_list:
        #   float_voiced_buffer_arrays.append(np.frombuffer(voiced_buffer, dtype=np.int16))
        #   # This doubles memory usage. Some wav files are already
        #   # 768MB in size. We are already dangerously close to the 2GB
        #   # memory limits imposed by Spark.
        #   # float_voiced_buffer_arrays.append(np.frombuffer(voiced_buffer, dtype=np.int16).astype(np.float32) / 32768.0)
        # del voiced_buffer_list
        df_rows.append({"start_ms": start_ms_list,
                        "end_ms": end_ms_list,
                        "voiced_buffer": voiced_buffer_list})
    return pd.DataFrame(df_rows)
  return vad

# @pandas_udf(IntegerType())
# def get_length_ms_udf():
#   pass

RESCORE_WITH_LM_OUTPUT_SCHEMA=StructType([StructField("transcribed_fragment", StringType())])
def rescore_with_lm(pdf: pd.DataFrame) -> pd.DataFrame:
  return
  # TODO
  # scorer_path = build_lm(pdf[0, 'text'])

  # scorer = Scorer(alpha, beta, scorer_path, alphabet)

  # out_pdf = pd.DataFrame(cols=["transcribed_fragment"])

  # for row in pdf.iterrows():
  #     # Be sure to borrow, not copy
  #     log_probs = np.frombuffer(row['log_probs'], dtype=np.float32)
  #     # May want to apply this outside the function, via spark's exp function
  #     probs = np.exp(log_probs)
  #     n_best_list = ctc_beam_search_decoder(probs, alphabet, beam_size, cutoff_prob, cutoff_top_n, scorer)
  #     out_pdf.append((n_best_list[0], ))
  # return out_pdf

def main(spark):
  audio_df = load_audio_files(spark, "gs://the-peoples-speech-west-europe/archive_org/small_dataset")
  # vad_out_dir = "gs://the-peoples-speech-west-europe/forced-aligner/vad-segments-dump/small_dataset"
  # audio_df = load_audio_files(spark, "gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA")
  # audio_df = audio_df.limit(100)
  vad_out_dir = "gs://the-peoples-speech-west-europe/forced-aligner/vad-segments-dump/Nov_6_2020/ALL_CAPTIONED_DATA_002"
  vad_udf = prepare_vad_udf(num_padding_frames=10, threshold=0.5,
                            aggressiveness=0, frame_duration_ms=30)
  vad_df = audio_df.withColumn("vad", vad_udf(audio_df.content, audio_df.format))

  # About 4GB used per file, being conservative to avoid out-of-memory errors

  exploded_voiced_buffer_df = vad_df.select(vad_df.id, F.posexplode(vad_df.vad.voiced_buffer))

  # print("COUNT:", exploded_voiced_buffer_df.count())
  # while True:
  #   print("DONE")
          
  
  tfrecord_df = exploded_voiced_buffer_df.select(
    exploded_voiced_buffer_df.id,
    exploded_voiced_buffer_df.col.alias("frames"), # .cast(ArrayType(FloatType()))
    # TODO: Replace this with the actual transcript
    lit("empty").alias("-"),
    F.concat_ws("-", exploded_voiced_buffer_df.id, exploded_voiced_buffer_df.pos).alias("uttid"),
  )

  # pyspark has no "transform" method... Ugh. It is accessible via
  # Spark SQL, though. We need to divide by 32768.0 after casting to
  # float.

  # For whatever reason, we need the cast(ArrayType(FloatType()))
  # part. Otherwise, spark infers an array of doubles...
  tfrecord_df = tfrecord_df.withColumn("frames", F.expr("transform(frames, x -> float(x)  / float(32768))").cast(ArrayType(FloatType())))

  # GALVEZ3:schema=
  # root
  #  |-- id: string (nullable = true)
  #  |-- frames: array (nullable = true)
  #  |    |-- element: double (containsNull = true)
  #  |-- -: string (nullable = false)
  #  |-- uttid: string (nullable = false)

  print("GALVEZ3:schema=")
  tfrecord_df.printSchema()

  # import sys; sys.exit()

  print("GALVEZ2", tfrecord_df.rdd.getNumPartitions())

  # transform(frames, x -> x / 32768.0)

  # tfrecord_df.printSchema()
  # tfrecord_pd = tfrecord_df.toPandas()
  # from IPython import embed; embed()
  # import sys; sys.exit()

  # print("GALVEZ: count:", tfrecord_df.count())
  # tfrecord_df.printSchema()

  # partitionBy("id").
  tfrecord_df.write.mode("overwrite").format("tfrecord").option("recordType", "SequenceExample").save(vad_out_dir)

  # while True:
  #   print("DONE")
  #   pass

  # Need to pass number of samples to decode process. Lingvo is a bit silly.
  # Is this running the entire pipeline twice?
  # num_samples = tfrecord_df.count()
  # print("GALVEZ:num_samples=", num_samples)

  # vad_df.select(vad_df.id, vad_df.vad.voiced_buffer).write.mode("overwrite").partitionBy("id").format("tfrecord").save("gs://the-peoples-speech-west-europe/forced-aligner/vad-segments-dump/segments.tfrecord")

  """ctpu up -name galv-tpu2 -tpu-only -tpu-size v3-8 -tf-version 2.2"""

  """
  bazel run //lingvo:trainer -- --logdir=${LOGDIR} \
  --mode=sync \
  --model=asr.librispeech_ctc.${CLS} \
  --logtostderr \
  --tpu=grpc://${TPUIP}:8470 \
  --job=decoder_dev 2>&1 | tee logs/${CLS}_${DATE}.log
  """

  # best_transcripts_df = log_probs_df.groupBy('id').applyInPandas(rescore_with_lm,
  #                                                                rescore_output_schema)
  # best_transcripts_df


if __name__ == '__main__':
    main(spark)
