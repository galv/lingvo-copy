from io import BytesIO
import wave

import numpy as np
import pandas as pd

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
import pyspark.sql.functions as F
from pyspark.sql.functions import array, array_contains, count, explode, lit, sum
from pyspark.sql.types import ArrayType, BinaryType, DoubleType, StructType, StructField, StringType, IntegerType, LongType

from lingvo.tools.audio_lib import DecodeToWav

from galvasr2.align.audio import AudioFormat, vad_split



spark = SparkSession.builder \
                    .master("local") \
                    .appName("Forced Aligner") \
                    .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
                    .getOrCreate()

                    # .config('spark.executor.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true')\
                    # .config('spark.driver.extraJavaOptions=-Dio.netty.tryReflectionSetAccessible=true')\

archive_schema = StructType([
    StructField("created", LongType(), True),
    StructField("d1", StringType(), True),
    StructField("d2", StringType(), True),
    StructField("dir", StringType(), True),
    StructField(
        "files",
        ArrayType(
            StructType([
                StructField("bitrate", StringType(), True),
                StructField("btih", StringType(), True),
                StructField("crc32", StringType(), True),
                StructField("format", StringType(), True),
                StructField("height", StringType(), True),
                StructField("length", StringType(), True),
                StructField("license", StringType(), True),
                StructField("md5", StringType(), True),
                StructField("mtime", StringType(), True),
                StructField("name", StringType(), True),
                StructField("original", StringType(), True),
                StructField("rotation", StringType(), True),
                StructField("sha1", StringType(), True),
                StructField("size", StringType(), True),
                StructField("source", StringType(), True),
                StructField("title", StringType(), True),
                StructField("track", StringType(), True),
                StructField("width", StringType(), True)
            ]), True), 
        True),
    StructField("files_count", LongType(), True),
    StructField("identifier", StringType(), True),
    StructField("item_last_updated", LongType(), True),
    StructField("item_size", LongType(), True),
    StructField(
        "metadata",
        StructType([
            StructField("Length", StringType(), True),
            StructField("addeddate", StringType(), True),
            StructField("adder", StringType(), True),
            StructField("aspect_ratio", StringType(), True),
            StructField("backup_location", StringType(), True),
            StructField("closed_captioning", StringType(), True),
            StructField("collection", ArrayType(StringType(), True), True),
            StructField("color", StringType(), True),
            StructField("contact", StringType(), True),
            StructField("coverage", StringType(), True),
            StructField("creator", StringType(), True),
            StructField("credits", StringType(), True),
            StructField("curation", StringType(), True),
            StructField("date", StringType(), True),
            StructField("description", StringType(), True),
            StructField("director", StringType(), True),
            StructField("duration", StringType(), True),
            StructField("format", StringType(), True),
            StructField("genre", StringType(), True),
            StructField("glob", StringType(), True),
            StructField("holder", StringType(), True),
            StructField("ia_orig__runtime", StringType(), True),
            StructField("identifier", StringType(), True),
            StructField("identifier-access", StringType(), True),
            StructField("identifier-ark", StringType(), True),
            StructField("imdb", StringType(), True),
            StructField("keywords", StringType(), True),
            StructField("language", StringType(), True),
            StructField("lcenseurl", StringType(), True),
            StructField("license", StringType(), True),
            StructField("licenseurl", StringType(), True),
            StructField("licensurl", StringType(), True),
            StructField("mediatype", StringType(), True),
            StructField("noarchivetorrent", StringType(), True),
            StructField("ocr", StringType(), True),
            StructField("omp-locally-produced", StringType(), True),
            StructField("omp-project", StringType(), True),
            StructField("own", StringType(), True),
            StructField("pbcore-genre", StringType(), True),
            StructField("pick", StringType(), True),
            StructField("ppi", StringType(), True),
            StructField("presenter", StringType(), True),
            StructField("producer", StringType(), True),
            StructField("publicdate", StringType(), True),
            StructField("publisher", StringType(), True),
            StructField("release_date", StringType(), True),
            StructField("repub_state", StringType(), True),
            StructField("resource", StringType(), True),
            StructField("runtime", StringType(), True),
            StructField("scanner", StringType(), True),
            StructField("segments", StringType(), True),
            StructField("series", StringType(), True),
            StructField("sound", StringType(), True),
            StructField("sponsor", StringType(), True),
            StructField("subject", StringType(), True),
            StructField("title", StringType(), True),
            StructField("tv-parental-guidelines", StringType(), True),
            StructField("updatedate", StringType(), True),
            StructField("updater", StringType(), True),
            StructField("upload_application", StringType(), True),
            StructField("uploader", StringType(), True),
            StructField("vimeo-height", StringType(), True),
            StructField("vimeo-id", StringType(), True),
            StructField("vimeo-n-entries", StringType(), True),
            StructField("vimeo-playlist", StringType(), True),
            StructField("vimeo-playlist-index", StringType(), True),
            StructField("vimeo-uploader", StringType(), True),
            StructField("vimeo-uploader-id", StringType(), True),
            StructField("vimeo-view-count", StringType(), True),
            StructField("vimeo-webpage-url", StringType(), True),
            StructField("vimeo-width", StringType(), True),
            StructField("year", StringType(), True),
            StructField("youtube-height", StringType(), True),
            StructField("youtube-id", StringType(), True),
            StructField("youtube-n-entries", StringType(), True),
            StructField("youtube-playlist", StringType(), True),
            StructField("youtube-playlist-index", StringType(), True),
            StructField("youtube-uploader", StringType(), True),
            StructField("youtube-uploader-id", StringType(), True),
            StructField("youtube-view-count", StringType(), True),
            StructField("youtube-webpage-url", StringType(), True),
            StructField("youtube-width", StringType(), True)
        ]), True),
])

# bytes, length, sampling_frequency, number_channels
def load_audio_files(spark, file_format: str, base_path: str):
    raw_audio_df = (spark.read.format("binaryFile")
                    .option("pathGlobFilter", f"*/*.{file_format}")
                    .load(base_path))
    # raw_audio_df = (spark.read.format("binaryFile")
    #                 .option("pathGlobFilter", "*.mp3")
    #                 .load("gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/bicycle_today_automobile_tomorrow"))
    # raw_audio_df = (spark.read.format("binaryFile")
    #                 .option("pathGlobFilter", f"*/*.{file_format}")
    #                 .load("gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA/"))
    return raw_audio_df.select('content', F.split(raw_audio_df.path, "[.]")[1].alias("format"),  F.reverse(F.split(raw_audio_df.path, "/"))[1].alias("id"))

# https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html#setting-arrow-batch-size

@pandas_udf("double")
def pandas_plus_one(v: pd.Series) -> pd.Series:
    return v + 1

def prepare_vad_udf(num_padding_frames, threshold, aggressiveness, frame_duration_ms):
    # Each audio file returns multiple voiced fragments. I need an Array, don't I?
    return_type = StructType(
        [
            StructField("start_ms", ArrayType(IntegerType())),
            StructField("end_ms", ArrayType(IntegerType())),
            StructField("voiced_buffer", ArrayType(BinaryType())),
        ]
    )
    # return_type = StructType([StructField("a", BinaryType()), StructField("b", StringType())])
        # [, 
        #  ArrayType(IntegerType()), ArrayType(BinaryType())]))
    # return_type = ArrayType(StructType([StructField("blah", IntegerType())]))
# StructType([StructField("start_ms", IntegerType()), 
#                                         StructField("end_ms", IntegerType()),
#                                         StructField("voiced_buffer", BinaryType())]))

    AUDIO_FORMAT = AudioFormat(sample_rate=16_000, channels=1, sample_byte_width=2)
    FRAME_DURATION_SAMPLES = (AUDIO_FORMAT.sample_rate * frame_duration_ms) // 1000
    FRAME_DURATION_BYTES = (FRAME_DURATION_SAMPLES * AUDIO_FORMAT.channels * 
                            AUDIO_FORMAT.sample_byte_width)
    @pandas_udf(return_type)
    def vad(audio_series: pd.Series, audio_types_series: pd.Series) -> pd.DataFrame:
        # return pd.DataFrame([audio_series, audio_types], cols=["a", "b"])
        # df = pd.DataFrame(columns=['start_ms', 'end_ms', 'voiced_buffer'])
        df_rows = []
        for audio_buffer, audio_type in zip(audio_series, audio_types_series):
            wav_bytes_buffer = BytesIO(DecodeToWav(audio_buffer, audio_type))
            # Is it safe to delete an arrow-allocated buffer? Unsure.
            # del audio_buffer
            with wave.open(wav_bytes_buffer, "rb") as fh:
                num_frames = fh.getnframes()
                assert fh.getframerate() == AUDIO_FORMAT.sample_rate
                assert fh.getnchannels() == AUDIO_FORMAT.channels
                assert fh.getsampwidth() == AUDIO_FORMAT.sample_byte_width
                pcm_buffer = fh.readframes(num_frames)
            del wav_bytes_buffer
            num_frames = len(pcm_buffer) // FRAME_DURATION_BYTES
            buffers = [pcm_buffer[FRAME_DURATION_BYTES * i: FRAME_DURATION_BYTES * (i + 1)] for i in range(num_frames)]
            del pcm_buffer
            generator = vad_split(buffers, AUDIO_FORMAT, num_padding_frames, 
                                  threshold, aggressiveness)
            voiced_buffer_list, start_ms_list, end_ms_list = zip(*generator)
            # start_ms_list = [int(x) for x in start_ms_list]
            # end_ms_list = [int(x) for x in end_ms_list]
            # Very first one appears to be -30.0...
            # assert all(x >= 0 for x in start_ms_list)
            df_rows.append({"start_ms": start_ms_list, 
                            "end_ms": end_ms_list, 
                            "voiced_buffer": voiced_buffer_list})
            # for voiced_buffer, start_ms, end_ms in generator:
            #     print("START:", start_ms)
            #     print("END:", end_ms)
        return pd.DataFrame(df_rows)
    return vad

# Need to have key present. Dump TFRecords


rescore_output_schema=StructType([StructField("transcribed_fragment", StringType())])
def rescore_with_lm(pdf: pd.DataFrame) -> pd.DataFrame:
    scorer_path = build_lm(pdf[0, 'text'])

    scorer = Scorer(alpha, beta, scorer_path, alphabet)

    out_pdf = pd.DataFrame(cols=["transcribed_fragment"])
    
    for row in pdf.iterrows():
        # Be sure to borrow, not copy
        log_probs = np.frombuffer(row['log_probs'], dtype=np.float32)
        # May want to apply this outside the function, via spark's exp function
        probs = np.exp(log_probs)
        n_best_list = ctc_beam_search_decoder(probs, alphabet, beam_size, cutoff_prob, cutoff_top_n, scorer)
        out_pdf.append((n_best_list[0], ))
    return out_pdf

def do_alignment():
    align()

def main(spark):
    audio_df = load_audio_files(spark, "mp3", "gs://the-peoples-speech-west-europe/archive_org/Nov_6_2020/ALL_CAPTIONED_DATA")
    # x = audio_df.collect()
    # print("GALVEZ:", x[0].id)

    vad_udf = prepare_vad_udf(num_padding_frames=10, threshold=0.5,
                              aggressiveness=0, frame_duration_ms=30)
    vad_df = audio_df.withColumn("vad", vad_udf(audio_df.content, audio_df.format))

    # print("GALVEZ:", len(vad_df.collect()[0].start_ms))

    # with open("voiced_buffers.npy", "wb") as fh:
    #     np.save(fh, np.array(vad_result[0].vad.voiced_buffer))

    # vad_result = vad_df.collect()
    # print(vad_result[0].vad.start_ms)

    # from IPython import embed; embed()

    vad_df.select(vad_df.vad.voiced_buffer).write.mode("overwrite").format("tfrecord").save("gs://the-peoples-speech-west-europe/forced-aligner/vad-segments-dump/segments.tfrecord")

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
