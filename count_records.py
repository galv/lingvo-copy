import tensorflow as tf

tf.enable_eager_execution()
sum(1 for _ in tf.data.TFRecordDataset(file_name))
