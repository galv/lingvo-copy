from functools import partial
import os

from absl import app
from absl import flags
import tensorflow as tf

flags.DEFINE_string("tfrecord", default="/home/ws15dgalvez/Librispeech/train/train.tfrecords-00097-of-00100", help="")
flags.DEFINE_integer("shards", default=0, help="")

FLAGS = flags.FLAGS

def reduce_func(prefix, key, dataset):
  filename = tf.strings.join([prefix, tf.strings.as_string(key), ".tfrecord"])
  print(filename)
  writer = tf.data.experimental.TFRecordWriter(filename)
  writer.write(dataset.map(lambda _, x: x))
  return tf.data.Dataset.from_tensors(filename)

def main(argv):
  del argv
  # /home/ws15dgalvez/Librispeech/train/train.tfrecords-00097-of-00100
  prefix = FLAGS.tfrecord + "_split_directory/"
  os.makedirs(prefix, exist_ok=True)
  partial_reduce_func = partial(reduce_func, prefix)
  dataset = tf.data.TFRecordDataset(FLAGS.tfrecord)
  # for i, x in dataset:
  # filename = tf.strings.join([prefix, tf.strings.as_string(key), ".tfrecord"])
  # tf.data.TFRecordWriter()
  #   print("GALV:, ", x)
  dataset = dataset.enumerate()
  if FLAGS.shards == 0:
    key_function = lambda i, _: i
  else:
    key_function = lambda i, _: i % FLAGS.shards
  
  dataset = dataset.apply(tf.data.experimental.group_by_window(
    key_function, partial_reduce_func, tf.int64.max
  ))

  list(dataset.as_numpy_iterator())

  assert tf.executing_eagerly()
  

if __name__ == '__main__':
  app.run(main)
