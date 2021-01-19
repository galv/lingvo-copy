from lingvo.tasks.asr import frontend as asr_frontend

@model_registry.RegisterSingleTaskModel
class InferenceOnly(base_model_params.SingleTaskModelParams):
  def RawInputParams(self):
    """
    Reads in a raw waveform, where samples are float32 and in the range[-1,1]
    """
    p = input_generator.AsrInput.Params()

    # This is copied from audio_lib.py. Ideally these configurations
    # would be split off into a function, but I want to minimize merge
    # conflicts with lingvo for now. Of course, a merge conflict would
    # indicate that parameters had changed, so my choice is clearly
    # wrong, but oh well.
    p.frontend = asr_frontend.MelAsrFrontend.Params()
    p.sample_rate = 16000.
    p.frame_size_ms = 25.
    p.frame_step_ms = 10.
    p.num_bins = 80
    p.lower_edge_hertz = 125.
    p.upper_edge_hertz = 7600.
    p.preemph = 0.97
    p.noise_scale = 0.
    p.pad_end = False

    p.file_datasource = datasource.PrefixedDataSource.Params()
    p.file_datasource.file_type = 'tfrecord'
    p.file_datasource.file_pattern_prefix = 'gs://the-peoples-speech-west-europe/Librispeech'

    # We have only
    p.frame_size = 1
    p.append_eos_frame = False

    p.pad_to_max_seq_length = True
    p.file_random_seed = 0
    p.file_buffer_size = 10000
    # Set this to whatever
    p.file_parallelism = 2

    # Should be set very high...
    p.source_max_length = 40_000 # 7412603 - 7389537
    p.bucket_upper_bound = [40_000]

    p.bucket_batch_limit = [1]

    return p

  def Inference(self):
    p = RawInputParams()
    p.file_datasource.file_pattern = '*.tfrecord'
    # Keep it simple for now. This should be overridable
    p.num_samples = 8
    return p

  def Task(self):
    p = ctc_model.CTCModel.Params()
    p.name = 'librispeech'

    # No default encoder params in this class.

    tp = p.train
    tp.learning_rate = 1e-4
    tp.lr_schedule = schedule.ContinuousSchedule.Params().Set(
        start_step=25_000, half_life_steps=5_000, min=1e-6)
    tp.scale_gradients = False
    tp.l2_regularizer_weight = None

    # Setting p.eval.samples_per_summary to a large value ensures that dev,
    # devother, test, testother are evaluated completely (since num_samples for
    # each of these sets is less than 5000), while train summaries will be
    # computed on 5000 examples.
    p.eval.samples_per_summary = 2700
    p.eval.decoder_samples_per_summary = 2700

    return p

  def ProgramSchedule(self):
    return program.SimpleProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=500,
        eval_dataset_names=['Inference'],
        eval_steps_per_loop=1,
        decode_steps_per_loop=0)
