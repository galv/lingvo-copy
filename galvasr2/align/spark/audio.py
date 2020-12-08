from collections import namedtuple

import numpy as np
from webrtcvad import Vad

AudioChunk = namdtuple('AudioChunk', ['byte_buffer', 'start_time_ms', 'end_time_ms'])

def vad_split(audio_frames: np.ndarray,
              num_padding_frames=10,
              threshold=0.5,
              aggressiveness=3):
    sample_rate = 16_000
    channels = 1
    width = 2  # bytes
    if aggressiveness not in [0, 1, 2, 3]:
        raise ValueError('VAD-splitting aggressiveness mode has to be one of 0, 1, 2, or 3')
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    vad = Vad(aggressiveness)
    voiced_frames = []
    frame_duration_ms = 0
    frame_index = 0
    for frame_index in range(audio_frames.shape[0] // sample_rate // )
    for frame_index, frame in enumerate(audio_frames):
        frame_duration_ms = get_pcm_duration(len(frame), audio_format) * 1000
        if int(frame_duration_ms) not in [10, 20, 30]:
            raise ValueError('VAD-splitting only supported for frame durations 10, 20, or 30 ms')
        is_speech = vad.is_speech(frame, sample_rate)
        # If this 10ms chunk doesn't have speech
        if not triggered:
            ring_buffer.append((frame, is_speech))
            # Then get the number of voiced 
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If half of the frames are voiced, then consider all frames voiced.

            # This is very suspicious. It's essentially dropping
            # unvoiced frames from the audio. Is that the right thing
            # to do?
            if num_voiced > threshold * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                # There should be no overlap whatsoever between voiced chunks. Hmm.
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > threshold * ring_buffer.maxlen:
                triggered = False
                # Confusing!
                yield b''.join(voiced_frames), \
                      frame_duration_ms * max(0, frame_index - len(voiced_frames)), \
                      frame_duration_ms * frame_index
                ring_buffer.clear()
                voiced_frames = []
    # This seems like clean up at the end. Could use return instead of
    # yield here, right? I don't think so...
    if len(voiced_frames) > 0:
        # It's a generator! A byte string. So VAD operates on raw bytes...
        yield b''.join(voiced_frames), \
              frame_duration_ms * (frame_index - len(voiced_frames)), \
              frame_duration_ms * (frame_index + 1)

def read_frames_from_file(audio_path, audio_format=DEFAULT_FORMAT, frame_duration_ms=30, yield_remainder=False):
    with AudioFile(audio_path, audio_format=audio_format) as wav_file:
        for frame in read_frames(wav_file, frame_duration_ms=frame_duration_ms, yield_remainder=yield_remainder):
            yield frame
