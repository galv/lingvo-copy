# -*- coding: utf-8 -*-
import os
import csv
import json
import heapq

from pathlib import Path
from functools import partial
from utils import MEGABYTE, GIGABYTE, Interleaved
from audio import Sample, DEFAULT_FORMAT, AUDIO_TYPE_WAV, AUDIO_TYPE_OPUS, SERIALIZABLE_AUDIO_TYPES

BIG_ENDIAN = 'big'
INT_SIZE = 4
BIGINT_SIZE = 2 * INT_SIZE
MAGIC = b'SAMPLEDB'

BUFFER_SIZE = 1 * MEGABYTE
CACHE_SIZE = 1 * GIGABYTE

SCHEMA_KEY = 'schema'
CONTENT_KEY = 'content'
MIME_TYPE_KEY = 'mime-type'
MIME_TYPE_TEXT = 'text/plain'
CONTENT_TYPE_SPEECH = 'speech'
CONTENT_TYPE_TRANSCRIPT = 'transcript'


class LabeledSample(Sample):
    """In-memory sample collection sample representing an utterance.
    Derived from util.audio.Sample and used by sample collection readers and writers."""
    def __init__(self, audio_type, raw_data, transcript, audio_format=DEFAULT_FORMAT, sample_id=None):
        """
        Creates an in-memory speech sample together with a transcript of the utterance (label).
        :param audio_type: See util.audio.Sample.__init__ .
        :param raw_data: See util.audio.Sample.__init__ .
        :param transcript: Transcript of the sample's utterance
        :param audio_format: See util.audio.Sample.__init__ .
        :param sample_id: Tracking ID
        """
        super().__init__(audio_type, raw_data, audio_format=audio_format)
        self.sample_id = sample_id
        self.transcript = transcript
        self.meta = None


class DirectSDBWriter:
    """Sample collection writer for creating a Sample DB (SDB) file"""
    def __init__(self, sdb_filename, buffering=BUFFER_SIZE, audio_type=AUDIO_TYPE_OPUS, id_prefix=None):
        self.sdb_filename = sdb_filename
        self.id_prefix = sdb_filename if id_prefix is None else id_prefix
        if audio_type not in SERIALIZABLE_AUDIO_TYPES:
            raise ValueError('Audio type "{}" not supported'.format(audio_type))
        self.audio_type = audio_type
        self.sdb_file = open(sdb_filename, 'wb', buffering=buffering)
        self.offsets = []
        self.num_samples = 0

        self.sdb_file.write(MAGIC)

        meta_data = {
            SCHEMA_KEY: [
                {CONTENT_KEY: CONTENT_TYPE_SPEECH, MIME_TYPE_KEY: audio_type},
                {CONTENT_KEY: CONTENT_TYPE_TRANSCRIPT, MIME_TYPE_KEY: MIME_TYPE_TEXT}
            ]
        }
        meta_data = json.dumps(meta_data).encode()
        self.write_big_int(len(meta_data))
        self.sdb_file.write(meta_data)

        self.offset_samples = self.sdb_file.tell()
        self.sdb_file.seek(2 * BIGINT_SIZE, 1)

    def write_int(self, n):
        return self.sdb_file.write(n.to_bytes(INT_SIZE, BIG_ENDIAN))

    def write_big_int(self, n):
        return self.sdb_file.write(n.to_bytes(BIGINT_SIZE, BIG_ENDIAN))

    def __enter__(self):
        return self

    def add(self, sample):
        def to_bytes(n):
            return n.to_bytes(INT_SIZE, BIG_ENDIAN)
        sample.change_audio_type(self.audio_type)
        opus = sample.audio.getbuffer()
        opus_len = to_bytes(len(opus))
        transcript = sample.transcript.encode()
        transcript_len = to_bytes(len(transcript))
        entry_len = to_bytes(len(opus_len) + len(opus) + len(transcript_len) + len(transcript))
        buffer = b''.join([entry_len, opus_len, opus, transcript_len, transcript])
        self.offsets.append(self.sdb_file.tell())
        self.sdb_file.write(buffer)
        sample.sample_id = '{}:{}'.format(self.id_prefix, self.num_samples)
        self.num_samples += 1
        return sample.sample_id

    def close(self):
        if self.sdb_file is None:
            return
        offset_index = self.sdb_file.tell()
        self.sdb_file.seek(self.offset_samples)
        self.write_big_int(offset_index - self.offset_samples - BIGINT_SIZE)
        self.write_big_int(self.num_samples)

        self.sdb_file.seek(offset_index + BIGINT_SIZE)
        self.write_big_int(self.num_samples)
        for index, offset in enumerate(self.offsets):
            self.write_big_int(offset)
        offset_end = self.sdb_file.tell()
        self.sdb_file.seek(offset_index)
        self.write_big_int(offset_end - offset_index - BIGINT_SIZE)
        self.sdb_file.close()
        self.sdb_file = None

    def __len__(self):
        return len(self.offsets)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SortingSDBWriter:  # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 sdb_filename,
                 tmp_sdb_filename=None,
                 cache_size=CACHE_SIZE,
                 buffering=BUFFER_SIZE,
                 audio_type=AUDIO_TYPE_OPUS,
                 buffered_samples=None,
                 id_prefix=None):
        self.sdb_filename = sdb_filename
        self.id_prefix = sdb_filename if id_prefix is None else id_prefix
        self.buffering = buffering
        self.tmp_sdb_filename = (sdb_filename + '.tmp') if tmp_sdb_filename is None else tmp_sdb_filename
        if audio_type not in SERIALIZABLE_AUDIO_TYPES:
            raise ValueError('Audio type "{}" not supported'.format(audio_type))
        self.audio_type = audio_type
        self.buffered_samples = buffered_samples
        self.tmp_sdb = DirectSDBWriter(self.tmp_sdb_filename,
                                       buffering=buffering,
                                       audio_type=audio_type,
                                       id_prefix='#pre-sorted')
        self.cache_size = cache_size
        self.meta_dict = {}
        self.meta_list = []
        self.buckets = []
        self.bucket = []
        self.bucket_offset = 0
        self.bucket_size = 0
        self.overall_size = 0

    def __enter__(self):
        return self

    def finish_bucket(self):
        if len(self.bucket) == 0:
            return
        self.bucket.sort(key=lambda s: s.duration)
        for sample in self.bucket:
            old_id = sample.sample_id
            new_id = self.tmp_sdb.add(sample)
            self.meta_dict[new_id] = self.meta_dict[old_id]
            del self.meta_dict[old_id]
        self.buckets.append((self.bucket_offset, self.bucket_offset + len(self.bucket)))
        self.bucket_offset += len(self.bucket)
        self.bucket = []
        self.overall_size += self.bucket_size
        self.bucket_size = 0

    def add(self, sample):
        if self.bucket_size > self.cache_size:
            self.finish_bucket()
        sample.change_audio_type(self.audio_type)
        sample.sample_id = '#unsorted:{}'.format(len(self.bucket))
        self.meta_dict[sample.sample_id] = sample.meta
        self.bucket.append(sample)
        self.bucket_size += len(sample.audio.getbuffer())
        return sample.sample_id

    def finalize(self):
        if self.tmp_sdb is None:
            return
        self.finish_bucket()
        num_samples = len(self.tmp_sdb)
        self.tmp_sdb.close()
        self.tmp_sdb = None
        if self.buffered_samples is None:
            avg_sample_size = self.overall_size / max(1, num_samples)
            max_cached_samples = self.cache_size / max(1, avg_sample_size)
            buffer_size = max(1, int(max_cached_samples / max(1, len(self.buckets))))
        else:
            buffer_size = self.buffered_samples
        sdb_reader = SDB(self.tmp_sdb_filename, buffering=self.buffering, id_prefix='#pre-sorted')

        def buffered_view(bucket):
            start, end = bucket
            buffer = []
            current_offset = start
            while current_offset < end:
                while len(buffer) < buffer_size and current_offset < end:
                    buffer.insert(0, sdb_reader[current_offset])
                    current_offset += 1
                while len(buffer) > 0:
                    yield buffer.pop(-1)

        bucket_views = list(map(buffered_view, self.buckets))
        interleaved = heapq.merge(*bucket_views, key=lambda s: s.duration)
        with DirectSDBWriter(self.sdb_filename,
                             buffering=self.buffering,
                             audio_type=self.audio_type,
                             id_prefix=self.id_prefix) as sdb_writer:
            for index, sample in enumerate(interleaved):
                old_id = sample.sample_id
                sdb_writer.add(sample)
                self.meta_list.append(self.meta_dict[old_id])
                del self.meta_dict[old_id]
                yield index / num_samples
        sdb_reader.close()
        os.unlink(self.tmp_sdb_filename)

    def close(self):
        for _ in self.finalize():
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SDB:  # pylint: disable=too-many-instance-attributes
    """Sample collection reader for reading a Sample DB (SDB) file"""
    def __init__(self, sdb_filename, buffering=BUFFER_SIZE, id_prefix=None):
        self.sdb_filename = sdb_filename
        self.id_prefix = sdb_filename if id_prefix is None else id_prefix
        self.sdb_file = open(sdb_filename, 'rb', buffering=buffering)
        self.offsets = []
        if self.sdb_file.read(len(MAGIC)) != MAGIC:
            raise RuntimeError('No Sample Database')
        meta_chunk_len = self.read_big_int()
        self.meta = json.loads(self.sdb_file.read(meta_chunk_len).decode())
        if SCHEMA_KEY not in self.meta:
            raise RuntimeError('Missing schema')
        self.schema = self.meta[SCHEMA_KEY]

        speech_columns = self.find_columns(content=CONTENT_TYPE_SPEECH, mime_type=SERIALIZABLE_AUDIO_TYPES)
        if not speech_columns:
            raise RuntimeError('No speech data (missing in schema)')
        self.speech_index = speech_columns[0]
        self.audio_type = self.schema[self.speech_index][MIME_TYPE_KEY]

        transcript_columns = self.find_columns(content=CONTENT_TYPE_TRANSCRIPT, mime_type=MIME_TYPE_TEXT)
        if not transcript_columns:
            raise RuntimeError('No transcript data (missing in schema)')
        self.transcript_index = transcript_columns[0]

        sample_chunk_len = self.read_big_int()
        self.sdb_file.seek(sample_chunk_len + BIGINT_SIZE, 1)
        num_samples = self.read_big_int()
        for _ in range(num_samples):
            self.offsets.append(self.read_big_int())

    def read_int(self):
        return int.from_bytes(self.sdb_file.read(INT_SIZE), BIG_ENDIAN)

    def read_big_int(self):
        return int.from_bytes(self.sdb_file.read(BIGINT_SIZE), BIG_ENDIAN)

    def find_columns(self, content=None, mime_type=None):
        criteria = []
        if content is not None:
            criteria.append((CONTENT_KEY, content))
        if mime_type is not None:
            criteria.append((MIME_TYPE_KEY, mime_type))
        if len(criteria) == 0:
            raise ValueError('At least one of "content" or "mime-type" has to be provided')
        matches = []
        for index, column in enumerate(self.schema):
            matched = 0
            for field, value in criteria:
                if column[field] == value or (isinstance(value, list) and column[field] in value):
                    matched += 1
            if matched == len(criteria):
                matches.append(index)
        return matches

    def read_row(self, row_index, *columns):
        columns = list(columns)
        column_data = [None] * len(columns)
        found = 0
        if not 0 <= row_index < len(self.offsets):
            raise ValueError('Wrong sample index: {} - has to be between 0 and {}'
                             .format(row_index, len(self.offsets) - 1))
        self.sdb_file.seek(self.offsets[row_index] + INT_SIZE)
        for index in range(len(self.schema)):
            chunk_len = self.read_int()
            if index in columns:
                column_data[columns.index(index)] = self.sdb_file.read(chunk_len)
                found += 1
                if found == len(columns):
                    return tuple(column_data)
            else:
                self.sdb_file.seek(chunk_len, 1)
        return tuple(column_data)

    def __getitem__(self, i):
        audio_data, transcript = self.read_row(i, self.speech_index, self.transcript_index)
        transcript = transcript.decode()
        sample_id = '{}:{}'.format(self.id_prefix, i)
        return LabeledSample(self.audio_type, audio_data, transcript, sample_id=sample_id)

    def __iter__(self):
        for i in range(len(self.offsets)):
            yield self[i]

    def __len__(self):
        return len(self.offsets)

    def close(self):
        if self.sdb_file is not None:
            self.sdb_file.close()

    def __del__(self):
        self.close()


class CSV:
    """Sample collection reader for reading a DeepSpeech CSV file"""
    def __init__(self, csv_filename):
        self.csv_filename = csv_filename
        self.rows = []
        csv_dir = Path(csv_filename).parent
        with open(csv_filename, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                wav_filename = Path(row['wav_filename'])
                if not wav_filename.is_absolute():
                    wav_filename = csv_dir / wav_filename
                self.rows.append((str(wav_filename), int(row['wav_filesize']), row['transcript']))
        self.rows.sort(key=lambda r: r[1])

    def __getitem__(self, i):
        wav_filename, _, transcript = self.rows[i]
        with open(wav_filename, 'rb') as wav_file:
            return LabeledSample(AUDIO_TYPE_WAV, wav_file.read(), transcript, sample_id=wav_filename)

    def __iter__(self):
        for i in range(len(self.rows)):
            yield self[i]

    def __len__(self):
        return len(self.rows)


def samples_from_file(filename, buffering=BUFFER_SIZE):
    """Retrieves the right sample collection reader from a filename"""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.sdb':
        return SDB(filename, buffering=buffering)
    if ext == '.csv':
        return CSV(filename)
    raise ValueError('Unknown file type: "{}"'.format(ext))


def samples_from_files(filenames, buffering=BUFFER_SIZE):
    """Retrieves a (potentially interleaving) sample collection reader from a list of filenames"""
    if len(filenames) == 0:
        raise ValueError('No files')
    if len(filenames) == 1:
        return samples_from_file(filenames[0], buffering=buffering)
    cols = list(map(partial(samples_from_file, buffering=buffering), filenames))
    return Interleaved(*cols, key=lambda s: s.duration)
