import glob

from dateutil.parser import parse
import webvtt

def parse_seconds(time_string):
  date_time = parse(time_string)
  duration = date_time.hour * 60 * 60 + date_time.minute * 60 + date_time.second + date_time.microsecond / 10**6
  return duration

total_seconds = 0

for vtt_file in glob.glob("output_dir/data_download/*.*en*.vtt"):
  # print(vtt_file)
  if "254116491-by.en-GB.vtt" in vtt_file:
    continue
  try:
    captions = webvtt.read(vtt_file)
  except (webvtt.errors.MalformedCaptionError, webvtt.errors.MalformedFileError):
    print("Bad file:", vtt_file)
  total_seconds += parse_seconds(captions[-1].end)
  
print(total_seconds / 60 / 60)
