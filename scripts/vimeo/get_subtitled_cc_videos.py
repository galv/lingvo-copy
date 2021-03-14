from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
import os
import sys
from threading import Lock, Thread
import time


# from absl import flags
import requests
import tqdm

# flags.define_string("")

# FLAGS = flags.FLAGS

API_URL = "https://api.vimeo.com"
ENDPOINT = f"{API_URL}/videos"
HEADERS = {'Authorization': f'bearer {os.environ["VIMEO_ACCESS_TOKEN"]}'}

NUM_THREADS = 16

class VimeoPageBeyond100Error(ValueError):
  pass

# {'error': 'An error has occurred while searching for videos. Please try again.', 'link': None, 'developer_message': 'Search is currently unavailable. Please retry your request.', 'error_code': 7300}
def robust_request(request_url):
  while True:
    next_response = requests.get(request_url, headers=HEADERS)
    next_response_json = next_response.json()
    if next_response_json.get("error_code") in (9000, 7300):
      # We are being rate limited. The quota will reset after 60 seconds
      # 7300 means that the search service is unavailable at the moment, meanwhile.
      time.sleep(5)
      continue
    elif next_response_json.get("error_code") in (2204, 2969):
      raise VimeoPageBeyond100Error()
    else:
      return next_response

class VideoMetadataIterator:
  def __init__(self, license, query):
    request_url = f"{ENDPOINT}?filter={license}&per_page=100&query={query}&fields=uri,duration"
    self.response = robust_request(request_url)
    self.response_json = self.response.json()
    self.length = self.response_json["total"]
    if self.length > 100 * 100:
      print(f"WARNING: {query}_{license} has {self.length} - {100 * 100} = {self.length - 100 * 100} more videos than can be requested")
      self.length = 100 * 100
    self.index = 0
    self.license = license
    self.query = query

  def __len__(self):
    return self.length

  def __iter__(self):
    return self

  def __next__(self):
    try:
      datum = self.response_json["data"][self.index]
      self.index += 1
      return datum
    except IndexError:
      self.index = 0
      if self.response_json["paging"]["next"]:
        try:
          self.response = robust_request(API_URL + self.response_json["paging"]["next"])
          self.response_json = self.response.json()
        except VimeoPageBeyond100Error:
          raise StopIteration
      else:
        raise StopIteration
      return next(self)
    except Exception as exc:
      print(f"GALVEZ:problem!{self.license} {self.query} {exc}")
      print("Offending response:", self.response_json)
      print("Offending URL:", self.response.request.url)
      raise

def run_wrapper(query_license):
  query, license = query_license
  run(query, license)

def run(query, license):
  for datum in VideoMetadataIterator(license, query):
    with LOCK:
      item_to_duration[datum["uri"]] = datum["duration"]

  # with open(f"output_dir/{query}_{license}.txt", "w") as fh:
  #   for datum in tqdm.tqdm(VideoMetadataIterator(license, query), desc=f"{query}_{license}"):
  #     uri = datum["uri"]
  #     duration = datum["duration"]
  #     fh.write(f"{uri},{duration}\n")

# https://vimeo.com/creativecommons/cc0/: 10 * 11_143 videos
# https://vimeo.com/creativecommons/by/: 10 * 161_770 videos

# https://docs.python.org/dev/library/functions.html#chr
LOCK = Lock()
item_to_duration = {}

def check_size():
  while True:
    with LOCK:
      total_count = len(item_to_duration)
    print("Total count:", total_count)
    print("CC-BY count:", 10 * 161_770)
    # This may be 1941168 instead...
    # 1617700
    # 1941168
    with open('cc_by_videos.json', 'w') as fh, LOCK:
      json.dump(item_to_duration, fh)
    time.sleep(10)

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
  # Note: The query string is case-insensitive, so no need for capital letters
  # pairs = list(product("abcdefghijklmnopqrstuvwxyz",
  #                      ["CC-BY", "CC0"],))
  pairs = list(product(map(lambda i: chr(i), range(0, 1_114_111 + 1)), ["CC-BY"]))
  thread = Thread(target=check_size)
  thread.start()
  list(tqdm.tqdm(executor.map(run_wrapper, pairs), total=len(pairs), desc="Total"))
thread.join()

# threads = []
# for query, license in product("abcdefghijklmnopqrstuvwxyz",
#                               ["CC-BY", "CC0"],):
#   thread = Thread(target=run, args=(query, license))
#   thread.start()
#   threads.append(thread)
# for thread in threads:
#   thread.join()
