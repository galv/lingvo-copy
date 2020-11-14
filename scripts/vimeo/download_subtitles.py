import youtube_dl
import glob
from tqdm import tqdm

# python -m youtube_dl --verbose --skip-download --all-subs --write-sub -o "cc_by_download_dir/%(id)s-%(license)s.%(ext)s" -i https://vimeo.com/creativecommons/by 2>&1 | tee --append download_log2.txt

input_dir = "output_dir/"
output_dir = "output_dir/data_download"


# def tqdm_hook(d):
#   print("GALVEZ")
#   pbar.update(1)

ydl_opts = dict(skip_download=True,
                writesubtitles=True,
                allsubtitles=True,
                ignoreerrors=True,
                outtmpl=f"{output_dir}/%(id)s-%(license)s.%(ext)s",
                download_archive="{output_dir}/already_downloaded_archive.archive",
                default_search="http://vimeo.com",
                # progress_hook=[tqdm_hook]
                )

video_ids = []
for file_name in glob.glob(f"{input_dir}/*.txt"):
  with open(file_name) as fh:
    lines = fh.readlines()
    for line in lines:
      video_ids.append(line.split(",")[0].replace('/videos/', '/'))

video_ids = sorted(set(video_ids))

print(f"GALVEZ:length={len(video_ids)}")

# video_ids = ["adsf10"] * 10
# pbar = tqdm(total=len(video_ids), miniters=1)

start = video_ids.index("/266776980")

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
  for video_id in tqdm(video_ids[start:]):
    ydl.download([video_id])
# pbar.close()
