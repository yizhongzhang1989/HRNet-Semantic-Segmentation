import os

video_dirs = [
    '/mnt/ssd/penwan/dataset/Spatial_Reconstruction/Supermarket-3',
    '/mnt/ssd/penwan/dataset/Spatial_Reconstruction/Supermarket-4']

for video_dir in video_dirs:
  videos = os.listdir(video_dir)
  for video in videos:
    if not video.endswith('.mp4'): continue
    cmds = ['python', 'tools/analysisVideo.py',
          '--input_video', os.path.join(video_dir, video),
          '--output_video', os.path.join(video_dir, video[:-4] + '-output-0.5.avi')]
    cmd = ' '.join(cmds)
    print(cmd)
    os.system(cmd)
