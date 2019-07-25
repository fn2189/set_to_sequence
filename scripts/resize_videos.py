"""
python scripts/resize_videos.py --input-folder ../s3-drive/RLY/RLYMedia --output-folder ../s3-drive/RLY/RLYMedia_resized


nb_frames: n = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames', '-of',
'default=nokey=1:noprint_wrappers=1', a[0]])
"""

import subprocess
from glob import glob
import argparse
import os 

def main():
    parser = argparse.ArgumentParser(description='resive video to desired format with ffmpeg')
    parser.add_argument('--input-folder', type=str, default='../s3-drive/RLY/RLYMedia',
                        help='folder from which we want to resize the videos')
    parser.add_argument('--output-folder', type=str, default='../s3-drive/RLY/RLYMedia_resized',
                        help='folder in which to store the resized_videos')
    
    args = parser.parse_args()
    
    videos = glob(args.input_folder +'/*')
    
     
    
    for _, video in enumerate(videos):
        basename = os.path.basename(video)
        output = os.path.join(args.output_folder, basename)
        subprocess.run(['ffmpeg', '-i', video, '-vf', 'scale=256:256', output])
        
        if (_+1) % 100 ==0:
            print(f'Done resizing video {_}')
        
if __name__ == '__main__':
    main()
  