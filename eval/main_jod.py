import re
import os
import subprocess
import argparse
import cv2 as cv
import numpy as np

def extract_from_result(text: str, prompt: str):
    m = re.search(prompt, text)
    return float(m.group(1))

def to_file(file_name, temp_name, WIDTH, HIEGHT, start_frame=None, end_frame=None, cache_dir=None, wh_bg=False):
    if os.path.isfile(file_name):
        return file_name
    else:
        for img_file in os.listdir(file_name):
            if img_file.endswith('.png'):
                img_index = int(img_file.split('.')[0])
                if img_index >= (start_frame) and img_index < (end_frame):
                    if not wh_bg:
                        spth = os.path.join(file_name, img_file)
                        tpth = os.path.join(cache_dir, '%08d.png' % (img_index - start_frame))
                        cmd = f'cp {spth} {tpth}'
                        os.system(cmd)
                    else:
                        spth = os.path.join(file_name, img_file)
                        tpth = os.path.join(cache_dir, '%08d.png' % (img_index - start_frame))
                        img = cv.cvtColor(cv.imread(spth, cv.IMREAD_UNCHANGED), cv.COLOR_BGR2BGRA)
                        img = img.astype(np.float32)
                        img /= 255.
                        bg_color = 1.0 if wh_bg else 0.0
                        img = img[..., :3]*img[..., 3:4] + (1.-img[..., 3:4])*bg_color
                        img = (img*255).astype(np.uint8)
                        cv.imwrite(tpth, img)
        os.system('ffmpeg -loglevel error -i {}/%08d.png -frames:v 300 -s {}x{} -c:v libx264 -qp 0 {}'
                  .format(cache_dir, WIDTH, HEIGHT, temp_name))
        return temp_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--gt', type=str, default=None)
    parser.add_argument('--downsample', type=int, default=None)
    parser.add_argument('--tmp_dir', type=str, default='/tmp/nerf_metric_temp')
    parser.add_argument('--tmp_gt_dir', type=str, default='/tmp/nerf_metric_temp_gt')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=300)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--wh_bg', action='store_true')
    parser.add_argument('--obj', type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(args.tmp_dir):
        os.mkdir(args.tmp_dir)
    if not os.path.exists(args.tmp_gt_dir):
        os.mkdir(args.tmp_gt_dir)
    args.tmp_dir = os.path.join(args.tmp_dir, args.obj)
    args.tmp_gt_dir = os.path.join(args.tmp_gt_dir, args.obj)
    os.system(f'rm -rf {args.tmp_dir}')
    os.system(f'mkdir {args.tmp_dir}')
    os.system(f'rm -rf {args.tmp_gt_dir}')
    os.system(f'mkdir {args.tmp_gt_dir}')
    
    #if args.downsample == 2:
    #    WIDTH = 1360
    #    HEIGHT = 1024
    #elif args.downsample == 4:
    #    WIDTH = 688
    #    HEIGHT = 512
    WIDTH = int(args.width/args.downsample)
    HEIGHT = int(args.height/args.downsample)

    file1 = to_file(args.output, os.path.join(args.tmp_dir, 'nerf_metric_temp1.mp4'), WIDTH, HEIGHT, start_frame=args.start_frame, end_frame=args.end_frame, cache_dir=args.tmp_dir, wh_bg=args.wh_bg)
    file2 = to_file(args.gt, os.path.join(args.tmp_gt_dir, 'nerf_metric_temp2.mp4'), WIDTH, HEIGHT, start_frame=args.start_frame, end_frame=args.end_frame, cache_dir=args.tmp_gt_dir, wh_bg=args.wh_bg)

    result = subprocess.check_output(['fvvdp', '--test', file1, '--ref', file2, '--gpu', '0', '--display', 'standard_fhd'])
    print(result.decode())
    #result = result.decode().strip()
    #result = float(result.split('=')[1])
    #print(result)
    #os.system('stty echo')
