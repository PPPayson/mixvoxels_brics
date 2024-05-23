import os

input_dir = '/gpfs/data/ssrinath/projects/brics_dyscene/dynamic_1/brics-tools/assets/objects/red_car/dynamic_data/mixvoxels/red_car_6/imgs_path_all'
mp4_list = []
num_files = len(os.listdir(input_dir))
for i in range(num_files):
    if os.path.isfile(os.path.join(input_dir, "cam_{}_comp_video.mp4".format(i))):
        mp4_list.append('cam_{}_comp_video.mp4'.format(i))
with open(os.path.join(input_dir, 'inputs.txt'), 'w') as o:
    for f in mp4_list:
        o.write('file \'' + f + '\'\n')
    o.close()


#ffmpeg -f concat -i inputs.txt -c copy concatenated.mp4

