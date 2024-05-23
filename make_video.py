import os
import imageio
input_dir = '/gpfs/data/ssrinath/projects/brics_dyscene/dynamic_1/brics-tools/assets/objects/flip_book/dynamic_data/mixvoxels/flip_book_1/imgs_path_all'
mp4_list = []
num_files = len(os.listdir(input_dir))
frames = []
for i in range(num_files):
    if os.path.isfile(os.path.join(input_dir, "cam_{}_comp_video.mp4".format(i))):
        mp4_list.append(os.path.join(input_dir, 'cam_{}_comp_video.mp4'.format(i)))

test_file = mp4_list[0]
reader = imageio.get_reader(test_file, 'ffmpeg')
count = 0

for i, im in enumerate(reader):
    count=i
    
index = 0
for i, f in enumerate(mp4_list):
    if i > count:
        index = i - count
    else:
        index = i
    reader = imageio.get_reader(f, 'ffmpeg')
    vid = reader.get_data(index)
    frames.append(vid)

writer = imageio.get_writer(os.path.join(input_dir, "concatenated_1_frame.mp4"), fps=30)

for frame in frames:
    writer.append_data(frame)
writer.close()