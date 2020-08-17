import os
nrows = 250
ncolumns = 250
channels = 3
batch_size = 4
sound_class = "/m/01b_21"
main_path = f'/Users/kylegood/Desktop/MastersProgram/cough_analyzer/'
neg_path = r'/Users/kylegood/Desktop/MastersProgram/data/cough/neg/'
pos_path = r'/Users/kylegood/Desktop/MastersProgram/data/cough/pos/'
cough_path = f'/Users/kylegood/Desktop/MastersProgram/cough_analyzer/img_data/pos_chunks/'
non_cough_path = f'/Users/kylegood/Desktop/MastersProgram/cough_analyzer/img_data/neg_chunks/'
cwd = os.getcwd()
paths = [neg_path, pos_path]
labels = 'neg_chunks/ pos_chunks/'.split()