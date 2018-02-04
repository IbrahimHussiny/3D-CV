The pose_and_K.mat file contains a python dictionary containing
'Baseline', 'K' and 'Pose'.

You can load the dictionary
as follows

calib = io.loadmat('pose_and_K.mat')
K = calib['K']
POSE = calib['Pose']
BASELINE= calib['Baseline']

# See kitty.py for more details

