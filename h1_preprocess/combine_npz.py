import numpy as np

# Load the individual npz files
train = np.load('train.npz')
test = np.load('test.npz')
val = np.load('val.npz')

# Prepare the combined datasets
combined_data = {
    'kinematics_train': train['data'],
    'labels_train': train['pid'],
    'kinematics_test': test['data'],
    'labels_test': test['pid'],
    'kinematics_val': val['data'],
    'labels_val': val['pid'],
}

# Save into a single npz file
np.savez('combinedh1.npz', **combined_data)