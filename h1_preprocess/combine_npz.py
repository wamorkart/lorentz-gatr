import numpy as np

# Load the individual npz files
train = np.load('train_100points.npz')
test = np.load('test_100points.npz')
val = np.load('val_100points.npz')

# Prepare the combined datasets
combined_data = {
    'kinematics_train': train['kinematics_train'],
    'labels_train': train['labels_train'],
    'event_scalars_train': train['data'],
    # 'jet_train':train['jet'],

    'kinematics_test': test['kinematics_test'],
    'labels_test': test['labels_test'],
    'event_scalars_test': test['data'],
    # 'jet_test':test['jet'],

    'kinematics_val': val['kinematics_val'],
    'labels_val': val['labels_val'],
    'event_scalars_val': val['data'],
    # 'jet_val':val['jet']

}

# Save into a single npz file
np.savez('combinedh1_withscalars_100particles.npz', **combined_data)