import numpy as np

# Load the individual npz files
train = np.load('train.npz')
test = np.load('test.npz')
val = np.load('val.npz')

# Prepare the combined datasets
combined_data = {
    'kinematics_train': train['kinematics_train'],
    'labels_train': train['pid'],
    # 'event_scalars_train': train['data'],
    # 'jet_train':train['jet'],

    'kinematics_test': test['kinematics_test'],
    'labels_test': test['pid'],
    # 'event_scalars_test': test['data'],
    # 'jet_test':test['jet'],

    'kinematics_val': val['kinematics_val'],
    'labels_val': val['pid'],
    # 'event_scalars_val': val['data'],
    # 'jet_val':val['jet']

}

# Save into a single npz file
np.savez('combinedh1_fourmomenta.npz', **combined_data)