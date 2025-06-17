import h5py
import numpy as np

train=h5py.File("train_H1.h5")
test=h5py.File("test_H1.h5")
val=h5py.File("val_H1.h5")

data_train=train["data"][:,:,[3,4,5,6]]
data_test=test["data"][:,:,[3,4,5,6]]
data_val=val["data"][:,:,[3,4,5,6]]

combined_data = {
    'kinematics_train': data_train,
    'labels_train': train['pid'],
    'kinematics_test': data_test,
    'labels_test': test['pid'],
    'kinematics_val': data_val,
    'labels_val': val['pid'],
}

np.savez('combinedh1_partfourvec.npz', **combined_data)



