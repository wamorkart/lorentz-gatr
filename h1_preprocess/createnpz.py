import numpy as np
import h5py as h5
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def extract_four_vectors_and_scalars(filename):
    """Extract particle 4-vectors (energy, px, py, pz) and event-level scalars from HDF5 file"""
    print(f"Processing {filename}...")
    
    with h5.File(filename, 'r') as f:
        # Print available keys for debugging
        print(f"Available keys in {filename}: {list(f.keys())}")
        
        four_vectors = None
        event_scalars = None
        
        # Extract particle 4-vectors
        if 'reco_particle_features' in f.keys():
            particle_data = f['reco_particle_features'][:]
            print(f"Particle data shape: {particle_data.shape}")
            
            # Based on your preprocessing script, the last 4 features should be [px, py, pz, energy]
            # Extracting the last 4 columns and reordering to [energy, px, py, pz]
            if particle_data.shape[-1] >= 4:
                px = particle_data[:, :, -4]   # px
                py = particle_data[:, :, -3]   # py  
                pz = particle_data[:, :, -2]   # pz
                energy = particle_data[:, :, -1]  # energy
                
                # Stack as 4-vectors: [energy, px, py, pz]
                four_vectors = np.stack([energy, px, py, pz], axis=-1)
                print(f"Four-vector shape: {four_vectors.shape}")
            else:
                print(f"Warning: Not enough features in {filename}")
                return None, None
        else:
            print(f"Warning: 'reco_particle_features' not found in {filename}")
            return None, None
        
        # Extract event-level scalars (first 4 features from reco_event_features)
        if 'reco_event_features' in f.keys():
            event_features = f['reco_event_features'][:]
            print(f"Event features shape: {event_features.shape}")
            
            # Extract first 4 event-level features
            if event_features.shape[1] >= 4:
                event_scalars = event_features[:, :4]  # Take first 4 features
                print(f"Event scalars shape: {event_scalars.shape}")
                print(f"Using first 4 event features as scalars")
            else:
                print(f"Warning: Less than 4 event features available in {filename}")
                # Pad with zeros if less than 4 features
                n_features = event_features.shape[1]
                padded_features = np.zeros((event_features.shape[0], 4))
                padded_features[:, :n_features] = event_features
                event_scalars = padded_features
                print(f"Padded event scalars shape: {event_scalars.shape}")
            
        else:
            print(f"Warning: 'reco_event_features' not found in {filename}")
            if four_vectors is not None:
                print("Computing event pseudorapidity from jet kinematics as fallback...")
                event_q2 = compute_event_q2(four_vectors)
                event_scalars = event_q2[:, np.newaxis]
                print(f"Computed event scalars shape: {event_scalars.shape}")
            else:
                print("Creating dummy event scalars (zeros)")
                event_scalars = np.zeros((0, 1))  # Will be handled in main()
        
        return four_vectors, event_scalars




def main():
    parser = argparse.ArgumentParser(description='Combine HDF5 files and create train/test/val splits')
    parser.add_argument('--file_rp', default='Rapgap.h5', help='First HDF5 file')
    parser.add_argument('--file_dj', default='Djangoh.h5', help='Second HDF5 file')
    parser.add_argument('--train_size', type=float, default=0.6, help='Training set fraction')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set fraction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate split sizes
    if abs(args.train_size + args.test_size + args.val_size - 1.0) > 1e-6:
        print("Warning: Split sizes don't sum to 1.0, normalizing...")
        total = args.train_size + args.test_size + args.val_size
        args.train_size /= total
        args.test_size /= total
        args.val_size /= total
    
    # Extract 4-vectors and scalars from both files
    four_vectors_rp, event_scalars_rp = extract_four_vectors_and_scalars(args.file_rp)
    four_vectors_dj, event_scalars_dj = extract_four_vectors_and_scalars(args.file_dj)
    
    if four_vectors_rp is None or four_vectors_dj is None:
        print("Error: Could not extract data from one or both files")
        return
    
    if event_scalars_rp is None or event_scalars_dj is None:
        print("Error: Could not extract event scalars from one or both files")
        return
    
    print(f"four_vectors_rp shape: {four_vectors_rp.shape}, four_vectors_dj shape: {four_vectors_dj.shape}")
    print(f"event_scalars_rp shape: {event_scalars_rp.shape}, event_scalars_dj shape: {event_scalars_dj.shape}")
    
    # Create labels (1 for Rapgap, 0 for Djangoh)
    pid_rp = np.ones(four_vectors_rp.shape[0])
    pid_dj = np.zeros(four_vectors_dj.shape[0])

    print(f"pid_rp shape: {pid_rp.shape}, pid_dj shape: {pid_dj.shape}")
    
    # Combine the data
    print("Combining datasets...")
    four_vectors_combined = np.concatenate([four_vectors_rp, four_vectors_dj], axis=0)
    event_scalars_combined = np.concatenate([event_scalars_rp, event_scalars_dj], axis=0)
    pid_combined = np.concatenate([pid_rp, pid_dj])
    
    # Shuffle all data together
    indices = np.arange(len(four_vectors_combined))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    
    four_vectors_combined = four_vectors_combined[indices]
    event_scalars_combined = event_scalars_combined[indices]
    pid_combined = pid_combined[indices]
    
    print(f"four_vectors_combined shape: {four_vectors_combined.shape}")
    print(f"event_scalars_combined shape: {event_scalars_combined.shape}")  
    print(f"pid_combined shape: {pid_combined.shape}")

    # Create splits
    total = four_vectors_combined.shape[0]
    ntrain = int(args.train_size * total)
    nval = int(args.val_size * total)
    ntest = total - ntrain - nval

    print(f"Split sizes: Train={ntrain}, Val={nval}, Test={ntest}")

    # Split the data
    train_data = four_vectors_combined[:ntrain]
    train_scalars = event_scalars_combined[:ntrain]
    train_pid = pid_combined[:ntrain]

    val_data = four_vectors_combined[ntrain:ntrain+nval]
    val_scalars = event_scalars_combined[ntrain:ntrain+nval]
    val_pid = pid_combined[ntrain:ntrain+nval]

    test_data = four_vectors_combined[ntrain+nval:]
    test_scalars = event_scalars_combined[ntrain+nval:]
    test_pid = pid_combined[ntrain+nval:]

    # Save individual splits
    print("Saving train.npz...")
    np.savez_compressed('train.npz', 
                       kinematics_train=train_data, 
                       labels_train=train_pid,
                       event_scalars_train=train_scalars)
    
    print("Saving test.npz...")
    np.savez_compressed('test.npz', 
                       kinematics_test=test_data, 
                       labels_test=test_pid,
                       event_scalars_test=test_scalars)
    
    print("Saving val.npz...")
    np.savez_compressed('val.npz', 
                       kinematics_val=val_data, 
                       labels_val=val_pid,
                       event_scalars_val=val_scalars)
    
    # Save combined file
    print("Combining the npz files into combinedh1_fourscalars.npz...")
    np.savez_compressed('combinedh1_fourscalars.npz', 
                       kinematics_train=train_data, 
                       labels_train=train_pid,
                       event_scalars_train=train_scalars,
                       kinematics_test=test_data, 
                       labels_test=test_pid,
                       event_scalars_test=test_scalars,
                       kinematics_val=val_data, 
                       labels_val=val_pid,
                       event_scalars_val=val_scalars)

    print("Done! Created train.npz, test.npz, val.npz, and combinedh1.npz files.")
    print(f"Final splits: Train={len(train_data)}, Test={len(test_data)}, Val={len(val_data)}")
    print(f"Event scalars have {event_scalars_combined.shape[1]} features")

if __name__ == "__main__":
    main()
# import numpy as np
# import h5py as h5
# import argparse
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle


# def extract_four_vectors(filename):
#     """Extract particle 4-vectors (energy, px, py, pz) from HDF5 file"""
#     print(f"Processing {filename}...")
    
#     with h5.File(filename, 'r') as f:
#         # Print available keys for debugging
#         print(f"Available keys in {filename}: {list(f.keys())}")
        
#         # Assuming the particle features are in 'reco_particle_features' or similar
#         # Based on your preprocessing script, the features should be in this format
#         if 'reco_particle_features' in f.keys():
#             particle_data = f['reco_particle_features'][:]
#             print(f"Particle data shape: {particle_data.shape}")
            
#             # Based on your preprocessing script, the last 4 features should be [px, py, pz, energy]
#             # Extracting the last 4 columns and reordering to [energy, px, py, pz]
#             if particle_data.shape[-1] >= 4:
#                 px = particle_data[:, :, -4]   # px
#                 py = particle_data[:, :, -3]   # py  
#                 pz = particle_data[:, :, -2]   # pz
#                 energy = particle_data[:, :, -1]  # energy
                
#                 # Stack as 4-vectors: [energy, px, py, pz]
#                 four_vectors = np.stack([energy, px, py, pz], axis=-1)
#                 print(f"Four-vector shape: {four_vectors.shape}")
#                 return four_vectors
#             else:
#                 print(f"Warning: Not enough features in {filename}")
#                 return None
#         else:
#             print(f"Warning: 'reco_particle_features' not found in {filename}")
#             return None

# def main():
#     parser = argparse.ArgumentParser(description='Combine HDF5 files and create train/test/val splits')
#     parser.add_argument('--file_rp', default='Rapgap.h5', help='First HDF5 file')
#     parser.add_argument('--file_dj', default='Djangoh.h5', help='Second HDF5 file')
#     parser.add_argument('--train_size', type=float, default=0.7, help='Training set fraction')
#     parser.add_argument('--test_size', type=float, default=0.2, help='Test set fraction')
#     parser.add_argument('--val_size', type=float, default=0.1, help='Validation set fraction')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
#     args = parser.parse_args()
    
#     # Validate split sizes
#     if abs(args.train_size + args.test_size + args.val_size - 1.0) > 1e-6:
#         print("Warning: Split sizes don't sum to 1.0, normalizing...")
#         total = args.train_size + args.test_size + args.val_size
#         args.train_size /= total
#         args.test_size /= total
#         args.val_size /= total
    
#     # Extract 4-vectors from both files
#     four_vectors_rp = extract_four_vectors(args.file_rp)
#     four_vectors_dj = extract_four_vectors(args.file_dj)
    
#     if four_vectors_rp is None or four_vectors_dj is None:
#         print("Error: Could not extract data from one or both files")
#         return
    
#     print(f"four_vectors_rp shape: {four_vectors_rp.shape},four_vectors_dj shape: {four_vectors_dj.shape} ")
#     pid_rp = np.ones(four_vectors_rp.shape[0])
#     pid_dj = np.zeros(four_vectors_dj.shape[0])

#     print(f"pid_rp shape; {pid_rp.shape}, pid_dj shape: {pid_dj.shape}")
#     # Combine the data
#     print("Combining datasets...")
#     # combined_data = np.concatenate([four_vectors_1, four_vectors_2], axis=0)

#     four_vectors_combined, pid_combined = shuffle(np.concatenate([four_vectors_rp, four_vectors_dj], axis=0),
#                                                   np.concatenate([pid_rp, pid_dj]))
    
#     print(f"four_vectors_combined --> {four_vectors_combined.shape}")
#     print(f"pid_combined --> {pid_combined.shape}")

#     total = four_vectors_combined.shape[0]
#     ntrain = int(0.6 * total)
#     nval = int(0.2 * total)
#     ntest = total - ntrain - nval  # remaining 20%

#     assert ntrain > ntest, "Training set should be larger than test set"

#     train_data = four_vectors_combined[:ntrain]
#     train_pid = pid_combined[:ntrain]

#     val_data = four_vectors_combined[ntrain:ntrain+nval]
#     val_pid = pid_combined[ntrain:ntrain+nval]

#     test_data = four_vectors_combined[ntrain+nval:]
#     test_pid = pid_combined[ntrain+nval:]

    
#     # # Save the splits
#     print("Saving train.npz...")
#     np.savez_compressed('train.npz', kinematics_train=train_data, labels_train=train_pid)
    
#     print("Saving test.npz...")
#     np.savez_compressed('test.npz', kinematics_test=test_data, labels_test=test_pid)
    
#     print("Saving val.npz...")
#     np.savez_compressed('val.npz', kinematics_val=val_data, labels_val=val_pid)
    
#     print("Combining the npz files into combined.npz...")
#     np.savez_compressed('combinedh1.npz', kinematics_train=train_data, labels_train=train_pid,kinematics_test=test_data, labels_test=test_pid,kinematics_val=val_data, labels_val=val_pid )

#     print("Done! Created train.npz, test.npz, and val.npz files.")

#     # print(f"Final splits: Train={len(train_data)}, Test={len(test_data)}, Val={len(val_data)}")

# if __name__ == "__main__":
#     main()