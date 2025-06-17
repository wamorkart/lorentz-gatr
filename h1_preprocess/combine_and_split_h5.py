import numpy as np
import h5py
import argparse
import os
import gc  # Garbage collector
from sklearn.utils import shuffle


dj=h5py.File("Djangoh_Eminus06_prep.h5")
rp=h5py.File("Rapgap_Eminus06_prep.h5")


dj_particle_fourvec=np.array(dj["reco_particle_epxpypz"])
rp_particle_fourvec=np.array(rp["reco_particle_epxpypz"])

dj_particle=np.array(dj["reco_particle_features"])
rp_particle=np.array(rp["reco_particle_features"])

dj_event=np.array(dj["reco_event_features"])
rp_event=np.array(rp["reco_event_features"])



print(f"djangoh --> {dj_particle.shape}, rapgap--> {rp_particle.shape}")
# particles_fourvec, particles, event, pid = shuffle(np.concatenate([dj_particle_fourvec, rp_particle_fourvec], axis=0),
#                                                    np.concatenate([dj_particle, rp_particle], axis=0),
#                                                    np.concatenate([dj_event, rp_event], axis=0),
#                                                    np.concatenate([np.zeros(dj_particle_fourvec.shape[0]),np.ones(rp_particle_fourvec.shape[0])] ))

particles_fourvec, pid, qsquare = shuffle(np.concatenate([dj_particle_fourvec, rp_particle_fourvec], axis=0),
                                                   np.concatenate([np.zeros(dj_particle_fourvec.shape[0]),np.ones(rp_particle_fourvec.shape[0])] ),
                                                   np.concatenate([dj_event[:,0], rp_event[:,0]], axis=0))
print(f"particles_fourved --> {particles_fourvec.shape}")
# print(f"particles --> {particles.shape}")
# print(f"event --> {event.shape}")
print(f"pid --> {pid.shape}")

total = particles_fourvec.shape[0]
ntrain = int(0.6 * total)
nval = int(0.2 * total)
ntest = total - ntrain - nval  # remaining 20%

assert ntrain > ntest, "Training set should be larger than test set"

# Split datasets
train_data = particles_fourvec[:ntrain]
train_pid = pid[:ntrain]
train_qsquare = qsquare[:ntrain]

val_data = particles_fourvec[ntrain:ntrain+nval]
val_pid = pid[ntrain:ntrain+nval]
val_qsquare = qsquare[ntrain:ntrain+nval]

test_data = particles_fourvec[ntrain+nval:]
test_pid = pid[ntrain+nval:]
test_qsquare = qsquare[ntrain+nval :]

# Save to NPZ
np.savez_compressed("train.npz", data=train_data, pid=train_pid, qsquare=train_qsquare)
np.savez_compressed("val.npz", data=val_data, pid=val_pid, qsquare=val_qsquare)
np.savez_compressed("test.npz", data=test_data, pid=test_pid, qsquare=test_qsquare)

# # Split and save
# with h5py.File('train.h5', "w") as fh5:
#     fh5.create_dataset('data', data=particles_fourvec[:ntrain])
#     fh5.create_dataset('pid', data=pid[:ntrain])

# with h5py.File('val.h5', "w") as fh5:
#     fh5.create_dataset('data', data=particles_fourvec[ntrain:ntrain+nval])
#     fh5.create_dataset('pid', data=pid[ntrain:ntrain+nval])

# with h5py.File('test.h5', "w") as fh5:
#     fh5.create_dataset('data', data=particles_fourvec[ntrain+nval:])
#     fh5.create_dataset('pid', data=pid[ntrain+nval:])

# def combine_and_split_h5(rapgap_file, djangoh_file, output_dir, 
#                         train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
#                         shuffle=True, seed=42, keep_combined=False,
#                         chunk_size=1000):
#     """
#     Combine two H5 files, optionally shuffle them, and split into train, validation, and test datasets.
#     Uses chunking to reduce memory usage.
    
#     Args:
#         rapgap_file: Path to the rapgap.h5 file
#         djangoh_file: Path to the djangoh.h5 file
#         output_dir: Directory to save the output H5 files
#         train_ratio: Fraction of data for training (default: 0.7)
#         val_ratio: Fraction of data for validation (default: 0.15)
#         test_ratio: Fraction of data for testing (default: 0.15)
#         shuffle: Whether to shuffle entries before splitting (default: True)
#         seed: Random seed for reproducibility (default: 42)
#         keep_combined: Whether to keep the combined file (default: False)
#         chunk_size: Number of entries to process at once (default: 1000)
#     """
#     # Set random seed for reproducibility
#     np.random.seed(seed)
    
#     # Verify ratios sum to 1
#     total_ratio = train_ratio + val_ratio + test_ratio
#     if not np.isclose(total_ratio, 1.0):
#         raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
#     # Create output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"Created output directory: {output_dir}")
    
#     print(f"Processing {rapgap_file} and {djangoh_file}")
    
#     # Get dataset information first
#     with h5py.File(rapgap_file, 'r') as rapgap_h5, h5py.File(djangoh_file, 'r') as djangoh_h5:
#         # Get the list of datasets from rapgap file
#         dataset_names = list(rapgap_h5.keys())
        
#         # Check if the same datasets exist in djangoh file
#         for name in dataset_names:
#             if name not in djangoh_h5:
#                 raise ValueError(f"Dataset '{name}' found in rapgap but not in djangoh")
        
#         # Get sizes and shapes
#         rapgap_size = rapgap_h5[dataset_names[0]].shape[0]
#         djangoh_size = djangoh_h5[dataset_names[0]].shape[0]
#         total_size = rapgap_size + djangoh_size
        
#         # Get dataset shapes
#         dataset_shapes = {}
#         dataset_dtypes = {}
#         for name in dataset_names:
#             # Get shape of the dataset excluding the first dimension (which varies)
#             if len(rapgap_h5[name].shape) > 1:
#                 dataset_shapes[name] = rapgap_h5[name].shape[1:]
#             else:
#                 dataset_shapes[name] = ()
#             dataset_dtypes[name] = rapgap_h5[name].dtype
            
#             # Check if shapes are compatible for concatenation
#             if dataset_shapes[name] != djangoh_h5[name].shape[1:]:
#                 raise ValueError(f"Dataset '{name}' has incompatible shapes: {rapgap_h5[name].shape} vs {djangoh_h5[name].shape}")
    
#     print(f"Found {rapgap_size} entries in rapgap and {djangoh_size} entries in djangoh")
#     print(f"Total dataset size: {total_size}")
    
#     # Create pid array
#     pid = np.concatenate([
#         np.ones(rapgap_size, dtype=np.int32),   # 1 for rapgap
#         np.zeros(djangoh_size, dtype=np.int32)  # 0 for djangoh
#     ])
    
#     # Create shuffle indices if needed
#     if shuffle:
#         print("Creating shuffle indices")
#         shuffle_indices = np.random.permutation(total_size)
#         # Shuffle pid
#         pid = pid[shuffle_indices]
    
#     # Calculate split sizes
#     train_size = int(total_size * train_ratio)
#     val_size = int(total_size * val_ratio)
#     test_size = total_size - train_size - val_size  # Ensure we use all data
    
#     print(f"Split sizes - Train: {train_size}, Validation: {val_size}, Test: {test_size}")
    
#     # Create output files
#     train_file = os.path.join(output_dir, "train.h5")
#     val_file = os.path.join(output_dir, "val.h5")
#     test_file = os.path.join(output_dir, "test.h5")
#     combined_file = os.path.join(output_dir, "combined.h5") if keep_combined else None
    
#     # Create output files with empty datasets
#     output_files = []
#     for split_name, split_size, output_file in [
#         ("Train", train_size, train_file),
#         ("Validation", val_size, val_file),
#         ("Test", test_size, test_file)
#     ]:
#         print(f"Creating {split_name} dataset at {output_file}")
#         with h5py.File(output_file, 'w') as output_h5:
#             # Create empty datasets
#             for name in dataset_names:
#                 shape = (split_size,) + dataset_shapes[name]
#                 output_h5.create_dataset(name, shape=shape, dtype=dataset_dtypes[name])
            
#             # Add pid dataset
#             output_h5.create_dataset('pid', shape=(split_size,), dtype=np.int32)
        
#         output_files.append(output_file)
    
#     # Create combined file if requested
#     if keep_combined:
#         print(f"Creating combined dataset at {combined_file}")
#         with h5py.File(combined_file, 'w') as combined_h5:
#             # Create empty datasets
#             for name in dataset_names:
#                 shape = (total_size,) + dataset_shapes[name]
#                 combined_h5.create_dataset(name, shape=shape, dtype=dataset_dtypes[name])
            
#             # Add pid dataset
#             combined_h5.create_dataset('pid', shape=(total_size,), dtype=np.int32)
    
#     # Process datasets in chunks
#     for name in dataset_names:
#         print(f"Processing dataset: {name}")
#         # Process in chunks to reduce memory usage
#         for chunk_start in range(0, rapgap_size, chunk_size):
#             chunk_end = min(chunk_start + chunk_size, rapgap_size)
#             chunk_size_current = chunk_end - chunk_start
            
#             print(f"  Processing rapgap chunk {chunk_start}:{chunk_end}")
            
#             # Read chunk from rapgap
#             with h5py.File(rapgap_file, 'r') as rapgap_h5:
#                 rapgap_chunk = rapgap_h5[name][chunk_start:chunk_end]
            
#             # Process chunk
#             process_chunk(name, rapgap_chunk, True, chunk_start, chunk_end, 
#                          rapgap_size, total_size, shuffle_indices,
#                          train_size, val_size, output_files, combined_file)
            
#             # Clear memory
#             del rapgap_chunk
#             gc.collect()
        
#         for chunk_start in range(0, djangoh_size, chunk_size):
#             chunk_end = min(chunk_start + chunk_size, djangoh_size)
#             chunk_size_current = chunk_end - chunk_start
            
#             print(f"  Processing djangoh chunk {chunk_start}:{chunk_end}")
            
#             # Read chunk from djangoh
#             with h5py.File(djangoh_file, 'r') as djangoh_h5:
#                 djangoh_chunk = djangoh_h5[name][chunk_start:chunk_end]
            
#             # Process chunk
#             process_chunk(name, djangoh_chunk, False, chunk_start, chunk_end, 
#                          djangoh_size, total_size, shuffle_indices,
#                          train_size, val_size, output_files, combined_file,
#                          offset=rapgap_size)
            
#             # Clear memory
#             del djangoh_chunk
#             gc.collect()
    
#     # Process pid array
#     print("Processing pid array")
#     for split_name, split_start, split_end, output_file in [
#         ("Train", 0, train_size, train_file),
#         ("Validation", train_size, train_size + val_size, val_file),
#         ("Test", train_size + val_size, total_size, test_file)
#     ]:
#         split_size = split_end - split_start
#         with h5py.File(output_file, 'r+') as output_h5:
#             output_h5['pid'][:] = pid[split_start:split_end]
    
#     # Save pid to combined file if requested
#     if keep_combined:
#         with h5py.File(combined_file, 'r+') as combined_h5:
#             combined_h5['pid'][:] = pid

# def process_chunk(name, chunk_data, is_rapgap, chunk_start, chunk_end, 
#                  source_size, total_size, shuffle_indices,
#                  train_size, val_size, output_files, combined_file, offset=0):
#     """
#     Process a chunk of data and write to output files.
    
#     Args:
#         name: Name of the dataset
#         chunk_data: Chunk of data to process
#         is_rapgap: Whether the chunk is from rapgap (True) or djangoh (False)
#         chunk_start: Start index in the source file
#         chunk_end: End index in the source file
#         source_size: Size of the source dataset
#         total_size: Total size of the combined dataset
#         shuffle_indices: Indices for shuffling (or None if not shuffling)
#         train_size: Size of the training dataset
#         val_size: Size of the validation dataset
#         output_files: List of output files [train_file, val_file, test_file]
#         combined_file: Path to combined file (or None if not keeping combined)
#         offset: Offset to apply to source indices (for djangoh data)
#     """
#     # Convert chunk indices to combined indices
#     source_indices = np.arange(chunk_start, chunk_end) + offset
    
#     # Get chunk size
#     chunk_size = chunk_end - chunk_start
    
#     # Get shuffled indices if needed
#     if shuffle_indices is not None:
#         # Find where these source indices ended up after shuffling
#         mask = np.isin(shuffle_indices, source_indices)
#         target_indices = np.where(mask)[0]
        
#         # Get the mapping from source to target
#         source_to_target = {s: t for s, t in zip(source_indices, target_indices)}
        
#         # Reorder chunk data according to shuffle
#         chunk_order = np.argsort([source_to_target[s] for s in source_indices])
#         chunk_data = chunk_data[chunk_order]
        
#         # Use target indices for further processing
#         indices = target_indices
#     else:
#         indices = source_indices
    
#     # Write to combined file if keeping
#     if combined_file is not None:
#         with h5py.File(combined_file, 'r+') as combined_h5:
#             combined_h5[name][indices] = chunk_data
    
#     # Split indices into train/val/test
#     train_indices = indices[indices < train_size]
#     val_indices = indices[(indices >= train_size) & (indices < train_size + val_size)]
#     test_indices = indices[indices >= train_size + val_size]
    
#     # Write to output files
#     for split_indices, split_start, split_file in [
#         (train_indices, 0, output_files[0]),
#         (val_indices, train_size, output_files[1]),
#         (test_indices, train_size + val_size, output_files[2])
#     ]:
#         if len(split_indices) > 0:
#             with h5py.File(split_file, 'r+') as output_h5:
#                 # Adjust indices relative to each split
#                 relative_indices = split_indices - split_start
                
#                 # Get the corresponding chunk data
#                 if shuffle_indices is not None:
#                     # Find which elements in the shuffled chunk correspond to this split
#                     mask = np.isin(target_indices, split_indices)
#                     split_data = chunk_data[mask]
#                 else:
#                     # Find which elements in the original chunk correspond to this split
#                     mask = np.isin(source_indices, split_indices)
#                     split_data = chunk_data[mask]
                
#                 # Write data
#                 output_h5[name][relative_indices] = split_data

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Combine H5 files, shuffle, and split into train/val/test')
#     parser.add_argument('--rapgap', type=str, required=True, help='Path to rapgap H5 file')
#     parser.add_argument('--djangoh', type=str, required=True, help='Path to djangoh H5 file')
#     parser.add_argument('--output_dir', type=str, default='split_data', help='Directory to save output H5 files')
#     parser.add_argument('--train_ratio', type=float, default=0.7, help='Fraction of data for training')
#     parser.add_argument('--val_ratio', type=float, default=0.15, help='Fraction of data for validation')
#     parser.add_argument('--test_ratio', type=float, default=0.15, help='Fraction of data for testing')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
#     parser.add_argument('--no_shuffle', action='store_true', help='Disable shuffling')
#     parser.add_argument('--keep_combined', action='store_true', help='Save the combined dataset as well')
#     parser.add_argument('--chunk_size', type=int, default=1000, help='Chunk size for processing')
    
#     args = parser.parse_args()
    
#     # Check if input files exist
#     if not os.path.isfile(args.rapgap):
#         raise FileNotFoundError(f"Rapgap file not found: {args.rapgap}")
#     if not os.path.isfile(args.djangoh):
#         raise FileNotFoundError(f"Djangoh file not found: {args.djangoh}")
    
#     # Verify ratios are valid
#     if args.train_ratio <= 0 or args.val_ratio < 0 or args.test_ratio < 0:
#         raise ValueError("Ratios must be positive (train_ratio > 0, val_ratio >= 0, test_ratio >= 0)")
    
#     # Combine and split the datasets
#     combine_and_split_h5(
#         args.rapgap,
#         args.djangoh,
#         args.output_dir,
#         args.train_ratio,
#         args.val_ratio,
#         args.test_ratio,
#         not args.no_shuffle,
#         args.seed,
#         args.keep_combined,
#         args.chunk_size
#     )
    
#     print(f"Successfully created train, validation, and test datasets in {args.output_dir}")
