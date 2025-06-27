# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path

# def load_and_plot_kinematics(file_path='combinedh1_withscalars.npz'):
#     """
#     Load NPZ file and plot kinematics data for different label values
#     """
#     # Load the data
#     print("Loading data...")
#     data = np.load(file_path)
    
#     # Extract the relevant arrays
#     kinematics_train = data['kinematics_train']  # Shape: (4200000, 30, 4)
#     event_scalars_train = data['event_scalars_train'] # Shape: (4200000, 30, 8)
#     jet_train = data['jet_train'] # Shape: (4200000, 4)
#     labels_train = data['labels_train']
    
#     print(f"Kinematics shape: {kinematics_train.shape}")
#     print(f"Event scalars shape: {event_scalars_train.shape}")
#     print(f"Jet shape: {jet_train.shape}")
#     print(f"Labels shape: {labels_train.shape}")
#     print(f"Unique labels: {np.unique(labels_train)}")
    
#     # Get unique labels
#     unique_labels = np.unique(labels_train)
    
#     # Feature names for clarity
#     feature_names = ['Energy', 'Px', 'Py', 'Pz']
#     scalar_names = ['delta eta', 'delta phi', 'log(1-rel pt)', 'log pt','log E', 'log(1-rel E)', ' delta R', 'charge']
#     jet_names = ['Jet pT','Jet eta', 'Jet mass', 'Jet multiplcity']
#     x_ranges = [[0,25], [-5,5], [-5,5], [0,10]]
#     scalar_x_ranges = [[-1,1], [-1,1], [-1.75,0], [-3,3], [-1.75,0], [-3,5], [0,1.2], [-2,2]]
#     jet_x_ranges = [[10,80],[-1.5,2.5],[0,25],[0,30]]

    
#     # Create subplots for each feature
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#     axes = axes.flatten()

    
    
#     # Plot each feature ([:,:,0], [:,:,1], [:,:,2], [:,:,3])
#     for feature_idx in range(4):
#         ax = axes[feature_idx]
        
#         # Extract data for current feature across all particles
#         feature_data = kinematics_train[:, :, feature_idx]  # Shape: (4200000, 30)
#         x_min, x_max = x_ranges[feature_idx]
#         for label in unique_labels:
#             # Get indices for current label
#             label_mask = labels_train == label
#             label_data = feature_data[label_mask]
            
#             # Flatten the data for histogram (all particles, all events)
#             flattened_data = label_data.flatten()
            
#             # Remove any NaN or infinite values
#             flattened_data = flattened_data[np.isfinite(flattened_data)]

#             label_name = "Djangoh" if label == 0 else "Rapgap"
            
#             # Plot histogram
#             ax.hist(flattened_data, bins=10, alpha=0.7, label=label_name, 
#                    density=True, histtype='step', linewidth=2, range=(x_min, x_max))
        
#         ax.set_xlabel(f'{feature_names[feature_idx]}')
#         ax.set_ylabel('Normalized Events')
#         ax.set_yscale('log')
#         # ax.set_title(f'Distribution of {feature_names[feature_idx]} by Label')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('kinematics_distributions.png', dpi=300, bbox_inches='tight')
#     plt.show()

#     fig, axes = plt.subplots(2, 4, figsize=(15, 12))
#     axes = axes.flatten()

#     for scalar_idx in range(8):
#         ax = axes[scalar_idx]
#         x_min, x_max = scalar_x_ranges[scalar_idx]

#         # Extract data for current scalar across all particles
#         scalar_data = event_scalars_train[:, :, scalar_idx]  # Shape: (4200000, 30)
        
#         # Collect all data for this scalar to determine reasonable range
#         all_scalar_data = []
        
#         for label in unique_labels:
#             # Get indices for current label
#             label_mask = labels_train == label
#             label_data = scalar_data[label_mask]
            
#             # Flatten the data for histogram (all particles, all events)
#             flattened_data = label_data.flatten()
            
#             # Remove any NaN or infinite values
#             flattened_data = flattened_data[np.isfinite(flattened_data)]
#             all_scalar_data.extend(flattened_data)
    
        
#         for label in unique_labels:
#             # Get indices for current label
#             label_mask = labels_train == label
#             label_data = scalar_data[label_mask]
            
#             # Flatten the data for histogram (all particles, all events)
#             flattened_data = label_data.flatten()
            
#             # Remove any NaN or infinite values
#             flattened_data = flattened_data[np.isfinite(flattened_data)]
            
#             # Set label name based on value
#             label_name = "Djangoh" if label == 0 else "Rapgap"
            
#             # Plot histogram with custom range
#             ax.hist(flattened_data, bins=10, alpha=0.7, label=label_name, 
#                    density=True, histtype='step', linewidth=2, range=(x_min, x_max))
        
#         # Set the x-axis limits
#         ax.set_xlim(x_min, x_max)
#         ax.set_xlabel(f'{scalar_names[scalar_idx]}')
#         ax.set_ylabel('Normalized Events')
#         # ax.set_title(f'{scalar_names[scalar_idx]}')
#         ax.set_yscale('log')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('event_scalars_distributions.png', dpi=300, bbox_inches='tight')
#     plt.show()

#      # Create subplots for each feature
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#     axes = axes.flatten()

#     # Plot each feature ([:,:,0], [:,:,1], [:,:,2], [:,:,3])
#     for feature_idx in range(4):
#         ax = axes[feature_idx]
        
#         # Extract data for current feature across all particles
#         feature_data = jet_train[:, feature_idx]  # Shape: (4200000, 4)
#         x_min, x_max = jet_x_ranges[feature_idx]
#         for label in unique_labels:
#             # Get indices for current label
#             label_mask = labels_train == label
#             label_data = feature_data[label_mask]
            
#             # Flatten the data for histogram (all particles, all events)
#             flattened_data = label_data.flatten()
            
#             # Remove any NaN or infinite values
#             flattened_data = flattened_data[np.isfinite(flattened_data)]

#             label_name = "Djangoh" if label == 0 else "Rapgap"
            
#             # Plot histogram
#             ax.hist(flattened_data, bins=10, alpha=0.7, label=label_name, 
#                    density=True, histtype='step', linewidth=2, range=(x_min, x_max))
        
#         ax.set_xlabel(f'{jet_names[feature_idx]}')
#         ax.set_ylabel('Normalized Events')
#         ax.set_yscale('log')
#         # ax.set_title(f'Distribution of {feature_names[feature_idx]} by Label')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('jet_distributions.png', dpi=300, bbox_inches='tight')
#     plt.show()

#     fig, axes = plt.subplots(2, 4, figsize=(15, 12))
#     axes = axes.flatten()


# if __name__ == "__main__":
#     # Set style for better plots
#     plt.style.use('seaborn-v0_8')
#     sns.set_palette("husl")
    
#     # Main plotting function
#     load_and_plot_kinematics()
    
#     # Correlation analysis
#     # plot_correlation_matrix()
    
#     print("\nPlots saved as:")
#     print("- kinematics_distributions.png")
#     # print("- kinematics_by_position.png") 
#     # print("- feature_correlations.png")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_plot_kinematics(file_path='combinedh1_withscalars.npz'):
    """
    Load NPZ file and plot kinematics data for different label values
    """
    # Load the data
    print("Loading data...")
    data = np.load(file_path)
    
    # Extract the relevant arrays
    kinematics_train = data['kinematics_train']  # Shape: (4200000, 30, 4)
    event_scalars_train = data['event_scalars_train'] # Shape: (4200000, 30, 8)
    jet_train = data['jet_train'] # Shape: (4200000, 4)
    labels_train = data['labels_train']
    
    print(f"Kinematics shape: {kinematics_train.shape}")
    print(f"Event scalars shape: {event_scalars_train.shape}")
    print(f"Jet shape: {jet_train.shape}")
    print(f"Labels shape: {labels_train.shape}")
    print(f"Unique labels: {np.unique(labels_train)}")
    
    # Get unique labels
    unique_labels = np.unique(labels_train)
    
    # Feature names for clarity
    feature_names = ['Energy', 'Px', 'Py', 'Pz']
    scalar_names = ['delta eta', 'delta phi', 'log(1-rel pt)', 'log pt','log E', 'log(1-rel E)', ' delta R', 'charge']
    jet_names = ['Jet pT','Jet eta', 'Jet mass', 'Jet multiplcity']
    x_ranges = [[0,25], [-5,5], [-5,5], [0,10]]
    scalar_x_ranges = [[-1,1], [-1,1], [-1.75,0], [-3,3], [-1.75,0], [-3,5], [0,1.2], [-2,2]]
    jet_x_ranges = [[10,80],[-1.5,2.5],[0,25],[0,30]]

    # Plot kinematics features - save each individually
    for feature_idx in range(4):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Extract data for current feature across all particles
        feature_data = kinematics_train[:, :, feature_idx]  # Shape: (4200000, 30)
        x_min, x_max = x_ranges[feature_idx]
        
        for label in unique_labels:
            # Get indices for current label
            label_mask = labels_train == label
            label_data = feature_data[label_mask]
            
            # Flatten the data for histogram (all particles, all events)
            flattened_data = label_data.flatten()
            
            # Remove any NaN or infinite values
            flattened_data = flattened_data[np.isfinite(flattened_data)]

            label_name = "Djangoh" if label == 0 else "Rapgap"
            
            # Plot histogram
            ax.hist(flattened_data, bins=10, alpha=0.7, label=label_name, 
                   density=True, histtype='step', linewidth=2, range=(x_min, x_max))
        
        ax.set_xlabel(f'{feature_names[feature_idx]}')
        ax.set_ylabel('Normalized Events')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'kinematics_{feature_names[feature_idx].lower()}_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    # Plot event scalars - save each individually
    for scalar_idx in range(8):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        x_min, x_max = scalar_x_ranges[scalar_idx]

        # Extract data for current scalar across all particles
        scalar_data = event_scalars_train[:, :, scalar_idx]  # Shape: (4200000, 30)
        
        # Collect all data for this scalar to determine reasonable range
        all_scalar_data = []
        
        for label in unique_labels:
            # Get indices for current label
            label_mask = labels_train == label
            label_data = scalar_data[label_mask]
            
            # Flatten the data for histogram (all particles, all events)
            flattened_data = label_data.flatten()
            
            # Remove any NaN or infinite values
            flattened_data = flattened_data[np.isfinite(flattened_data)]
            all_scalar_data.extend(flattened_data)
        
        for label in unique_labels:
            # Get indices for current label
            label_mask = labels_train == label
            label_data = scalar_data[label_mask]
            
            # Flatten the data for histogram (all particles, all events)
            flattened_data = label_data.flatten()
            
            # Remove any NaN or infinite values
            flattened_data = flattened_data[np.isfinite(flattened_data)]
            
            # Set label name based on value
            label_name = "Djangoh" if label == 0 else "Rapgap"
            
            # Plot histogram with custom range
            ax.hist(flattened_data, bins=10, alpha=0.7, label=label_name, 
                   density=True, histtype='step', linewidth=2, range=(x_min, x_max))
        
        # Set the x-axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(f'{scalar_names[scalar_idx]}')
        ax.set_ylabel('Normalized Events')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Clean filename by removing spaces and special characters
        clean_name = scalar_names[scalar_idx].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        plt.savefig(f'event_scalar_{clean_name}_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    # Plot jet features - save each individually
    for feature_idx in range(4):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Extract data for current feature across all particles
        feature_data = jet_train[:, feature_idx]  # Shape: (4200000, 4)
        x_min, x_max = jet_x_ranges[feature_idx]
        
        for label in unique_labels:
            # Get indices for current label
            label_mask = labels_train == label
            label_data = feature_data[label_mask]
            
            # Flatten the data for histogram (all particles, all events)
            flattened_data = label_data.flatten()
            
            # Remove any NaN or infinite values
            flattened_data = flattened_data[np.isfinite(flattened_data)]

            label_name = "Djangoh" if label == 0 else "Rapgap"
            
            # Plot histogram
            ax.hist(flattened_data, bins=10, alpha=0.7, label=label_name, 
                   density=True, histtype='step', linewidth=2, range=(x_min, x_max))
        
        ax.set_xlabel(f'{jet_names[feature_idx]}')
        ax.set_ylabel('Normalized Events')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Clean filename by removing spaces and special characters
        clean_name = jet_names[feature_idx].replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(f'jet_{clean_name.lower()}_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == "__main__":
    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Main plotting function
    load_and_plot_kinematics()
    
    print("\nIndividual plots saved as:")
    print("Kinematics plots:")
    print("- kinematics_energy_distribution.png")
    print("- kinematics_px_distribution.png")
    print("- kinematics_py_distribution.png")
    print("- kinematics_pz_distribution.png")
    print("\nEvent scalar plots:")
    print("- event_scalar_delta_eta_distribution.png")
    print("- event_scalar_delta_phi_distribution.png")
    print("- event_scalar_log1_rel_pt_distribution.png")
    print("- event_scalar_log_pt_distribution.png")
    print("- event_scalar_log_E_distribution.png")
    print("- event_scalar_log1_rel_E_distribution.png")
    print("- event_scalar__delta_R_distribution.png")
    print("- event_scalar_charge_distribution.png")
    print("\nJet plots:")
    print("- jet_jet_pt_distribution.png")
    print("- jet_jet_eta_distribution.png")
    print("- jet_jet_mass_distribution.png")
    print("- jet_jet_multiplcity_distribution.png")