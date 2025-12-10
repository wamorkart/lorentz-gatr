import numpy as np
import matplotlib.pyplot as plt

def load_and_plot_kinematics(file_path='combinedh1_withscalars.npz'):
    """
    Load NPZ file and plot kinematics data for different label values
    """

    print("Loading data...")
    data = np.load(file_path)

    kinematics_train = data['kinematics_train']
    event_scalars_train = data['event_scalars_train']
    jet_train = data['jet_train']
    labels_train = data['labels_train']

    print(f"Kinematics shape: {kinematics_train.shape}")
    print(f"Event scalars shape: {event_scalars_train.shape}")
    print(f"Jet shape: {jet_train.shape}")
    print(f"Labels shape: {labels_train.shape}")
    print(f"Unique labels: {np.unique(labels_train)}")

    unique_labels = np.unique(labels_train)

    # -----------------------------------------------------------------------
    # Pretty labels for plot x-axes (LaTeX formatted)
    # -----------------------------------------------------------------------
    feature_labels = ['Energy', 'Px', 'Py', 'Pz']

    scalar_labels = [
        r'$\Delta \eta$',
        r'$\Delta \phi$',
        r'$\log(1 - p_T^\mathrm{rel})$',
        r'$\log p_T$',
        r'$\log E$',
        r'$\log(1 - E^\mathrm{rel})$',
        r'$\Delta R$',
        'Charge'
    ]

    jet_labels = ['Jet $p_T$', 'Jet $\eta$', 'Jet mass', 'Jet multiplicity']

    # -----------------------------------------------------------------------
    # Clean ASCII names ONLY used for saving filenames
    # -----------------------------------------------------------------------
    feature_keys = ['energy', 'px', 'py', 'pz']

    scalar_keys = [
        'delta_eta',
        'delta_phi',
        'log1_rel_pt',
        'log_pt',
        'log_E',
        'log1_rel_E',
        'delta_R',
        'charge'
    ]

    jet_keys = ['jet_pt', 'jet_eta', 'jet_mass', 'jet_multiplicity']

    # Ranges
    x_ranges = [[0,25], [-5,5], [-5,5], [0,10]]
    scalar_x_ranges = [[-3,3], [-1.5,1.5], [-1.75,0], [-3,3], [-1.75,0], [-3,5], [0,1.2], [-2,2]]
    jet_x_ranges = [[10,80],[-1.5,2.5],[0,25],[0,30]]

    # -----------------------------------------------------------------------
    # PLOT KINEMATICS
    # -----------------------------------------------------------------------
    for idx in range(4):
        for particle_type in ['all', 'leading']:
            fig, ax = plt.subplots(figsize=(8, 6))
            x_min, x_max = x_ranges[idx]

            if particle_type == 'all':
                feature_data = kinematics_train[:, :, idx].copy()  # all particles
            else:
                feature_data = kinematics_train[:, 0, idx].copy()  # leading particle only

            for label in unique_labels:
                mask = labels_train == label
                if particle_type == 'all':
                    flattened = feature_data[mask].flatten()
                else:
                    flattened = feature_data[mask]
                flattened = flattened[np.isfinite(flattened)]

                label_name = "Djangoh" if label == 0 else "Rapgap"

                ax.hist(flattened, bins=10, alpha=0.7, label=label_name,
                        density=True, histtype='step', linewidth=2,
                        range=(x_min, x_max))

            ax.set_xlabel(feature_labels[idx])
            ax.set_ylabel('Normalized Counts')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.savefig(f'kinematics_{feature_keys[idx]}_{particle_type}_distribution.png', dpi=300)
            plt.show()
            plt.close()

    # -----------------------------------------------------------------------
    # PLOT EVENT SCALARS
    # -----------------------------------------------------------------------
    for idx in range(8):
        for particle_type in ['all', 'leading']:
            fig, ax = plt.subplots(figsize=(8, 6))
            x_min, x_max = scalar_x_ranges[idx]

            if particle_type == 'all':
                scalar_data = event_scalars_train[:, :, idx].copy()  # all particles
            else:
                scalar_data = event_scalars_train[:, 0, idx].copy()  # leading particle only

            for label in unique_labels:
                mask = labels_train == label
                if particle_type == 'all':
                    flattened = scalar_data[mask].flatten()
                else:
                    flattened = scalar_data[mask]
                flattened = flattened[np.isfinite(flattened)]

                label_name = "Djangoh" if label == 0 else "Rapgap"

                ax.hist(flattened, bins=10, alpha=0.7, label=label_name,
                        density=True, histtype='step', linewidth=2,
                        range=(x_min, x_max))

            ax.set_xlim(x_min, x_max)
            ax.set_xlabel(scalar_labels[idx])
            ax.set_ylabel('Normalized Counts')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.savefig(f'event_scalar_{scalar_keys[idx]}_{particle_type}_distribution.png', dpi=300)
            plt.show()
            plt.close()

    # -----------------------------------------------------------------------
    # PLOT JET FEATURES (unchanged)
    # -----------------------------------------------------------------------
    for idx in range(4):
        fig, ax = plt.subplots(figsize=(8, 6))
        x_min, x_max = jet_x_ranges[idx]

        feature_data = jet_train[:, idx]

        for label in unique_labels:
            mask = labels_train == label
            flattened = feature_data[mask]
            flattened = flattened[np.isfinite(flattened)]

            label_name = "Djangoh" if label == 0 else "Rapgap"

            ax.hist(flattened, bins=10, alpha=0.7, label=label_name,
                    density=True, histtype='step', linewidth=2,
                    range=(x_min, x_max))

        ax.set_xlabel(jet_labels[idx])
        ax.set_ylabel('Normalized Counts')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'jet_{jet_keys[idx]}_distribution.png', dpi=300)
        plt.show()
        plt.close()


if __name__ == "__main__":
    plt.style.use('fast')
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['axes.titlesize'] = 18

    load_and_plot_kinematics()
