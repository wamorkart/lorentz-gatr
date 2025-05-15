import os
import uproot3 as uproot
import pandas as pd
import numpy as np
import awkward
from optparse import OptionParser
from sklearn.utils import shuffle
import h5py
import fastjet as fj


def find_files_with_string(directory, string):
    matching_files = []
    for filename in os.listdir(directory):
        if string in filename:
            matching_files.append(filename)
    return matching_files


def convert_to_np(file_list, base_path, name, max_part=30, nevts=30000000):
    data_dict = {
        'jet': [],
        'data': [],
    }

    mask_list = ['y', 'ptmiss', 'Empz']
    jet_list = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_e', 'jet_mass']
    particle_list = ['jet_part_pt', 'jet_part_eta', 'jet_part_phi', 'jet_part_e', 'jet_part_charge', 'jet_part_idx']

    nevt = 0
    # maxevt = int(3.5e6)
    maxevt = min(nevts, int(3.5e6))

    for f in file_list:
        print("evaluating file {}".format(f))
        if nevt >= maxevt:
            break
        print("Currently keeping {} events".format(nevt))
        try:
            tmp_file = uproot.open(os.path.join(base_path, f))['{}/minitree'.format(name)]
        except:
            print('No TTree found, skipping')
            continue

        print("loaded file")
        mask_evt = (tmp_file['Q2'].array()[:] > 150)
        data_dict['jet'].append(np.stack([tmp_file[feat].array()[mask_evt].pad(1, clip=True).fillna(0).regular() for feat in jet_list], -1))
        data_dict['data'].append(np.stack([tmp_file[feat].array()[mask_evt].pad(max_part, clip=True).fillna(0).regular() for feat in particle_list], -1))
        data_dict['jet'][-1] = np.squeeze(data_dict['jet'][-1])

        mask_reco = np.stack([tmp_file[feat].array()[mask_evt] for feat in mask_list], -1)
        print("Number of events: {}".format(mask_reco.shape[0]))

        pass_reco = (mask_reco[:, 0] > 0.08) & (mask_reco[:, 0] < 0.7) & (mask_reco[:, 1] < 10.0) & (mask_reco[:, 2] > 45.) & (mask_reco[:, 2] < 65)

        data_dict['jet'][-1] = data_dict['jet'][-1][pass_reco]
        data_dict['data'][-1] = data_dict['data'][-1][pass_reco]
        del mask_reco, pass_reco

        pass_reco = (data_dict['jet'][-1][:, 0] > 10) & (data_dict['jet'][-1][:, 1] > -1.5) & (data_dict['jet'][-1][:, 1] < 2.75)
        data_dict['jet'][-1] = data_dict['jet'][-1][pass_reco]
        data_dict['data'][-1] = data_dict['data'][-1][pass_reco]

        nevt += data_dict['jet'][-1].shape[0]

        mask_part = (data_dict['data'][-1][:, :, 1] > -1.5) & (data_dict['data'][-1][:, :, 1] < 2.75) & (data_dict['data'][-1][:, :, -1] == 0)
        data_dict['data'][-1] *= mask_part[:, :, None]
        mask_evt = np.sum(data_dict['data'][-1][:, :, 0], 1) > 0

        del mask_part
        print("Rejecting {}".format(1.0 - 1.0 * np.sum(mask_evt) / mask_evt.shape[0]))
        data_dict['data'][-1] = data_dict['data'][-1][mask_evt]
        data_dict['jet'][-1] = data_dict['jet'][-1][mask_evt]

    data_dict['data'] = np.concatenate(data_dict['data'])[:maxevt]
    data_dict['jet'] = np.concatenate(data_dict['jet'])[:maxevt]

    del tmp_file
    return data_dict


def _calculate_jet_features(jet):
    tau_11, tau_11p5, tau_12, tau_20, sumpt = 0, 0, 0, 0, 0
    for constituent in jet.constituents():
        pt = constituent.pt()
        delta_r = jet.delta_R(constituent)
        tau_11 += pt * delta_r ** 1
        tau_11p5 += pt * delta_r ** 1.5
        tau_12 += pt * delta_r ** 2
        tau_20 += pt ** 2
        sumpt += pt

    return {
        'tau_11': np.log(tau_11 / jet.pt()) if jet.pt() > 0 else 0.0,
        'tau_11p5': np.log(tau_11p5 / jet.pt()) if jet.pt() > 0 else 0.0,
        'tau_12': np.log(tau_12 / jet.pt()) if jet.pt() > 0 else 0.0,
        'tau_20': tau_20 / (jet.pt() ** 2) if jet.pt() > 0 else 0.0,
        'ptD': np.sqrt(tau_20) / sumpt if sumpt > 0 else 0.0,
    }



def make_np_entries(particles, jets):
    mask = particles[:, :, 0] > 0
    NFEAT = 8
    points = np.zeros((particles.shape[0], particles.shape[1], NFEAT))

    # particles = ['jet_part_pt', 'jet_part_eta', 'jet_part_phi', 'jet_part_e', 'jet_part_charge', 'jet_part_idx']

    part_deltaeta = particles[:,:,1]
    part_deltaphi = particles[:,:,2]
    part_relpt = np.ma.log(1.0 - particles[:,:,0]).filled(0)
    part_pt = np.exp(np.ma.log(particles[:,:,0]*jets[:,0,None]).filled(0))
    part_rele = np.ma.log(1.0 - particles[:,:,3]).filled(0)
    part_e = np.exp(np.ma.log(particles[:,:,3]*jets[:,3,None]).filled(0))
    part_hypot = np.hypot(points[:,:,0],points[:,:,1])
    part_charge = particles[:,:,4]


    

    # pt = particles[:, :, 0]
    # eta = particles[:, :, 1]
    # phi = particles[:, :, 2]
    # e = particles[:, :, 3]
    # charge = particles[:, :, 4]

    

    


    points *= mask[:, :, None]
    jets = np.concatenate([jets, np.sum(mask, -1)[:, None]], -1)
    jet_eta = jets[:,1]
    jet_phi = jets[:,2]

    part_eta = part_deltaeta + jet_eta[:, np.newaxis]
    part_phi = part_deltaphi + jet_phi[:, np.newaxis]

    part_px = part_pt * np.cos(part_phi)
    part_py = part_pt * np.sin(part_phi)
    part_pz = part_pt * np.sinh(part_eta)

    points[:, :, 0] = part_pt
    points[:, :, 1] = part_eta
    points[:, :, 2] = part_phi
    points[:, :, 3] = part_e
    points[:, :, 4] = part_px
    points[:, :, 5] = part_py
    points[:, :, 6] = part_pz
    points[:, :, 7] = part_charge

    jets = np.delete(jets, [2, 3], axis=1)

    clustered_jets = []
    for i in range(particles.shape[0]):
        fj_particles = []
        for j in range(particles.shape[1]):
            if particles[i, j, 0] <= 0:
                continue
            fj_particles.append(fj.PseudoJet(part_px[i, j], part_py[i, j], part_pz[i, j], part_e[i, j]))
        if not fj_particles:
            clustered_jets.append([0] * 5)
            continue

        jet_def = fj.JetDefinition(fj.antikt_algorithm, 1.0)
        cs = fj.ClusterSequence(fj_particles, jet_def)
        inclusive_jets = cs.inclusive_jets()

        if not inclusive_jets:
            clustered_jets.append([0] * 5)
            continue

        features = _calculate_jet_features(inclusive_jets[0])
        clustered_jets.append([
            features['tau_11'],
            features['tau_11p5'],
            features['tau_12'],
            features['tau_20'],
            features['ptD'],
        ])

    clustered_jets = np.array(clustered_jets)
    jets = np.concatenate([jets, clustered_jets], axis=-1)
    return points, jets


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--npoints", type=int, default=30, help="Number of particles per event")
    parser.add_option("--folder", type="string", default='/global/cfs/cdirs/m3246/vmikuni/H1v2/PET', help="Folder containing input files")

    (flags, args) = parser.parse_args()

    file_list = find_files_with_string(flags.folder + '/out_ep0607', 'Django_Eplus0607_')
    data_django = convert_to_np(file_list, flags.folder + '/out_ep0607', name='Django', max_part=flags.npoints)
    dj_p, dj_j = make_np_entries(data_django['data'], data_django['jet'])

    assert not np.any(np.isnan(dj_p)), "ERROR: NAN in particles"
    assert not np.any(np.isnan(dj_j)), "ERROR: NAN in jets"

    file_list = find_files_with_string(flags.folder + '/out_ep0607', 'Rapgap_Eplus0607_')
    data_rapgap = convert_to_np(file_list, flags.folder + '/out_ep0607', name='Rapgap', max_part=flags.npoints)
    rp_p, rp_j = make_np_entries(data_rapgap['data'], data_rapgap['jet'])

    assert not np.any(np.isnan(rp_p)), "ERROR: NAN in particles"
    assert not np.any(np.isnan(rp_j)), "ERROR: NAN in jets"

    particles, jets, pid = shuffle(
        np.concatenate([rp_p, dj_p], 0),
        np.concatenate([rp_j, dj_j], 0),
        np.concatenate([np.ones(rp_j.shape[0]), np.zeros(dj_j.shape[0])], 0),
    )

    print(particles.shape)
    total = particles.shape[0]

    ntrain = int(0.6 * total)
    nval = int(0.2 * total)
    ntest = total - ntrain - nval  # remaining 20%

    assert ntrain > ntest, "Training set should be larger than test set"

    # Split and save
    with h5py.File('train.h5', "w") as fh5:
        fh5.create_dataset('data', data=particles[:ntrain])
        fh5.create_dataset('jet', data=jets[:ntrain])
        fh5.create_dataset('pid', data=pid[:ntrain])

    with h5py.File('val.h5', "w") as fh5:
        fh5.create_dataset('data', data=particles[ntrain:ntrain+nval])
        fh5.create_dataset('jet', data=jets[ntrain:ntrain+nval])
        fh5.create_dataset('pid', data=pid[ntrain:ntrain+nval])

    with h5py.File('test.h5', "w") as fh5:
        fh5.create_dataset('data', data=particles[ntrain+nval:])
        fh5.create_dataset('jet', data=jets[ntrain+nval:])
        fh5.create_dataset('pid', data=pid[ntrain+nval:])
    # ntrain = int(2.5e6)
    # nval = int(3.0e6)
    # with h5py.File('train.h5', "w") as fh5:
    #     fh5.create_dataset('data', data=particles[:ntrain])
    #     fh5.create_dataset('jet', data=jets[:ntrain])
    #     fh5.create_dataset('pid', data=pid[:ntrain])

    # with h5py.File('val.h5', "w") as fh5:
    #     fh5.create_dataset('data', data=particles[ntrain:nval])
    #     fh5.create_dataset('jet', data=jets[ntrain:nval])
    #     fh5.create_dataset('pid', data=pid[ntrain:nval])

    # with h5py.File('test.h5', "w") as fh5:
    #     fh5.create_dataset('data', data=particles[nval:])
    #     fh5.create_dataset('jet', data=jets[nval:])
    #     fh5.create_dataset('pid', data=pid[nval:])
