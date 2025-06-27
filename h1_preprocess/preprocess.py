import os
import uproot as uproot
import pandas as pd
import numpy as np
import awkward as ak
from optparse import OptionParser
from sklearn.utils import shuffle
import h5py

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
    particle_list = ['jet_part_pt', 'jet_part_eta', 'jet_part_phi',
                     'jet_part_e', 'jet_part_charge', 'jet_part_idx']

    nevt = 0
    maxevt = int(3.5e6)
    # maxevt = int(1e6)


    for ifile, f in enumerate(file_list):
        if nevt >= maxevt:
            print(f"Reached maxevt = {maxevt}, stopping file reading.")
            break

        print(f"Evaluating file {f}")
        try:
            tmp_file = uproot.open(os.path.join(base_path, f))[f'{name}/minitree']
        except (OSError, ValueError, KeyError, Exception) as e:
            print(f'Error opening file {f}: {type(e).__name__}: {e}')
            print('File appears to be corrupted or incomplete, skipping')
            continue

        print("Loaded file")
        print(f"Number of events in file: {len(tmp_file)}")

        mask_evt = tmp_file['Q2'].array()[:] > 150
        jets = np.stack([
            ak.to_numpy(ak.fill_none(ak.pad_none(tmp_file[feat].array()[mask_evt], 1, clip=True), 0))
            for feat in jet_list
        ], -1)
        jets = np.squeeze(jets)

        particles = np.stack([
            ak.to_numpy(ak.fill_none(ak.pad_none(tmp_file[feat].array()[mask_evt], max_part, clip=True), 0))
            for feat in particle_list
        ], -1)

        mask_reco = np.stack([
            tmp_file[feat].array()[mask_evt] for feat in mask_list
        ], -1)

        print(f"Number of events: {len(mask_reco)}")

        pass_reco = (
            (mask_reco[:, 0] > 0.08) & (mask_reco[:, 0] < 0.7) &
            (mask_reco[:, 1] < 10.0) &
            (mask_reco[:, 2] > 45.) & (mask_reco[:, 2] < 65)
        )
        jets = jets[pass_reco]
        particles = particles[pass_reco]

        pass_reco = (
            (jets[:, 0] > 10) &
            (jets[:, 1] > -1.5) & (jets[:, 1] < 2.75)
        )
        jets = jets[pass_reco]
        particles = particles[pass_reco]

        mask_part = (
            (particles[:, :, 1] > -1.5) &
            (particles[:, :, 1] < 2.75) &
            (particles[:, :, -1] == 0)
        )
        particles *= mask_part[:, :, None]
        mask_evt = np.sum(particles[:, :, 0], axis=1) > 0

        print("Rejecting {:.2f}%".format(
            100 * (1.0 - np.sum(mask_evt) / mask_evt.shape[0])
        ))

        jets = jets[mask_evt]
        particles = particles[mask_evt]

        nevt += jets.shape[0]

        data_dict['jet'].append(jets)
        data_dict['data'].append(particles)

        if nevt >= maxevt:
            print(f"Reached maxevt = {maxevt} after file {f}, stopping.")
            break

    if len(data_dict['jet']) == 0 or len(data_dict['data']) == 0:
        print("No valid events found.")
        return {'jet': np.array([]), 'data': np.array([])}

    data_dict['jet'] = np.concatenate(data_dict['jet'])[:maxevt]
    data_dict['data'] = np.concatenate(data_dict['data'])[:maxevt]

    return data_dict



def make_np_entries(particles,jets):
    mask = particles[:,:,0]>0
    NFEAT=8
    points = np.zeros((particles.shape[0],particles.shape[1],NFEAT))

    points[:,:,0] = particles[:,:,1] #delta eta
    points[:,:,1] = particles[:,:,2] #deltaphi
    points[:,:,2] = np.ma.log(1.0 - particles[:,:,0]).filled(0) #ln(1-relpt)
    points[:,:,3] = np.ma.log(particles[:,:,0]*jets[:,0,None]).filled(0) #ln(pt)
    
    points[:,:,4] = np.ma.log(1.0 - particles[:,:,3]).filled(0) #ln(1-relE)
    points[:,:,5] = np.ma.log(particles[:,:,3]*jets[:,3,None]).filled(0) #(ln E)
    points[:,:,6] = np.hypot(points[:,:,0],points[:,:,1]) #(delta R?)
    points[:,:,7] = particles[:,:,4] #charge?



    particle_pt = np.exp(points[:,:,3])
    particle_eta = points[:,:,0] + jets[:,1][:, np.newaxis]
    particle_phi = points[:,:,1] + jets[:,2][:, np.newaxis]
    particle_energy = np.exp(points[:,:,5])
 
    particle_px = particle_pt * np.cos(particle_phi)
    particle_py = particle_pt * np.sin(particle_phi)
    particle_pz = particle_pt * np.sinh(particle_eta)

    four_momentum = np.stack([particle_energy, particle_px, particle_py, particle_pz], axis=-1)
    # print(f"particle four_momentum: {four_momentum}")

    
    points*=mask[:,:,None]
    four_momentum *= mask[:, :, None]
    jets = np.concatenate([jets,np.sum(mask,-1)[:,None]],-1)
    #delete phi and energy
    jets = np.delete(jets,[2,3],axis=1)
    return points,jets, four_momentum

if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=30, help="Number of particles per event")
    parser.add_option("--folder", type="string", default='/pscratch/sd/v/vmikuni/PET/H1', help="Folder containing input files")

    (flags, args) = parser.parse_args()


    file_list = find_files_with_string(flags.folder+'/out_ep0607', 'Django_Eplus0607_')
    data_django = convert_to_np(file_list,flags.folder+'/out_ep0607',name='Django',max_part=flags.npoints)
    dj_p,dj_j,dj_4mom = make_np_entries(data_django['data'],data_django['jet'])
    # dj_4mom = data_django['four_momenta'] 
        
    assert np.any(np.isnan(dj_p)) == False, "ERROR: NAN in particles"
    assert np.any(np.isnan(dj_j)) == False, "ERROR: NAN in jets"


    file_list = find_files_with_string(flags.folder+'/out_ep0607', 'Rapgap_Eplus0607_')
    data_rapgap = convert_to_np(file_list,flags.folder+'/out_ep0607',name='Rapgap',max_part=flags.npoints)
    rp_p,rp_j, rp_4mom = make_np_entries(data_rapgap['data'],data_rapgap['jet'])
    # rp_4mom = data_rapgap['four_momenta']  # Get four-momentum

        
    assert np.any(np.isnan(rp_p)) == False, "ERROR: NAN in particles"
    assert np.any(np.isnan(rp_j)) == False, "ERROR: NAN in jets"

    particles, jets, pid, four_momentum = shuffle(np.concatenate([rp_p,dj_p],0),
                                   np.concatenate([rp_j,dj_j],0),
                                   np.concatenate([np.ones(rp_j.shape[0]),np.zeros(dj_j.shape[0])],0),
                                   np.concatenate([rp_4mom, dj_4mom], 0)
                                   )


    # ntrain = int(2.5e6)
    # nval = int(3.0e6)
    ntotal = particles.shape[0]
    ntrain = int(0.6 * ntotal)
    nval = int(0.2 * ntotal)
    ntest = ntotal - ntrain - nval
    np.savez('train_100points.npz',
             data=particles[:ntrain],
             jet=jets[:ntrain],
             labels_train=pid[:ntrain],
             kinematics_train=four_momentum[:ntrain])

    np.savez('val_100points.npz',
         data=particles[ntrain:ntrain+nval],
         jet=jets[ntrain:ntrain+nval],
         labels_val=pid[ntrain:ntrain+nval],
         kinematics_val=four_momentum[ntrain:ntrain+nval])

    np.savez('test_100points.npz',
         data=particles[ntrain+nval:],
         jet=jets[ntrain+nval:],
         labels_test=pid[ntrain+nval:],
         kinematics_test=four_momentum[ntrain+nval:])

    print(f"Saved training data: {ntrain} events")
    print(f"Saved validation data: {nval} events")  
    print(f"Saved test data: {ntest} events")


    