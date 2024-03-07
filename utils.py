import numpy as np

def bin_spectrum(wav, spec, wavs_per_bin):

    new_wav = wav[::wavs_per_bin]
    inds = np.concatenate([
        np.where(np.isclose(w, wav))[0] for w in new_wav
    ])
    inds[-1] = -1
    binned_spec = np.array([
        np.sum(np.array(
            spec[:, inds[i]:inds[i+1]]
        ), axis=1) 
        for i in range(len(inds) - 1)
    ])
    return wav[inds][:-1] + np.diff(wav[inds]), binned_spec