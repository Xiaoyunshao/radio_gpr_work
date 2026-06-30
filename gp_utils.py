import numpy as np
import torch

import bayeslim as ba
from bayeslim import VisData, telescope_model, dataset
import gprlim


def load_uvdata(dfile, bls=None, freq_chans=None, time_ints=None, pols=None, inflate_by_red=False, device=None, dtype=None):
    """
    Load a pyuvdata UVData and convert data to torch tensor.

    Parameters
    ----------
    dfile : str
        Filepath to UVH5 dataset.

    Returns
    -------
    data, info
    """
    try:
        from pyuvdata import UVData
        # load uvdata
        uvd = UVData()

        # get antenna metadata
        uvd.read(dfile, freq_chans=freq_chans, polarizations=pols, read_data=False, axis='blt')
        antpos, ants = uvd.get_enu_data_ants()
        antp = dict(zip(ants.tolist(), torch.as_tensor(antpos, device=device, dtype=dtype)))
        all_bls = uvd.get_antpairs()

        # now load the data
        uvd.read(dfile, bls=bls, freq_chans=freq_chans, polarizations=pols)

        if time_ints is not None:
            uvd.select(times=np.unique(uvd.time_array)[time_ints])

        # ensure (ant1<ant2) conjugation
        uvd.conjugate_bls('ant1<ant2')

        # inflate to redundancies
        if inflate_by_red:
            uvd.inflate_by_redundancy()

        # unravel to (Npols, 1, Nbls, Ntimes, Nfreqs) and convert to tensor
        data = uvd.data_array.reshape(uvd.Ntimes, uvd.Nbls, uvd.Nfreqs, uvd.Npols)
        data = np.moveaxis(data, 0, 1)
        data = np.moveaxis(data, -1, 0)[:, None]

        # get metadata
        bls = uvd.get_antpairs()
        freqs = uvd.freq_array
        times = np.unique(uvd.time_array)
        lsts = np.unique(np.unwrap(uvd.lst_array))
        lat, lon, alt = uvd.telescope.location_lat_lon_alt_degrees
        pols = uvd.polarization_array.tolist()

        vd = ba.VisData()
        telescope = telescope_model.TelescopeModel((lon, lat, alt))
        vd.setup_meta(telescope, antp)
        vd.setup_data(bls, times, freqs, pol=pols, data=torch.as_tensor(data))

    except ValueError:
        # load BayesLIM VisData

        # get metadata
        vd = VisData.from_hdf5(dfile, read_data=False)
        vd = dataset.concat_VisData(vd, 'time')
        all_bls = vd.bls

        # now load data
        vd = VisData.from_hdf5(dfile, bl=bls, freq_inds=freq_chans, pol=pols)
        vd = dataset.concat_VisData(vd, 'time')

        if time_ints is not None:
            vd.select(time_inds=time_ints, inplace=True)

        # get metadata
        bls = vd.bls
        freqs = vd.freqs
        times = np.unique(vd.times)
        lon, lat = vd.telescope.location[:2]
        lsts = telescope_model.JD2LST(vd.times, lon)
        pols = vd.pol

        # get antenna dictionary
        antp = vd.antpos

    # convert to dtype and/or device
    vd.push(dtype)
    vd.push(device)

    # get redundancies
    from bayeslim.telescope_model import build_reds
    reds = build_reds(antp)[0]

    # get metadata
    meta = {
        'antp': antp,
        'bls': bls,
        'all_bls': all_bls,
        'freqs': torch.as_tensor(freqs, device=device, dtype=dtype),
        'times': torch.as_tensor(times, device=device, dtype=dtype),
        'lsts': torch.as_tensor(lsts, device=device, dtype=dtype),
        'reds': reds,
        'lat': lat,
        'lon': lon,
        'pols': pols
    }

    return vd, meta


def compute_noise_var(auto, dt, dnu):
    """
    Compute noise variance for cross-correlation visibilities

    Parameters
    ----------
    auto : tensor
        Auto-correlation tensor of shape (Ntimes, Nfreqs)
    dt : float
        Time integration [sec]
    dnu : float
        Frequency channelization [Hz]

    """
    return auto / dt / dnu


def coherent_average(data, wgts, lsts, freqs, Navg, bls, antp, lat):
    """
    Coherently average visibilities across time.
    
    Parameters
    ----------
    data : tensor
        Visibility tensor of shape (Nbls, Ntimes, Nfreqs, Npols)
    wgts : tensor
        Visibility weights of shape (-1, Ntimes, -1, -1)
    lsts : tensor
        LST array of time axis [radians]. Assumes uniformly spaced.
    freqs : tensor
        Frequency bins [Hz]
    Navg : int
        Number of time integrations to average
    bls : list
        List of antenna pairs for each baseline
    antp : dict
        Antenna position dictionary
    lat : tuple
        Latitude in degrees of telescope location

    Returns
    -------
    tensor
    """
    # get dLST
    Ngroups = int(np.ceil(len(lsts) / Navg))
    dlst = []
    for i in range(Ngroups):
        _lsts = lsts[i*Navg:(i+1)*Navg]
        dlst.extend(_lsts - _lsts[len(_lsts)//2])
    dlst = torch.as_tensor(dlst)

    # get rephasing term
    bl_vecs = torch.stack([antp[bl[1]] - antp[bl[0]] for bl in bls])
    phs = vis_rephase(dlst, lat, bl_vecs, freqs)[..., None]

    # iterate over lst groups
    avg_d = []
    for i in range(Ngroups):
        idx = slice(i*Navg, (i+1)*Navg)
        wgt = wgts[:, idx]
        wsum = wgt.sum(axis=1, keepdims=True).clip(1e-10)
        avg_d.append(((data[:, idx] / phs[:, idx]) * wgt).sum(axis=1, keepdims=True) / wsum)

    avg_d = torch.cat(avg_d, axis=1)

    return avg_d


def vis_rephase(dlst, lat, blvecs, freqs):
    """
    Generate a rephasing tensor for drift-scan,
    zenith-pointing interferometric visibilities.

    Parameters
    ----------
    dlst : tensor
        Delta-LST in radians to move fringe center
    lat : float
        Earth latitude of telescope in degrees
    blvecs : tensor
        3D baseline vectors in ENU coordinates of shape (Nbls, 3)
    freqs : tensor
        Observing frequenices in Hz
    device : str
        Device to operate one

    Returns
    -------
    tensor
        Rephasing vector to multiply complex visibilities
        of shape (Nbls, Ntimes, Nfreqs)
    """
    dlst, lat = torch.as_tensor(dlst), torch.as_tensor(lat)
    dlst, lat = torch.atleast_1d(dlst), torch.atleast_1d(lat)

    # get zero vector
    zero = torch.tensor([0.], device=dlst.device)

    # get top2eq matrix (1, 3, 3)
    top2eq_mat = _top2eq_m(zero, lat * np.pi / 180)

    # get eq2top matrix (Nlst, 3, 3)
    eq2top_mat = _eq2top_m(-dlst, lat * np.pi / 180)

    # get full rotation matrix (Nlst, 3, 3)
    rot = torch.einsum("...jk,...kl->...jl", eq2top_mat, top2eq_mat)

    # get new s-hat vector (Nlsts, 3)
    s_zenith = torch.tensor([0.0, 0.0, 1.0], device=dlst.device)
    s_prime = torch.einsum("...ij,j->...i", rot, s_zenith)

    # dot bl with difference of pointing vectors to get new u: Zhang, Y. et al. 2018 (Eqn. 22)
    # note that we pre-divided s_diff by c so this is in units of tau.
    s_diff_over_c = (s_prime - s_zenith) / 2.99792458e8
    tau = torch.einsum("...i,ki->...k", s_diff_over_c, blvecs)  # (Nlst, Nbls)

    # get phasor (Nbls, Nlst, Nfreqs)
    phasor = torch.exp(2j * np.pi * freqs * tau.T[..., None])

    return phasor


def _eq2top_m(ha, dec):
    """
    Return the 3x3 matrix converting equatorial coordinates to topocentric
    at the given hour angle (ha) and declination (dec).

    Returned array has the number of ha's or dec's in the first dimension, so is
    shape ``(Nha, 3, 3)``.

    Borrowed from pyuvdata which borrowed from aipy

    Args:
        ha : float or ndarray
        dec: float
    """
    ha = torch.as_tensor(ha)
    dec = torch.ones_like(ha) * dec
    sin_H, cos_H = torch.sin(ha), torch.cos(ha)
    sin_d, cos_d = torch.sin(dec), torch.cos(dec)
    mat = torch.stack(
        [sin_H, cos_H, torch.zeros_like(ha),
         -sin_d * cos_H, sin_d * sin_H, cos_d,
         cos_d * cos_H, -cos_d * sin_H, sin_d]
    )
    mat = mat.reshape(3, 3, -1).moveaxis(2, 0)

    return mat


def _top2eq_m(ha, dec):
    """Return the 3x3 matrix converting topocentric coordinates to equatorial
    at the given hour angle (ha) and declination (dec).

    Returned array has the number of ha's or dec's in the first dimension, so is
    shape ``(Nha, 3, 3)``.

    Slightly changed from aipy to simply write the matrix instead of inverting.
    Borrowed from pyuvdata which borrowed from aipy.

    Args:
        ha : float or ndarray
        dec: float
    """
    ha = torch.as_tensor(ha)
    dec = torch.ones_like(ha) * dec
    sin_H, cos_H = torch.sin(ha), torch.cos(ha)
    sin_d, cos_d = torch.sin(dec), torch.cos(dec)
    mat = torch.stack(
        [sin_H, -cos_H * sin_d, cos_d * cos_H,
         cos_H, sin_d * sin_H, -cos_d * sin_H,
         torch.zeros_like(ha), cos_d, sin_d]
    )
    mat = mat.reshape(3, 3, -1).moveaxis(2, 0)

    return mat


def get_leaf_ants(antpos):
    """return lower leaf of full HERA split-hex array. antpos must be only hera 320 core."""
    antpos = antpos - antpos.mean(0)
    th = -np.pi/6
    _R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    leaf = (antpos[:, 1] < 5) & ((_R @ antpos[:, :2].T).T[:, 0] < 10)
    return leaf



def inpaint_freq_1d(vd, flags, freq_kernel, inv_wgts, method='woodbury', rcond=1e-12, **kwargs):
    """
    Frequency-only inpainting

    Parameters
    ----------
    vd : bayeslim.VisData
    flags : tensor
        Shape (-1, Ntimes, Nfreqs)
    freq_kernel : Kernel
    inv_wgts : tensor
        Shape (-1, Ntimes, Nfreqs)
    """
    # inpaint along freq
    inp_y, mdl = gprlim.models.inpaint_1d(
        freq_kernel, vd.freqs/1e6, vd.data[0,0]*~flags, inv_wgts, flags, dim=-1, method=method, rcond=rcond, **kwargs
    )

    return inp_y, mdl


def inpaint_time_freq_1d(
    vd, flags, time_kernel, freq_kernel, inv_wgts, method='woodbury', rcond=1e-12, noise_mult=100, **kwargs
    ):
    """
    Time 1D inpaint, then freq 1D inpaint


    Parameters
    ----------
    vd : bayeslim.VisData
    flags : tensor
        Shape (-1, Ntimes, Nfreqs)
    time_kernel : Kernel
    freq_kernel : Kernel
    inv_wgts : tensor
        Shape (-1, Ntimes, Nfreqs)
    """
    times = (vd.times - vd.times[0]) * 24 * 3600
    inp_y, mdl = gprlim.models.inpaint_1d(
        time_kernel, times, vd.data[0,0]*~flags, inv_wgts, flags, dim=-2, method=method, rcond=rcond, **kwargs
    )
    fully_flagged = (flags.all(dim=1, keepdim=True)).expand_as(flags)
    inv_wgts[flags & ~fully_flagged] = inv_wgts[~flags].mean() * noise_mult

    inp_y, mdl = gprlim.models.inpaint_1d(
        freq_kernel, vd.freqs/1e6, inp_y, inv_wgts, flags, dim=-1, method=method, rcond=rcond, **kwargs
    )

    return inp_y, mdl


def inpaint_time_freq_2d(
    vd, flags, time_kernel, freq_kernel, inv_wgts, method='cg', cg_tol=1e-3, n_threads=8, **kwargs
    ):
    """
    Joint 2D inpainting

    Parameters
    ----------
    vd : bayeslim.VisData
    flags : tensor
        Shape (-1, Ntimes, Nfreqs)
    time_kernel : Kernel
    freq_kernel : Kernel
    inv_wgts : tensor
        Shape (-1, Ntimes, Nfreqs)
    """
    times = (vd.times - vd.times[0]) * 24 * 3600
    inp_y, mdl = gprlim.models.inpaint_2d(
        time_kernel, freq_kernel, times, vd.freqs/1e6, vd.data[0,0]*~flags, inv_wgts, flags,
        method=method, cg_tol=cg_tol, cg_max_iter=100000, n_threads=n_threads, **kwargs
    )

    return inp_y, mdl

def inpaint_time_freq_2d_freq_1d(
    vd, flags, time_kernel, freq_kernel, inv_wgts, cg_tol=1e-3, rcond=1e-12, n_threads=8, noise_mult=100, **kwargs
    ):
    """
    CG 2D inpaint, then 1D freq inpaint

    Parameters
    ----------
    vd : bayeslim.VisData
    flags : tensor
        Shape (-1, Ntimes, Nfreqs)
    time_kernel : Kernel
    freq_kernel : Kernel
    inv_wgts : tensor
        Shape (-1, Ntimes, Nfreqs)
    """
    # first do joint 2D inpaint
    times = (vd.times - vd.times[0]) * 24 * 3600
    inp_y, mdl = gprlim.models.inpaint_2d(
        time_kernel, freq_kernel, times, vd.freqs/1e6, vd.data[0,0]*~flags, inv_wgts, flags,
        method='cg', cg_tol=cg_tol, cg_max_iter=100000, n_threads=n_threads, **kwargs
    )

    # now use it as prior for freq only inpaint
    inv_wgts = inv_wgts.clone()
    inv_wgts[flags] = inv_wgts[~flags].mean() * noise_mult

    # inpaint along freq
    inp_y, mdl = gprlim.models.inpaint_1d(
        freq_kernel, vd.freqs/1e6, inp_y, inv_wgts, flags, dim=-1, method='woodbury', rcond=rcond, **kwargs
    )

    return inp_y, mdl
