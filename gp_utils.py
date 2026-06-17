import numpy as np
import torch


def load_uvdata(dfile, bls=None, freq_chans=None, time_ints=None, pols=None, inflate_by_red=False, device=None, dtype=None):
    """
    Load a pyuvdata UVData and convert data to torch tensor.

    Parameters
    ----------
    dfile : str
        Filepath to UVH5 dataset.
    """
    from pyuvdata import UVData

    # load uvdata
    uvd = UVData()
    uvd.read(dfile, bls=bls, freq_chans=freq_chans, polarizations=pols)

    if time_ints is not None:
        uvd.select(times=np.unique(uvd.time_array)[time_ints])

    # get antenna dictionary
    antpos = uvd.telescope.get_enu_antpos()
    ants = uvd.telescope.antenna_numbers
    antp = dict(zip(ants.tolist(), torch.as_tensor(antpos, device=device, dtype=dtype)))

    # inflate to redundancies
    if inflate_by_red:
        uvd.inflate_by_redundancy()

    # unravel to (Nbls, Ntimes, Nfreqs, Npols) and convert to tensor
    data = uvd.data_array.reshape(uvd.Ntimes, uvd.Nbls, uvd.Nfreqs, uvd.Npols)
    data = np.moveaxis(data, 0, 1)

    # convert to torch tensor
    data = torch.as_tensor(data, device=device, dtype=dtype)

    # get redundancies
    from hera_cal.redcal import get_pos_reds
    reds = get_pos_reds(antp, include_autos=True)

    # get metadata
    meta = {
        'antp': antp,
        'bls': uvd.get_antpairs(),
        'freqs': torch.as_tensor(uvd.freq_array, device=device, dtype=dtype),
        'times': torch.as_tensor(np.unique(uvd.time_array), device=device, dtype=dtype),
        'lsts': torch.as_tensor(np.unique(np.unwrap(uvd.lst_array)), device=device, dtype=dtype),
        'reds': reds,
        'lat': uvd.telescope.location_lat_lon_alt_degrees[0],
        'lon': uvd.telescope.location_lat_lon_alt_degrees[1],
    }

    return data, meta


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

