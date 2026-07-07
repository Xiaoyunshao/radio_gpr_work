# utils_inpaint.py

import numpy as np
import torch

import sys
sys.path.append('/Users/alliswell48/radio_gpr_work/shao/gprlim')
import gprlim

def make_flags(
    flag_types,
    Ntimes,
    Nfreqs,
    device="cpu",
    time_ints=None,
    freq_chans=None,
    real_flag_path="gpr_data/version2/zen.h6c_idr3.2459951.flag_waterfall_round2.npz",
    real_flag_key="flags",
    batch_size=1,
):
    """
    Construct artificial and/or real flags.

    Parameters
    ----------
    flag_types : list[str]
        List of flag types to apply. Available options are:

        - 'real'
        - 'narrowband_full_time'
        - 'wideband_full_time'
        - 'ultraband_short_time'
        - 'fullband_short_time'
        - 'narrowband_short_time'
        - 'narrowband_intermittent'

    Ntimes : int
        Number of time integrations.

    Nfreqs : int
        Number of frequency channels.

    device : str or torch.device
        Torch device.

    time_ints : slice or array-like, optional
        Time indices used when loading real flags.

    freq_chans : slice or array-like, optional
        Frequency indices used when loading real flags.

    real_flag_path : str
        Path to the real flag .npz file.

    real_flag_key : str
        Key name inside the .npz file.

    batch_size : int
        First dimension of output flags. Default is 1.

    Returns
    -------
    flags : torch.BoolTensor
        Boolean flag array with shape (batch_size, Ntimes, Nfreqs).
        True means flagged, False means unflagged.
    """

    flags = torch.zeros(
        (batch_size, Ntimes, Nfreqs),
        dtype=torch.bool,
        device=device,
    )

    # --------------------------------------------------
    # 1. Real flags
    # --------------------------------------------------
    if "real" in flag_types:
        if time_ints is None or freq_chans is None:
            raise ValueError(
                "When using flag_types=['real'], you must provide time_ints and freq_chans."
            )

        flg = np.load(real_flag_path)[real_flag_key][time_ints][:, freq_chans, 0]
        flg = torch.as_tensor(flg, dtype=torch.bool, device=device)

        # flg shape: (Ntimes, Nfreqs)
        # flags shape: (batch_size, Ntimes, Nfreqs)
        flags = flags + flg
        
        


    # --------------------------------------------------
    # 2. Narrowband, full time
    # --------------------------------------------------
    if "narrowband_full_time" in flag_types:
        # 1 channel wide, all times
        torch.manual_seed(100)

        Nflag = 15
        rand = torch.randint(
            low=0,
            high=Nfreqs,
            size=(Nflag,),
            device=device,
        )

        flags[:, :, rand] = True

    # --------------------------------------------------
    # 3. Wideband, full time
    # --------------------------------------------------
    if "wideband_full_time" in flag_types:
        # 10-20 channels wide, all times
        torch.manual_seed(101)

        Nflag = 4

        for _ in range(Nflag):
            f_width = torch.randint(
                low=10,
                high=21,
                size=(1,),
                device=device,
            ).item()

            f0 = torch.randint(
                low=0,
                high=Nfreqs - f_width + 1,
                size=(1,),
                device=device,
            ).item()

            f1 = f0 + f_width
            flags[:, :, f0:f1] = True

    # --------------------------------------------------
    # 4. Ultraband, short time
    # --------------------------------------------------
    if "ultraband_short_time" in flag_types:
        # 20-30 channels wide, 1-4 time integrations
        torch.manual_seed(102)

        Nflag = 10

        for _ in range(Nflag):
            f_width = torch.randint(
                low=20,
                high=31,
                size=(1,),
                device=device,
            ).item()

            t_width = torch.randint(
                low=1,
                high=5,
                size=(1,),
                device=device,
            ).item()

            f0 = torch.randint(
                low=0,
                high=Nfreqs - f_width + 1,
                size=(1,),
                device=device,
            ).item()

            t0 = torch.randint(
                low=0,
                high=Ntimes - t_width + 1,
                size=(1,),
                device=device,
            ).item()

            f1 = f0 + f_width
            t1 = t0 + t_width

            flags[:, t0:t1, f0:f1] = True

    # --------------------------------------------------
    # 5. Fullband, short time
    # --------------------------------------------------
    if "fullband_short_time" in flag_types:
        # all channels wide, 1-3 time integrations
        torch.manual_seed(104)

        Nflag = 5

        for _ in range(Nflag):
            t_width = torch.randint(
                low=1,
                high=4,
                size=(1,),
                device=device,
            ).item()

            t0 = torch.randint(
                low=0,
                high=Ntimes - t_width + 1,
                size=(1,),
                device=device,
            ).item()

            t1 = t0 + t_width
            flags[:, t0:t1, :] = True

    # --------------------------------------------------
    # 6. Narrowband, short time
    # --------------------------------------------------
    if "narrowband_short_time" in flag_types:
        # 1-3 channels wide, 3-10 time integrations
        torch.manual_seed(103)

        Nflag = 20

        for _ in range(Nflag):
            f_width = torch.randint(
                low=1,
                high=4,
                size=(1,),
                device=device,
            ).item()

            t_width = torch.randint(
                low=3,
                high=11,
                size=(1,),
                device=device,
            ).item()

            f0 = torch.randint(
                low=0,
                high=Nfreqs - f_width + 1,
                size=(1,),
                device=device,
            ).item()

            t0 = torch.randint(
                low=0,
                high=Ntimes - t_width + 1,
                size=(1,),
                device=device,
            ).item()

            f1 = f0 + f_width
            t1 = t0 + t_width

            flags[:, t0:t1, f0:f1] = True

    # --------------------------------------------------
    # 7. Narrowband, intermittent
    # --------------------------------------------------
    if "narrowband_intermittent" in flag_types:
        # 1-3 channels wide, repeated intermittent chunks in time
        torch.manual_seed(105)

        Nflag = 5
        chunk_len = 61
        n_chunks = 3

        # Divide the time axis into n_chunks large regions.
        # One chunk is placed randomly inside each region.
        edges = torch.linspace(
            0,
            Ntimes,
            n_chunks + 1,
            device=device,
        ).long()

        for _ in range(Nflag):
            f_width = torch.randint(
                low=1,
                high=4,
                size=(1,),
                device=device,
            ).item()

            f0 = torch.randint(
                low=0,
                high=Nfreqs - f_width + 1,
                size=(1,),
                device=device,
            ).item()

            f1 = f0 + f_width

            for i in range(n_chunks):
                region_start = edges[i].item()
                region_end = edges[i + 1].item()

                max_t0 = region_end - chunk_len

                if max_t0 <= region_start:
                    continue

                t0 = torch.randint(
                    low=region_start,
                    high=max_t0 + 1,
                    size=(1,),
                    device=device,
                ).item()

                t1 = t0 + chunk_len

                flags[:, t0:t1, f0:f1] = True

    return flags






def make_time_kernel(
    kernel_type,
    freqs,
    bl_vec,
    lat,
    default_buf=1.0,
    default_min_hw=0.5,
):
    """
    Make time covariance kernel.

    Parameters
    ----------
    kernel_type : str
        Either "tophat" or "custom".

    freqs : array-like
        Frequencies in MHz.

    bl_vec : tensor or array-like
        Baseline vector.

    lat : float
        Telescope latitude.

    default_buf : float
        Fringe-rate buffer.

    default_min_hw : float
        Minimum fringe-rate half-width.

    Returns
    -------
    time_kernel : gpytorch kernel
        Time covariance kernel.
    """

    if kernel_type == "tophat":

        time_kernel = gprlim.kernels.default_time_kernel(
            freqs * 1e6,
            bl_vec,
            lat,
            ml_scale=2e0,
            fz_scale=1e-3,
            fr_scale=1e3,
            buffer=default_buf,
            min_hw=default_min_hw,
            only_global_amp=True,
        )

    elif kernel_type == "custom":

        time_kernel = gprlim.kernels.default_time_kernel(
            freqs * 1e6,
            bl_vec,
            lat,
            ml_scale=1e3,
            fz_scale=3e-1,
            fr_scale=1e-4,
            buffer=default_buf,
            min_hw=default_min_hw,
            only_global_amp=True,
        )

    else:
        raise ValueError(
            f"Unknown kernel_type={kernel_type}. "
            "Available options are 'tophat' and 'custom'."
        )

    return time_kernel


def make_freq_kernel(
    kernel_type,
    bl_vec,
    default_buffer=100.0,
    default_min_delay=50.0,
):
    """
    Make frequency covariance kernel.

    Parameters
    ----------
    kernel_type : str
        Either "tophat" or "custom".

    bl_vec : tensor or array-like
        Baseline vector.

    default_buffer : float
        Supra-horizon buffer in delay space.

    default_min_delay : float
        Minimum horizon delay.

    Returns
    -------
    freq_kernel : gpytorch kernel
        Frequency covariance kernel.
    """

    if kernel_type == "tophat":

        freq_kernel = gprlim.kernels.default_freq_kernel(
            bl_vec,
            only_global_amp=True,
            ml_scale=3e0,
            pf_scale=1e0,
            wd_scale=1e0,
            lk_scale=1e3,
            real=True,
            buffer=default_buffer,
            min_delay=default_min_delay,
        )

    elif kernel_type == "custom":

        freq_kernel = gprlim.kernels.default_freq_kernel(
            bl_vec,
            only_global_amp=True,
            ml_scale=1e3,
            pf_scale=1e-1,
            wd_scale=1e-3,
            lk_scale=1e-3,
            real=True,
            lk_kern="twinrbf",
            buffer=default_buffer,
            min_delay=default_min_delay,
        )

    else:
        raise ValueError(
            f"Unknown kernel_type={kernel_type}. "
            "Available options are 'tophat' and 'custom'."
        )

    return freq_kernel

