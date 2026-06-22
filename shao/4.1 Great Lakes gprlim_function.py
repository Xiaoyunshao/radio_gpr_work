import numpy as np
import gprlim
from gprlim import models

# (Nbls, Ntimes, Nfreq, Npol)=(1,1600,600,1)

# EoR only
d_eor = np.load("generate_data/eor_data.npz", allow_pickle=True)
eor_data_raw = d_eor["data"]
freqs_eor = d_eor["freqs"]   # unit MHz
times_eor = d_eor["times"]

# EoR + foreground
d_fg = np.load("generate_data/eor_fg_data.npz", allow_pickle=True)
eor_fg_data_raw = d_fg["data"]

# EoR + foreground + noise
d_efn = np.load("generate_data/eor_fg_noise.npz", allow_pickle=True)
eor_fg_noise_raw = d_efn["data"]

# Thermal noise
d_noise = np.load("generate_data/noise.npz", allow_pickle=True)
noise_raw = d_noise["noise"]
noise_var_raw = d_noise["noise_var"]


# Flags
d_flag = np.load("generate_data/flags.npz", allow_pickle=True)
flags_raw = d_flag["flags"]

# Check original shapes
print("Original shapes")
print("eor_data_raw      :", eor_data_raw.shape)
print("eor_fg_data_raw   :", eor_fg_data_raw.shape)
print("eor_fg_noise_raw  :", eor_fg_noise_raw.shape)
print("noise_raw         :", noise_raw.shape)
print("noise_var_raw     :", noise_var_raw.shape)
print("flags_raw         :", flags_raw.shape)




# (1, 1600, 600, 1) -> (1, 20, 180, 1)

choose7=1600
eor_data_raw      = eor_data_raw[:, :choose7, :, :]
eor_fg_data_raw   = eor_fg_data_raw[:, :choose7, :, :]
eor_fg_noise_raw  = eor_fg_noise_raw[:, :choose7, :, :]
noise_raw         = noise_raw[:, :choose7, :, :]
noise_var_raw     = noise_var_raw[:, :choose7, :, :]
flags_raw         = flags_raw[:, :choose7, :, :]

# 如果频率数组长度是 600，也同步截取
freqs_eor = freqs_eor[:]

# 如果 times_eor 长度是 1600，也同步截取
times_eor = times_eor[:choose7]

print("\nTrimmed shapes")
print("eor_data_raw      :", eor_data_raw.shape)
print("eor_fg_data_raw   :", eor_fg_data_raw.shape)
print("eor_fg_noise_raw  :", eor_fg_noise_raw.shape)
print("noise_raw         :", noise_raw.shape)
print("noise_var_raw     :", noise_var_raw.shape)
print("flags_raw         :", flags_raw.shape)



# convert: (Nbls, Ntimes, Nfreq, Npol)
#       -> (Npol, Nbls，Ntimes,  Nfreq)

eor_data     = np.transpose(eor_data_raw,     (3, 0, 1, 2))
eor_fg_data  = np.transpose(eor_fg_data_raw,  (3, 0, 1, 2))
eor_fg_noise = np.transpose(eor_fg_noise_raw, (3, 0, 1, 2))
noise         = np.transpose(noise_raw,        (3, 0, 1, 2))
noise_var     = np.transpose(noise_var_raw,    (3, 0, 1, 2))
flags         = np.transpose(flags_raw,        (3, 0, 1, 2))

# Check converted shapes
print("\nConverted shapes")
print("eor_data      :", eor_data.shape)
print("eor_fg_data   :", eor_fg_data.shape)
print("eor_fg_noise  :", eor_fg_noise.shape)
print("noise         :", noise.shape)
print("noise_var     :", noise_var.shape)
print("flags         :", flags.shape)



import numpy as np
from matplotlib import pyplot as plt
#d = np.load("/Users/alliswell48/simpleQE/notebooks/generate_datat/gpr_inpaint_input.npz")

#fchans = [(20, 21), (40, 41), (80, 81), (95, 96), (120, 121),[170,171]]

data0 = eor_fg_noise
inp_flags0 = flags 
inv_wgts0 = noise_var
freqs0 = freqs_eor
fg0=eor_fg_data -eor_data 
noise0=noise



inv_wgts0 = inv_wgts0.copy()
inv_wgts0[inp_flags0] = 2e10     # ≫ outputscale(1e3),又 ≪ 1e10


print(data0.shape)
print(inp_flags0.shape)
print(inv_wgts0.shape)
print(freqs0.shape)
print(fg0.shape)
print(noise0.shape)



import torch

# numpy → torch
data = torch.tensor(data0)
inp_flags = torch.tensor(inp_flags0)
inv_wgts = torch.tensor(inv_wgts0)
freqs = torch.tensor(freqs0)

# 统一 dtype
freqs = freqs.double()               # Hz, float32
data = data.to(torch.complex128)     # complex64
inv_wgts = inv_wgts.double()         # float32
inp_flags = inp_flags.bool()        # bool

# 检查
print("data.shape =", data.shape)
print("inp_flags.shape =", inp_flags.shape)
print("inv_wgts.shape =", inv_wgts.shape)
print("freqs.shape =", freqs.shape)

print("data.dtype =", data.dtype)
print("inp_flags.dtype =", inp_flags.dtype)
print("inv_wgts.dtype =", inv_wgts.dtype)
print("freqs.dtype =", freqs.dtype)



# per band, per red-group, per-pol inpainting
def gpr_inpaint(
    data,
    inv_wgts,
    inp_flags,
    freqs,
    bl_len,
    kernA_var=0.01, #add 137 
    center_y=True,
    Ndeg=1,
    horizon_buffer=150,
    min_horizon_dly=150,
    kernA='sinc',  #add 137  ,最外层 
    kernB='sinc',  #中层
    kernB_buffer=200,
    kernB_max_dly=1000,
    kernB_rel_var=0.01,
    kernC='sinc',   # 最内层
    kernC_buffer=500,
    kernC_max_dly=1200,
    kernC_rel_var=0.1,
):
    """
    Per band, per red-group, per-polarization GPR inpainting

    Parameters
    ----------
    data : tensor, complex
        Complex visibility tensor (Nbls, Ntimes, Nfreqs)
    inv_wgts : tensor, float
        Inverse data weights (i.e. noise variance)
    inp_flags : tensor, bool
        Boolean mask indicating where to inpaint
    freqs : tensor
        Frequencies [Hz]
    bl_len : tensor
        Baseline ENU length [meters]
    kernA_var : float
        Variance of kernA
    center_y : bool
        If True, subtract data mean before inpainting
    horizon_buffer : float
        Baseline horizon buffer [ns]
    min_horizon_dly : float
        Minimum threshold for delay of baseline horizon [ns]
    kernB_buffer : float
        Buffer [ns] of kernB relative to kernA
    kernB_max_dly : float
        Maximum delay of kernB [ns]
    kernB_rel_var : float
        Relative amplitude (variance) of kernB relative to kernA
    kernC_buffer : float
        Buffer [ns] of kernC relative to kernB
    kernC_max_dly : float
        Maximum delay of kernC [ns]
    kernC_rel_var : float
        Relative amplitude (variance) of kernC relative to kernB

    Returns
    -------
    inp_y : tensor
        Copy of data with inpainted model in flags
    mdl : tensor
        Inpaint model
    """
    
    
    
    Nbls, Ntimes, Nfreqs = data.shape[-3:]  #忽律pol 极化维度
    train_y = torch.vstack([data.real.reshape(-1, Nfreqs), data.imag.reshape(-1, Nfreqs)])

    train_xt = freqs[:, None]  # [MHz] Size([3，1])
    #freqs[:]=>freqs[:, None].shape 等价于freqs.Size([3])=>freqs.Size([3，1])

    train_x = train_xt.expand((1, Nfreqs, 1)) # Size([3，1]) =>Size([1, 3, 1])
    inp_flags = torch.vstack([inp_flags.reshape(-1, Nfreqs), inp_flags.reshape(-1, Nfreqs)])
    train_wgts = torch.vstack([inv_wgts.reshape(-1, Nfreqs), inv_wgts.reshape(-1, Nfreqs)]) / 2  # real/imag

    # tau [ns] to sinc length-scale [MHz]
    tau2ls = lambda tau: 1e3 / 2 / tau  # ns
    bl_tau = max([bl_len / 2.99792e8 * 1e9 + horizon_buffer, min_horizon_dly])  # ns

    # get mean and covar model of 3-sinc mixture




    mean, covar = models.multi_kernel_mixture(
        mean_fix_constant=True,
        mean_set_constant=0,
        kern2=kernA,
        kern2_set_lengthscale=tau2ls(bl_tau),
        kern2_set_outputscale=kernA_var,
        kern1=kernB,
        kern1_set_lengthscale=tau2ls(min([bl_tau + kernB_buffer, kernB_max_dly])),
        kern1_set_outputscale=kernB_rel_var,
        kern0=kernC,
        kern0_set_lengthscale=tau2ls(min([bl_tau + kernC_buffer, kernC_max_dly])),
        kern0_set_outputscale=kernC_rel_var,
    )


    # get GPModel
    model, y_offset = models.fixednoise_gp_1d(
        train_x, train_y, mean, covar, inv_wgts=train_wgts, center_y=center_y, Ndeg=Ndeg,

    )
    y_offset = y_offset if y_offset is not None else 0



    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.1,
        #max_iter=1
    )

    loss, opt = models.optimize_kernel(
        model,
        Niter=20,  #12,30
        opt=optimizer,
        thresh=None,
        #thresh=1e-5
    )


    for name, param in model.named_parameters():
        print(name, param.shape, param.requires_grad)

    print()
    print('Consrain parameter:')
    for name, constraint in model.named_constraints():
        print(name, constraint)



    np.save(
        "output/gpr_loss.npy",
        np.array([l.item() for l in loss])
    )

    print("Saved gpr_loss.npy")

    # get inpaint model and inpainted data
    with torch.no_grad():
        inp_y, mdl = model.inpaint(inp_flags, y_offset=y_offset, to_complex=True)
    inp_y = inp_y.reshape(-1, Ntimes, Nfreqs)

    # inp_y.reshape(-1, 2, 3) =》 把 GP 内部的 batch 形式重新恢复成（ Nbls,Ntimes, Nfreq)形式 
    #=>之后也可再自行加1维,变成 # shape = (Npol, Nbls, Ntimes, Nfreq)
    #                                 = (1,    1,    2,      3)

    print(mdl)
    mdl = mdl.reshape(-1, Ntimes, Nfreqs)  #同理
    #mdl inpaint 所有点
    #inp_y inpaint flag点
    return inp_y, mdl, model, y_offset

# Per-band, per-redundant-group, per-polarization inpainting example

# data shape: (Npol, Nbls, Ntimes, Nfreq)
# here: (1, 1, Ntimes, Nfreq)



inp_y, mdl, model, y_offset = gpr_inpaint(
    data,
    inv_wgts,
    inp_flags,
    freqs,
    bl_len = 14.6,   # HERA baseline length [m]
    kernA_var=0.01,
    center_y=True,
    Ndeg=1,
    horizon_buffer=150,
    min_horizon_dly=150,
    kernA="sinc",          # outermost kernel
    kernB="sinc",          # middle kernel
    kernB_buffer=200,
    kernB_max_dly=1000,
    kernB_rel_var=0.01,
    kernC="sinc",          # innermost kernel
    kernC_buffer=500,
    kernC_max_dly=1200,
    kernC_rel_var=0.1,
)

# 1. 保存整个模型
torch.save(model, "output/gpr_model.pt")

# 2. 保存数值结果
np.savez(
    "output/gpr_outputs.npz",
    inp_y=inp_y.cpu().numpy(),
    mdl=mdl.cpu().numpy(),
    y_offset=np.asarray(y_offset)
)

print("Saved gpr_model.pt")
print("Saved gpr_outputs.npz")

