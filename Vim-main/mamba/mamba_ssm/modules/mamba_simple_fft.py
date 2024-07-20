# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from turtle import pd
from typing import Optional  #typing: 用于类型注解，提供了 Optional 类型。

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat #einops: 提供简单而强大的Numpy/PyTorch/TF/JAX操作，用于重塑张量。

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from torch.nn.functional import silu


'''FFTLayer: 这是一个简单的FFT（快速傅里叶变换）层，进行双重傅里叶变换，并返回结果的实部。'''
class FFTLayer(nn.Module):
    def __init__(self):
        super().__init__()  #调用父类的构造方法
    @torch.cuda.amp.autocast(enabled=False)   #enabled=False 表示禁用自动混合精度。
    def forward(self, x):
        return torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real  #对倒数第一第二维做傅里叶变换（二维傅里叶变换），因为这两维一般对应图像的高度和宽度



class Mamba_FFT(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
#时间常数投影（dt projection）相关参数。
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,

        conv_bias=True, #卷积层的偏置参数。
        bias=False,  #线性层的偏置参数。
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",  #双向Mamba类型，默认为 "none"。
        high_freq=0.9, #高频阈值
        low_freq=0.1  #低频阈值
    ):
        factory_kwargs = {"device": device, "dtype": dtype}

        # 调用父类 nn.Module 的初始化方法
        super().__init__()

        # 初始化子类特有的属性
        self.high_freq = high_freq
        self.low_freq = low_freq
        self.fftlayer = FFTLayer()  # FFT层的初始化。FFTLayer 用于对输入进行傅里叶变换
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv   #卷积核的大小，这个值决定了一维卷积操作中使用的卷积核的大小
        self.expand = expand  #表示扩展因子，用于增加内部特征维度。这个值通常用于扩展输入特征的维度。

        '''
        知识补充：
        经卷积后的矩阵尺寸大小计算公式为：
        N = (W - F + 2P) / S + 1
        
        1. 输入图片大小 W * W
        2. Filter大小F * F
        3. 步长 S
        4. padding的像素数 P     padding 一圈是1，半圈是0.5
        '''

        self.d_inner = int(self.expand * self.d_model)   #计算扩展后的内部特征维度
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  #表示时间常数投影的秩。这个值用于时间常数投影层的计算。
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type


        # 定义线性投影层
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)  #将输入投影到更高维度。

        # 定义一维卷积层，用于处理输入特征。  （用于正向特征处理）
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,   #卷积核的大小
            groups=self.d_inner,  #分组卷积的组数
            padding=d_conv - 1,  #填充的大小
            **factory_kwargs,   #包括设备和数据类型的其他参数。
        )

        self.activation = "silu"
        self.act = nn.SiLU()  #使用 SiLU（Sigmoid Linear Unit）作为激活函数

        self.x_proj = nn.Linear( #投影层，用于生成时间常数和状态参数。
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)  #时间常数投影层。


#初始化时间常数投影参数
        # 初始化特殊的 dt（时间常数） 投影以在初始化时保留方差
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 初始化时间常数投影层 (self.dt_proj) 的偏置 (bias) 参数，使其在特定的范围内（dt_min 到 dt_max 之间）。这有助于确保模型在训练开始时的稳定性和合理性。
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # softplus的反面: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # 我们的初始化会将所有 Linear.bias 设置为零，需要将此标记为 _no_reinit
        self.dt_proj.bias._no_reinit = True





        # 初始化 S4D（Skip-Shift-Scale）的参数 A_log 和 D

#S4D 是 "State Space Sequence Model for Speech Denoising" 的缩写。
#S4D 是一种用于序列建模的状态空间模型（State Space Model，SSM），特别应用于语音去噪任务
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # 初始化双向 Mamba 的参数 A_b_log 和 D_b
        assert bimamba_type == "v2"  #一个断言语句，用于检查 bimamba_type 是否为 "v2"。

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True 




        self.conv1d_b = nn.Conv1d(  #一维卷积层，用于处理输入特征。  （用于反向特征处理）
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )





        self.x_proj_b = nn.Linear(  # 线性投影层，用于生成时间常数和状态参数。
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # 时间常数投影层，用于生成时间常数
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs) #时间常数投影层。

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True


        #一个输出投影层，用于将高维度特征还原到原始维度。
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # add by HJ
        self.fft_conv = nn.Conv1d(self.d_inner*2, self.d_inner*2, 5, 1, 2)   #一个一维卷积层，用于在频域中处理特征数据。
        #输入通道，输出通道，卷积核大小，步长，填充（padding）
        self.gate_mapping = nn.Linear(self.d_model*128, 3)   #一个线性层，用于映射和生成门控分数
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((self.d_model, 2048))   #自适应平均池化层，用于将输入特征的空间维度调整到指定大小。
        self.elu = nn.ELU()  #使用ELU（Exponential Linear Unit）作为激活函数。用于引入非线性变换



    '''
    这个 forward 函数实现了一个复杂的前向传播过程，涉及到多种操作，包括线性变换、卷积操作、快速傅里叶变换（FFT）、门控机制以及状态更新等。
    '''
    def forward(self, hidden_states, inference_params=None, No_block=-1, SpectralGatingBlocks=None, GateModules=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states  #返回：与hidden_states形状相同
        """

        '''1. 输入处理和初始化'''
        batch, seqlen, dim = hidden_states.shape  #批次大小， 序列长度， 特征维度
        #处理了推理过程中的状态缓存
        conv_state, ssm_state = None, None  #



        '''2. 处理推理过程中的状态缓存'''
        if inference_params is not None:  #检查是否提供了推理参数 inference_params,如果提供了推理参数，表示当前处于推理阶段，需要处理状态缓存。
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:  #seqlen_offset 通常表示当前处理的序列在整个序列中的偏移量，用于管理状态更新。
                # The states are updated inplace
                #step 方法使用当前的 hidden_states、conv_state 和 ssm_state 进行推理，并更新状态。
                #step 方法返回三个值：输出 out、更新后的 conv_state 和 ssm_state。
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out  #直接返回输出 out，结束当前前向传播。

        # We do matmul and transpose BLH -> HBL at the same time


        '''3. 对输入的隐藏状态进行线性变换和重新排列'''
        # (d_inner * 2, d_model) @ (d_model, b*l) -> (d_inner * 2, b*l)
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",  # (d_inner * 2, b*l) -> (b, d_inner * 2, l)
            l=seqlen,
        )
        # d: 特征维度(d_model):即每个时间步的数据包含的特征数量
        # b: 批次大小(batch_size)，
        # l: 序列长度(sequence_length):每个输入序列包含的时间步的数量

        # self.in_proj.bias 是一个偏置向量，其形状为 (d_inner * 2,)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype),
                                "d -> d 1")  # "d -> d 1") 将偏置向量的形状从 (d_inner * 2,) 转换为 (d_inner * 2, 1)
            # （通过广播机制）将偏置加到线性变换的结果上。

        ##最终xz在双向传播前的形式为（b, d_inner * 2, l)
        #扫描是按照时间步（sequence length）方向进行的，也就是在序列长度（seqlen）维度上进行的。这样可以处理每一个时间步的数据，逐步构建输出序列。
        #每一个时间步并不是整个序列长度，而是序列中的一个特定位置。序列长度表示整个输入序列的长度，而时间步指的是在序列中某一个具体的位置。
        # 例如，如果序列长度是10，那么时间步可以是1到10之间的任何一个整数。

        '''4. 计算参数 A 和 A_b'''
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)      将A_log指数运算后取负值



        # In the backward pass we write dx and dz next to each other to avoid torch.cat


        '''5. 快速路径和双向 Mamba 处理'''
        '''5.1. 使用快速路径和双向 Mamba 类型 v2'''
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states

            '''双向扫描策略'''
            if self.bimamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())   #初始化反向的参数，exp()指数函数会将每个元素 x 转换为 e^x。


                '''mamba_inner_fn_no_out_proj 函数对输入 xz 进行卷积操作
                其中 out 是正向计算的结果
                out_b 是对 xz 进行翻转后（xz.flip([-1])）计算得到的反向结果
                '''

                '''xz 是输入张量，其形状为 (batch, dim, seqlen)。其中：
                
                batch 是批次大小。
                dim 是特征维度。
                seqlen 是序列长度。
                '''
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )

                '''这部分代码对翻转后的 xz 输入进行单向扫描。
                flip([-1]) 操作翻转了输入的最后一个维度（即序列维度），这样可以实现对输入的反向处理'''
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
                # add by HJ
                # fft_res = self.fftlayer(rearrange(out + out_b.flip([-1]), "b d l -> b l d"))
                # out = F.linear(fft_res + rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                # original
                # ################################## add by HJ ##################################


                '''5.2. 快速傅里叶变换（FFT）'''
                #将xz张量沿着最后一个维度（dim=1）分割成两部分，x和z
                x, z = xz.chunk(2, dim=1) # (1, 8, 307200)
                #这里假设xz的形状是(B, D, L)，其中D是2，因此分割后x和z的形状将是(B, 1, L)
                x_fft = torch.fft.fftn(x, dim=(-2, -1))  #对倒数第二个和最后一个维度进行二维FFT   最后两个维度为长度和高度
                x_fft_real = x_fft.real
                x_fft_imag = x_fft.imag
                x_fft = torch.cat([x_fft_real, x_fft_imag],dim = 1)
                #这行代码将实部和虚部沿第二个维度（dim=1）拼接起来。拼接后的x_fft将包含实部和虚部信息，形状变为(B, 2, L)

                ###### 对 x_fft 进行处理  我觉得就是Wf（）
                if SpectralGatingBlocks is not None:
                    x_fft = self.act(x_fft * SpectralGatingBlocks.to('cuda'))
                else:
                    x_fft = self.act(self.fft_conv(x_fft))

                x_fft_real , x_fft_imag= torch.chunk(x_fft, 2, dim = 1)  #分割实部和虚部
                x_fft = torch.complex(x_fft_real, x_fft_imag)  #将实部和虚部转建立一个复数张量

                '''5.3.频率滤波和逆傅里叶变换（IFFT）'''
                # 根据No_block的值应用不同的滤波器
                if No_block == 0:  #低频滤波器：保留低频成分
                    filter_threshold = self.low_freq  #低频阈值为0.1
                    abs_x_fft = torch.abs(x_fft)  #对FFT变换后的复数信号 x_fft 取绝对值，得到其幅度频谱。
                    high_pass_filter = abs_x_fft < filter_threshold
                    x_fft = x_fft * high_pass_filter
                elif No_block == 1:  #带通滤波器：保留特定频率范围内的成分
                    low_filter_threshold = self.low_freq  #低频阈值
                    high_filter_threshold = self.high_freq  #高频阈值
                    abs_x_fft = torch.abs(x_fft)
                    #创造一个布尔掩码，用于选择性地保留特定频率范围(0.1~0.9)内的成分。
                    band_pass_filter = (abs_x_fft > low_filter_threshold) & (abs_x_fft < high_filter_threshold)
                    x_fft = x_fft * band_pass_filter

                elif No_block == 2:  #高频滤波器：保留高频成分
                    filter_threshold = self.high_freq
                    abs_x_fft = torch.abs(x_fft)
                    high_pass_filter = abs_x_fft > filter_threshold
                    x_fft = x_fft * high_pass_filter
                ## 执行逆快速傅里叶变换
                x_fft_out = torch.fft.ifftn(x_fft, dim = (-2,-1))  #对倒数第二个和最后一个维度进行IFFT，将频域信息转化为时域信息



                ''' 激活和池化（对每个通道上的所有元素进行最大池化，只保留一个最大值）'''
                z_out = F.max_pool1d(self.act(z), kernel_size=z.shape[-1]) # avg_pool1d


                # 归一化
                out_f = F.layer_norm(torch.abs(x_fft_out * z_out),  normalized_shape=(x_fft_out.shape[-1], ))
                ################################## end ##################################

                '''5.4. 添加门控模块'''
                if GateModules is not None:

                    #设备设置：将 GateModules 中的每个线性层（gate_fc1, gate_fc2, gate_fc3）移动到与 hidden_states 相同的设备上。
                    device = hidden_states.device
                    gate_fc1, gate_fc2, gate_fc3 = GateModules
                    gate_fc1 = gate_fc1.to(device)
                    gate_fc2 = gate_fc2.to(device)
                    gate_fc3 = gate_fc3.to(device)

                    #自适应平均池化：对 hidden_states 进行自适应平均池化，并将其转置（将hidden_states 的最后两个维度交换）
                    gate_logits = self.elu(gate_fc3(self.elu(gate_fc2(self.elu(gate_fc1(self.adaptive_avg_pool(hidden_states.transpose(-1, -2))))))))
                    gate_score = self.gate_mapping(torch.reshape(gate_logits, (batch, -1)))

                    # add here
                    # gate_score = F.softmax(gate_score, dim=-1)
                    # gate_score = torch.sigmoid(gate_score)
                    #预测与三个特征相对应的三个比例
                    score_out, score_out_b, score_out_f = gate_score[:,0], gate_score[:,1], gate_score[:,2]
                    out *= score_out[:, None, None]
                    out_b *= score_out_b[:, None, None]
                    out_f *= score_out_f[:, None, None]

                #线性层进行投影并输出
                out = F.linear(rearrange(out + out_b.flip([-1]) + out_f, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)

                '''mamba类型不是v2'''
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )


            '''6.非快速路径处理'''

        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )
            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out   #最终的输出




    '''定义了一个 step 函数，用于在推理过程中对模型进行单步更新操作。该函数包含了卷积步骤（Conv step）和状态空间模型步骤（SSM step），并处理了相应的状态更新'''
    def step(self, hidden_states, conv_state, ssm_state):

        '''函数签名和初始检查'''
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        '''卷积步骤（Conv step）'''
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )


        '''线性投影和参数计算'''
        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        '''状态空间模型步骤（SSM step）'''
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )


        '''输出结果'''
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state


    '''定义了 allocate_inference_cache 函数，用于在推理（inference）过程中为模型分配缓存状态。
        该函数主要负责初始化卷积状态和状态空间模型状态的张量，并返回这些张量以供推理过程中使用。'''

    '''函数签名和设备设置'''
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device

        '''卷积状态（conv_state）初始化'''
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        '''状态空间模型状态（ssm_state）初始化'''
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        '''返回初始化的状态张量'''
        return conv_state, ssm_state


    '''这段代码定义了 _get_states_from_cache 函数，用于在推理过程中从缓存中获取或初始化模型的状态，包括卷积状态（conv_state）和状态空间模型状态（ssm_state）。
        该函数确保每一层都有对应的状态缓存，如果缓存中没有对应的状态，则初始化状态并存储在缓存中。如果缓存中已有对应的状态，则直接获取并在必要时重置状态'''

    '''函数签名和初始检查'''
    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None

        '''检查缓存中是否已有对应的状态'''
        if self.layer_idx not in inference_params.key_value_memory_dict:
            '''如果缓存中没有对应的状态，则初始化状态'''
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)

            '''如果缓存中已有对应的状态，则直接获取状态'''
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            '''重置状态（如果需要）'''
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()

        '''返回状态'''
        return conv_state, ssm_state




'''用于包装一个混合器类（mixer_cls），并结合 LayerNorm 或 RMSNorm 进行归一化处理以及残差连接'''
class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"


    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual


    '''这个 allocate_inference_cache 函数主要是用于在推理（inference）过程中分配缓存'''
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
