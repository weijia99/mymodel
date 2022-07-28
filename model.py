# 构建glow模块
import torch
import torch.nn as nn
from math import log, pi, exp
from torch.nn import functional as F
import numpy as np
from scipy import linalg as la

# 进行求解矩阵的

logabs = lambda x: torch.log(torch.abs(x))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActNorm(nn.Module):
    """
    logdet计算行列式的值，输入通道数，actnorm就是普通的线性变换
    """

    def __init__(self, in_channel, logdet=True):
        """
        loc就是放射变换的w，scale是平移量
        initial全是0，表示还没有对参数进行初始化，需要初始化
        :param in_channel:
        :param logdet:
        """

        # 4个维度，并加入到module里面，4个维度=bs*c*h*w
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.initialized = nn.Parameter(torch.tensor(0, dtype=torch.uint8), requires_grad=False)

        self.logdet = logdet

    def initialize(self, input):
        """
        求出std和mean进行归一化
        只保留channel的std
        :param input:
        :return:
        """
        with torch.no_grad():
            mean = torch.mean(input, dim=(0, 2, 3), keepdim=True)
            std = torch.std(input, dim=(0, 2, 3), keepdim=True)
            # 使用均值进行填充
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        batch_size, _, height, width = input.shape
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)
        logdet = torch.tile(torch.tensor([logdet], device=device), (batch_size,))
        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (self.loc + input)

    def reverse(self, output):
        """
        返回最初的input，根据上面的公式反解就可以
        :param output:
        :return:
        """
        return output / self.scale - self.loc


class InvCon2d(nn.Module):
    """
    论文使用对焦矩阵简化求解
    """

    def __init__(self, in_channel):
        super().__init__()
        weight = torch.randn(in_channel, in_channel)
        # qr分解，变成对角
        q, _ = torch.qr(weight)
        # 然后进行解压，变成4维度
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        batch_size, _, height, width = input.shape
        out = F.conv2d(input, self.weight)
        logdet = (
                height * width * torch.slogdet(self.weight.squeeze().double()[1].float)
        )
        return out, logdet

    def reverse(self, output):
        """
        求你，只需要把矩阵进行求逆就可以得到input
        :param output:
        :return:
        """
        return F.conv2d(
            output, self.weight.squeeze().inversr().unsqueeze(2).unsqueeze(2)
        )


# 基于论文的lu分解
class InvCon2dLu(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        # qr分解，q是正交矩阵
        q, _ = la.qr(weight)
        # 行列式部位0 ，lu一定可以
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        # 获取对焦元素构成数组
        w_s = np.diag(w_u)
        # 对角线元素全是0
        w_u = np.triu(w_u, 1)
        # 对角线元素为0，其余全是1
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p).to(device)
        w_l = torch.from_numpy(w_l).to(device)
        w_s = torch.from_numpy(w_s).to(device)
        w_u = torch.from_numpy(w_u).to(device)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask)).to(device)
        self.register_buffer("l_mask", torch.from_numpy(l_mask)).to(device)
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def calc_weight(self):
        weight = (
                self.w_p
                @ (self.w_l * self.l_mask + self.l_eye)
                @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def forward(self, input):
        batch_size, _, height, width = input.shape
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)
        logdet = torch.tile(torch.tensor([logdet], device=device), (batch_size,))

        return out, logdet

    def reverse(self, output):
        weight = self.calc_weight()
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


# 接下来就是仿射实现，再吃之前需要一个全0的卷积
class ZeroCon2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        # 全0初始化
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.scale = nn.Parameter(torch.zeros((1, out_channel, 1, 1)))

    def forward(self, input):
        # 全1填充
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(input)
        out = out * torch.exp(self.scale * 3)
        return out


class AffineCoupling(nn.Module):
    """
    放射层的实现
    """

    def __init__(self, in_channel, filter_size=512, affine=True):
        super(AffineCoupling, self).__init__()
        self.affine = affine
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroCon2d(filter_size, in_channel if self.affine else in_channel // 2)

        )
        # 对2个卷积层的参数进行初始化
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        """
        仿射层，需要进行拆分图片，然后在进入net，最后再和weight结合
        :param input:
        :return:
        """
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out
        return torch.cat([out_a, in_b], 1)


# flow实现
class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super.__init__()
        self.actnorm = ActNorm(in_channel)
        if conv_lu:
            self.invconv = InvCon2dLu(in_channel)
        else:
            self.invconv = InvCon2d(in_channel)
        self.coupling = AffineCoupling(in_channel, affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)
        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, output):
        # 反过来，先仿射，再act norm
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input


# 最后一个就是block实现
def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroCon2d(in_channel * 2, in_channel * 4)

        else:
            self.prior = ZeroCon2d(in_channel * 4, in_channel * 8)

        self.h_zero = nn.Parameter(torch.zeros(1, in_channel * 4, 8, 8), requires_grad=True)

        self.label_embedding = nn.Embedding(40, 32)
        self.proj_layer = nn.Linear(32, in_channel * 4)

    def forward(self, input, label):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            condition = self.label_embedding(label)
            condition = F.softplus(self.proj_layer(condition))
            condition = condition.unsqueeze(-1).unsqueeze(-1)

            mean, log_sd = self.prior(self.h_zero + condition).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, label, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                condition = self.label_embedding(label)
                condition = F.softplus(self.proj_layer(condition))
                condition = condition.unsqueeze(-1).unsqueeze(-1)
                # zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(self.h_zero + condition).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


# 最后一部glow构成
class Glow(nn.Module):
    def __init__(
            self, in_channel, n_flow, n_block, affine=True, conv_lu=True
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

        self.classifier_net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(48 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 40)
        )

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        logits =self.classifier_net(z_new)
        return log_p_sum, logdet, z_outs,logits

    def reverse(self,z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input
