import collections
import re
import queue
import threading

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

try:
    from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
except ImportError:
    ReduceAddCoalesced = Broadcast = None


class SlavePipe(collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class FutureResult:
    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, 'Previous result has\'t been fetched.'
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res


class SyncMaster:
    _MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = self._MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dimensions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


class SynchronizedBatchNorm2d(_BatchNorm):
    _ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
    _MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        assert ReduceAddCoalesced is not None, 'Can not use Synchronized Batch Normalization without CUDA support.'

        super().__init__(
            num_features, eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats
        )

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        assert input.size(1) == self.num_features, 'Channel size mismatch: got {}, expect {}.'.format(input.size(1),
                                                                                                      self.num_features)
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(self._ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(self._ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], self._MasterMessage(*broadcasted[i * 2:i * 2 + 2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        if hasattr(torch, 'no_grad'):
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, bias_var.clamp(self.eps) ** -0.5

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()
        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)

        ks = int(parsed.group(2))
        self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2

        self.mlp_shared = nn.Sequential(nn.Conv2d(3, nhidden, kernel_size=ks, padding=pw), nn.ReLU())

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, degraded_image):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        degraded_face = F.interpolate(degraded_image, size=x.size()[2:], mode="bilinear", align_corners=True)

        actv = self.mlp_shared(degraded_face)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        # Attributes
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        self.conv_0 = nn.utils.spectral_norm(self.conv_0)
        self.conv_1 = nn.utils.spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = nn.utils.spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = 'spadesyncbatch3x3'
        self.norm_0 = SPADE(spade_config_str, fin, 18)
        self.norm_1 = SPADE(spade_config_str, fmiddle, 18)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, 18)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, degraded_image):
        x_s = self.shortcut(x, seg, degraded_image)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg, degraded_image)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, degraded_image)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg, degraded_image):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, degraded_image))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADEGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.sw, self.sh = 8, 8

        self.fc = nn.Conv2d(3, 1024, 3, padding=1)

        self.head_0 = SPADEResnetBlock(1024, 1024)

        self.G_middle_0 = SPADEResnetBlock(1024, 1024)
        self.G_middle_1 = SPADEResnetBlock(1024, 1024)

        self.up_0 = SPADEResnetBlock(1024, 512)
        self.up_1 = SPADEResnetBlock(512, 256)
        self.up_2 = SPADEResnetBlock(256, 128)
        self.up_3 = SPADEResnetBlock(128, 64)

        self.conv_img = nn.Conv2d(64, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()

    def init_weights(self, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
                if init_type == "normal":
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "xavier_uniform":
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == "none":  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(init_type, gain)

    def forward(self, input_, degraded_image):
        x = F.interpolate(degraded_image, size=(self.sh, self.sw), mode="bilinear", align_corners=True)
        x = self.fc(x)

        x = self.head_0(x, input_, degraded_image)

        x = self.up(x)
        x = self.G_middle_0(x, input_, degraded_image)

        x = self.G_middle_1(x, input_, degraded_image)

        x = self.up(x)
        x = self.up_0(x, input_, degraded_image)
        x = self.up(x)
        x = self.up_1(x, input_, degraded_image)
        x = self.up(x)
        x = self.up_2(x, input_, degraded_image)
        x = self.up(x)
        x = self.up_3(x, input_, degraded_image)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x


class Pix2PixModel(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor

        self.netG = SPADEGenerator()
        self.netG.load_state_dict(state_dict)

    def forward(self, face):
        with torch.no_grad():
            return self.generate_fake(*self.preprocess_input(face))

    def preprocess_input(self, face):
        if torch.cuda.is_available():
            label = torch.zeros(
                (2, 18, 256, 256), dtype=torch.float32, device=torch.device('cuda')
            )
            face = face.cuda()
        else:
            label = torch.zeros((2, 18, 256, 256), dtype=torch.float32)
        face = face.unsqueeze(0)
        return label, face

    def generate_fake(self, input_semantics, degraded_image):
        fake_image = self.netG(input_semantics, degraded_image)
        return fake_image
