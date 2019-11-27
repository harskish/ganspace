'''
Running statistics on the GPU using pytorch.

RunningTopK maintains top-k statistics for a set of channels in parallel.
RunningQuantile maintains (sampled) quantile statistics for a set of channels.
'''

import torch, math, numpy
from collections import defaultdict

class RunningTopK:
    '''
    A class to keep a running tally of the the top k values (and indexes)
    of any number of torch feature components.  Will work on the GPU if
    the data is on the GPU.

    This version flattens all arrays to avoid crashes.
    '''
    def __init__(self, k=100, state=None):
        if state is not None:
            self.set_state_dict(state)
            return
        self.k = k
        self.count = 0
        # This version flattens all data internally to 2-d tensors,
        # to avoid crashes with the current pytorch topk implementation.
        # The data is puffed back out to arbitrary tensor shapes on ouput.
        self.data_shape = None
        self.top_data = None
        self.top_index = None
        self.next = 0
        self.linear_index = 0
        self.perm = None

    def add(self, data):
        '''
        Adds a batch of data to be considered for the running top k.
        The zeroth dimension enumerates the observations.  All other
        dimensions enumerate different features.
        '''
        if self.top_data is None:
            # Allocation: allocate a buffer of size 5*k, at least 10, for each.
            self.data_shape = data.shape[1:]
            feature_size = int(numpy.prod(self.data_shape))
            self.top_data = torch.zeros(
                    feature_size, max(10, self.k * 5), out=data.new())
            self.top_index = self.top_data.clone().long()
            self.linear_index = 0 if len(data.shape) == 1 else torch.arange(
                feature_size, out=self.top_index.new()).mul_(
                        self.top_data.shape[-1])[:,None]
        size = data.shape[0]
        sk = min(size, self.k)
        if self.top_data.shape[-1] < self.next + sk:
            # Compression: if full, keep topk only.
            self.top_data[:,:self.k], self.top_index[:,:self.k] = (
                    self.result(sorted=False, flat=True))
            self.next = self.k
            free = self.top_data.shape[-1] - self.next
        # Pick: copy the top sk of the next batch into the buffer.
        # Currently strided topk is slow.  So we clone after transpose.
        # TODO: remove the clone() if it becomes faster.
        cdata = data.contiguous().view(size, -1).t().clone()
        td, ti = cdata.topk(sk, sorted=False)
        self.top_data[:,self.next:self.next+sk] = td
        self.top_index[:,self.next:self.next+sk] = (ti + self.count)
        self.next += sk
        self.count += size

    def result(self, sorted=True, flat=False):
        '''
        Returns top k data items and indexes in each dimension,
        with channels in the first dimension and k in the last dimension.
        '''
        k = min(self.k, self.next)
        # bti are top indexes relative to buffer array.
        td, bti = self.top_data[:,:self.next].topk(k, sorted=sorted)
        # we want to report top indexes globally, which is ti.
        ti = self.top_index.view(-1)[
                (bti + self.linear_index).view(-1)
                ].view(*bti.shape)
        if flat:
            return td, ti
        else:
            return (td.view(*(self.data_shape + (-1,))),
                    ti.view(*(self.data_shape + (-1,))))

    def to_(self, device):
        self.top_data = self.top_data.to(device)
        self.top_index = self.top_index.to(device)
        if isinstance(self.linear_index, torch.Tensor):
            self.linear_index = self.linear_index.to(device)

    def state_dict(self):
        return dict(
                constructor=self.__module__ + '.' +
                    self.__class__.__name__ + '()',
                k=self.k,
                count=self.count,
                data_shape=tuple(self.data_shape),
                top_data=self.top_data.cpu().numpy(),
                top_index=self.top_index.cpu().numpy(),
                next=self.next,
                linear_index=(self.linear_index.cpu().numpy()
                    if isinstance(self.linear_index, torch.Tensor)
                    else self.linear_index),
                perm=self.perm)

    def set_state_dict(self, dic):
        self.k = dic['k'].item()
        self.count = dic['count'].item()
        self.data_shape = tuple(dic['data_shape'])
        self.top_data = torch.from_numpy(dic['top_data'])
        self.top_index = torch.from_numpy(dic['top_index'])
        self.next = dic['next'].item()
        self.linear_index = (torch.from_numpy(dic['linear_index'])
                if len(dic['linear_index'].shape) > 0
                else dic['linear_index'].item())

class RunningQuantile:
    """
    Streaming randomized quantile computation for torch.

    Add any amount of data repeatedly via add(data).  At any time,
    quantile estimates (or old-style percentiles) can be read out using
    quantiles(q) or percentiles(p).

    Accuracy scales according to resolution: the default is to
    set resolution to be accurate to better than 0.1%,
    while limiting storage to about 50,000 samples.

    Good for computing quantiles of huge data without using much memory.
    Works well on arbitrary data with probability near 1.

    Based on the optimal KLL quantile algorithm by Karnin, Lang, and Liberty
    from FOCS 2016.  http://ieee-focs.org/FOCS-2016-Papers/3933a071.pdf
    """

    def __init__(self, resolution=6 * 1024, buffersize=None, seed=None,
            state=None):
        if state is not None:
            self.set_state_dict(state)
            return
        self.depth = None
        self.dtype = None
        self.device = None
        self.resolution = resolution
        # Default buffersize: 128 samples (and smaller than resolution).
        if buffersize is None:
            buffersize = min(128, (resolution + 7) // 8)
        self.buffersize = buffersize
        self.samplerate = 1.0
        self.data = None
        self.firstfree = [0]
        self.randbits = torch.ByteTensor(resolution)
        self.currentbit = len(self.randbits) - 1
        self.extremes = None
        self.size = 0

    def _lazy_init(self, incoming):
        self.depth = incoming.shape[1]
        self.dtype = incoming.dtype
        self.device = incoming.device
        self.data = [torch.zeros(self.depth, self.resolution,
            dtype=self.dtype, device=self.device)]
        self.extremes = torch.zeros(self.depth, 2,
                dtype=self.dtype, device=self.device)
        self.extremes[:,0] = float('inf')
        self.extremes[:,-1] = -float('inf')

    def to_(self, device):
        """Switches internal storage to specified device."""
        if device != self.device:
            old_data = self.data
            old_extremes = self.extremes
            self.data = [d.to(device) for d in self.data]
            self.extremes = self.extremes.to(device)
            self.device = self.extremes.device
            del old_data
            del old_extremes

    def add(self, incoming):
        if self.depth is None:
            self._lazy_init(incoming)
        assert len(incoming.shape) == 2
        assert incoming.shape[1] == self.depth, (incoming.shape[1], self.depth)
        self.size += incoming.shape[0]
        # Convert to a flat torch array.
        if self.samplerate >= 1.0:
            self._add_every(incoming)
            return
        # If we are sampling, then subsample a large chunk at a time.
        self._scan_extremes(incoming)
        chunksize = int(math.ceil(self.buffersize / self.samplerate))
        for index in range(0, len(incoming), chunksize):
            batch = incoming[index:index+chunksize]
            sample = sample_portion(batch, self.samplerate)
            if len(sample):
                self._add_every(sample)

    def _add_every(self, incoming):
        supplied = len(incoming)
        index = 0
        while index < supplied:
            ff = self.firstfree[0]
            available = self.data[0].shape[1] - ff
            if available == 0:
                if not self._shift():
                    # If we shifted by subsampling, then subsample.
                    incoming = incoming[index:]
                    if self.samplerate >= 0.5:
                        # First time sampling - the data source is very large.
                        self._scan_extremes(incoming)
                    incoming = sample_portion(incoming, self.samplerate)
                    index = 0
                    supplied = len(incoming)
                ff = self.firstfree[0]
                available = self.data[0].shape[1] - ff
            copycount = min(available, supplied - index)
            self.data[0][:,ff:ff + copycount] = torch.t(
                    incoming[index:index + copycount,:])
            self.firstfree[0] += copycount
            index += copycount

    def _shift(self):
        index = 0
        # If remaining space at the current layer is less than half prev
        # buffer size (rounding up), then we need to shift it up to ensure
        # enough space for future shifting.
        while self.data[index].shape[1] - self.firstfree[index] < (
                -(-self.data[index-1].shape[1] // 2) if index else 1):
            if index + 1 >= len(self.data):
                return self._expand()
            data = self.data[index][:,0:self.firstfree[index]]
            data = data.sort()[0]
            if index == 0 and self.samplerate >= 1.0:
                self._update_extremes(data[:,0], data[:,-1])
            offset = self._randbit()
            position = self.firstfree[index + 1]
            subset = data[:,offset::2]
            self.data[index + 1][:,position:position + subset.shape[1]] = subset
            self.firstfree[index] = 0
            self.firstfree[index + 1] += subset.shape[1]
            index += 1
        return True

    def _scan_extremes(self, incoming):
        # When sampling, we need to scan every item still to get extremes
        self._update_extremes(
                torch.min(incoming, dim=0)[0],
                torch.max(incoming, dim=0)[0])

    def _update_extremes(self, minr, maxr):
        self.extremes[:,0] = torch.min(
                torch.stack([self.extremes[:,0], minr]), dim=0)[0]
        self.extremes[:,-1] = torch.max(
                torch.stack([self.extremes[:,-1], maxr]), dim=0)[0]

    def _randbit(self):
        self.currentbit += 1
        if self.currentbit >= len(self.randbits):
            self.randbits.random_(to=2)
            self.currentbit = 0
        return self.randbits[self.currentbit]

    def state_dict(self):
        return dict(
                constructor=self.__module__ + '.' +
                    self.__class__.__name__ + '()',
                resolution=self.resolution,
                depth=self.depth,
                buffersize=self.buffersize,
                samplerate=self.samplerate,
                data=[d.cpu().numpy()[:,:f].T
                    for d, f in zip(self.data, self.firstfree)],
                sizes=[d.shape[1] for d in self.data],
                extremes=self.extremes.cpu().numpy(),
                size=self.size)

    def set_state_dict(self, dic):
        self.resolution = int(dic['resolution'])
        self.randbits = torch.ByteTensor(self.resolution)
        self.currentbit = len(self.randbits) - 1
        self.depth = int(dic['depth'])
        self.buffersize = int(dic['buffersize'])
        self.samplerate = float(dic['samplerate'])
        firstfree = []
        buffers = []
        for d, s in zip(dic['data'], dic['sizes']):
            firstfree.append(d.shape[0])
            buf = numpy.zeros((d.shape[1], s), dtype=d.dtype)
            buf[:,:d.shape[0]] = d.T
            buffers.append(torch.from_numpy(buf))
        self.firstfree = firstfree
        self.data = buffers
        self.extremes = torch.from_numpy((dic['extremes']))
        self.size = int(dic['size'])
        self.dtype = self.extremes.dtype
        self.device = self.extremes.device

    def minmax(self):
        if self.firstfree[0]:
            self._scan_extremes(self.data[0][:,:self.firstfree[0]].t())
        return self.extremes.clone()

    def median(self):
        return self.quantiles([0.5])[:,0]

    def mean(self):
        return self.integrate(lambda x: x) / self.size

    def variance(self):
        mean = self.mean()[:,None]
        return self.integrate(lambda x: (x - mean).pow(2)) / (self.size - 1)

    def stdev(self):
        return self.variance().sqrt()

    def _expand(self):
        cap = self._next_capacity()
        if cap > 0:
            # First, make a new layer of the proper capacity.
            self.data.insert(0, torch.zeros(self.depth, cap,
                dtype=self.dtype, device=self.device))
            self.firstfree.insert(0, 0)
        else:
            # Unless we're so big we are just subsampling.
            assert self.firstfree[0] == 0
            self.samplerate *= 0.5
        for index in range(1, len(self.data)):
            # Scan for existing data that needs to be moved down a level.
            amount = self.firstfree[index]
            if amount == 0:
                continue
            position = self.firstfree[index-1]
            # Move data down if it would leave enough empty space there
            # This is the key invariant: enough empty space to fit half
            # of the previous level's buffer size (rounding up)
            if self.data[index-1].shape[1] - (amount + position) >= (
                    -(-self.data[index-2].shape[1] // 2) if (index-1) else 1):
                self.data[index-1][:,position:position + amount] = (
                        self.data[index][:,:amount])
                self.firstfree[index-1] += amount
                self.firstfree[index] = 0
            else:
                # Scrunch the data if it would not.
                data = self.data[index][:,:amount]
                data = data.sort()[0]
                if index == 1:
                    self._update_extremes(data[:,0], data[:,-1])
                offset = self._randbit()
                scrunched = data[:,offset::2]
                self.data[index][:,:scrunched.shape[1]] = scrunched
                self.firstfree[index] = scrunched.shape[1]
        return cap > 0

    def _next_capacity(self):
        cap = int(math.ceil(self.resolution * (0.67 ** len(self.data))))
        if cap < 2:
            return 0
        # Round up to the nearest multiple of 8 for better GPU alignment.
        cap = -8 * (-cap // 8)
        return max(self.buffersize, cap)

    def _weighted_summary(self, sort=True):
        if self.firstfree[0]:
            self._scan_extremes(self.data[0][:,:self.firstfree[0]].t())
        size = sum(self.firstfree) + 2
        weights = torch.FloatTensor(size) # Floating point
        summary = torch.zeros(self.depth, size,
                dtype=self.dtype, device=self.device)
        weights[0:2] = 0
        summary[:,0:2] = self.extremes
        index = 2
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            summary[:,index:index + ff] = self.data[level][:,:ff]
            weights[index:index + ff] = 2.0 ** level
            index += ff
        assert index == summary.shape[1]
        if sort:
            summary, order = torch.sort(summary, dim=-1)
            weights = weights[order.view(-1).cpu()].view(order.shape)
        return (summary, weights)

    def quantiles(self, quantiles, old_style=False):
        if self.size == 0:
            return torch.full((self.depth, len(quantiles)), torch.nan)
        summary, weights = self._weighted_summary()
        cumweights = torch.cumsum(weights, dim=-1) - weights / 2
        if old_style:
            # To be convenient with torch.percentile
            cumweights -= cumweights[:,0:1].clone()
            cumweights /= cumweights[:,-1:].clone()
        else:
            cumweights /= torch.sum(weights, dim=-1, keepdim=True)
        result = torch.zeros(self.depth, len(quantiles),
                dtype=self.dtype, device=self.device)
        # numpy is needed for interpolation
        if not hasattr(quantiles, 'cpu'):
            quantiles = torch.Tensor(quantiles)
        nq = quantiles.cpu().numpy()
        ncw = cumweights.cpu().numpy()
        nsm = summary.cpu().numpy()
        for d in range(self.depth):
            result[d] = torch.tensor(numpy.interp(nq, ncw[d], nsm[d]),
                    dtype=self.dtype, device=self.device)
        return result

    def integrate(self, fun):
        result = None
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            term = torch.sum(
                    fun(self.data[level][:,:ff]) * (2.0 ** level),
                    dim=-1)
            if result is None:
                result = term
            else:
                result += term
        if result is not None:
            result /= self.samplerate
        return result

    def percentiles(self, percentiles):
        return self.quantiles(percentiles, old_style=True)

    def readout(self, count=1001, old_style=True):
        return self.quantiles(
                torch.linspace(0.0, 1.0, count), old_style=old_style)

    def normalize(self, data):
        '''
        Given input data as taken from the training distirbution,
        normalizes every channel to reflect quantile values,
        uniformly distributed, within [0, 1].
        '''
        assert self.size > 0
        assert data.shape[0] == self.depth
        summary, weights = self._weighted_summary()
        cumweights = torch.cumsum(weights, dim=-1) - weights / 2
        cumweights /= torch.sum(weights, dim=-1, keepdim=True)
        result = torch.zeros_like(data).float()
        # numpy is needed for interpolation
        ndata = data.cpu().numpy().reshape((data.shape[0], -1))
        ncw = cumweights.cpu().numpy()
        nsm = summary.cpu().numpy()
        for d in range(self.depth):
            normed = torch.tensor(numpy.interp(ndata[d], nsm[d], ncw[d]),
                dtype=torch.float, device=data.device).clamp_(0.0, 1.0)
            if len(data.shape) > 1:
                normed = normed.view(*(data.shape[1:]))
            result[d] = normed
        return result


class RunningConditionalQuantile:
    '''
    Equivalent to a map from conditions (any python hashable type)
    to RunningQuantiles.  The reason for the type is to allow limited
    GPU memory to be exploited while counting quantile stats on many
    different conditions, a few of which are common and which benefit
    from GPU, but most of which are rare and would not all fit into
    GPU RAM.

    To move a set of conditions to a device, use rcq.to_(device, conds).
    Then in the future, move the tallied data to the device before
    calling rcq.add, that is, rcq.add(cond, data.to(device)).

    To allow the caller to decide which conditions to allow to use GPU,
    rcq.most_common_conditions(n) returns a list of the n most commonly
    added conditions so far.
    '''
    def __init__(self, resolution=6 * 1024, buffersize=None, seed=None,
            state=None):
        self.first_rq = None
        self.call_stats = defaultdict(int)
        self.running_quantiles = {}
        if state is not None:
            self.set_state_dict(state)
            return
        self.rq_args = dict(resolution=resolution, buffersize=buffersize,
                seed=seed)

    def add(self, condition, incoming):
        if condition not in self.running_quantiles:
            self.running_quantiles[condition] = RunningQuantile(**self.rq_args)
            if self.first_rq is None:
                self.first_rq = self.running_quantiles[condition]
        self.call_stats[condition] += 1
        rq = self.running_quantiles[condition]
        # For performance reasons, the caller can move some conditions to
        # the CPU if they are not among the most common conditions.
        if rq.device is not None and (rq.device != incoming.device):
            rq.to_(incoming.device)
        self.running_quantiles[condition].add(incoming)

    def most_common_conditions(self, n):
        return sorted(self.call_stats.keys(),
                key=lambda c: -self.call_stats[c])[:n]

    def collected_add(self, conditions, incoming):
        for c in conditions:
            self.add(c, incoming)

    def conditional(self, c):
        return self.running_quantiles[c]

    def collected_quantiles(self, conditions, quantiles, old_style=False):
        result = torch.zeros(
                size=(len(conditions), self.first_rq.depth, len(quantiles)),
                dtype=self.first_rq.dtype,
                device=self.first_rq.device)
        for i, c in enumerate(conditions):
            if c in self.running_quantiles:
                result[i] = self.running_quantiles[c].quantiles(
                        quantiles, old_style)
        return result

    def collected_normalize(self, conditions, values):
        result = torch.zeros(
                size=(len(conditions), values.shape[0], values.shape[1]),
                dtype=torch.float,
                device=self.first_rq.device)
        for i, c in enumerate(conditions):
            if c in self.running_quantiles:
                result[i] = self.running_quantiles[c].normalize(values)
        return result

    def to_(self, device, conditions=None):
        if conditions is None:
            conditions = self.running_quantiles.keys()
        for cond in conditions:
            if cond in self.running_quantiles:
                self.running_quantiles[cond].to_(device)

    def state_dict(self):
        conditions = sorted(self.running_quantiles.keys())
        result = dict(
                constructor=self.__module__ + '.' +
                    self.__class__.__name__ + '()',
                rq_args=self.rq_args,
                conditions=conditions)
        for i, c in enumerate(conditions):
            result.update({
                '%d.%s' % (i, k): v
                for k, v in self.running_quantiles[c].state_dict().items()})
        return result

    def set_state_dict(self, dic):
        self.rq_args = dic['rq_args'].item()
        conditions = list(dic['conditions'])
        subdicts = defaultdict(dict)
        for k, v in dic.items():
            if '.' in k:
                p, s = k.split('.', 1)
                subdicts[p][s] = v
        self.running_quantiles = {
                c: RunningQuantile(state=subdicts[str(i)])
                for i, c in enumerate(conditions)}
        if conditions:
            self.first_rq = self.running_quantiles[conditions[0]]

    # example usage:
    # levels = rqc.conditional(()).quantiles(1 - fracs)
    # denoms = 1 - rqc.collected_normalize(cats, levels)
    # isects = 1 - rqc.collected_normalize(labels, levels)
    # unions = fracs + denoms[cats] - isects
    # iou = isects / unions




class RunningCrossCovariance:
    '''
    Running computation. Use this when an off-diagonal block of the
    covariance matrix is needed (e.g., when the whole covariance matrix
    does not fit in the GPU).

    Chan-style numerically stable update of mean and full covariance matrix.
    Chan, Golub. LeVeque. 1983. http://www.jstor.org/stable/2683386
    '''
    def __init__(self, state=None):
        if state is not None:
            self.set_state_dict(state)
            return
        self.count = 0
        self._mean = None
        self.cmom2 = None
        self.v_cmom2 = None

    def add(self, a, b):
        if len(a.shape) == 1:
            a = a[None, :]
            b = b[None, :]
        assert(a.shape[0] == b.shape[0])
        if len(a.shape) > 2:
            a, b = [d.view(d.shape[0], d.shape[1], -1).permute(0, 2, 1
                ).contiguous().view(-1, d.shape[1]) for d in [a, b]]
        batch_count = a.shape[0]
        batch_mean = [d.sum(0) / batch_count for d in [a, b]]
        centered = [d - bm for d, bm in zip([a, b], batch_mean)]
        # If more than 10 billion operations, divide into batches.
        sub_batch = -(-(10 << 30) // (a.shape[1] * b.shape[1]))
        # Initial batch.
        if self._mean is None:
            self.count = batch_count
            self._mean = batch_mean
            self.v_cmom2 = [c.pow(2).sum(0) for c in centered]
            self.cmom2 = a.new(a.shape[1], b.shape[1]).zero_()
            progress_addbmm(self.cmom2, centered[0][:,:,None],
                    centered[1][:,None,:], sub_batch)
            return
        # Update a batch using Chan-style update for numerical stability.
        oldcount = self.count
        self.count += batch_count
        new_frac = float(batch_count) / self.count
        # Update the mean according to the batch deviation from the old mean.
        delta = [bm.sub_(m).mul_(new_frac)
                for bm, m in zip(batch_mean, self._mean)]
        for m, d in zip(self._mean, delta):
            m.add_(d)
        # Update the cross-covariance using the batch deviation
        progress_addbmm(self.cmom2, centered[0][:,:,None],
                centered[1][:,None,:], sub_batch)
        self.cmom2.addmm_(alpha=new_frac * oldcount,
                mat1=delta[0][:,None], mat2=delta[1][None,:])
        # Update the variance using the batch deviation
        for c, vc2, d in zip(centered, self.v_cmom2, delta):
            vc2.add_(c.pow(2).sum(0))
            vc2.add_(d.pow_(2).mul_(new_frac * oldcount))

    def mean(self):
        return self._mean

    def variance(self):
        return [vc2 / (self.count - 1) for vc2 in self.v_cmom2]

    def stdev(self):
        return [v.sqrt() for v in self.variance()]

    def covariance(self):
        return self.cmom2 / (self.count - 1)

    def correlation(self):
        covariance = self.covariance()
        rstdev = [s.reciprocal() for s in self.stdev()]
        cor = rstdev[0][:,None] * covariance * rstdev[1][None,:]
        # Remove NaNs
        cor[torch.isnan(cor)] = 0
        return cor

    def to_(self, device):
        self._mean = [m.to(device) for m in self._mean]
        self.v_cmom2 = [vcs.to(device) for vcs in self.v_cmom2]
        self.cmom2 = self.cmom2.to(device)

    def state_dict(self):
        return dict(
                constructor=self.__module__ + '.' +
                    self.__class__.__name__ + '()',
                count=self.count,
                mean_a=self._mean[0].cpu().numpy(),
                mean_b=self._mean[1].cpu().numpy(),
                cmom2_a=self.v_cmom2[0].cpu().numpy(),
                cmom2_b=self.v_cmom2[1].cpu().numpy(),
                cmom2=self.cmom2.cpu().numpy())

    def set_state_dict(self, dic):
        self.count = dic['count'].item()
        self._mean = [torch.from_numpy(dic[k]) for k in ['mean_a', 'mean_b']]
        self.v_cmom2 = [torch.from_numpy(dic[k])
                for k in ['cmom2_a', 'cmom2_b']]
        self.cmom2 = torch.from_numpy(dic['cmom2'])

def progress_addbmm(accum, x, y, batch_size):
    '''
    Break up very large adbmm operations into batches so progress can be seen.
    '''
    from .progress import default_progress
    if x.shape[0] <= batch_size:
        return accum.addbmm_(x, y)
    progress = default_progress(None)
    for i in progress(range(0, x.shape[0], batch_size), desc='bmm'):
        accum.addbmm_(x[i:i+batch_size], y[i:i+batch_size])
    return accum


def sample_portion(vec, p=0.5):
    bits = torch.bernoulli(torch.zeros(vec.shape[0], dtype=torch.uint8,
        device=vec.device), p)
    return vec[bits]

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("error")
    import time
    import argparse
    parser = argparse.ArgumentParser(
        description='Test things out')
    parser.add_argument('--mode', default='cpu', help='cpu or cuda')
    parser.add_argument('--test_size', type=int, default=1000000)
    args = parser.parse_args()

    # An adverarial case: we keep finding more numbers in the middle
    # as the stream goes on.
    amount = args.test_size
    quantiles = 1000
    data = numpy.arange(float(amount))
    data[1::2] = data[-1::-2] + (len(data) - 1)
    data /= 2
    depth = 50
    test_cuda = torch.cuda.is_available()
    alldata = data[:,None] + (numpy.arange(depth) * amount)[None, :]
    actual_sum = torch.FloatTensor(numpy.sum(alldata * alldata, axis=0))
    amt = amount // depth
    for r in range(depth):
        numpy.random.shuffle(alldata[r*amt:r*amt+amt,r])
    if args.mode == 'cuda':
        alldata = torch.cuda.FloatTensor(alldata)
        dtype = torch.float
        device = torch.device('cuda')
    else:
        alldata = torch.FloatTensor(alldata)
        dtype = torch.float
        device = None
    starttime = time.time()
    qc = RunningQuantile(resolution=6 * 1024)
    qc.add(alldata)
    # Test state dict
    saved = qc.state_dict()
    # numpy.savez('foo.npz', **saved)
    # saved = numpy.load('foo.npz')
    qc = RunningQuantile(state=saved)
    assert not qc.device.type == 'cuda'
    qc.add(alldata)
    actual_sum *= 2
    ro = qc.readout(1001).cpu()
    endtime = time.time()
    gt = torch.linspace(0, amount, quantiles+1)[None,:] + (
            torch.arange(qc.depth, dtype=torch.float) * amount)[:,None]
    maxreldev = torch.max(torch.abs(ro - gt) / amount) * quantiles
    print("Maximum relative deviation among %d perentiles: %f" % (
        quantiles, maxreldev))
    minerr = torch.max(torch.abs(qc.minmax().cpu()[:,0] -
            torch.arange(qc.depth, dtype=torch.float) * amount))
    maxerr = torch.max(torch.abs((qc.minmax().cpu()[:, -1] + 1) -
            (torch.arange(qc.depth, dtype=torch.float) + 1) * amount))
    print("Minmax error %f, %f" % (minerr, maxerr))
    interr = torch.max(torch.abs(qc.integrate(lambda x: x * x).cpu()
            - actual_sum) / actual_sum)
    print("Integral error: %f" % interr)
    medianerr = torch.max(torch.abs(qc.median() -
        alldata.median(0)[0]) / alldata.median(0)[0]).cpu()
    print("Median error: %f" % interr)
    meanerr = torch.max(
            torch.abs(qc.mean() - alldata.mean(0)) / alldata.mean(0)).cpu()
    print("Mean error: %f" % meanerr)
    varerr = torch.max(
            torch.abs(qc.variance() - alldata.var(0)) / alldata.var(0)).cpu()
    print("Variance error: %f" % varerr)
    counterr = ((qc.integrate(lambda x: torch.ones(x.shape[-1]).cpu())
                - qc.size) / (0.0 + qc.size)).item()
    print("Count error: %f" % counterr)
    print("Time %f" % (endtime - starttime))
    # Algorithm is randomized, so some of these will fail with low probability.
    assert maxreldev < 1.0
    assert minerr == 0.0
    assert maxerr == 0.0
    assert interr < 0.01
    assert abs(counterr) < 0.001
    print("OK")
