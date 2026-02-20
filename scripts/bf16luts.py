#!python3
import warnings
from torch import (tensor, int32, bfloat16, uint16, int16, float32, float64, log2,
                   int32, empty, int64, arange, set_printoptions, argmin, stack, inf, ones)
set_printoptions(30)


INT_MIN, INT_MAX = tensor(-2**15, dtype=int32), tensor(2**15 - 1, dtype=int32)


# forward rescale always takes tensor int -> tensor int.  Create
# common helper that also works on floats.
def rescale_helper(scale, in_zp, out_zp):
    return lambda x: ((x - in_zp) * scale + out_zp).clip(INT_MIN, INT_MAX)
# Rescales should round during the forward pass, but keep the inverse
# continuous.
class Rescale:
    def __init__(self, scale, in_zp, out_zp):
        self.shift_factor = int((14 - log2(scale).floor()) // 4 * 4)
        self.scale_factor = (scale * (1 << self.shift_factor)).round().int()
        self.in_zp, self.out_zp = in_zp, out_zp
        self.params = [str(z) for z in
                       (self.scale_factor, self.shift_factor, in_zp, out_zp)]
        self.I = rescale_helper(1/scale, out_zp, in_zp)
    def __call__(self, x):
        return ((((x.to(int32) - self.in_zp).to(int32) * self.scale_factor
                  + (1 << (self.shift_factor - 1)))
                 // (1 << self.shift_factor)
                 ).int() + self.out_zp
                ).clip(INT_MIN, INT_MAX).to(int16)
    def __repr__(self):
        return "scale, shift, in_zp, out_zp: " + ", ".join(self.params)


# torq LUTs are pre-packed with base/slop information.  Emulate that
# packing.
def pack(base, slope):
    base, slope = base.round().to(int16), slope.round().to(int16)
    return (slope.to(int32) << 16) + base.view(uint16).to(int32)

def unpack(packed):
    return (packed % (1 << 16)).to(int16), (packed >> 16).to(int16)


# basic lut utilities
def domain_to_index_delta(d):
    return (d >> 7) + 256, d % (1 << 7)

def interp(base, slope, delta):
    base, slope, delta = base.to(int32), slope.to(int32), delta.to(int32)
    return (base + ((slope * delta + (1 << 6))) // (1 << 7)
            ).clip(INT_MIN, INT_MAX).to(int16)

def run_lut(lut, x):
    index, delta = domain_to_index_delta(x)
    base, slope = unpack(lut[index.int()])
    return interp(base, slope, delta)


# some error functions
def torq_error(x, y):
    x, y = x.view(bfloat16).to(float64), y.view(bfloat16).to(float64)
    error = (x - y).abs() / (x.abs() + y.abs() + 1e-6)
    x_inf, y_inf, x_ninf, y_ninf = x == inf, y == inf, x == -inf, y == -inf
    error[(x_inf & y_inf) | (x_ninf & y_ninf)] = 0
    error[(x_inf ^ y_inf) | (x_ninf ^ y_ninf)] = inf
    error[x.isnan() | y.isnan()] = inf
    assert not error.isnan().sum()
    return error


def generate_tols(gt):
    max_bittol = 30
    deviations = arange(-max_bittol, max_bittol, dtype=int32).reshape(-1, 1)
    x = (gt.to(int32) + deviations).clip(INT_MIN, INT_MAX).to(int16)
    return torq_error(x, gt).unique().sort().values


def fit_points(a, delta, gt, start, end):
    delta = delta.to(int32)
    v = a.I(gt).round().to(int32)
    assert all(a(v) == gt)
    Δk = tensor((-1, 1),dtype=int32).reshape(2, 1)

    for tol in generate_tols(gt):

        tols = tensor([tol] * len(gt), dtype=float64)
        # table endpoints must be exact
        if start:
            tols[0] = 0
        if end:
            tols[-1] = 0

        # find the limits for table output given bittol
        v_min, v_max = v.clone(), v.clone()
        while any(slack := ((v_min > INT_MIN) &
                            (torq_error(a(v_min - 1), gt) <= tols))):
            v_min -= slack.int()
        while any(slack := ((v_max < INT_MAX) &
                            (torq_error(a(v_max + 1), gt) <= tols))):
            v_max += slack.int()
        assert all(v_min == v_min.clip(INT_MIN, INT_MAX))
        assert all(v_max == v_max.clip(INT_MIN, INT_MAX))

        # constrain the possible bases and slopes based on above
        # limits.  Firstly, consider that at least one optimal
        # solution is bounded by the consecutive slopes:
        # v = base + (slope * delta + (1<<6)) // (1 >> 7)
        # v = base + (slope * delta + (1<<6)) / (1 >> 7) + k; k in [0, 1)
        # (1 << 7) * (v - k) = (1 << 7) * base + slope * delta + (1<<6)
        # (1 << 7) * (Δv - Δk) = slope * Δdelta
        # slope = (1 << 7) * (Δv - Δk) / Δdelta
        Δv, Δdelta = v[1:] - v[:-1], delta[1:] - delta[:-1]
        slopes = 128 * (Δv - Δk).to(float64) / (Δdelta).to(float32)
        slope_min = slopes.floor().int().min().clip(INT_MIN, INT_MAX)
        slope_max = slopes.ceil().int().max().clip(INT_MIN, INT_MAX)
        base_min, base_max = INT_MIN, INT_MAX
        moved = True
        failed = False
        while moved:
            moved = False
            # the constraint on v and slope imply constraints on the base
            b_min = v_min - (delta * slope_max + (1<<6)) // (1<<7)
            # clipping means it's fine to undershoot int_min
            b_min[v_min == INT_MIN] = INT_MIN
            b_min = b_min.max()
            if b_min > base_min:
                moved = True
                base_min = b_min
            b_max = v_max - (delta * slope_min + (1<<6)) // (1<<7)
            # clipping means it's fine to overshoot int_max
            b_max[v_max == INT_MAX] = INT_MAX
            b_max = b_max.min()
            if b_max < base_max:
                moved = True
                base_max = b_max
            # the constraints on the base imply constraints on the slope
            while any(interp(base_min, slope_max, delta) > v_max):
                slope_max -= 1
                moved = True
            while any(interp(base_max, slope_min, delta) < v_min):
                slope_min += 1
                moved = True
            # if we rule out everything, then move on to next tolerance
            if slope_min > slope_max or base_min > base_max:
                break
        else:
            # check that we actually found a solution
            base = arange(base_min, base_max + 1).reshape(-1, 1, 1)
            slope = arange(slope_min, slope_max + 1).reshape(-1, 1)
            best_tol_found = torq_error(a(interp(base, slope, delta)), gt
                                        ).max(-1).values.min()
            if best_tol_found == tol:
                # found solution.  Return.
                return base, slope
            assert best_tol_found > tol, 'somehow missed solution at prior tol'
    else:
        assert False, "exhausted tols?"


def fudge_index(domain_start, domain_end):
    rescale = Rescale(scale=(INT_MAX - INT_MIN) / (domain_end - domain_start),
                       in_zp=domain_start,
                       out_zp=INT_MIN)
    return int(rescale(domain_end) != INT_MAX)


def approximate_function(f, negative, domain=None):
    if domain is None:
        # negative integers are negative bfloat16 values
        if negative:
            domain = arange(-2**15, 0, dtype=int16)
        else:
            domain = arange(0, 2**15, dtype=int16)
    # ignore nans
    domain = domain[~domain.view(bfloat16).isnan()]

    # find the interesting part of the function.  The endpoints had
    # better be constant functions going away from the interesting
    # stuff.
    image = f(domain.view(bfloat16)).view(int16)
    image_start, image_end = image[[0, -1]]
    index_start = argmin((image == image_start).int()) - 1
    index_end = len(image) - argmin((image.flip(0) == image_end).int())
    fudge = fudge_index(*domain[[int(index_start), int(index_end)]])
    index_end += fudge
    image = image[index_start:index_end + 1]
    domain = domain[index_start:index_end + 1]

    # note that the domain is ordered/increasing, but the image need not be.
    domain_start, domain_end = domain[[0, -1]]
    range_start, range_end = image.min(), image.max()

    # approximate f(x) = {a(b(c([x])))} where b is a lookup table, a
    # and c are rescales that maximize the utility of b, '[]'
    # represents bf16->int16 bitcast, and '{}' represents int16->bf16
    # bitcast.
    a = Rescale(scale=(range_end - range_start) / (INT_MAX - INT_MIN),
                in_zp=INT_MIN,
                out_zp=range_start)
    assert a(INT_MIN) == range_start, str(a(INT_MIN))
    assert a(INT_MAX) == range_end, str(a(INT_MAX))
    c = Rescale(scale=(INT_MAX - INT_MIN) / (domain_end - fudge - domain_start),
                in_zp=domain_start,
                out_zp=INT_MIN)
    assert c(domain_start) == INT_MIN, str(c(domain_start))
    assert c(domain_end) == INT_MAX, str(c(domain_end))

    # sort inputs/outputs by table region
    regions = {i : [] for i in range(512)}
    for x, gt in zip(domain, image, strict=True):
        i, delta = domain_to_index_delta(c(x))
        regions[i.item()].append((delta, gt))

    # iterate over table regions and generate b
    b = empty(512, dtype=int32)
    for i, region in regions.items():
        start, end = i == 0, i == 511
        delta, gt = tensor(region).T
        base, slope = fit_points(a, delta, gt, start, end)

        # dims are initially base, slope, delta
        y = a(interp(base, slope, delta))
        torq_err = torq_error(y, gt)

        # endpoints must match exactly
        if start:
            valid = torq_err[..., :1] == 0
        elif end:
            valid = torq_err[..., -1:] == 0
        else:
            valid = ones(len(base), len(slope), 1, dtype=bool)

        # minimize max torq err
        max_torq_err = torq_err.max(-1, keepdim=True).values
        final_max_torq_err = max_torq_err[valid].min()
        valid &= max_torq_err == final_max_torq_err

        # minimize total torq error squared
        total_torq_err_squared = (torq_err**2).sum(-1, keepdim=True)
        final_total_torq_err_squared = total_torq_err_squared[valid].min()
        valid &= total_torq_err_squared == final_total_torq_err_squared

        # Anything valid at this point is good, so pick the first one
        base_i, slope_i = divmod(argmin((~valid).int()).item(), len(slope))
        b[i] = pack(base[base_i].reshape(1), slope[slope_i].reshape(1))

        # sanity checks...
        y = a(interp(*unpack(b[i]), delta))
        torq_err = torq_error(y, gt)
        assert torq_err.max() == final_max_torq_err
        assert (torq_err**2).sum() == final_total_torq_err_squared

    # print results for hardcoding in compiler
    print("a:", a)
    print('b:', b)
    print("c:", c)

    # summarize performance
    simulated = a(run_lut(b, c(domain)))
    torq_err = torq_error(simulated, image)
    print('torq_errs', torq_err.sort())
    print('torq_err max, rms', torq_err.max(), (torq_err**2).mean()**.5)
    failed = torq_err > 1e-2
    if failed.sum():
        print(80*'=')
        print('some values will fail test framework!')
        bad = domain[failed]
        i, _ = domain_to_index_delta(c(bad))
        print('failed LUT indicies', i.unique())
        extremes = bad[[0, -1]].view(bfloat16)
        print('these correspond to the approximate input range', extremes)
        print('and output range', f(extremes))
        print(80*'=')
    print()


if __name__ == "__main__":

    # comment as appropriate.

    from torch import sigmoid
    print("sigmoid of negative")
    approximate_function(sigmoid, negative=True)
    print("sigmoid of positive")
    approximate_function(sigmoid, negative=False)

    from torch import exp
    print("exp of negative")
    approximate_function(exp, negative=True)
    print("exp of positive")
    approximate_function(exp, negative=False)

    print("limited_reciprocal")
    def limited_reciprocal(x):
        x = x.clone()
        x[x < 1] = 1
        x[x > (2**12)] = 2**12
        return 1 / x
    approximate_function(limited_reciprocal, negative=False)
