#!/usr/bin/env python3
from warnings import catch_warnings, simplefilter

from torch import arange, empty, exp, bfloat16, set_flush_denormal, \
    logical_not, int32, int16, int8, tensor, where, softmax, stack, int64
from torch.nn import Module

# Ensure subnormal/denormal numbers to flush to zero
assert set_flush_denormal(True), "system doesn't support flushing denormal"


IN_SHAPE = 1, 100, 16

# useful integers for bit hacking.  '_'s split into sign_exponent_mantissa.
sign = 0b1_00000000_0000000  # sign bit
expo = 0b0_11111111_0000000  # exponent bits
mant = 0b0_00000000_1111111  # mantissa bits
inf  = 0b0_11111111_0000000  # infinity
nan  = 0b1_11111111_1111111  # cannonical nan (torch)
#nan = 0b0_11111111_1000000  # cannonical nan (tensorflow)
#one = 0b0_01111111_0000000  # FYI


# Section 1.9 of the tosa 1.0.0 draft specification states "At least
# one NaN encoding must be supported.".  We can at least simplify by
# dropping support for miscelaneous nans.  Also in that same section:
# "Subnormal values must either be supported or flushed to
# sign-preserved zero."  I had an implementation of 1/x which was bit
# accurate... or at least I did before tensorflow updated their
# implementation of bfloat division.  The new one better supports
# subnormal output (old one only dealt with subnormal inputs
# correctly).  We are trying to squeeze water from rocks here, so lets
# just flush subnormals.
class OneOverX(Module):

    def __init__(self):
        super().__init__()

        # generate bfloat values with sequential integer representations
        all_bfloat_values = arange(2**16).to(int16).view(bfloat16)

        # calculate the inverse
        all_inverse = (1 / all_bfloat_values).view(int16)

        # populate the lut
        self.lut = empty(128, dtype=int8)
        seen = set()
        for val_int, inv_int in enumerate(all_inverse):

            # subnormal and inf/nan excluded
            if not ((val_int ^ expo) & expo) or not (val_int & expo):
                continue
            # super large numbers go to zero.  ignore them
            if val_int & (expo | mant) > 0b0_11111101_0000000:
                continue

            # pull out input and output mantissas
            val_mant = val_int & mant
            inv_mant = inv_int & mant

            # if seen already, validate that the value is what we expect
            if val_mant in seen:
                assert inv_mant == self.lut[val_mant].to(int16)
            else:
                # otherwise, add the value to our table
                self.lut[val_mant] = inv_mant
                seen.add(val_mant)

    def forward(self, x):
        # get readx for integer operations
        x = x.view(int16)

        # save sign bit
        x_sign = sign & x

        # remove sign bit
        #x = x ^ x_sign
        # iree struggles to lower xor.  use and instead
        x = x & (~sign)

        x_mant = x & mant
        is_nan = (x > inf).to(int16)
        is_subnormal = logical_not(expo & x)
        # 'big' here means so big that 1/x maps to zero
        is_not_big = logical_not(x > 0b0_11111101_0000000).to(int16)
        computed_expo = 0b0_11111101_0000000 - (x & expo)
        computed_expo += logical_not(x_mant) * 0b0_00000001_0000000
        # index needs to be int64 to work around nonsensical lowering bug
        computed = (self.lut[x_mant.to(int64)] | computed_expo) * is_not_big
        computed = computed.view(bfloat16) + tensor(is_subnormal * inf,
                                                    dtype=int16
                                                    ).view(bfloat16)
        return (computed.view(int16) | (is_nan * nan) | x_sign).view(bfloat16)


# EX = e^x.  This algo can be significantly simplified for softmax
# since we subtract the max.  Hence, we only need the lut for the
# negative input.  But for completion, here is a full e^x
class EX(Module):
    mantissa_bit_masks = tensor([2**i for i in range(7, -1, -1)], dtype=int16)
    bit_offsets = arange(7, -1, -1)
    neg_exp_lut = exp(tensor(7 * [0] + [-2**i for i in range(-127, 7)] + (128 - (7 - 1)) * [-2**7]
                             )).to(bfloat16)
    pos_exp_lut = exp(tensor(7 * [0] + [ 2**i for i in range(-127, 7)] + (128 - (7 - 1)) * [ 2**7]
                             )).to(bfloat16)

    def forward(self, x):
        x = x.view(int16)
        x_sign = x & sign

        #x = x ^ x_sign
        # iree struggles to lower xor giving:
        #EX.mlir:13:10: error: failed to legalize operation 'torch.operator' that was explicitly marked illegal
        #    %8 = torch.operator "torch.aten.__xor__.Tensor"(%6, %7) : (!torch.vtensor<[8192,8],si16>, !torch.vtensor<[8192,8],si16>) -> !torch.vtensor<[8192,8],si16>
        # use and instead
        x = x & (~sign)

        is_nan = (x > inf).to(int16)
        x_expo = (x & expo) >> 7

        # define "signifacand" to be mantissa with leading zero
        #x_sgnf = (x & mant) | 0b10000000
        # The above fails because, ironically, it seems or only lowers
        # succesfully with torch int16, not python int, and and only
        # works with python int, not torch int16.  Use the below instead
        x_sgnf = (x & mant) | tensor(0b10000000, dtype=int16)
        lut_index = x_expo.reshape(*x_expo.shape, 1) + self.bit_offsets

        #mask = (x_sgnf.reshape(*x_sgnf.shape, 1) & self.mantissa_bit_masks
        #         ).to(bool)
        # When making this work, you should think about it like the
        # commented line above.  However, tensor &'s don't lower to
        # linalg now, so force scalar &'s by casting to python int.
        lut_index *= stack([(x_sgnf & int(bit_mask)).to(bool)
                            for bit_mask in self.mantissa_bit_masks],
                           -1)

        # ternary operator
        computed = where(x_sign[..., None].to(bool),
                         self.neg_exp_lut[lut_index],
                         self.pos_exp_lut[lut_index]).prod(-1).view(int16)
        return (computed | (is_nan * nan)).view(bfloat16)


# A slightly more simple version.  Just handles negative inputs
class NegEX(Module):
    mantissa_bit_masks = tensor([2**i for i in range(7, -1, -1)], dtype=int16)
    bit_offsets = arange(7, -1, -1)
    neg_exp_lut = exp(tensor(7 * [0] + [-2**i for i in range(-127, 7)] + (128 - (7 - 1)) * [-2**7]
                             )).to(bfloat16)

    def forward(self, x):
        x = x.view(int16)

        #x = x ^ x_sign
        # iree struggles to lower xor giving:
        #EX.mlir:13:10: error: failed to legalize operation 'torch.operator' that was explicitly marked illegal
        #    %8 = torch.operator "torch.aten.__xor__.Tensor"(%6, %7) : (!torch.vtensor<[8192,8],si16>, !torch.vtensor<[8192,8],si16>) -> !torch.vtensor<[8192,8],si16>
        # use and instead
        x = x & (~sign)

        is_nan = (x > inf).to(int16)
        x_expo = (x & expo) >> 7

        # define "signifacand" to be mantissa with leading zero
        #x_sgnf = (x & mant) | 0b10000000
        # The above fails because, ironically, it seems or only lowers
        # succesfully with torch int16, not python int, and and only
        # works with python int, not torch int16.  Use the below instead
        x_sgnf = (x & mant) | tensor(0b10000000, dtype=int16)
        lut_index = x_expo.reshape(*x_expo.shape, 1) + self.bit_offsets

        #mask = (x_sgnf.reshape(*x_sgnf.shape, 1) & self.mantissa_bit_masks
        #         ).to(bool)
        # When making this work, you should think about it like the
        # commented line above.  However, tensor &'s don't lower to
        # linalg now, so force scalar &'s by casting to python int.
        lut_index *= stack([(x_sgnf & int(bit_mask)).to(bool)
                            for bit_mask in self.mantissa_bit_masks],
                           -1)

        # no ternary operator
        computed = self.neg_exp_lut[lut_index].prod(-1).view(int16)
        return (computed | (is_nan * nan)).view(bfloat16)

        lut_index *= (x_sgnf[..., None] & self.mantissa_bit_masks).to(bool)


class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.ex = NegEX()
        self.one_over_x = OneOverX()

    def forward(self, x):
        x -= x.max(-1, keepdims=True).values
        ex = self.ex(x)
        norm = ex.sum(-1, keepdims=True)
        return self.one_over_x(norm) * ex


if __name__ == "__main__":

    print('For best results, set environment variable '
          'IREE_BUILD_DIR to latest iree.')
    exec(open(__file__.rsplit('/', 1)[0] + '/lib.py').read())
    with catch_warnings():
        simplefilter("ignore")
        test_and_save(OneOverX(),
                      lambda x: (1 / x).to(bfloat16))
        test_and_save(EX(),
                      lambda x: exp(x),
                      bittol=3)
        test_and_save(NegEX(),
                      lambda x: where(x < 0, exp(x), exp(-x)),
                      bittol=3)
        test_and_save(Softmax(),
                      lambda x: softmax(x, -1),
                      bittol=2)
    print('done')
