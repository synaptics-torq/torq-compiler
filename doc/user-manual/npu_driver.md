# NPU Kernel Driver

On Torq-enabled Astra targets (SL2610), the NPU is driven by the `syna_npu` kernel module, which
the Astra SDK deploys as the `torq-kernel-module` package. The Torq runtime talks to it through
`/dev/torq`. This page describes the driver settings that change NPU performance.

## NPU Frequency Boost

The driver scales the NPU clock with Linux `devfreq` (the SL2610 NPU runs at 800 MHz or 1 GHz).
To run inferences at the highest frequency, it boosts the NPU to the maximum frequency:

- when the runtime loads a model on the NPU, so that the first inference doesn't wait for the
  frequency change either;
- while a job runs on the NPU;
- for `boost_hold_ms` milliseconds after the last job completes, so that the next job doesn't
  wait for the clock and voltage to be raised again (about 3.6 ms on SL2610).

After that, the `devfreq` governor chooses the frequency from the NPU load again.

The default of 100 ms keeps the boost between the jobs of an inference and between consecutive
inferences (for example a 30 fps camera stream), and needs no change when loading the module.

### `boost_hold_ms` Module Parameter

| Parameter | Default | Description |
|---|---|---|
| `boost_hold_ms` | `100` | Time in ms the frequency boost is kept after the last job completes. `0` releases the boost as soon as a job completes. |

Set it when loading the module:

```shell
# modprobe syna_npu boost_hold_ms=200
```

or persistently, in a `modprobe.d` file:

```shell
# echo "options syna_npu boost_hold_ms=200" > /etc/modprobe.d/syna_npu.conf
```

or at runtime (it applies from the next job completion):

```shell
# cat /sys/module/syna_npu/parameters/boost_hold_ms
100
# echo 200 > /sys/module/syna_npu/parameters/boost_hold_ms
```

A longer hold avoids the frequency change for workloads with longer gaps between inferences, at
the cost of keeping the NPU at its highest frequency and voltage for longer when idle.

The current frequency and the frequency changes are visible in the `devfreq` `sysfs` entries of
the NPU device, for example:

```shell
# cat /sys/class/devfreq/*.synpu/cur_freq
# cat /sys/class/devfreq/*.synpu/trans_stat
```
