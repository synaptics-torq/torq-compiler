# NPU Load Status

Torq exposes cumulative NPU inference time through a Linux `sysfs` entry created by the kernel driver. This is the supported way to show NPU load status on Torq-enabled Astra targets: user-space tools can sample NPU activity and compute a utilization percentage over a fixed time window.

## Sysfs Path

On Torq-enabled Astra targets, the driver creates the statistics group under:

```shell
/sys/class/misc/torq/statistics/
```

The currently exposed counter is:

```shell
/sys/class/misc/torq/statistics/torq_inference_time
```

## `torq_inference_time`

This file contains the cumulative time spent executing NPU inference since the last reset. The value is a 64-bit integer expressed in microseconds.

Example:

```shell
# cat /sys/class/misc/torq/statistics/torq_inference_time
32233264
```

## Clearing the Counter

Writing to `torq_inference_time` resets the accumulated value to `0`.

Example:

```shell
# cat /sys/class/misc/torq/statistics/torq_inference_time
32233264
# echo 0 > /sys/class/misc/torq/statistics/torq_inference_time
# cat /sys/class/misc/torq/statistics/torq_inference_time
0
```

## Calculating NPU Load

Because the counter is cumulative, NPU load is typically measured over a fixed observation window:

1. Reset the counter.
2. Wait for a known time window.
3. Read the accumulated inference time.
4. Convert it to a percentage of the observation window.

For a 1 second window, compute NPU load as:

```text
npu_load_percent = (inference_time_us * 100) / 1000000
```

Example shell workflow:

```shell
STAT_FILE=/sys/class/misc/torq/statistics/torq_inference_time
WINDOW_US=1000000

echo 0 > "$STAT_FILE"
usleep "$WINDOW_US"
inference_time_us=$(cat "$STAT_FILE")
npu_load=$((inference_time_us * 100 / WINDOW_US))
echo "Average NPU usage: ${npu_load}%"
```

## Notes

- This counter reports cumulative NPU execution time, not end-to-end host latency.
- To obtain a meaningful utilization percentage, the counter must be reset before each measurement window.
- Access to `sysfs` statistics may require elevated privileges depending on the target system configuration.
- If the statistics group is not present, verify that the target is running a driver version that includes the merged sysfs statistics support.
