// SPDX-License-Identifier: GPL-2.0
// Copyright 2026 Synaptics Incorporated

#include <linux/module.h>
#include <linux/device.h>
#include <linux/devfreq.h>
#include <linux/pm_opp.h>
#include <linux/clk.h>
#include <linux/regulator/consumer.h>
#include <linux/platform_device.h>
#include <linux/slab.h>
#include <linux/err.h>
#include <linux/of.h>

#include "torq_core_linux.h"
#include "torq_devfreq.h"

#define TORQ_MAX_FREQ	1000000000UL

struct torq_devfreq {
	struct device *dev;
	struct clk *core_clk;
	struct clk *clk_high;
	struct clk *clk_low;
	struct regulator *vcore;
	struct devfreq *devfreq;
	unsigned long cur_freq;
};

static struct torq_devfreq *torq_get_devfreq_data(struct device *dev)
{
	struct torq_module *torq_dev;

	if (!dev)
		return NULL;

	torq_dev = dev_get_drvdata(dev);
	if (!torq_dev)
		return NULL;

	return torq_dev->devfreq_data;
}

static int torq_opp_get_voltages(struct device *dev,
				 struct dev_pm_opp *opp,
				 unsigned long *min_uV,
				 unsigned long *target_uV,
				 unsigned long *max_uV)
{
	struct device_node *opp_np;
	u32 voltages[3];
	int count;
	int ret;

	if (!dev || !opp || !min_uV || !target_uV || !max_uV)
		return -EINVAL;

	opp_np = dev_pm_opp_get_of_node(opp);
	if (!opp_np) {
		dev_err(dev, "failed to get OPP device-tree node\n");
		return -ENODEV;
	}

	count = of_property_count_u32_elems(opp_np, "opp-microvolt");
	if (count < 0) {
		ret = count;

		dev_err(dev,
			"failed to count opp-microvolt values: %d\n",
			ret);
		goto out_put_node;
	}

	if (count == 1) {
		ret = of_property_read_u32(opp_np,
					   "opp-microvolt",
					   &voltages[0]);
		if (ret) {
			dev_err(dev,
				"failed to read opp-microvolt: %d\n",
				ret);
			goto out_put_node;
		}

		/*
		 * Single-value format:
		 *
		 * opp-microvolt = <target>;
		 */
		*target_uV = voltages[0];
		*min_uV = voltages[0];
		*max_uV = voltages[0];
	} else if (count == 3) {
		ret = of_property_read_u32_array(opp_np,
						 "opp-microvolt",
						 voltages,
						 ARRAY_SIZE(voltages));
		if (ret) {
			dev_err(dev,
				"failed to read opp-microvolt triplet: %d\n",
				ret);
			goto out_put_node;
		}

		/*
		 * OPP triplet format:
		 *
		 * opp-microvolt = <target min max>;
		 */
		*target_uV = voltages[0];
		*min_uV = voltages[1];
		*max_uV = voltages[2];
	} else {
		dev_err(dev,
			"unsupported opp-microvolt element count: %d\n",
			count);

		ret = -EINVAL;
		goto out_put_node;
	}

	if (!*target_uV || !*min_uV || !*max_uV) {
		dev_err(dev,
			"invalid zero OPP voltage: "
			"min=%lu target=%lu max=%lu uV\n",
			*min_uV, *target_uV, *max_uV);

		ret = -EINVAL;
		goto out_put_node;
	}

	if (*min_uV > *target_uV || *target_uV > *max_uV) {
		dev_err(dev,
			"invalid OPP voltage range: "
			"min=%lu target=%lu max=%lu uV\n",
			*min_uV, *target_uV, *max_uV);

		ret = -EINVAL;
		goto out_put_node;
	}

	ret = 0;

out_put_node:
	of_node_put(opp_np);

	return ret;
}

static int torq_set_opp_voltage(struct torq_devfreq *tdf,
				unsigned long min_uV,
				unsigned long target_uV,
				unsigned long max_uV)
{
	int ret;

	ret = regulator_set_voltage_triplet(tdf->vcore,
					    min_uV,
					    target_uV,
					    max_uV);
	if (ret) {
		dev_err(tdf->dev,
			"failed to set voltage "
			"min=%lu target=%lu max=%lu uV: %d\n",
			min_uV, target_uV, max_uV, ret);
		return ret;
	}

	dev_dbg(tdf->dev,
		"requested voltage "
		"min=%lu target=%lu max=%lu uV\n",
		min_uV, target_uV, max_uV);

	return 0;
}

static int torq_devfreq_target(struct device *dev,
			       unsigned long *freq,
			       u32 flags)
{
	struct torq_devfreq *tdf;
	struct dev_pm_opp *opp;
	struct clk *target_parent;
	unsigned long requested_freq;
	unsigned long old_freq;
	unsigned long target_freq;
	unsigned long min_uV;
	unsigned long target_uV;
	unsigned long max_uV;
	unsigned long actual_freq;
	int ret;

	tdf = torq_get_devfreq_data(dev);
	if (!tdf || !freq)
		return -EINVAL;

	requested_freq = *freq;
	old_freq = clk_get_rate(tdf->core_clk);
	target_freq = requested_freq;

	opp = devfreq_recommended_opp(dev, &target_freq, flags);
	if (IS_ERR(opp)) {
		ret = PTR_ERR(opp);

		dev_err(dev,
			"failed to get recommended OPP for %lu Hz: %d\n",
			requested_freq, ret);
		return ret;
	}

	ret = torq_opp_get_voltages(dev,
				    opp,
				    &min_uV,
				    &target_uV,
				    &max_uV);

	dev_pm_opp_put(opp);

	if (ret)
		return ret;

	if (target_freq >= TORQ_MAX_FREQ)
		target_parent = tdf->clk_high;
	else
		target_parent = tdf->clk_low;

	dev_info(dev,
		 "OPP request: requested=%lu target=%lu old=%lu Hz, "
		 "voltage=%lu/%lu/%lu uV\n",
		 requested_freq,
		 target_freq,
		 old_freq,
		 min_uV,
		 target_uV,
		 max_uV);

	if (target_freq > old_freq) {
		ret = torq_set_opp_voltage(tdf,
					   min_uV,
					   target_uV,
					   max_uV);
		if (ret)
			return ret;

		ret = clk_set_parent(tdf->core_clk, target_parent);
		if (ret) {
			dev_err(dev,
				"failed to switch clock parent for "
				"frequency increase: %d\n",
				ret);
			return ret;
		}
	} else if (target_freq < old_freq) {
		ret = clk_set_parent(tdf->core_clk, target_parent);
		if (ret) {
			dev_err(dev,
				"failed to switch clock parent for "
				"frequency decrease: %d\n",
				ret);
			return ret;
		}

		ret = torq_set_opp_voltage(tdf,
					   min_uV,
					   target_uV,
					   max_uV);
		if (ret) {
			dev_warn(dev,
				 "clock lowered to %lu Hz, but voltage "
				 "could not be relaxed to %lu/%lu/%lu uV\n",
				 target_freq,
				 min_uV,
				 target_uV,
				 max_uV);
		}
	} else {
		if (clk_get_parent(tdf->core_clk) != target_parent) {
			ret = clk_set_parent(tdf->core_clk,
					     target_parent);
			if (ret) {
				dev_err(dev,
					"failed to correct clock parent: %d\n",
					ret);
				return ret;
			}
		}

		ret = torq_set_opp_voltage(tdf,
					   min_uV,
					   target_uV,
					   max_uV);
		if (ret) {
			dev_warn(dev,
				 "frequency unchanged at %lu Hz, but "
				 "failed to update voltage request\n",
				 target_freq);
		}
	}

	actual_freq = clk_get_rate(tdf->core_clk);

	tdf->cur_freq = actual_freq;
	*freq = actual_freq;

	if (actual_freq != target_freq) {
		dev_warn(dev,
			 "frequency mismatch: "
			 "requested=%lu target=%lu actual=%lu Hz\n",
			 requested_freq,
			 target_freq,
			 actual_freq);
	} else {
		dev_info(dev,
			 "frequency changed: %lu -> %lu Hz, "
			 "voltage=%lu/%lu/%lu uV\n",
			 old_freq,
			 actual_freq,
			 min_uV,
			 target_uV,
			 max_uV);
	}
	return 0;
}

static int torq_devfreq_get_cur_freq(struct device *dev,
				     unsigned long *freq)
{
	struct torq_devfreq *tdf;

	tdf = torq_get_devfreq_data(dev);
	if (!tdf || !freq)
		return -EINVAL;

	tdf->cur_freq = clk_get_rate(tdf->core_clk);
	*freq = tdf->cur_freq;

	return 0;
}

static int torq_devfreq_get_status(struct device *dev,
				   struct devfreq_dev_status *stat)
{
	struct torq_devfreq *tdf;

	tdf = torq_get_devfreq_data(dev);
	if (!tdf || !stat)
		return -EINVAL;

	memset(stat, 0, sizeof(*stat));

	stat->current_frequency = clk_get_rate(tdf->core_clk);
	stat->busy_time = 0;
	stat->total_time = 0;

	return 0;
}

static struct devfreq_dev_profile torq_devfreq_profile = {
	.polling_ms = 100,
	.target = torq_devfreq_target,
	.get_dev_status = torq_devfreq_get_status,
	.get_cur_freq = torq_devfreq_get_cur_freq,
};

int torq_devfreq_init(struct torq_module *torq_dev)
{
	struct torq_devfreq *tdf;
	struct device *dev;
	int ret;

	if (!torq_dev || !torq_dev->pdev)
		return -EINVAL;

	dev = &torq_dev->pdev->dev;

	tdf = devm_kzalloc(dev, sizeof(*tdf), GFP_KERNEL);
	if (!tdf)
		return -ENOMEM;

	tdf->dev = dev;

	tdf->core_clk = devm_clk_get(dev, "core");
	if (IS_ERR(tdf->core_clk))
		return dev_err_probe(dev,
				     PTR_ERR(tdf->core_clk),
				     "failed to get core clock\n");

	tdf->clk_high = devm_clk_get(dev, "clk_high");
	if (IS_ERR(tdf->clk_high))
		return dev_err_probe(dev,
				     PTR_ERR(tdf->clk_high),
				     "failed to get high clock\n");

	tdf->clk_low = devm_clk_get(dev, "clk_low");
	if (IS_ERR(tdf->clk_low))
		return dev_err_probe(dev,
				     PTR_ERR(tdf->clk_low),
				     "failed to get low clock\n");

	tdf->vcore = devm_regulator_get(dev, "npu");
	if (IS_ERR(tdf->vcore))
		return dev_err_probe(dev,
				     PTR_ERR(tdf->vcore),
				     "failed to get npu regulator\n");

	ret = devm_pm_opp_of_add_table(dev);
	if (ret)
		return dev_err_probe(dev,
				     ret,
				     "failed to add OPP table\n");

	tdf->cur_freq = clk_get_rate(tdf->core_clk);

	torq_dev->devfreq_data = tdf;

	tdf->devfreq = devm_devfreq_add_device(dev,
					      &torq_devfreq_profile,
					      "performace",
					      NULL);
	if (IS_ERR(tdf->devfreq)) {
		ret = PTR_ERR(tdf->devfreq);
		torq_dev->devfreq_data = NULL;

		return dev_err_probe(dev,
				     ret,
				     "failed to add devfreq device\n");
	}

	dev_info(dev,
		 "TORQ devfreq initialized: current=%lu Hz, "
		 "low-parent=%lu Hz, high-parent=%lu Hz\n",
		 tdf->cur_freq,
		 clk_get_rate(tdf->clk_low),
		 clk_get_rate(tdf->clk_high));

	return 0;
}
EXPORT_SYMBOL_GPL(torq_devfreq_init);

void torq_devfreq_exit(struct torq_module *torq_dev)
{
	struct torq_devfreq *tdf;

	if (!torq_dev || !torq_dev->pdev)
		return;

	tdf = torq_dev->devfreq_data;
	if (!tdf)
		return;

	torq_dev->devfreq_data = NULL;
}
EXPORT_SYMBOL_GPL(torq_devfreq_exit);
