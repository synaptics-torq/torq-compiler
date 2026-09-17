// SPDX-License-Identifier: GPL-2.0
// Copyright 2026 Synaptics Incorporated

#include <linux/module.h>
#include <linux/device.h>
#include <linux/devfreq.h>
#include <linux/pm_opp.h>
#include <linux/clk.h>
#include <linux/regulator/consumer.h>
#include <linux/platform_device.h>
#include <linux/pm_qos.h>
#include <linux/slab.h>
#include <linux/units.h>
#include <linux/workqueue.h>
#include <linux/err.h>
#include <linux/of.h>

#include "torq_core_linux.h"
#include "torq_devfreq.h"

#define TORQ_MAX_FREQ	1000000000UL

struct torq_opp_entry {
	unsigned long freq;
	unsigned long min_uV;
	unsigned long target_uV;
	unsigned long max_uV;
};

struct torq_devfreq {
	struct device *dev;
	struct clk *core_clk;
	struct clk *clk_high;
	struct clk *clk_low;
	struct regulator *vcore;
	struct devfreq *devfreq;
	struct torq_opp_entry *opp_table;

	struct devfreq_simple_ondemand_data ondemand_data;
	struct dev_pm_qos_request boost_min_freq_req;
	struct work_struct boost_release_work;

	unsigned int opp_count;
	unsigned long cur_freq;
	s32 boost_freq_khz;

	spinlock_t stats_lock;
	ktime_t last_sample;
	ktime_t busy_start;
	u64 busy_time_ns;
	unsigned int active_jobs;

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

	if (!tdf->vcore)
		return 0;

	ret = regulator_set_voltage_triplet(tdf->vcore,
					    min_uV,
					    target_uV,
					    max_uV);
	if (ret) {
		dev_dbg(tdf->dev,
			"failed to set voltage "
			"min=%lu target=%lu max=%lu uV: %d\n",
			min_uV, target_uV, max_uV, ret);
		return ret;
	}

	return 0;
}

static const struct torq_opp_entry *
torq_find_opp_entry(struct torq_devfreq *tdf, unsigned long freq)
{
	unsigned int i;

	for (i = 0; i < tdf->opp_count; i++) {
		if (tdf->opp_table[i].freq == freq)
			return &tdf->opp_table[i];
	}

	return NULL;
}

static bool torq_devfreq_is_busy(struct torq_devfreq *tdf)
{
	unsigned long flags;
	bool busy;

	spin_lock_irqsave(&tdf->stats_lock, flags);
	busy = tdf->active_jobs != 0;
	spin_unlock_irqrestore(&tdf->stats_lock, flags);

	return busy;
}

static void torq_devfreq_release_boost(struct work_struct *work)
{
	struct torq_devfreq *tdf;
	unsigned long flags;
	bool busy;
	int ret;

	tdf = container_of(work, struct torq_devfreq, boost_release_work);

	spin_lock_irqsave(&tdf->stats_lock, flags);
	busy = tdf->active_jobs != 0;
	spin_unlock_irqrestore(&tdf->stats_lock, flags);
	if (busy)
		return;

	ret = dev_pm_qos_update_request(&tdf->boost_min_freq_req,
					PM_QOS_MIN_FREQUENCY_DEFAULT_VALUE);
	if (ret < 0)
		dev_warn(tdf->dev, "failed to release frequency boost: %d\n",
			 ret);
}

static void torq_devfreq_remove_boost_request(void *data)
{
	struct torq_devfreq *tdf = data;

	cancel_work_sync(&tdf->boost_release_work);
	if (dev_pm_qos_request_active(&tdf->boost_min_freq_req))
		dev_pm_qos_remove_request(&tdf->boost_min_freq_req);
}

static int torq_devfreq_target(struct device *dev,
			       unsigned long *freq,
			       u32 flags)
{
	struct torq_devfreq *tdf;
	struct dev_pm_opp *opp;
	struct clk *target_parent;
	const struct torq_opp_entry *opp_entry;
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

	/* The NPU clock parent must not change while a job is running. */
	if (torq_devfreq_is_busy(tdf)) {
		*freq = old_freq;
		return 0;
	}

	opp = devfreq_recommended_opp(dev, &target_freq, flags);
	if (IS_ERR(opp)) {
		ret = PTR_ERR(opp);

		dev_err(dev,
			"failed to get recommended OPP for %lu Hz: %d\n",
			requested_freq, ret);
		return ret;
	}

	dev_pm_opp_put(opp);

	if (tdf->vcore) {
		opp_entry = torq_find_opp_entry(tdf, target_freq);
		if (!opp_entry) {
			dev_err(dev,
				"no voltage data for OPP %lu Hz\n",
				target_freq);
			return -ENOENT;
		}

		min_uV = opp_entry->min_uV;
		target_uV = opp_entry->target_uV;
		max_uV = opp_entry->max_uV;
	}

	if (target_freq >= TORQ_MAX_FREQ)
		target_parent = tdf->clk_high;
	else
		target_parent = tdf->clk_low;

	if (target_freq > old_freq) {
		if (tdf->vcore) {
			ret = torq_set_opp_voltage(tdf,
						min_uV,
						target_uV,
						max_uV);
			if (ret)
				return ret;
		}

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
			dev_dbg(dev,
				"failed to switch clock parent for "
				"frequency decrease: %d\n",
				ret);
			return ret;
		}
		if (tdf->vcore) {
			ret = torq_set_opp_voltage(tdf,
						min_uV,
						target_uV,
						max_uV);
			if (ret) {
				dev_dbg(dev,
					"clock lowered to %lu Hz, but voltage "
					"could not be relaxed to %lu/%lu/%lu uV\n",
					target_freq,
					min_uV,
					target_uV,
					max_uV);
			}
		}
	}
	actual_freq = clk_get_rate(tdf->core_clk);

	tdf->cur_freq = actual_freq;
	*freq = actual_freq;

	if (actual_freq != target_freq) {
		dev_dbg(dev,
			 "frequency mismatch: "
			 "requested=%lu target=%lu actual=%lu Hz\n",
			 requested_freq,
			 target_freq,
			 actual_freq);
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
	unsigned long flags;
	ktime_t now;
	u64 busy_ns;
	u64 total_ns;

	tdf = torq_get_devfreq_data(dev);
	if (!tdf || !stat)
		return -EINVAL;

	now = ktime_get();

	spin_lock_irqsave(&tdf->stats_lock, flags);
	total_ns = ktime_to_ns(ktime_sub(now, tdf->last_sample));
	busy_ns = tdf->busy_time_ns;

	if (tdf->active_jobs) {
		busy_ns += ktime_to_ns(ktime_sub(now, tdf->busy_start));
		tdf->busy_start = now;
	}
	tdf->busy_time_ns = 0;
	tdf->last_sample = now;
	spin_unlock_irqrestore(&tdf->stats_lock, flags);

	if (busy_ns > total_ns)
		busy_ns = total_ns;

	memset(stat, 0, sizeof(*stat));
	stat->busy_time = busy_ns;
	stat->total_time = total_ns;
	stat->current_frequency = clk_get_rate(tdf->core_clk);

	return 0;
}

static struct devfreq_dev_profile torq_devfreq_profile = {
	.polling_ms = 100,
	.target = torq_devfreq_target,
	.get_dev_status = torq_devfreq_get_status,
	.get_cur_freq = torq_devfreq_get_cur_freq,
};

static int torq_build_opp_table(struct torq_devfreq *tdf)
{
	struct device *dev = tdf->dev;
	struct dev_pm_opp *opp;
	unsigned long freq = 0;
	int count;
	int i;
	int ret;

	count = dev_pm_opp_get_opp_count(dev);
	if (count <= 0)
		return count ? count : -ENODEV;

	tdf->opp_table = devm_kcalloc(dev,
				      count,
				      sizeof(*tdf->opp_table),
				      GFP_KERNEL);
	if (!tdf->opp_table)
		return -ENOMEM;

	for (i = 0; i < count; i++) {
		struct torq_opp_entry *entry = &tdf->opp_table[i];

		opp = dev_pm_opp_find_freq_ceil(dev, &freq);
		if (IS_ERR(opp)) {
			ret = PTR_ERR(opp);
			dev_err(dev,
				"failed to find OPP %d: %d\n",
				i, ret);
			return ret;
		}

		entry->freq = freq;

		ret = torq_opp_get_voltages(dev,
					    opp,
					    &entry->min_uV,
					    &entry->target_uV,
					    &entry->max_uV);

		dev_pm_opp_put(opp);

		if (ret)
			return ret;

		if (freq == ULONG_MAX)
			break;

		freq++;
	}

	tdf->opp_count = count;
	return 0;
}

int torq_devfreq_job_start(struct torq_module *torq_dev)
{
	struct torq_devfreq *tdf;
	unsigned long flags;
	ktime_t now;
	int ret;

	if (!torq_dev)
		return -EINVAL;

	tdf = torq_dev->devfreq_data;
	if (!tdf)
		return 0;

	/* Serialize against boost release from the preceding completion. */
	cancel_work_sync(&tdf->boost_release_work);

	ret = dev_pm_qos_update_request(&tdf->boost_min_freq_req,
					tdf->boost_freq_khz);
	if (ret < 0) {
		dev_err(tdf->dev, "failed to request frequency boost: %d\n",
			ret);
		return ret;
	}

	now = ktime_get();

	spin_lock_irqsave(&tdf->stats_lock, flags);

	if (tdf->active_jobs++ == 0)
		tdf->busy_start = now;

	spin_unlock_irqrestore(&tdf->stats_lock, flags);

	return 0;
}
EXPORT_SYMBOL_GPL(torq_devfreq_job_start);

void torq_devfreq_job_complete(struct torq_module *torq_dev)
{
	struct torq_devfreq *tdf;
	unsigned long flags;
	ktime_t now;
	bool release_boost = false;

	if (!torq_dev)
		return;

	tdf = torq_dev->devfreq_data;
	if (!tdf)
		return;

	now = ktime_get();

	spin_lock_irqsave(&tdf->stats_lock, flags);

	if (!tdf->active_jobs) {
		spin_unlock_irqrestore(&tdf->stats_lock, flags);
		return;
	}

	tdf->active_jobs--;

	if (!tdf->active_jobs) {
		tdf->busy_time_ns += ktime_to_ns(ktime_sub(now,
							      tdf->busy_start));
		release_boost = true;
	}

	spin_unlock_irqrestore(&tdf->stats_lock, flags);

	if (release_boost)
		schedule_work(&tdf->boost_release_work);
}
EXPORT_SYMBOL_GPL(torq_devfreq_job_complete);

int torq_devfreq_init(struct torq_module *torq_dev)
{
	struct torq_devfreq *tdf;
	struct dev_pm_opp *opp;
	struct device *dev;
	unsigned long boost_freq = ULONG_MAX;
	int ret;

	dev = &torq_dev->pdev->dev;

	tdf = devm_kzalloc(dev, sizeof(*tdf), GFP_KERNEL);
	if (!tdf)
		return -ENOMEM;

	tdf->dev = dev;

	spin_lock_init(&tdf->stats_lock);
	INIT_WORK(&tdf->boost_release_work, torq_devfreq_release_boost);
	tdf->last_sample = ktime_get();
	tdf->busy_start = 0;
	tdf->busy_time_ns = 0;
	tdf->active_jobs = 0;

	tdf->core_clk = devm_clk_get_enabled(dev, "core");
	if (IS_ERR(tdf->core_clk))
		return dev_err_probe(dev,
					PTR_ERR(tdf->core_clk),
					"failed to get and enable core clock\n");

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

	tdf->vcore = devm_regulator_get_optional(dev, "npu");
	if (IS_ERR(tdf->vcore)) {
		ret = PTR_ERR(tdf->vcore);

		if (ret == -ENODEV) {
			tdf->vcore = NULL;
			dev_dbg(dev, "no NPU regulator configured\n");
		} else {
			return dev_err_probe(dev, ret,
						"failed to get NPU regulator\n");
		}
	}

	ret = devm_pm_opp_of_add_table(dev);
	if (ret == -ENODEV) {
		dev_dbg(dev, "no OPP table, devfreq disabled\n");
		return 0;
	}

	if (ret)
		return dev_err_probe(dev, ret,
					"failed to add OPP table\n");

	opp = dev_pm_opp_find_freq_floor(dev, &boost_freq);
	if (IS_ERR(opp))
		return dev_err_probe(dev, PTR_ERR(opp),
					"failed to find maximum OPP\n");
	dev_pm_opp_put(opp);

	if (boost_freq > (unsigned long)S32_MAX * HZ_PER_KHZ)
		return dev_err_probe(dev, -ERANGE,
					"maximum OPP exceeds PM QoS range\n");
	tdf->boost_freq_khz = DIV_ROUND_UP(boost_freq, HZ_PER_KHZ);

	if (tdf->vcore) {
		ret = torq_build_opp_table(tdf);
		if (ret)
			return dev_err_probe(dev, ret,
						"failed to build cached OPP table\n");
	}

	tdf->cur_freq = clk_get_rate(tdf->core_clk);

	torq_dev->devfreq_data = tdf;

	tdf->ondemand_data.upthreshold = 50;
	tdf->ondemand_data.downdifferential = 30;

	tdf->devfreq = devm_devfreq_add_device(dev,
						&torq_devfreq_profile,
						"simple_ondemand",
						&tdf->ondemand_data);
	if (IS_ERR(tdf->devfreq)) {
		ret = PTR_ERR(tdf->devfreq);
		torq_dev->devfreq_data = NULL;

		return dev_err_probe(dev,
				     ret,
				     "failed to add devfreq device\n");
	}

	ret = dev_pm_qos_add_request(dev, &tdf->boost_min_freq_req,
				     DEV_PM_QOS_MIN_FREQUENCY,
				     PM_QOS_MIN_FREQUENCY_DEFAULT_VALUE);
	if (ret < 0) {
		torq_dev->devfreq_data = NULL;
		return dev_err_probe(dev, ret,
					"failed to add frequency boost request\n");
	}

	ret = devm_add_action_or_reset(dev,
				       torq_devfreq_remove_boost_request, tdf);
	if (ret) {
		torq_dev->devfreq_data = NULL;
		return ret;
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

	if (!torq_dev)
		return;

	tdf = torq_dev->devfreq_data;
	if (!tdf)
		return;

	torq_devfreq_remove_boost_request(tdf);
	torq_dev->devfreq_data = NULL;
}
EXPORT_SYMBOL_GPL(torq_devfreq_exit);
