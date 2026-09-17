/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright 2026 Synaptics Incorporated
 */

#ifndef __TORQ_DEVFREQ_H__
#define __TORQ_DEVFREQ_H__

#include <linux/platform_device.h>

struct torq_module;

int torq_devfreq_init(struct torq_module *torq_dev);
void torq_devfreq_exit(struct torq_module *torq_dev);
int torq_devfreq_job_start(struct torq_module *torq_dev);
void torq_devfreq_job_complete(struct torq_module *torq_dev);
#endif /* __TORQ_DEVFREQ_H__ */
