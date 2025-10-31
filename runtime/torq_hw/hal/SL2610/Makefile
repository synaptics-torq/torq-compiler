# SPDX-License-Identifier: GPL-2.0-only
#

obj-$(CONFIG_SYNAPTICS_NPU) += syna_npu.o
syna_npu-y += src/torq_core_linux.o

HEADER_PATH += -I$(src)/inc/
ccflags-y += $(HEADER_PATH)
ccflags-y += -Werror
