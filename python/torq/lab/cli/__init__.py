# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command-line presentation of ``torq-lab``.

``parser`` builds the argparse tree and translates CLI arguments into domain
objects; ``commands`` dispatches the subcommands and hosts the installed CLI
``main()``; ``interactive`` is the no-argument prompt workflow; ``output``
holds the plan/summary view models and their renderers.
"""
