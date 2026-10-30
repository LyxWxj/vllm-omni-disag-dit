# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Keep this package free of eager model imports. The pipeline registry imports
# the topology module directly, while the diffusion registry lazily imports the
# heavyweight Qwen-Image implementation in the worker process.
