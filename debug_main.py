# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
Debug-friendly entry point for model training. This version runs in single process mode.
"""

import logging
import os
import socket

from typing import List, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide excessive tensorflow debug messages
import sys

# import fbgemm_gpu  # noqa: F401, E402
import gin

import torch
import tempfile

from absl import app, flags

# Import train_fn early so gin can recognize it
from generative_recommenders.research.trainer.train import train_fn

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def delete_flags(FLAGS, keys_to_delete: List[str]) -> None:  # pyre-ignore [2]
    keys = [key for key in FLAGS._flags()]
    for key in keys:
        if key in keys_to_delete:
            delattr(FLAGS, key)


def find_free_port(start_port: int = 12355, max_attempts: int = 100) -> int:
    """
    Find a free port starting from start_port.
    
    Args:
        start_port: Starting port number
        max_attempts: Maximum number of ports to try
        
    Returns:
        Available port number
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts}")


delete_flags(flags.FLAGS, ["gin_config_file", "master_port"])
flags.DEFINE_string("gin_config_file", None, "Path to the config file.")
flags.DEFINE_integer("master_port", 12355, "Master port.")
FLAGS = flags.FLAGS  # pyre-ignore [5]


def _main(argv) -> None:  # pyre-ignore [2]
    if FLAGS.gin_config_file is not None:
        logging.info(f"Loading gin config from {FLAGS.gin_config_file}")
        gin.parse_config_file(FLAGS.gin_config_file)

    # Run in single process mode for debugging
    # Set world_size=1 and rank=0 for single GPU debugging
    world_size = 1
    rank = 0
    
    # Check if we have GPU available and set appropriate backend
    if torch.cuda.is_available():
        logging.info(f"CUDA available with {torch.cuda.device_count()} devices")
        backend = "nccl"
    else:
        logging.info("No CUDA available, using CPU mode with gloo backend")
        backend = "gloo"
        # Set environment variable to override backend in train.py
        os.environ["TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    
    # Find a free port to avoid conflicts
    try:
        free_port = find_free_port(FLAGS.master_port)
        logging.info(f"Using port {free_port} for distributed training")
    except RuntimeError:
        logging.warning(f"Could not find free port starting from {FLAGS.master_port}, using default")
        free_port = FLAGS.master_port
    
    # Set up environment for single process training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(free_port)
    
    try:
        train_fn(rank, world_size, free_port)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.info("This error might be due to the train_fn expecting GPU but running on CPU.")
        logging.info("To debug this properly, you may need to modify the training code for CPU compatibility.")
        raise e


def main() -> None:
    app.run(_main)


if __name__ == "__main__":
    main() 