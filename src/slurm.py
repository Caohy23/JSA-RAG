# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import signal
import socket
import subprocess
import sys
from logging import getLogger

import torch

logger = getLogger()

GLOO_GROUP = None


def sig_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))

    prod_id = int(os.environ["SLURM_PROCID"])
    logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
        os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
    else:
        logger.warning("Not the main process, no need to requeue.")
    sys.exit(-1)


def term_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Bypassing SIGTERM.")


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    # logger.warning("Signal handler installed.")


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    # params.is_slurm_job = 'SLURM_JOB_ID' in os.environ
    params.is_slurm_job = "SLURM_JOB_ID" in os.environ and not "WORLD_SIZE" in os.environ
    has_local_rank = hasattr(params, "local_rank")

    # SLURM job
    if params.is_slurm_job and has_local_rank:

        assert params.local_rank == -1  # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_JOB_NUM_NODES",
            "SLURM_NTASKS",
            "SLURM_TASKS_PER_NODE",
            "SLURM_MEM_PER_NODE",
            "SLURM_MEM_PER_CPU",
            "SLURM_NODEID",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "SLURM_TASK_PID",
        ]

        PREFIX = "%i - " % int(os.environ["SLURM_PROCID"])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            # print(PREFIX + "%s: %s" % (name, str(value)))

        # # job ID
        # params.job_id = os.environ['SLURM_JOB_ID']

        # number of nodes / node ID
        params.n_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        params.node_id = int(os.environ["SLURM_NODEID"])

        # local rank on the current node / global rank
        params.local_rank = int(os.environ["SLURM_LOCALID"])
        params.global_rank = int(os.environ["SLURM_PROCID"])

        # number of processes / GPUs per node
        params.world_size = int(os.environ["SLURM_NTASKS"])
        params.n_gpu_per_node = params.world_size // params.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
        params.main_addr = hostnames.split()[0].decode("utf-8")
        assert 10001 <= params.main_port <= 20000 or params.world_size == 1
        # print(PREFIX + "Master address: %s" % params.master_addr)
        # print(PREFIX + "Master port   : %i" % params.master_port)

        # set environment variables for 'env://'
        os.environ["MASTER_ADDR"] = params.main_addr
        os.environ["MASTER_PORT"] = str(params.main_port)
        os.environ["WORLD_SIZE"] = str(params.world_size)
        os.environ["RANK"] = str(params.global_rank)

        params.is_distributed = True

    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    elif has_local_rank and params.local_rank != -1:

        assert params.main_port == -1

        # read environment variables
        params.global_rank = int(os.environ["RANK"])
        params.world_size = int(os.environ["WORLD_SIZE"])
        params.n_gpu_per_node = params.world_size
        # params.n_gpu_per_node = int(os.environ["NGPU"])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
        params.is_distributed = True

    else:
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.is_distributed = False
        params.n_nodes = 1
        params.node_id = 0
        params.n_gpu_per_node = 1

    # define whether this is the master process / if we are in distributed mode
    params.is_main = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1

    # summary
    PREFIX = "%i - " % params.global_rank

    # set GPU device
    if params.is_distributed:
        torch.cuda.set_device(params.local_rank)
        device = torch.device("cuda", params.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.device = device

    # initialize multi-GPU
    if params.is_distributed:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        # print("Initializing PyTorch distributed ...")
        # Fix for if gloo sockets are inconsistent
        p1 = subprocess.Popen(["ip", "r"], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["grep", "default"], stdin=p1.stdout, stdout=subprocess.PIPE)
        p1.stdout.close()
        gloo_socket_ifname = subprocess.check_output(["awk", "{print $5}"], stdin=p2.stdout).decode("utf-8").strip()
        p2.stdout.close()
        os.environ["GLOO_SOCKET_IFNAME"] = gloo_socket_ifname
        os.environ["GLOO_SOCKET_IFNAME"] = "eth0"

        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
            timeout=datetime.timedelta(seconds=100000)
        )
        global GLOO_GROUP

        GLOO_GROUP = torch.distributed.new_group(
            list(range(params.world_size)), backend="gloo", timeout=datetime.timedelta(0, 600)
        )


def get_gloo_group():
    global GLOO_GROUP
    assert GLOO_GROUP is not None
    return GLOO_GROUP
