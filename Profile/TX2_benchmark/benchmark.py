# The latency is benchmarked with batch size = 1
import argparse
import datetime
import numpy as np
import time
import torch
import json
import os
import onnx
import subprocess
import logging
import csv
import re
from onnxsim import simplify
import sys
sys.path.append("../")
from models import build_model

def print_and_log(s):
    print(s)
    logging.info(s)

def get_args_parser():
    parser = argparse.ArgumentParser('Latency in Jetson GPU measurement script', add_help=False)
    # Model Configuration
    parser.add_argument('--model', default=None, type=str,
                        help='the type of the model to benchmark')
    parser.add_argument('--seq_len', default=None, type=int,
                        help='the seq_len of the model')
    parser.add_argument('--dim', default=None, type=int,
                        help='the dim of the model')
    parser.add_argument('--heads', default=None, type=int,
                        help='the number of heads of the model')
    # Profile Configuration
    parser.add_argument('--enable_op_profiling', action='store_true', default=False, help='Profiling each operator')
    return parser

@torch.no_grad()
def compute_latency(model_name, model_config, enable_op_profiling=False):
    latency_mean = -1
    torch.cuda.empty_cache()

    model = build_model(model_name, model_config)

    os.makedirs("../onnx_models", exist_ok=True)

    if not os.path.exists("../onnx_models/{}-simpler.onnx".format(model_name)):

        model = model.cuda().eval()
        inputs = torch.rand(1, model_config["seq_len"], model_config["dim"]).cuda()

        torch.onnx.export(model, inputs, "../onnx_models/{}.onnx".format(model_name), input_names=['input'],
                        output_names=['output'])


        # simplify model
        # load your predefined ONNX model
        onnx_model = onnx.load("../onnx_models/{}.onnx".format(model_name))
        onnx_model_simpler, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"

        onnx.save(onnx_model_simpler, "../onnx_models/{}-simpler.onnx".format(model_name))

        del onnx_model
        del onnx_model_simpler
        torch.cuda.empty_cache()

    trtexec_run = "./trtexec --onnx=../onnx_models/{}-simpler.onnx --batch={}".format(model_name, 1)
    if enable_op_profiling:
        trtexec_run+=" --dumpProfile --exportProfile=benchmark_logs/{}.json".format(model_name)
    result = subprocess.check_output(trtexec_run, shell=True)

    result = result.decode("utf-8")
    logging.info(result)
    for line in result.split("\n"):
        if "mean" in line and "end to end" in line:
            latency_mean = float(re.split("mean: | ms",line)[1])

    print_and_log("{} deployed by trtexec, Latency per image: {} ms".format(model_name, latency_mean))
    return latency_mean

def main(args):

    model_name = "{}-seq_len_{}-dim_{}-heads_{}".format(args.model, args.seq_len, args.dim, args.heads)

    basic_arch = {
        "seq_len": args.seq_len,
        "dim": args.dim,
        "heads": args.heads,
    }

    os.makedirs("benchmark_logs", exist_ok=True)

    logging.basicConfig(filename=os.path.join("benchmark_logs", "{}.log".format(model_name)),level=logging.DEBUG)
 
    compute_latency(model_name,
                    basic_arch,
                    enable_op_profiling=args.enable_op_profiling)

if __name__=="__main__":
    parser = argparse.ArgumentParser('Latency in Jetson GPU measurement script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)