# The latency is benchmarked with batch size = 1
import argparse
import os
import json

def get_args_parser():
    parser = argparse.ArgumentParser('TensorRT operator json parser script', add_help=False)
    # Model Configuration
    parser.add_argument('--model', default=None, type=str,
                        help='the type of the model to benchmark')
    parser.add_argument('--seq_len', default=None, type=int,
                        help='the seq_len of the model')
    parser.add_argument('--dim', default=None, type=int,
                        help='the dim of the model')
    parser.add_argument('--heads', default=None, type=int,
                        help='the number of heads of the model')
    return parser

def main(args):

    model_name = "{}-seq_len_{}-dim_{}-heads_{}".format(args.model, args.seq_len, args.dim, args.heads)
    with open("benchmark_logs/{}.json".format(model_name)) as f:
        trace = json.load(f)
    total_percentage = 0
    for d in trace:
        if "name" in d:
            if "oftmax" in d["name"]:
                total_percentage += d["percentage"]
    print("Model: {}, softmax percentage: {}%".format(model_name, total_percentage))


if __name__=="__main__":
    parser = argparse.ArgumentParser('TensorRT operator json parser script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)