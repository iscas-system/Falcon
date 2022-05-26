import os,argparse
import sys

from utils.logger import set_logger

LOG = "/var/log/falctl.log"
logger = set_logger(os.path.basename(__file__), LOG)

all_m = ['bert-large','densenet-201','gru','inception-v2','inception-v4','mobilenet-v2','resnet-101','resnet-152-v2','roberta','tacotron2','transformer','vgg16']
all_o = ['add','batch_norm','bias_add','concat','conv1d','conv2d','dense','multiply','relu','sigmoid','split','strided_slice','subtract','tanh','transpose']

def runParser(args):
    if args.model:
        if args.model not in all_m:
            print('Unsupported model %s' % args.model)
            sys.exit(1)
        os.system('run-baselines %s' % args.model)
    elif args.all:
        os.system('run-baselines')
        
def analyzeParser(args):
    if args.model:
        if args.model not in all_m:
            print('Unsupported model %s' % args.model)
            sys.exit(1)
    elif args.operator:
        if args.operator not in all_o:
            print('Unsupported operator %s' % args.operator)
            sys.exit(1)

'''
CMD line parser
'''
parser = argparse.ArgumentParser(prog="falctl", description="A configuration recommender tool for deep learning reference.")

subparsers = parser.add_subparsers()

'''
falctl evaluate
'''
parser_run = subparsers.add_parser("evaluate", help="comparing falcon with baselines")
megroup = parser_run.add_mutually_exclusive_group()
megroup.add_argument("--model", required=False, metavar="[bert-large|densenet-201|gru|inception-v2|inception-v4|mobilenet-v2|resnet-101|resnet-152-v2|roberta|tacotron2|transformer|vgg16]", type=str,
                                help="evaluating a specific DL model")
megroup.add_argument("--all", required=False, type=bool, nargs='?', const=True,
                                help="evaluating all DL models")

parser_run.set_defaults(func=runParser)

'''
falctl analyze
'''
parser_analyze = subparsers.add_parser("analyze", help="analyzing DL models and operators")
megroup = parser_analyze.add_mutually_exclusive_group()
megroup.add_argument("--model", required=False, metavar="[bert-large|densenet-201|gru|inception-v2|inception-v4|mobilenet-v2|resnet-101|resnet-152-v2|roberta|tacotron2|transformer|vgg16]", type=str,
                                help="analyzing a specific DL model")
megroup.add_argument("--operator", required=False, metavar="[add|batch_norm|bias_add|concat|conv1d|conv2d|dense|multiply|relu|sigmoid|split|strided_slice|subtract|tanh|transpose]", type=str,
                                help="analyzing a specific DL model")

parser_analyze.set_defaults(func=analyzeParser)

os.putenv('LANG', 'en_US.UTF-8')
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
elif len(sys.argv)==2:
    if sys.argv[1] == 'evaluate':
        parser_run.print_help(sys.stderr)
        sys.exit(1)
    elif sys.argv[1] == 'analyze':
        parser_analyze.print_help(sys.stderr)
        sys.exit(1)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)
args = parser.parse_args()
args.func(args)
