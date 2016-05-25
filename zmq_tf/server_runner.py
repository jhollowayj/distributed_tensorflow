import server
from networks import NetworkType

### COMMAND LINE ARGUMENTS ###
import argparse
parser = argparse.ArgumentParser()
### SERVER-RUNNER-PARAMS
parser.add_argument('--test_client', '-test', default=False,  action='store_true', required=False, help="")
### SERVER-PARAMS
parser.add_argument('--just_one_server', '-jos', default=True,  action='store_false', required=False, help="")
parser.add_argument('--grad_update_cnt_before_send', '-send', default=10,  type=int, required=False, help="")
parser.add_argument('--serverType', '-t', default=NetworkType.World,  type=int, required=False, help="")
parser.add_argument('--server_learning_rate', '-lr', default=1,  type=int, required=False, help="")
parser.add_argument('--tensorflow_random_seed', '-tfrs', default=54321,  type=int, required=False, help="")
parser.add_argument('--requested_gpu_vram_percent', '-vram', default=0.05,  type=float, required=False, help="")
parser.add_argument('--device_to_use', '-device', default=1,  type=int, required=False, help="")
parser.add_argument('--verbose', '-v', default=2,  type=int, required=False, help="")
parser.add_argument('--ckpt_save_interval', '-save', default=100,  type=int, required=False, help="")
parser.add_argument('--weights_ckpt_folder', '-f', default="/tmp/mod_dnn_research/", type=str, required=False, help="")
parser.add_argument('--load_ckpt_file_on_start', '-load', default=False, action='store_true', required=False, help="")
args = parser.parse_args()
### COMMAND LINE ARGUMENTS ###

s = server.ModDNN_ZMQ_Server(
    just_one_server = args.just_one_server,
    grad_update_cnt_before_send = args.grad_update_cnt_before_send,
    serverType = args.serverType, 
    server_learning_rate = args.server_learning_rate,
    tensorflow_random_seed = args.tensorflow_random_seed,
    requested_gpu_vram_percent = args.requested_gpu_vram_percent,
    device_to_use = args.device_to_use,
    verbose = args.verbose,
    ckpt_save_interval = args.ckpt_save_interval,
    weights_ckpt_folder = args.weights_ckpt_folder,
    load_ckpt_file_on_start = args.load_ckpt_file_on_start,
)

if args.test_client:
    s.testClient()
else:
    s.startPolling()
