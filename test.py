import os
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import mteb
import argparse
from model import TemperatureSentenceTransformer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--task", type=str)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--temperature", type=float, default=1)

args = parser.parse_args()

model = TemperatureSentenceTransformer(args.model, temperature=args.temperature)

tasks = mteb.get_tasks(tasks=[args.task])
evaluation = mteb.MTEB(tasks=tasks)

output_dir = f"./results/{args.model.split('/')[-1]}/{args.task}/t{args.temperature}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
results = evaluation.run(model, encode_kwargs={'batch_size': args.batch_size}, output_folder=f"{output_dir}", save_predictions=True)