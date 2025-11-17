import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import random
import argparse
import numpy as np

#from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

auth_token="your_token"
hf_cache = '/your_path/huggingface_cache'
#accelerator = Accelerator(gradient_accumulation_steps=1)

def parse_args():
    parser = argparse.ArgumentParser(description='LLM finetuning')
    parser.add_argument('--seed', default=2023, type=int, help='The random seed.')
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3-8B', type=str, help='model name')
    parser.add_argument('--model_checkpoint', type=str, default='/your_path', help='The folder for storing model checkpoints.')
    parser.add_argument('--task', type=str, default='commonsense', help='The evaluation tasks.')
    parser.add_argument('--BATCH_SIZE', type=int, default=1, help='Batch size for model training.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU using, i.e. \'0,1,2\'')     
    parser.add_argument('--parallel', default=False, action='store_true', help='paralleled computing')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['PYHTONHASHSEED'] = str(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #np.random.seed(args.seed)
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch_cache = '/your_path/torch/hub'
    if not os.path.isdir(torch_cache):
        os.makedirs(torch_cache, exist_ok=True)
    torch.hub.set_dir(torch_cache)
    #metric_name = task2metrc[args.task_name]
    import datasets
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    
    #########################################
    # check model list
    #########################################
    model_list = [
        "meta-llama/Meta-Llama-3-8B","meta-llama/Llama-2-7b-hf",
    ]
    if args.model not in model_list:
        raise ValueError("Only support the language classification models including: ".format(model_list))

    config = AutoConfig.from_pretrained(args.model, finetuning_task="text-generation", token = auth_token, cache_dir = hf_cache)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, token = auth_token, cache_dir = hf_cache)
    #Do not need to reset the classifier, the pretrained parameters does not include classification layer.

    checkpoint = torch.load(args.model_checkpoint)
    model.load_state_dict(checkpoint)   
    model.to(DEVICE)

    if args.task == "zeroshot":
        print("----------------------------------------")
        print("Now evaluating the model on standard 8 zero-shot datasets......")
        from methods.data_loader import zero_shot_eval
        results = zero_shot_eval(model, tokenizer, DEVICE)
        avg = 0
        print("----------------------------------------")
        for k,v in results.items():
            print("********** Accuracy on {}: {:.2f}%. **********\n".format(k, v*100))
            avg += v*100
        print("********** Accuracy in average: {:.2f}%. **********\n".format(avg/8))
        print("----------------------------------------")
    else:
        if args.task == "math":
            #task_list = ["gsm8k", "hendrycks_math"] 
            task_list = ["gsm8k"] 
        elif args.task == "commonsense":
            task_list=["boolq","piqa","social_iqa","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"]
        elif args.task == "instruction":
            task_list=["ifeval"]

        print("----------------------------------------")
        print("Now evaluating the model by EleutherAI's LM-Evaluation-Harness")
        from lm_eval import evaluator 
        from lm_eval.models.huggingface import HFLM 
        num_fewshot = 0
        if "70b" in args.model or "65b" in args.model:
            limit = 2000
        else:
            limit = None
        #model = model.merge_and_unload()
        model = HFLM(pretrained=model, backend = "causal", trust_remote_code = True)
        results = evaluator.simple_evaluate(
            model = model,
            tasks = task_list,
            num_fewshot=num_fewshot,
            batch_size=args.BATCH_SIZE,
            device = DEVICE,
            limit = limit,
            check_integrity = False,
        )
        printing_results = results["results"]
        print("{} shot evaluation".format(num_fewshot))
        for k,v in printing_results.items():
            print("{}: {}".format(k,v))    

if __name__=='__main__':
    main()