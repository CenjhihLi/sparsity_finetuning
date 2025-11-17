import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import random
import time
import json
import argparse
import shortuuid
import bitsandbytes as bnb

from methods.data_loader import mtbench

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
)
#from accelerate import Accelerator

auth_token="your_token"
hf_cache = '/your_path/huggingface_cache'
#accelerator = Accelerator(gradient_accumulation_steps=1)


def quantize_and_dequantized(weight):
    device = weight.device
    weight_nf4 = bnb.nn.Params4bit(weight.to("cpu"), 
                                    requires_grad=False, 
                                    compress_statistics=False, 
                                    quant_type="nf4", 
                                    #quant_storage = torch.uint8,
                                    )
    weight_nf4 = weight_nf4.to(device)
    weight_dequantized = bnb.functional.dequantize_4bit(
        weight_nf4.data, weight_nf4.quant_state
    ).to(torch.float32)
    return weight_nf4, weight_dequantized


def parse_args():
    parser = argparse.ArgumentParser(description='LLM finetuning')
    parser.add_argument('--seed', default=2023, type=int, help='The random seed.')
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3-8B', type=str, help='model name')
    parser.add_argument('--quantization', default=False, action='store_true', help='quantize model')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='Dataset choosing.')
    parser.add_argument('--BATCH_SIZE', type=int, default=1, help='Batch size for model training.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU using, i.e. \'0,1,2\'')     
    parser.add_argument('--parallel', default=False, action='store_true', help='paralleled computing')
    args = parser.parse_args()
    return args


#"/mtbench/question.jsonl"

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
    
    #########################################
    # check model list
    #########################################
    model_list = [
        "meta-llama/Meta-Llama-3-8B","meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", "google/gemma-7b",
    ]
    if args.model not in model_list:
        raise ValueError("Only support the language classification models including: ".format(model_list))

    config = AutoConfig.from_pretrained(args.model, finetuning_task="text-generation", token = auth_token, cache_dir = hf_cache)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, token = auth_token, cache_dir = hf_cache)
    model.to(DEVICE)
    if args.quantization:
        for name, param in model.named_parameters():
            weight_nf4, weight_dequantized = quantize_and_dequantized(param)
            param.data = weight_dequantized.data
    #Do not need to reset the classifier, the pretrained parameters does not include classification layer.

    print("\n----------------------------------------")
    for name, param in model.named_parameters():
        print ("Parameters: {}, requires_grad: {}, size: {}.".format(name, param.requires_grad, param.size()))
    print("----------------------------------------\n")

    ###################################################################
    #Evaluate models on MT-bench
    mtbench_dataset = mtbench()
    
    first_instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    second_instruction = "Below is a conversation with a previous instruction, a previous response, and a new instruction that describes a task. Write a response that appropriately completes the request from new instruction.\n\n"
    def append_prompt(prompt, message):
        prompt = prompt + "### Instruction:\n{instruction}\n\n### Response:".format(instruction = message)
        return prompt

    temperature_config = {
        "writing": 0.7,
        "roleplay": 0.7,
        "extraction": 0.0,
        "math": 0.0,
        "coding": 0.0,
        "reasoning": 0.0,
        "stem": 0.1,
        "humanities": 0.1,
        }
    #model_generations = []

    if args.quantization:
        answer_file = './mtbench/' + args.model.split('/')[-1] + '_quantization.jsonl'
    else:
        answer_file = './mtbench/' + args.model.split('/')[-1] + '.jsonl'
    
    for i, item in enumerate(mtbench_dataset):
        temperature = temperature_config[item['category']]
        if temperature < 1e-4:
            do_sample = False
        else:
            do_sample = True
        
        instruction = first_instruction 
        prompt = ""
        answers = []
        for question in item["turns"]:
            prompt = append_prompt(prompt, question)

            input_ids = tokenizer([instruction+prompt]).input_ids
            output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample = do_sample,
                        temperature = temperature,
                        max_new_tokens = 512,
                    )
            output = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("### Response:")[-1]
           
            answers.append(output)
            prompt = prompt + "\n" + output + "\n\n"
            instruction = second_instruction
        model_gen = {
            "question_id": item["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": args.model,
            "choices": [{"index": 0, "turns": answers}],
            "tstamp": time.time(),
        }
        with open(os.path.expanduser(answer_file), "a") as fout:
            fout.write(json.dumps(model_gen) + "\n")
    """
    Evaluate by using the following prompt via GPT-4 or Claude-3
    
    You are a helpful assistant.[Instruction]\nPlease act as an impartial judge and evaluate the quality of 
    the response provided by an AI assistant to the user question displayed below. 
    Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, 
    and level of detail of the response. Begin your evaluation by providing a short explanation. 
    Be as objective as possible. 
    After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: 
    \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n
    [Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]
    """
    ###################################################################
    
    if args.dataset == "zeroshot":
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
        print("----------------------------------------")
        print("Now evaluating the model by EleutherAI's LM-Evaluation-Harness")
        from lm_eval import evaluator 
        from lm_eval.models.huggingface import HFLM 
        task_list=["boolq","piqa","social_iqa","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa", "gsm8k", "ifeval"] 
        num_fewshot = 0
        if "70b" in args.model or "65b" in args.model:
            limit = 2000
        else:
            limit = None

        model = HFLM(pretrained=model, tokenizer = tokenizer, backend = "causal")
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