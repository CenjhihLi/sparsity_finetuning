import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import random
import argparse
import time
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import bitsandbytes as bnb

from torch.nn import Linear
from torch.autograd import Variable
from methods.data_loader import flanv2, wikitext2, alpaca_llama, alpacagpt4_llama, gsm8k, zeroshot_llama
from evaluate import load
from peft import get_peft_model, prepare_model_for_kbit_training
from torch.utils.tensorboard import SummaryWriter
#from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
    BitsAndBytesConfig,
)

auth_token="your_token"
hf_cache = '/your_path/huggingface_cache'
#accelerator = Accelerator(gradient_accumulation_steps=1)

def parse_args():
    parser = argparse.ArgumentParser(description='LLM finetuning')
    parser.add_argument('--seed', default=2023, type=int, help='The random seed.')
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3-8B', type=str, help='model name')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='Dataset choosing.')
    parser.add_argument('--task_name', type=str, default='text-generation', help='Task choosing.')
    parser.add_argument('--n_samples', type=int, default=256, help='Number of samples for finetuning text generation model.')
    parser.add_argument('--max_length', type=int, default=2048, help=(
            "The maximum length of total input sequence tokenization. Sequences longer will be truncated, sequences shorter will be padded."))
    parser.add_argument('--pad_to_max_length', default=False, action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",)
    parser.add_argument('--data_path', default='/your_path', type=str, help='Data path')
    parser.add_argument('--model_checkpoint', type=str, default='/your_path', help='The folder for storing model checkpoints.')
    parser.add_argument('--store_frequency', type=int, default=5, help='Storing model checkpoints each how much epochs.')
    parser.add_argument('--finetuning_rank', default=64, type=int, help='control the fine-tuning features by a fixed rank instead of using ratio')
    parser.add_argument('--finetuning_density', default=0.012, type=float, help='control the fine-tuning density of rosa')
    parser.add_argument('--finetune_attn_as_linear', default=False, action='store_true', help='Finetune attention layer as independent linear layers.')
    parser.add_argument('--method', default='lora', type=str, help='peft method choice', choices=['qlora', 'loftq', 'qpissa', 'qdora', 'qrosa', 'gptq', 'ours', 'ours_loftq', 'memory_test', 'fft'])
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for model training.')
    parser.add_argument('--warmup_steps', type=float, default=0, help='Warm up steps (Learning rate will not dacay) for model training.')
    parser.add_argument('--BATCH_SIZE', type=int, default=1, help='Batch size for model training.')
    parser.add_argument('--EPOCHS', type=int, default=10, help='Max epochs for model training.')    
    parser.add_argument('--START_EPOCH', type=int, default=1, help='Start epochs for model training. (continue last training)')    
    parser.add_argument('--gpu', type=str, default='0', help='GPU using, i.e. \'0,1,2\'')     
    parser.add_argument('--parallel', default=False, action='store_true', help='paralleled computing')
    parser.add_argument('--finetuning_ratio', default=0.024, type=float, help='finetuning ratio')
    parser.add_argument('--p_dropout', default=0.0, type=float, help='finetuning dropout prob.')
    parser.add_argument('--n_estimate', default=5, type=int, help='n_estimate for ZOtaylor')
    parser.add_argument('--n_iter', default=5, type=int, help='n_iter for loftQ and qpissa initialization')
    parser.add_argument('--variant', default='PruFT', type=str, help='combining other method')
    parser.add_argument('--finetune_num_heads', default=False, action='store_true', help='Finetuning attention head.')
    parser.add_argument('--head_finetuning_ratio', default=0.05, type=float, help='Head finetuning ratio, work if finetune_num_heads is True.')
    parser.add_argument('--bottleneck', default=False, action='store_true', help='Bottleneck or uniform')
    parser.add_argument('--finetuning_type', default='l2', type=str, help='Finetuning type', choices=['random', 'ZOtaylor', 'taylor', 'l2', 'l1', 'wanda', 'hessian', '2ndtaylor'])
    parser.add_argument('--label_aggregator', default='None', type=str, help='Aggregator of labelwise importance score', choices=['max', 'mean', 'GreaterQuantileAvg', 'QuantilesAvg', 'Top5Avg', 'Top5MeanAvg'])
    parser.add_argument('--finetuning_channel', default='out', type=str, help='Finetuning channel. "in": in channel, "out": out channel, "half": half in and half out', choices=['out', 'in', 'half'])
    parser.add_argument('--global_finetuning', default=False, action='store_true', help='Global finetuning')  
    args = parser.parse_args()
    return args

def train_text_generation(model, device, data_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    # List to store negative log likelihoods, for computing perplexity
    #nlls = []
    total_num = len(data_loader.dataset)
    print("Training sample size: {}, number of batchs: {}.".format(total_num, len(data_loader)))
    for batch_idx, item in enumerate(data_loader):
        if isinstance(item, list):
            # Prepare inputs and move to device
            inputs, targets = item
            inputs = Variable(inputs).to(device)
            targets = Variable(targets).to(device)
            # Forward pass through the model
            lm_output = model(inputs, labels = targets)
        elif isinstance(item, dict):
            for k, v in item.items():
                item[k] = Variable(v).to(device)
            lm_output = model(**item)
            targets = item["labels"]
        loss = lm_output.loss #negative log likelihood
        optimizer.zero_grad()
        loss.backward()
        #accelerator.backward(loss)
        optimizer.step()
        sum_loss += loss.data.item()
        #sum_loss += accelerator.gather(loss).data.item()
        # Append to list of negative log likelihoods
        #nlls.append(loss)
        
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(targets), total_num,
                       100. * (batch_idx + 1) / len(data_loader), loss.item()))
        torch.cuda.empty_cache()
    avg_loss = sum_loss / len(data_loader)
    #ppl = torch.exp(torch.stack(nlls).mean())
    print('epoch:{},loss:{}'.format(epoch, avg_loss))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()
    return avg_loss#, ppl.item()


@torch.no_grad()
def val_text_generation(model, device, data_loader, metric):
    model.eval()
    sum_loss = 0
    # List to store negative log likelihoods, for computing perplexity
    output_metric = None
    if metric == "perplexity":
        nlls = []
    elif metric in ["F1", "accuracy", "spearmanr" ]:
        metric_fn = load(metric)
    total_num = len(data_loader.dataset)
    print("Training sample size: {}, number of batchs: {}.".format(total_num, len(data_loader)))    
    with torch.no_grad():
        for batch_idx, item in enumerate(data_loader):
            if isinstance(item, list):
                # Prepare inputs and move to device
                inputs, targets = item
                inputs = Variable(inputs).to(device)
                targets = Variable(targets).to(device)
                # Forward pass through the model
                lm_output = model(inputs, labels = targets)
            elif isinstance(item, dict):
                for k, v in item.items():
                    item[k] = Variable(v).to(device)
                lm_output = model(**item)
                targets = item["labels"]
            loss = lm_output.loss #negative log likelihood
            sum_loss += loss.data.item()
            # Append to list of negative log likelihoods
            if metric == "perplexity":
                nlls.append(loss)
            elif metric in ["F1", "accuracy", "spearmanr" ]:
                metric_fn.add_batch(
                    predictions = lm_output.logits.argmax(dim=-1).to(torch.float32),
                    references = targets.to(torch.float32),
                )

        if metric == "perplexity":
            output_metric = torch.exp(torch.stack(nlls).mean()).item()
        elif metric in ["F1", "accuracy", "spearmanr" ]:
            output_metric = metric_fn.compute()[metric]

        avgloss = sum_loss / len(data_loader)
        print('\nVal set: Average loss: {:.4f}, perplexity: {:.4f}\n'.format(avgloss, output_metric))
        torch.cuda.empty_cache()
        return avgloss, output_metric

def merge_model(args, target_modules, quant_model):
    config = AutoConfig.from_pretrained(args.model, finetuning_task="text-generation", token = auth_token, cache_dir = hf_cache)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, token = auth_token, trust_remote_code=True, 
                                                 device_map={"":0}, cache_dir = hf_cache)
    use_dora = True if args.method == "qdora" else False
    lora_alpha = args.finetuning_rank if "ours" in args.method else 16
    lora_dropout = 0.0 if "ours" in args.method else 0.1
    from peft import LoraConfig
    peft_config = LoraConfig(
            task_type = "CAUSAL_LM", 
            inference_mode=False, 
            r = args.finetuning_rank, 
            lora_alpha = lora_alpha, 
            lora_dropout = lora_dropout, #0.0: no dropout, >0.0: dropout
            bias= "none", 
            use_dora = use_dora,
            target_modules = target_modules,
            )
    model = get_peft_model(model, peft_config)
    with torch.no_grad():
        state_dict = {}
        for name, param in quant_model.named_parameters():
            if 'base_layer' in name:
                state_dict[name] = bnb.functional.dequantize_4bit(param.data, param.quant_state).to(device = 'cpu', dtype = torch.float32)
            elif 'lora' in name:
                state_dict[name] = param.data.to('cpu')
        model.load_state_dict(state_dict, strict=False)
    model = model.merge_and_unload()
    return model
        
def build_our_model(args, DEVICE, train_loader):
    from methods.finetuner_runner import Finetuner
    from methods.finetuner_importance import RandomImportance, NormImportance, TaylorImportance, HessianImportance, WandAImportance, ZerothOrderTaylorImportance
    start_time = time.time()
    if args.finetuning_type == 'random':
        imp = RandomImportance()
    elif args.finetuning_type == 'taylor':
        imp = TaylorImportance()
    elif args.finetuning_type == 'ZOtaylor':
        imp = ZerothOrderTaylorImportance(n_estimate = args.n_estimate)
    elif args.finetuning_type == 'l2':
        imp = NormImportance(p=2)
    elif args.finetuning_type == 'l1':
        imp = NormImportance(p=1)
    elif args.finetuning_type == 'wanda':
        imp = WandAImportance()
    elif args.finetuning_type == 'hessian':
        imp = HessianImportance()
    else: raise NotImplementedError
    config = AutoConfig.from_pretrained(args.model, finetuning_task="text-generation", token = auth_token, cache_dir = hf_cache)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, token = auth_token, trust_remote_code=True, 
                                                 device_map={"":0}, cache_dir = hf_cache)
    #########################################
    # Build network finetuner
    #########################################
    num_heads = {}
    ignored_layers = []
    ignored_layers.append(model.lm_head)
    #########################################
    # Ignore classification modules
    # All heads should be pruned simultaneously, so we group channels by head.
    # Since requires_grad are set to be True, all parameters in the ignored layers will be finetuned.
    #########################################

    if args.finetune_attn_as_linear:
        print("Finetune attention layer as independent linear layers.")
       
    customized_finetuner = None
    for name, module in model.named_modules():
        if (not args.finetune_attn_as_linear) and name.endswith("self_attn"):
            ignored_layers.append(module)
        if 'norm' in name or 'embed_tokens' in name:
            ignored_layers.append(module)
        
    if args.finetuning_channel == "in":
        print("Finetuning in channels/features.")
    unwrapped_parameters = None
    finetuner = Finetuner(
        model, 
        global_finetuning=args.global_finetuning, # If False, a uniform pruning ratio will be assigned to different layers.
        importance=imp, # importance criterion for parameter selection
        finetuning_ratio=args.finetuning_ratio, # target pruning ratio
        rank = args.finetuning_rank if args.finetuning_rank>0 else None,
        ignored_layers=ignored_layers,
        num_heads=num_heads, # number of heads in self attention
        finetune_num_heads=args.finetune_num_heads, # finetuning entire heads (default: False)
        finetune_head_dims=not args.finetune_num_heads, # finetuning head_dim (default: True)
        head_finetuning_ratio = args.head_finetuning_ratio, # only works when finetune_num_heads=True (default: 0.0)
        round_to = 1 if args.finetuning_rank>0 else 4,
        finetuning_channel = args.finetuning_channel,
        dtype = "fp32",
        p_dropout = args.p_dropout,
        variant = args.variant,
        customized_finetuner = customized_finetuner
        )
    
    torch.cuda.reset_peak_memory_stats(0)
    if isinstance(imp, (WandAImportance, TaylorImportance, HessianImportance)):
        if isinstance(imp, (TaylorImportance, HessianImportance)):
            model.zero_grad()
            print("Accumulating gradients for finetuning settings...")
        if isinstance(imp, HessianImportance):
            imp.zero_grad()    

        if isinstance(imp, WandAImportance):        
            imp.register_module(model, len(train_loader.dataset))

        for k, item_dict in enumerate(train_loader):
            if k>=args.taylor_batchs: break
            for key, tensor in item_dict.items():
                item_dict[key] = tensor.to(DEVICE)

            output = model(**item_dict)

            lbls = item_dict["labels"]
            if isinstance(imp, HessianImportance):
                loss = F.cross_entropy(output.logits, lbls, reduction='none')
                for l in loss:
                    model.zero_grad()
                    l.backward(retain_graph=True)
                    imp.accumulate_grad(model)
            elif isinstance(imp, TaylorImportance):
                loss = output.loss
                loss.backward()
        if isinstance(imp, WandAImportance):        
            imp.remove_module(model)
        
    elif isinstance(imp, (ZerothOrderTaylorImportance)):
        imp.projected_grad_estimate(model, DEVICE, train_loader, ignored_layers, random_seed = args.seed)
        
    finetuner.run() 
    end_time = time.time()
    print("Max memory allocated in fine-tuning settings: {}GB".format(torch.cuda.max_memory_allocated()/(1024**3)))
    print("Completed setting the finetuned parameters, took {:.2f}s.".format(end_time - start_time))
    torch.cuda.empty_cache()
    return model

def build_peft_model(args, DEVICE, train_loader, target_modules):
    # This is for pissa and loftq
    config = AutoConfig.from_pretrained(args.model, finetuning_task="text-generation", token = auth_token, cache_dir = hf_cache)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, token = auth_token, trust_remote_code=True, 
                                                 device_map={"":0}, cache_dir = hf_cache)
    from peft import LoraConfig
    peft_config = LoraConfig(
            task_type = "CAUSAL_LM", 
            inference_mode=False, 
            r = args.finetuning_rank, 
            lora_alpha = 16, 
            lora_dropout = 0.1, #0.0: no dropout, >0.0: dropout
            bias= "none", 
            #init_lora_weights="pissa_niter_4" if args.method == "qpissa" else True,
            target_modules = target_modules,
        )
    model = get_peft_model(model, peft_config)
    return model


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
    if not os.path.isdir(args.model_checkpoint):
        os.makedirs(args.model_checkpoint, exist_ok=True)
    torch.hub.set_dir(torch_cache)
    #import datasets
    #datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    #metric_name = task2metrc[args.task_name]
    
    #########################################
    # check model list
    #########################################
    model_list = [
        "meta-llama/Meta-Llama-3-8B","meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "mistralai/Mistral-7B-v0.1", "google/gemma-7b",
    ]
    if args.model not in model_list:
        raise ValueError("Only support the language classification models including: ".format(model_list))
    #########################################
    # load dataloader
    #########################################
    
    if args.dataset == "flanv2":
        traindata_list, valdata_list = flanv2(args)
        #if args.task_name == 'MMLU':
        #    metric = "accuracy" 
        #elif args.task_name == 'QA':
        #    metric = "F1" 
        #MMLU: 5 shot setting, accuracy
        #QA: 1-shot setting, F1
        #TODO: how to compute that
    elif args.dataset == "wikitext2":
        train_loader, val_loader = wikitext2(args)
        #if args.task_name == 'text-generation':
        #metric = "perplexity" 
    elif args.dataset == "alpaca":
        train_loader, val_loader = alpaca_llama(args)
        #if args.task_name == 'text-generation':
        #metric = "spearmanr" 
    elif args.dataset == "alpacagpt4":
        train_loader, val_loader = alpacagpt4_llama(args)
    elif args.dataset == "zeroshot":
        train_loader, val_loader = zeroshot_llama(args)
    elif args.dataset == "gsm8k":
        train_loader, val_loader = gsm8k(args)
    
    metric = "perplexity" 

    print("Dataset: {}, task: {}, metric: {}".format(args.dataset, args.task_name, metric))
    config = AutoConfig.from_pretrained(args.model, finetuning_task="text-generation", token = auth_token, cache_dir = hf_cache)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype = torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, token = auth_token, trust_remote_code=True, device_map={"":0},
                                                quantization_config = quantization_config, 
                                                torch_dtype=torch.bfloat16,
                                                cache_dir = hf_cache,
                                                )
    model = prepare_model_for_kbit_training(model)
    #Do not need to reset the classifier, the pretrained parameters does not include classification layer.

    model_name = args.model.split("/")[-1]
    checkpoint_folder = os.path.join(os.path.join(args.model_checkpoint, model_name), args.dataset + "_" + args.task_name)  
    if args.method == 'qrosa': #Not implemented
        checkpoint_folder = os.path.join(checkpoint_folder, "rank" + str(args.finetuning_rank*2))
    else:
        checkpoint_folder = os.path.join(checkpoint_folder, "rank" + str(args.finetuning_rank))
    
    if not os.path.isdir(checkpoint_folder) and not (args.method == "memory_test"):
        os.makedirs(checkpoint_folder, exist_ok=True)

    target_modules = []
    if args.finetune_attn_as_linear:
        print("Finetune attention layer as independent linear layers.")
    else:
        print("Freeze and not train attention layers.")
    for n, m in model.named_modules():
        if (
            isinstance(m, Linear) and m != model.lm_head 
            and ((args.finetune_attn_as_linear) or (not args.finetune_attn_as_linear and "self_attn" not in n))
        ):
            target_modules.append(n)

    import copy
    if args.method == "qlora":
        from peft import LoraConfig
        peft_config = LoraConfig(
                task_type = "CAUSAL_LM", 
                inference_mode=False, 
                r = args.finetuning_rank, 
                lora_alpha = 16, 
                lora_dropout = 0.1, #0.0: no dropout, >0.0: dropout
                bias= "none", 
                target_modules = target_modules,
            )
    elif args.method == "memory_test":
        from peft import LoraConfig
        peft_config = LoraConfig(
                task_type = "CAUSAL_LM", 
                inference_mode=False, 
                r = args.finetuning_rank, 
                lora_alpha = args.finetuning_rank, 
                lora_dropout = 0.0, #0.0: no dropout, >0.0: dropout
                bias= "none", 
                target_modules = target_modules,
            )
    elif args.method == "loftq":
        from peft import LoraConfig
        ft_model = build_peft_model(args, DEVICE, train_loader, target_modules)
        peft_config = LoraConfig(
                task_type = "CAUSAL_LM", 
                inference_mode=False, 
                #init_lora_weights="loftq",
                r = args.finetuning_rank, 
                lora_alpha = 16, 
                lora_dropout = 0.1, #0.0: no dropout, >0.0: dropout
                bias= "none", 
                target_modules = target_modules,
            )
    elif args.method == "qdora":
        from peft import LoraConfig
        peft_config = LoraConfig(
                task_type = "CAUSAL_LM", 
                inference_mode=False, 
                r = args.finetuning_rank, 
                lora_alpha = 16, 
                lora_dropout = 0.1, #0.0: no dropout, >0.0: dropout
                bias= "none", 
                use_dora = True,
                target_modules = target_modules,
            )
    elif args.method == "qpissa":
        from peft import LoraConfig
        ft_model = build_peft_model(args, DEVICE, train_loader, target_modules)
        peft_config = LoraConfig(
                task_type = "CAUSAL_LM", 
                inference_mode=False, 
                #init_lora_weights="pissa_niter_4",
                r = args.finetuning_rank, 
                lora_alpha = 4, 
                lora_dropout = 0.1, #0.0: no dropout, >0.0: dropout
                bias= "none", 
                target_modules = target_modules,
            )
    elif 'ours' in args.method:
        from peft import LoraConfig
        ft_model = build_our_model(args, DEVICE, train_loader)
        peft_config = LoraConfig(
                task_type = "CAUSAL_LM", 
                inference_mode=False, 
                r = args.finetuning_rank, 
                lora_alpha = args.finetuning_rank, 
                lora_dropout = 0.0, #0.0: no dropout, >0.0: dropout
                bias= "none", 
                target_modules = target_modules,
            )
    elif args.method in ['qrosa', 'gptq']:
        raise NotImplementedError("qrosa and gptq haven't been implemented.")
    
    if args.method not in ["fft"]:
        model = get_peft_model(model, peft_config)

    if 'ours' in args.method or args.method in ['qpissa', 'loftq']:
        state_dict = {}
        printed_param = ft_model.get_parameter("base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight") \
                        if args.method in ['qpissa', 'loftq']\
                        else ft_model.get_parameter("model.layers.0.self_attn.q_proj.basic_layer.weight")
        weight_nf4, weight_dequantized = quantize_and_dequantized(printed_param)
        del printed_param
        print("weight_nf4 dtype: {}, size: {}, quant_state: {}".format(weight_nf4.dtype, weight_nf4.size(), weight_nf4.quant_state))
        print(weight_nf4)
        for name, param in model.named_parameters():                
            if 'ours' in args.method and 'lora_B' in name:
                param.requires_grad_(False) 
            if "base_layer" in name:
                if 'ours' in args.method:
                    ##############################################################################
                    # base_model.model.model.layers.0.self_attn.q_proj   (.base_layer.weight     )
                    # base_model.model.model.layers.0.self_attn.q_proj   (.lora_A.default.weight )
                    # base_model.model.model.layers.0.self_attn.q_proj   (.lora_B.default.weight )
                    ##############################################################################
                    # model.layers.0.self_attn.q_proj   (.basic_layer.weight   )
                    # model.layers.0.self_attn.q_proj   (.finetuned_out_weight )
                    ##############################################################################
                    path = name.replace('base_model.model.', '').replace('.base_layer.weight', '').split('.')
                    module = ft_model
                    for p in path:
                        module = getattr(module, p)
                    del path
                    state_dict[name.replace('base_layer', 'lora_A.default')] = module.finetuned_out_weight.data
                    state_dict[name.replace('base_layer', 'lora_B.default')] = module.finetuned_out_mapping.transpose(0,1).data
                    if 'loftq' in args.method:
                        weight = module.basic_layer.weight.to(device=DEVICE, dtype=torch.float32)
                        size_mapping = module.finetuned_out_mapping
                        finetune_idxs = size_mapping.argmax(dim=1).tolist()
                        res = weight.clone()
                        for i in range(args.n_iter):
                            torch.cuda.empty_cache()
                            # Quantization
                            weight_nf4, weight_dequantized = quantize_and_dequantized(res)

                            res = weight - weight_dequantized

                            # select the residual via idx
                            finetuned_weight = torch.index_select(res, 0, torch.LongTensor(finetune_idxs).to(res.device))
                            output = finetuned_weight.movedim(0, -1).matmul(size_mapping).movedim(-1, 0)
                            res = weight - output
                        state_dict[name.replace('base_layer', 'lora_A.default')] = finetuned_weight.data
                        ratio = (1-(weight - (weight_dequantized + output)).norm(p='nuc')/(weight - weight_dequantized).norm(p='nuc'))
                        row_weight = torch.index_select(weight, 0, torch.LongTensor(finetune_idxs).to(res.device))
                        row_weight_dequantized = torch.index_select(weight_dequantized, 0, torch.LongTensor(finetune_idxs).to(res.device))
                        ratio_row = (1-( row_weight - (row_weight_dequantized + finetuned_weight)).norm(p='nuc')/(row_weight - row_weight_dequantized).norm(p='nuc'))
                        print("Param name: {}, initial quantization error reduction ratio: {:.2f}%, error reduction ratio for only fine-tuning rows: {:.2f}%.".format(name.replace('base_model.model.', '').replace('.base_layer.weight', ''), ratio*100, ratio_row*100))
                        del weight, ratio, row_weight, row_weight_dequantized, res, output, finetuned_weight, ratio_row, finetune_idxs, size_mapping
                elif args.method == 'qpissa':
                    weight = ft_model.get_parameter(name)
                    res = weight.to(torch.float32)
                    r = args.finetuning_rank
                    for i in range(args.n_iter):
                        U, S, Vh = torch.linalg.svd(res, full_matrices=False)
                        lora_B = U @ (torch.sqrt(torch.diag(S)[:, :r]))
                        lora_A = torch.sqrt(torch.diag(S)[:r, :]) @ Vh
                        output = lora_B @ lora_A
                        res = weight - output
                        weight_nf4, weight_dequantized = quantize_and_dequantized(res)
                        res = weight - weight_dequantized
                    state_dict[name.replace("base_layer", "lora_A.default")] = lora_A.data
                    state_dict[name.replace("base_layer", "lora_B.default")] = lora_B.data
                    ratio = (1-(weight - (weight_dequantized + output)).norm(p='nuc')/(weight - weight_dequantized).norm(p='nuc'))
                    print("Param name: {}, initial quantization error reduction ratio: {:.2f}%".format(name.replace('base_model.model.', '').replace('.base_layer.weight', ''), ratio*100))
                    del weight, ratio, lora_A, lora_B, res, output, r, U, S, Vh
                elif args.method == 'loftq':
                    weight = ft_model.get_parameter(name)
                    res = weight.clone().to(torch.float32)
                    r = args.finetuning_rank
                    for i in range(args.n_iter):
                        weight_nf4, weight_dequantized = quantize_and_dequantized(res)
                        res = weight - weight_dequantized
                        # Decompose the residual by SVD
                        U, S, Vh = torch.linalg.svd(res, full_matrices=False)
                        lora_B = U @ (torch.sqrt(torch.diag(S)[:, :r]))
                        lora_A = torch.sqrt(torch.diag(S)[:r, :]) @ Vh
                        output = lora_B @ lora_A
                        res = weight - output
                    state_dict[name.replace("base_layer", "lora_A.default")] = lora_A.data
                    state_dict[name.replace("base_layer", "lora_B.default")] = lora_B.data
                    ratio = (1-(weight - (weight_dequantized + output)).norm(p='nuc')/(weight - weight_dequantized).norm(p='nuc'))
                    print("Param name: {}, initial quantization error reduction ratio: {:.2f}%".format(name.replace('base_model.model.', '').replace('.base_layer.weight', ''), ratio*100))
                    del weight, ratio, lora_A, lora_B, res, output, r, U, S, Vh
        model.load_state_dict(state_dict, strict=False)
        del ft_model, weight_nf4, weight_dequantized, state_dict
        torch.cuda.empty_cache()
    #########################################
    
    print("{} to be finetuned.".format(model_name))
    print(model)

    optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0,
        )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)
    decay_type = 'step linear decay'
    #optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    #scheduler = get_scheduler(
    #    name = "linear",
    #    optimizer=optimizer,
    #    num_warmup_steps = int(args.EPOCHS * len(train_loader) * 0.03),
    #    num_training_steps = args.EPOCHS * len(train_loader), 
    #    )
    #decay_type = 'linear decay'
    print("Optimizer settings: \nlearning rate: {}\nscheduler: {} with warmup steps: {}\nnum_training_steps: {}".format(
        args.learning_rate, decay_type, int(args.EPOCHS * len(train_loader) * 0.03), args.EPOCHS * len(train_loader)
    ))
    #train_loader, val_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, val_loader, model, optimizer, scheduler)
    

    #finetune the model here
    print("\n----------------------------------------")
    n_trainable_params = 0
    n_params = 0
    for name, param in model.named_parameters():
        if args.method == "memory_test" and 'lora_B' in name:
            param.requires_grad_(False) 
        print ("Parameters: {}, requires_grad: {}, size: {}.".format(name, param.requires_grad, param.size()))
        n_params += param.numel()
        if param.requires_grad:
            n_trainable_params += param.numel()
    print ("Number of all parameters: {}.\nNumber of trainable parameters: {}.".format(n_params, n_trainable_params))
    print("----------------------------------------\n")
        
    print("Testing perplexity of the pretrained foundation model {}".format(model_name))
    # Reset the peak memory stats
    torch.cuda.reset_peak_memory_stats(0)
    start_time = time.time()
    loss_ori, ppl_ori = val_text_generation(model, DEVICE, val_loader, metric)
    end_time = time.time()
    print("Max memory allocated in validation: {}GB".format(torch.cuda.max_memory_allocated()/(1024**3)))
    print('Inference of {} took {:.2f}s'.format(args.method, end_time - start_time))
    Best_metric = ppl_ori
    checkpoint = {'model': model.state_dict(), 'best_metric': Best_metric}
    checkpoint_name = model_name + ("_attn_as_linear" if args.finetune_attn_as_linear else "") + "_" + args.method
    
    checkpoint_file = os.path.join(checkpoint_folder, checkpoint_name + '_epoch_{}.pth'.format(args.START_EPOCH-1))
    if args.START_EPOCH==1:
        if not args.method == "memory_test":
            torch.save(checkpoint, os.path.join(checkpoint_folder, checkpoint_name + '_best.pth') )
        training_loss_history = []
        val_loss_history = []
        metric_history = []
    else:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model'])   
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        training_loss_history = checkpoint['training_loss']
        val_loss_history = checkpoint['val_loss']
        metric_history = checkpoint['metric']
        Best_metric = checkpoint['best_metric']

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir=checkpoint_folder)

    for epoch in range(args.START_EPOCH, args.EPOCHS + 1):
        # Reset the peak memory stats
        torch.cuda.reset_peak_memory_stats(0)

        start_time = time.time()
        train_loss = train_text_generation(model, DEVICE, train_loader, optimizer, epoch)
        scheduler.step()
        end_time = time.time()
        # Check the maximum memory allocated
        print("Max memory allocated in training: {}GB".format(torch.cuda.max_memory_allocated()/(1024**3)))
        print('Training of {} took {:.2f}s'.format(args.method, end_time - start_time))
        
        # Reset the peak memory stats
        torch.cuda.reset_peak_memory_stats(0)
        start_time = time.time()
        val_loss, val_metric = val_text_generation(model, DEVICE, val_loader, metric)
        end_time = time.time()
        # Check the maximum memory allocated
        print("Max memory allocated in validation: {}GB".format(torch.cuda.max_memory_allocated()/(1024**3)))
        print('Inference of {} took {:.2f}s'.format(args.method, end_time - start_time))
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar(metric+"/val", val_metric, epoch)
        
        training_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        metric_history.append(val_metric)
        #accelerator.wait_for_everyone()
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'training_loss': training_loss_history,
            'val_loss': val_loss_history,
            'metric': metric_history,
            'best_metric': Best_metric,
            }
        if (metric == "perplexity" and val_metric < Best_metric) or (metric in ["spearmanr", "F1", "accuracy" ] and val_metric > Best_metric):
            checkpoint['best_metric'] = val_metric
            if not args.method == "memory_test":
                torch.save(checkpoint, os.path.join(checkpoint_folder, checkpoint_name + '_best.pth') )
            Best_metric = val_metric
        print('Best metric: {}\n'.format( Best_metric))
        if epoch % args.store_frequency == 0 and not args.method == "memory_test":
            torch.save(checkpoint, os.path.join(checkpoint_folder, checkpoint_name + '_epoch_{}.pth'.format(epoch)) )
    #accelerator.end_training()
    if args.method != "fft":
        model = merge_model(args, target_modules, model)
        torch.cuda.empty_cache()
    if not args.method == "memory_test":
        torch.save(model.state_dict(), os.path.join(checkpoint_folder, checkpoint_name + '_final_merged.pth') )
    writer.close()

    #=============================================================================================

    print("Testing perplexity of the finel finetuned model...")
    loss_Finetuned, metric_Finetuned = val_text_generation(model, DEVICE, val_loader, metric)
    print('Best metric: {}\n'.format( Best_metric))

    print("----------------------------------------")
    print("Now evaluating the model by EleutherAI's LM-Evaluation-Harness")
    print("Note that we only use this for gsm8k in the paper.")
    from lm_eval import evaluator 
    from lm_eval.models.huggingface import HFLM 
    num_fewshot = 0
    if "70b" in args.model or "65b" in args.model:
        limit = 2000
    else:
        limit = None

    if args.dataset == "zeroshot":
        task_list=["boolq","piqa","social_iqa","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"]
    elif args.dataset == "gsm8k":
        #task_list = ["gsm8k", "hendrycks_math"] 
        task_list = ["gsm8k"] 
    elif args.dataset == "coding":
        task_list=["code2text_python",]
    else:
        task_list=["boolq","piqa","social_iqa","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa", "ifeval"]
        #task_list=["boolq","piqa","social_iqa","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"]

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