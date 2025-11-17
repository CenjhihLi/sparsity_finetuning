import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import random
import time
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.nn import Linear
from torch.autograd import Variable
from methods.data_loader import flanv2, wikitext2, alpaca_llama, alpacagpt4_llama, gsm8k, zeroshot_llama
from evaluate import load
from peft import get_peft_model
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
    #parser.add_argument('--use_dora', default=False, action='store_true', help='Finetune the model using dora.')
    parser.add_argument('--method', default='lora', type=str, help='peft method choice', choices=['lora', 'pissa', 'dora', 'vera', 'rosa'])
    parser.add_argument('--p_dropout', default = 0.1, type=float, help='finetuning dropout prob.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for model training.')
    parser.add_argument('--warmup_steps', type=float, default=0, help='Warm up steps (Learning rate will not dacay) for model training.')
    parser.add_argument('--BATCH_SIZE', type=int, default=1, help='Batch size for model training.')
    parser.add_argument('--EPOCHS', type=int, default=10, help='Max epochs for model training.')    
    parser.add_argument('--START_EPOCH', type=int, default=1, help='Start epochs for model training. (continue last training)')    
    parser.add_argument('--gpu', type=str, default='0', help='GPU using, i.e. \'0,1,2\'')     
    parser.add_argument('--parallel', default=False, action='store_true', help='paralleled computing')
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
            #pred = out.logits
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
    #metric_name = task2metrc[args.task_name]
    #import datasets
    #datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    
    #########################################
    # check model list
    #########################################
    model_list = [
        "meta-llama/Meta-Llama-3-8B","meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", "google/gemma-7b",
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, token = auth_token, cache_dir = hf_cache)
    #Do not need to reset the classifier, the pretrained parameters does not include classification layer.

    model_name = args.model.split("/")[-1]
    checkpoint_folder = os.path.join(os.path.join(args.model_checkpoint, model_name), args.dataset + "_" + args.task_name)  
    if args.method == "rosa":
        checkpoint_folder = os.path.join(checkpoint_folder, "rank" + str(args.finetuning_rank*2))
    else:
        checkpoint_folder = os.path.join(checkpoint_folder, "rank" + str(args.finetuning_rank))
    
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)

    for p in model.parameters():
        p.requires_grad_(True)

    target_modules = []
    if args.finetune_attn_as_linear:
        print("Finetune attention layer as independent linear layers.")
    else:
        print("Freeze and not train attention layers.")
    for n, m in model.named_modules():
        if (
            isinstance(m, Linear) and m != model.lm_head 
            and ((args.finetune_attn_as_linear) or 
                 (not args.finetune_attn_as_linear and 
                    ('up_proj' in n or 'gate_proj' in n or 'o_proj' in n or 'down_proj' in n)))
        ):
            target_modules.append(n)

    import copy
    if args.method == "lora":
        from peft import LoraConfig
        peft_config = LoraConfig(
                task_type = "CAUSAL_LM", 
                inference_mode=False, 
                r = args.finetuning_rank, 
                lora_alpha = 16, 
                lora_dropout = args.p_dropout, #0.0: no dropout, >0.0: dropout 
                bias= "none", 
                target_modules = target_modules,
            )
    elif args.method == "dora":
        from peft import LoraConfig
        peft_config = LoraConfig(
                task_type = "CAUSAL_LM", 
                inference_mode=False, 
                r = args.finetuning_rank, 
                lora_alpha = 16, 
                lora_dropout = args.p_dropout, #0.0: no dropout, >0.0: dropout
                bias= "none", 
                use_dora = True,
                target_modules = target_modules,
            )
    elif args.method == "pissa":
        from peft import LoraConfig
        peft_config = LoraConfig(
                task_type = "CAUSAL_LM", 
                inference_mode=False, 
                #init_lora_weights="pissa_niter_4",
                init_lora_weights="pissa",
                r = args.finetuning_rank, 
                lora_alpha = 16, 
                lora_dropout = 0.1, #0.0: no dropout, >0.0: dropout
                bias= "none", 
                target_modules = target_modules,
            )
    elif args.method == "vera":
        from peft import VeraConfig
        peft_config = VeraConfig(
                r = args.finetuning_rank, 
                vera_dropout = args.p_dropout, #0.0: no dropout, >0.0: dropout
                bias= "none", 
                target_modules = target_modules,
                save_projection = False,
            )
    elif args.method == "rosa":
        from peft.tuners.rosa import RosaConfig, RosaScheduler
        peft_config = RosaConfig(
                task_type="CAUSAL_LM", 
                r = args.finetuning_rank,
                lora_alpha = 16,
                lora_dropout = args.p_dropout, #0.0: no dropout, >0.0: dropout
                bias = "none", 
                target_modules = target_modules,

                d = args.finetuning_density, 
                spa_num_grads=1, 
                grad_acc_mode='mean_squared',    
                rosa_dtype = 'fp32', #out of memory
                #rosa_dtype = 'bf16', 
                schedule='wl64', 
            )

    model = get_peft_model(model, peft_config)
    model.to(DEVICE)

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

    print("num_training_steps: {} with {}".format(args.EPOCHS * len(train_loader), decay_type))
    #train_loader, val_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, val_loader, model, optimizer, scheduler)
    
    #finetune the model here
    print("\n----------------------------------------")
    n_trainable_params = 0
    n_params = 0
    for name, param in model.named_parameters():
        print ("Parameters: {}, requires_grad: {}, size: {}.".format(name, param.requires_grad, param.size()))
        n_params += param.numel()
        if param.requires_grad:
            n_trainable_params += param.numel()
    print ("Number of all parameters: {}.\nNumber of trainable parameters: {}.".format(n_params, n_trainable_params))
    print("----------------------------------------\n")

    print("Testing perplexity of the pretrained foundation model {}".format(model_name))
    loss_ori, ppl_ori = val_text_generation(model, DEVICE, val_loader, metric)
    Best_metric = ppl_ori
    checkpoint = {'model': model.state_dict(), 'best_metric': Best_metric}
    checkpoint_name = model_name + ("_attn_as_linear" if args.finetune_attn_as_linear else "") + "_" + args.method
    checkpoint_name = checkpoint_name + ("_nodropout" if args.p_dropout==0.0 else "")
    checkpoint_file = os.path.join(checkpoint_folder, checkpoint_name + '_epoch_{}.pth'.format(args.START_EPOCH-1))
    if args.START_EPOCH==1:
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
            torch.save(checkpoint, os.path.join(checkpoint_folder, checkpoint_name + '_best.pth') )
            Best_metric = val_metric
        print('Best metric: {}\n'.format( Best_metric))
        if epoch % args.store_frequency == 0:
            torch.save(checkpoint, os.path.join(checkpoint_folder, checkpoint_name + '_epoch_{}.pth'.format(epoch)) )
    #accelerator.end_training()
    model = model.merge_and_unload()
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
        #task_list=["boolq","piqa","social_iqa","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa", "ifeval"]
        task_list=["boolq","piqa","social_iqa","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"]

    model = HFLM(pretrained=model, tokenizer = tokenizer, backend = "causal", trust_remote_code = True)
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