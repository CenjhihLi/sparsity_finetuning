import torchvision.transforms as transforms
import torch
import copy
import os
import re

from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import Caltech101, ImageFolder
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, DataCollatorWithPadding, default_data_collator, GenerationConfig
from accelerate import Accelerator
from datasets import load_dataset
from datasets import Dataset as hf_Dataset

__all__ = ["imagenet_1k", "tiny_imagenet", "transform", "tiny_imagenet_c", "cifar100", "caltech101", "glue", "flanv2", "alpaca", "alpaca_llama", "wikitext2"]

accelerator = Accelerator()

use_imagenet_mean_std = True
imagenet_mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5]
imagenet_std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5]

use_cifar_mean_std = True
cifar_mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343] if use_imagenet_mean_std else [0.5, 0.5, 0.5]
cifar_std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404] if use_imagenet_mean_std else [0.5, 0.5, 0.5]

imagenet_transform = transforms.Compose([
    transforms.Resize((256,256), interpolation=InterpolationMode.BILINEAR, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

cifar_transform = transforms.Compose([
    transforms.Resize((224,224), interpolation=InterpolationMode.BILINEAR, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
])

caltech101_transform = transforms.Compose([
    transforms.Resize((256,256), interpolation=InterpolationMode.BILINEAR, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

auth_token="your_token"
hf_cache = '/your_path/huggingface_cache'

def imagenet_map(instances):
    instances['image'] = [imagenet_transform(img.convert("RGB")) for img in instances['image']]
    return instances

def cifar_map(instances):
    instances['image'] = [cifar_transform(img.convert("RGB")) for img in instances['image']]
    return instances

glue_columns = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli_mismatched": ("premise", "hypothesis"),
    "mnli_matched": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

alpaca_prompt_template = {
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:",
}
def generate_prompt(data_point, label = True):
    #if len(data_point["input"].replace(' ',''))==0:
    if len(data_point["input"])==0:
        prompt = alpaca_prompt_template["prompt_no_input"].format(instruction=data_point["instruction"])
    else:
        prompt = alpaca_prompt_template["prompt_input"].format(instruction = data_point["instruction"], input = data_point["input"])
    if label:
        prompt = prompt + data_point["output"]
    return prompt
#To obtain response: output.split("### Response:")[1].strip()

class imageDataset(Dataset):
    def __init__(self, dataset, data_map) -> None:
        dataset.set_format("torch")
        dataset.set_transform(data_map)
        self.dataset = dataset
    
    def __getitem__(self, index):
        item = self.dataset[index]
        return item["image"], item["label"]
    
    def __len__(self):
        return len(self.dataset)

class myLMDataset(TensorDataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
    
    def __getitem__(self, index):
        item = self.dataset[index]
        return item[0], item[1]
    
    def __len__(self):
        return len(self.dataset)

class imagenet_C_Dataset(Dataset):
    def __init__(self, dataset, method, severity=1) -> None:
        def imagenet_c_map(instances):
            instances['image'] = [method(imagenet_transform(img.convert("RGB")), severity=severity) for img in instances['image']]
            return instances
        dataset.set_format("torch")
        dataset.set_transform(imagenet_c_map)
        self.dataset = dataset
    
    def __getitem__(self, index):
        item = self.dataset[index]
        return item["image"], item["label"]
    
    def __len__(self):
        return len(self.dataset)

def label_order_tiny_imagenet(args):
    from datasets import load_dataset    
    train_data = load_dataset('zh-plus/tiny-imagenet', split = 'train')
    train_data = imageDataset(train_data, data_map = imagenet_map)

    val_data = load_dataset('zh-plus/tiny-imagenet', split = 'valid')
    val_data = imageDataset(val_data, data_map = imagenet_map)
    
    """
    train_loader_list = []
    val_loader_list = []
    for label in range(200):
        train_loader = DataLoader(Subset(train_data, range(500*label, 500*(label+1))), batch_size = args.BATCH_SIZE,  shuffle=False)
        val_loader = DataLoader(Subset(val_data, range(500*label, 500*(label+1))), batch_size = args.BATCH_SIZE,  shuffle=False)
        train_loader_list.append(train_loader)
        val_loader_list.append(val_loader)

    return train_loader_list, val_loader_list
    """
    train_loader = DataLoader(train_data, batch_size = args.BATCH_SIZE,  shuffle=False)
    val_loader = DataLoader(val_data, batch_size = args.BATCH_SIZE,  shuffle=False)

    return train_loader, val_loader


def tiny_imagenet(args):
    from datasets import load_dataset    
    train_data = load_dataset('zh-plus/tiny-imagenet', split = 'train')
    train_data = imageDataset(train_data, data_map = imagenet_map)

    val_data = load_dataset('zh-plus/tiny-imagenet', split = 'valid')
    val_data = imageDataset(val_data, data_map = imagenet_map)

    if args.parallel:
        train_sampler = DistributedSampler(train_data)
        val_sampler = DistributedSampler(val_data)
        train_loader = DataLoader(train_data, sampler = train_sampler, batch_size = args.BATCH_SIZE,  shuffle=False)
        val_loader = DataLoader(val_data, sampler = val_sampler, batch_size = args.BATCH_SIZE,  shuffle=False)
    else:
        train_loader = DataLoader(train_data, batch_size = args.BATCH_SIZE,  shuffle=True)
        val_loader = DataLoader(val_data, batch_size = args.BATCH_SIZE,  shuffle=False)

    return train_loader, val_loader

def imagenet_1k(args):
    from datasets import load_dataset    
    train_data = load_dataset("imagenet-1k", trust_remote_code=True, token=auth_token, split = 'train')
    train_data = imageDataset(train_data, data_map = imagenet_map)

    val_data = load_dataset("imagenet-1k", trust_remote_code=True, token=auth_token, split = 'validation')
    val_data = imageDataset(val_data, data_map = imagenet_map)

    if args.parallel:
        train_sampler = DistributedSampler(train_data)
        val_sampler = DistributedSampler(val_data)
        train_loader = DataLoader(train_data, sampler = train_sampler, batch_size = args.BATCH_SIZE,  shuffle=False)
        val_loader = DataLoader(val_data, sampler = val_sampler, batch_size = args.BATCH_SIZE,  shuffle=False)
    else:
        train_loader = DataLoader(train_data, batch_size = args.BATCH_SIZE,  shuffle=True)
        val_loader = DataLoader(val_data, batch_size = args.BATCH_SIZE,  shuffle=False)

    return train_loader, val_loader

def label_order_cifar100(args):
    from datasets import load_dataset, concatenate_datasets
    raw_data = load_dataset('uoft-cs/cifar100', split = 'train')
    raw_data = raw_data.rename_column("img", "image")
    raw_data = raw_data.rename_column("fine_label", "label")
    train_data = []
    for label in range(100):
        train_data.append(raw_data.filter(lambda example: example["label"]==label))
    train_data = concatenate_datasets(train_data)
    train_data = imageDataset(train_data, data_map = cifar_map)

    raw_data = load_dataset('uoft-cs/cifar100', split = 'test')
    raw_data = raw_data.rename_column("img", "image")
    raw_data = raw_data.rename_column("fine_label", "label")
    val_data = []
    for label in range(100):
        val_data.append(raw_data.filter(lambda example: example["label"]==label))
    val_data = concatenate_datasets(val_data)
    val_data = imageDataset(val_data, data_map = cifar_map)

    if args.parallel:
        train_sampler = DistributedSampler(train_data)
        val_sampler = DistributedSampler(val_data)
        train_loader = DataLoader(train_data, sampler = train_sampler, batch_size = args.BATCH_SIZE,  shuffle=False)
        val_loader = DataLoader(val_data, sampler = val_sampler, batch_size = args.BATCH_SIZE,  shuffle=False)
    else:
        train_loader = DataLoader(train_data, batch_size = args.BATCH_SIZE,  shuffle=False)
        val_loader = DataLoader(val_data, batch_size = args.BATCH_SIZE,  shuffle=False)

    return train_loader, val_loader

def cifar100(args):
    from datasets import load_dataset    
    train_data = load_dataset('uoft-cs/cifar100', split = 'train')
    train_data = train_data.rename_column("img", "image")
    train_data = train_data.rename_column("fine_label", "label")
    train_data = imageDataset(train_data, data_map = cifar_map)

    val_data = load_dataset('uoft-cs/cifar100', split = 'test')
    val_data = val_data.rename_column("img", "image")
    val_data = val_data.rename_column("fine_label", "label")
    val_data = imageDataset(val_data, data_map = cifar_map)

    if args.parallel:
        train_sampler = DistributedSampler(train_data)
        val_sampler = DistributedSampler(val_data)
        train_loader = DataLoader(train_data, sampler = train_sampler, batch_size = args.BATCH_SIZE,  shuffle=False)
        val_loader = DataLoader(val_data, sampler = val_sampler, batch_size = args.BATCH_SIZE,  shuffle=False)
    else:
        train_loader = DataLoader(train_data, batch_size = args.BATCH_SIZE,  shuffle=True)
        val_loader = DataLoader(val_data, batch_size = args.BATCH_SIZE,  shuffle=False)

    return train_loader, val_loader

def caltech101(args):
    dataset = Caltech101(args.data_path, download = True)
    del dataset
    path = os.path.join(os.path.join(args.data_path, 'caltech101'), "101_ObjectCategories")
    dataset = ImageFolder(path, transform=imagenet_transform)
    n_sample = len(dataset)
    n_train = int(0.75*n_sample)
    train_data, val_data = torch.utils.data.random_split(dataset, [n_train, n_sample-n_train])
    train_loader = DataLoader(train_data, batch_size = args.BATCH_SIZE,  shuffle=True)
    val_loader = DataLoader(val_data, batch_size = args.BATCH_SIZE,  shuffle=False)
    return train_loader, val_loader

def glue(args):
    raw_datasets = load_dataset("glue", args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir = hf_cache)
    column1, column2 = glue_columns[args.task_name]
    padding = "max_length" if args.pad_to_max_length else False

    def preprocessing(instances):
        # Tokenize the texts
        texts = (
            (instances[column1],) if column2 is None else (instances[column1], instances[column2])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
        result["labels"] = instances["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocessing, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    train_dataset = processed_datasets["train"]
    val_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator # convert to tensor
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
        #with padding

    train_loader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=args.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, collate_fn=data_collator, batch_size=args.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def flanv2(args):
    #"SirNeural/flan_v2", that has 336 million inputs, is too large
    #"ostapeno/flanv2_100k_2" has 100k inputs
    raw_datasets = load_dataset("ostapeno/flanv2_100k_2", split = 'train')
    raw_datasets = raw_datasets.train_test_split()
    train_dataset = raw_datasets["train"]
    val_dataset = raw_datasets["test"]
    del raw_datasets

    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    tokenizer.pad_token_id =(0)
    tokenizer.padding_side = "left"
    import random
    random.seed(args.seed)
    traininput = torch.zeros((args.n_samples, args.max_length), dtype=torch.int64)
    traintarget = torch.zeros((args.n_samples, args.max_length), dtype=torch.int64)
    for i in range(args.n_samples):
        while True:
            index = random.randint(0, len(train_dataset) - 1)
            encoding_prompt = tokenizer(train_dataset[index]['user'] + train_dataset[index]['assistant'], return_tensors='pt')
            if encoding_prompt.input_ids.shape[1] <= args.max_length:
                encoding_prompt = tokenizer(train_dataset[index]['user'] + train_dataset[index]['assistant'],
                                            max_length=args.max_length,
                                            padding=True,
                                            return_tensors='pt')
                break
        input = encoding_prompt.input_ids.squeeze()
        traininput[i,:] = input.squeeze()
        traintarget[i,:] = target.squeeze()
    train_dataset = TensorDataset(traininput, traintarget)
    train_loader = DataLoader(train_dataset, batch_size = args.BATCH_SIZE,  shuffle=False)
    
    valinput = torch.zeros((args.n_samples, args.max_length), dtype=torch.int64)
    valtarget = torch.zeros((args.n_samples, args.max_length), dtype=torch.int64)
    for _ in range(args.n_samples//4):
        while True:
            index = random.randint(0, len(val_dataset) - 1)
            encoding_prompt = tokenizer(val_dataset[index]['user'] + val_dataset[index]['assistant'], return_tensors='pt')
            if encoding_prompt.input_ids.shape[1] <= args.max_length:
                encoding_prompt = tokenizer(val_dataset[index]['user'],
                                            truncation=True,
                                            max_length=args.max_length,
                                            padding=False,
                                            return_tensors='pt')
                encoding_target = tokenizer(val_dataset[index]['assistant'],
                                            truncation=True,
                                            max_length=args.max_length,
                                            padding=False,
                                            return_tensors='pt')
                break
        input = encoding_prompt.input_ids.squeeze()
        target = encoding_target.input_ids.squeeze()
        target[:-args.max_length] = -100
        valinput[i,:] = input.squeeze()
        valtarget[i,:] = target.squeeze()
    val_dataset = TensorDataset(valinput, valtarget)
    val_loader = DataLoader(val_dataset, batch_size = args.BATCH_SIZE,  shuffle=False)
    
    #MMLU: 5 shot setting, accuracy
    #QA: 1-shot setting, F1
    #TODO: compute those metrics
    return train_loader, val_loader

def alpaca_llama(args):
    rawdata = load_dataset('tatsu-lab/alpaca', split='train')
    rawdata = rawdata.train_test_split()
    train_dataset = rawdata["train"]
    val_dataset = rawdata["test"]
    del rawdata

    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(data_point):
        prompt = torch.tensor(tokenizer.encode(generate_prompt(data_point, label = False)), dtype=torch.int64)
        input = tokenizer.encode(generate_prompt(data_point, label = True))
        input.append(tokenizer.eos_token_id)
        input = torch.tensor(input, dtype=torch.int64)
        labels = copy.deepcopy(input)
        labels[: len(prompt)] = -1
        input_mask = input.ge(0)
        label_mask = labels.ge(0)
        input[~input_mask] = 0
        labels[~label_mask] = -100

        return {
            "input_ids": input.tolist(),
            "labels": labels.tolist(),
            "attention_mask":input_mask.tolist(),
        }
    
    def preprocessing(data_point):
        for k, v in data_point.items():
            padding_length = args.max_length - len(v)
            if tokenizer.padding_side == "right":
                data_point[k] = torch.tensor(v + [tokenizer.pad_token_id] * padding_length, dtype=torch.int64)
            elif tokenizer.padding_side == "left":
                data_point[k] = torch.tensor([tokenizer.pad_token_id] * padding_length + v, dtype=torch.int64)
        return data_point
    
    import random
    random.seed(args.seed)

    def sample_fn(dataset):
        sampled_dataset = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
        }
        sampling_pool = list(range(len(dataset)))
        for i in range(args.n_samples):
            while True:
                random.shuffle(sampling_pool)
                index = sampling_pool.pop(0)
                tokenized_datapoint = tokenize_function(dataset[index])
                if len(tokenized_datapoint["input_ids"]) <= args.max_length and  tokenized_datapoint["input_ids"][-1] == tokenizer.eos_token_id:
                    for k,v in preprocessing(tokenized_datapoint).items():
                        sampled_dataset[k].append(v)
                    break
            if len(sampling_pool)<1:
                break
        return sampled_dataset
    train_dataset = hf_Dataset.from_dict(sample_fn(train_dataset)).with_format("torch")
    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    val_dataset = hf_Dataset.from_dict(sample_fn(val_dataset)).with_format("torch")
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

def alpacagpt4_llama(args):
    rawdata = load_dataset('vicgalle/alpaca-gpt4', split='train')
    rawdata = rawdata.train_test_split()
    train_dataset = rawdata["train"]
    val_dataset = rawdata["test"]
    del rawdata

    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    def tokenize_function(data_point):
        prompt = torch.tensor(tokenizer.encode(generate_prompt(data_point, label = False)), dtype=torch.int64)
        input = tokenizer.encode(generate_prompt(data_point, label = True))
        input.append(tokenizer.eos_token_id)
        input = torch.tensor(input, dtype=torch.int64)
        labels = copy.deepcopy(input)
        labels[: len(prompt)] = -1
        input_mask = input.ge(0)
        label_mask = labels.ge(0)
        input[~input_mask] = 0
        labels[~label_mask] = -100

        return {
            "input_ids": input.tolist(),
            "labels": labels.tolist(),
            "attention_mask":input_mask.tolist(),
        }
    
    def preprocessing(data_point):
        for k, v in data_point.items():
            padding_length = args.max_length - len(v)
            if tokenizer.padding_side == "right":
                data_point[k] = torch.tensor(v + [tokenizer.pad_token_id] * padding_length, dtype=torch.int64)
            elif tokenizer.padding_side == "left":
                data_point[k] = torch.tensor([tokenizer.pad_token_id] * padding_length + v, dtype=torch.int64)
        return data_point
    
    import random
    random.seed(args.seed)

    def sample_fn(dataset):
        sampled_dataset = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
        }
        sampling_pool = list(range(len(dataset)))
        for i in range(args.n_samples):
            while True:
                random.shuffle(sampling_pool)
                index = sampling_pool.pop(0)
                tokenized_datapoint = tokenize_function(dataset[index])
                if len(tokenized_datapoint["input_ids"]) <= args.max_length and  tokenized_datapoint["input_ids"][-1] == tokenizer.eos_token_id:
                    for k,v in preprocessing(tokenized_datapoint).items():
                        sampled_dataset[k].append(v)
                    break
            if len(sampling_pool)<1:
                break
        return sampled_dataset
    train_dataset = hf_Dataset.from_dict(sample_fn(train_dataset)).with_format("torch")
    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    val_dataset = hf_Dataset.from_dict(sample_fn(val_dataset)).with_format("torch")
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

def mtbench():
    val_dataset = load_dataset('philschmid/mt-bench', split='train') #80 rows
    return val_dataset

def hendrycks_math(args):
    train_dataset = load_dataset('EleutherAI/hendrycks_math', 'main', split='train') #7.47k
    val_dataset = load_dataset('EleutherAI/hendrycks_math', 'main', split='test') #1.32k

    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    tokenizer.pad_token_id = tokenizer.eos_token_id
        
    def tokenize_function(data_point):
        prompt_text = "\nQuestion: {}\nSolution:".format(data_point["question"])
        prompt = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.int64)

        prompt_text = "\nQuestion: {}\nSolution: {}".format(data_point["question"], data_point["answer"])
        input = tokenizer.encode(prompt_text)
        input.append(tokenizer.eos_token_id)
        input = torch.tensor(input, dtype=torch.int64)
        labels = copy.deepcopy(input)
        labels[: len(prompt)] = -1
        input_mask = input.ge(0)
        label_mask = labels.ge(0)
        input[~input_mask] = 0
        labels[~label_mask] = -100

        return {
            "input_ids": input.tolist(),
            "labels": labels.tolist(),
            "attention_mask":input_mask.tolist(),
        }
    
    def preprocessing(data_point):
        for k, v in data_point.items():
            padding_length = args.max_length - len(v)
            if tokenizer.padding_side == "right":
                data_point[k] = torch.tensor(v + [tokenizer.pad_token_id] * padding_length, dtype=torch.int64)
            elif tokenizer.padding_side == "left":
                data_point[k] = torch.tensor([tokenizer.pad_token_id] * padding_length + v, dtype=torch.int64)
        return data_point
    
    import random
    random.seed(args.seed)

    def sample_fn(dataset):
        sampled_dataset = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
        }
        sampling_pool = list(range(len(dataset)))
        for i in range(args.n_samples):
            while True:
                random.shuffle(sampling_pool)
                index = sampling_pool.pop(0)
                tokenized_datapoint = tokenize_function(dataset[index])
                if len(tokenized_datapoint["input_ids"]) <= args.max_length and  tokenized_datapoint["input_ids"][-1] == tokenizer.eos_token_id:
                    for k,v in preprocessing(tokenized_datapoint).items():
                        sampled_dataset[k].append(v)
                    break
            if len(sampling_pool)<1:
                break
        return sampled_dataset
    train_dataset = hf_Dataset.from_dict(sample_fn(train_dataset)).with_format("torch")
    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    val_dataset = hf_Dataset.from_dict(sample_fn(val_dataset)).with_format("torch")
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def gsm8k(args):
    train_dataset = load_dataset('openai/gsm8k', 'main', split='train') #7.47k
    val_dataset = load_dataset('openai/gsm8k', 'main', split='test') #1.32k

    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    tokenizer.pad_token_id = tokenizer.eos_token_id
        
    def tokenize_function(data_point):
        prompt_text = "\nQuestion: {}\nSolution:".format(data_point["question"])
        prompt = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.int64)

        prompt_text = "\nQuestion: {}\nSolution: {}".format(data_point["question"], data_point["answer"])
        input = tokenizer.encode(prompt_text)
        input.append(tokenizer.eos_token_id)
        input = torch.tensor(input, dtype=torch.int64)
        labels = copy.deepcopy(input)
        labels[: len(prompt)] = -1
        input_mask = input.ge(0)
        label_mask = labels.ge(0)
        input[~input_mask] = 0
        labels[~label_mask] = -100

        return {
            "input_ids": input.tolist(),
            "labels": labels.tolist(),
            "attention_mask":input_mask.tolist(),
        }
    
    def preprocessing(data_point):
        for k, v in data_point.items():
            padding_length = args.max_length - len(v)
            if tokenizer.padding_side == "right":
                data_point[k] = torch.tensor(v + [tokenizer.pad_token_id] * padding_length, dtype=torch.int64)
            elif tokenizer.padding_side == "left":
                data_point[k] = torch.tensor([tokenizer.pad_token_id] * padding_length + v, dtype=torch.int64)
        return data_point
    
    import random
    random.seed(args.seed)

    def sample_fn(dataset):
        sampled_dataset = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
        }
        sampling_pool = list(range(len(dataset)))
        for i in range(args.n_samples):
            while True:
                random.shuffle(sampling_pool)
                index = sampling_pool.pop(0)
                tokenized_datapoint = tokenize_function(dataset[index])
                if len(tokenized_datapoint["input_ids"]) <= args.max_length and  tokenized_datapoint["input_ids"][-1] == tokenizer.eos_token_id:
                    for k,v in preprocessing(tokenized_datapoint).items():
                        sampled_dataset[k].append(v)
                    break
            if len(sampling_pool)<1:
                break
        return sampled_dataset
    train_dataset = hf_Dataset.from_dict(sample_fn(train_dataset)).with_format("torch")
    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    val_dataset = hf_Dataset.from_dict(sample_fn(val_dataset)).with_format("torch")
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

def wikitext2(args):
    traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    encoding_train_data = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    encoding_test_data = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    
    import random
    random.seed(args.seed)
    traininput = torch.zeros((args.n_samples, args.max_length), dtype=torch.int64)
    traintarget = torch.zeros((args.n_samples, args.max_length), dtype=torch.int64)
    for i in range(args.n_samples):
        start_loc = random.randint(0, encoding_train_data.input_ids.shape[1] - args.max_length - 1)
        end_loc = start_loc + args.max_length
        input = encoding_train_data.input_ids[:, start_loc:end_loc]
        target = input.clone()
        target[:, :-args.max_length] = -100
        traininput[i,:] = input.squeeze()
        traintarget[i,:] = target.squeeze()
    train_dataset = TensorDataset(traininput, traintarget)
    train_loader = DataLoader(train_dataset, batch_size = args.BATCH_SIZE,  shuffle=False)

    valinput = torch.zeros((args.n_samples, args.max_length), dtype=torch.int64)
    valtarget = torch.zeros((args.n_samples, args.max_length), dtype=torch.int64)
    for i in range(args.n_samples):
        start_loc = random.randint(0, encoding_test_data.input_ids.shape[1] - args.max_length - 1)
        end_loc = start_loc + args.max_length
        input = encoding_test_data.input_ids[:, start_loc:end_loc]
        target = input.clone()
        target[:, :-args.max_length] = -100
        valinput[i,:] = input.squeeze()
        valtarget[i,:] = target.squeeze()
    val_dataset = TensorDataset(valinput, valtarget)
    val_loader = DataLoader(val_dataset, batch_size = args.BATCH_SIZE,  shuffle=False)
    return train_loader, val_loader

def tiny_imagenet_c(args):
    import collections

    d = collections.OrderedDict()
    from methods.create_corruption import gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur
    from methods.create_corruption import zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform
    from methods.create_corruption import pixelate, jpeg_compression, speckle_noise, gaussian_blur, spatter, saturate
    d['GaussianNoise'] = gaussian_noise
    d['ShotNoise'] = shot_noise
    d['ImpulseNoise'] = impulse_noise
    d['DefocusBlur'] = defocus_blur
    d['GlassBlur'] = glass_blur
    d['MotionBlur'] = motion_blur
    d['ZoomBlur'] = zoom_blur
    d['Snow'] = snow
    d['Frost'] = frost
    d['Fog'] = fog
    d['Brightness'] = brightness
    d['Contrast'] = contrast
    d['Elastic'] = elastic_transform
    d['Pixelate'] = pixelate
    d['JPEG'] = jpeg_compression
    d['SpeckleNoise'] = speckle_noise
    d['GaussianBlur'] = gaussian_blur
    d['Spatter'] = spatter
    d['Saturate'] = saturate
    from torch.utils.data import DataLoader
    from datasets import load_dataset    
    val_loader_list = []
    #DATA_PATH = "/imagenet/ILSVRC/Tiny-ImageNet-C"
    #transform = transforms.Compose([
    #    transforms.Resize((224,224), interpolation=InterpolationMode.BILINEAR, antialias=True),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=mean, std=std)
    #    ])
    method = d[args.noise]
    for i in range(5):
        val_data = load_dataset('zh-plus/tiny-imagenet', split = 'valid')
        val_data = imagenet_C_Dataset(dataset = val_data, method = method, severity = i+1)        
        val_loader_list.append(DataLoader(val_data, batch_size = args.BATCH_SIZE,  shuffle=False))
    return val_loader_list

"""
A reminder of our instruction prompt:
alpaca_prompt_template = {
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n
    ### Instruction:\n{instruction}\n\n
    ### Input:\n{input}\n\n
    ### Response:",
    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n
    ### Instruction:\n{instruction}\n\n
    ### Response:",
}

"""
ai2arc_answer_mapping = {
    "1": "answer1",
    "A": "answer1",
    "2": "answer2",
    "B": "answer2",
    "3": "answer3",
    "C": "answer3",
    "4": "answer4",
    "D": "answer4",
    "5": "answer5",
    "E": "answer5",
}
ai2arc_index_mapping = {
    "1": "1",
    "A": "1",
    "2": "2",
    "B": "2",
    "3": "3",
    "C": "3",
    "4": "4",
    "D": "4",
    "5": "5",
    "E": "5",
}
def zero_shot_dataset(task, training = False):
    if task == 'boolq':
        if training:
            #"train"  9427 rows
            dataset = load_dataset("google/boolq", split='train')
        else:
            #"validation"  3270 rows
            dataset = load_dataset("google/boolq", split='validation')
        def generate_zero_shot_prompt(data_point):
            instruction = "Please answer the following yes/no question by selecting \"1. no\" or \"2. yes\".\n" 
            instruction = instruction + "{}\n{}\n1. No, it is false.\n2. Yes, it is true.".format(data_point["passage"], data_point["question"])
            label = ["1", "2"][data_point["answer"]] # false/true
            return {
                "prompt": alpaca_prompt_template["prompt_no_input"].format(instruction=instruction),
                "correct_answer": label, #false/true
            }
    elif task == 'piqa':
        if training:
            #"train" about 16113 rows
            dataset = load_dataset("ybisk/piqa", split='train', trust_remote_code = True)
        else:
            #"validation" 1838 rows
            dataset = load_dataset("ybisk/piqa", split='validation', trust_remote_code = True)
        def generate_zero_shot_prompt(data_point):
            instruction = "Please select the correct or most plausible choice for the following question.\n" 
            instruction = instruction + "{}\n1. {}\n2. {}".format(data_point["goal"], data_point["sol1"], data_point["sol2"])
            label = ["1", "2"][data_point["label"]] # 0 refers to solution1 and 1 refers to solution2
            return {
                "prompt": alpaca_prompt_template["prompt_no_input"].format(instruction=instruction),
                "correct_answer": label,
            }
    elif task == 'social_i_qa':
        if training:
            #"train" 33410 rows
            dataset = load_dataset("allenai/social_i_qa", split='train', trust_remote_code = True)
        else:
            #"validation" 1954 rows
            dataset = load_dataset("allenai/social_i_qa", split='validation', trust_remote_code = True)
        def generate_zero_shot_prompt(data_point):
            instruction = "Please select the correct or most plausible choice for the following question.\n" 
            instruction = instruction + "{}\n{}".format(data_point["context"], data_point["question"])
            for i, choice in enumerate([data_point["answerA"], data_point["answerB"], data_point["answerC"]]):
                instruction = instruction + "\n{}. ".format(i+1) + choice 
            label = data_point["label"] # 1 -> answer1, 2 -> answer2, 3 -> answer3
            return {
                "prompt": alpaca_prompt_template["prompt_no_input"].format(instruction=instruction),
                "correct_answer": label,
            }
    elif task == 'openbookqa':
        if training:
            #"train" 4957 rows 
            dataset = load_dataset("allenai/openbookqa", "main", split='train')
        else:
            #"validation" 500 rows "test" 500 rows
            dataset = load_dataset("allenai/openbookqa", "main", split='test')
        def generate_zero_shot_prompt(data_point):
            instruction = "Please select the correct or most plausible choice for the following question.\n" 
            instruction = instruction + data_point["question_stem"]
            for i, choice in enumerate(data_point["choices"]["text"]):
                instruction = instruction + "\n{}. ".format(i+1) + choice 
            label = ai2arc_index_mapping[data_point["answerKey"]] # 1,A -> answer1, 2,B -> answer2, ...
            return {
                "prompt": alpaca_prompt_template["prompt_no_input"].format(instruction=instruction),
                "correct_answer": label,
            }
    elif task == 'ARC-Challenge':
        if training:
            #"train" 1119 rows 
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split='train')
        else:   
            #"validation" 299 rows "test" 1172 rows
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split='test')
        def generate_zero_shot_prompt(data_point):
            instruction = "Please select the correct or most plausible choice for the following question.\n" 
            instruction = instruction + data_point["question"]
            for i, choice in enumerate(data_point["choices"]["text"]):
                instruction = instruction + "\n{}. ".format(i+1) + choice 
            label = ai2arc_index_mapping[data_point["answerKey"]] # 1,A -> answer1, 2,B -> answer2, ...
            return {
                "prompt": alpaca_prompt_template["prompt_no_input"].format(instruction=instruction),
                "correct_answer": label,
            }
    elif task == 'ARC-Easy':
        if training:
            #"train" 2251 rows 
            dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split='train')
        else:
            #"validation" 570 rows "test" 2376 
            dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split='test')
        def generate_zero_shot_prompt(data_point):
            instruction = "Please select the correct or most plausible choice for the following question.\n" 
            instruction = instruction + data_point["question"]
            for i, choice in enumerate(data_point["choices"]["text"]):
                instruction = instruction + "\n{}. ".format(i+1) + choice 
            label = ai2arc_index_mapping[data_point["answerKey"]] # 1,A -> answer1, 2,B -> answer2, ...
            return {
                "prompt": alpaca_prompt_template["prompt_no_input"].format(instruction=instruction),
                "correct_answer": label,
            }
    elif task == 'hellaswag':
        if training:
            #"train" 39905 rows
            dataset = load_dataset("Rowan/hellaswag", split='train', trust_remote_code = True)
        else:
            #"validation" 10042 rows
            dataset = load_dataset("Rowan/hellaswag", split='validation', trust_remote_code = True)
        def generate_zero_shot_prompt(data_point):
            instruction = "Please select the correct or most plausible choice for the following question.\n" 
            instruction = instruction + "{}: {} {}".format(data_point["activity_label"], data_point["ctx_a"], data_point["ctx_b"].capitalize())
            for i, choice in enumerate(data_point["endings"]):
                instruction = instruction + "\n{}. ".format(i+1) + choice 
            label = str(int(data_point["label"])+1) #0 -> ending1, 1 -> ending2, 2 -> ending3, 3 -> ending4
            return {
                "prompt": alpaca_prompt_template["prompt_no_input"].format(instruction=instruction),
                "correct_answer": label,
            }
    elif task == 'winogrande':
        if training:
            #"train" 40398 
            dataset = load_dataset("allenai/winogrande", "winogrande_xl", split='train', trust_remote_code = True)
        else:
            #"validation" 1267  "test" 1767
            dataset = load_dataset("allenai/winogrande", "winogrande_xl", split='validation', trust_remote_code = True)
        def generate_zero_shot_prompt(data_point):
            instruction = "Please select the most appropriate option to complete the following sentense.\n" 
            instruction = instruction + "{}\n1. {}\n2. {}".format(data_point["sentence"], data_point["option1"], data_point["option2"])
            label = data_point["answer"] #"1" -> option1, "2" -> option2
            return {
                "prompt": alpaca_prompt_template["prompt_no_input"].format(instruction=instruction),
                "correct_answer": label,
            }
    
    processed_datasets = dataset.map(generate_zero_shot_prompt, remove_columns=dataset.column_names)
    return processed_datasets

def extract_answer(task, sentence: str) -> str:
    sentence_ = sentence.strip().lower()
    if task == 'boolq':
        pred_answers = re.findall(r'yes|no|true|false', sentence_)
        if not pred_answers:
            if sentence_[0] in ['1', '2']:
                return sentence_[0]
            else:
                return ""
        return {'yes':'2', 'no':'1', 'true':'2', 'false':'1'}[pred_answers[0]]
    elif task in ['piqa', 'social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa', 'hellaswag', 'winogrande']:
        if len(sentence_)>0:
            if sentence_[0] in ai2arc_index_mapping.keys():
                return ai2arc_index_mapping[sentence_[0]]
        return ""

def zero_shot_eval(model, 
                   tokenizer, 
                   device,
                   temperature=0.1,
                   top_p=0.75,
                   top_k=40,
                   num_beams=4,
                   max_new_tokens=64,
                   n_samples=500,
                   **kwargs,
                   ):
    tasks = ["boolq","piqa","social_i_qa","hellaswag","winogrande","ARC-Challenge","ARC-Easy","openbookqa"]
    results = {}
    for task in tasks:
        datasets = zero_shot_dataset(task).shuffle(seed = 1234)
        total = min(len(datasets), n_samples)
        correct = 0
        print("Starting evaluation on task {}.\n".format(task))
        for i, item in enumerate(datasets):
            if i >= n_samples: 
                break
            prompt = item["prompt"]
            label = item["correct_answer"]
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                **kwargs,
                )
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                    generation_config=generation_config,
                    #num_return_sequences=1, 
                    #do_sample=False,
                    )
            output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
            
            print(f"Model generation:\n{output}")
            output = output.split("### Response:")[-1].strip()
            if len(output)>0:
                prediction = extract_answer(task, output)
            else:
                prediction = ""
            if prediction == label:
                correct += 1
            print('prediction:', prediction)
            print('label:', label)
        acc = correct/total
        print("********** Accuracy on {}: {:.2f}%. **********\n".format(task, acc*100))
        results[task] = acc
    return results


def zeroshot_llama(args):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, token = auth_token, use_fast=False, cache_dir = hf_cache)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    def tokenize_function(prompt_ctx, input_ctx):
        prompt = torch.tensor(tokenizer.encode(prompt_ctx), dtype=torch.int64)
        input = tokenizer.encode(input_ctx)
        input.append(tokenizer.eos_token_id)
        input = torch.tensor(input, dtype=torch.int64)
        labels = copy.deepcopy(input)
        labels[: len(prompt)] = -1
        input_mask = input.ge(0)
        label_mask = labels.ge(0)
        input[~input_mask] = 0
        labels[~label_mask] = -100

        return {
            "input_ids": input.tolist(),
            "labels": labels.tolist(),
            "attention_mask":input_mask.tolist(),
        }
    
    def preprocessing(data_point):
        for k, v in data_point.items():
            padding_length = args.max_length - len(v)
            if tokenizer.padding_side == "right":
                data_point[k] = torch.tensor(v + [tokenizer.pad_token_id] * padding_length, dtype=torch.int64)
            elif tokenizer.padding_side == "left":
                data_point[k] = torch.tensor([tokenizer.pad_token_id] * padding_length + v, dtype=torch.int64)
        return data_point
    
    import random
    random.seed(args.seed)

    def sample_fn(zero_shot_data, training):
        sampled_dataset = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
        }
        
        for task in  ["boolq","piqa","social_i_qa","hellaswag","winogrande","ARC-Challenge","ARC-Easy","openbookqa"]:
            processed_datasets = zero_shot_data(task, training)
            sampling_pool = list(range(len(processed_datasets)))
            for i in range(int(args.n_samples/8)):
                while True:
                    random.shuffle(sampling_pool)
                    index = sampling_pool.pop(0)
                    datapoint = processed_datasets[index]
                    prompt_ctx = datapoint["prompt"]
                    input_ctx = prompt_ctx + datapoint["correct_answer"]
                    tokenized_datapoint = tokenize_function(prompt_ctx, input_ctx)
                    if len(tokenized_datapoint["input_ids"]) <= args.max_length and tokenized_datapoint["input_ids"][-1] == tokenizer.eos_token_id:
                        for k,v in preprocessing(tokenized_datapoint).items():
                            sampled_dataset[k].append(v)
                        break
                if len(sampling_pool)<1:
                    break
        return sampled_dataset
    
    train_dataset = hf_Dataset.from_dict(sample_fn(zero_shot_dataset, True)).with_format("torch")
    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    val_dataset = hf_Dataset.from_dict(sample_fn(zero_shot_dataset, False)).with_format("torch")
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader