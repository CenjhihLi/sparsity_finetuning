# sparsity_finetuning

This is the official implementation of ["An Efficient Sparse Fine-Tuning with Low Quantization Error via Neural Network Pruning"](https://openreview.net/forum?id=w3b67v5EzD).

---

## Acknowledgement

Our pruning-based fine-tuning method leverages the [`Torch-Pruning`](https://github.com/VainF/Torch-Pruning) library. Many
parts of the pruning logic (e.g., dependency graph construction and layer pruning
rules) were adapted from or inspired by the original [`Torch-Pruning`](https://github.com/VainF/Torch-Pruning) codebase.
Their framework greatly simplified the development of our method. We appreciate their high-quality and well-structured codebase.

---

## Environment building

Most dependencies are included in `requirements.txt`.

- For **RoSA** and its **spops**, please refer to: [https://github.com/IST-DASLab/peft-rosa](https://github.com/IST-DASLab/peft-rosa)  
- For **Language Model Evaluation Harness**, please refer to: [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)  
- For **FastChat**, please refer to: [https://github.com/lm-sys/FastChat](https://github.com/lm-sys/FastChat).  
Note that we added Gemini 2.5 as a judge model since it is cheaper than GPT-4o. 


## Reproducibility

- Training datasets are loaded from HuggingFace Hub.  
- A subset of the dataset is sampled for training. Due to ongoing updates of the HuggingFace loading pipeline and potential changes in dataset metadata, exact data sampling may vary slightly even with the same random seed.  
- Exact reproduction of results is **not guaranteed**, but overall trends and main results should remain consistent.  
- Although we use a relatively small number of training examples for most tasks due to computational constraints on our cluster, especially for zeroshot datasets, we strongly recommend increasing the sample size whenever possible to improve robustness.

---

## Usage

### Fine-tuning with LoRA
```
python finetune_text_generation_model_peft.py \  
  --EPOCHS 3 \  
  --learning_rate 1e-4 \  
  --model meta-llama/Meta-Llama-3-8B \  
  --dataset alpacagpt4 \  
  --method lora \  
  --finetune_attn_as_linear \  
  --finetuning_rank 32 \  
  --max_length 512 \  
  --n_samples 1024
```

### Fine-tuning with our method
```
python finetune_text_generation_model_ours.py \  
  --EPOCHS 3 \  
  --learning_rate 1e-4 \  
  --model meta-llama/Meta-Llama-3-8B \  
  --dataset alpacagpt4 \  
  --finetune_attn_as_linear \  
  --finetuning_rank 64 \  
  --finetuning_type ZOtaylor \  
  --n_estimate 2 \  
  --p_dropout 0.0 \  
  --param_dtype fp32 \  
  --max_length 512 \  
  --n_samples 1024  
```

### Fine-tuning Quantized Llama-3-8B with our method and PEFTs
```
python finetune_LLM_quantization.py \
  --EPOCHS 3 \
  --model meta-llama/Meta-Llama-3-8B \
  --dataset gsm8k \
  --learning_rate 1e-4 \
  --method ours_loftq \
  --finetune_attn_as_linear \
  --finetuning_rank 64 \
  --max_length 512 \
  --n_samples 2048 
```

The flag --finetune_attn_as_linear makes the script fine-tune all linear layers of the model.
Without this flag, the query, key, and value projections will not be fine-tuned.
See the description in Section 6.1 of the paper.  

In ours_loftq, our method initializes the update matrix $\Delta\mathbf{W}$ with the corresponding rows extracted from the quantization residual $W^{res}$.

### Evaluate the finetuned models
```
python finetuned_LLM_eval.py \
  --model meta-llama/Meta-Llama-3-8B \
  --task zeroshot \
  --model_checkpoint {your_path_of_checkpoints.pth} 

```

### Generating MT Bench answers
```
python generate_mtbench_answers.py \  
  --model meta-llama/Meta-Llama-3-8B \  
  --answer_path ./mtbench/Meta-Llama-3-8B_attn_as_linear_lora_rank64.jsonl \  
  --model_checkpoint {your_path_of_checkpoints.pth}  
```

Then evaluate the results following the instructions in FastChat.fastchat.llm_judge.