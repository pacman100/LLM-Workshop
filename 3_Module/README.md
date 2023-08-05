# Training and Evaluating LLMs and their Best Practices

## Scaling Laws and Cost of Pre-Training LLMs

Research Paper: [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)

* Aprrox parameters in Decoder LLMs: 12 \* n_layers \* d_model^2
* All parameters participate roughly in 1 add and 1 multiply per token in forward pass and 2X that in backward pass. So the training compute is:
* Approx Comput6 required = 6 \* P (num_params) \* D (num_tokens) [6 = 2 ops [sum, multiply] \* (1 [forward] + 2 [backward])]
* Context length doesn't matter much: context_compute/total_compute ~ n_ctx / (12 \* d_model) where d_model >> n_ctx.

Let's put some numbers with Llama V2 70B model example:

1. If we have a mobel with 70B params trained on 2T tokens, the approx compute required is: 6 \* 7e10 * 2e12  = **8.4e23 FLOPs**. 
2. A100s GPUs perform ~ 2.8e13 FLOPs (in float32 or bfloat16). Hoever, the most performant implementation manages to have 50% of peak performance due to GPU memory limits and distributed communication constraints, i.e., reasonable Model Flops Utilization (MFU) being **~1.4e13 FLOPs**.
3. Number of hours required = 8.4e23/(1.5e14\*3600) = 8.4e23/5.4e17 = **1.55e6 A100 hours**
4. **Note:** Above estimate doesn't consider hyperparamter runs, failure of nodes and restarts. There will be hyperparameter runs on smaller scales and numerous failures and restarts from checkpoints. Paper reports **1.72e6 A100 GPU hours**. This is ~10% more than our estimated GPU hours.
4. Approx cost of training = 1.72e6 \* ($1.8/hr) ~ **$3M** 

![Scaling Laws](../assets/ScalingLaw.png)
[Source [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)]

Precise Scaling laws for the performance of LLMs as a function of nodel paramneters count, datset size and training compute. Performance is mostly about overcoming bottlenecks such as not enough data, parameters, compute.


**How to choose the model size and dataset size given a compute budget?**

![chincilla](../assets/chinchilla.png)
[Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf) [Chincilla Paper]

Chincilla paper's main finding is that the current large language models are far too large for their compute budget and are not being trained on enough data. They outline that 70B model (4X smaller compared to 280B Gopher) trained on 4X data (1.3T tokens compared to 300B tokens used by Gopher) outperform 280B Gopher model on every task they evaluated.

However, recent trends shows that model performance goes on improving with dataset size. For example, Llam V2 &B model was trained on 1T tokens.

## Should You Purchase an LLM or Train Your Own?

Let's go over a nice blog by Weights and Biases on this: (Should You Purchase an LLM or Train Your Own?)[https://wandb.ai/wandb_fc/LLM%20Best%20Practices/reports/Should-You-Purchase-an-LLM-or-Train-Your-Own---VmlldzozNjU5NjYy]

### Few more Thoughts:
1. If you are not working with sensitive/PII data such as healthcare/financial recods, start with commercial APIs to explore LLM capabilities and get to market faster as it doesn't require lots of data. Once you have product in production and get considerable amount of amount, start thinking on improving the product by continued pretraining/fine-tuning of available Open-Source LLMs such as Llama, Falcon, MPT, Starcoder, CodeGen ...
2. In most cases, there is never a need to pre-train from stratch given the quality of Open-Source models today. It only makes sense if you are trying to make innovations in modeling and data such as changing model architecture, tokenizer, niche data such as financial records (BloombergGPT) ...

## How to pre-train/fine-tune LLMs?

Let's look at approaches required for pre-training or fine-tuning LLMs. The difference between them is that fine-tuning usually is on a downstream task with limited dataset. In all other aspects such as training and evaluating, they are mostly similar. 

Challenges:
1. Models are so large that they can't fit on a single GPU. How do we even enable training of such large models?
2. How to reduce the communication overhead in a distributed cluster setup?
3. How to increase the throughput of the training without hurting the performance?
4. How to save/load such large checkpoints?
5. How do we reduce the overheads in reading and processing large datasets?

Let's go over 2 great blogs explaining this in detail:

1. [Efficient Training on a Single GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one)
2. [Efficient Training on Multiple GPUs](https://huggingface.co/docs/transformers/perf_train_gpu_many)

## Evaluating LLMs

Let's go over this insightful section on evaluating LLMs by Eugene Yan:
[Evals: To measure performance](https://eugeneyan.com/writing/llm-patterns/#evals-to-measure-performance)

## Additional Resources
[Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/)