## Comparison

Below plot shows the eval loss, train loss and learning rate scheduler for QLoRA vs full fine-tuning. We observe that full fine-tuning leads to slightly lower loss and converges a bit faster compared to QLoRA. The learning rate for peft fine-tuning is 10X more than that of full fine-tuning.

![plots](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/full_finetuning_vs_qlora.png)

To make sure that our PEFT model doesn't lead to catastrophic forgetting, we run the Python Human Eval on it. Below are the results. We can observe that the performance on humaneval-python is comparable for both the base `StarCoder-15B` and the PEFT model `smangrul/peft-lora-starcoder15B-v2-personal-copilot-A100-40GB-colab`.

| | |
|---|---|
| Model | Pass@1 |
|StarCoder-15B | 33.57|
|smangrul/peft-lora-starcoder15B-v2-personal-copilot-A100-40GB-colab| 33.37 |

As we don't have a benchmark, we will look at some qualitative samples. Inference Code for full fine-tuned model and peft model are available at [full_finetuned_inference.py](https://github.com/pacman100/DHS-LLM-Workshop/blob/main/personal_copilot/inference/full_finetuned_inference.py) and [peft_inference.py](https://github.com/pacman100/DHS-LLM-Workshop/blob/main/personal_copilot/inference/peft_inference.py), respectively. In our manual analysis, we noticed that the QLoRA led to slight overfitting and as such we down weigh it by creating new weighted adapter with weight 0.8 via `add_weighted_adapter` utility of PEFT.


Below screenshots of a table show the predictions by Fully fine-tuned model in comparison with the PEFT QloRA model:
![qualitative_comparison_1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/qualitative_comparison_1.png)
![qualitative_comparison_2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/qualitative_comparison_2.png)

We can observe that the generations from both the variants are as per expectations. Awesome! 🚀

## Finetuning your own Code Chat Assistant

So far, the models we trained were specifically trained as personal co-pilot for code completion tasks. They aren't trained to carry out conversations or for question answering. `Octocoder` and `StarChat` are great examples of such models. This section briefly describes how to achieve that.

Resources: 

1. Codebase: [link](https://github.com/pacman100/DHS-LLM-Workshop/tree/main/code_assistant/training). It uses the recently added Flash Attention V2 support in Transformers. 
2. Colab notebook : [link](https://colab.research.google.com/drive/1XFyePK-3IoyX81RM94JO73CcIZtAU4i4?usp=sharing). Make sure to choose A100 GPU with High RAM setting.
3. Model: [bigcode/stacoderplus](https://huggingface.co/bigcode/starcoderplus)
4. Dataset: [smangrul/code-chat-assistant-v1](https://huggingface.co/datasets/smangrul/code-chat-assistant-v1). Mix of `LIMA+GUANACO` with proper formatting in a ready-to-train format.
5. Trained Model: [smangrul/peft-lora-starcoderplus-chat-asst-A100-40GB-colab](https://huggingface.co/smangrul/peft-lora-starcoderplus-chat-asst-A100-40GB-colab) 

## Dance of LoRAs

If you have dabbled with Stable Diffusion models and LoRAs for making your own Dreambooth models, you might be familiar with the concepts of combining different LoRAs with different weights, using a LoRA model with a different base model than the one on which it was trained. In text/code domain, this remains unexplored territory. We carry out experiments in this regard and have observed very promising findings. Are you ready? Let's go! 🚀

### Mix-and-Match LoRAs

PEFT currently supports 3 ways of combining LoRA models, `linear`, `svd` and `cat`. For more details, refer: [tuners#peft.LoraModel.add_weighted_adapter](https://huggingface.co/docs/peft/main/en/package_reference/tuners#peft.LoraModel.add_weighted_adapter).

Please refer the notebook [Dance of LoRAs.ipynb]() for all the inference code and various combinations. Below, we will explore only a few important scenarios.

In the notebook, notice that we are loading the chat assistant on top of `starcoder` instead of `starcodeplus` on which it was fine-tuned. 

Here, we will consider 2 abilities, i.e., `chatting/QA` and `code-completion` on 2 data distributions, i.e., `top 10 public hf codebase` and `generic codebase`. That gives us 4 axes on which to evaluate things. We will be carrying out qualitative analysis. 

#### First, let us consider `chatting/QA` task. 

Let's disable adapters and see the outputs on a `generic` and `hf code` questions, specifically.

![disabled_chat_generic](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/disabled_chat.png)

We can observe that it fails for both cases as the base model `starcoder` is only meant for code completion and is unsuitable for `chatting/question-answering`.

Now, let's enable the `assistant` adapter.

![assistant_chat_generic](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/assistant_chat_generic.png)
![assistant_chat_hf](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/assistant_chat_hf.png)

We can observe that generic question regarding scrapy is being answered properly. However, it is failing for the HF code related question which wasn't part of its pretraining data.

Finally, let's enable the `copilot` adapter.

![copilot_chat_generic](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/copilot_chat_generic.png)
![copilot_chat_hf](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/copilot_chat_hf.png)

We can observe that it performs similar to disabled case because this LoRA was also specifically fine-tuned for code-completion.

##### Let us now consider `code-completion` task.

Let's disable adapters and see the outputs on a `generic` and `hf code` code blocks, specifically.

![disabled_code_generic](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/disabled_code_generic.png)
![disabled_code_hf](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/disabled_code_hf.png)

Observe that the code completion for the generic two-sum is as expected. However, the HF code completion fails with wrong params to `LoraConfig` as the base model hasn't seen it in its pretraining data.

Time for us to check the `assistant` adapter for code-completion task.

![assistant_code_generic](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/assistant_code_generic.png)
![assistant_code_hf](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/assistant_code_hf.png)

We can observe that the `assistant` performs similar to disabled case as it was trained on natural language conversations which didn't have any HF code repos. 

Finally, let's enable the `copilot` adapter.

![copilot_code_generic](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/copilot_code_generic.png)
![copilot_code_hf](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/copilot_code_hf.png)

We can observe that the `copilot` adapter gets it right in both case. Therefore, it performs as expected for code-completions when working with HF specific codebase as well as generic codebases.

**Now, as a user, I want to combine the ability of `assistant` as well as `copilot`. This will enable me to use it for code completion while coding in IDE and then have it as a chatbot to answer my questions regarding APIs, classes, methods, documentation. It should be able to provide answers to questions like `How do I use x`, `Please write a code snippet for Y` on my codebase.**

PEFT allows you do it by via `add_weighted_adapter`. Let's create a new adapter `code_buddy` with equal weights to `assistant` and `copilot` adapters.

![combining_loras](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/combining_loras.png)

Now, let's see how `code_buddy` performs on the `chatting/question_answering` tasks.

![mix_chat_generic](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/mix_chat_generic.png)
![mix_chat_hf](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/mix_chat_hf.png)

We can observe that `code_buddy` is performing much better than `assistant` and `copilot` adapters. It is able to answer the generic question of computing quantiles as well as write a code snippet to show how to use a specific HF repo API. However, it is also hallucinating the wrong links to guide which remains a caveat for thes LLMs.

Below is the performance of `code_buddy` on code completions task.

![mix_code_generic](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/mix_code_generic.png)
![mix_code_hf](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/mix_code_hf.png)

We can observe that `code_buddy` is performing on par with `copilot` which was specifically finetuned for this task.


## Transfer LoRAs to different base models

We can also transfer the LoRA models to different base models.
We will take the fresh off the press `Octocoder` model and apply on it the LoRA we trained above with `starcoder` base model. Please go through the following notebook []() for the entire code.

**Performance on the Code Completion task**

![octocoder_code_generic](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/octocoder_code_generic.png)
![octocoder_code_hf](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/octocoder_code_hf.png)

We can observe that `octocoder` is performing great. It is able to complete generic as well as HF specific code snippets.

**Performance on the Chatting/QA task**

As Octocoder is trained to answer questions and carry out conversations about coding, let's see if it can use our LoRA adapter to answer HF specific questions.

First, let's see the output with adapter disabled to make sure it isn't part of the training data of Octocoder:

![octocoder_disabled_chat_hf](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/octocoder_disabled_chat_hf.png)

We can see that it fails to correctly use the API of LoraConfig or to create a PEFT model. Now, let's see it performance with the adapter enabled.

![octocoder_chat_hf](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/octocoder_chat_hf.png)
![octocoder_chat_generic](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/personal_copilot/octocoder_chat_generic.png)

Yay! It correctly answers in detail how to create LoraConfig and related peft model along with correctly using the model name, dataset name as well as param values of LoraConfig. Also note that it does a great job at answering the generic query of using 
scrapy for crawling.