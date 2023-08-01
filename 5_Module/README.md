# Parameter Efficient Fine Tuning methods and developing Gradio Applications

[Parameter-Efficient Fine-Tuning (PEFT) Presentation](https://docs.google.com/presentation/d/1fY5w1_3lPu7CttZOjNG8InuN6-n-H3COWKuSB9uuhZs/edit?usp=sharing)


## PEFT Hands-on

### Get your own Code Assistant to help with coding questions

1. Go to [Code Assistant Dataset generation notebook](../code_assistant/dataset_generation/) and see how the conversational dataset is generated to PEFT tune the StarCoder to be able to converse.
2. Go to [train.py](../code_assistant/training/train.py) to see how the training code is leveraging PEFT, SFTTrainer from `trl` library and how easy it is to tune **16B** parameter  Starcoder-16B on a single A6000 48GB GPU.
3. Run the training with [run_peft.sh](../code_assistant/training/run_peft.sh) 

### Get your own personal co-pilot trained on your internal codebase

1. Go to [train.py](../personal_copilot/training/train.py) for the training code using 🤗 Accelerate, 🤗 Transformers Trainer and 🤗 PEFT.
2. Go to [run_peft.sh](../personal_copilot/training/run_peft.sh) to run using PEFT.
3. Infer using the notebook [PEFT Personal Code Co-Pilot.ipynb](../5_Module/PEFT_Personal_Code_CoPilot.ipynb).

## Gradio Application

1. Go to [chat_gradio_app](../5_Module/chat_gradio_app)
2. install requirements
```
pip install -r requirements.txt
```
3. Go through the app code at [app.py](../5_Module/chat_gradio_app/app.py)
4. Run the app
```
python app.py
```
5. Wait for 10-15 minutes for the app to load the big models and merge the peft models into it. Play around with the model! 🤗

https://github.com/pacman100/DHS-LLM-Workshop/assets/13534540/673fb4a7-37b4-4581-a90c-ef157ee3441b

## AutoTrain Advanced - No Code solution making it the EASIEST!

Check out [The EASIEST way to finetune LLAMA-v2 on local machine!](https://youtu.be/3fsn19OI_C8) to know how to use Autotrain Advanced for PEFT training wihtout having to write any code.

Codebase: [autotrain-advanced](https://github.com/huggingface/autotrain-advanced)

All that you need to run is:
```
pip install autotrain-advanced

autotrain llm --train --project_name my-llm --model meta-llama/Llama-2-7b-hf --data_path . --use_peft --use_int4 --learning_rate 2e-4 --train_batch_size 12 --num_train_epochs 3 --trainer sft
``` 
