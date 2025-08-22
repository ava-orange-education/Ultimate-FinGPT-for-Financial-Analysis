
# !pip install transformers --upgrade
# !pip install accelerate
# !pip install -U bitsandbytes
# !pip install loguru
# !pip install --upgrade peft
# # !pip install transformers==4.40.1
# !pip install  datasets 
# !pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


import pandas as pd

import json
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer
import datasets
from datasets import load_from_disk, load_dataset, Dataset
import torch
import torch.nn.functional as F

json_data_list = []  # Var to save json data

# Save to a jsonl file and store in json_data_list
with open("Data/dataset_new.jsonl", 'r') as f:
    for line in f:
        json_line = json.loads(line.strip())
        json_data_list.append(json_line)

#check
print(f"testing json data list")
print(json_data_list[0]['target'], json_data_list[0]['context'])

#Tokenization is the process of converting input text into tokens that can be fed into the model.
#Specifies the model  working with. #do not forget to add it to your finegrained access on HuggingFace 
model_name = 'meta-llama/Meta-Llama-3-8B'   
jsonl_path = 'Data/dataset_new.jsonl'
#The path where the processed dataset will be saved after tokenization
save_path = 'Data/dataset_new' 
#Maximum sequence length for the inputs. If an input exceeds this length, it will either be truncated or skipped.
max_seq_length = 128 # please change if you ahve longer sentences   
#A flag that determines whether to skip overlength examples that exceed max_seq_length
skip_overlength = True    #


#This preprocess function tokenizes the promt and target, 
# combines them into Input ids, trims or pads the squence to the maximum squence length.
def preprocess(tokenizer, config, example, max_seq_length):
  prompt = example['context']
  target = example['target']
  prompt_ids = tokenizer.encode(   #ids refers to the numerical identifiers that correspond to tokens.
      prompt,
      max_length = max_seq_length,
      truncation = True
      )
  target_ids = tokenizer.encode(
      target,
      max_length = max_seq_length,
      truncation = True,
      add_special_tokens = False
      )
  input_ids = prompt_ids + target_ids + [config.eos_token_id]  #[config.eos_token_id] is a sign that marks the end of the list.
  return {'input_ids':input_ids,'seq_len':len(prompt_ids)}

# now define tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, device_map='auto')


# check
example = json_data_list[0]
prompt = example['context']
target = example['target']

example['target']


#following fuction Uses yield to return one preprocessed feature at a time, making the function a generator.
#This allows you to iterate over the processed features one by one without loading everything into memory at once.

def read_jsonl(path, max_seq_length, skip_overlength=False):
    #Initializes a tokenizer using a pre-trained model specified by model_name.
    tokenizer = AutoTokenizer.from_pretrained(    
        model_name, trust_remote_code=True)
    #Loads the configuration for the model. 
    # device_map='auto' helps automatically map the model to available devices (e.g., GPU or CPU).
    config = AutoConfig.from_pretrained(    
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            #Preprocesses each example by tokenizing it and converting it into input_ids using the preprocess() function,
            #which takes the tokenizer, config, example, and max_seq_length as inputs.
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]  #Truncates the input_ids to ensure they do not exceed max_seq_length.
            yield feature


save_path = 'Data/dataset_new1'

dataset = datasets.Dataset.from_generator(
    lambda: read_jsonl(jsonl_path, max_seq_length, skip_overlength)
    )
dataset.save_to_disk(save_path)



# Load Dataset
loaded_dataset = load_from_disk('Data/dataset_new1')

# Check the structure of Dataset
print(loaded_dataset)

# Print the first sample of the dataset
print(loaded_dataset['input_ids'][0])


from typing import List, Dict, Optional
import torch
from loguru import logger
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers import LlamaForCausalLM


training_args = TrainingArguments(
    output_dir='Model/',    # Path to save the fine-tuned model
    logging_steps = 500,               # Log every 500 steps
    # max_steps=10000,                 # Maximum number of training steps (commented out, can be enabled)
    num_train_epochs = 2,              # Number of training epochs (train for 2 epochs)
    per_device_train_batch_size=4,     # Batch size of 4 for training on each device (GPU/CPU)
    gradient_accumulation_steps=8,     # Accumulate gradients for 8 steps before updating weights
    learning_rate=1e-4,                # Learning rate set to 1e-4
    weight_decay=0.01,                 # Weight decay (L2 regularization) set to 0.01
    warmup_steps=1000,                 # Warm up the learning rate for the first 1000 steps
    fp16=True,                         # Enable FP16 mixed precision training to save memory and speed up training
    # bf16=True,                       # Enable BF16 mixed precision training (commented out)
    torch_compile = False,             # Whether to enable Torch compile (`False` means not enabled)
    load_best_model_at_end = True,     # Load the best-performing model at the end of training
    # evaluation_strategy="steps",       # Evaluation strategy is set to evaluate every few steps
    eval_strategy="steps",
    save_steps=500,  # # Save the model every 500 steps
    metric_for_best_model="loss",
    remove_unused_columns=False,       # Whether to remove unused columns during training (keep all columns)
    logging_dir="./logs",
)


# quantitative allocation
q_config = BitsAndBytesConfig(load_in_4bit=False,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.float16
                                )

# get access to huggingFace

import os
import shutil

# Retrieve the token from Colab Secrets
hf_token = 'YOUR HF TOKEN'
os.environ["HF_TOKEN"] = hf_token

from huggingface_hub import login
login(token=hf_token)

# to avoid runtime error
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# check before proceeding
from transformers.utils import is_bitsandbytes_available
print(is_bitsandbytes_available() ) # should be True, if not re install



model = LlamaForCausalLM.from_pretrained(
        model_name,
        quantization_config = q_config,
        trust_remote_code=True,
        device_map='auto'
    )



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# LoRA for Llama3
target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['llama']  # Modules for the Llama model
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_modules,
    bias='none',
)

# Loading LoRA for Llama3 models using PEFT (Parameter-Efficient Fine-Tuning)
model = get_peft_model(model, lora_config)

# Print the number of trainable parameters
print_trainable_parameters(model)


resume_from_checkpoint = None
if resume_from_checkpoint is not None:
    checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(
            resume_from_checkpoint, 'adapter_model.bin'
        )
        resume_from_checkpoint = False
    if os.path.exists(checkpoint_name):
        logger.info(f'Restarting from {checkpoint_name}')
        adapters_weights = torch.load(checkpoint_name)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        logger.info(f'Checkpoint {checkpoint_name} not found')


model.print_trainable_parameters()


dataset = load_from_disk(save_path)
# set seed to standardise the output
dataset = dataset.train_test_split(0.2, shuffle=True, seed = 42)


# prepare the data for input
def data_collator(features: list) -> dict:
    # Check if pad_token_id is None, if it is then use eos_token_id as the padding value
    if tokenizer.pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id  # Use eos_token_id as a fill symbol
    else:
        pad_token_id = tokenizer.pad_token_id

    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)

    input_ids = []
    labels_list = []

    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]

        # Padding with calculated pad_token_id
        labels = (
            [pad_token_id] * (seq_len - 1) + ids[(seq_len - 1) :] + [pad_token_id] * (longest - ids_l)
        )
        ids = ids + [pad_token_id] * (longest - ids_l)

        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)

    return {
        "input_ids": input_ids,
        "labels": labels,
    }

from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback

# Train
# Took about 10 compute units. You will need WandDb password
writer = SummaryWriter()
# trainer = ModifiedTrainer(
trainer = Trainer(
    model=model,
    args=training_args,             # Trainer args
    train_dataset=dataset["train"], # Training set
    eval_dataset=dataset["test"],   # Testing set
    data_collator=data_collator,    # Data Collator
    callbacks=[TensorBoardCallback(writer)],
)
trainer.train()
writer.close()

# if you want to use this pretrained model, save to disk or google drive
# model_output_dir = 'Model/' # or your google drive
# model.save_pretrained(model_output_dir)


from transformers import AutoModelForSequenceClassification
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast

# Load Models for instruction tokenization
base_model = "meta-llama/Meta-Llama-3-8B"
tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

#get the finetuned model for inference
model = model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


#test with your sentence
# Make test prompts
prompt = [
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: Bitcoin is experiencing a significant growth. The market sentiment around Bitcoin is very positive.
Answer: ''',
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: 'BNB is experiencing a significant adoption.'
Answer: '''
]

#Now that prompt is defined, we can proceed with tokenization
tokens = tokenizer(prompt, return_tensors='pt', padding=True, max_length=128).to(device)
res = model.generate(**tokens, max_length=128)
res_sentences = [tokenizer.decode(i) for i in res]
out_text = [o.split("Answer: ")[1] for o in res_sentences]


# Show results
for sentiment in out_text:
    print(sentiment)



