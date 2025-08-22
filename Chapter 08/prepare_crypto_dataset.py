#!pip install datasets tqdm 

# !pip install transformers --upgrade
# !pip install accelerate
# !pip install -U bitsandbytes
# !pip install loguru
# !pip install --upgrade peft
# # !pip install transformers==4.40.1
# !pip install  datasets 
# !pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import pandas as pd
import datasets

import json
from tqdm.notebook import tqdm


data = pd.read_csv('Data/crypto_sentiment_data.csv')

#Convert the Pandas dataframe back to a Hugging Face Dataset object.
data = datasets.Dataset.from_pandas(data)

tmp_dataset = datasets.concatenate_datasets([data]*2) #Create a list that contains 2 data
train_dataset = tmp_dataset
print(tmp_dataset.num_rows)

all_dataset = train_dataset.shuffle(seed = 42)
print(all_dataset.shape)



def format_examle(example:dict) -> dict:    
  
  context = f"Instruction:{example['instruction']}\n"   #Initializes a string variable context using an f-string to format the instruction.
  if example.get('input'):     #Checks if the example dictionary has an input key and whether it contains a value.
    context += f"Input:{example['input']}\n"
  context += 'Answer: '
  target = example['output']
  # This creates the format of json data.
  return {"context": context , "target":target}  


# Iterate over each row of the dataset 
data_list = []
for item in all_dataset.to_pandas().itertuples():    
  tmp = {}
  tmp['instruction'] = item.instruction
  tmp['input'] = item.input
  tmp['output'] = item.output
  data_list.append(tmp)

# test
print(print(data_list[0]))

# save to a json file
with open("Data/dataset_new.jsonl",'w') as f:
  for example in tqdm(data_list,desc = 'formatting..'):
    f.write(json.dumps(format_examle(example)) + '\n')


j



