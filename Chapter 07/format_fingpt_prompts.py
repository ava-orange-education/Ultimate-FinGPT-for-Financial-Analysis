from datasets import Dataset
from transformers import AutoTokenizer
import json


# Sample instruction dataset (simulate from previously prepared DataFrame)

def format_fingpt_prompts():
    instruction_data = [
        {
            "instruction": "Extract named entities, sentiment classification, and topic classification from the input text.",
            "input": "QuantumSoft Technologies secures major government contract worth $1.2B.",
            "output": json.dumps({
                "entities": ["QuantumSoft Technologies"],
                "sentiment": "absolutely positive",
                "topic": "Government Contract"
            }, indent=2)
        },
        {
            "instruction": "Generate possible market scenarios based on increased trade tensions and assess the impact on technology stocks.",
            "input": "Scenario: Escalating trade tensions between US and China.\nFocus: Impact on semiconductor and AI sectors.",
            "output": json.dumps({
                "scenarios": [
                    "Moderate disruption to chip supply chains; minor decline in hardware stock prices.",
                    "Significant delays in AI hardware deployment; major decline in key semiconductor stocks."
                ],
                "impacted_sectors": ["Semiconductors", "Artificial Intelligence"],
                "risk_level": "High"
            }, indent=2)
        }
    ]

    # Convert to Hugging Face dataset
    hf_dataset = Dataset.from_list(instruction_data)

    # Prompt template
    def format_fingpt_prompt(example):
        return f"""### Instruction:
    {example['instruction']}

    ### Input:
    {example['input']}

    ### Response:
    {example['output']}"""

    # Apply prompt formatting
    hf_dataset = hf_dataset.map(lambda x: {"text": format_fingpt_prompt(x)})

    # Load tokenizer for LLaMA 2 and Tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained("ANY HUGGINGFACE model") # example, meta-llama/Llama-2-7b-chat-hf
    tokenizer.pad_token = tokenizer.eos_token

    # 
    tokenized_dataset = hf_dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
        batched=True
    )

    # Format for PyTorch training
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    return tokenized_dataset