import random
import json
import pandas as pd

# Recreate a small mock dataset since the environment was reset
companies = ['QuantumSoft Technologies', 'NexGen Robotics']
sentiments = ['positive', 'negative', 'neutral', 'absolutely positive', 'absolutely negative']
texts = {
    'positive': 'Analysts upgrade {company} to Buy.',
    'negative': '{company} faces short-term headwinds.',
    'neutral': '{company} maintains its current guidance.',
    'absolutely positive': '{company} secures major government contract worth $1.2B.',
    'absolutely negative': '{company} under investigation for regulatory violations.'
}

# Generate  samples
samples = []
n = 20
for i in range(n): # Replace n with number of samples
    company = random.choice(companies)
    sentiment = random.choice(sentiments)
    text = texts[sentiment].format(company=company)

    # use appropriate topics
    if "contract" in text.lower():
        topic = "Government Contract"
    elif "guidance" in text.lower():
        topic = "Earnings Guidance"
    elif "investigation" in text.lower():
        topic = "Regulatory Risk"
    elif "upgrade" in text.lower():
        topic = "Analyst Opinion"
    else:
        topic = "General Market News"

    samples.append({
        "instruction": "Extract named entities, sentiment classification, and topic classification from the input text.",
        "input": text,
        "output": json.dumps({
            "entities": [company],
            "sentiment": sentiment,
            "topic": topic
        }, indent=2)
    })

# Split into train and eval
train_df = pd.DataFrame(samples[:15])
eval_df = pd.DataFrame(samples[15:])

train_df.to_csv('Data/train_df.csv')
eval_df.to_csv('Data/eval_df.csv')
