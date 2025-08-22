import pandas as pd

# Recreate all prior samples (Method 1 + Method 2 + Method 3)
# Method 1: NER + Sentiment + Topic Classification (5 examples)
companies = ['QuantumSoft Technologies', 'NexGen Robotics']
sentiments = ['positive', 'negative', 'neutral', 'absolutely positive', 'absolutely negative']
texts = {
    'positive': 'Analysts upgrade {company} to Buy.',
    'negative': '{company} faces short-term headwinds.',
    'neutral': '{company} maintains its current guidance.',
    'absolutely positive': '{company} secures major government contract worth $1.2B.',
    'absolutely negative': '{company} under investigation for regulatory violations.'
}
method1_data = []
for i in range(10):
    company = random.choice(companies)
    sentiment = random.choice(sentiments)
    text = texts[sentiment].format(company=company)
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
    method1_data.append({
        "instruction": "Extract named entities, sentiment classification, and topic classification from the input text.",
        "input": text,
        "output": json.dumps({
            "entities": [company],
            "sentiment": sentiment,
            "topic": topic
        }, indent=2)
    })

# Method 2: Sentiment Shift Detection & Event Impact (2 examples)
method2_data = [
    {
        "instruction": "Analyze the sentiment trend before and after a market event, and determine if the sentiment shifted significantly.",
        "input": (
            "Before Event: QuantumSoft Technologies shows steady sentiment across sources.\n"
            "Event: TechCloud Services faces regulatory investigation.\n"
            "After Event: Increased negative sentiment observed around QuantumSoft Technologies."
        ),
        "output": json.dumps({
            "sentiment_shift": "Significant Negative Shift",
            "affected_entities": ["QuantumSoft Technologies", "TechCloud Services"],
            "trigger_event": "Regulatory Investigation",
            "potential_impact": "Sector-wide reputation risk and investor concern"
        }, indent=2)
    },
    {
        "instruction": "Assess whether a market event resulted in a detectable sentiment shift in related companies.",
        "input": (
            "Event: CEO resignation at NexGen Robotics.\n"
            "Sentiment Data: Neutral sentiment turns slightly negative across multiple platforms.\n"
            "Market Data: Increased volatility in NexGen's stock price."
        ),
        "output": json.dumps({
            "sentiment_shift": "Mild Negative Shift",
            "affected_entities": ["NexGen Robotics"],
            "trigger_event": "Executive Turnover",
            "potential_impact": "Temporary loss of investor confidence"
        }, indent=2)
    }
]

# Method 3: Scenario Simulation (2 examples)
method3_data = [
    {
        "instruction": "Generate possible market scenarios based on increased trade tensions and assess the impact on technology stocks.",
        "input": (
            "Scenario: Escalating trade tensions between US and China.\n"
            "Focus: Impact on semiconductor and AI sectors."
        ),
        "output": json.dumps({
            "scenarios": [
                "Moderate disruption to chip supply chains; minor decline in hardware stock prices.",
                "Significant delays in AI hardware deployment; major decline in key semiconductor stocks."
            ],
            "impacted_sectors": ["Semiconductors", "Artificial Intelligence"],
            "risk_level": "High"
        }, indent=2)
    },
    {
        "instruction": "Simulate market conditions following a macroeconomic shock and evaluate possible portfolio responses.",
        "input": (
            "Event: Sudden interest rate hike by the Federal Reserve.\n"
            "Portfolio: Heavy allocation in tech growth stocks."
        ),
        "output": json.dumps({
            "scenarios": [
                "Valuation compression in growth stocks; recommend short-term rotation into defensives.",
                "Increased volatility; hedge with options or inverse ETFs."
            ],
            "recommended_actions": ["Reduce tech exposure", "Increase allocation to cash and bonds"],
            "risk_level": "Medium to High"
        }, indent=2)
    }
]

# Combine all methods into a single dataset
all_samples = method1_data + method2_data + method3_data
all_samples_df = pd.DataFrame(all_samples)

all_samples_df.to_csv('Data/all_analyticalmethods_samples.csv')