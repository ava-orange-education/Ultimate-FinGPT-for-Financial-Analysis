import random
import json
import pandas as pd

# Analytical Method 2: Sentiment Shift Detection & Event Impact
method2_samples = [
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

# Analytical Method 3: Market Scenario Simulation
method3_samples = [
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

# Combine and convert to DataFrame
instruction_samples = method2_samples + method3_samples
instruction_df = pd.DataFrame(instruction_samples)

instruction_df.to_json('Data/method_2_3.json')
