from tool_pattern.tool import Tool, tool


@tool
def suggest_clothing(temperature: float, description: str) -> str:
    advice = ""

    if temperature < 10:
        advice += "Wear a heavy coat. "
    elif temperature < 20:
        advice += "Wear a light sweater or jacket. "
    else:
        advice += "Light clothing is fine. "

    if "rain" in description.lower():
        advice += "Take an umbrella."
    elif "snow" in description.lower():
        advice += "Wear boots and a waterproof jacket."

    return advice or "Wear comfortable clothes."

