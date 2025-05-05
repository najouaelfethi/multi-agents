from multiagent_pattern.agent import Agent
from tool_pattern.get_weather_tool import get_weather
from tool_pattern.suggest_clothing_tool import suggest_clothing
from planning_pattern.react_agent import ReactAgent
from multiagent_pattern .crewai import Crew

with Crew() as crew:
    # --------------------Agent 1: Weather agent---------------------
    weather_agent = Agent(
        name="WeatherAgent",
        backstory="You are a weather expert. You can provide accurate weather information for any city.",
        task_description="Provide the weather for Casablanca city.",
        task_expected_output="temperature: <float>\ndescription: <string>",
        tools=[get_weather]
    )

    # --------------------------Agent 2: Clothing advisor agent----------------------
    clothing_agent = Agent(
        name="ClothingAdvisorAgent",
        backstory="You are a clothing advisor. Based on weather data, suggest appropriate clothing.",
        task_description="Use the temperature <float> and description <string> to suggest suitable clothes.",
        task_expected_output="Return a clothing advice text.",
        tools=[suggest_clothing]
    )

    weather_agent >> clothing_agent

#Run crew
crew.run()
