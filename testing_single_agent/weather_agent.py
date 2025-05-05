from planning_pattern.react_agent import ReactAgent
from tool_pattern.tool_agent import ToolAgent
from tool_pattern.get_weather_tool import get_weather
from tool_pattern.suggest_clothing_tool import suggest_clothing

#--------------Weather Agent: 2 Tools---------------

agent = ReactAgent(tools=[get_weather, suggest_clothing])
response=agent.run(user_msg="What's the weather like in Casablanca? and what clothes do you suggest for me to wear?")
print(response)