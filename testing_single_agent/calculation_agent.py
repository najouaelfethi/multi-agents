from planning_pattern.react_agent import ReactAgent
from tool_pattern.calculation_tool import sum_two_elements,multiply_two_elements,compute_log

#-------------Calculation Agent: 3 Tools---------------------

agent = ReactAgent(tools=[sum_two_elements, multiply_two_elements, compute_log])
response=agent.run(user_msg="I want to calculate the sum of 1234 and 5678 and multiply the result by 5. Then, I want to take the logarithm of this result")
print(response)