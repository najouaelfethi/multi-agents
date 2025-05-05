from reflection_pattern.reflection_agent import ReflectionAgent

# Initialize the agent
agent = ReflectionAgent(
    model="gpt-4o-mini" 
)

user_msg = "Generate a Python implementation of the Merge Sort algorithm"


generation_system_prompt = ""
reflection_system_prompt = ""

final_response = agent.run(
    user_msg=user_msg,
    generation_system_prompt=generation_system_prompt,
    reflection_system_prompt=reflection_system_prompt,
    n_steps=3,
    verbose=1 #shows detailed output
)

print("\nFINAL RESPONSE:\n", final_response)
