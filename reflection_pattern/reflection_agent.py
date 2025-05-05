from colorama import Fore
from dotenv import load_dotenv
import os
from openai import OpenAI
from utils.completions import build_prompt_structure
from utils.completions import completions_create #sends prompt to LLm
from utils.completions import FixedFirstChatHistory #fixed the first message while adding new ones
from utils.completions import update_chat_history
from utils.logging import fancy_step_tracker

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
openai_org = os.getenv("OPENAI_ORG_ID")

BASE_GENERATION_SYSTEM_PROMPT = """
Your are a helpful assistant, your task is to generate the best content possible for the user's request.
If the user gives feedback, revise your previous answer
"""

BASE_REFLECTION_SYSTEM_PROMPT = """
You are tasked to review the user's response. If there are problems or improvements, list them.
If everything is fine, just reply: <OK>
When reviewing, check for any duplicated lines, repeated code, or unnecessary comments and suggest corrections if found.
"""


class ReflectionAgent:

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=openai_key, organization=openai_org)
        self.model = model

    #private method to send propmt(chat history) to LLM
    def _request_completion(
        self,
        history: list, #list of messages forming the conversation
        verbose: int = 0,
        log_title: str = "COMPLETION",
        log_color: str = "",
    ):
        output = completions_create(self.client, history, self.model)

        #Print extra information; like what the agent is doing at each step
        if verbose > 0:
            print(log_color, f"\n\n{log_title}\n\n", output)

        return output

    def generate(self, generation_history: list, verbose: int = 0) -> str:
        return self._request_completion(
            generation_history, verbose, log_title="GENERATION", log_color=Fore.BLUE
        )

    def reflect(self, reflection_history: list, verbose: int = 0) -> str:
        return self._request_completion(
            reflection_history, verbose, log_title="REFLECTION", log_color=Fore.GREEN
        )

    def run(
        self,
        user_msg: str,
        generation_system_prompt: str = "",
        reflection_system_prompt: str = "",
        n_steps: int = 3,
        verbose: int = 0,
    ) -> str:
        generation_system_prompt += BASE_GENERATION_SYSTEM_PROMPT
        reflection_system_prompt += BASE_REFLECTION_SYSTEM_PROMPT

        generation_history = FixedFirstChatHistory(
            [
                build_prompt_structure(prompt=generation_system_prompt, role="system"),
                build_prompt_structure(prompt=user_msg, role="user"),
            ],
            total_length=3,
        )

        reflection_history = FixedFirstChatHistory(
            [build_prompt_structure(prompt=reflection_system_prompt, role="system")],
            total_length=3,#max 3 messages
        )

        for step in range(n_steps):
            if verbose > 0:
                fancy_step_tracker(step, n_steps)

            # Generate the response
            generation = self.generate(generation_history, verbose=verbose)
            update_chat_history(generation_history, generation, "assistant")
            update_chat_history(reflection_history, generation, "user")

            # Reflect and critique the generation
            critique = self.reflect(reflection_history, verbose=verbose)

            if "<OK>" in critique:
                # If no additional suggestions are made, stop the loop
                print(
                    Fore.CYAN,
                    "\n\nStop Sequence found. Stopping the reflection loop ... \n\n",
                )
                break

            update_chat_history(generation_history, critique, "user")
            update_chat_history(reflection_history, critique, "assistant")

        return generation
