import json
from typing import Callable


def get_fn_signature(fn: Callable) -> dict:
    """
    Generates the signature for a given function.

    Args:
        fn (Callable): The function whose signature needs to be extracted.

    Returns:
        dict: A dictionary containing the function's name, description,
              and parameter types.
    example:
    <tools> {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location location (str): The city and state, e.g. Madrid, Barcelona unit (str): The unit. It can take two values; 'celsius', 'fahrenheit'",
    "parameters": {
        "properties": {
            "location": {
                "type": "str"
            },
            "unit": {
                "type": "str"
            }
        }
    }
    }
    </tools>
    """
    fn_signature: dict = {
        "name": fn.__name__,
        "description": fn.__doc__,
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
    
    # Get parameter types from function annotations
    for param_name, param_type in fn.__annotations__.items():
        if param_name != "return":
            fn_signature["parameters"]["properties"][param_name] = {
                "type": param_type.__name__ if hasattr(param_type, "__name__") else str(param_type)
            }
    
    return fn_signature

#maybe delete it later
def validate_arguments(tool_call: dict, tool_signature: dict) -> dict:
    """
    Validates and converts arguments in the input dictionary to match the expected types.

    Args:
        tool_call (dict): A dictionary containing the arguments passed to the tool.
        tool_signature (dict): The expected function signature and parameter types.

    Returns:
        dict: The tool call dictionary with the arguments converted to the correct types if necessary.
    """
    properties = tool_signature["parameters"]["properties"]

    # Type mapping for conversion
    type_mapping = {
        "int": int,
        "str": str,
        "bool": bool,
        "float": float,
        "list": list,
        "dict": dict
    }

    for arg_name, arg_value in tool_call["arguments"].items():
        if arg_name in properties:
            expected_type = properties[arg_name].get("type")
            if expected_type in type_mapping:
                try:
                    if not isinstance(arg_value, type_mapping[expected_type]):
                        tool_call["arguments"][arg_name] = type_mapping[expected_type](arg_value)
                except (ValueError, TypeError):
                    raise ValueError(f"Could not convert argument '{arg_name}' to type {expected_type}")

    return tool_call


class Tool:
    """
    A class representing a tool that wraps a callable and its signature.

    Attributes:
        name (str): The name of the tool (function).
        fn (Callable): The function that the tool represents.
        fn_signature (str): JSON string representation of the function's signature.
    """

    def __init__(self, name: str, fn: Callable, fn_signature: str):
        self.name = name
        self.fn = fn
        self.fn_signature = fn_signature

    def __str__(self):
        return self.fn_signature

    def run(self, **kwargs):
        """
        Executes the tool (function) with provided arguments.

        Args:
            **kwargs: Keyword arguments passed to the function.

        Returns:
            The result of the function call.
        """
        return self.fn(**kwargs)

#A decorator that wraps a function into a Tool object.
def tool(fn: Callable):
    """
    A decorator that wraps a function into a Tool object.

    Args:
        fn (Callable): The function to be wrapped.

    Returns:
        Tool: A Tool object containing the function, its name, and its signature.
    """

    def wrapper():
        fn_signature = get_fn_signature(fn)
        return Tool(
            name=fn_signature.get("name"), fn=fn, fn_signature=json.dumps(fn_signature)
        )
    return wrapper()

