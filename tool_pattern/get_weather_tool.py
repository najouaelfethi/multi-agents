import json
import requests
from tool_pattern.tool import tool


@tool
def get_weather(city: str):
    """
    Fetch the current weather for any city using the OpenWeatherMap API.
    It returns the temperature, weather description, and humidity.
    """

    api_key = "58a1809bd5d800285b6ff1070121ae2b"  
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()  

        data = response.json()

        weather_info = {
            "city": city,
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"]
        }

        return json.dumps(weather_info)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return json.dumps({"error": str(e)})
    
