from openai import OpenAI
import config.constants as config
def base_model():
    client = OpenAI(
        api_key = config.OPENAIAPI 
    )

    return client