from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # loads OPENAI_API_KEY from .env if present
print(os.getenv("OPENAI_API_KEY")[:8])
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#def main():
    #r = client.responses.create(model="gpt-5-mini", input="Reply with 'OK' only.")
    #print(r.output_text)

    # --- Streaming example (optional) ---
    # with client.responses.stream(
    #     model="gpt-5-mini",
    #     input="Stream the word 'StreamingOK' only."
    # ) as stream:
    #     for event in stream:
    #         if event.type == "response.output_text.delta":
    #             print(event.delta, end="", flush=True)
    #     print()

#if __name__ == "__main__":
   # main()
