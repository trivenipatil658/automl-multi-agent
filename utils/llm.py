import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from the .env file (e.g. GROQ_API_KEY)
load_dotenv()


def get_llm_response(prompt):
    # Read the Groq API key from environment variables
    api_key = os.getenv("GROQ_API_KEY")

    # Raise an error early if the key is missing — avoids confusing API errors later
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    # Initialize the Groq client with the API key
    client = Groq(api_key=api_key)

    try:
        # Send the prompt to LLaMA 3.3 70B model via Groq's chat completions API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        # Extract and return the text content from the first response choice
        return response.choices[0].message.content

    except Exception as e:
        # Wrap any API error in a RuntimeError with a clear message
        raise RuntimeError(f"LLM request failed: {e}")


# Quick test — run this file directly to verify the LLM connection works
if __name__ == "__main__":
    reply = get_llm_response("Explain machine learning in one line")
    print(reply)
