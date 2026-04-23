import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def get_llm_response(prompt):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}")


if __name__ == "__main__":
    reply = get_llm_response("Explain machine learning in one line")
    print(reply)
