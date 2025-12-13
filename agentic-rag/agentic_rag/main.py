from dotenv import load_dotenv

# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
load_dotenv(override=False)

from graph.graph import app


if __name__ == "__main__":
    question = "What is React?"
    print(app.invoke(input={"question": question}))
