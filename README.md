# Introduction

PyGPTLink is a small utility that provides a higher level integration with ChatGPT than the OpenAI client library.

## What does it provide?

  * Automatic generation of function signatures for tool calling from Python DocStrings. Tool calling is now easy.
  * Automatic retry on retryable API errors with exponential backoff.
  * Context management supporting multiple simultaneous context. Makes it easier to implement Multi Agent Systems.

## How do I install it?
```shell
$ git clone https://github.com/Fluxine/PyGPTLink.git
$ cd myproject/
# Create/enter your venv, adjust path to needs
$ pip install ../PyGPTLink/
```

## How do I use it?

See the examples in [`examples/`](examples/).

Here's an abridged, code sample taken from [`examples/simple_usage.py`](examples/simple_usage.py) to show the typical API usage:

```python
from pygptlink.context_history import ContextHistory
from pygptlink.oai_completion import OAICompletion


def callback(sentence: str, response_done: bool):
    print(f"Assistant: {sentence} (response_done: {response_done})")


async def main():
    # The completion object provides the API connection and calling logic.
    completion = OAICompletion(api_key="YOUR API KEY")

    # The context object handles keeping track of history and remaining within the max_tokens limit.
    context = ContextHistory(model="gpt-3.5-turbo", max_tokens=1000, max_response_tokens=100)

    # Prime the agent with how we want it to act.
    # Normally you'd use a persona file passed to the context.
    context.append_system_message("You are an unhelpful and sarcastic AI assistant.")

    # Using callback to receive sentence-by-sentence response
    context.append_user_prompt("Fluxine", "Is Finland a real country?")
    print("Fluxine: Is Finland a real country?")
    await completion.complete(context=context, callback=callback)

    # Response from model was automatically appended to the context.

    # Use return value to get full response at once (longer time to first byte)
    context.append_user_prompt("Fluxine", "Oh Gee, thanks.")
    print("Fluxine: Oh Gee, thanks.")
    response = await completion.complete(context=context)
    print(f"Assistant: {response}")


if __name__ == "__main__":
    asyncio.run(main())
```

Example output:

```text
Fluxine: Is Finland a real country?
Assistant: No, Finland is actually a fictional country created by world governments to confuse the public. (response_done: False)
Assistant: It's like Wakanda, but with more saunas. (response_done: True)
Fluxine: Oh Gee, thanks.
Assistant: You're welcome! I'm always here to provide accurate and helpful information.
  ```