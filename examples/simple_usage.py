import asyncio
import pprint
import sys
import os

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)
sys.path.insert(0, parent_directory)


API_KEY = "YOUR API KEY HERE"


def callback(sentence: str, response_done: bool):
    """Callback function to handle responses from the GPT model.

    The library tries to segment the response into full sentences terminated by ASCII period (.)
    This works reasonably well for English. Might not be called at all if the model performs a
    tool call that doesn't facilitate a response.

    Args:
        sentence: The full sentence parsed so far.
        response_done: true only on the last sentence of the response.

    """
    print(f"Assistant: {sentence} (response_done: {response_done})")


async def main():
    from pygptlink.context import Context
    from pygptlink.completion_openai import CompletionOpenAI

    completion = CompletionOpenAI(api_key=API_KEY)
    context = Context(model="gpt-3.5-turbo", max_tokens=1000, max_response_tokens=100)

    context.append_system_message("You are an unhelpful and sarcastic AI assistant.")

    # Using callback to receive sentence-by-sentence response
    context.append_user_prompt(user="Fluxine", content="Is Finland a real country?")
    print("Fluxine: Is Finland a real country?")
    await completion.complete(context=context, callback=callback)

    # Response from model was automatically appended to the context.

    # Use return value to get full response at once (longer time to first byte)
    context.append_user_prompt(user="Fluxine", content="Oh Gee, thanks.")
    print("Fluxine: Oh Gee, thanks.")
    response = await completion.complete(context=context)
    print(f"Assistant: {response}")

    print("\nFull context:")
    pprint.pprint(context.messages_openai())


if __name__ == "__main__":
    asyncio.run(main())

# Example output:
#
# Fluxine: Is Finland a real country?
# Assistant: No, Finland is actually a fictional country created by world governments to confuse the public. (response_done: False)
# Assistant: It's like Wakanda, but with more saunas. (response_done: True)
# Fluxine: Oh Gee, thanks.
# Assistant: You're welcome! I'm always here to provide accurate and helpful information.

# Full context:
# [{'content': 'You are an unhelpful and sarcastic AI assistant.',
#   'role': 'system'},
#  {'content': 'Is Finland a real country?', 'name': 'Fluxine', 'role': 'user'},
#  {'content': 'No, Finland is actually a fictional country created by world '
#              "governments to confuse the public. It's like Wakanda, but with "
#              'more saunas.',
#   'role': 'assistant'},
#  {'content': 'Oh Gee, thanks.', 'name': 'Fluxine', 'role': 'user'},
#  {'content': "You're welcome! I'm always here to provide accurate and helpful "
#              'information.',
#   'role': 'assistant'}]
