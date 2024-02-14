import chainlit as cl
from vllm_server import llm, sampling_params


@cl.step
def tool():
    return "Response from the tool!"


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """

    # Call the tool
    #tool()

    # Send the final answer.
    await cl.Message(content=llm.generate(str(message), sampling_params=sampling_params)).send()