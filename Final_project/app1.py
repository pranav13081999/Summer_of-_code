import gradio as gr
from multimodal_chat import get_answer

# Define the chatbot function
def chatbot(question):
    answer = get_answer(question)
    return answer

# Define the Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="InstiGPT",
    description="A simple chatbot that answers qustions related to insti"
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()


