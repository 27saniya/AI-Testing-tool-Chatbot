import streamlit as st
from chatbot_text_image import Chatbot_Response
import os

# App title
st.set_page_config(page_title="🤖💬 AI Testing Tool Chatbot")

# Chatbot Credentials
with st.sidebar:
    st.title('🤖💬 AI Testing Tool Chatbot')
    # Load your chatbot image
    current_directory = os.getcwd()
    chatbot_image = os.path.join(current_directory, f'Icon.jpeg')

    # Add the image to the sidebar
    st.sidebar.image(chatbot_image, use_column_width=True)
    
    
# Initialize the chatbot
chatbot = Chatbot_Response()  # Modify this line with your chatbot initialization

# Store chatbot generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How can I assist you today?"}]

# Display chat messages
for message in st.session_state.messages:
    if "image" in message:
        st.image(message["image"], caption="User Uploaded Image", use_column_width=True)
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating chatbot response
def generate_response(prompt_input, uploaded_image=None):
    # Replace this with your chatbot logic to generate a response
    if prompt_input.lower() == "bye":
          return "Chatbot: Goodbye!"
          
    if prompt_input.lower() == "hi":
        return "Chatbot: Hello"
       

    if "thanks" in prompt_input.lower():
        return "Chatbot: You are welcome"
      
      
    if "zoom" in prompt_input.lower():
        return "At this time, we don't have this information. Please contact the admin."
    
    if uploaded_image:
        # If image is uploaded, send it to chatbot.answer_question_with_image()
        response = chatbot.answer_question_with_image(prompt_input, uploaded_image)
    else:
        response = chatbot.answer_question(prompt_input)
    return response

# User-provided prompt
if user_input := st.text_input("User: "):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Convert to bytes
    uploaded_image_bytes = uploaded_image.getvalue()
    st.session_state.messages.append({"role": "user", "content": "Uploaded an image:", "image": uploaded_image_bytes})


# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(user_input, uploaded_image)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)