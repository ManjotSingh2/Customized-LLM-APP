### North Indian Diet Expert Chatbot
# Overview
The Retrieval-Augmented Generation (RAG) LLM Chatbot is designed to provide accurate and contextually relevant responses by combining retrieval-based methods with generation-based models. This chatbot leverages external knowledge from a PDF document to enhance its responses, making it a powerful tool for domain-specific information retrieval and conversation. In this project, the chatbot specializes in providing dietary advice related to North Indian cuisine.

# Features
1. Contextually Relevant Responses: Combines retrieval-based and generation-based approaches to provide accurate responses.
2. Knowledge Integration: Uses external documents to supplement the chatbot's responses with up-to-date and domain-specific information.
3. Interactive User Interface: Utilizes Gradio to provide a user-friendly interaction platform.
4. North Indian Diet Expertise: Provides dietary advice, meal plans, healthy recipes, and nutritional information specific to North Indian cuisine.

# Setup Instructions
Create a requirements.txt file with the following content:

gradio
huggingface_hub
PyMuPDF
sentence-transformers
faiss-cpu

Prepare the PDF

Ensure you have a PDF file named North_Indian_Diet.pdf that contains information about the North Indian diet, recipes, nutrition, etc. Place this file in the same directory as your app.py.

# Usage


Example Interactions

Healthy North Indian Breakfast
User: Can you suggest a healthy North Indian breakfast?

Nutritional Benefits of Chickpeas
User: What are the nutritional benefits of chickpeas?

Balanced North Indian Meal Plan
User: How can I plan a balanced North Indian meal?

# Customization
System Message
Modify the system message in the respond function to change the chatbot's expertise area.

system_message = "You are a North Indian diet expert. You provide dietary advice, suggest meal plans, and answer questions related to North Indian cuisine and nutrition. Feel free to ask about healthy recipes, nutritional benefits of foods, or meal planning tips."

Response Parameters
Adjust the max_tokens, temperature, and top_p parameters to fine-tune the chatbot's responses.

gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),

Adding More Examples
Include more example questions in the examples section to guide users on how to interact with the chatbot.

examples=[
    ["Can you suggest a healthy North Indian breakfast?"],
    ["What are the nutritional benefits of chickpeas?"],
    ["How can I plan a balanced North Indian meal?"]
]

# Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.
