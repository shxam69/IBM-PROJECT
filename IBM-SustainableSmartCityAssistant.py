!pip install transformers torch gradio PyPDF2 -q  
import gradio as gr  
import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM  import PyPDF2  
import io  
# --- 1. Load Model and Tokenizer (with optimization for Colab) ---  
# Check for GPU availability  
device = "cuda" if torch.cuda.is_available() else "cpu"  
dtype = torch.float16 if device == "cuda" else torch.float32  
model_name = "ibm-granite/granite-3.2-2b-instruct"  print(f"Loading model '{model_name}' on device: {device}...")  
try:  
    tokenizer = AutoTokenizer.from_pretrained(model_name)      model = AutoModelForCausalLM.from_pretrained(  
        model_name,  
        torch_dtype=dtype,  
        device_map="auto" if device == "cuda" else None  
    )  
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token  
    print("Model loaded successfully.")  except Exception as e:  
    print(f"Error loading model: {e}")  
    raise  
# --- 2. Core Functions ---  
def generate_response(prompt, max_new_tokens=256, temperature=0.6):  
    """Generates a text response based on a given prompt."""  
    try:  
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)          if device == "cuda":  
            inputs = {k: v.to(model.device) for k, v in inputs.items()}  
        with torch.no_grad():  
            outputs = model.generate(  
                **inputs,  
                max_new_tokens=max_new_tokens,                  temperature=temperature,  
                do_sample=True,  
                pad_token_id=tokenizer.eos_token_id              )  
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)  
# Robustly clean up the output to remove the input prompt  
        if response.startswith(prompt):  
            response = response[len(prompt):].strip()  
        return response  
    except Exception as e:  
        return f"An error occurred during text generation: {str(e)}"  
def extract_text_from_pdf(pdf_file):  
    """Extracts text from an uploaded PDF file."""      if pdf_file is None:  
        return ""  
    try:  
        pdf_reader = PyPDF2.PdfReader(pdf_file)          text = ""  
        for page in pdf_reader.pages:  
            text += page.extract_text() + "\n"  
        return text  
    except Exception as e:  
        return f"Error reading PDF: {str(e)}"  
def eco_chat_response(message, history):  
    """Handles the eco-friendly chatbot conversation."""  
    if not message.strip():  
        yield "Please enter a question to get a response!"  
        return  
# Added explicit instruction to the system prompt  
    system_prompt = "You are an AI assistant specialized in providing practical, actionable, and eco-friendly tips for sustainable living. Only provide a direct answer to the user's question. Do not ask a follow-up question."  
    conversation_history = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history])  
    full_prompt = f"{system_prompt}\n\n{conversation_history}\nUser: {message}\nAssistant:"  
    yield generate_response(full_prompt, max_new_tokens=256)  
def general_ai_assistant(message, history):  
    """Handles the general-purpose AI assistant conversation."""  
    if not message.strip():  
        yield "Hello! How can I help you today?"  
        return  
# Added explicit instruction to the system prompt  
    system_prompt = "You are a helpful, general-purpose AI assistant. Your goal is to provide clear and concise answers to a wide range of questions. Only provide a direct answer. Do not ask a follow-up question."  
    conversation_history = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history])  
    full_prompt = f"{system_prompt}\n\n{conversation_history}\nUser: {message}\nAssistant:"  
    yield generate_response(full_prompt, max_new_tokens=300)  
def policy_summarization(pdf_file, policy_text):  
    """Summarizes a policy from a PDF or pasted text."""  
    content = ""  
    if pdf_file is not None:  
        content = extract_text_from_pdf(pdf_file)  
        if "Error" in content:  
            return content  
    elif policy_text.strip():  
        content = policy_text  
    if not content.strip():  
        return "Please upload a PDF or paste text to summarize!"  
    summary_prompt = f"Summarize the following policy document and extract the most important points, key provisions, and implications:\n\n{content}"  
return generate_response(summary_prompt, max_new_tokens=400)  
def check_policy_compliance(text, policy_type):  
    """Checks a text for compliance with a given policy type."""  
    if not text.strip():  
        return "Please provide text to check for compliance."  
    prompts = {  
        "Environmental": f"Analyze the following text for its alignment with environmental sustainability principles, focusing on resource use, waste reduction, and eco-friendly practices. Provide a summary of its compliance status:\n\n{text}",  
        "Social Justice": f"Analyze the following text for its alignment with social justice and equity principles, focusing on fair labor, community impact, and diversity. Provide a summary of its compliance status:\n\n{text}",  
        "Ethical AI Use": f"Analyze the following text for its alignment with ethical AI principles, focusing on fairness, transparency, and accountability. Provide a summary of its compliance status:\n\n{text}"  
    }  
    compliance_prompt = prompts.get(policy_type, "Please select a valid policy type.")  
    if compliance_prompt.startswith("Please select"):  
        return compliance_prompt  
    return generate_response(compliance_prompt, max_new_tokens=500)  
def analyze_text(text):  
    """Performs sentiment analysis and keyword extraction on a given text."""  
    if not text.strip():  
        return "Please provide text to analyze.", "Please provide text to analyze."  
    sentiment_prompt = f"Analyze the sentiment of the following text: '{text}'. Is the sentiment positive, negative, or neutral? Explain your reasoning."  
    sentiment_result = generate_response(sentiment_prompt, max_new_tokens=100)  
    keywords_prompt = f"Extract the most important keywords from the following text, separated by commas: '{text}'."  
    keywords_result = generate_response(keywords_prompt, max_new_tokens=50)  
    return sentiment_result, keywords_result  
def submit_feedback(feedback_text):  
    """A placeholder function for submitting feedback."""  
    if not feedback_text.strip():  
        gr.Warning("Please enter your feedback before submitting.")          return "No feedback submitted."  
    gr.Info("Thank you for your feedback! It has been submitted.")      return "Feedback received."  
# --- 3. Create Gradio Interface with all features ---  
with gr.Blocks(  
    theme=gr.themes.Glass(),  
    title="Eco Assistant & Policy Analyzer"  
) as app:  
    gr.Markdown(  
        """  
#    Multi-Purpose AI Assistant     
        A powerful tool for sustainable living, policy analysis, and more.          """  
    )  
    with gr.Tabs():  
# Tab 1: Eco-Friendly Chatbot  
        with gr.TabItem("   Eco-Friendly Chatbot"):  
            gr.Markdown("### Ask me anything about sustainable living!")  
            gr.ChatInterface(  
                fn=eco_chat_response,  
                title="Eco-Friendly Chatbot",  
                description="Your personal assistant for a greener lifestyle.",  
                chatbot=gr.Chatbot(height=500, label="Conversation", show_label=True, show_copy_button=True),  
                submit_btn="Ask   ",  
                examples=[  
                    ["How can I reduce plastic waste?"],  
                    ["What are some tips for saving water at home?"],  
                    ["Give me ideas for a zero-waste lifestyle."]  
                ]  
            )  
# Tab 2: General AI Assistant  
        with gr.TabItem("   General AI Assistant"):  
            gr.Markdown("### Your all-purpose AI assistant.")  
            gr.ChatInterface(  
                fn=general_ai_assistant,  
                title="General AI Assistant",  
                description="Ask me any general question you have.",  
                chatbot=gr.Chatbot(height=500, label="Conversation", show_label=True, show_copy_button=True),  
                submit_btn="Ask   ",  
                examples=[  
                    ["What is the capital of France?"],  
                    ["Explain the theory of relativity."],  
                    ["Write a short poem about the ocean."]  
                ]  
            )  
# Tab 3: Policy Summarizer  
        with gr.TabItem("   Policy Summarizer"):  
            gr.Markdown("### Upload a PDF or paste text to get a quick summary.")  
            with gr.Column():  
                with gr.Row():  
                    gr.Markdown("#### Step 1: Provide the Document")  
                with gr.Row():  
                    pdf_upload = gr.File(label="Upload a PDF file", file_types=[".pdf"])  
                gr.Markdown("---")  
                with gr.Row():  
                    gr.Markdown("#### Or, paste the text here:")  
                with gr.Row():  
                    policy_text_input = gr.Textbox(label="Paste document text", placeholder="Paste the full policy text here...", lines=10)  
            with gr.Row():  
                summarize_btn = gr.Button("  Summarize Policy Document", variant="primary", scale=1)              with gr.Row():  
                summary_output = gr.Textbox(label="Policy Summary & Key Points", lines=20, interactive=False)  
            summarize_btn.click(  
                policy_summarization,  
                inputs=[pdf_upload, policy_text_input],                  outputs=summary_output,              )  
# Tab 4: Policy Compliance Check  
        with gr.TabItem("  Policy Compliance Check"):  
            gr.Markdown("### Check a text for compliance with specific principles.")  
            with gr.Column():  
                text_to_check = gr.Textbox(label="Text to Check", lines=10, placeholder="Paste a project description, policy statement, or other text to analyze.")  
                policy_options = gr.Radio(  
                    ["Environmental", "Social Justice", "Ethical AI Use"],  
                    label="Choose Policy Type"  
                )  
                compliance_output = gr.Textbox(label="Compliance Report", lines=10, interactive=False)                  check_btn = gr.Button("Analyze Compliance", variant="primary")  
            check_btn.click(  
                check_policy_compliance,  
                inputs=[text_to_check, policy_options],  
                outputs=compliance_output,  
            )  
# Tab 5: Text Analyzer  
        with gr.TabItem("   Text Analyzer"):  
            gr.Markdown("### Analyze text for sentiment and keywords.")  
            with gr.Column():  
                analyze_input = gr.Textbox(label="Text to Analyze", lines=10, placeholder="Enter text here to analyze its sentiment and extract keywords.")  
            with gr.Row():  
                analyze_btn = gr.Button("Analyze Text", variant="primary")  
            with gr.Row():  
                sentiment_output = gr.Textbox(label="Sentiment Analysis", lines=5, interactive=False)  
                keywords_output = gr.Textbox(label="Extracted Keywords", lines=5, interactive=False)  
            analyze_btn.click(  
                analyze_text,  
                inputs=analyze_input,  
                outputs=[sentiment_output, keywords_output],  
            )  
# Tab 6: Feedback  
        with gr.TabItem("  Feedback"):  
            gr.Markdown("### We'd love to hear your thoughts!")  
            with gr.Column():  
                feedback_input = gr.Textbox(label="Your Feedback", placeholder="Tell us what you think...", lines=5)  
                feedback_btn = gr.Button("Submit Feedback", variant="secondary")  
            feedback_output = gr.Textbox(label="Status", interactive=False)  
            feedback_btn.click(  
                submit_feedback,  
                inputs=feedback_input,  
                outputs=feedback_output,  
            )  
# Launch the app with sharing enabled for Google Colab  app.launch(share=True)
