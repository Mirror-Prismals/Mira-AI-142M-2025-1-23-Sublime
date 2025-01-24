from flask import Flask, request, jsonify, render_template_string
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize Flask app
app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_path = r"M:\checkpoint-1300"  # Replace with your model path
print("Loading the fine-tuned model...")
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Define the text generation function
def generate_text(prompt, max_length=1024, temperature=0.7, top_k=50, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# HTML template for the chatbot UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>half asleep</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #808000; /* Forest green */
            color: #006060;
            margin: 0;
            padding: 0;
        }
        #chat-container {
            width: 80%;
            margin: 50px auto;
            background-color: #002020;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
            padding: 20px;
        }
        #chat-log {
            height: 400px;
            overflow-y: auto;
            background-color: #111111;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        #chat-log div {
            margin: 5px 0;
        }
        #user-message {
            width: calc(100% - 100px);
            padding: 10px;
            border: none;
            border-radius: 5px;
        }
        #send-btn {
            padding: 10px;
            background-color: #0000ff; /* Dark red */
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        #send-btn:hover {
            background-color: #8080ff; /* Brown */
        }
    </style>
</head>
<body>
    <div id="chat-container">
        <h1>Mira's Rainforest Chatbot</h1>
        <div id="chat-log"></div>
        <input id="user-message" type="text" placeholder="Type your message here..." />
        <button id="send-btn">Send</button>
    </div>
    <script>
        const chatLog = document.getElementById('chat-log');
        const userMessage = document.getElementById('user-message');
        const sendBtn = document.getElementById('send-btn');

        sendBtn.addEventListener('click', () => {
            const message = userMessage.value.trim();
            if (message) {
                chatLog.innerHTML += `<div><strong>You:</strong> ${message}</div>`;
                userMessage.value = '';

                fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: message })
                })
                .then(response => response.json())
                .then(data => {
                    chatLog.innerHTML += `<div><strong>Mira:</strong> ${data.response}</div>`;
                    chatLog.scrollTop = chatLog.scrollHeight;
                })
                .catch(err => {
                    console.error(err);
                    chatLog.innerHTML += `<div style="color: red;"><strong>Error:</strong> Unable to get a response.</div>`;
                });
            }
        });

        userMessage.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                sendBtn.click();
            }
        });
    </script>
</body>
</html>
"""

# Flask routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    response = generate_text(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    # Use your local IP address to make the app accessible on your network
    app.run(host='0.0.0.0', port=5000, debug=True)
