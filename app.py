from flask import Flask, request, jsonify, send_from_directory
import os
import opengradient as og

app = Flask(__name__)

client = og.Client(
    private_key=os.environ.get("OG_PRIVATE_KEY"),
)

MODELS = {
    "gpt-4o": "gpt-4o",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")
    model_key = data.get("model", "llama3")
    model_cid = MODELS.get(model_key, MODELS["llama3"])

    tx_hash, response = og.infer_llm(
        model_cid=model_cid,
        prompt=message,
        max_tokens=500,
        temperature=0.7
    )

    return jsonify({
        "response": str(response),
        "tx_hash": tx_hash,
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
