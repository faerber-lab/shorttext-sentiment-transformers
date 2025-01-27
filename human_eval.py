from flask import Flask, request, render_template, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Global variables
data = None
classified_data = []
current_index = 0
classes = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]

@app.route('/')
def index():
    return render_template("index.html", classes=classes)

@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    global data, classified_data, current_index

    # Load dataset
    dataset_path = request.form['dataset_path']
    dataset_path = "data/public_data/dev/track_a/eng.csv"
    try:
        data = pd.read_csv(dataset_path)
        if 'id' not in data.columns or 'text' not in data.columns:
            return jsonify({"error": "Dataset must contain 'id' and 'text' columns."}), 400

        classified_data = []
        current_index = 0
        return jsonify({"message": "Dataset loaded successfully.", "total_examples": len(data)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_example', methods=['GET'])
def get_example():
    global data, current_index

    # Number of examples requested
    num_examples = int(request.args.get('num_examples', 1))

    if data is None:
        return jsonify({"error": "Dataset not loaded."}), 400

    if current_index >= len(data):
        return jsonify({"message": "All examples have been classified."})

    # Slice the requested number of examples
    examples = data.iloc[current_index:current_index + num_examples]
    current_index += num_examples
    
    print(examples)

    # Convert to JSON format and return
    return jsonify(examples[['id', 'text']].to_dict(orient='records'))


@app.route('/submit_classification', methods=['POST'])
def submit_classification():
    global classified_data

    classifications = request.json['classifications']
    classified_data.extend(classifications)

    return jsonify({"message": "Classifications submitted."})

@app.route('/save_classifications', methods=['POST'])
def save_classifications():
    data = request.json.get('data', [])
    output_path = request.json.get('output_path', 'classified_data.csv')

    try:
        # Convert the data to a DataFrame
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        return jsonify({"message": f"Classified data saved to {output_path}."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)