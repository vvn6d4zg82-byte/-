from flask import Flask, request, jsonify
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

model = None
GESTURE_NAMES = {
    0: "拳头",
    1: "手掌", 
    2: "指物",
    3: "胜利",
    4: "点赞"
}

@app.route('/train', methods=['POST'])
def train():
    global model
    data = request.json
    
    X = np.array(data['features'])
    y = np.array(data['labels'])
    
    print(f"Training with {len(y)} samples...")
    model = KNeighborsClassifier(n_neighbors=3, weights='distance')
    model.fit(X, y)
    
    return jsonify({'message': f'Model trained with {len(y)} samples!', 'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'gesture': None, 'error': 'Model not trained'})
    
    data = request.json
    features = np.array(data['features'])
    pred = model.predict([features])[0]
    gesture = GESTURE_NAMES.get(int(pred), "未知")
    
    return jsonify({'gesture': gesture, 'confidence': 0.9})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'model_loaded': model is not None})

if __name__ == '__main__':
    print("=== Gesture Server ===")
    print("Run: python gesture_server.py")
    print("Endpoints:")
    print("  POST /train - Train model")
    print("  POST /predict - Predict gesture")
    print("  GET /status - Check status")
    app.run(host='0.0.0.0', port=5000, debug=True)