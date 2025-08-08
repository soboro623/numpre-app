from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# モデルをロード
model = load_model('mnist_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    # 画像を読み込み（グレースケール変換 → リサイズ）
    img = Image.open(file).convert('L').resize((28, 28))
    
    # numpy配列に変換 → 色を反転（白地に黒文字なら不要） → 正規化
    img = np.array(img)
    img = 255 - img  # 黒地に白文字ならこのまま。白地に黒文字なら消してOK
    img = img / 255.0
    
    # 形状を (1, 28, 28, 1) に変換
    img = img.reshape(1, 28, 28, 1)

    # モデルで予測
    prediction = model.predict(img)
    predicted_label = int(np.argmax(prediction))

    return f'予測された数字は: {predicted_label} です'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
