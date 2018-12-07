from flask import Flask, render_template, jsonify, request
import new_predict

app = Flask(__name__)

@app.route('/')
def send_landing_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_func():
    name = request.form['name']
    msg = request.form['complain']
    predicted_class, confidence = new_predict.predict_text(msg)

    
    return jsonify({
        'name':name,
        'class': predicted_class,
        'confidence': confidence,
        'original':msg,
        'msg':'your comment predicted as {}'.format(predicted_class)
    })

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)