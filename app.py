#!flask/bin/python
from flask import Flask, request
from captcha_solver.solver import solver
import os

UPLOAD_FOLDER = './upload'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

solver = solver()

@app.route('/', methods=['POST'])
def upload_file():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return {
                    'data': 'wrong data'
                }
            file = request.files['file']
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            image_text = solver.solve(path)

            os.remove(path)
            return {
                'data': image_text
            }
    except:
        os.remove(path)
        return {
            'data': 'can not recognize'
        }


if __name__ == '__main__':
    app.run(debug=True)
