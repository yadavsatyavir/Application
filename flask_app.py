from flask import Flask
app = Flask(__name__)

@app.route('/')
def homepage():
    return "<h1>Hello heroku</h1><p>It is currently -UPDATED </p>"

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0',port=80)
