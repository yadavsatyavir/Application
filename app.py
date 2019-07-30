from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template(
            "./home.html"  # name of template
            )

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='0.0.0.0',port=80)
