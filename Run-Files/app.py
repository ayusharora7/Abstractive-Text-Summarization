
from flask import Flask, render_template, jsonify, request
from flask_wtf import FlaskForm
from flask_pagedown import PageDown
from flask_pagedown.fields import PageDownField
from wtforms.fields import SubmitField
import requests
import json
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
pagedown = PageDown(app)


class PageDownFormExample(FlaskForm):
    pagedown = PageDownField('Type the text you want to translate and click "Translate".')
    submit = SubmitField('Translate')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = PageDownFormExample()
    text = None
    if form.validate_on_submit():
        source = form.pagedown.data.lower()
        source = re.sub(r"([?.!,:;¿])", r" \1 ", source)
        source = re.sub(r'[" "]+', " ", source)
        url = "http://127.0.0.1:5000/"
        headers = {"Content-Type": "application/json"}
        data = [{"src": source, "id": 100}]
        response = requests.post(url, json=data, headers=headers)
        translation = response.text
        jsn = json.loads(translation)
        text = jsn[0][0]['tgt']
        text = re.sub(r" ([?.!,:،؛؟¿])", r"\1", text)
    return render_template('index.html', form=form, language=language, text=text)




if __name__ == '__main__':
    app.run(debug=True)
