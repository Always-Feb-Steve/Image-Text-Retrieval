# Flask-based image search engine web interface
from flask import Flask, redirect, url_for, request
import json
import shutil
import os

app = Flask(__name__)

# ── Paths (update these to match your environment) ──────────────────────────
input_filename = "./meta_data/output.json"
pic_folder     = "./pics/"
static_folder  = "./static/"
# ─────────────────────────────────────────────────────────────────────────────

# Load the inverted index
with open(input_filename, "rb") as fp:
    buffer = fp.read()
dict_key_2_pic = json.loads(str(buffer, encoding="UTF-8"))
for key in dict_key_2_pic:
    dict_key_2_pic[key] = [str(v) for v in dict_key_2_pic[key]]

os.makedirs(static_folder, exist_ok=True)


@app.route('/success/<tag>.html')
def success(tag):
    """Display images matching the searched tag."""
    html = "<p align=center>Search Results<br /><br />"
    pic_list = []

    for pic in dict_key_2_pic[tag]:
        pic = str(pic)
        if pic.startswith('['):
            pic = pic.split('\'')[1]

        src = os.path.join(pic_folder, pic + ".jpg")
        dst = os.path.join(static_folder, pic + ".jpg")

        if os.path.exists(src):
            shutil.copyfile(src, dst)
            print("Serving:", pic)
            html += '<img src="/static/{}.jpg" alt="{}" />'.format(pic, tag)
            pic_list.append(pic)

    print("Results:", pic_list)
    return html


@app.route('/fail/<tag1>/')
def fail(tag1):
    """Return a message when no results are found."""
    return 'Sorry, no results found for keyword: {}'.format(tag1)


@app.route('/SearchEngine/', methods=['GET'])
def SearchEngine():
    """Main search endpoint — accepts a tag via GET parameter."""
    content = request.args.get('tag')
    if content in dict_key_2_pic:
        return redirect(url_for('success', tag=content))
    else:
        return redirect(url_for('fail', tag1=content))


if __name__ == '__main__':
    app.run(debug=False)  # Listens continuously for incoming requests
