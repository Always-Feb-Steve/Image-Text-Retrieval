# Flask-based image search engine web interface
from flask import Flask, request, send_from_directory
from markupsafe import escape

from search import Searcher, pic_folder

app = Flask(__name__)
searcher = Searcher()  # loads the model and embeds all images once

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Image-Text Retrieval</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; }}
  form {{ display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }}
  input[type=text] {{ flex: 1; min-width: 200px; padding: 8px; font-size: 16px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }}
  .grid figure {{ margin: 0; }}
  .grid img {{ width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 6px; }}
  figcaption {{ font-size: 12px; color: #666; }}
</style></head>
<body>
<h2>Image-Text Retrieval</h2>
<form action="/SearchEngine/" method="get">
  <input type="text" name="tag" value="{query}" placeholder="e.g. a dog running on the beach">
  <label><input type="radio" name="mode" value="semantic" {semantic_checked}> model</label>
  <label><input type="radio" name="mode" value="keyword" {keyword_checked}> keyword</label>
  <button type="submit">Search</button>
</form>
{body}
</body></html>"""


def render(query="", mode="semantic", body=""):
    return PAGE.format(query=escape(query), body=body,
                       semantic_checked="checked" if mode != "keyword" else "",
                       keyword_checked="checked" if mode == "keyword" else "")


@app.route('/')
def index():
    return render()


@app.route('/images/<pic_id>.jpg')
def image(pic_id):
    """Serve a Flickr8k image straight from the dataset folder."""
    return send_from_directory(pic_folder, pic_id + ".jpg")


@app.route('/SearchEngine/', methods=['GET'])
def SearchEngine():
    """Main search endpoint — accepts a free-text query via the `tag` GET parameter."""
    query = request.args.get('tag', '').strip()
    mode = request.args.get('mode', 'semantic')
    if not query:
        return render(mode=mode)

    results = searcher.keyword(query, 24) if mode == "keyword" else searcher.semantic(query, 24)
    if not results:
        return render(query, mode, "<p>Sorry, no results found for: {}</p>".format(escape(query)))

    cells = "".join(
        '<figure><img src="/images/{0}.jpg" alt="{1}"><figcaption>{2}</figcaption></figure>'.format(
            escape(pic_id), escape(query), "similarity {:.3f}".format(score) if score is not None else "")
        for pic_id, score in results)
    return render(query, mode, '<div class="grid">{}</div>'.format(cells))


if __name__ == '__main__':
    app.run(debug=False)  # http://localhost:5000/
