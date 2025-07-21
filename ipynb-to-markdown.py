from nbconvert import MarkdownExporter
import nbformat
from pathlib import Path

# list of notebooks
notebook_list = [
    "notebook/lab-prompting.ipynb",
    "notebook/lab-Scipy-Optimize.ipynb",
    "notebook/lab-peft-from-scratch.ipynb",
]

# save the file path
output = "docs/notebook"
for path in notebook_list:
    path = Path(path)
    with open(path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    # Convert to Markdown
    exporter = MarkdownExporter()
    body, _ = exporter.from_notebook_node(notebook)
    # save the .md file
    with open(output + "/" + path.stem + ".md", "w") as f:
        f.write(body)
    print(f"Converted {path.name} to Markdown and saved as {output}/{path.stem}.md")