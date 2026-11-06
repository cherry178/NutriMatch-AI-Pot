with open('data/stitch_template.html', 'r', encoding='utf-8') as f:
    text = f.read()

import re

m = re.search(r'id=["\']tab-panel-explorer["\'][^>]*>(.*?)id=["\']tab-panel-learning-loop["\']', text, re.DOTALL)
if m:
    content = m.group(1)
    print("Explorer panel length:", len(content))
    grids = re.findall(r'<div[^>]*class=["\'][^"\']*grid[^"\']*["\'][^>]*>', content)
    for g in grids:
        print("Grid tag:", g)
else:
    print("Explorer panel not found with regex")
