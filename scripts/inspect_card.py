with open('data/stitch_template.html', 'r', encoding='utf-8') as f:
    text = f.read()

import re

# Find the first card inside grid-cols-1 lg:grid-cols-2
m = re.search(r'<div class="grid grid-cols-1 lg:grid-cols-2 gap-space-lg">(.*?)</div>\s*<!-- Latent Vector', text, re.DOTALL)
if m:
    cards_html = m.group(1)
    print("Cards HTML length:", len(cards_html))
    # Print the first 1200 characters of the card
    print(cards_html[:1500])
