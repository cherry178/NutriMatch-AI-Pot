with open('data/stitch_template.html', 'r', encoding='utf-8') as f:
    text = f.read()

import re

# Find major section headers
headers = re.findall(r'<h[1-4][^>]*>(.*?)</h[1-4]>', text, re.IGNORECASE)
print("=== Headings ===")
for h in headers:
    clean_h = re.sub('<[^<]+?>', '', h).strip()
    if clean_h:
        print(" -", clean_h)

# Find tabs
tab_buttons = re.findall(r'<button[^>]*class="[^"]*tab[^"]*"[^>]*>(.*?)</button>', text, re.IGNORECASE | re.DOTALL)
print("\n=== Tabs ===")
for tb in tab_buttons:
    print(" -", re.sub('<[^<]+?>', '', tb).strip())

# Check if tabs have container IDs or data attributes
containers = re.findall(r'id="([^"]+)"', text)
print("\n=== Element IDs ===")
for cid in containers[:25]:
    print(" -", cid)
