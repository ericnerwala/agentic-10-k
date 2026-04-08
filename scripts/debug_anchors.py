from extract import extract_10k_text
import re

html = extract_10k_text('data/folder_1/folder_1/0000004281-20-000038.txt')
print('HTML length:', len(html))

# Find all <a id="..."> anchors
anchor_id_pattern = re.compile(r'<a\s[^>]*\bid="([^"]+)"[^>]*>', re.IGNORECASE)
matches = list(anchor_id_pattern.finditer(html))
print('Total anchors:', len(matches))

for m in matches[:10]:
    offset = m.start()
    lookahead = html[offset: offset+400]
    plain = re.sub(r'<[^>]+>', ' ', lookahead)
    plain = re.sub(r'\s+', ' ', plain).strip()[:150]
    print(f'  id={m.group(1)[:30]}...')
    print(f'  text={repr(plain[:120])}')
    print()

# Also check if item2 appears as a content section with a nearby anchor
print('\n--- Searching for Item 2 bold headers ---')
item2_hits = [(m.start(), m.group(0)) for m in re.finditer(r'Item\s*2[\.\s]', html, re.IGNORECASE)]
print(f'Found {len(item2_hits)} Item 2 references')
for offset, hit in item2_hits[:5]:
    # Look for nearest preceding <a id="...">
    preceding = html[max(0, offset-500):offset]
    anchor_matches = list(re.finditer(r'<a\s[^>]*\bid="([^"]+)"[^>]*>', preceding, re.IGNORECASE))
    nearest_anchor = anchor_matches[-1].group(1) if anchor_matches else 'NONE'
    context = re.sub(r'<[^>]+>', ' ', html[offset:offset+200])
    context = re.sub(r'\s+', ' ', context).strip()[:100]
    print(f'  offset={offset} nearest_anchor={nearest_anchor[:20]}... context={repr(context)}')
