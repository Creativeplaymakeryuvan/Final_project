import re
import json

# Extracted text as a list of lines
extracted_text = [
    "LOREMIPSUM",
    "Lorem Ipsum",
    "112233 Lorem Street, Lorem, Dolor, AMET",
    "1x Lorem",
    "1x Ipsum Dolor",
    "1x Sit Amet",
    "2x Consectetuer",
    "1x Adipiscing Elit",
    "1x Sed Diam",
    "$ 25.00",
    "$ 15.00",
    "$ 10.00",
    "$ 20.00",
    "$ 10.00",
    "$ 15.00",
    "TOTAL AMOUNT",
    "CASH",
    "CHANGE",
    "$ 95.00",
    "$100.00",
    "$",
    "5.00",
    "designed by ® freepik"
]

# Initialize variables to store structured data
store_name = ""
cost = 0.0
item_list = []

# Regular expression patterns for matching item lines and dollar amounts
item_pattern = re.compile(r"(\d+x)\s+(.+)")
cost_pattern = re.compile(r"\$\s?(\d+\.\d{2})")

# Parse the text
for line in extracted_text:
    line = line.strip()

    # Check for the store name
    if not store_name and re.search(r"\d{5}", line):  # Assuming the store name is before the address
        store_name = extracted_text[0]  # First line is often the store name

    # Check for items with quantity (e.g., "1x Lorem Ipsum")
    item_match = item_pattern.match(line)
    if item_match:
        quantity, item_name = item_match.groups()
        item_list.append({
            "quantity": quantity,
            "item": item_name.strip()
        })

    # Check for costs (e.g., "$ 25.00")
    cost_match = cost_pattern.match(line)
    if cost_match:
        cost = float(cost_match.group(1))

# Compile the results
result = {
    "store_name": store_name,
    "total_cost": cost,
    "items": item_list
}

# Print the structured data
print(json.dumps(result, indent=4))
