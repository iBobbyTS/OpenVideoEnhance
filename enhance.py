import json
from ove.utils.enhancer import enhance

# Args
with open('options.json', 'r') as f:
    options = json.load(f)

if __name__ == '__main__':
    enhance(
        **options
    )
