import os
import sys
root = '/'.join(os.path.abspath(__file__).split('/')[:-1])
if os.getcwd() != root:
    os.chdir(root)
    sys.path.append(root)

import json
from ove.utils.enhancer import enhance
# from ove.utils.event_handler import enhance


if __name__ == '__main__':
    with open('options.json', 'r') as f:
        options = json.load(f)
    enhance(
        **options
    )
