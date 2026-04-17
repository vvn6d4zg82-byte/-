import tarfile
import os
import sys

print('Opening tar...', file=sys.stderr)
tar = tarfile.open('tasks-vision.tgz', 'r:gz')
print('Extracting...', file=sys.stderr)
tar.extractall('.')
print('Done!', file=sys.stderr)
tar.close()

for root, dirs, files in os.walk('.'):
    for f in files:
        if 'hand' in f.lower() and '.task' in f:
            print(f'Found: {os.path.join(root, f)}')