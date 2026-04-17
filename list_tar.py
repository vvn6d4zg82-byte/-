import tarfile
import os

tar = tarfile.open('tasks-vision.tgz', 'r:gz')
print('All files in package:')
for m in tar.getmembers():
    print(f'  {m.name}')
tar.close()