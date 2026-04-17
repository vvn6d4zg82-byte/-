import gzip
import shutil

with gzip.open('hand_landmarker.task', 'rb') as f_in:
    with open('hand_landmarker_unzipped.task', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print("解压完成!")

import zipfile
try:
    z = zipfile.ZipFile('hand_landmarker_unzipped.task')
    print("ZIP有效! 内容:")
    for n in z.namelist():
        print(f"  {n}")
    z.close()
except Exception as e:
    print(f"不是ZIP: {e}")
    with open('hand_landmarker_unzipped.task', 'rb') as f:
        print(f"文件头: {f.read(20)}")