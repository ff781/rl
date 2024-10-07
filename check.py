import os
import numpy, h5py
from tqdm import tqdm

total_files = sum([len(files) for _, _, files in os.walk('.')])
pbar = tqdm(total=total_files, desc="Checking files")

running_min = {}
running_max = {}

"main/walker_walk/random/64px"
"""
>>> a['height'].shape
(501,)
>>> a['velocity'].shape
(501, 9)
>>> a['orientations'].shape
(501, 14)
"""
"""
action: min = -1.0, max = 1.0
reward: min = 0.0, max = 2.0
image: min = 0, max = 255
discount: min = 1.0, max = 1.0
orientations: min = -1.0, max = 1.0
height: min = 0.037800513207912445, max = 1.3525866270065308
velocity: min = -60.131019592285156, max = 55.77828598022461
position: min = -0.8250628113746643, max = 0.7039156556129456
"""

for a, b, c in os.walk('.'):
    for f in c:
        if f.endswith(".npz"):
            f = os.path.join(a, f)
            assert "64px" in f
            d = numpy.load(f)
            g = [
                numpy.nonzero(d['is_first'])[0].tolist() == [0],
                numpy.nonzero(d['is_last'])[0].tolist() == [500],
                numpy.nonzero(d['is_terminal'])[0].tolist() == []
            ]
            print(f, d['reward'].max())
            # for k,v in d.items():
            #     if k not in ['is_first', 'is_last', 'is_terminal', "image"]:
            #         if k not in running_min or v.min() < running_min[k]:
            #             running_min[k] = v.min()
            #         if k not in running_max or v.max() > running_max[k]:
            #             running_max[k] = v.max()
            if not all(g):
                print(f, g)
        elif f.endswith(".hdf5"):
            f = os.path.join(a, f)
            assert "84px" in f
            d = h5py.File(f)
            g = [
                numpy.nonzero(d['step_type'][:]==0)[0].tolist() == list(range(0,d['step_type'][:].shape[0],501)),
                numpy.nonzero(d['step_type'][:]==1)[0].tolist() == [i for i in range(d['step_type'][:].shape[0]) if i % 501 != 0 and i % 501 != 500],
                numpy.nonzero(d['step_type'][:]==2)[0].tolist() == list(range(500,d['step_type'][:].shape[0],501)),
                (d['discount'][:]==1).all()
            ]
            # for k,v in d.items():
            #     if k not in ['step_type', 'discount', 'observation']:
            #         if k not in running_min or v[:].min() < running_min[k]:
            #             running_min[k] = v[:].min()
            #         if k not in running_max or v[:].max() > running_max[k]:
            #             running_max[k] = v[:].max()
            if not all(g):
                print(f, g)
        else:
            assert False, f
        if "height" in d:
            print(f)
        pbar.update(1)

pbar.close()

print("\nRunning min and max for each key:")
for k in running_min.keys():
    print(f"{k}: min = {running_min[k]}, max = {running_max[k]}")