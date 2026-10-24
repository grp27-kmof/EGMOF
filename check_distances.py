import glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
from pymatgen.core import Structure

for f in sorted(glob.glob("test_output/cifs/*.cif")):
    s = Structure.from_file(f)
    d = s.distance_matrix
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    i, j = np.unravel_index(d.argmin(), d.shape)
    pair = f"{s[i].specie}-{s[j].specie}"
    flag = "BROKEN" if dmin < 0.8 else ("CHECK" if dmin < 0.9 else "OK")
    print(f"{f.split('/')[-1]:30s} min_dist={dmin:.2f} Å ({pair:6s})  {flag}")
