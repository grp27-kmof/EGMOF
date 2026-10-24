import glob, warnings
warnings.filterwarnings("ignore")
from pymatgen.core import Structure

for f in sorted(glob.glob("test_output/cifs/*.cif")):
    try:
        s = Structure.from_file(f)
        metals = sorted({el.symbol for el in s.composition.elements if el.is_metal})
        print(f"{f.split('/')[-1]:30s} atoms={len(s):5d}  "
              f"metals={','.join(metals) or 'none':12s}  "
              f"a,b,c={s.lattice.a:.1f},{s.lattice.b:.1f},{s.lattice.c:.1f}  "
              f"formula={s.composition.reduced_formula}")
    except Exception as e:
        print(f"{f}: FAILED to read ({e})")
