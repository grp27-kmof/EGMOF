"""Single-linker structure check shared by generation, screening and verification.

Definitions:
    - Organic fragment: after deleting all metal atoms (and PORMAKE "X" dummies),
      a covalently bonded group of atoms (periodic ASE NeighborList,
      natural_cutoffs, mult=1.15, bothways=True, minimum image) with >= 2 carbon
      atoms or containing a ring. Lone O, OH, H2O and CO2/COO (1 C) are not
      organic fragments; they belong to the metal node.
    - Two fragments are the SAME linker only if their molecular graphs are
      isomorphic (networkx, nodes labelled by element). RDKit SMILES/InChIKey
      are labels only ("unparsed" if bond perception fails).
    - A structure PASSES only if: (a) min interatomic distance >= 1.0 A,
      (b) >= 1 organic fragment, (c) the organic fragments form exactly
      ``expected_unique`` unique graphs (1 for "single"/"single-linker",
      2 for "ligand-linker"), (d) at least one metal atom, (e) every unique
      linker matches the PORMAKE building block(s) named in the CIF name
      (heavy-atom graph, H ignored), (f) every unique linker passes the RDKit
      chemistry rules.
      With expected_unique=None (organic_mode "any") only (a) is checked.

Expected linker (e): PORMAKE joins blocks by deleting each X-X pair and bonding
the two atoms next to them. The carboxylate C(O)O of e.g. a Zn4O node belongs to
the NODE block, so the linker that remains after deleting metals is
"organic block without X" + "the node's organic stub at each connection point".
The reference graph is built exactly that way from the block xyz files.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs

MIN_DIST_THRESHOLD = 1.0
NEIGHBOR_MULT = 1.15
_RDKIT_CHARGES = (0, -1, 1, -2, 2, -3, 3, -4, 4)
_RDKIT_MAX_ITERATIONS = 10_000  # per charge, DetermineBondOrders(maxIterations=...)
_MAX_VALENCE = {"C": 4, "O": 2, "N": 3}
_MAX_STUB_COMBINATIONS = 64
_GLOBAL_NAME_SUFFIX = re.compile(r"__r(\d+)$")  # generate_V1.py --n-valid: <name>__rXX
_REJECTED_PREFIX = re.compile(r"^r\d+_")  # generate_V1.py --n-valid: rejected/rXX_<name>
_EXPECTED_UNIQUE_BY_MODE: dict[str, int | None] = {
    "any": None,
    "single": 1,
    "single-linker": 1,
    "ligand-linker": 2,
}


def expected_unique_for_mode(
    organic_mode: str = "any",
    single_ligand: bool = False,
    ligand_linker: bool = False,
) -> int | None:
    """Number of unique linker graphs required for an organic mode (None = no linker check).

    The legacy flags are resolved exactly like ``run_mof2desc_and_select``: they only
    apply when organic_mode is "any", and ligand_linker wins over single_ligand.
    """
    mode = organic_mode
    if mode == "any":
        if ligand_linker:
            mode = "ligand-linker"
        elif single_ligand:
            mode = "single"
    if mode not in _EXPECTED_UNIQUE_BY_MODE:
        raise ValueError(f"Unknown organic_mode: {organic_mode}")
    return _EXPECTED_UNIQUE_BY_MODE[mode]


def strip_run_tags(name: str) -> str:
    """CIF stem without the generate_V1.py tags (rejected/rXX_ prefix, __rXX suffix)."""
    stem = Path(str(name)).name
    if stem.endswith(".cif"):
        stem = stem[:-4]
    return _GLOBAL_NAME_SUFFIX.sub("", _REJECTED_PREFIX.sub("", stem))


@lru_cache(maxsize=1)
def _metal_node_names() -> frozenset[str]:
    from .desc2mof import __desc2mof_dir__

    return frozenset((Path(__desc2mof_dir__) / "data" / "metal_node.txt").read_text().split())


def topology_node_linker_key(name: str, linker_inchikeys: list[str]) -> tuple | None:
    """(topology, metal node(s), linker InChIKeys) of a MOF; None if a linker is unlabelled."""
    inchikeys = sorted(linker_inchikeys)
    if not inchikeys or "unparsed" in inchikeys:
        return None
    tokens = strip_run_tags(name).split("+")
    metal_nodes = tuple(sorted({tok for tok in tokens[1:] if tok in _metal_node_names()}))
    return tokens[0], metal_nodes, tuple(inchikeys)


def _is_metal(symbol: str) -> bool:
    from pymatgen.core import Element

    return bool(Element(symbol).is_metal)


def _to_atoms(obj: Any) -> Atoms:
    """Convert a CIF path, ASE Atoms, pymatgen Structure or PORMAKE Framework to Atoms."""
    if isinstance(obj, Atoms):
        return obj
    if isinstance(obj, (str, Path)):
        from ase.io import read

        try:
            return read(str(obj))
        except Exception:  # fall back to pymatgen's CIF parser
            from pymatgen.core import Structure
            from pymatgen.io.ase import AseAtomsAdaptor

            return AseAtomsAdaptor.get_atoms(Structure.from_file(str(obj)))
    if hasattr(obj, "atoms") and isinstance(obj.atoms, Atoms):
        return obj.atoms
    try:
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as e:
        raise TypeError(f"unsupported input type: {type(obj).__name__}") from e
    if isinstance(obj, Structure):
        return AseAtomsAdaptor.get_atoms(obj)
    raise TypeError(f"unsupported input type: {type(obj).__name__}")


def _min_distance(atoms: Atoms) -> float:
    if len(atoms) < 2:
        return float("inf")
    dm = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dm, np.inf)
    return float(dm.min())


def _bond_graph(
    organic: Atoms,
) -> tuple[nx.Graph, dict[int, list[tuple[int, np.ndarray]]]]:
    """Periodic covalent bond graph; also returns (j, cell offset) of every bond i->j."""
    graph = nx.Graph()
    symbols = organic.get_chemical_symbols()
    for i, symbol in enumerate(symbols):
        graph.add_node(i, element=symbol)
    bonds: dict[int, list[tuple[int, np.ndarray]]] = {i: [] for i in range(len(organic))}
    if len(organic) == 0:
        return graph, bonds

    neighbor_list = NeighborList(
        natural_cutoffs(organic, mult=NEIGHBOR_MULT),
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    neighbor_list.update(organic)
    for i in range(len(organic)):
        indices, cell_offsets = neighbor_list.get_neighbors(i)
        for j, offset in zip(indices, cell_offsets):
            j = int(j)
            if i == j:
                continue  # bond to its own periodic image
            graph.add_edge(i, j)
            bonds[i].append((j, np.asarray(offset)))
    return graph, bonds


def _unwrapped_positions(
    organic: Atoms, nodes: list[int], bonds: dict[int, list[tuple[int, np.ndarray]]]
) -> dict[int, np.ndarray]:
    """Unwrap one fragment across periodic boundaries (for RDKit only)."""
    positions = organic.positions
    cell = np.asarray(organic.cell)
    unwrapped = {nodes[0]: positions[nodes[0]].copy()}
    queue = deque([nodes[0]])
    while queue:
        i = queue.popleft()
        for j, offset in bonds[i]:
            if j in unwrapped:
                continue
            image = positions[j] + offset @ cell
            unwrapped[j] = unwrapped[i] + (image - positions[i])
            queue.append(j)
    return unwrapped


# ── RDKit: labels + chemistry rules ──────────────────────────────────────────


def _rdkit_mol(symbols: list[str], coords: list[np.ndarray]):
    """Sanitized RDKit molecule from 3D coordinates, or None if bond perception fails."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdDetermineBonds

        RDLogger.DisableLog("rdApp.*")
        xyz = f"{len(symbols)}\nfragment\n" + "".join(
            f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n" for s, p in zip(symbols, coords)
        )
        mol = Chem.MolFromXYZBlock(xyz)
        if mol is None:
            return None
        rdDetermineBonds.DetermineConnectivity(mol)
        for charge in _RDKIT_CHARGES:
            candidate = Chem.RWMol(mol)
            try:
                # capped: the bond-order search is exponential on distorted fragments
                # (minutes per CIF); those fragments do not resolve with more iterations
                rdDetermineBonds.DetermineBondOrders(
                    candidate, charge=charge, maxIterations=_RDKIT_MAX_ITERATIONS
                )
                Chem.SanitizeMol(candidate)
            except Exception:  # try the next total charge
                continue
            return candidate.GetMol()
    except Exception:  # RDKit missing or crashed
        return None
    return None


def _rdkit_labels(mol) -> tuple[str, str]:
    """(SMILES, InChIKey) of a sanitized molecule, or ("unparsed", "unparsed")."""
    if mol is None:
        return "unparsed", "unparsed"
    try:
        from rdkit import Chem

        smiles = Chem.MolToSmiles(mol, canonical=True) or "unparsed"
        inchikey = Chem.MolToInchiKey(mol) or "unparsed"
        return smiles, inchikey
    except Exception:  # labels are optional
        return "unparsed", "unparsed"


def _chemistry_problem(mol) -> str | None:
    """First violated chemistry rule of one linker, or None if it looks sane."""
    if mol is None:
        return "RDKit bond perception / sanitization failed"
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) == 3 and any(mol.GetAtomWithIdx(i).GetSymbol() == "C" for i in ring):
            return "3-membered ring containing C"
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        limit = _MAX_VALENCE.get(symbol)
        valence = atom.GetTotalValence()
        if limit is not None and valence > limit:
            return f"{symbol}{atom.GetIdx()} valence {valence} > {limit}"
        if symbol == "C":
            n_oxygen = sum(nb.GetSymbol() == "O" for nb in atom.GetNeighbors())
            if n_oxygen > 2:
                return f"C{atom.GetIdx()} bonded to {n_oxygen} O"
    return None


# ── expected linker from the PORMAKE building blocks in the MOF name ─────────


def _parse_mof_name(name: str) -> tuple[list[str], int | None]:
    """Building-block tokens of a CIF name and the --n-valid round tag (if any)."""
    stem = Path(str(name)).name
    if stem.endswith(".cif"):
        stem = stem[:-4]
    stem = _REJECTED_PREFIX.sub("", stem)
    match = _GLOBAL_NAME_SUFFIX.search(stem)
    round_idx = int(match.group(1)) if match else None
    stem = _GLOBAL_NAME_SUFFIX.sub("", stem)
    return stem.split("+")[1:], round_idx


def default_new_bb_dir(cif_path: str | Path) -> Path:
    """new_bbs/ folder that holds the Gen_* blocks of a CIF written by generate_V1.py.

    <out>/V1_x/cifs/<name>.cif              -> <out>/V1_x/new_bbs
    <out>/V1_x/cifs/<name>__rXX.cif         -> <out>/V1_x/rounds/round_XX/new_bbs
    <out>/V1_x/rounds/round_XX/cifs/<name>  -> <out>/V1_x/rounds/round_XX/new_bbs
    """
    path = Path(cif_path)
    _, round_idx = _parse_mof_name(path.name)
    base = path.parent.parent
    if round_idx is not None:
        return base / "rounds" / f"round_{round_idx:02d}" / "new_bbs"
    return base / "new_bbs"


def _load_block(name: str, new_bb_dir: str | None):
    """PORMAKE BuildingBlock: database for N*/E*, new_bb_dir for Gen_*."""
    import pormake as pm
    from pormake.building_block import BuildingBlock

    if name.startswith("Gen_"):
        if new_bb_dir is None:
            raise FileNotFoundError(f"{name}: new_bbs folder unknown")
        path = Path(new_bb_dir) / f"{name}.xyz"
    else:
        path = Path(pm.Database().bb_dir) / f"{name}.xyz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return BuildingBlock(str(path))


def _block_graph(bb) -> tuple[nx.Graph, list[int]]:
    """Heavy-atom graph of a block (H and X removed) and the atom bonded to each X."""
    symbols = bb.atoms.get_chemical_symbols()
    full = nx.Graph()
    full.add_nodes_from((i, {"element": s}) for i, s in enumerate(symbols))
    full.add_edges_from((int(i), int(j)) for i, j in bb.bonds)
    anchors = []
    for x in bb.connection_point_indices:
        neighbors = [n for n in full.neighbors(int(x)) if symbols[n] != "X"]
        if neighbors:
            anchors.append(neighbors[0])
    heavy = full.subgraph([i for i, s in enumerate(symbols) if s not in ("H", "X")]).copy()
    return heavy, anchors


def _stub_match(left: dict, right: dict) -> bool:
    return left["element"] == right["element"] and left.get("anchor") == right.get("anchor")


def _node_stubs(bb) -> list[nx.Graph]:
    """Organic piece a metal-containing block adds at each connection point.

    E.g. C(O)O for a carboxylate node; an empty graph when X sits on a metal.
    The stub graph stores its attachment atom in graph.graph["anchor"].
    """
    heavy, anchors = _block_graph(bb)
    organic = heavy.subgraph(
        [n for n, d in heavy.nodes(data=True) if not _is_metal(d["element"])]
    )
    stubs = []
    for anchor in anchors:
        if _is_metal(heavy.nodes[anchor]["element"]):
            stubs.append(nx.Graph())
            continue
        stub = organic.subgraph(nx.node_connected_component(organic, anchor)).copy()
        for n in stub.nodes:
            stub.nodes[n]["anchor"] = n == anchor
        stub.graph["anchor"] = anchor
        stubs.append(stub)
    return stubs


def _attach_stubs(bb, stub_options: list[nx.Graph]) -> list[nx.Graph]:
    """Expected linker graphs of one organic block for every stub assignment."""
    heavy, anchors = _block_graph(bb)
    combos = list(product(range(len(stub_options)), repeat=len(anchors)))
    if len(combos) > _MAX_STUB_COMBINATIONS:  # many stub types: same stub at every X
        combos = [(k,) * len(anchors) for k in range(len(stub_options))]
    graphs = []
    for combo in combos:
        graph = nx.Graph()
        graph.add_nodes_from((("b", n), {"element": d["element"]}) for n, d in heavy.nodes(data=True))
        graph.add_edges_from((("b", u), ("b", v)) for u, v in heavy.edges)
        for x_idx, (anchor, k) in enumerate(zip(anchors, combo)):
            stub = stub_options[k]
            graph.add_nodes_from(
                (("s", x_idx, n), {"element": d["element"]}) for n, d in stub.nodes(data=True)
            )
            graph.add_edges_from((("s", x_idx, u), ("s", x_idx, v)) for u, v in stub.edges)
            if "anchor" in stub.graph:
                graph.add_edge(("b", anchor), ("s", x_idx, stub.graph["anchor"]))
        graphs.append(graph)
    return graphs


@lru_cache(maxsize=512)
def _expected_linker_graphs_cached(blocks: tuple[str, ...], new_bb_dir: str | None) -> tuple:
    loaded = [_load_block(name, new_bb_dir) for name in dict.fromkeys(blocks) if name != "E0"]
    has_metal = [
        any(_is_metal(s) for s in bb.atoms.get_chemical_symbols() if s != "X") for bb in loaded
    ]
    stub_options: list[nx.Graph] = []
    for bb, metal in zip(loaded, has_metal):
        if not metal:
            continue
        for stub in _node_stubs(bb):
            if not any(nx.is_isomorphic(stub, s, node_match=_stub_match) for s in stub_options):
                stub_options.append(stub)
    if not stub_options:
        stub_options = [nx.Graph()]
    references = []
    for bb, metal in zip(loaded, has_metal):
        if not metal:
            references.extend(_attach_stubs(bb, stub_options))
    return tuple(references)


def expected_linker_graphs(mof_name: str, new_bb_dir: str | Path | None = None) -> list[nx.Graph]:
    """Heavy-atom graphs the organic blocks of ``mof_name`` must have in the built MOF.

    Raises:
        FileNotFoundError: a block xyz (database or new_bbs/) is missing.
    """
    blocks, _ = _parse_mof_name(mof_name)
    return list(
        _expected_linker_graphs_cached(
            tuple(blocks), str(new_bb_dir) if new_bb_dir is not None else None
        )
    )


def _heavy(graph: nx.Graph) -> nx.Graph:
    return graph.subgraph([n for n, d in graph.nodes(data=True) if d["element"] != "H"])


def _same_heavy_linker(extracted: nx.Graph, reference: nx.Graph) -> bool:
    """Same heavy-atom formula and isomorphic heavy-atom graph."""
    formula = Counter(d["element"] for _, d in extracted.nodes(data=True))
    if formula != Counter(d["element"] for _, d in reference.nodes(data=True)):
        return False
    return nx.is_isomorphic(extracted, reference, node_match=_element_match)


def _formula(symbols: list[str]) -> str:
    counts = Counter(symbols)
    order = (["C", "H"] if "C" in counts else []) + sorted(
        s for s in counts if not ("C" in counts and s in ("C", "H"))
    )
    return "".join(s if counts[s] == 1 else f"{s}{counts[s]}" for s in order if s in counts)


def _element_match(left: dict, right: dict) -> bool:
    return left["element"] == right["element"]


def check_structure(
    atoms_or_structure: Any,
    expected_unique: int | None = 1,
    min_dist_threshold: float = MIN_DIST_THRESHOLD,
    mof_name: str | None = None,
    new_bb_dir: str | Path | None = None,
) -> dict:
    """Check that a MOF contains exactly ``expected_unique`` kinds of sane, undistorted linker.

    Args:
        atoms_or_structure: CIF path, ASE Atoms, pymatgen Structure or PORMAKE Framework.
        expected_unique: Required number of unique linker graphs (see
            ``expected_unique_for_mode``). None = only the overlap check is applied.
        min_dist_threshold: Minimum allowed interatomic distance in Angstrom.
        mof_name: PORMAKE name (topology+N..+E..) whose blocks give the expected
            linker. Default: the file name when a CIF path is given.
        new_bb_dir: Folder with the Gen_* block xyz files. Default for a CIF path:
            ``default_new_bb_dir(path)``.

    Returns:
        Dict with keys passed, reason ("ok" | "overlap" | "no_organic" | "no_metal" |
        "multiple_linkers" | "too_few_linkers" | "no_reference" | "linker_distorted" |
        "bad_chemistry" | "error: ..."), min_dist, n_fragments, n_unique_graphs,
        linker_smiles, linker_inchikeys, linker_formula, linker_reference,
        chemistry_issue. Never raises.
    """
    result: dict = {
        "passed": False,
        "reason": "error: not checked",
        "min_dist": float("nan"),
        "n_fragments": 0,
        "n_unique_graphs": 0,
        "linker_smiles": [],
        "linker_inchikeys": [],
        "linker_formula": "",
        "linker_reference": "",
        "chemistry_issue": "",
    }
    try:
        if isinstance(atoms_or_structure, (str, Path)):
            if mof_name is None:
                mof_name = Path(atoms_or_structure).name
            if new_bb_dir is None:
                new_bb_dir = default_new_bb_dir(atoms_or_structure)

        atoms = _to_atoms(atoms_or_structure)
        atoms = atoms[[i for i, s in enumerate(atoms.get_chemical_symbols()) if s != "X"]]
        symbols = atoms.get_chemical_symbols()
        metal_flags = [_is_metal(s) for s in symbols]
        n_metal = sum(metal_flags)

        min_dist = _min_distance(atoms)
        result["min_dist"] = min_dist

        if expected_unique is None:  # organic_mode "any": overlap check only
            result["passed"] = bool(min_dist >= min_dist_threshold)
            result["reason"] = "ok" if result["passed"] else "overlap"
            return result

        organic = atoms[[i for i, is_metal in enumerate(metal_flags) if not is_metal]]
        graph, bonds = _bond_graph(organic)
        organic_symbols = organic.get_chemical_symbols()

        # unique graphs: fragments are grouped by WL hash (different hash = different
        # linker); isomorphism is only tested against representatives with the same hash
        buckets: dict[str, list[tuple[nx.Graph, list[int]]]] = {}
        n_fragments = 0
        for component in nx.connected_components(graph):
            nodes = sorted(component)
            fragment = graph.subgraph(nodes).copy()
            n_carbon = sum(organic_symbols[i] == "C" for i in nodes)
            if n_carbon < 2 and not nx.cycle_basis(fragment):
                continue  # O, OH, H2O, CO2/COO -> part of the metal node
            n_fragments += 1
            wl_hash = nx.weisfeiler_lehman_graph_hash(fragment, node_attr="element")
            same_hash = buckets.setdefault(wl_hash, [])
            if not any(
                nx.is_isomorphic(rep, fragment, node_match=_element_match)
                for rep, _ in same_hash
            ):
                same_hash.append((fragment, nodes))

        representatives = [rep for reps in buckets.values() for rep in reps]
        result["n_fragments"] = n_fragments
        result["n_unique_graphs"] = len(representatives)

        # labels + chemistry rules (RDKit, sanitize must succeed)
        formulas = []
        chemistry_issues = []
        for _, nodes in representatives:
            unwrapped = _unwrapped_positions(organic, nodes, bonds)
            frag_symbols = [organic_symbols[i] for i in nodes]
            mol = _rdkit_mol(frag_symbols, [unwrapped[i] for i in nodes])
            smiles, inchikey = _rdkit_labels(mol)
            result["linker_smiles"].append(smiles)
            result["linker_inchikeys"].append(inchikey)
            formulas.append(_formula(frag_symbols))
            problem = _chemistry_problem(mol)
            if problem is not None:
                chemistry_issues.append(f"{_formula(frag_symbols)}: {problem}")
        result["linker_formula"] = ";".join(formulas)
        result["chemistry_issue"] = "; ".join(chemistry_issues)

        # expected-linker match against the building blocks named in the CIF
        distorted = False
        no_reference = False
        if mof_name is None:
            no_reference = True
            result["linker_reference"] = "no MOF name"
        else:
            try:
                references = expected_linker_graphs(mof_name, new_bb_dir)
            except Exception as e:  # missing xyz / unreadable block
                references = []
                result["linker_reference"] = f"{type(e).__name__}: {e}"[:200]
            if not references:
                no_reference = True
                result["linker_reference"] = result["linker_reference"] or "no organic block"
            else:
                unmatched = [
                    _formula([d["element"] for _, d in rep.nodes(data=True)])
                    for rep, _ in representatives
                    if not any(_same_heavy_linker(_heavy(rep), ref) for ref in references)
                ]
                distorted = bool(unmatched)
                result["linker_reference"] = (
                    f"mismatch: {';'.join(unmatched)}" if unmatched else "matched"
                )

        if not min_dist >= min_dist_threshold:
            result["reason"] = "overlap"
        elif n_fragments == 0:
            result["reason"] = "no_organic"
        elif len(representatives) > expected_unique:
            result["reason"] = "multiple_linkers"
        elif len(representatives) < expected_unique:
            result["reason"] = "too_few_linkers"
        elif n_metal == 0:
            result["reason"] = "no_metal"
        elif no_reference:
            result["reason"] = "no_reference"
        elif distorted:
            result["reason"] = "linker_distorted"
        elif chemistry_issues:
            result["reason"] = "bad_chemistry"
        else:
            result["reason"] = "ok"
            result["passed"] = True
    except Exception as e:  # check_structure must never raise
        result["passed"] = False
        result["reason"] = f"error: {type(e).__name__}: {e}"[:200]
    return result
