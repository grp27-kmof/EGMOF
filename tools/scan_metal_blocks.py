#!/usr/bin/env python3
import json
from collections import deque
from pathlib import Path

import pormake as pm
import networkx as nx
from ase.neighborlist import NeighborList, natural_cutoffs
from pymatgen.core import Element
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "src" / "egmof" / "desc2mof" / "data"
OUTPUT_PATH = DATA_DIR / "metal_block_organics.json"


def _bonded_fragments(atoms):
    symbols = atoms.get_chemical_symbols()
    keep = [
        i
        for i, symbol in enumerate(symbols)
        if symbol != "X" and not Element(symbol).is_metal
    ]

    full_neighbor_list = NeighborList(
        natural_cutoffs(atoms, mult=1.15),
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    full_neighbor_list.update(atoms)
    atoms_bonded_to_x = set()
    for i in range(len(atoms)):
        neighbors, _ = full_neighbor_list.get_neighbors(i)
        if symbols[i] == "X":
            atoms_bonded_to_x.update(int(j) for j in neighbors)
        elif any(symbols[int(j)] == "X" for j in neighbors):
            atoms_bonded_to_x.add(i)

    organic = atoms[keep]
    if len(organic) == 0:
        return []

    neighbor_list = NeighborList(
        natural_cutoffs(organic, mult=1.15),
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    neighbor_list.update(organic)
    adj = [[] for _ in organic]
    for i in range(len(organic)):
        indices, _ = neighbor_list.get_neighbors(i)
        adj[i].extend(indices)

    fragments = []
    seen = set()
    for root in range(len(organic)):
        if root in seen:
            continue
        queue = deque([root])
        component = []
        seen.add(root)
        while queue:
            i = queue.popleft()
            component.append(i)
            for j in adj[i]:
                if j not in seen:
                    seen.add(j)
                    queue.append(j)
        original_indices = {keep[i] for i in component}
        fragments.append((organic[component], bool(original_indices & atoms_bonded_to_x)))
    return fragments


def _fragment_molecule(fragment):
    xyz = (
        f"{len(fragment)}\nfragment\n"
        + "".join(
            f"{symbol} {position[0]:.8f} {position[1]:.8f} {position[2]:.8f}\n"
            for symbol, position in zip(fragment.get_chemical_symbols(), fragment.positions)
        )
    )
    mol = Chem.MolFromXYZBlock(xyz)
    if mol is None:
        return None
    for charge in (0, -1, 1, -2, 2, -3, 3):
        candidate = Chem.RWMol(mol)
        try:
            rdDetermineBonds.DetermineBonds(candidate, charge=charge)
            return candidate
        except Exception:
            continue
    return None


def _fragment_graph(fragment, adjacency):
    graph = nx.Graph()
    for i, symbol in enumerate(fragment.get_chemical_symbols()):
        graph.add_node(i, element=symbol)
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            if i < j:
                graph.add_edge(i, j)
    return graph


def _graph_key(graph):
    return nx.weisfeiler_lehman_graph_hash(
        graph,
        node_attr="element",
        edge_attr=None,
    )


def _rings_present(graph):
    return len(list(nx.simple_cycles(graph.to_directed()))) > 0


def _fragment_molecule_graph(fragment):
    neighbor_list = NeighborList(
        natural_cutoffs(fragment, mult=1.15),
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    neighbor_list.update(fragment)
    adjacency = [[] for _ in fragment]
    for i in range(len(fragment)):
        indices, _ = neighbor_list.get_neighbors(i)
        adjacency[i].extend(indices)
    return _fragment_graph(fragment, adjacency)


def scan_block(block_name):
    try:
        atoms = pm.Database().get_bb(block_name).atoms
    except Exception:
        return {
            "has_organic": False,
            "inchikeys": [],
            "smiles": [],
            "graph_keys": [],
            "linker_stubs": [],
        }
    smiles = set()
    inchikeys = set()
    graph_keys = set()
    linker_stubs = set()

    for fragment, touches_x in _bonded_fragments(atoms):
        symbols = fragment.get_chemical_symbols()
        carbon_count = sum(symbol == "C" for symbol in symbols)

        mol = _fragment_molecule(fragment)
        graph = nx.Graph()
        for i, symbol in enumerate(symbols):
            graph.add_node(i, element=symbol)
        try:
            graph = _fragment_molecule_graph(fragment)
        except Exception:
            pass
        ring = len(list(nx.simple_cycles(graph.to_directed()))) > 0
        if carbon_count < 2 and not ring:
            continue

        graph_key = _graph_key(graph)
        if touches_x:
            linker_stubs.add(graph_key)
            continue

        graph_keys.add(graph_key)

        if mol is not None:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            inchi_key = Chem.MolToInchiKey(mol)
            if canonical:
                smiles.add(canonical)
            if inchi_key:
                inchikeys.add(inchi_key[:14])

    return {
        "has_organic": bool(graph_keys),
        "inchikeys": sorted(inchikeys),
        "smiles": sorted(smiles),
        "graph_keys": sorted(graph_keys),
        "linker_stubs": sorted(linker_stubs),
    }


def main():
    block_names = set()
    for filename in ("metal_node.txt", "metal_edge.txt"):
        block_names.update((DATA_DIR / filename).read_text().split())

    result = {
        block_name: scan_block(block_name)
        for block_name in sorted(block_names)
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    count = sum(entry["has_organic"] for entry in result.values())
    print(f"Metal blocks with standalone embedded ligands: {count}/{len(result)}")
    print("N85:")
    print(json.dumps(result.get("N85", {}), indent=2, sort_keys=True))
    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
