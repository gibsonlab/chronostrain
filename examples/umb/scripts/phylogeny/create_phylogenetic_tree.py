from typing import Tuple, List
import click
import numpy as np
from pathlib import Path

from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from skbio import DistanceMatrix
from skbio.tree import nj


def load_distances(dist_values_path: Path, strain_ordering_path: Path) -> Tuple[np.ndarray, List[str]]:
    full_matrix = np.load(dist_values_path)
    with open(strain_ordering_path, "rt") as f:
        strain_ids = [x.strip() for x in f]
    full_matrix[np.isinf(full_matrix)] = 1.0
    return full_matrix, strain_ids


def get_newick(node, parent_dist, leaf_names, newick='') -> str:
    """
    https://stackoverflow.com/questions/28222179/save-dendrogram-to-newick-format
    Convert scipy.cluster.hierarchy.to_tree()-output to Newick format.

    :param node: output of scipy.cluster.hierarchy.to_tree()
    :param parent_dist: output of scipy.cluster.hierarchy.to_tree().dist
    :param leaf_names: list of leaf names
    :param newick: leave empty, this variable is used in recursion.
    :returns: tree in Newick format
    """
    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parent_dist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parent_dist - node.dist, newick)
        else:
            newick = ");"
        newick = get_newick(node.get_left(), node.dist, leaf_names, newick=newick)
        newick = get_newick(node.get_right(), node.dist, leaf_names, newick=",%s" % (newick))
        newick = "(%s" % (newick)
        return newick


@click.command()
@click.option(
    '--output-path', '-o', 'output_path',
    type=click.Path(path_type=Path, dir_okay=False), required=True
)
@click.option(
    '--distance-array', '-da', 'distance_array_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True
)
@click.option(
    '--distance-ordering', '-do', 'distance_ordering_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True
)
def main(output_path: Path, distance_array_path: Path, distance_ordering_path: Path):
    output_path.parent.mkdir(exist_ok=True, parents=True)
    temp_dir = output_path.parent / '__dist_tmp'
    temp_dir.mkdir(exist_ok=True)

    # ========== Create tree.
    if output_path.exists():
        print("Already found pre-computed tree {}".format(output_path))
        return

    distances, acc_ordering = load_distances(distance_array_path, distance_ordering_path)
    print("Computing neighbor joining tree.")
    dm = DistanceMatrix(distances, acc_ordering)
    newick_str = nj(dm, result_constructor=str)
    print("Producing newick tree.")
    with open(output_path, "wt") as f:
        print(newick_str, file=f)

    # ============ scipy linkage function
    # print("Computing linkage via hierarchical clustering.")
    # Z = linkage(
    #     squareform(distances), method='complete'
    # )
    # tree = to_tree(Z, False)
    #
    # print("Producing newick tree.")
    # newick_str = get_newick(tree, tree.dist, acc_ordering)
    # with open(output_path, "wt") as f:
    #     print(newick_str, file=f)

    # ============ old code using biopython, NJ is slow!
    # print("Constructing tree from distance matrix.")
    # tree = DistanceTreeConstructor().nj(distances)
    # # erase internal node names. Necessary for SynerClust?
    # for clade in tree.get_nonterminals():
    #     clade.name = ""
    #
    # # Save the tree.
    # Phylo.write([tree], output_path, tree_format)
    # print("Created tree {}".format(output_path))
    # print("To run SynerClust, the user might need to manually delete the root node distance (`:0.000`).")


if __name__ == "__main__":
    main()
