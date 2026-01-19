""" Module containing a class that holds the tree search
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from aizynthfinder.chem.serialization import MoleculeDeserializer, MoleculeSerializer
from aizynthfinder.search.andor_trees import AndOrSearchTreeBase, SplitAndOrTree
from aizynthfinder.search.retrostar.cost import MoleculeCost
from aizynthfinder.search.retrostar.nodes import MoleculeNode
from aizynthfinder.utils.exceptions import RejectionException
from aizynthfinder.utils.logging import logger

if TYPE_CHECKING:
    from aizynthfinder.chem import RetroReaction
    from aizynthfinder.context.config import Configuration
    from aizynthfinder.reactiontree import ReactionTree
    from aizynthfinder.utils.type_utils import List, Optional, Sequence


class SearchTree(AndOrSearchTreeBase):
    """
    Encapsulation of the Retro* search tree (an AND/OR tree).

    :ivar config: settings of the tree search algorithm
    :ivar root: the root node

    :param config: settings of the tree search algorithm
    :param root_smiles: the root will be set to a node representing this molecule, defaults to None
    """

    def __init__(
        self, config: Configuration, root_smiles: Optional[str] = None
    ) -> None:
        super().__init__(config, root_smiles)
        self._mol_nodes: List[List[MoleculeNode]] = []
        self._logger = logger()
        self.molecule_cost = MoleculeCost(config)
        self.root_slvd_bool_l = []

        if root_smiles:
            self.root = []
            for root_smi in root_smiles.split("."):
                current_root = MoleculeNode.create_root(
                    root_smi, config, self.molecule_cost
                )
                self.root.append(current_root)
                self.root_slvd_bool_l.append(False)
                self._mol_nodes.append([current_root])
        else:
            self.root = None

        self._routes: List[ReactionTree] = []

        self.profiling = {
            "expansion_calls": 0,
            "reactants_generations": 0,
        }
        self.retro_bm_wdth = config.search.retro_bm_width
        print("\n Retrostar search algorithm beam width ", self.retro_bm_wdth)
        self.route_group_done = []

    @classmethod
    def from_json(cls, filename: str, config: Configuration) -> SearchTree:
        """
        Create a new search tree by deserialization from a JSON file

        :param filename: the path to the JSON node
        :param config: the configuration of the search tree
        :return: a deserialized tree
        """

        def _find_mol_nodes(node):
            for child_ in node.children:
                tree._mol_nodes.append(child_)  # pylint: disable=protected-access
                for grandchild in child_.children:
                    _find_mol_nodes(grandchild)

        tree = cls(config)
        with open(filename, "r") as fileobj:
            dict_ = json.load(fileobj)
        mol_deser = MoleculeDeserializer(dict_["molecules"])
        tree.root = MoleculeNode.from_dict(
            dict_["tree"], config, mol_deser, tree.molecule_cost
        )
        tree._mol_nodes.append(tree.root)  # pylint: disable=protected-access
        for child in tree.root.children:
            _find_mol_nodes(child)
        return tree

    @property
    def mol_nodes(self) -> Sequence[MoleculeNode]:  # type: ignore
        """Return the molecule nodes of the tree"""
        return self._mol_nodes

    def one_iteration(self) -> bool:
        """
        Perform one iteration of
            1. Selection
            2. Expansion
            3. Update

        :raises StopIteration: if the search should be pre-maturely terminated
        :return: if a solution was found
        :rtype: bool
        """
        if self.root is None:
            raise ValueError("Root is undefined. Cannot make an iteration")

        self._routes = []
        
        selected_nodes_and_root_ids = self._select()

        if selected_nodes_and_root_ids is None:
            self._logger.debug("No expandable nodes in Retro* iteration")
            raise StopIteration
        next_node_list, root_id_list = selected_nodes_and_root_ids

        self._expand(next_node_list, root_id_list)
        for next_node in next_node_list:
            if not next_node.children:
                next_node.expandable = False

            self._update(next_node)
        self.root_slvd_bool_l = [r.solved for r in self.root]
        return self.root_slvd_bool_l

    def routes(self, root_id: int) -> List[ReactionTree]:
        """
        Extracts and returns routes from the AND/OR tree

        :return: the routes
        """
        if root_id in self.route_group_done:
            assert self._routes is not None
        else:
            self._routes = SplitAndOrTree(self.root[root_id], self.config.stock).routes
            self.route_group_done.append(root_id)
        return self._routes

    def serialize(self, filename: str) -> None:
        """
        Seralize the search tree to a JSON file

        :param filename: the path to the JSON file
        :type filename: str
        """
        if self.root is None:
            raise ValueError("Cannot serialize tree as root is not defined")

        mol_ser = MoleculeSerializer()
        dict_ = {"tree": self.root.serialize(mol_ser), "molecules": mol_ser.store}
        with open(filename, "w") as fileobj:
            json.dump(dict_, fileobj, indent=2)

    def _expand(self, node_list: MoleculeNode, root_id_list: List[int]) -> None:
        batch_reactions, batch_priors = self.config.expansion_policy([node.mol for node in node_list])
        assert len(node_list) == len(root_id_list)

        for i in range(len(node_list)):
            root_id = root_id_list[i]
            node = node_list[i]
            reactions = batch_reactions[i]
            priors = batch_priors[i]
            self.profiling["expansion_calls"] += 1

            if not reactions:
                continue

            costs = -np.log(np.clip(priors, 1e-3, 1.0))
            reactions_to_expand = []
            reaction_costs = []
            for reaction, cost in zip(reactions, costs):
                try:
                    self.profiling["reactants_generations"] += 1
                    _ = reaction.reactants
                except:  # pylint: disable=bare-except
                    continue
                if not reaction.reactants:
                    continue
                for idx, _ in enumerate(reaction.reactants):
                    rxn_copy = reaction.copy(idx)
                    if self._filter_reaction(rxn_copy):
                        continue
                    reactions_to_expand.append(rxn_copy)
                    reaction_costs.append(cost)

            for cost, rxn in zip(reaction_costs, reactions_to_expand):
                new_nodes = node.add_stub(cost, rxn)
                self._mol_nodes[root_id].extend(new_nodes)

    def _filter_reaction(self, reaction: RetroReaction) -> bool:
        if not self.config.filter_policy.selection:
            return False
        try:
            self.config.filter_policy(reaction)
        except RejectionException as err:
            self._logger.debug(str(err))
            return True
        return False

    def _select(self) -> Optional[MoleculeNode]:
        selected_nodes = []
        root_id_list = []

        for root_id, mol_nodes_l in enumerate(self._mol_nodes):
            if self.root_slvd_bool_l[root_id] is True:
                continue
            scores = np.asarray(
                [
                    node.target_value for node in mol_nodes_l if node.expandable
                ]
            )
            if len(scores) == 0:
                continue

            idx = [i for i in range(len(mol_nodes_l)) if mol_nodes_l[i].expandable]

            selected_nodes.append(mol_nodes_l[idx[np.argsort(scores)[0]]])
            root_id_list.append(root_id)

        if len(selected_nodes) == 0:
            return None
        return selected_nodes, root_id_list

    @staticmethod
    def _update(node: MoleculeNode) -> None:
        v_delta = node.close()
        if node.parent and np.isfinite(v_delta):
            node.parent.update(v_delta, from_mol=node.mol)
