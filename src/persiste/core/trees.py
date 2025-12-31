"""
Core tree utilities for PERSISTE.

Provides a generic tree interface that plugins can use for tree-based
likelihood computation. Supports multiple backends (dendropy, ete3, simple).

Key design principles:
1. Backend-agnostic interface - plugins don't care how tree was parsed
2. Efficient structure extraction - arrays for JAX/numpy operations
3. Lazy loading of optional dependencies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class TreeNode:
    """
    Generic tree node representation.
    
    Attributes:
        id: Unique node identifier
        name: Node name (for tips, usually taxon name)
        parent_id: ID of parent node (None for root)
        children_ids: List of child node IDs
        branch_length: Length of branch leading to this node
        is_tip: Whether this is a leaf node
    """
    id: int
    name: Optional[str] = None
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    branch_length: float = 0.0
    is_tip: bool = False


@dataclass
class TreeStructure:
    """
    Extracted tree structure optimized for likelihood computation.
    
    This is the core data structure that pruning algorithms use.
    It's backend-agnostic - can be created from dendropy, ete3, or simple parser.
    
    Attributes:
        n_nodes: Total number of nodes
        n_tips: Number of tip nodes
        nodes: List of TreeNode objects
        tip_indices: Indices of tip nodes
        internal_indices: Indices of internal nodes
        root_index: Index of root node
        postorder: Node indices in postorder (tips first, root last)
        branch_lengths: Array of branch lengths indexed by node
        parent_indices: Array of parent indices (-1 for root)
        children_array: Array of (parent, child1, child2) for internal nodes
        tip_names: List of tip names in order of tip_indices
    """
    n_nodes: int
    n_tips: int
    nodes: List[TreeNode]
    tip_indices: List[int]
    internal_indices: List[int]
    root_index: int
    postorder: List[int]
    branch_lengths: np.ndarray
    parent_indices: np.ndarray
    children_array: np.ndarray  # Shape: (n_internal, 3) for binary trees
    tip_names: List[str]
    
    @classmethod
    def from_newick(cls, newick: str, backend: str = "auto") -> 'TreeStructure':
        """
        Parse Newick string into TreeStructure.
        
        Args:
            newick: Newick format tree string
            backend: Parser backend ("dendropy", "simple", "auto")
            
        Returns:
            TreeStructure instance
        """
        if backend == "auto":
            # Try dendropy first, fall back to simple
            try:
                return cls._from_dendropy(newick)
            except ImportError:
                return cls._from_simple(newick)
        elif backend == "dendropy":
            return cls._from_dendropy(newick)
        elif backend == "simple":
            return cls._from_simple(newick)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path], backend: str = "auto") -> 'TreeStructure':
        """Load tree from Newick file."""
        with open(filepath, 'r') as f:
            newick = f.read().strip()
        return cls.from_newick(newick, backend=backend)
    
    @classmethod
    def _from_dendropy(cls, newick: str) -> 'TreeStructure':
        """Parse using dendropy (recommended for complex trees)."""
        try:
            import dendropy
        except ImportError:
            raise ImportError(
                "dendropy required for tree parsing. "
                "Install with: pip install dendropy"
            )
        
        tree = dendropy.Tree.get(data=newick, schema="newick")
        
        # Extract structure
        nodes = []
        node_to_idx = {}
        
        # Postorder traversal
        for i, node in enumerate(tree.postorder_node_iter()):
            node_to_idx[node] = i
            
            tree_node = TreeNode(
                id=i,
                name=node.taxon.label if node.taxon else None,
                branch_length=node.edge_length if node.edge_length else 0.0,
                is_tip=node.is_leaf(),
            )
            nodes.append(tree_node)
        
        # Set parent and children relationships
        for node in tree.postorder_node_iter():
            idx = node_to_idx[node]
            
            if node.parent_node:
                nodes[idx].parent_id = node_to_idx[node.parent_node]
            
            for child in node.child_nodes():
                nodes[idx].children_ids.append(node_to_idx[child])
        
        return cls._build_from_nodes(nodes, node_to_idx[tree.seed_node])
    
    @classmethod
    def _from_simple(cls, newick: str) -> 'TreeStructure':
        """Simple Newick parser (no external dependencies)."""
        # Remove trailing semicolon and whitespace
        newick = newick.strip().rstrip(';')
        
        nodes = []
        node_counter = [0]
        
        def parse_node(s: str, parent_id: Optional[int] = None) -> Tuple[int, str]:
            """Parse a node and return (node_id, remaining_string)."""
            s = s.strip()
            
            if s.startswith('('):
                # Internal node
                node_id = node_counter[0]
                node_counter[0] += 1
                
                node = TreeNode(
                    id=node_id,
                    parent_id=parent_id,
                    is_tip=False,
                )
                nodes.append(node)
                
                # Find matching closing paren
                depth = 1
                i = 1
                while depth > 0 and i < len(s):
                    if s[i] == '(':
                        depth += 1
                    elif s[i] == ')':
                        depth -= 1
                    i += 1
                
                # Parse children
                children_str = s[1:i-1]
                remaining = s[i:]
                
                # Split children by comma (but not inside parens)
                children_parts = []
                depth = 0
                start = 0
                for j, c in enumerate(children_str):
                    if c == '(':
                        depth += 1
                    elif c == ')':
                        depth -= 1
                    elif c == ',' and depth == 0:
                        children_parts.append(children_str[start:j])
                        start = j + 1
                children_parts.append(children_str[start:])
                
                # Parse each child
                for child_str in children_parts:
                    child_id, _ = parse_node(child_str.strip(), node_id)
                    node.children_ids.append(child_id)
                
                # Parse name and branch length after closing paren
                if remaining:
                    # Parse optional name
                    name_end = 0
                    for j, c in enumerate(remaining):
                        if c in ':,;':
                            break
                        name_end = j + 1
                    
                    if name_end > 0:
                        node.name = remaining[:name_end].strip()
                    
                    remaining = remaining[name_end:]
                    
                    # Parse branch length
                    if remaining.startswith(':'):
                        colon_end = len(remaining)
                        for j, c in enumerate(remaining[1:], 1):
                            if c in ',;':
                                colon_end = j
                                break
                        try:
                            node.branch_length = float(remaining[1:colon_end])
                        except ValueError:
                            pass
                        remaining = remaining[colon_end:]
                
                return node_id, remaining
            
            else:
                # Tip node
                node_id = node_counter[0]
                node_counter[0] += 1
                
                # Find end of name
                end = len(s)
                for j, c in enumerate(s):
                    if c in ':,;':
                        end = j
                        break
                
                name = s[:end].strip() if end > 0 else None
                remaining = s[end:]
                
                # Parse branch length
                branch_length = 0.0
                if remaining.startswith(':'):
                    colon_end = len(remaining)
                    for j, c in enumerate(remaining[1:], 1):
                        if c in ',;':
                            colon_end = j
                            break
                    try:
                        branch_length = float(remaining[1:colon_end])
                    except ValueError:
                        pass
                    remaining = remaining[colon_end:]
                
                node = TreeNode(
                    id=node_id,
                    name=name,
                    parent_id=parent_id,
                    branch_length=branch_length,
                    is_tip=True,
                )
                nodes.append(node)
                
                return node_id, remaining
        
        root_id, _ = parse_node(newick)
        
        return cls._build_from_nodes(nodes, root_id)
    
    @classmethod
    def _build_from_nodes(cls, nodes: List[TreeNode], root_index: int) -> 'TreeStructure':
        """Build TreeStructure from list of nodes."""
        n_nodes = len(nodes)
        
        # Sort nodes by ID for consistent indexing
        nodes = sorted(nodes, key=lambda n: n.id)
        
        # Identify tips and internals
        tip_indices = [n.id for n in nodes if n.is_tip]
        internal_indices = [n.id for n in nodes if not n.is_tip]
        n_tips = len(tip_indices)
        
        # Build arrays
        branch_lengths = np.array([n.branch_length for n in nodes])
        parent_indices = np.array([n.parent_id if n.parent_id is not None else -1 for n in nodes])
        
        # Compute postorder traversal first (needed for children_array ordering)
        postorder = cls._compute_postorder(nodes, root_index)
        
        # Build children array for internal nodes (binary trees)
        # IMPORTANT: Sort in postorder so children are processed before parents
        children_list = []
        for node_id in postorder:
            n = nodes[node_id]
            if not n.is_tip and len(n.children_ids) >= 2:
                children_list.append([n.id, n.children_ids[0], n.children_ids[1]])
        children_array = np.array(children_list, dtype=np.int32) if children_list else np.zeros((0, 3), dtype=np.int32)
        
        # Get tip names
        tip_names = [nodes[i].name or f"tip_{i}" for i in tip_indices]
        
        return cls(
            n_nodes=n_nodes,
            n_tips=n_tips,
            nodes=nodes,
            tip_indices=tip_indices,
            internal_indices=internal_indices,
            root_index=root_index,
            postorder=postorder,
            branch_lengths=branch_lengths,
            parent_indices=parent_indices,
            children_array=children_array,
            tip_names=tip_names,
        )
    
    @staticmethod
    def _compute_postorder(nodes: List[TreeNode], root_index: int) -> List[int]:
        """Compute postorder traversal."""
        result = []
        
        def visit(node_id: int):
            node = nodes[node_id]
            for child_id in node.children_ids:
                visit(child_id)
            result.append(node_id)
        
        visit(root_index)
        return result
    
    def get_tip_index_map(self) -> Dict[str, int]:
        """Map tip names to their indices."""
        return {name: idx for idx, name in zip(self.tip_indices, self.tip_names)}
    
    def __repr__(self) -> str:
        return f"TreeStructure({self.n_tips} tips, {self.n_nodes} nodes)"


class Tree:
    """
    Simple tree class for validation and testing.
    
    This is a lightweight wrapper that allows programmatic tree construction
    without parsing Newick strings.
    """
    
    def __init__(self):
        self.nodes = []
        self.node_counter = 0
        self.root = None
    
    def add_node(self, parent: Optional[int] = None, branch_length: float = 1.0, 
                 is_leaf: bool = False, name: Optional[str] = None) -> int:
        """Add a node to the tree."""
        node_id = self.node_counter
        self.node_counter += 1
        
        node = TreeNode(
            id=node_id,
            name=name,
            parent_id=parent,
            branch_length=branch_length,
            is_tip=is_leaf,
        )
        
        self.nodes.append(node)
        
        # Update parent's children
        if parent is not None:
            self.nodes[parent].children_ids.append(node_id)
        else:
            self.root = node_id
        
        return node_id
    
    def get_children(self, node_id: int) -> List[int]:
        """Get children of a node."""
        return self.nodes[node_id].children_ids
    
    def get_branch_length(self, node_id: int) -> float:
        """Get branch length of a node."""
        return self.nodes[node_id].branch_length
    
    def is_leaf(self, node_id: int) -> bool:
        """Check if node is a leaf."""
        return self.nodes[node_id].is_tip
    
    def get_node_name(self, node_id: int) -> str:
        """Get name of a node."""
        return self.nodes[node_id].name or f"node_{node_id}"
    
    def get_leaves(self) -> List[int]:
        """Get all leaf node IDs."""
        return [n.id for n in self.nodes if n.is_tip]


def load_tree(filepath: Union[str, Path], backend: str = "auto") -> TreeStructure:
    """
    Load a phylogenetic tree from file.
    
    This is the main entry point for tree loading in PERSISTE.
    
    Args:
        filepath: Path to Newick file
        backend: Parser backend ("dendropy", "simple", "auto")
        
    Returns:
        TreeStructure ready for likelihood computation
    """
    return TreeStructure.from_file(filepath, backend=backend)
