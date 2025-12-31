/// Tree structure for phylogenetic inference.
///
/// Represents a rooted phylogenetic tree with branch lengths.
/// Nodes are indexed 0..n_nodes, with tips first, then internal nodes.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Tree {
    pub n_nodes: usize,
    pub n_tips: usize,
    pub parent_indices: Vec<i32>,
    pub branch_lengths: Vec<f64>,
    pub children: Vec<Vec<usize>>,
    pub tip_indices: Vec<usize>,
    pub root_index: usize,
    pub tip_name_to_idx: HashMap<String, usize>,
}

impl Tree {
    /// Build tree from structure arrays (from Python TreeStructure).
    ///
    /// # Arguments
    /// * `parent_indices` - Parent index for each node (-1 for root)
    /// * `branch_lengths` - Branch length for each node
    /// * `n_tips` - Number of tip nodes
    pub fn from_structure(
        parent_indices: Vec<i32>,
        branch_lengths: Vec<f64>,
        n_tips: usize,
    ) -> Result<Self, String> {
        let n_nodes = parent_indices.len();
        
        if branch_lengths.len() != n_nodes {
            return Err("parent_indices and branch_lengths must have same length".to_string());
        }
        
        // Build children dictionary
        let mut children = vec![Vec::new(); n_nodes];
        for (child_idx, &parent_idx) in parent_indices.iter().enumerate() {
            if parent_idx >= 0 {
                children[parent_idx as usize].push(child_idx);
            }
        }
        
        // Find root (node with no parent)
        let root_index = parent_indices.iter()
            .position(|&p| p == -1)
            .ok_or("No root node found (no node with parent_idx == -1)")?;
        
        // Tip indices are first n_tips nodes
        let tip_indices: Vec<usize> = (0..n_tips).collect();
        
        Ok(Tree {
            n_nodes,
            n_tips,
            parent_indices,
            branch_lengths,
            children,
            tip_indices,
            root_index,
            tip_name_to_idx: HashMap::new(), // Not needed for structure-based construction
        })
    }
    
    /// Get post-order traversal of nodes (tips to root).
    pub fn post_order(&self) -> Vec<usize> {
        let mut order = Vec::new();
        let mut visited = vec![false; self.n_nodes];
        
        fn visit(
            node: usize,
            tree: &Tree,
            visited: &mut Vec<bool>,
            order: &mut Vec<usize>,
        ) {
            if visited[node] {
                return;
            }
            
            // Visit children first
            for &child in &tree.children[node] {
                visit(child, tree, visited, order);
            }
            
            visited[node] = true;
            order.push(node);
        }
        
        visit(self.root_index, self, &mut visited, &mut order);
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tree_creation() {
        let taxon_names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let tree = Tree::from_newick("((A:1,B:1):1,C:1);", &taxon_names).unwrap();
        
        assert_eq!(tree.n_tips, 3);
        assert_eq!(tree.n_nodes, 5);
    }
    
    #[test]
    fn test_post_order() {
        let taxon_names = vec!["A".to_string(), "B".to_string()];
        let tree = Tree::from_newick("(A:1,B:1);", &taxon_names).unwrap();
        
        let order = tree.post_order();
        // Root should be last
        assert_eq!(order.last(), Some(&tree.root_index));
    }
}
