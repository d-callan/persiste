//! Assembly state representation.
//!
//! States are compositional equivalence classes, not specific molecules.
//! Mirrors the Python AssemblyState but optimized for Rust performance.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

/// Unique identifier for an assembly state (used for caching).
pub type AssemblyStateId = u64;

/// Compositional state in assembly theory.
///
/// Represents an equivalence class of molecular assemblies,
/// not a specific molecule. This keeps the state space tractable.
///
/// Key properties:
/// - Immutable (Clone only)
/// - Hashable (can be HashMap keys)
/// - Compositional (multiset of parts, not molecular graph)
/// - Compact (assembly depth instead of full history)
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssemblyState {
    /// Multiset of building blocks: part_name -> count
    parts: BTreeMap<String, u32>,
    /// Integer assembly index proxy
    assembly_depth: u32,
    /// Optional structural motif labels
    motifs: BTreeSet<String>,
    /// Cached hash for fast lookup
    cached_id: AssemblyStateId,
}

impl Hash for AssemblyState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cached_id.hash(state);
    }
}

impl AssemblyState {
    /// Create a new assembly state from parts.
    ///
    /// # Arguments
    /// * `parts` - List of building block identifiers (can have duplicates)
    /// * `depth` - Assembly depth
    /// * `motifs` - Optional set of motif labels
    pub fn new(parts: &[&str], depth: u32, motifs: Option<&[&str]>) -> Self {
        let mut parts_map = BTreeMap::new();
        for part in parts {
            *parts_map.entry(part.to_string()).or_insert(0) += 1;
        }

        let motifs_set: BTreeSet<String> = motifs
            .unwrap_or(&[])
            .iter()
            .map(|s| s.to_string())
            .collect();

        let cached_id = Self::compute_id(&parts_map, depth, &motifs_set);

        Self {
            parts: parts_map,
            assembly_depth: depth,
            motifs: motifs_set,
            cached_id,
        }
    }

    /// Create from pre-counted parts map.
    pub fn from_parts_map(
        parts: BTreeMap<String, u32>,
        depth: u32,
        motifs: BTreeSet<String>,
    ) -> Self {
        let cached_id = Self::compute_id(&parts, depth, &motifs);
        Self {
            parts,
            assembly_depth: depth,
            motifs,
            cached_id,
        }
    }

    /// Create empty state (depth 0, no parts).
    pub fn empty() -> Self {
        Self::new(&[], 0, None)
    }

    /// Compute deterministic ID from state components.
    fn compute_id(
        parts: &BTreeMap<String, u32>,
        depth: u32,
        motifs: &BTreeSet<String>,
    ) -> AssemblyStateId {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();

        // Hash parts in sorted order (BTreeMap is already sorted)
        for (part, count) in parts {
            part.hash(&mut hasher);
            count.hash(&mut hasher);
        }

        depth.hash(&mut hasher);

        // Hash motifs in sorted order
        for motif in motifs {
            motif.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Get the unique state ID.
    #[inline]
    pub fn id(&self) -> AssemblyStateId {
        self.cached_id
    }

    /// Get assembly depth.
    #[inline]
    pub fn depth(&self) -> u32 {
        self.assembly_depth
    }

    /// Get parts as reference to the internal map.
    #[inline]
    pub fn parts(&self) -> &BTreeMap<String, u32> {
        &self.parts
    }

    /// Get motifs as reference.
    #[inline]
    pub fn motifs(&self) -> &BTreeSet<String> {
        &self.motifs
    }

    /// Total number of parts (with multiplicity).
    pub fn total_parts(&self) -> u32 {
        self.parts.values().sum()
    }

    /// Alias for total_parts.
    #[inline]
    pub fn size(&self) -> u32 {
        self.total_parts()
    }

    /// Check if state contains a specific part.
    pub fn contains_part(&self, part: &str) -> bool {
        self.parts.contains_key(part)
    }

    /// Check if state has a specific motif.
    pub fn contains_motif(&self, motif: &str) -> bool {
        self.motifs.contains(motif)
    }

    /// Check if this state is a subassembly of another.
    ///
    /// True if all parts of this state are contained in other with at least the same count.
    pub fn is_subassembly_of(&self, other: &AssemblyState) -> bool {
        for (part, &count) in &self.parts {
            match other.parts.get(part) {
                Some(&other_count) if other_count >= count => continue,
                _ => return false,
            }
        }
        true
    }

    /// Get parts as a flat list (with duplicates).
    pub fn get_parts_list(&self) -> Vec<String> {
        let mut result = Vec::new();
        for (part, &count) in &self.parts {
            for _ in 0..count {
                result.push(part.clone());
            }
        }
        result
    }

    /// Create a new state by joining with another part.
    pub fn join_with(&self, part: &str) -> Self {
        let mut new_parts = self.parts.clone();
        *new_parts.entry(part.to_string()).or_insert(0) += 1;
        Self::from_parts_map(new_parts, self.assembly_depth + 1, self.motifs.clone())
    }

    /// Create a new state by joining two states.
    pub fn join_states(a: &AssemblyState, b: &AssemblyState) -> Self {
        let mut new_parts = a.parts.clone();
        for (part, &count) in &b.parts {
            *new_parts.entry(part.clone()).or_insert(0) += count;
        }
        let new_depth = a.assembly_depth.max(b.assembly_depth) + 1;
        let new_motifs: BTreeSet<String> = a.motifs.union(&b.motifs).cloned().collect();
        Self::from_parts_map(new_parts, new_depth, new_motifs)
    }
}

impl std::fmt::Display for AssemblyState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parts_str: Vec<String> = self
            .parts
            .iter()
            .map(|(p, c)| {
                if *c > 1 {
                    format!("{}×{}", p, c)
                } else {
                    p.clone()
                }
            })
            .collect();
        let motifs_str = if self.motifs.is_empty() {
            String::new()
        } else {
            format!(" [{}]", self.motifs.iter().cloned().collect::<Vec<_>>().join(", "))
        };
        write!(
            f,
            "State(d={}: {}{})",
            self.assembly_depth,
            parts_str.join(", "),
            motifs_str
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = AssemblyState::new(&["A", "B", "A"], 2, None);
        assert_eq!(state.depth(), 2);
        assert_eq!(state.total_parts(), 3);
        assert!(state.contains_part("A"));
        assert!(state.contains_part("B"));
        assert!(!state.contains_part("C"));
    }

    #[test]
    fn test_state_equality() {
        let s1 = AssemblyState::new(&["A", "B", "A"], 2, None);
        let s2 = AssemblyState::new(&["A", "A", "B"], 2, None);
        assert_eq!(s1, s2);
        assert_eq!(s1.id(), s2.id());
    }

    #[test]
    fn test_subassembly() {
        let small = AssemblyState::new(&["A"], 1, None);
        let large = AssemblyState::new(&["A", "B"], 2, None);
        assert!(small.is_subassembly_of(&large));
        assert!(!large.is_subassembly_of(&small));
    }

    #[test]
    fn test_join_with() {
        let state = AssemblyState::new(&["A"], 1, None);
        let joined = state.join_with("B");
        assert_eq!(joined.total_parts(), 2);
        assert_eq!(joined.depth(), 2);
    }
}
