//! Hot-spot detection for JIT compilation.
//!
//! Identifies code regions that would benefit from JIT compilation:
//! - Frequently executed loops (loop headers with high iteration counts)
//! - Hot functions (called many times)
//! - Stable type profiles (monomorphic code paths)

use std::fmt;

use rustc_hash::FxHashMap;

use super::types::TypeProfile;

/// Threshold for considering a loop "hot" and ready for JIT
pub const LOOP_HOT_THRESHOLD: u32 = 1000;

/// Threshold for considering a function "hot"
pub const FUNCTION_HOT_THRESHOLD: u32 = 100;

/// Minimum observations before we trust type feedback
pub const MIN_TYPE_OBSERVATIONS: u32 = 10;

/// Identifies a hot-spot in the code
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HotSpotKind {
    /// A loop (instruction range: start_ip..end_ip)
    Loop {
        /// IP of the loop header (where iteration check happens)
        header_ip: usize,
        /// IP of the loop end (jump back target)
        end_ip: usize,
        /// IP of the exit jump target
        exit_ip: usize,
    },
    /// A function
    Function {
        /// Name of the function
        name: String,
        /// Start IP of function body
        start_ip: usize,
    },
    /// A frequently executed basic block
    BasicBlock { start_ip: usize, end_ip: usize },
}

impl fmt::Display for HotSpotKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HotSpotKind::Loop {
                header_ip, end_ip, ..
            } => {
                write!(f, "loop@{}..{}", header_ip, end_ip)
            }
            HotSpotKind::Function { name, start_ip } => {
                write!(f, "fn {}@{}", name, start_ip)
            }
            HotSpotKind::BasicBlock { start_ip, end_ip } => {
                write!(f, "block@{}..{}", start_ip, end_ip)
            }
        }
    }
}

/// Information about a detected hot-spot
#[derive(Debug, Clone)]
pub struct HotSpot {
    /// What kind of hot-spot this is
    pub kind: HotSpotKind,
    /// Execution count (iterations for loops, calls for functions)
    pub execution_count: u32,
    /// Type profile for this region
    pub type_profile: TypeProfile,
    /// Whether this hot-spot has been JIT compiled
    pub is_compiled: bool,
    /// Priority score for JIT compilation (higher = more important)
    pub priority: u32,
}

impl HotSpot {
    pub fn new(kind: HotSpotKind) -> Self {
        Self {
            kind,
            execution_count: 0,
            type_profile: TypeProfile::new(),
            is_compiled: false,
            priority: 0,
        }
    }

    /// Increment execution count and recalculate priority
    pub fn record_execution(&mut self) {
        self.execution_count = self.execution_count.saturating_add(1);
        self.update_priority();
    }

    /// Update JIT compilation priority based on execution count and type stability
    fn update_priority(&mut self) {
        // Base priority from execution count
        let mut score = self.execution_count;

        // Bonus for stable types (monomorphic code)
        if !self.type_profile.is_empty() {
            let (start, end) = self.ip_range();
            if self.type_profile.region_is_jit_candidate(start, end) {
                score = score.saturating_mul(2);
            }
        }

        // Loops get higher priority than functions (more iterations = more benefit)
        if matches!(self.kind, HotSpotKind::Loop { .. }) {
            score = score.saturating_mul(3);
        }

        self.priority = score;
    }

    /// Get the IP range covered by this hot-spot
    pub fn ip_range(&self) -> (usize, usize) {
        match &self.kind {
            HotSpotKind::Loop {
                header_ip, end_ip, ..
            } => (*header_ip, *end_ip),
            HotSpotKind::Function { start_ip, .. } => (*start_ip, usize::MAX), // Functions extend to Return
            HotSpotKind::BasicBlock { start_ip, end_ip } => (*start_ip, *end_ip),
        }
    }

    /// Check if this hot-spot is ready for JIT compilation
    pub fn is_hot(&self) -> bool {
        let threshold = match &self.kind {
            HotSpotKind::Loop { .. } => LOOP_HOT_THRESHOLD,
            HotSpotKind::Function { .. } => FUNCTION_HOT_THRESHOLD,
            HotSpotKind::BasicBlock { .. } => LOOP_HOT_THRESHOLD,
        };
        self.execution_count >= threshold && !self.is_compiled
    }

    /// Check if this hot-spot should be JIT compiled
    /// Requires being hot AND having stable types
    pub fn should_compile(&self) -> bool {
        if !self.is_hot() {
            return false;
        }

        // Require some type observations
        if self.type_profile.is_empty() {
            return false;
        }

        let (start, end) = self.ip_range();
        self.type_profile.region_is_jit_candidate(start, end)
    }
}

/// Detects and tracks hot-spots in executing code
#[derive(Debug, Default)]
pub struct HotSpotDetector {
    /// Hot-spots indexed by their header/start IP
    hot_spots: FxHashMap<usize, HotSpot>,
    /// Map from function name to start IP (for function tracking)
    function_ips: FxHashMap<String, usize>,
    /// Backward jump targets (potential loop headers)
    loop_headers: FxHashMap<usize, LoopInfo>,
    /// Total number of JIT-compiled regions
    compiled_count: usize,
}

/// Information about a detected loop
#[derive(Debug, Clone)]
struct LoopInfo {
    /// IP of the backward jump instruction
    back_edge_ip: usize,
    /// IP of the exit (jump target when loop ends)
    exit_ip: usize,
}

impl HotSpotDetector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a loop detected during bytecode analysis
    /// Called when we see a backward jump (Jump or JumpIfFalse to earlier IP)
    pub fn register_loop(&mut self, header_ip: usize, back_edge_ip: usize, exit_ip: usize) {
        self.loop_headers.insert(
            header_ip,
            LoopInfo {
                back_edge_ip,
                exit_ip,
            },
        );

        let kind = HotSpotKind::Loop {
            header_ip,
            end_ip: back_edge_ip,
            exit_ip,
        };
        self.hot_spots
            .entry(header_ip)
            .or_insert_with(|| HotSpot::new(kind));
    }

    /// Register a function for tracking
    pub fn register_function(&mut self, name: &str, start_ip: usize) {
        self.function_ips.insert(name.to_string(), start_ip);

        let kind = HotSpotKind::Function {
            name: name.to_string(),
            start_ip,
        };
        self.hot_spots
            .entry(start_ip)
            .or_insert_with(|| HotSpot::new(kind));
    }

    /// Record execution at a specific IP
    /// Returns true if this location is now hot and should be considered for JIT
    pub fn record_execution(&mut self, ip: usize) -> bool {
        if let Some(hotspot) = self.hot_spots.get_mut(&ip) {
            hotspot.record_execution();
            return hotspot.is_hot();
        }
        false
    }

    /// Record a function call
    pub fn record_function_call(&mut self, name: &str) -> bool {
        if let Some(&start_ip) = self.function_ips.get(name) {
            return self.record_execution(start_ip);
        }
        false
    }

    /// Record a loop iteration (at the loop header)
    pub fn record_loop_iteration(&mut self, header_ip: usize) -> bool {
        self.record_execution(header_ip)
    }

    /// Get a hot-spot by its start IP
    pub fn get(&self, ip: usize) -> Option<&HotSpot> {
        self.hot_spots.get(&ip)
    }

    /// Get a mutable reference to a hot-spot's type profile
    pub fn get_type_profile_mut(&mut self, ip: usize) -> Option<&mut TypeProfile> {
        self.hot_spots.get_mut(&ip).map(|hs| &mut hs.type_profile)
    }

    /// Mark a hot-spot as compiled
    pub fn mark_compiled(&mut self, ip: usize) {
        if let Some(hotspot) = self.hot_spots.get_mut(&ip) {
            hotspot.is_compiled = true;
            self.compiled_count += 1;
        }
    }

    /// Get all hot-spots that are ready for JIT compilation, sorted by priority
    pub fn get_compilation_candidates(&self) -> Vec<&HotSpot> {
        let mut candidates: Vec<_> = self
            .hot_spots
            .values()
            .filter(|hs| hs.should_compile())
            .collect();
        candidates.sort_by(|a, b| b.priority.cmp(&a.priority));
        candidates
    }

    /// Get all hot loops (for debugging/stats)
    pub fn get_hot_loops(&self) -> Vec<&HotSpot> {
        self.hot_spots
            .values()
            .filter(|hs| matches!(hs.kind, HotSpotKind::Loop { .. }) && hs.is_hot())
            .collect()
    }

    /// Get all hot functions (for debugging/stats)
    pub fn get_hot_functions(&self) -> Vec<&HotSpot> {
        self.hot_spots
            .values()
            .filter(|hs| matches!(hs.kind, HotSpotKind::Function { .. }) && hs.is_hot())
            .collect()
    }

    /// Check if an IP is a known loop header
    pub fn is_loop_header(&self, ip: usize) -> bool {
        self.loop_headers.contains_key(&ip)
    }

    /// Get loop exit IP if this is a loop header
    pub fn get_loop_exit_ip(&self, header_ip: usize) -> Option<usize> {
        self.loop_headers.get(&header_ip).map(|info| info.exit_ip)
    }

    /// Check if a loop at the given header IP is hot and ready for JIT
    pub fn is_loop_hot(&self, header_ip: usize) -> bool {
        self.hot_spots
            .get(&header_ip)
            .map(|hs| hs.is_hot())
            .unwrap_or(false)
    }

    /// Get statistics about hot-spot detection
    pub fn stats(&self) -> HotSpotStats {
        let total = self.hot_spots.len();
        let hot_loops = self
            .hot_spots
            .values()
            .filter(|hs| matches!(hs.kind, HotSpotKind::Loop { .. }) && hs.is_hot())
            .count();
        let hot_functions = self
            .hot_spots
            .values()
            .filter(|hs| matches!(hs.kind, HotSpotKind::Function { .. }) && hs.is_hot())
            .count();
        let compiled = self.compiled_count;

        let hottest = self
            .hot_spots
            .values()
            .max_by_key(|hs| hs.execution_count)
            .map(|hs| (hs.kind.clone(), hs.execution_count));

        HotSpotStats {
            total_tracked: total,
            hot_loops,
            hot_functions,
            compiled_regions: compiled,
            hottest_spot: hottest,
        }
    }

    /// Clear all tracking data (for benchmarking fresh runs)
    pub fn reset(&mut self) {
        for hs in self.hot_spots.values_mut() {
            hs.execution_count = 0;
            hs.is_compiled = false;
            hs.type_profile.clear();
        }
        self.compiled_count = 0;
    }
}

/// Statistics about hot-spot detection
#[derive(Debug, Clone)]
pub struct HotSpotStats {
    pub total_tracked: usize,
    pub hot_loops: usize,
    pub hot_functions: usize,
    pub compiled_regions: usize,
    pub hottest_spot: Option<(HotSpotKind, u32)>,
}

impl fmt::Display for HotSpotStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Hot-spot Statistics:")?;
        writeln!(f, "  Total tracked regions: {}", self.total_tracked)?;
        writeln!(f, "  Hot loops: {}", self.hot_loops)?;
        writeln!(f, "  Hot functions: {}", self.hot_functions)?;
        writeln!(f, "  JIT compiled regions: {}", self.compiled_regions)?;
        if let Some((kind, count)) = &self.hottest_spot {
            writeln!(f, "  Hottest spot: {} ({}x)", kind, count)?;
        }
        Ok(())
    }
}
