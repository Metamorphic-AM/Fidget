//! Counters describing how the mesher placed its vertices.
//!
//! These exist so the quality tests can report *why* a mesh came out the way
//! it did, not only *that* it did: how many cell vertices needed a lower rank,
//! a boundary-constrained solve or the mass point to stay inside their cell,
//! how often the edge search saw a sign flip, and how often a NaN gradient
//! forced a snap.  Relaxed atomics; the cost is a handful of increments per
//! leaf.

#![allow(missing_docs)]

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// How a cell vertex was placed by [`QuadraticErrorSolver::solve_in`]
/// (`super::qef::QuadraticErrorSolver::solve_in`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    /// The unconstrained solve at the rank chosen by the eigenvalue cutoff
    /// landed inside the cell.
    Interior,
    /// A lower-rank solve landed inside the cell.
    LowerRank,
    /// The best solution constrained to one of the cell's faces.
    Face,
    /// The best solution constrained to one of the cell's edges.
    Edge,
    /// The best of the cell's corners.
    Corner,
    /// The mass point of the intersections.
    MassPoint,
}

pub static INTERIOR: AtomicUsize = AtomicUsize::new(0);
pub static LOWER_RANK: AtomicUsize = AtomicUsize::new(0);
pub static FACE: AtomicUsize = AtomicUsize::new(0);
pub static EDGE: AtomicUsize = AtomicUsize::new(0);
pub static CORNER: AtomicUsize = AtomicUsize::new(0);
pub static MASS_POINT: AtomicUsize = AtomicUsize::new(0);
/// Edge searches where a sample disagreed in sign with the corner it was
/// taken from (floating-point rounding between batches).
pub static EDGE_SEARCH_FLIPS: AtomicUsize = AtomicUsize::new(0);
/// Cell vertices snapped to an intersection because its gradient was NaN.
pub static NAN_SNAPS: AtomicUsize = AtomicUsize::new(0);
/// Cells collapsed into their parent during simplification.
pub static COLLAPSES: AtomicUsize = AtomicUsize::new(0);
/// Dual polygons the hybrid triangulation had to fan because neither
/// diagonal gave two triangles facing along the edge.
pub static HYBRID_FANS: AtomicUsize = AtomicUsize::new(0);
/// Sign components whose crossings lay on two sheets and received a vertex
/// per sheet.
pub static SPLITS: AtomicUsize = AtomicUsize::new(0);
/// Polygons emitted across a cell face to close the seam between the sheet
/// vertices of a split cell.
pub static STITCHES: AtomicUsize = AtomicUsize::new(0);
/// Dual polygons where a coarser neighbour still held several vertices and
/// the nearest to the edge crossing was taken.
pub static COARSE_PICKS: AtomicUsize = AtomicUsize::new(0);
/// Emitted triangles whose normal points against the surface direction
/// along their edge (folds that no triangulation could avoid).
pub static BACKWARD_TRIANGLES: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn record(p: Placement) {
    let c = match p {
        Placement::Interior => &INTERIOR,
        Placement::LowerRank => &LOWER_RANK,
        Placement::Face => &FACE,
        Placement::Edge => &EDGE,
        Placement::Corner => &CORNER,
        Placement::MassPoint => &MASS_POINT,
    };
    c.fetch_add(1, Relaxed);
}

/// A copy of every counter.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    pub interior: usize,
    pub lower_rank: usize,
    pub face: usize,
    pub edge: usize,
    pub corner: usize,
    pub mass_point: usize,
    pub edge_search_flips: usize,
    pub nan_snaps: usize,
    pub collapses: usize,
    pub hybrid_fans: usize,
    pub backward_triangles: usize,
    pub splits: usize,
    pub stitches: usize,
    pub coarse_picks: usize,
}

impl Snapshot {
    /// Cell vertices placed in total.
    pub fn placed(&self) -> usize {
        self.interior + self.lower_rank + self.face + self.edge + self.corner + self.mass_point
    }
    /// Cell vertices that the unconstrained solve would have put outside
    /// their cell.
    pub fn escaped(&self) -> usize {
        self.placed() - self.interior
    }
}

impl std::fmt::Display for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "placed={} interior={} lower_rank={} face={} edge={} corner={} mass_point={} \
             edge_flips={} nan_snaps={} collapses={} hybrid_fans={} backward_tris={} splits={} stitches={} coarse_picks={}",
            self.placed(),
            self.interior,
            self.lower_rank,
            self.face,
            self.edge,
            self.corner,
            self.mass_point,
            self.edge_search_flips,
            self.nan_snaps,
            self.collapses,
            self.hybrid_fans,
            self.backward_triangles,
            self.splits,
            self.stitches,
            self.coarse_picks,
        )
    }
}

/// Zero every counter.
pub fn reset() {
    for c in [
        &INTERIOR, &LOWER_RANK, &FACE, &EDGE, &CORNER, &MASS_POINT,
        &EDGE_SEARCH_FLIPS, &NAN_SNAPS, &COLLAPSES, &HYBRID_FANS, &BACKWARD_TRIANGLES, &SPLITS, &STITCHES, &COARSE_PICKS,
    ] {
        c.store(0, Relaxed);
    }
}

/// Read every counter.
pub fn snapshot() -> Snapshot {
    Snapshot {
        interior: INTERIOR.load(Relaxed),
        lower_rank: LOWER_RANK.load(Relaxed),
        face: FACE.load(Relaxed),
        edge: EDGE.load(Relaxed),
        corner: CORNER.load(Relaxed),
        mass_point: MASS_POINT.load(Relaxed),
        edge_search_flips: EDGE_SEARCH_FLIPS.load(Relaxed),
        nan_snaps: NAN_SNAPS.load(Relaxed),
        collapses: COLLAPSES.load(Relaxed),
        hybrid_fans: HYBRID_FANS.load(Relaxed),
        backward_triangles: BACKWARD_TRIANGLES.load(Relaxed),
        splits: SPLITS.load(Relaxed),
        stitches: STITCHES.load(Relaxed),
        coarse_picks: COARSE_PICKS.load(Relaxed),
    }
}
