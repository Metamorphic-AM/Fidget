//! Octree construction and meshing
//!
//! This module implements
//! [Manifold Dual Contouring](https://people.engr.tamu.edu/schaefer/research/dualsimp_tvcg.pdf),
//! to generate a triangle mesh from an implicit surface (or anything
//! implementing [`Shape`](fidget_core::shape::Shape)).
//!
//! The resulting meshes should be
//! - Manifold
//! - Watertight
//! - Preserving sharp features (corners / edges)
//!
//! However, they may contain self-intersections, and are not guaranteed to
//! catch thin features (below the sampling grid resolution).
//!
//! The resulting [`Mesh`] objects can be written out as STL files.
//!
//! Here's a full example, meshing a sphere:
//!
//! ```
//! use fidget_core::{
//!     context::Tree,
//!     vm::VmShape
//! };
//! use fidget_mesh::{Octree, Settings};
//!
//! let radius_squared = Tree::x().square()
//!     + Tree::y().square()
//!     + Tree::z().square();
//! let tree: Tree = radius_squared.sqrt() - 0.6;
//! let shape = VmShape::from(tree);
//! let settings = Settings {
//!     depth: 4,
//!     ..Default::default()
//! };
//! let o = Octree::build(&shape, &settings).unwrap();
//! let mesh = o.walk_dual();
//!
//! // Open a file to write, e.g.
//! // let mut f = std::fs::File::create("out.stl")?;
//! # let mut f = vec![];
//! mesh.write_stl(&mut f)?;
//! # Ok::<(), std::io::Error>(())
//! ```
#![warn(missing_docs)]

mod builder;
mod cell;
mod codegen;
mod dc;
mod frame;
mod octree;
mod output;
mod qef;

use fidget_core::render::{CancelToken, ThreadPool};
use std::sync::Arc;

#[doc(hidden)]
pub mod types;

// Re-export the main Octree type as public
pub use octree::Octree;

/// Callback supplying additional QEF plane constraints at each edge
/// zero-crossing processed by the MDC leaf vertex-placement routine.
///
/// At a sharp crease or corner in a mesh-backed oracle, Fidget's own
/// gradient-slice evaluator only returns a single outward normal — one side
/// of the crease — which leaves the QEF solver a single-plane problem and
/// the resulting vertex blunts the feature.  A hook of this type may be
/// installed on [`Settings`] so the meshing pipeline can ask an external
/// oracle (in Morphé's case, the FCPW mesh oracle reached through the C++
/// FFI) for the *full* list of outward face normals meeting at the closest
/// point.  Each returned plane is accumulated into the cell's QEF in
/// addition to the primary gradient, letting the solver locate the sharp
/// edge / corner exactly.
///
/// Arguments:
/// - `pos`         — intersection position in model space `[x, y, z]`.
/// - `cell_extent` — the containing cell's edge-length extent in model
///                   space `[x, y, z]`.  Callers may use this to scope any
///                   internal spatial query.
///
/// Return value: a list of `[nx, ny, nz, offset_d]` plane equations.  Each
/// entry is fed to the QEF via an additional `add_intersection` call at the
/// given position with the given outward normal.  Returning an empty vec is
/// equivalent to having no hook installed for this cell.
///
/// The closure is shared across rayon workers during parallel octree
/// construction, hence `Send + Sync`.
pub type FeaturesFn =
    Arc<dyn Fn([f32; 3], [f32; 3]) -> Vec<[f32; 4]> + Send + Sync>;

////////////////////////////////////////////////////////////////////////////////

/// An indexed 3D mesh
#[derive(Default, Debug)]
pub struct Mesh {
    /// Triangles, as indexes into [`self.vertices`](Self::vertices)
    pub triangles: Vec<nalgebra::Vector3<usize>>,
    /// Vertex positions
    pub vertices: Vec<nalgebra::Vector3<f32>>,
}

impl Mesh {
    /// Builds a new mesh
    pub fn new() -> Self {
        Self::default()
    }
}

/// Settings when building an octree and mesh
pub struct Settings<'a> {
    /// Depth to recurse in the octree
    pub depth: u8,

    /// Viewport to provide a world-to-model transform
    pub world_to_model: nalgebra::Matrix4<f32>,

    /// Thread pool to use for rendering
    ///
    /// If this is `None`, then rendering is done in a single thread; otherwise,
    /// the provided pool is used.
    pub threads: Option<&'a ThreadPool>,

    /// Token to cancel rendering
    pub cancel: CancelToken,

    /// Optional callback supplying additional QEF plane constraints at each
    /// edge zero-crossing during MDC leaf vertex placement.
    ///
    /// When `None` (the default), the QEF is solved using only the single
    /// gradient Fidget's grad-slice evaluator produces at each edge — the
    /// original single-normal behaviour, unchanged from upstream.
    ///
    /// When `Some(...)`, the closure is invoked at each edge intersection
    /// with the position and cell extent, and every returned plane is
    /// accumulated into the cell's QEF in addition to the primary gradient.
    /// See [`FeaturesFn`] for full semantics.
    pub features_at: Option<FeaturesFn>,
}

impl Default for Settings<'_> {
    fn default() -> Self {
        Self {
            depth: 3,
            world_to_model: nalgebra::Matrix4::identity(),
            threads: Some(&ThreadPool::Global),
            cancel: CancelToken::new(),
            features_at: None,
        }
    }
}
