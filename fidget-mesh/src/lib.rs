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
/// Counters describing how the mesher placed its vertices
pub mod stats;

use fidget_core::render::{CancelToken, ThreadPool};

#[doc(hidden)]
pub mod types;

// Re-export the main Octree type as public
pub use octree::Octree;

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

/// How each dual polygon is turned into triangles
///
/// See the comment above the polygon emission in `dc.rs` for the trade-offs.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Triangulation {
    /// A fan of triangles about the point where the surface crosses the
    /// edge (upstream Fidget; the envelope construction of Ju and Udeshi,
    /// 2006).  Up to four triangles per polygon, none of which can intersect
    /// a triangle of another polygon when vertices stay inside their cells.
    Fan,
    /// Two triangles along the more coplanar diagonal (libfive's rule).
    /// Half the triangles of the fan.  The default: with the bounded solve
    /// it matches the fan's cleanliness where vertices stay in their cells,
    /// and where a feature vertex is allowed out it folds less than the fan.
    #[default]
    QuadSplit,
    /// The quad split when both triangles of a diagonal face along the edge
    /// as the surface does and the edge pierces one of them, the fan
    /// otherwise.
    Hybrid,
}

/// Settings when building an octree and mesh
pub struct Settings<'a> {
    /// Depth to recurse in the octree
    pub depth: u8,

    /// How each dual polygon is turned into triangles
    pub triangulation: Triangulation,

    /// Whether cell vertices are kept inside their cells (the bounded QEF
    /// solve).  `false` reproduces the earlier behaviour, where a vertex
    /// may land anywhere the planes through its edge crossings meet; kept
    /// for comparison only.
    pub bounded: bool,

    /// Whether to merge groups of leaf cells whose surface a single vertex
    /// describes well enough (adaptive simplification).  Merged cells sit
    /// next to finer ones, and the polygons across that boundary are the
    /// ones that can still fold; `false` keeps every leaf at full depth.
    pub collapse: bool,

    /// Viewport to provide a world-to-model transform
    pub world_to_model: nalgebra::Matrix4<f32>,

    /// Thread pool to use for rendering
    ///
    /// If this is `None`, then rendering is done in a single thread; otherwise,
    /// the provided pool is used.
    pub threads: Option<&'a ThreadPool>,

    /// Token to cancel rendering
    pub cancel: CancelToken,
}

impl Default for Settings<'_> {
    fn default() -> Self {
        Self {
            depth: 3,
            triangulation: Triangulation::default(),
            collapse: true,
            bounded: true,
            world_to_model: nalgebra::Matrix4::identity(),
            threads: Some(&ThreadPool::Global),
            cancel: CancelToken::new(),
        }
    }
}
