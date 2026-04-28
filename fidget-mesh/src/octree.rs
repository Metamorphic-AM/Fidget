//! An octree data structure and implementation of Manifold Dual Contouring

use super::{
    Mesh, Settings,
    builder::MeshBuilder,
    cell::{Cell, CellIndex, CellVertex, Leaf},
    codegen::CELL_TO_VERT_TO_EDGES,
    frame::Frame,
    qef::QuadraticErrorSolver,
    types::{Axis, CellMask, Corner, Edge},
};
use fidget_core::{
    eval::Function,
    render::{CancelToken, RenderHandle, RenderHints, ThreadPool},
    shape::{Shape, ShapeBulkEval, ShapeTracingEval, ShapeVars},
    types::Grad,
};
use std::collections::VecDeque;

/// Octree storing occupancy and vertex positions for Manifold Dual Contouring
#[derive(Debug)]
pub struct Octree {
    pub(crate) root: Cell<3>,
    pub(crate) cells: Vec<[Cell<3>; 8]>,

    /// Cell vertices, given as positions within the cell
    ///
    /// This is indexed by cell leaf index; the exact shape depends heavily on
    /// the number of intersections and vertices within each leaf.
    pub(crate) verts: Vec<CellVertex<3>>,
}

impl Octree {
    /// Builds an octree with the root marked as invalid
    pub(crate) fn new() -> Self {
        Octree {
            root: Cell::Invalid,
            cells: vec![],
            verts: vec![],
        }
    }

    /// Builds an octree to the given depth, with user-provided variables
    ///
    /// The shape is evaluated on the region specified by `settings.bounds`.
    ///
    /// Returns `None` if processing is cancelled by the [`CancelToken`] in
    /// [`Settings`].
    pub fn build_with_vars<F: Function + RenderHints + Clone>(
        shape: &Shape<F>,
        vars: &ShapeVars<f32>,
        settings: &Settings,
    ) -> Option<Self> {
        // Transform the shape given our world-to-model matrix
        let t = settings.world_to_model;
        if t == nalgebra::Matrix4::identity() {
            Self::build_inner(shape, vars, settings)
        } else {
            let shape = shape.with_transform(t);
            let mut out = Self::build_inner(&shape, vars, settings)?;

            // Apply the transform from [-1, +1] back to model space
            for v in &mut out.verts {
                let p: nalgebra::Point3<f32> = v.pos.into();
                let q = t.transform_point(&p);
                v.pos = q.coords;
            }
            Some(out)
        }
    }

    /// Builds an octree to the given depth
    ///
    /// If the shape uses variables other than `x`, `y`, `z`, then
    /// [`build_with_vars`](Octree::build_with_vars) should be used instead (and
    /// this function will return an error).
    ///
    /// The shape is evaluated on the region specified by `settings.bounds`.
    pub fn build<F: Function + RenderHints + Clone>(
        shape: &Shape<F>,
        settings: &Settings,
    ) -> Option<Self> {
        Self::build_with_vars(shape, &ShapeVars::new(), settings)
    }

    fn build_inner<F: Function + RenderHints + Clone, T: Sync>(
        shape: &Shape<F, T>,
        vars: &ShapeVars<f32>,
        settings: &Settings,
    ) -> Option<Self> {
        if let Some(threads) = settings.threads {
            Self::build_inner_mt(
                shape,
                vars,
                settings.depth,
                &settings.cancel,
                threads,
            )
        } else {
            let mut eval = RenderHandle::new(shape.clone());
            let mut out = OctreeBuilder::new();
            let mut hermite = LeafHermiteData::default();
            if !out.recurse(
                &mut eval,
                vars,
                CellIndex::default(),
                settings.depth,
                &settings.cancel,
                &mut hermite,
                false, // in_batch — no outer batch window
                None,  // parent_ctx — root call, no parent octree row
            ) {
                return None;
            }
            // Z1: final flush catches any deferred leaves that weren't resolved
            // by an inner simplification-boundary flush (e.g. trees shallower
            // than the first simplification depth).
            if !out.flush_deferred_leaves(
                &mut eval,
                vars,
                &settings.cancel,
            ) {
                return None;
            }
            Some(out.octree)
        }
    }

    /// Multithreaded constructor
    fn build_inner_mt<F: Function + RenderHints + Clone, T: Sync>(
        shape: &Shape<F, T>,
        vars: &ShapeVars<f32>,
        max_depth: u8,
        cancel: &CancelToken,
        threads: &ThreadPool,
    ) -> Option<Self> {
        let mut root = Octree::new();
        let mut todo = VecDeque::new();
        todo.push_back(CellIndex::<3>::default());
        let mut fixup = vec![];
        let mut hermites = vec![];

        // We want a number of tasks that's significantly larger than our thread
        // count, so that we can fully saturate all cores even if tasks take
        // different amounts of time.
        let target_count =
            (8usize.pow(u32::from(max_depth))).min(threads.thread_count() * 10);
        while todo.len() < target_count {
            let next = todo.pop_front().unwrap();

            // Reserve new cells for the 8x children
            let index = root.cells.len();
            root.cells.push([Cell::Invalid; 8]);
            hermites.push([LeafHermiteData::default(); 8]);
            for i in Corner::<3>::iter() {
                let cell = next.child(index, i);
                todo.push_back(cell);
            }
            fixup.push((next, index));
        }

        use rayon::prelude::*;

        struct Output {
            cell: CellIndex<3>,
            octree: Octree,
            hermite: LeafHermiteData,
        }
        let mut rh = RenderHandle::new(shape.clone());
        let _ = rh.i_tape(&mut vec![]); // pre-populate interval tape
        let out = threads.run(|| {
            todo.par_iter()
                .map_init(
                    || {
                        let builder = OctreeBuilder::new();
                        (builder, rh.clone())
                    },
                    |(builder, eval), cell| {
                        let mut hermite = LeafHermiteData::default();
                        // Patch our cell so that it builds at index 0
                        let local_cell = CellIndex {
                            index: None,
                            ..*cell
                        };
                        if !builder.recurse(
                            eval,
                            vars,
                            local_cell,
                            max_depth,
                            cancel,
                            &mut hermite,
                            false, // in_batch — no outer batch window
                            None,  // parent_ctx — worker root, no parent row
                        ) {
                            return None;
                        }
                        // Z1: final flush catches any deferred leaves that
                        // weren't resolved by an inner simplification-boundary
                        // flush.
                        if !builder.flush_deferred_leaves(eval, vars, cancel) {
                            return None;
                        }
                        let octree = std::mem::replace(
                            &mut builder.octree,
                            Octree::new(),
                        );
                        Some(Output {
                            octree,
                            cell: *cell,
                            hermite,
                        })
                    },
                )
                .collect::<Option<Vec<_>>>()
        })?;

        // Copy hermite data into arrays, and compute cumulative offsets
        let mut cell_offsets = vec![root.cells.len()];
        let mut vert_offsets = vec![0];
        for o in &out {
            let (i, j) = o.cell.index.unwrap();
            hermites[i][j as usize] = o.hermite;
            let c = cell_offsets.last().unwrap() + o.octree.cells.len();
            cell_offsets.push(c);
            let v = vert_offsets.last().unwrap() + o.octree.verts.len();
            vert_offsets.push(v);
        }
        root.cells.reserve(*cell_offsets.last().unwrap());
        root.verts.reserve(*vert_offsets.last().unwrap());

        for (i, o) in out.into_iter().enumerate() {
            assert_eq!(cell_offsets[i], root.cells.len());
            assert_eq!(vert_offsets[i], root.verts.len());
            let remap_cell = |c| match c {
                Cell::Leaf(Leaf { mask, index }) => Cell::Leaf(Leaf {
                    mask,
                    index: index + vert_offsets[i],
                }),
                Cell::Branch { index } => Cell::Branch {
                    index: index + cell_offsets[i],
                },
                Cell::Full | Cell::Empty => c,
                Cell::Invalid => panic!(),
            };
            root.cells.extend(
                o.octree.cells.into_iter().map(|cs| cs.map(remap_cell)),
            );
            root.verts.extend(o.octree.verts);
            root[o.cell] = remap_cell(o.octree.root);
        }

        // Walk back up the tree, merging cells as we go
        for (cell, index) in fixup.into_iter().rev() {
            let h = hermites[index];
            root[cell] = root.check_done(
                cell,
                index,
                h,
                cell.index
                    .map(|(i, j)| &mut hermites[i][j as usize])
                    .unwrap_or(&mut LeafHermiteData::default()),
            );
        }
        Some(root)
    }

    /// Recursively walks the dual of the octree, building a mesh
    pub fn walk_dual(&self) -> Mesh {
        let mut mesh = MeshBuilder::default();

        mesh.cell(self, CellIndex::default());
        mesh.take()
    }

    pub(crate) fn is_leaf(&self, cell: CellIndex<3>) -> bool {
        match self[cell] {
            Cell::Leaf(..) | Cell::Full | Cell::Empty => true,
            Cell::Branch { .. } => false,
            Cell::Invalid => panic!(),
        }
    }

    /// Looks up the given child of a cell.
    ///
    /// If the cell is a leaf node, returns that cell instead.
    ///
    /// # Panics
    /// If the cell is [`Invalid`](Cell::Invalid)
    pub(crate) fn child<C: Into<Corner<3>>>(
        &self,
        cell: CellIndex<3>,
        child: C,
    ) -> CellIndex<3> {
        let child = child.into();

        match self[cell] {
            Cell::Leaf { .. } | Cell::Full | Cell::Empty => cell,
            Cell::Branch { index } => cell.child(index, child),
            Cell::Invalid => panic!(),
        }
    }

    /// Checks the set of 8 children starting at the given index for completion
    ///
    /// If all are empty or full, then pro-actively collapses the cells (freeing
    /// them if they're at the tail end of the array).
    fn check_done(
        &mut self,
        cell: CellIndex<3>,
        index: usize,
        hermite_data: [LeafHermiteData; 8],
        hermite: &mut LeafHermiteData, // output
    ) -> Cell<3> {
        // Check out the children
        let mut full_count = 0;
        let mut empty_count = 0;
        for i in 0..8 {
            match self.cells[index][i] {
                Cell::Invalid => {
                    panic!("found invalid cell during meshing")
                }
                Cell::Full => {
                    full_count += 1;
                }
                Cell::Empty => {
                    empty_count += 1;
                }
                Cell::Branch { .. } => return Cell::Branch { index },
                Cell::Leaf(Leaf { .. }) => (),
            }
        }

        // If all of the branches are empty or full, then we're going to
        // record an Empty or Full cell in the parent and don't need the
        // 8x children.  Drop them by resizing the array
        let out = if full_count == 8 {
            Cell::Full
        } else if empty_count == 8 {
            Cell::Empty
        } else if let Some(leaf) =
            self.try_collapse(cell, index, hermite_data, hermite)
        {
            Cell::Leaf(leaf)
        } else {
            Cell::Branch { index }
        };

        // If we can collapse the cell, then we'll be recording an Empty / Full
        // / Leaf cell in the parent and don't need the 8x children.
        //
        // We can drop them if they happen to be at the tail end of the
        // octree; otherwise, we'll satisfy ourselves with setting them to
        // invalid.  We will always be at the tail end of the octree during
        // single-threaded evaluation, it's only during merging octrees from
        // multiple threads that we'd have cells midway through the array.
        if !matches!(out, Cell::Branch { .. }) {
            if index == self.cells.len() - 1 {
                self.cells.resize(index, [Cell::Invalid; 8]);
            } else {
                self.cells[index] = [Cell::Invalid; 8];
            }
        }
        out
    }

    /// Try to collapse the given cell
    ///
    /// Writes to `self.verts` and returns the leaf if successful; otherwise
    /// returns `None`.
    fn try_collapse(
        &mut self,
        cell: CellIndex<3>,
        index: usize,
        hermite_data: [LeafHermiteData; 8],
        hermite: &mut LeafHermiteData, // output
    ) -> Option<Leaf<3>> {
        let mask = self.collapsible(index)?;
        *hermite = LeafHermiteData::merge(hermite_data)?;

        // Empty / full cells should never be produced here.  The
        // only way to get an empty / full cell is for all eight
        // corners to be empty / full; if that was the case, then
        // either:
        //
        // - The interior vertices match, in which case this should
        //   have been collapsed into a single empty / full cell
        // - The interior vertices *do not* match, in which case the
        //   cell should not be marked as collapsible.
        let (pos, new_err) = hermite.solve();
        if new_err >= hermite.qef_err * 2.0 || !cell.bounds.contains(pos) {
            return None;
        }
        // Record the newly-collapsed leaf
        hermite.qef_err = new_err;

        let index = self.verts.len();
        self.verts.push(pos);

        // Install cell intersections, of which there must only
        // be one (since we only collapse manifold cells)
        let edges = &CELL_TO_VERT_TO_EDGES[mask.index()];
        debug_assert_eq!(edges.len(), 1);
        for e in edges[0] {
            let i = hermite.intersections[e.to_undirected().index()];
            self.verts.push(CellVertex { pos: i.pos.xyz() });
        }

        Some(Leaf { mask, index })
    }

    /// Checks whether the set of 8 cells beginning at `root` can be collapsed.
    ///
    /// Only topology is checked, based on the three predicates from "Dual
    /// Contouring of Hermite Data" (Ju et al, 2002), §4.1
    pub(crate) fn collapsible(&self, root: usize) -> Option<CellMask<3>> {
        // Copy cells to a local variable for simplicity
        let cells = self.cells[root];

        let mut mask = 0;
        for (i, &c) in cells.iter().enumerate() {
            let b = match c {
                Cell::Leaf(Leaf { mask, .. }) => {
                    if CELL_TO_VERT_TO_EDGES[mask.index()].len() > 1 {
                        return None;
                    }
                    (mask.index() & (1 << i) != 0) as u8
                }
                Cell::Empty => 0,
                Cell::Full => 1,
                Cell::Branch { .. } => return None,
                Cell::Invalid => panic!(),
            };
            mask |= b << i;
        }

        use super::frame::{XYZ, YZX, ZXY};
        for (t, u, v) in [XYZ::frame(), YZX::frame(), ZXY::frame()] {
            //  - The sign in the middle of a coarse edge must agree with the
            //    sign of at least one of the edge’s two endpoints.
            for i in 0..4 {
                let a = (u * ((i & 1) != 0)) | (v * ((i & 2) != 0));
                let b = a | t;
                let center = cells[a.index()].corner(b);

                if [a, b]
                    .iter()
                    .all(|v| ((mask & (1 << v.index())) != 0) != center)
                {
                    return None;
                }
            }

            //  - The sign in the middle of a coarse face must agree with the
            //    sign of at least one of the face’s four corners.
            for i in 0..2 {
                let a: Corner<3> = (t * (i & 1 == 0)).into();
                let b = a | u;
                let c = a | v;
                let d = a | u | v;

                let center = cells[a.index()].corner(d);

                if [a, b, c, d]
                    .iter()
                    .all(|v| ((mask & (1 << v.index())) != 0) != center)
                {
                    return None;
                }
            }
            //  - The sign in the middle of a coarse cube must agree with the
            //    sign of at least one of the cube’s eight corners.
            for _i in 0..1 {
                // Doing this in the t,u,v loop isn't strictly necessary, but it
                // preserves the structure nicely.
                let center = cells[0].corner(t | u | v);
                if (0..8).all(|v| ((mask & (1 << v)) != 0) != center) {
                    return None;
                }
            }
        }

        // The outer cell must not be empty or full at this point; if it was
        // empty or full and the other conditions had been met, then it should
        // have been collapsed already.
        debug_assert_ne!(mask, 255);
        debug_assert_ne!(mask, 0);

        // TODO: this check may not be necessary, because we're doing *manifold*
        // dual contouring; the collapsed cell can have multiple vertices.
        if CELL_TO_VERT_TO_EDGES[mask as usize].len() == 1 {
            Some(CellMask::new(mask))
        } else {
            None
        }
    }
}

impl std::ops::Index<CellIndex<3>> for Octree {
    type Output = Cell<3>;

    fn index(&self, i: CellIndex<3>) -> &Self::Output {
        match i.index {
            None => &self.root,
            Some((i, j)) => &self.cells[i][j as usize],
        }
    }
}

impl std::ops::IndexMut<CellIndex<3>> for Octree {
    fn index_mut(&mut self, i: CellIndex<3>) -> &mut Self::Output {
        match i.index {
            None => &mut self.root,
            Some((i, j)) => &mut self.cells[i][j as usize],
        }
    }
}

/// Z1: a leaf-cell whose evaluation has been deferred so that many leaves can
/// be processed in a single pipelined pass (corner batch → edge-search batch ×4
/// → gradient batch → per-cell QEF solve).
///
/// The queued cell has `Cell::Invalid` written to `self.octree[cell]` as a
/// placeholder; the flush pass replaces it with the real `Cell::Empty`,
/// `Cell::Full`, or `Cell::Leaf(..)` value.
///
/// `parent_idx` and `corner_idx` pin the queued cell to its slot in
/// `self.octree.cells[parent_idx][corner_idx]`, and are the address that
/// flush phase 5 uses to write this leaf's `LeafHermiteData` into
/// `self.hermite_cells[parent_idx][corner_idx]`.  That hermite entry is
/// consumed later by `collapse_subtree()` so `try_collapse` can run
/// bottom-up across the flushed subtree, restoring MDC's adaptive
/// coarsening on top of the batched-evaluation win.
#[derive(Copy, Clone, Debug)]
struct DeferredLeaf {
    cell: CellIndex<3>,
    parent_idx: usize,
    corner_idx: u8,
}

/// Data structure for an under-construction octree
//
// Note: cannot `#[derive(Debug)]` because the `ShapeBulkEval<F::*>` and
// `ShapeTracingEval<F::*>` evaluator fields are generic over `F`'s
// associated types, which don't carry `Debug` bounds — `#[derive(Debug)]`
// would silently propagate `where F::FloatSliceEval: Debug` (etc.) clauses
// to every caller.  The manual impl below skips those fields via
// `finish_non_exhaustive` so the struct can still participate in tracing /
// `{:?}` debug prints wherever callers reach for them.
pub(crate) struct OctreeBuilder<F: Function + RenderHints> {
    /// In-construction octree
    pub(crate) octree: Octree,

    eval_float_slice: ShapeBulkEval<F::FloatSliceEval>,
    eval_interval: ShapeTracingEval<F::IntervalEval>,
    eval_grad_slice: ShapeBulkEval<F::GradSliceEval>,

    tape_storage: Vec<F::TapeStorage>,
    shape_storage: Vec<F::Storage>,
    workspace: F::Workspace,

    /// Z1: queue of leaves pending a pipelined batch evaluation.
    ///
    /// Filled by `recurse()` at `cell.depth == max_depth` when a batching
    /// window is active; drained by `flush_deferred_leaves()` at the
    /// simplification boundary that opened the window.
    deferred_leaves: Vec<DeferredLeaf>,

    /// Z1: per-branch hermite data, sharing the row-index space with
    /// `self.octree.cells`.  Used only inside a batching window, because
    /// scalar recursion propagates hermite via stack-local `&mut
    /// LeafHermiteData` parameters and doesn't need this storage.  For each
    /// parent-branch `p` and corner `c` in 0..8:
    ///
    /// * If `self.octree.cells[p][c]` is `Cell::Leaf`, then
    ///   `self.hermite_cells[p][c]` holds that leaf's real hermite data
    ///   (populated by `flush_deferred_leaves` phase 5).
    /// * If it's `Cell::Full` or `Cell::Empty`, the hermite entry stays at
    ///   default, which is the correct contribution for `merge()` (empty
    ///   intersections, `qef_err = QEF_ERR_EMPTY`).
    /// * If it's `Cell::Branch`, the hermite entry is unused; the inner
    ///   branch's hermite is produced on-demand by a recursive
    ///   `collapse_subtree` call.
    ///
    /// Growth strategy is lazy: `recurse` does NOT push a row here alongside
    /// every `octree.cells.push(..)` because each `[LeafHermiteData; 8]`
    /// entry is ~7 KB and scalar-only builds would pay that cost for
    /// nothing.  Instead, `flush_deferred_leaves` resizes this `Vec` up to
    /// `octree.cells.len()` at the start of each flush.  Every branch
    /// inside the active window exists by the time flush runs (flush is
    /// called at the window root, after all 8 children recursed), so the
    /// resize covers every index that the subsequent `collapse_subtree`
    /// pass will read.  Entries outside an active batching window are
    /// meaningless and should not be read.
    hermite_cells: Vec<[LeafHermiteData; 8]>,
}

impl<F: Function + RenderHints> std::fmt::Debug for OctreeBuilder<F> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        // Only print the fields whose types don't depend on `F`'s associated
        // types — `ShapeBulkEval<F::FloatSliceEval>` and friends don't carry
        // `Debug` bounds on their generics, so explicitly formatting them
        // would force us to add `where F::FloatSliceEval: Debug` (etc.)
        // clauses here and propagate them to every caller.  The original
        // `#[derive(Debug)]` silently emitted those bounds into its generated
        // impl, but our hand-written impl doesn't get that treatment for
        // free.  These fields are internal evaluator scratch state and
        // wouldn't be meaningful in debug output anyway; `finish_non_exhaustive`
        // signals the partial view.
        f.debug_struct("OctreeBuilder")
            .field("octree", &self.octree)
            .field("deferred_leaves_len", &self.deferred_leaves.len())
            .field("hermite_cells_len", &self.hermite_cells.len())
            .finish_non_exhaustive()
    }
}

impl<F: Function + RenderHints> Default for OctreeBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Function + RenderHints> OctreeBuilder<F> {
    /// Builds a new octree builder, which allocates data for 8 root cells
    pub(crate) fn new() -> Self {
        Self {
            octree: Octree::new(),
            eval_float_slice: Shape::<F>::new_float_slice_eval(),
            eval_grad_slice: Shape::<F>::new_grad_slice_eval(),
            eval_interval: Shape::<F>::new_interval_eval(),
            tape_storage: vec![],
            shape_storage: vec![],
            workspace: Default::default(),
            deferred_leaves: vec![],
            hermite_cells: vec![],
        }
    }

    /// Recurse down the octree, building the given cell
    ///
    /// Writes to `self.o.cells[cell]`, which must be reserved
    ///
    /// If a leaf is written, then `hermite` is populated
    ///
    /// The `in_batch` flag indicates whether any ancestor has opened a
    /// batching window via `simplify_tree_during_meshing`. When true, leaves
    /// at `max_depth` are pushed onto `self.deferred_leaves` instead of being
    /// processed inline; they are flushed at the recurse frame that opened
    /// the window.
    ///
    /// `parent_ctx` is `Some((parent_idx, corner_idx))` whenever this frame
    /// was called from another recurse frame that had reserved a row in
    /// `self.octree.cells` for its 8 children.  It is the address of this
    /// frame's slot in its parent's cells row, and `DeferredLeaf` entries
    /// carry a copy so that `flush_deferred_leaves` knows which
    /// `self.hermite_cells[parent_idx][corner_idx]` slot to populate.  The
    /// single-threaded root call and each multithreaded worker's root call
    /// pass `None` (they have no parent octree row to write back to).
    ///
    /// Returns `true` if recursion completed normally, or `false` if it was
    /// cancelled.
    #[must_use]
    fn recurse<T>(
        &mut self,
        eval: &mut RenderHandle<F, T>,
        vars: &ShapeVars<f32>,
        cell: CellIndex<3>,
        max_depth: u8,
        cancel: &CancelToken,
        hermite: &mut LeafHermiteData,
        in_batch: bool,
        parent_ctx: Option<(usize, u8)>,
    ) -> bool {
        if cancel.is_cancelled() {
            return false;
        }
        let (i, r) = self
            .eval_interval
            .eval_v(
                eval.i_tape(&mut self.tape_storage),
                cell.bounds[crate::types::X],
                cell.bounds[crate::types::Y],
                cell.bounds[crate::types::Z],
                vars,
            )
            .unwrap();
        self.octree[cell] = if i.upper() < 0.0 {
            Cell::Full
        } else if i.lower() > 0.0 {
            Cell::Empty
        } else {
            // Z1: this frame opens a batching window if `simplify_tree_during_meshing`
            // fires here AND no ancestor has already opened one. The frame that
            // opens the window owns the flush at the end of its subtree.
            //
            // We only open a window if we actually have descendants to batch
            // across — if this frame is itself the max-depth leaf, there's
            // nothing to amortise, and deferring would leave a Cell::Invalid
            // placeholder that an ancestor's scalar `check_done` would panic
            // on.
            let opens_window = F::simplify_tree_during_meshing(cell.depth)
                && !in_batch
                && cell.depth != max_depth as usize;
            let sub_tape = if F::simplify_tree_during_meshing(cell.depth) {
                if let Some(trace) = r.as_ref() {
                    eval.simplify(
                        trace,
                        &mut self.workspace,
                        &mut self.shape_storage,
                        &mut self.tape_storage,
                    )
                } else {
                    eval
                }
            } else {
                eval
            };
            let child_in_batch = in_batch || opens_window;
            if cell.depth == max_depth as usize {
                if child_in_batch {
                    // Z1: queue the leaf for pipelined batch evaluation.
                    // Write Cell::Invalid as a placeholder; flush replaces it.
                    //
                    // `parent_ctx` must be Some here: the only way to reach
                    // `child_in_batch == true` is via a parent frame that
                    // reserved an octree row and called us with that row's
                    // (index, corner) pair.
                    let (parent_idx, corner_idx) = parent_ctx
                        .expect("deferred leaf reached without a parent slot");
                    self.deferred_leaves.push(DeferredLeaf {
                        cell,
                        parent_idx,
                        corner_idx,
                    });
                    Cell::Invalid
                } else {
                    self.leaf(sub_tape, vars, cell, hermite)
                }
            } else {
                // Reserve new cells for the 8x children.  `hermite_cells` is
                // NOT grown eagerly here — doing so would allocate ~7 KB
                // per row even for scalar-only builds, which adds up fast
                // on deep meshes.  Instead, `flush_deferred_leaves` resizes
                // `hermite_cells` to match `octree.cells` right before it
                // writes; all batching-window-internal branches exist by
                // the time flush runs, so the resize covers every index
                // that `collapse_subtree` will later read.
                let index = self.octree.cells.len();
                self.octree.cells.push([Cell::Invalid; 8]);
                let mut hermite_child = [LeafHermiteData::default(); 8];
                for i in Corner::<3>::iter() {
                    let child = cell.child(index, i);
                    if !self.recurse(
                        sub_tape,
                        vars,
                        child,
                        max_depth,
                        cancel,
                        &mut hermite_child[i.index()],
                        child_in_batch,
                        Some((index, i.index() as u8)),
                    ) {
                        return false;
                    }
                }

                // Z1: if we opened the batching window here, flush all queued
                // leaves under this subtree before anything looks at the
                // cells row — after flush, `self.octree.cells[index][..]` no
                // longer contains Cell::Invalid placeholders for this
                // subtree's leaves and `self.hermite_cells[index][..]` has
                // real hermite data for each flushed leaf.
                if opens_window
                    && !self.flush_deferred_leaves(sub_tape, vars, cancel)
                {
                    return false;
                }

                if opens_window {
                    // Window root: the subtree is fully resolved now, so we
                    // can run the normal bottom-up try_collapse pass over
                    // it.  `collapse_subtree` writes the resulting hermite
                    // into `*hermite` for this frame, exactly matching the
                    // scalar `check_done` contract.  This is what restores
                    // MDC's adaptive coarsening on top of Z1's batching —
                    // without it, every max-depth cell that spans the
                    // surface sticks around as its own leaf and the mesh
                    // comes out uniformly dense.
                    self.collapse_subtree(cell, index, hermite)
                } else if in_batch {
                    // Strictly inside a batching window (neither root nor
                    // scalar).  Leaves may still be deferred somewhere under
                    // us, so we can't call check_done yet — the window root
                    // will run collapse_subtree over its own subtree and
                    // will recurse down through us at that time, at which
                    // point our row in self.octree.cells will be fully
                    // populated.  Record ourselves as a plain Branch for
                    // now; hermite flows through self.hermite_cells rather
                    // than through `*hermite`.
                    Cell::Branch { index }
                } else {
                    // Scalar path: business as usual.
                    self.octree.check_done(cell, index, hermite_child, hermite)
                }
            }
        };
        true
    }

    /// Z1: bottom-up `try_collapse` traversal of a subtree whose leaves are
    /// all resolved (no `Cell::Invalid` placeholders).
    ///
    /// Called by `recurse` at the frame that opens a batching window, once
    /// `flush_deferred_leaves` has run.  Walks the subtree rooted at
    /// `index` in post-order: for each `Cell::Branch` child at corner `c`,
    /// recursively collapses the sub-subtree (which overwrites
    /// `self.octree.cells[index][c]` with the collapsed result); for each
    /// `Cell::Leaf` child, pulls hermite from
    /// `self.hermite_cells[index][c]`; for `Cell::Full` / `Cell::Empty`,
    /// uses default hermite (zero contribution to `merge`).
    ///
    /// The gathered `[LeafHermiteData; 8]` is then handed to the normal
    /// `Octree::check_done`, which runs the vanilla `try_collapse` logic —
    /// including QEF-error gating and the tail-of-array cleanup — and
    /// writes the merged result into `*hermite_out`.  The returned
    /// `Cell<3>` is the new value for this frame's slot in its parent's
    /// row; the caller is responsible for writing it.
    ///
    /// Never descends into leaves, never re-evaluates the shape — it is a
    /// pure post-processing pass over the octree and hermite arrays.
    fn collapse_subtree(
        &mut self,
        cell: CellIndex<3>,
        index: usize,
        hermite_out: &mut LeafHermiteData,
    ) -> Cell<3> {
        let mut hermite_child = [LeafHermiteData::default(); 8];
        for i in 0..8 {
            match self.octree.cells[index][i] {
                Cell::Branch { index: child_idx } => {
                    let child_cell =
                        cell.child(index, Corner::<3>::new(i as u8));
                    let mut child_hermite = LeafHermiteData::default();
                    let new_cell = self.collapse_subtree(
                        child_cell,
                        child_idx,
                        &mut child_hermite,
                    );
                    self.octree.cells[index][i] = new_cell;
                    hermite_child[i] = child_hermite;
                }
                Cell::Leaf(..) => {
                    hermite_child[i] = self.hermite_cells[index][i];
                }
                Cell::Full | Cell::Empty => {
                    // Default-constructed hermite: no intersections, no
                    // face QEF contributions, `qef_err = QEF_ERR_EMPTY`.
                    // That's exactly how `merge` expects empty / full
                    // children to present.
                }
                Cell::Invalid => {
                    panic!(
                        "collapse_subtree: Cell::Invalid at \
                         cells[{index}][{i}] — flush must run first"
                    )
                }
            }
        }

        // `cells[index]` now contains this row's post-collapse state
        // (every Cell::Branch has been replaced with its collapsed
        // Leaf/Full/Empty result; Cell::Leaf/Full/Empty entries are
        // unchanged).  Save it before calling `check_done`, because
        // `check_done` resets `cells[index]` to `[Cell::Invalid; 8]`
        // or truncates whenever it decides to collapse (`out !=
        // Branch`).  That reset is harmless in pre-Z1 scalar DFS —
        // the not-at-tail reset branch never fires, because scalar
        // DFS only collapses when all 8 children are leaf-ish and
        // therefore no descendant rows exist past this one — but it
        // IS fatal here.  `collapse_subtree` is a bottom-up pass
        // over a fully-built subtree, so sibling branches that
        // didn't collapse keep their rows alive, meaning our own
        // row is usually NOT at the tail and `check_done` takes the
        // write-`[Invalid; 8]` branch.  Those Invalid rows then
        // escape the worker's octree and crash the MT-merge
        // `remap_cell` pass, which walks every row linearly and
        // panics on any `Cell::Invalid` it sees.
        //
        // The row is orphan after we return (the caller overwrites
        // its parent's slot with `out`, which is non-Branch), so
        // restoring the saved content doesn't affect tree
        // traversal — it just keeps linear walks from blowing up.
        let saved_row = self.octree.cells[index];
        let out = self.octree.check_done(cell, index, hermite_child, hermite_out);
        if !matches!(out, Cell::Branch { .. }) {
            // `check_done` may have truncated; grow back if so.
            if self.octree.cells.len() <= index {
                self.octree
                    .cells
                    .resize(index + 1, [Cell::Invalid; 8]);
            }
            self.octree.cells[index] = saved_row;
        }
        out
    }

    /// Evaluates the given leaf
    ///
    /// Writes the leaf vertex to `self.o.verts`, hermite data to
    /// `self.hermite`, and the leaf data to `self.leafs`.  Does **not** write
    /// anything to `self.o.cells`; the cell is returned instead.
    fn leaf<T>(
        &mut self,
        eval: &mut RenderHandle<F, T>,
        vars: &ShapeVars<f32>,
        cell: CellIndex<3>,
        hermite_cell: &mut LeafHermiteData,
    ) -> Cell<3> {
        let mut xs = [0.0; 8];
        let mut ys = [0.0; 8];
        let mut zs = [0.0; 8];
        for i in Corner::<3>::iter() {
            let [x, y, z] = cell.corner(i);
            xs[i.index()] = x;
            ys[i.index()] = y;
            zs[i.index()] = z;
        }

        let out = self
            .eval_float_slice
            .eval_v(eval.f_tape(&mut self.tape_storage), &xs, &ys, &zs, vars)
            .unwrap();
        debug_assert_eq!(out.len(), 8);

        // Build a mask of active corners, which determines cell
        // topology / vertex count / active edges / etc.
        let mask = out
            .iter()
            .enumerate()
            .filter(|(_i, v)| **v < 0.0)
            .fold(0, |acc, (i, _v)| acc | (1 << i));

        // Early exit if the cell is completely empty or full
        if mask == 0 {
            return Cell::Empty;
        } else if mask == 255 {
            return Cell::Full;
        }

        let mask = CellMask::new(mask);

        // Start and endpoints in 3D space for intersection searches
        let mut start = [nalgebra::Vector3::zeros(); 12];
        let mut end = [nalgebra::Vector3::zeros(); 12];
        let mut edge_count = 0;

        // Convert from the corner mask to start and end-points in 3D space,
        // relative to the cell bounds (i.e. as a `Matrix3<u16>`).
        for vs in CELL_TO_VERT_TO_EDGES[mask.index()].iter() {
            for e in *vs {
                // Find the axis that's being used by this edge
                let axis = e.start().index() ^ e.end().index();
                debug_assert_eq!(axis.count_ones(), 1);
                debug_assert!(axis < 8);

                // Pick a position closer to the filled side
                let (a, b) = if e.end().index() & axis != 0 {
                    (0, u16::MAX)
                } else {
                    (u16::MAX, 0)
                };

                // Convert the intersection to a 3D position
                let mut v = nalgebra::Vector3::zeros();
                let i = (axis.trailing_zeros() + 1) % 3;
                let j = (axis.trailing_zeros() + 2) % 3;
                v[i as usize] = if e.start() & Axis::new(1 << i) {
                    u16::MAX
                } else {
                    0
                };
                v[j as usize] = if e.start() & Axis::new(1 << j) {
                    u16::MAX
                } else {
                    0
                };

                v[axis.trailing_zeros() as usize] = a;
                start[edge_count] = v;
                v[axis.trailing_zeros() as usize] = b;
                end[edge_count] = v;
                edge_count += 1;
            }
        }
        // Slice off the unused sections of the arrays
        let start = &mut start[..edge_count]; // always inside
        let end = &mut end[..edge_count]; // always outside

        // Scratch arrays for edge search
        const EDGE_SEARCH_SIZE: usize = 16;
        const EDGE_SEARCH_DEPTH: usize = 4;
        let xs =
            &mut [0.0; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];
        let ys =
            &mut [0.0; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];
        let zs =
            &mut [0.0; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];

        // This part looks hairy, but it's just doing an N-ary search along each
        // edge to find the intersection point.
        for _ in 0..EDGE_SEARCH_DEPTH {
            // Populate edge search arrays
            let mut i = 0;
            for (start, end) in start.iter().zip(end.iter()) {
                for j in 0..EDGE_SEARCH_SIZE {
                    let pos = ((start.map(|i| i as u32)
                        * (EDGE_SEARCH_SIZE - j - 1) as u32)
                        + (end.map(|i| i as u32) * j as u32))
                        / ((EDGE_SEARCH_SIZE - 1) as u32);
                    debug_assert!(pos.max() <= u16::MAX.into());

                    let pos = cell.pos(pos.map(|p| p as u16));
                    xs[i] = pos.x;
                    ys[i] = pos.y;
                    zs[i] = pos.z;
                    i += 1;
                }
            }
            debug_assert_eq!(i, EDGE_SEARCH_SIZE * edge_count);

            // Do the actual evaluation
            let out = self
                .eval_float_slice
                .eval_v(eval.f_tape(&mut self.tape_storage), xs, ys, zs, vars)
                .unwrap();

            // Update start and end positions based on evaluation
            for ((start, end), search) in start
                .iter_mut()
                .zip(end.iter_mut())
                .zip(out.chunks(EDGE_SEARCH_SIZE))
            {
                // The search must be inside-to-outside
                debug_assert!(search[0] < 0.0);
                debug_assert!(search[EDGE_SEARCH_SIZE - 1] >= 0.0);
                let frac = search
                    .iter()
                    .enumerate()
                    .find(|(_i, v)| **v >= 0.0)
                    .unwrap()
                    .0;
                debug_assert!(frac > 0);
                debug_assert!(frac < EDGE_SEARCH_SIZE);

                let f = |frac| {
                    ((start.map(|i| i as u32)
                        * (EDGE_SEARCH_SIZE - frac - 1) as u32)
                        + (end.map(|i| i as u32) * frac as u32))
                        / ((EDGE_SEARCH_SIZE - 1) as u32)
                };

                let a = f(frac - 1);
                let b = f(frac);

                debug_assert!(a.max() <= u16::MAX.into());
                debug_assert!(b.max() <= u16::MAX.into());
                *start = a.map(|v| v as u16);
                *end = b.map(|v| v as u16);
            }
        }

        // Populate intersections to the average of start and end
        let intersections: arrayvec::ArrayVec<nalgebra::Vector3<u16>, 12> =
            start
                .iter()
                .zip(end.iter())
                .map(|(a, b)| {
                    ((a.map(|v| v as u32) + b.map(|v| v as u32)) / 2)
                        .map(|v| v as u16)
                })
                .collect();

        let xs = &mut [Grad::from(0.0); 12 * EDGE_SEARCH_SIZE]
            [..intersections.len()];
        let ys = &mut [Grad::from(0.0); 12 * EDGE_SEARCH_SIZE]
            [..intersections.len()];
        let zs = &mut [Grad::from(0.0); 12 * EDGE_SEARCH_SIZE]
            [..intersections.len()];

        for (i, xyz) in intersections.iter().enumerate() {
            let pos = cell.pos(*xyz);
            xs[i] = Grad::new(pos.x, 1.0, 0.0, 0.0);
            ys[i] = Grad::new(pos.y, 0.0, 1.0, 0.0);
            zs[i] = Grad::new(pos.z, 0.0, 0.0, 1.0);
        }

        // TODO: special case for cells with multiple gradients ("features")
        let grads = self
            .eval_grad_slice
            .eval_v(eval.g_tape(&mut self.tape_storage), xs, ys, zs, vars)
            .unwrap();

        let mut verts: arrayvec::ArrayVec<_, 4> = arrayvec::ArrayVec::new();
        let mut i = 0;
        for vs in CELL_TO_VERT_TO_EDGES[mask.index()].iter() {
            let mut force_point = None;
            let mut qef = QuadraticErrorSolver::new();
            for e in vs.iter() {
                let pos = nalgebra::Vector3::new(xs[i].v, ys[i].v, zs[i].v);
                let grad: nalgebra::Vector4<f32> = grads[i].into();

                // If a point has invalid gradients, then it's _probably_ on a
                // sharp feature in the mesh, so we should just snap to that
                // point specifically.  This means we don't solve the QEF, and
                // instead mark it as invalid.
                if grad.iter().any(|f| f.is_nan()) {
                    force_point = Some(pos);
                    hermite_cell.qef_err = QEF_ERR_INVALID;
                    break;
                }
                qef.add_intersection(pos, grad);

                // Record this intersection in the Hermite data for the leaf
                let edge_index = e.to_undirected().index();
                hermite_cell.intersections[edge_index] = LeafIntersection {
                    pos: nalgebra::Vector4::new(pos.x, pos.y, pos.z, 1.0),
                    grad,
                };

                i += 1;
            }

            if let Some(pos) = force_point {
                verts.push(CellVertex { pos });
            } else {
                let (pos, err) = qef.solve();
                verts.push(pos);

                // We overwrite the error here, because it's only used when
                // collapsing cells, which only occurs if there's a single
                // vertex; last-error-wins works fine in that case.
                hermite_cell.qef_err = err;
            }
        }

        let index = self.octree.verts.len();
        self.octree.verts.extend(verts);
        self.octree.verts.extend(
            intersections
                .into_iter()
                .map(|pos| CellVertex { pos: cell.pos(pos) }),
        );

        Cell::Leaf(Leaf { mask, index })
    }

    /// Z1: processes all queued leaves in a single pipelined pass.
    ///
    /// For each queued cell this function mirrors the work that the scalar
    /// `leaf()` would have done, but collects the 8-corner sample, the 4×N
    /// edge-search samples, and the N gradient samples from *every* queued
    /// leaf into three flat evaluator calls — one big corner batch, four big
    /// edge-search batches, and one big gradient batch. Per-cell QEF solve
    /// and final vertex write-back are then done in a scalar sweep at the
    /// end.
    ///
    /// The caller is responsible for invoking this with `eval` = the same
    /// simplified `RenderHandle` that was active when the cells were queued,
    /// i.e. the one at the simplification boundary that opened the batching
    /// window.
    ///
    /// Returns `false` if cancelled; `true` (and empties `deferred_leaves`)
    /// otherwise.
    #[must_use]
    fn flush_deferred_leaves<T>(
        &mut self,
        eval: &mut RenderHandle<F, T>,
        vars: &ShapeVars<f32>,
        cancel: &CancelToken,
    ) -> bool {
        // Keep `hermite_cells` aligned with `octree.cells` BEFORE the early
        // return, so the invariant "inside an active batching window,
        // `hermite_cells.len() >= octree.cells.len()`" holds unconditionally
        // once flush has been called — including when the window had no
        // mixed-sign max-depth leaves (deferred_leaves empty).
        if self.hermite_cells.len() < self.octree.cells.len() {
            self.hermite_cells
                .resize(self.octree.cells.len(), [LeafHermiteData::default(); 8]);
        }
        if self.deferred_leaves.is_empty() {
            return true;
        }
        if cancel.is_cancelled() {
            return false;
        }

        const EDGE_SEARCH_SIZE: usize = 16;
        const EDGE_SEARCH_DEPTH: usize = 4;

        let n = self.deferred_leaves.len();

        // --- Phase 1: corner batch (8 samples per queued cell) --------------
        let mut corner_xs: Vec<f32> = Vec::with_capacity(n * 8);
        let mut corner_ys: Vec<f32> = Vec::with_capacity(n * 8);
        let mut corner_zs: Vec<f32> = Vec::with_capacity(n * 8);
        for dl in &self.deferred_leaves {
            for i in Corner::<3>::iter() {
                let [x, y, z] = dl.cell.corner(i);
                corner_xs.push(x);
                corner_ys.push(y);
                corner_zs.push(z);
            }
        }
        let corner_out = self
            .eval_float_slice
            .eval_v(
                eval.f_tape(&mut self.tape_storage),
                &corner_xs,
                &corner_ys,
                &corner_zs,
                vars,
            )
            .unwrap();

        // Decide per-cell: empty / full (resolved now) vs. needs edge search.
        // `masks[i] = Some(m)` means cell `i` has mixed signs and will go
        // through edge search; `None` means it was resolved immediately.
        let mut masks: Vec<Option<CellMask<3>>> = Vec::with_capacity(n);
        for (i, dl) in self.deferred_leaves.iter().enumerate() {
            let corner_vals = &corner_out[i * 8..(i + 1) * 8];
            let mask_bits = corner_vals
                .iter()
                .enumerate()
                .filter(|(_, v)| **v < 0.0)
                .fold(0u8, |acc, (i, _)| acc | (1 << i));
            if mask_bits == 0 {
                self.octree[dl.cell] = Cell::Empty;
                masks.push(None);
            } else if mask_bits == 255 {
                self.octree[dl.cell] = Cell::Full;
                masks.push(None);
            } else {
                masks.push(Some(CellMask::new(mask_bits)));
            }
        }

        // --- Phase 2: build the flat edge list across all surviving cells ---
        // For every surviving cell, for every vertex `vs` in its topology,
        // for every edge `e` in that vertex, push one `EdgeSlot` into `edges`.
        // `per_cell_edge_bases[i]` = first-edge-index for surviving cell `i`;
        // a sentinel trailing entry makes `bases[i+1] - bases[i]` give the
        // per-cell edge count without special-casing the last cell.
        struct EdgeSlot {
            /// Index into `self.deferred_leaves` / `masks`
            cell_idx: usize,
            start: nalgebra::Vector3<u16>,
            end: nalgebra::Vector3<u16>,
        }
        let mut edges: Vec<EdgeSlot> = Vec::new();
        let mut per_cell_edge_bases: Vec<usize> = Vec::with_capacity(n + 1);
        for (i, mask_opt) in masks.iter().enumerate() {
            per_cell_edge_bases.push(edges.len());
            let Some(mask) = mask_opt else { continue };
            for vs in CELL_TO_VERT_TO_EDGES[mask.index()].iter() {
                for e in *vs {
                    let axis = e.start().index() ^ e.end().index();
                    debug_assert_eq!(axis.count_ones(), 1);
                    debug_assert!(axis < 8);

                    let (a, b) = if e.end().index() & axis != 0 {
                        (0u16, u16::MAX)
                    } else {
                        (u16::MAX, 0u16)
                    };

                    let mut v = nalgebra::Vector3::zeros();
                    let ui = (axis.trailing_zeros() + 1) % 3;
                    let vi = (axis.trailing_zeros() + 2) % 3;
                    v[ui as usize] = if e.start() & Axis::new(1 << ui) {
                        u16::MAX
                    } else {
                        0
                    };
                    v[vi as usize] = if e.start() & Axis::new(1 << vi) {
                        u16::MAX
                    } else {
                        0
                    };

                    let mut s = v;
                    s[axis.trailing_zeros() as usize] = a;
                    let mut ee = v;
                    ee[axis.trailing_zeros() as usize] = b;

                    edges.push(EdgeSlot {
                        cell_idx: i,
                        start: s,
                        end: ee,
                    });
                }
            }
        }
        per_cell_edge_bases.push(edges.len());

        if edges.is_empty() {
            // All queued cells were fully empty/full — nothing more to do.
            self.deferred_leaves.clear();
            return true;
        }

        let total_edges = edges.len();

        // --- Phase 3: 4-iteration N-ary edge search, batched ---------------
        let mut search_xs: Vec<f32> = vec![0.0; total_edges * EDGE_SEARCH_SIZE];
        let mut search_ys: Vec<f32> = vec![0.0; total_edges * EDGE_SEARCH_SIZE];
        let mut search_zs: Vec<f32> = vec![0.0; total_edges * EDGE_SEARCH_SIZE];

        for _ in 0..EDGE_SEARCH_DEPTH {
            for (ei, edge) in edges.iter().enumerate() {
                let cell = self.deferred_leaves[edge.cell_idx].cell;
                for j in 0..EDGE_SEARCH_SIZE {
                    let pos = ((edge.start.map(|i| i as u32)
                        * (EDGE_SEARCH_SIZE - j - 1) as u32)
                        + (edge.end.map(|i| i as u32) * j as u32))
                        / ((EDGE_SEARCH_SIZE - 1) as u32);
                    debug_assert!(pos.max() <= u16::MAX.into());
                    let pos_f = cell.pos(pos.map(|p| p as u16));
                    let idx = ei * EDGE_SEARCH_SIZE + j;
                    search_xs[idx] = pos_f.x;
                    search_ys[idx] = pos_f.y;
                    search_zs[idx] = pos_f.z;
                }
            }

            let out = self
                .eval_float_slice
                .eval_v(
                    eval.f_tape(&mut self.tape_storage),
                    &search_xs,
                    &search_ys,
                    &search_zs,
                    vars,
                )
                .unwrap();

            for (ei, edge) in edges.iter_mut().enumerate() {
                let search = &out[ei * EDGE_SEARCH_SIZE
                    ..(ei + 1) * EDGE_SEARCH_SIZE];
                debug_assert!(search[0] < 0.0);
                debug_assert!(search[EDGE_SEARCH_SIZE - 1] >= 0.0);
                let frac = search
                    .iter()
                    .enumerate()
                    .find(|(_, v)| **v >= 0.0)
                    .unwrap()
                    .0;
                debug_assert!(frac > 0);
                debug_assert!(frac < EDGE_SEARCH_SIZE);

                let f = |frac| {
                    ((edge.start.map(|i| i as u32)
                        * (EDGE_SEARCH_SIZE - frac - 1) as u32)
                        + (edge.end.map(|i| i as u32) * frac as u32))
                        / ((EDGE_SEARCH_SIZE - 1) as u32)
                };

                let a = f(frac - 1);
                let b = f(frac);
                debug_assert!(a.max() <= u16::MAX.into());
                debug_assert!(b.max() <= u16::MAX.into());
                edge.start = a.map(|v| v as u16);
                edge.end = b.map(|v| v as u16);
            }
        }

        // --- Phase 4: one big gradient batch at the final intersection ------
        let mut grad_xs: Vec<Grad> = Vec::with_capacity(total_edges);
        let mut grad_ys: Vec<Grad> = Vec::with_capacity(total_edges);
        let mut grad_zs: Vec<Grad> = Vec::with_capacity(total_edges);
        for edge in &edges {
            let cell = self.deferred_leaves[edge.cell_idx].cell;
            let xyz = ((edge.start.map(|v| v as u32)
                + edge.end.map(|v| v as u32))
                / 2)
            .map(|v| v as u16);
            let pos = cell.pos(xyz);
            grad_xs.push(Grad::new(pos.x, 1.0, 0.0, 0.0));
            grad_ys.push(Grad::new(pos.y, 0.0, 1.0, 0.0));
            grad_zs.push(Grad::new(pos.z, 0.0, 0.0, 1.0));
        }
        let grads = self
            .eval_grad_slice
            .eval_v(
                eval.g_tape(&mut self.tape_storage),
                &grad_xs,
                &grad_ys,
                &grad_zs,
                vars,
            )
            .unwrap();

        // --- Phase 5: per-cell QEF solve + Cell::Leaf write-back -----------
        // Mirrors the cursor-advancement semantics of the scalar `leaf()`
        // including the NaN-break behaviour (cursor is only incremented on
        // successful non-NaN add).
        //
        // Also populates `self.hermite_cells[parent_idx][corner_idx]` with
        // per-edge intersections + QEF error, so the post-flush
        // `collapse_subtree` pass can run `try_collapse` bottom-up over this
        // subtree exactly the way scalar MDC would.  Without this, every
        // surface-crossing cell survives as its own leaf and the mesh comes
        // out at uniform max-depth density.
        for (i, mask_opt) in masks.iter().enumerate() {
            let Some(mask) = mask_opt else { continue };
            let dl = &self.deferred_leaves[i];
            let base = per_cell_edge_bases[i];
            let n_edges = per_cell_edge_bases[i + 1] - base;

            // Per-leaf hermite data we're about to fill in.  Starts blank.
            let mut hermite_cell = LeafHermiteData::default();

            let mut verts: arrayvec::ArrayVec<CellVertex<3>, 4> =
                arrayvec::ArrayVec::new();
            let mut cursor = 0usize;
            for vs in CELL_TO_VERT_TO_EDGES[mask.index()].iter() {
                let mut force_point: Option<nalgebra::Vector3<f32>> = None;
                let mut qef = QuadraticErrorSolver::new();
                for e in vs.iter() {
                    let global = base + cursor;
                    let pos = nalgebra::Vector3::new(
                        grad_xs[global].v,
                        grad_ys[global].v,
                        grad_zs[global].v,
                    );
                    let grad: nalgebra::Vector4<f32> = grads[global].into();

                    if grad.iter().any(|f| f.is_nan()) {
                        // Match scalar leaf()'s NaN handling: force-point
                        // the vertex and mark hermite as QEF_ERR_INVALID so
                        // `merge` rejects this subtree for collapsing.
                        force_point = Some(pos);
                        hermite_cell.qef_err = QEF_ERR_INVALID;
                        break;
                    }
                    qef.add_intersection(pos, grad);

                    // Record this intersection in the leaf's hermite data,
                    // using the same edge-index convention scalar leaf()
                    // uses (`e.to_undirected().index()`).
                    let edge_index = e.to_undirected().index();
                    hermite_cell.intersections[edge_index] =
                        LeafIntersection {
                            pos: nalgebra::Vector4::new(
                                pos.x, pos.y, pos.z, 1.0,
                            ),
                            grad,
                        };

                    cursor += 1;
                }

                if let Some(pos) = force_point {
                    verts.push(CellVertex { pos });
                } else {
                    let (pos, err) = qef.solve();
                    verts.push(pos);
                    // Last-error-wins, matching scalar leaf() — only
                    // meaningful when the mask has a single vertex (the
                    // only case where try_collapse actually looks at it).
                    hermite_cell.qef_err = err;
                }
            }

            let vert_index = self.octree.verts.len();
            self.octree.verts.extend(verts);
            // Append the per-edge intersection positions — these serve as the
            // edge vertices that `walk_dual` emits on boundary edges, and
            // must match the order of the intersections produced inside the
            // scalar `leaf()`.
            for ei_in_cell in 0..n_edges {
                let global = base + ei_in_cell;
                let pos = nalgebra::Vector3::new(
                    grad_xs[global].v,
                    grad_ys[global].v,
                    grad_zs[global].v,
                );
                self.octree.verts.push(CellVertex { pos });
            }

            self.octree[dl.cell] = Cell::Leaf(Leaf {
                mask: *mask,
                index: vert_index,
            });

            // Publish hermite to the parent's slot so collapse_subtree can
            // read it.  `hermite_cells` is grown in lock-step with
            // `octree.cells`, so this index is always in bounds.
            self.hermite_cells[dl.parent_idx][dl.corner_idx as usize] =
                hermite_cell;
        }

        self.deferred_leaves.clear();
        true
    }
}

////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Default, Debug)]
struct LeafIntersection {
    /// Intersection position is xyz; w is 1 if the intersection is present
    pos: nalgebra::Vector4<f32>,
    /// Gradient is xyz; w is the distance field value at this point
    grad: nalgebra::Vector4<f32>,
}

impl From<LeafIntersection> for QuadraticErrorSolver {
    fn from(i: LeafIntersection) -> Self {
        let mut qef = QuadraticErrorSolver::default();
        if i.pos.w != 0.0 {
            qef.add_intersection(i.pos.xyz(), i.grad);
        }
        qef
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct LeafHermiteData {
    intersections: [LeafIntersection; 12],
    face_qefs: [QuadraticErrorSolver; 6],
    center_qef: QuadraticErrorSolver,

    /// Error found when solving this QEF (if positive), or a special value
    qef_err: f32,
}

/// This QEF is not populated
const QEF_ERR_EMPTY: f32 = -1.0;

/// This QEF is known to be invalid and should be disregarded
const QEF_ERR_INVALID: f32 = -2.0;

impl Default for LeafHermiteData {
    fn default() -> Self {
        Self {
            intersections: Default::default(),
            face_qefs: Default::default(),
            center_qef: Default::default(),
            qef_err: QEF_ERR_EMPTY,
        }
    }
}

impl LeafHermiteData {
    /// Merges an octree subdivision of leaf hermite data
    ///
    /// Returns `None` if any of the leafs have invalid QEFs (typically due to
    /// NANs in normal computation).
    fn merge(leafs: [LeafHermiteData; 8]) -> Option<Self> {
        let mut out = Self::default();
        use super::types::{X, Y, Z};

        if leafs.iter().any(|v| v.qef_err == QEF_ERR_INVALID) {
            return None;
        }

        // Accumulate intersections along edges
        for t in [X, Y, Z] {
            let u = t.next();
            let v = u.next();
            for edge in 0..4 {
                let mut start = Corner::<3>::new(0);
                if edge & 1 != 0 {
                    start = start | u;
                }
                if edge & 2 != 0 {
                    start = start | v;
                }
                let end = start | t;

                // Canonical edge as a value in the 0-12 range
                let edge = Edge::new((t.index() * 4 + edge) as u8);

                // One or the other leaf has an intersection
                let a = leafs[start.index()].intersections[edge.index()];
                let b = leafs[end.index()].intersections[edge.index()];
                match (a.pos.w > 0.0, b.pos.w > 0.0) {
                    (true, false) => out.intersections[edge.index()] = a,
                    (false, true) => out.intersections[edge.index()] = b,
                    (false, false) => (),
                    (true, true) => panic!("duplicate intersection"),
                }
            }
        }

        // Accumulate face QEFs along edges
        for t in [X, Y, Z] {
            let u = t.next();
            let v = t.next();
            for face in 0..2 {
                let a = if face == 1 {
                    t.into()
                } else {
                    Corner::<3>::new(0)
                };
                let b = a | u;
                let c = a | v;
                let d = a | u | v;
                let f = t.index() * 2 + face;
                for q in [a, b, c, d] {
                    out.face_qefs[f] += leafs[q.index()].face_qefs[f];
                }
                // Edges oriented along the v axis on this face
                let edge_index_v = v.index() * 4 + face * 2 + 1;
                out.face_qefs[f] +=
                    leafs[a.index()].intersections[edge_index_v].into();
                out.face_qefs[f] +=
                    leafs[b.index()].intersections[edge_index_v].into();

                // Edges oriented along the u axis on this face
                let edge_index_u = v.index() * 4 + face * 2 + 1;
                out.face_qefs[f] +=
                    leafs[a.index()].intersections[edge_index_u].into();
                out.face_qefs[f] +=
                    leafs[c.index()].intersections[edge_index_u].into();
            }
        }

        // Accumulate center QEFs
        for t in [X, Y, Z] {
            let u = t.next();
            let v = t.next();

            // Accumulate the four inner face QEFs
            let a = Corner::<3>::new(0);
            let b = a | u;
            let c = a | v;
            let d = a | u | v;
            for q in [a, b, c, d] {
                out.center_qef += leafs[q.index()].face_qefs[t.index() * 2 + 1];
            }

            // Edges oriented along the u axis
            out.center_qef +=
                leafs[a.index()].intersections[u.index() * 4 + 3].into();
            out.center_qef +=
                leafs[b.index()].intersections[u.index() * 4 + 3].into();

            // We skip edges oriented on the v axis, because they'll be counted
            // by one of the other iterations through the loop
        }
        for leaf in leafs {
            out.center_qef += leaf.center_qef;
        }

        // Accumulate minimum QEF error among valid child QEFs
        out.qef_err = f32::INFINITY;
        for e in leafs.iter().map(|q| q.qef_err).filter(|&e| e >= 0.0) {
            out.qef_err = out.qef_err.min(e);
        }

        Some(out)
    }

    /// Solves the combined QEF.
    pub fn solve(&self) -> (CellVertex<3>, f32) {
        let mut qef = self.center_qef;
        for &i in &self.intersections {
            qef += i.into();
        }
        for &f in &self.face_qefs {
            qef += f;
        }
        qef.solve()
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::{Edge, X, Y, Z};
    use fidget_core::{
        context::{Context, Tree},
        render::ThreadPool,
        shape::EzShape,
        var::Var,
        vm::VmShape,
    };
    use std::collections::BTreeMap;

    fn depth0_single_thread() -> Settings<'static> {
        Settings {
            depth: 0,
            threads: None,
            ..Default::default()
        }
    }

    fn depth1_single_thread() -> Settings<'static> {
        Settings {
            depth: 1,
            threads: None,
            ..Default::default()
        }
    }

    fn sphere(center: [f32; 3], radius: f32) -> Tree {
        let (x, y, z) = Tree::axes();
        ((x - center[0]).square()
            + (y - center[1]).square()
            + (z - center[2]).square())
        .sqrt()
            - radius
    }

    fn cube(bx: [f32; 2], by: [f32; 2], bz: [f32; 2]) -> Tree {
        let (x, y, z) = Tree::axes();
        let x_bounds = (bx[0] - x.clone()).max(x - bx[1]);
        let y_bounds = (by[0] - y.clone()).max(y - by[1]);
        let z_bounds = (bz[0] - z.clone()).max(z - bz[1]);
        x_bounds.max(y_bounds).max(z_bounds)
    }

    #[test]
    fn test_cube_edge() {
        const EPSILON: f32 = 1e-3;
        let f = 2.0;
        let shape = VmShape::from(cube([-f, f], [-f, 0.3], [-f, 0.6]));
        // This should be a cube with a single edge running through the root
        // node of the octree, with an edge vertex at [0, 0.3, 0.6]
        let octree = Octree::build(&shape, &depth0_single_thread()).unwrap();
        assert_eq!(octree.verts.len(), 5);
        let v = octree.verts[0].pos;
        let expected = nalgebra::Vector3::new(0.0, 0.3, 0.6);
        assert!(
            (v - expected).norm() < EPSILON,
            "bad edge vertex {v:?}; expected {expected:?}"
        );
    }

    fn cone(
        corner: nalgebra::Vector3<f32>,
        tip: nalgebra::Vector3<f32>,
        radius: f32,
    ) -> Tree {
        let dir = tip - corner;
        let length = dir.norm();
        let dir = dir.normalize();

        let corner = corner.map(|v| Tree::constant(v as f64));
        let dir = dir.map(|v| Tree::constant(v as f64));

        let (x, y, z) = Tree::axes();
        let point = nalgebra::Vector3::new(x, y, z);
        let offset = point.clone() - corner.clone();

        // a is the distance along the corner-tip direction
        let a = offset.x.clone() * dir.x.clone()
            + offset.y.clone() * dir.y.clone()
            + offset.z.clone() * dir.z.clone();

        // Position of the nearest point on the corner-tip axis
        let a_pos = corner + dir * a.clone();

        // b is the orthogonal distance
        let offset = point - a_pos;
        let b = (offset.x.clone().square()
            + offset.y.clone().square()
            + offset.z.clone().square())
        .sqrt();

        b - radius * (1.0 - a / length)
    }

    #[test]
    fn test_mesh_basic() {
        let shape = VmShape::from(sphere([0.0; 3], 0.2));

        // If we only build a depth-0 octree, then it's a leaf without any
        // vertices (since all the corners are empty)
        let octree = Octree::build(&shape, &depth0_single_thread()).unwrap();
        assert!(octree.cells.is_empty()); // root only
        assert_eq!(Cell::Empty, octree.root);
        assert!(octree.verts.is_empty());

        let empty_mesh = octree.walk_dual();
        assert!(empty_mesh.vertices.is_empty());
        assert!(empty_mesh.triangles.is_empty());

        // Now, at depth-1, each cell should be a Leaf with one vertex
        let octree = Octree::build(&shape, &depth1_single_thread()).unwrap();
        assert_eq!(octree.cells.len(), 1); // one fanout of 8 cells
        assert_eq!(Cell::Branch { index: 0 }, octree.root);

        // Each of the 6 edges is counted 4 times and each cell has 1 vertex
        assert_eq!(octree.verts.len(), 6 * 4 + 8, "incorrect vertex count");

        // Each cell is a leaf with 4 vertices (3 edges, 1 center)
        for o in octree.cells[0].iter() {
            let Cell::Leaf(Leaf { index, mask }) = *o else {
                panic!()
            };
            assert_eq!(mask.count_ones(), 1);
            assert_eq!(index % 4, 0);
        }

        let sphere_mesh = octree.walk_dual();
        assert!(sphere_mesh.vertices.len() > 1);
        assert!(!sphere_mesh.triangles.is_empty());
    }

    #[test]
    fn test_sphere_verts() {
        let shape = VmShape::from(sphere([0.0; 3], 0.2));

        let octree = Octree::build(&shape, &depth1_single_thread()).unwrap();
        let sphere_mesh = octree.walk_dual();

        let mut edge_count = 0;
        for v in &sphere_mesh.vertices {
            // Edge vertices should be found via binary search and therefore
            // should be close to the true crossing point
            let x_edge = v.x != 0.0;
            let y_edge = v.y != 0.0;
            let z_edge = v.z != 0.0;
            let edge_sum = x_edge as u8 + y_edge as u8 + z_edge as u8;
            assert!(edge_sum == 1 || edge_sum == 3);
            if edge_sum == 1 {
                assert!(
                    (v.norm() - 0.2).abs() < 2.0 / u16::MAX as f32,
                    "edge vertex {v:?} is not at radius 0.2"
                );
                edge_count += 1;
            } else {
                // The sphere looks like a box at this sampling depth, since the
                // edge intersections are all axis-aligned; we'll check that
                // corner vertices are all at [±0.2, ±0.2, ±0.2]
                assert!(
                    (v.abs() - nalgebra::Vector3::new(0.2, 0.2, 0.2)).norm()
                        < 2.0 / 65535.0,
                    "cell vertex {v:?} is not at expected position"
                );
            }
        }
        assert_eq!(edge_count, 6);
    }

    #[test]
    fn test_sphere_manifold() {
        let shape = VmShape::from(sphere([0.0; 3], 0.85));

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 5,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&shape, &settings).unwrap();
            let sphere_mesh = octree.walk_dual();

            check_for_vertex_dupes(&sphere_mesh).unwrap();
            check_for_edge_matching(&sphere_mesh).unwrap();
        }
    }

    #[test]
    fn test_cube_verts() {
        let shape = VmShape::from(cube([-0.1, 0.6], [-0.2, 0.75], [-0.3, 0.4]));

        let octree = Octree::build(&shape, &depth1_single_thread()).unwrap();
        let mesh = octree.walk_dual();
        const EPSILON: f32 = 2.0 / u16::MAX as f32;
        assert!(!mesh.vertices.is_empty());
        for v in &mesh.vertices {
            // Edge vertices should be found via binary search and therefore
            // should be close to the true crossing point
            let x_edge = v.x != 0.0;
            let y_edge = v.y != 0.0;
            let z_edge = v.z != 0.0;
            let edge_sum = x_edge as u8 + y_edge as u8 + z_edge as u8;
            assert!(edge_sum == 1 || edge_sum == 3);

            if edge_sum == 1 {
                assert!(
                    (x_edge
                        && ((v.x - -0.1).abs() < EPSILON
                            || (v.x - 0.6).abs() < EPSILON))
                        || (y_edge
                            && ((v.y - -0.2).abs() < EPSILON
                                || (v.y - 0.75).abs() < EPSILON))
                        || (z_edge
                            && ((v.z - -0.3).abs() < EPSILON
                                || (v.z - 0.4).abs() < EPSILON)),
                    "bad edge position {v:?}"
                );
            } else {
                assert!(
                    ((v.x - -0.1).abs() < EPSILON
                        || (v.x - 0.6).abs() < EPSILON)
                        && ((v.y - -0.2).abs() < EPSILON
                            || (v.y - 0.75).abs() < EPSILON)
                        && ((v.z - -0.3).abs() < EPSILON
                            || (v.z - 0.4).abs() < EPSILON),
                    "bad vertex position {v:?}"
                );
            }
        }
    }

    #[test]
    fn test_plane_center() {
        const EPSILON: f32 = 1e-3;
        for dx in [0.0, 0.25, -0.25, 2.0, -2.0] {
            for dy in [0.0, 0.25, -0.25, 2.0, -2.0] {
                for offset in [0.0, -0.2, 0.2] {
                    let (x, y, z) = Tree::axes();
                    let f = x * dx + y * dy + z + offset;
                    let shape = VmShape::from(f);
                    let octree =
                        Octree::build(&shape, &depth0_single_thread()).unwrap();

                    assert!(octree.cells.is_empty()); // root only
                    let pos = octree.verts[0].pos;
                    let mut mass_point = nalgebra::Vector3::zeros();
                    for v in &octree.verts[1..] {
                        mass_point += v.pos;
                    }
                    mass_point /= (octree.verts.len() - 1) as f32;
                    assert!(
                        (pos - mass_point).norm() < EPSILON,
                        "bad vertex position at dx: {dx}, dy: {dy}, \
                         offset: {offset} => {pos:?} != {mass_point:?}"
                    );
                    let mut eval = VmShape::new_point_eval();
                    let tape = shape.ez_point_tape();
                    for v in &octree.verts {
                        let v = v.pos;
                        let (r, _) = eval.eval(&tape, v.x, v.y, v.z).unwrap();
                        assert!(r.abs() < EPSILON, "bad value at {v:?}: {r}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_cone_vert() {
        // Test both in-cell and out-of-cell cone vertices
        for tip in [
            nalgebra::Vector3::new(0.2, 0.3, 0.4),
            nalgebra::Vector3::new(1.2, 1.3, 1.4),
        ] {
            let corner = nalgebra::Vector3::new(-1.0, -1.0, -1.0);
            let shape = VmShape::from(cone(corner, tip, 0.1));

            let mut eval = VmShape::new_point_eval();
            let tape = shape.ez_point_tape();
            let (v, _) = eval.eval(&tape, tip.x, tip.y, tip.z).unwrap();
            assert!(v.abs() < 1e-6, "bad tip value: {v}");
            let (v, _) =
                eval.eval(&tape, corner.x, corner.y, corner.z).unwrap();
            assert!(v < 0.0, "bad corner value: {v}");

            let octree =
                Octree::build(&shape, &depth0_single_thread()).unwrap();
            assert!(octree.cells.is_empty()); // root only
            assert_eq!(octree.verts.len(), 4);

            let pos = octree.verts[0].pos;
            assert!(
                (pos - tip).norm() < 1e-3,
                "bad vertex position: expected {tip:?}, got {pos:?}"
            );
        }
    }

    fn test_mesh_manifold_inner(threads: Option<&ThreadPool>, mask: u8) {
        let mut shape = vec![];
        for j in Corner::<3>::iter() {
            if mask & (1 << j.index()) != 0 {
                shape.push(sphere(
                    [
                        if j & X { 0.5 } else { 0.0 },
                        if j & Y { 0.5 } else { 0.0 },
                        if j & Z { 0.5 } else { 0.0 },
                    ],
                    0.1,
                ));
            }
        }
        let Some(start) = shape.pop() else { return };
        let shape = shape.into_iter().fold(start, |acc, s| acc.min(s));

        // Now, we have our shape, which is 0-8 spheres placed at the
        // corners of the cell spanning [0, 0.25]
        let shape = VmShape::from(shape);
        let settings = Settings {
            depth: 2,
            threads,
            ..Default::default()
        };
        let octree = Octree::build(&shape, &settings).unwrap();

        let mesh = octree.walk_dual();
        if mask != 0 && mask != 255 {
            assert!(!mesh.vertices.is_empty());
            assert!(!mesh.triangles.is_empty());
        }

        if let Err(e) = check_for_vertex_dupes(&mesh) {
            panic!("mask {mask:08b} has {e}");
        }
        if let Err(e) = check_for_edge_matching(&mesh) {
            panic!("mask {mask:08b} has {e}");
        }
    }

    #[test]
    fn test_mesh_manifold_single_thread() {
        for mask in 0..=255 {
            test_mesh_manifold_inner(None, mask)
        }
    }

    #[test]
    fn test_mesh_manifold_multi_thread() {
        for mask in 0..=255 {
            test_mesh_manifold_inner(Some(&ThreadPool::Global), mask)
        }
    }

    #[test]
    fn test_collapsible() {
        let shape = VmShape::from(sphere([0.0; 3], 0.1));
        let octree = Octree::build(&shape, &depth1_single_thread()).unwrap();
        assert!(octree.collapsible(0).is_none());

        // If we have a single corner sphere, then the builder will collapse the
        // branch for us, leaving just a leaf.
        let shape = VmShape::from(sphere([-1.0; 3], 0.1));
        let octree = Octree::build(&shape, &depth1_single_thread()).unwrap();
        assert!(matches!(octree.root, Cell::Leaf { .. }));

        // Test with a single thread and a deeper tree, which should still work
        let octree = Octree::build(
            &shape,
            &Settings {
                depth: 4,
                threads: None,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            matches!(octree.root, Cell::Leaf { .. }),
            "root should be a leaf, not {:?}",
            octree.root
        );

        // This should also work with multiple threads!
        let octree = Octree::build(
            &shape,
            &Settings {
                depth: 4,
                threads: Some(&ThreadPool::Global),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            matches!(octree.root, Cell::Leaf { .. }),
            "root should be a leaf, not {:?}",
            octree.root
        );

        let shape = VmShape::from(sphere([-1.0, 0.0, 1.0], 0.1));
        let octree = Octree::build(&shape, &depth1_single_thread()).unwrap();
        assert!(octree.collapsible(0).is_none());

        let a = sphere([-1.0; 3], 0.1);
        let b = sphere([1.0; 3], 0.1);
        let shape = VmShape::from(a.min(b));
        let octree = Octree::build(&shape, &depth1_single_thread()).unwrap();
        assert!(octree.collapsible(0).is_none());
    }

    #[test]
    fn test_empty_collapse() {
        // Make a very smol sphere that won't be sampled
        let shape = VmShape::from(sphere([0.1; 3], 0.05));

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 1,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&shape, &settings).unwrap();
            assert_eq!(
                octree.root,
                Cell::Empty,
                "failed to collapse octree with threads: {:?}",
                threads.map(|t| t.thread_count())
            );
        }
    }

    #[test]
    fn test_colonnade_manifold() {
        const COLONNADE: &str = include_str!("../../models/colonnade.vm");
        let (ctx, root) = Context::from_text(COLONNADE.as_bytes()).unwrap();
        let tape = VmShape::new(&ctx, root).unwrap();

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 5,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&tape, &settings).unwrap();
            let mesh = octree.walk_dual();
            // Note: the model has duplicate vertices!
            if let Err(e) = check_for_edge_matching(&mesh) {
                panic!(
                    "colonnade model has {e} with threads: {:?}",
                    threads.map(|t| t.thread_count())
                );
            }
        }
    }

    #[test]
    fn colonnade_bounds() {
        const COLONNADE: &str = include_str!("../../models/colonnade.vm");
        let (ctx, root) = Context::from_text(COLONNADE.as_bytes()).unwrap();
        let tape = VmShape::new(&ctx, root).unwrap();

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 8,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&tape, &settings).unwrap();
            let mesh = octree.walk_dual();
            for v in mesh.vertices.iter() {
                assert!(
                    v.x < 1.0
                        && v.x > -1.0
                        && v.y < 1.0
                        && v.y > -1.0
                        && v.z < 1.0
                        && v.z > -0.5,
                    "invalid vertex {v:?} with threads {:?}",
                    threads.map(|t| t.thread_count())
                );
            }
        }
    }

    #[test]
    fn bear_bounds() {
        const COLONNADE: &str = include_str!("../../models/bear.vm");
        let (ctx, root) = Context::from_text(COLONNADE.as_bytes()).unwrap();
        let tape = VmShape::new(&ctx, root).unwrap();

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 5,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&tape, &settings).unwrap();
            let mesh = octree.walk_dual();
            for v in mesh.vertices.iter() {
                assert!(
                    v.x < 1.0
                        && v.x > -0.75
                        && v.y < 1.0
                        && v.y > -0.75
                        && v.z < 0.75
                        && v.z > -0.75,
                    "invalid vertex {v:?} with threads {:?}",
                    threads.map(|t| t.thread_count())
                );
            }
        }
    }

    fn check_for_vertex_dupes(mesh: &Mesh) -> Result<(), String> {
        let mut verts = mesh.vertices.clone();
        verts.sort_by_key(|k| (k.x.to_bits(), k.y.to_bits(), k.z.to_bits()));
        for i in 1..verts.len() {
            if verts[i - 1] == verts[i] {
                return Err(format!("duplicate vertices at {}", verts[i]));
            }
        }
        Ok(())
    }

    fn check_for_edge_matching(mesh: &Mesh) -> Result<(), String> {
        let mut edges: BTreeMap<_, usize> = BTreeMap::new();
        for t in &mesh.triangles {
            for edge in [(t.x, t.y), (t.y, t.z), (t.z, t.x)] {
                if t.x == t.y || t.y == t.z || t.x == t.z {
                    return Err("triangle with duplicate edges".to_string());
                }
                *edges.entry(edge).or_default() += 1;
            }
        }
        for (&(a, b), &i) in &edges {
            if i != 1 {
                return Err(format!(
                    "duplicate edge ({a}, {b}) between {:?} {:?}",
                    mesh.vertices[a], mesh.vertices[b]
                ));
            }
            if !edges.contains_key(&(b, a)) {
                return Err("unpaired edges".to_owned());
            }
        }
        Ok(())
    }

    #[test]
    fn test_qef_merging() {
        let mut hermite = LeafHermiteData::default();

        // Add a dummy intersection with a non-zero normal; we'll be counting
        // mass points to check that the merging went smoothly
        let grad = nalgebra::Vector4::new(1.0, 0.0, 0.0, 0.0);
        let pos = nalgebra::Vector4::new(0.0, 0.0, 0.0, 1.0);
        hermite.intersections.fill(LeafIntersection { pos, grad });

        // Ensure that only one of the two sub-edges per edge contains an
        // intersection, because that's checked for in `LeafHermiteData::merge`
        let mut hermites = [hermite; 8];
        for i in 0..12 {
            let e = Edge::new(i);
            let (_start, end) = e.corners();
            hermites[end.index()].intersections[i as usize] =
                LeafIntersection::default();
        }

        let merged = LeafHermiteData::merge(hermites).unwrap();
        for i in merged.intersections {
            assert_eq!(i.grad, grad);
            assert_eq!(i.pos, pos);
        }
        // Each face in the merged cell should include the accumulation of four
        // edges from lower cells (but nothing more, because lower cells didn't
        // have face QEFs populated)
        for (i, f) in merged.face_qefs.iter().enumerate() {
            assert_eq!(
                f.mass_point().w,
                4.0,
                "bad accumulated QEF on face {i}"
            );
        }
        assert_eq!(
            merged.center_qef.mass_point().w,
            6.0,
            "bad accumulated QEF in center"
        );
    }

    #[test]
    fn test_qef_near_planar() {
        let shape = VmShape::from(sphere([0.0; 3], 0.75));

        let settings = Settings {
            depth: 4,
            ..Default::default()
        };

        let octree = Octree::build(&shape, &settings).unwrap().walk_dual();
        for v in octree.vertices.iter() {
            let n = v.norm();
            assert!(n > 0.7 && n < 0.8, "invalid vertex at {v:?}: {n}");
        }
    }

    #[test]
    fn test_mesh_vars() {
        let (x, y, z) = Tree::axes();
        let v = Var::new();
        let c = Tree::from(v);
        let sphere = (x.square() + y.square() + z.square()).sqrt() - c;
        let shape = VmShape::from(sphere);

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 4,
                threads,
                ..Default::default()
            };

            for r in [0.5, 0.75] {
                let mut vars = ShapeVars::new();
                vars.insert(v.index().unwrap(), r);
                let octree = Octree::build_with_vars(&shape, &vars, &settings)
                    .unwrap()
                    .walk_dual();
                for v in octree.vertices.iter() {
                    let n = v.norm();
                    assert!(
                        n > r - 0.05 && n < r + 0.05,
                        "invalid vertex at {v:?}: {n} != {r} with threads {:?}",
                        threads.map(|t| t.thread_count())
                    );
                }
            }
        }
    }

    #[test]
    fn test_octree_cancel() {
        let (x, y, z) = Tree::axes();
        let sphere = (x.square() + y.square() + z.square()).sqrt() - 1.0;
        let shape = VmShape::from(sphere);

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 4,
                threads,
                ..Default::default()
            };
            let c = settings.cancel.clone();
            c.cancel();

            assert!(Octree::build(&shape, &settings).is_none());
        }
    }
}
