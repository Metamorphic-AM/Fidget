//! Dual contouring implementation

use crate::{
    Octree, Triangulation,
    builder::MeshBuilder,
    cell::{Cell, CellIndex, Leaf, NO_SPLIT},
    frame::{Frame, XYZ, YZX, ZXY},
    types::{Corner, Edge, X, Y, Z},
};
use std::sync::atomic::Ordering::Relaxed;

pub fn dc_cell(octree: &Octree, cell: CellIndex<3>, out: &mut MeshBuilder) {
    if let Cell::Branch { .. } = octree[cell] {
        for i in Corner::iter() {
            out.cell(octree, octree.child(cell, i));
        }

        // Helper function for DC face calls
        fn dc_faces<T: Frame>(
            octree: &Octree,
            cell: CellIndex<3>,
            out: &mut MeshBuilder,
        ) {
            let (t, u, v) = T::frame();
            for c in [Corner::new(0), u.into(), v.into(), u | v] {
                out.face::<T>(
                    octree,
                    octree.child(cell, c),
                    octree.child(cell, c | t),
                );
            }
        }
        dc_faces::<XYZ>(octree, cell, out);
        dc_faces::<YZX>(octree, cell, out);
        dc_faces::<ZXY>(octree, cell, out);

        #[allow(unused_parens)]
        for i in [false, true] {
            out.edge::<XYZ>(
                octree,
                octree.child(cell, (X * i)),
                octree.child(cell, (X * i) | Y),
                octree.child(cell, (X * i) | Y | Z),
                octree.child(cell, (X * i) | Z),
            );
            out.edge::<YZX>(
                octree,
                octree.child(cell, (Y * i)),
                octree.child(cell, (Y * i) | Z),
                octree.child(cell, (Y * i) | X | Z),
                octree.child(cell, (Y * i) | X),
            );
            out.edge::<ZXY>(
                octree,
                octree.child(cell, (Z * i)),
                octree.child(cell, (Z * i) | X),
                octree.child(cell, (Z * i) | X | Y),
                octree.child(cell, (Z * i) | Y),
            );
        }
    }
}

/// Handles two cells which share a common face
///
/// `lo` is below `hi` on the `T` axis; the cells share a `UV` face where
/// `T-U-V` is a right-handed coordinate system.
pub fn dc_face<T: Frame>(
    octree: &Octree,
    lo: CellIndex<3>,
    hi: CellIndex<3>,
    out: &mut MeshBuilder,
) {
    if octree.is_leaf(lo) && octree.is_leaf(hi) {
        stitch_face::<T>(octree, lo, hi, out);
        return;
    }
    let (t, u, v) = T::frame();
    out.face::<T>(
        octree,
        octree.child(lo, t),
        octree.child(hi, Corner::new(0)),
    );
    out.face::<T>(octree, octree.child(lo, t | u), octree.child(hi, u));
    out.face::<T>(octree, octree.child(lo, t | v), octree.child(hi, v));
    out.face::<T>(octree, octree.child(lo, t | u | v), octree.child(hi, u | v));
    #[allow(unused_parens)]
    for i in [false, true] {
        out.edge::<T::Next>(
            octree,
            octree.child(lo, (u * i) | t),
            octree.child(lo, (u * i) | v | t),
            octree.child(hi, (u * i) | v),
            octree.child(hi, (u * i)),
        );
        out.edge::<<T::Next as Frame>::Next>(
            octree,
            octree.child(lo, (v * i) | t),
            octree.child(hi, (v * i)),
            octree.child(hi, (v * i) | u),
            octree.child(lo, (v * i) | u | t),
        );
    }
}

/// Closes the seam left on a face by a leaf with a vertex per sheet
///
/// The crossings on a face come in pairs, one pair for each run of inside
/// corners along the face's boundary, and the dual polygons of a pair share
/// the mesh edge between the two cells' vertices.  When either cell gives the
/// two crossings of a pair different vertices (see [`Leaf::split`]), those
/// polygons no longer share an edge and the surface would be open between
/// them.  The polygon through the two cells' vertices for the pair closes it:
/// a triangle when one cell split, a quad when both did.  It is the chamfer
/// that stands in for a crease lying beyond the reach of the cell's vertex.
///
/// `lo` is below `hi` on the `T` axis, as in [`dc_face`].
fn stitch_face<T: Frame>(
    octree: &Octree,
    lo: CellIndex<3>,
    hi: CellIndex<3>,
    out: &mut MeshBuilder,
) {
    let (Cell::Leaf(a), Cell::Leaf(b)) = (octree[lo], octree[hi]) else {
        return;
    };
    if a.split == NO_SPLIT && b.split == NO_SPLIT {
        return;
    }
    let (t, u, v) = T::frame();
    let (ui, vi) = (u.index() as u8, v.index() as u8);

    // The face's corners c00, c10, c01, c11 (in u, v) and which are inside,
    // read from the deeper leaf: a coarser neighbour has a single vertex and
    // its mask says nothing about this sub-face.
    let inside: [bool; 4] = if lo.depth >= hi.depth {
        [
            a.mask & Corner::from(t),
            a.mask & (t | u),
            a.mask & (t | v),
            a.mask & (t | u | v),
        ]
    } else {
        [
            b.mask & Corner::new(0),
            b.mask & Corner::from(u),
            b.mask & Corner::from(v),
            b.mask & (u | v),
        ]
    };

    // The face's four edges: (edge in `lo`, edge in `hi`, start corner, end
    // corner, flip).  `flip` is whether the dual polygon of the edge, for an
    // inside start corner, runs from `hi`'s vertex to `lo`'s rather than the
    // other way; it follows from the cell order in `dc_edge`.
    #[allow(clippy::identity_op)]
    let edges: [(u8, u8, usize, usize, bool); 4] = [
        (ui * 4 + 2, ui * 4 + 0, 0, 1, false), // along u at v = 0
        (ui * 4 + 3, ui * 4 + 1, 2, 3, true),  // along u at v = 1
        (vi * 4 + 1, vi * 4 + 0, 0, 2, true),  // along v at u = 0
        (vi * 4 + 3, vi * 4 + 2, 1, 3, false), // along v at u = 1
    ];

    // Runs of inside corners along the face's boundary
    let mut label = [0usize, 1, 2, 3];
    loop {
        let mut changed = false;
        for (p, q) in [(0, 1), (2, 3), (0, 2), (1, 3)] {
            if inside[p] && inside[q] && label[p] != label[q] {
                let m = label[p].min(label[q]);
                label[p] = m;
                label[q] = m;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut crossings: arrayvec::ArrayVec<(usize, usize), 4> =
        arrayvec::ArrayVec::new();
    for (k, e) in edges.iter().enumerate() {
        if inside[e.2] != inside[e.3] {
            let inner = if inside[e.2] { e.2 } else { e.3 };
            crossings.push((k, label[inner]));
        }
    }

    // The vertex of `leaf` for one of this face's edges
    let vertex_of = |cell: CellIndex<3>, leaf: &Leaf<3>, edge: u8| {
        if cell.depth == lo.depth.max(hi.depth) {
            octree
                .leaf_edge(leaf, Edge::new(edge))
                .map(|i| leaf.index + i.vert.0 as usize)
        } else {
            // A coarser neighbour has a single vertex (see `dc_edge`)
            (0..12)
                .filter_map(|j| octree.leaf_edge(leaf, Edge::new(j)))
                .next()
                .map(|i| leaf.index + i.vert.0 as usize)
        }
    };

    let tri_normal = |a: nalgebra::Vector3<f32>,
                      b: nalgebra::Vector3<f32>,
                      c: nalgebra::Vector3<f32>|
     -> nalgebra::Vector3<f32> {
        let n = (b - a).cross(&(c - a));
        let m = n.norm();
        if m > 1e-30 { n / m } else { n }
    };

    for comp in 0..4 {
        let pair: arrayvec::ArrayVec<usize, 4> = crossings
            .iter()
            .filter(|(_, c)| *c == comp)
            .map(|(k, _)| *k)
            .collect();
        if pair.len() != 2 {
            debug_assert!(pair.is_empty());
            continue;
        }
        let (k1, k2) = (pair[0], pair[1]);
        let (Some(a1), Some(a2), Some(b1), Some(b2)) = (
            vertex_of(lo, &a, edges[k1].0),
            vertex_of(lo, &a, edges[k2].0),
            vertex_of(hi, &b, edges[k1].1),
            vertex_of(hi, &b, edges[k2].1),
        ) else {
            continue;
        };
        if a1 == a2 && b1 == b2 {
            continue;
        }
        // The crease apex of each split cell's component, if it has one:
        // the stitch then runs sheet - apex - sheet on that side, and the
        // two apexes' edge is the crease itself.
        let apex_a = (a1 != a2)
            .then(|| octree.leaf_apex(&a, Edge::new(edges[k1].0)))
            .flatten();
        let apex_b = (b1 != b2)
            .then(|| octree.leaf_apex(&b, Edge::new(edges[k1].1)))
            .flatten();
        // The dual polygon of the first crossing runs from `lo`'s vertex to
        // `hi`'s when the start corner is inside and the edge is not flipped,
        // or both are reversed; the second crossing's polygon runs the other
        // way.  The stitch runs against both.
        let start_outside = !inside[edges[k1].2];
        let a_to_b = start_outside == edges[k1].4;
        debug_assert_eq!(
            a_to_b,
            !(!inside[edges[k2].2] == edges[k2].4),
            "face pair polygons must traverse the shared edge in opposite directions"
        );
        // Polygon order: b1, a1, [apex_a], a2, b2, [apex_b] when the first
        // crossing's polygon runs from `lo` to `hi`, the mirror otherwise
        let mut poly: arrayvec::ArrayVec<usize, 6> = arrayvec::ArrayVec::new();
        if a_to_b {
            poly.push(b1);
            poly.push(a1);
            poly.extend(apex_a);
            poly.push(a2);
            poly.push(b2);
            poly.extend(apex_b);
        } else {
            poly.push(a1);
            poly.push(b1);
            poly.extend(apex_b);
            poly.push(b2);
            poly.push(a2);
            poly.extend(apex_a);
        }
        let mut pts: arrayvec::ArrayVec<usize, 6> = arrayvec::ArrayVec::new();
        for p in poly {
            if pts.last() != Some(&p) {
                pts.push(p);
            }
        }
        if pts.len() > 1 && pts.first() == pts.last() {
            pts.pop();
        }
        if pts.len() < 3 {
            continue;
        }
        let ps: arrayvec::ArrayVec<_, 6> =
            pts.iter().map(|&i| octree.verts[i].pos).collect();
        let mut emit = |i: usize, j: usize, k: usize| {
            let vi = out.vertex(pts[i], &octree.verts);
            let vj = out.vertex(pts[j], &octree.verts);
            let vk = out.vertex(pts[k], &octree.verts);
            out.triangle(vi, vj, vk);
        };
        if let Some(f) = apex_a
            .or(apex_b)
            .and_then(|apex| pts.iter().position(|&p| p == apex))
        {
            // Fan from an apex: its edges to the sheet vertices on each
            // side lie on the sheets, and its edge to the other cell's
            // vertex or apex is the crease
            for j in 1..(pts.len() - 1) {
                emit(f, (f + j) % pts.len(), (f + j + 1) % pts.len());
            }
        } else if pts.len() == 3 {
            emit(0, 1, 2);
        } else {
            let score_02 = tri_normal(ps[0], ps[1], ps[2])
                .dot(&tri_normal(ps[0], ps[2], ps[3]));
            let score_13 = tri_normal(ps[0], ps[1], ps[3])
                .dot(&tri_normal(ps[1], ps[2], ps[3]));
            if score_02 >= score_13 {
                emit(0, 1, 2);
                emit(0, 2, 3);
            } else {
                emit(0, 1, 3);
                emit(1, 2, 3);
            }
        }
        crate::stats::STITCHES.fetch_add(1, Relaxed);
    }
}

/// Handles four cells that share a common edge aligned on axis `T`
///
/// Cells positions are in the order `[0, U, U | V, U]`, i.e. a right-handed
/// winding about `+T` (where `T, U, V` is a right-handed coordinate frame)
///
/// - `dc_edge<X>` is `[0, Y, Y | Z, Z]`
/// - `dc_edge<Y>` is `[0, Z, Z | X, X]`
/// - `dc_edge<Z>` is `[0, X, X | Y, Y]`
pub fn dc_edge<T: Frame>(
    octree: &Octree,
    a: CellIndex<3>,
    b: CellIndex<3>,
    c: CellIndex<3>,
    d: CellIndex<3>,
    out: &mut MeshBuilder,
) {
    let cs = [a, b, c, d];
    if cs.iter().all(|v| octree.is_leaf(*v)) {
        // If any of the leafs are Empty or Full, then this edge can't
        // include a sign change.  TODO: can we make this any -> all if we
        // collapse empty / filled leafs into Empty / Full cells?
        let leafs = cs.map(|cell| match octree[cell] {
            Cell::Leaf(leaf) => Some(leaf),
            Cell::Empty | Cell::Full => None,
            Cell::Branch { .. } => unreachable!(),
            Cell::Invalid => panic!(),
        });
        if leafs.iter().any(Option::is_none) {
            return;
        }
        let leafs = leafs.map(Option::unwrap);

        // TODO: should we pick a canonically deepest leaf instead of the first
        // among the four that's at the deepest depth?
        let deepest = (0..4).max_by_key(|i| cs[*i].depth).unwrap();

        let (t, _u, _v) = T::frame();

        // Each leaf has an edge associated with it
        #[allow(clippy::identity_op)]
        let edges = [
            Edge::new((t.index() * 4 + 3) as u8),
            Edge::new((t.index() * 4 + 2) as u8),
            Edge::new((t.index() * 4 + 0) as u8),
            Edge::new((t.index() * 4 + 1) as u8),
        ];

        // Find the starting sign of the relevant edge, bailing out early if
        // there isn't a sign change here.  All of the deepest edges should show
        // the same sign change, so it doesn't matter which one we pick here.
        let starting_sign = {
            let (start, end) = edges[deepest].corners();
            let start = !(leafs[deepest].mask & start);
            let end = !(leafs[deepest].mask & end);
            // If there is no sign change, then there's nothing to do here.
            if start == end {
                return;
            }
            start
        };

        // Iterate over each of the edges, assigning a vertex if the sign change
        // lines up.
        let mut verts = [None; 4];
        for i in 0..4 {
            if cs[i].depth == cs[deepest].depth {
                let (start, end) = edges[i].corners();
                let s = !(leafs[i].mask & start);
                let e = !(leafs[i].mask & end);
                debug_assert_eq!(s, starting_sign);
                debug_assert_eq!(e, !starting_sign);
                verts[i] = octree.leaf_edge(&leafs[i], edges[i]);
            } else {
                // A coarser neighbour normally holds a single vertex (the
                // collapse only merges cells whose children do); when it
                // does not, the vertex nearest the crossing on the deepest
                // leaf's edge stands in.
                let crossing = octree
                    .leaf_edge(&leafs[deepest], edges[deepest])
                    .map(|x| octree.verts[leafs[deepest].index + x.edge.0 as usize].pos);
                let mut best: Option<crate::types::Intersection> = None;
                let mut several = false;
                for j in 0..12 {
                    let Some(x) = octree.leaf_edge(&leafs[i], Edge::new(j)) else {
                        continue;
                    };
                    match best {
                        None => best = Some(x),
                        Some(b) if b.vert != x.vert => {
                            several = true;
                            if let Some(c) = crossing {
                                let d = |v: crate::types::Offset| {
                                    (octree.verts[leafs[i].index + v.0 as usize].pos - c).norm()
                                };
                                if d(x.vert) < d(b.vert) {
                                    best = Some(x);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if several {
                    crate::stats::COARSE_PICKS.fetch_add(1, Relaxed);
                }
                verts[i] = best;
            }
        }

        let verts = verts.map(Option::unwrap);

        // -------------------------------------------------------------------
        // Dual polygon emission
        //
        // The four cells around the edge each contribute one vertex; the
        // polygon through them is the dual of this edge.  Three ways of
        // turning it into triangles are available (see `Triangulation`):
        //
        // * `Fan`: upstream Fidget's construction, a fan of triangles about
        //   the point where the surface crosses the edge.  Because that point
        //   lies on the edge shared by all four cells and each cell vertex
        //   lies inside its own cell, every triangle lies within the two
        //   cells that share a face containing the edge.  This is the
        //   envelope argument of Ju and Udeshi (2006): triangles confined to
        //   non-overlapping envelopes cannot intersect.  It costs up to four
        //   triangles per polygon.
        // * `QuadSplit`: libfive's two-triangle split along the diagonal
        //   whose triangles are most coplanar.  Half the triangles, but a
        //   twisted polygon still gets a folded triangle.
        // * `Hybrid` (default): the quad split when both triangles of a
        //   diagonal face along the edge the way the surface does, and the
        //   fan when neither diagonal does.  This is the rule Ju and Udeshi's
        //   own implementation uses (a diagonal is kept when the primal edge
        //   pierces its triangles), and the one Occupancy-Based Dual
        //   Contouring (Hwang and Sung, 2024) adopted from it.
        //
        // Vertices escaping their cells defeat all three; `solve_in` in the
        // QEF keeps them inside.
        // -------------------------------------------------------------------

        // Mesh index and position for the 4 corner vertices
        let abs = [
            leafs[0].index + verts[0].vert.0 as usize,
            leafs[1].index + verts[1].vert.0 as usize,
            leafs[2].index + verts[2].vert.0 as usize,
            leafs[3].index + verts[3].vert.0 as usize,
        ];
        let ps = abs.map(|i| octree.verts[i].pos);

        // The point where the surface crosses the edge, from the deepest leaf
        let edge_abs = leafs[deepest].index + verts[deepest].edge.0 as usize;

        // Cyclic winding order.  Upstream used `winding = 3` for
        // `starting_sign == true` (== -1 mod 4), so slot j stepped to j-1;
        // the same orientation is kept so triangle normals point outward.
        let order: [usize; 4] = if starting_sign {
            [0, 3, 2, 1]
        } else {
            [0, 1, 2, 3]
        };

        // Collapse consecutive slots whose cells refer to the same octree
        // cell (level-of-detail collapses at neighbour boundaries): those
        // slots carry the same vertex, so the polygon degenerates.
        let mut slots: [usize; 4] = [0; 4];
        let mut n = 0usize;
        for &s in &order {
            let same_as_prev = n > 0 && cs[slots[n - 1]].index == cs[s].index;
            if !same_as_prev {
                slots[n] = s;
                n += 1;
            }
        }
        if n > 1 && cs[slots[0]].index == cs[slots[n - 1]].index {
            n -= 1;
        }
        if n < 3 {
            return;
        }

        // Unit face normal; zero for a degenerate triangle.
        let tri_normal = |a: nalgebra::Vector3<f32>,
                          b: nalgebra::Vector3<f32>,
                          c: nalgebra::Vector3<f32>|
         -> nalgebra::Vector3<f32> {
            let n = (b - a).cross(&(c - a));
            let m = n.norm();
            if m > 1e-30 { n / m } else { n }
        };

        // The surface normal at the crossing points from the inside corner to
        // the outside corner along the edge; `starting_sign` is true when the
        // start corner is outside, so the normal then points along -t.
        let expected = if starting_sign { -1.0_f32 } else { 1.0 };
        let t_index = t.index();
        let forward = |a: usize, b: usize, c: usize| -> bool {
            tri_normal(ps[a], ps[b], ps[c])[t_index] * expected > 0.0
        };

        // Whether the primal edge passes through the triangle: the crossing
        // point, seen along the edge's axis, lies inside the triangle's
        // projection.  Tao Ju's intersection-free implementation keeps a
        // diagonal only when the edge pierces one of its triangles; a
        // diagonal whose triangles both miss the edge wraps around it and
        // reaches into a neighbouring cell's polygons.
        let edge_pos = octree.verts[edge_abs].pos;
        let (u_index, v_index) = ((t_index + 1) % 3, (t_index + 2) % 3);
        let pierces = |a: usize, b: usize, c: usize| -> bool {
            let pu = |p: nalgebra::Vector3<f32>| (p[u_index], p[v_index]);
            let (ax, ay) = pu(ps[a]);
            let (bx, by) = pu(ps[b]);
            let (cx, cy) = pu(ps[c]);
            let (ex, ey) = pu(edge_pos);
            let side = |x0: f32, y0: f32, x1: f32, y1: f32| {
                (x1 - x0) * (ey - y0) - (y1 - y0) * (ex - x0)
            };
            let s0 = side(ax, ay, bx, by);
            let s1 = side(bx, by, cx, cy);
            let s2 = side(cx, cy, ax, ay);
            (s0 >= 0.0 && s1 >= 0.0 && s2 >= 0.0)
                || (s0 <= 0.0 && s1 <= 0.0 && s2 <= 0.0)
        };

        let mut emit = |a: usize, b: usize, c: usize| {
            if !forward(a, b, c) {
                crate::stats::BACKWARD_TRIANGLES.fetch_add(1, Relaxed);
            }
            let va = out.vertex(abs[a], &octree.verts);
            let vb = out.vertex(abs[b], &octree.verts);
            let vc = out.vertex(abs[c], &octree.verts);
            out.triangle(va, vb, vc);
        };

        // Fan about the edge crossing
        let fan = |out: &mut MeshBuilder| {
            let e = out.vertex(edge_abs, &octree.verts);
            for j in 0..n {
                let a = slots[j];
                let b = slots[(j + 1) % n];
                let va = out.vertex(abs[a], &octree.verts);
                let vb = out.vertex(abs[b], &octree.verts);
                if tri_normal(ps[a], ps[b], octree.verts[edge_abs].pos)[t_index]
                    * expected
                    <= 0.0
                {
                    crate::stats::BACKWARD_TRIANGLES.fetch_add(1, Relaxed);
                }
                out.triangle(va, vb, e);
            }
        };

        match octree.triangulation {
            Triangulation::Fan => fan(out),
            Triangulation::QuadSplit | Triangulation::Hybrid => {
                let hybrid = octree.triangulation == Triangulation::Hybrid;
                if n == 3 {
                    let (a, b, c) = (slots[0], slots[1], slots[2]);
                    if hybrid && !forward(a, b, c) {
                        crate::stats::HYBRID_FANS.fetch_add(1, Relaxed);
                        fan(out);
                    } else {
                        emit(a, b, c);
                    }
                } else {
                    let (a, b, c, d) = (slots[0], slots[1], slots[2], slots[3]);
                    // Diagonal AC: triangles (a,b,c) + (a,c,d)
                    // Diagonal BD: triangles (a,b,d) + (b,c,d)
                    let score_ac = tri_normal(ps[a], ps[b], ps[c])
                        .dot(&tri_normal(ps[a], ps[c], ps[d]));
                    let score_bd = tri_normal(ps[a], ps[b], ps[d])
                        .dot(&tri_normal(ps[b], ps[c], ps[d]));
                    // A crease runs through the two vertices that sit on it
                    // (rank 2 or more); the diagonal between them keeps the
                    // crease an edge of the mesh, where the coplanarity
                    // score would cut across it.
                    let rk = |k: usize| octree.vert_rank[abs[k]];
                    let crease_ac = rk(a) >= 2
                        && rk(c) >= 2
                        && rk(b) < 2
                        && rk(d) < 2
                        && forward(a, b, c)
                        && forward(a, c, d);
                    let crease_bd = rk(b) >= 2
                        && rk(d) >= 2
                        && rk(a) < 2
                        && rk(c) < 2
                        && forward(a, b, d)
                        && forward(b, c, d);
                    let prefer_ac = if crease_ac {
                        true
                    } else if crease_bd {
                        false
                    } else {
                        score_ac >= score_bd
                    };
                    let (ac_ok, bd_ok) = if hybrid {
                        (
                            forward(a, b, c)
                                && forward(a, c, d)
                                && (pierces(a, b, c) || pierces(a, c, d)),
                            forward(a, b, d)
                                && forward(b, c, d)
                                && (pierces(a, b, d) || pierces(b, c, d)),
                        )
                    } else {
                        (true, true)
                    };
                    if ac_ok && (!bd_ok || prefer_ac) {
                        emit(a, b, c);
                        emit(a, c, d);
                    } else if bd_ok {
                        emit(a, b, d);
                        emit(b, c, d);
                    } else {
                        crate::stats::HYBRID_FANS.fetch_add(1, Relaxed);
                        fan(out);
                    }
                }
            }
        }
    } else {
        let (t, u, v) = T::frame();

        #[allow(unused_parens)]
        for i in [false, true] {
            out.edge::<T>(
                octree,
                octree.child(a, (t * i) | u | v),
                octree.child(b, (t * i) | v),
                octree.child(c, (t * i)),
                octree.child(d, (t * i) | u),
            )
        }
    }
}
