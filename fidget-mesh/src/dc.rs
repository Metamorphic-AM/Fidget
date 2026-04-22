//! Dual contouring implementation

use crate::{
    Octree,
    builder::MeshBuilder,
    cell::{Cell, CellIndex},
    frame::{Frame, XYZ, YZX, ZXY},
    types::{Corner, Edge, X, Y, Z},
};

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
                verts[i] = leafs[i].edge(edges[i]);
            } else {
                // We declare that only *manifold* leaf cells can be neighbors
                // to smaller leaf cells.  This means that there's only one
                // vertex to pick here.
                let mut iter =
                    (0..12).filter_map(|j| leafs[i].edge(Edge::new(j)));
                verts[i] = iter.next();
                if iter.any(|other| other.vert != verts[i].unwrap().vert) {
                    panic!("invalid leaf vertex at {a:?}");
                }
            }
        }

        let verts = verts.map(Option::unwrap);

        // -------------------------------------------------------------------
        // Morphé patch: libfive-style quad-split dual emission
        //
        // Upstream Fidget emits a FAN of up to 4 triangles anchored at the
        // on-surface edge-intersection vertex `i` (see `leafs[deepest]
        // .edge`).  Because each cell's QEF feature vertex lives *inside*
        // its cell, the four QEF vertices bounding a shared edge can sit
        // at significantly different positions relative to the on-surface
        // point, which produces tiny fold / flap triangles on any curved
        // surface — the classic Fidget DC artifact.
        //
        // libfive instead walks the 4 cells' QEF vertices directly as a
        // quad and splits it along the diagonal whose resulting pair of
        // triangles is the most co-planar.  This halves triangle count
        // and structurally prevents flap-fold topology.  See
        // `libfive/src/render/brep/dc/dc_mesher.cpp` — the
        // `norms[0].dot(norms[3]) > norms[1].dot(norms[2])` heuristic.
        //
        // We use *face* normals of the candidate triangles (purely
        // topological, no shape evaluation needed) rather than libfive's
        // per-vertex surface normals.  This is a strictly stronger
        // flap-suppressor since flaps are exactly the case where the
        // two face normals oppose; the heuristic eliminates them by
        // construction.
        // -------------------------------------------------------------------

        // Collect mesh indices + world positions for the 4 corner vertices
        let abs = [
            leafs[0].index + verts[0].vert.0 as usize,
            leafs[1].index + verts[1].vert.0 as usize,
            leafs[2].index + verts[2].vert.0 as usize,
            leafs[3].index + verts[3].vert.0 as usize,
        ];
        let vs = [
            out.vertex(abs[0], &octree.verts),
            out.vertex(abs[1], &octree.verts),
            out.vertex(abs[2], &octree.verts),
            out.vertex(abs[3], &octree.verts),
        ];
        let ps = [
            octree.verts[abs[0]].pos,
            octree.verts[abs[1]].pos,
            octree.verts[abs[2]].pos,
            octree.verts[abs[3]].pos,
        ];

        // Cyclic winding order.  Fidget's original code used
        // `winding = 3` for `starting_sign == true` (== -1 mod 4), so
        // slot j stepped to j-1; we preserve the same orientation here
        // so emitted triangle normals still point outward.
        let order: [usize; 4] = if starting_sign {
            [0, 3, 2, 1]
        } else {
            [0, 1, 2, 3]
        };

        // Collapse consecutive slots whose cells refer to the same
        // octree cell (level-of-detail collapses at neighbour
        // boundaries): those slots carry the same QEF vertex, so the
        // polygon degenerates.
        let mut loop_slots: [usize; 4] = [0; 4];
        let mut loop_len = 0usize;
        for &s in &order {
            let same_as_prev = if loop_len == 0 {
                false
            } else {
                cs[loop_slots[loop_len - 1]].index == cs[s].index
            };
            if !same_as_prev {
                loop_slots[loop_len] = s;
                loop_len += 1;
            }
        }
        // Wrap-around dedup (first == last after a full walk)
        if loop_len > 1
            && cs[loop_slots[0]].index == cs[loop_slots[loop_len - 1]].index
        {
            loop_len -= 1;
        }

        // Face-normal helper.  Returns zero vector for degenerate
        // triangles; dot-product comparison handles that gracefully.
        let tri_normal = |a: nalgebra::Vector3<f32>,
                          b: nalgebra::Vector3<f32>,
                          c: nalgebra::Vector3<f32>|
         -> nalgebra::Vector3<f32> {
            let n = (b - a).cross(&(c - a));
            let m = n.norm();
            if m > 1e-30 { n / m } else { n }
        };

        match loop_len {
            3 => {
                let (a, b, c) =
                    (loop_slots[0], loop_slots[1], loop_slots[2]);
                out.triangle(vs[a], vs[b], vs[c]);
            }
            4 => {
                let (a, b, c, d) = (
                    loop_slots[0],
                    loop_slots[1],
                    loop_slots[2],
                    loop_slots[3],
                );
                // Diagonal AC: triangles (a,b,c) + (a,c,d)
                // Diagonal BD: triangles (a,b,d) + (b,c,d)
                let n_ac_1 = tri_normal(ps[a], ps[b], ps[c]);
                let n_ac_2 = tri_normal(ps[a], ps[c], ps[d]);
                let n_bd_1 = tri_normal(ps[a], ps[b], ps[d]);
                let n_bd_2 = tri_normal(ps[b], ps[c], ps[d]);
                let score_ac = n_ac_1.dot(&n_ac_2);
                let score_bd = n_bd_1.dot(&n_bd_2);
                if score_ac >= score_bd {
                    out.triangle(vs[a], vs[b], vs[c]);
                    out.triangle(vs[a], vs[c], vs[d]);
                } else {
                    out.triangle(vs[a], vs[b], vs[d]);
                    out.triangle(vs[b], vs[c], vs[d]);
                }
            }
            _ => {
                // 0, 1, or 2 unique cells -> no triangle
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
