use super::cell::{CellBounds, CellVertex};
use crate::stats::Placement;

/// Solver for a quadratic error function to position a vertex within a cell
#[derive(Copy, Clone, Debug, Default)]
pub struct QuadraticErrorSolver {
    /// A^T A term
    ata: nalgebra::Matrix3<f32>,

    /// A^T B term
    atb: nalgebra::Vector3<f32>,

    /// B^T B term
    btb: f32,

    /// Mass point of intersections is stored as XYZ / W, so that summing works
    mass_point: nalgebra::Vector4<f32>,
}

impl std::ops::AddAssign for QuadraticErrorSolver {
    fn add_assign(&mut self, rhs: Self) {
        self.ata += rhs.ata;
        self.atb += rhs.atb;
        self.btb += rhs.btb;
        self.mass_point += rhs.mass_point;
    }
}

/// Relative eigenvalue cutoff for the rank of the unconstrained solve.
///
/// Skip any eigenvalues that are small relative to the maximum eigenvalue.
/// This is very much a tuned value (alas!).  If the value is too small, then
/// we incorrectly pick high-rank solutions, which may shoot vertices out of
/// their cells in near-planar situations.  If the value is too large, then we
/// incorrectly pick low-rank solutions, which makes us less likely to snap to
/// sharp features.
///
/// For example, our cone test needs to use a rank-3 solver for eigenvalues of
/// [1.5633028, 1.430821, 0.0058764853] (a dynamic range of 2e3); while the
/// bear model needs to use a rank-2 solver for eigenvalues of [2.87, 0.13,
/// 5.64e-7] (a dynamic range of 10^7).  We pick 10^3 here somewhat arbitrarily
/// to be within that range.
const EIGENVALUE_CUTOFF_RELATIVE: f32 = 1e-3;

/// Fraction of a cell's width by which a vertex may sit outside the cell and
/// still be accepted.
///
/// A sharp feature does not stop at cell boundaries: a cell that both faces
/// of a crease pass through, while the crease itself runs just past the cell,
/// has its best vertex on that crease, outside the cell.  Rejecting it rounds
/// or serrates the crease.  Accepting vertices up to this margin outside the
/// cell keeps such features while still rejecting the far shots of
/// near-planar cells, which are what fold the mesh.  In cells.
const CONTAINMENT_MARGIN: f32 = 1.0;

/// Smallest singular value of `A^T A` (normals are unit length, so this is
/// in units of one sample's weight) for a direction to count as strongly
/// supported.  A vertex is allowed outside its cell only when every retained
/// direction is this strong: that is a crease or corner that runs past the
/// cell, which the vertex must follow.  A near-planar cell whose extra
/// directions rest on a few degrees of normal variation keeps its vertex
/// inside, since letting it out folds the surface.  The value is libfive's
/// `EIGENVALUE_CUTOFF`.
const STRONG_FEATURE_CUTOFF: f32 = 0.1;


impl QuadraticErrorSolver {
    pub fn new() -> Self {
        Self {
            ata: nalgebra::Matrix3::zeros(),
            atb: nalgebra::Vector3::zeros(),
            btb: 0.0,
            mass_point: nalgebra::Vector4::zeros(),
        }
    }

    #[cfg(test)]
    pub fn mass_point(&self) -> nalgebra::Vector4<f32> {
        self.mass_point
    }

    /// Adds a new intersection to the QEF
    ///
    /// `pos` is the position of the intersection and is accumulated in the mass
    /// point.  `grad` is the gradient at the surface, and is normalized in this
    /// function.
    pub fn add_intersection(
        &mut self,
        pos: nalgebra::Vector3<f32>,
        grad: nalgebra::Vector4<f32>,
    ) {
        // TODO: correct for non-zero distance value in grad.w
        self.mass_point += nalgebra::Vector4::new(pos.x, pos.y, pos.z, 1.0);
        let norm = grad.xyz().normalize();
        self.ata += norm * norm.transpose();
        self.atb += norm * norm.dot(&pos);
        self.btb += norm.dot(&pos).powi(2);
    }

    /// QEF error at `pos`, clamped to a small positive value for ease of
    /// comparison
    fn error_at(&self, pos: &nalgebra::Vector3<f32>) -> f32 {
        ((pos.transpose() * self.ata * pos - 2.0 * pos.transpose() * self.atb)
            [0]
            + self.btb)
            .max(1e-6)
    }

    /// Solve the given QEF, minimizing towards the mass point
    ///
    /// The vertex is not constrained to any cell; see [`Self::solve_in`] for
    /// the bounded variant that the mesher uses.
    ///
    /// Also returns the QEF error as the second item in the tuple
    pub fn solve(&self) -> (CellVertex<3>, f32) {
        let s = Decomposed::new(self);
        let pos = s.solve_rank(s.rank);
        (CellVertex { pos }, self.error_at(&pos))
    }

    /// Solve the given QEF, keeping the vertex within `bounds`
    ///
    /// The solution is the first of the following that lies within the cell
    /// (to a small tolerance), in this order:
    ///
    /// 1. the unconstrained solution at the rank chosen by the eigenvalue
    ///    cutoff, minimizing towards the mass point;
    /// 2. the unconstrained solution at each lower rank in turn, which pulls
    ///    the vertex back towards the mass point along the least certain
    ///    directions first;
    /// 3. the lowest-error solution constrained to one of the cell's faces,
    ///    edges or corners (the bounded solve of libfive's simplex mesher);
    /// 4. the mass point itself, which is a convex combination of points on
    ///    the cell's edges and therefore always inside.
    ///
    /// Keeping every vertex inside its cell is what makes the dual polygons
    /// around an edge lie within the cells that share the edge, which is the
    /// condition under which dual contouring output is free of
    /// self-intersections (Ju and Udeshi, 2006).
    ///
    /// Returns the vertex, its QEF error, and how it was placed.
    pub fn solve_in(
        &self,
        bounds: &CellBounds<3>,
    ) -> (CellVertex<3>, f32, Placement) {
        self.solve_bounded(bounds, true)
    }

    /// As [`Self::solve_in`], but the vertex is never allowed outside the
    /// cell, feature or not.  Used for the vertex of one sheet of a cell that
    /// holds several, whose crease is carried by other cells.
    pub fn solve_contained(
        &self,
        bounds: &CellBounds<3>,
    ) -> (CellVertex<3>, f32, Placement) {
        self.solve_bounded(bounds, false)
    }

    /// Rank of the solve at the eigenvalue cutoff: 1 on a plane, 2 on a
    /// crease, 3 at a corner
    pub fn rank(&self) -> u8 {
        Decomposed::new(self).rank as u8
    }

    /// The unconstrained solution when it is a strongly supported crease or
    /// corner (rank 2 or 3), wherever it lies; `None` on a plane or a weak
    /// feature
    pub fn feature_point(&self) -> Option<nalgebra::Vector3<f32>> {
        let s = Decomposed::new(self);
        if s.rank >= 2 && s.singular_values[s.rank - 1] >= STRONG_FEATURE_CUTOFF {
            let p = s.solve_rank(s.rank);
            p.iter().all(|c| c.is_finite()).then_some(p)
        } else {
            None
        }
    }

    fn solve_bounded(
        &self,
        bounds: &CellBounds<3>,
        allow_feature_outside: bool,
    ) -> (CellVertex<3>, f32, Placement) {
        let s = Decomposed::new(self);
        let inside = |p: &nalgebra::Vector3<f32>| contains(bounds, p);

        // 1: the unconstrained solve at the cutoff rank.  Inside the cell it
        // is accepted as it is.  Outside, it is still accepted within the
        // margin when every retained direction is strongly supported: the
        // cell then holds two or three sheets of a feature whose crease or
        // corner runs past the cell, and the vertex belongs on it.
        let pos = s.solve_rank(s.rank);
        let strong = s.rank >= 1
            && s.singular_values[s.rank - 1] >= STRONG_FEATURE_CUTOFF;
        let exc = excess(bounds, &pos);
        if exc <= 1e-4
            || (allow_feature_outside && strong && exc <= CONTAINMENT_MARGIN)
        {
            return (CellVertex { pos }, self.error_at(&pos), Placement::Interior);
        }

        // 2: unconstrained solves at decreasing rank, which pull the vertex
        // back towards the mass point along the least certain directions
        for rank in (1..s.rank).rev() {
            let pos = s.solve_rank(rank);
            if inside(&pos) {
                return (CellVertex { pos }, self.error_at(&pos), Placement::LowerRank);
            }
        }

        // 3: constrained to the boundary of the cell.  Only reached when the
        // unconstrained solution escapes, so the extra solves are rare.
        // A plane or a crease restricted to a face leaves a line or a point
        // of equal error, and the cell's faces offer several of them; of
        // candidates with the same error the one nearest the mass point is
        // taken, so the vertex stays among its crossings rather than at the
        // far end of a face.
        let cutoff = s.cutoff;
        let center = s.center;
        let mut best: Option<(nalgebra::Vector3<f32>, f32, Placement, f32)> = None;
        let mut consider = |pos: nalgebra::Vector3<f32>, how: Placement| {
            if !inside(&pos) {
                return;
            }
            let err = self.error_at(&pos);
            let dist = (pos - center).norm();
            let better = match best {
                None => true,
                Some((_, b_err, _, b_dist)) => {
                    let tol = 1e-9 + 1e-3 * b_err.max(err);
                    err + tol < b_err || ((err - b_err).abs() <= tol && dist < b_dist)
                }
            };
            if better {
                best = Some((pos, err, how, dist));
            }
        };
        for k in 0..3 {
            for side in [bounds.bounds[k].lower(), bounds.bounds[k].upper()] {
                consider(self.solve_on_face(&s.center, k, side, cutoff), Placement::Face);
            }
        }
        for k in 0..3 {
            let i = (k + 1) % 3;
            let j = (k + 2) % 3;
            for ci in [bounds.bounds[i].lower(), bounds.bounds[i].upper()] {
                for cj in [bounds.bounds[j].lower(), bounds.bounds[j].upper()] {
                    consider(
                        self.solve_on_edge(&s.center, k, (i, ci), (j, cj), cutoff),
                        Placement::Edge,
                    );
                }
            }
        }
        for c in 0..8 {
            let pos = nalgebra::Vector3::new(
                if c & 1 != 0 { bounds.bounds[0].upper() } else { bounds.bounds[0].lower() },
                if c & 2 != 0 { bounds.bounds[1].upper() } else { bounds.bounds[1].lower() },
                if c & 4 != 0 { bounds.bounds[2].upper() } else { bounds.bounds[2].lower() },
            );
            consider(pos, Placement::Corner);
        }
        if let Some((pos, err, how, _)) = best {
            return (CellVertex { pos }, err, how);
        }

        // 4: the mass point
        let pos = s.center;
        (CellVertex { pos }, self.error_at(&pos), Placement::MassPoint)
    }

    /// Minimizes the QEF on the plane `x[k] = c`, towards the mass point
    /// projected onto that plane
    fn solve_on_face(
        &self,
        center: &nalgebra::Vector3<f32>,
        k: usize,
        c: f32,
        cutoff: f32,
    ) -> nalgebra::Vector3<f32> {
        let i = (k + 1) % 3;
        let j = (k + 2) % 3;
        let mut m = *center;
        m[k] = c;
        // Stationarity of E(m + d) restricted to d[k] = 0:
        //   ata[I,I] d[I] = (atb - ata m)[I]
        let r = self.atb - self.ata * m;
        let a = nalgebra::Matrix2::new(
            self.ata[(i, i)],
            self.ata[(i, j)],
            self.ata[(j, i)],
            self.ata[(j, j)],
        );
        let b = nalgebra::Vector2::new(r[i], r[j]);
        let svd = nalgebra::linalg::SVD::new(a, true, true);
        let d = svd.solve(&b, cutoff).unwrap_or_else(|_| nalgebra::Vector2::zeros());
        let mut pos = m;
        pos[i] += d[0];
        pos[j] += d[1];
        pos
    }

    /// Minimizes the QEF on the line `x[i] = ci, x[j] = cj`, towards the mass
    /// point projected onto that line
    fn solve_on_edge(
        &self,
        center: &nalgebra::Vector3<f32>,
        k: usize,
        (i, ci): (usize, f32),
        (j, cj): (usize, f32),
        cutoff: f32,
    ) -> nalgebra::Vector3<f32> {
        let mut m = *center;
        m[i] = ci;
        m[j] = cj;
        let r = self.atb - self.ata * m;
        let a = self.ata[(k, k)];
        let mut pos = m;
        if a > cutoff {
            pos[k] += r[k] / a;
        }
        pos
    }
}

/// How far `p` lies outside `bounds`, as the largest excess over any axis in
/// units of that axis's cell width; 0 inside, infinite for a non-finite point
pub(crate) fn excess(bounds: &CellBounds<3>, p: &nalgebra::Vector3<f32>) -> f32 {
    let mut worst = 0.0_f32;
    for a in 0..3 {
        if !p[a].is_finite() {
            return f32::INFINITY;
        }
        let b = bounds.bounds[a];
        let w = b.width().max(1e-30);
        let e = (b.lower() - p[a]).max(p[a] - b.upper()).max(0.0) / w;
        worst = worst.max(e);
    }
    worst
}

/// Whether `p` lies within `bounds`, to a small tolerance for features that
/// sit exactly on a cell face
fn contains(bounds: &CellBounds<3>, p: &nalgebra::Vector3<f32>) -> bool {
    excess(bounds, p) <= 1e-4
}

/// The SVD of `A^T A`, with its singular values sorted and the rank chosen
/// by the relative eigenvalue cutoff
struct Decomposed {
    svd: nalgebra::linalg::SVD<f32, nalgebra::U3, nalgebra::U3>,
    /// Singular values in decreasing order
    singular_values: [f32; 3],
    /// Number of singular values above the cutoff
    rank: usize,
    /// Absolute cutoff, relative to the largest singular value
    cutoff: f32,
    /// Mass point
    center: nalgebra::Vector3<f32>,
    /// `A^T B` shifted to minimize towards the mass point
    atb: nalgebra::Vector3<f32>,
}

impl Decomposed {
    fn new(q: &QuadraticErrorSolver) -> Self {
        // This gets a little tricky; see
        // https://www.mattkeeter.com/projects/qef for a walkthrough of QEF math
        // and references to primary sources.
        let center = q.mass_point.xyz() / q.mass_point.w;
        let atb = q.atb - q.ata * center;

        let svd = nalgebra::linalg::SVD::new(q.ata, true, true);

        // nalgebra doesn't always actually order singular values (?!?)
        // https://github.com/dimforge/nalgebra/issues/1215
        let mut singular_values =
            svd.singular_values.data.0[0].map(ordered_float::OrderedFloat);
        singular_values.sort();
        singular_values.reverse();
        let singular_values = singular_values.map(|o| o.0);

        let cutoff = singular_values[0].abs() * EIGENVALUE_CUTOFF_RELATIVE;

        // Intuition about `rank`:
        // 0 => all eigenvalues are invalid (?!), use the center point
        // 1 => the first eigenvalue is valid, this must be planar
        // 2 => the first two eigenvalues are valid, this is a planar or an edge
        // 3 => all eigenvalues are valid, this is a planar, edge, or corner
        let rank = (0..3)
            .find(|i| singular_values[*i].abs() < cutoff)
            .unwrap_or(3);

        Self {
            svd,
            singular_values,
            rank,
            cutoff,
            center,
            atb,
        }
    }

    /// Unconstrained minimizer keeping the `rank` largest singular values
    fn solve_rank(&self, rank: usize) -> nalgebra::Vector3<f32> {
        if rank == 0 {
            return self.center;
        }
        // Singular values at or below `epsilon` are treated as zero by
        // `solve`, so passing the first discarded value keeps exactly the
        // `rank` largest ones.
        let epsilon = self.singular_values.get(rank).cloned().unwrap_or(0.0);
        let sol = self.svd.solve(&self.atb, epsilon);
        sol.map(|c| c + self.center).unwrap_or(self.center)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use fidget_core::types::Interval;
    use nalgebra::{Vector3, Vector4};

    #[test]
    fn qef_rank2() {
        let mut q = QuadraticErrorSolver::new();
        q.add_intersection(
            Vector3::new(-0.5, -0.75, -0.75),
            Vector4::new(0.24, 0.12, 0.0, 0.0),
        );
        q.add_intersection(
            Vector3::new(-0.75, -1.0, -0.6),
            Vector4::new(0.0, 0.0, 0.31, 0.0),
        );
        q.add_intersection(
            Vector3::new(-0.50, -1.0, -0.6),
            Vector4::new(0.0, 0.0, 0.31, 0.0),
        );
        let (_out, err) = q.solve();
        assert_eq!(err, 1e-6);
    }

    #[test]
    fn qef_near_planar() {
        let mut q = QuadraticErrorSolver::new();
        q.add_intersection(
            Vector3::new(-0.5, -0.25, 0.4999981),
            Vector4::new(-0.66666776, -0.33333388, 0.66666526, -1.2516975e-6),
        );
        q.add_intersection(
            Vector3::new(-0.5, -0.25, 0.50),
            Vector4::new(-0.6666667, -0.33333334, 0.6666667, 0.0),
        );
        q.add_intersection(
            Vector3::new(-0.5, -0.25, 0.50),
            Vector4::new(-0.6666667, -0.33333334, 0.6666667, 0.0),
        );
        let (out, err) = q.solve();
        assert_eq!(err, 1e-6);
        let expected = Vector3::new(-0.5, -0.25, 0.5);
        assert!(
            (out.pos - expected).norm() < 1e-3,
            "expected {expected:?}, got {:?}",
            out.pos
        );
    }

    fn unit_cell() -> CellBounds<3> {
        CellBounds {
            bounds: [Interval::new(0.0, 1.0); 3],
        }
    }

    /// A corner inside the cell is reproduced exactly
    #[test]
    fn bounded_keeps_an_interior_corner() {
        let mut q = QuadraticErrorSolver::new();
        q.add_intersection(Vector3::new(0.3, 0.5, 0.5), Vector4::new(1.0, 0.0, 0.0, 0.0));
        q.add_intersection(Vector3::new(0.5, 0.3, 0.5), Vector4::new(0.0, 1.0, 0.0, 0.0));
        q.add_intersection(Vector3::new(0.5, 0.5, 0.3), Vector4::new(0.0, 0.0, 1.0, 0.0));
        let (v, _, how) = q.solve_in(&unit_cell());
        assert_eq!(how, Placement::Interior);
        assert!((v.pos - Vector3::new(0.3, 0.3, 0.3)).norm() < 1e-5, "{:?}", v.pos);
    }

    /// Two planes whose crease lies outside the cell: the vertex stays
    /// inside, on the cell boundary nearest the crease, rather than escaping
    #[test]
    fn bounded_solution_stays_inside() {
        let mut q = QuadraticErrorSolver::new();
        // Planes x = 1.5 and y = 1.5 sampled on the cell's edges, with a
        // slight tilt so the solve is rank 2.
        q.add_intersection(Vector3::new(1.0, 0.2, 0.5), Vector4::new(1.0, 0.05, 0.0, 0.0));
        q.add_intersection(Vector3::new(0.2, 1.0, 0.5), Vector4::new(0.05, 1.0, 0.0, 0.0));
        q.add_intersection(Vector3::new(0.6, 0.6, 0.5), Vector4::new(0.7, 0.7, 0.0, 0.0));
        let cell = unit_cell();
        let (unbounded, _) = q.solve();
        let (v, _, how) = q.solve_in(&cell);
        assert!(contains(&cell, &v.pos), "{:?} via {how:?}", v.pos);
        // The unconstrained problem is rank deficient along z, so the test is
        // only meaningful if the free solve really left the cell.
        if !contains(&cell, &unbounded.pos) {
            assert_ne!(how, Placement::Interior);
        }
    }

    /// Every placement path returns a point inside the cell
    #[test]
    fn bounded_never_escapes() {
        let cell = unit_cell();
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..2000 {
            let mut q = QuadraticErrorSolver::new();
            let n = 2 + (rnd() * 4.0) as usize;
            for _ in 0..n {
                // Intersection on a cell edge
                let axis = (rnd() * 3.0) as usize;
                let mut p = Vector3::new(rnd().round(), rnd().round(), rnd().round());
                p[axis] = rnd();
                // Arbitrary, possibly contradictory, normal
                let g = Vector3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5);
                if g.norm() < 1e-3 {
                    continue;
                }
                q.add_intersection(p, Vector4::new(g.x, g.y, g.z, 0.0));
            }
            if q.mass_point.w == 0.0 {
                continue;
            }
            let (v, err, how) = q.solve_in(&cell);
            // Inside the cell, or at most the feature margin outside it when
            // the solve was strongly supported in every retained direction.
            let exc = excess(&cell, &v.pos);
            assert!(
                exc <= CONTAINMENT_MARGIN + 1e-4,
                "{:?} via {how:?} is {exc} cells outside",
                v.pos
            );
            if how != Placement::Interior {
                assert!(contains(&cell, &v.pos), "{:?} via {how:?}", v.pos);
            }
            assert!(err.is_finite());
        }
    }
}
