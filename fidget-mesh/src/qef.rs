use super::cell::CellVertex;

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

    /// Solve the given QEF, minimizing towards the mass point
    ///
    /// `cell_extent` is the largest axis-aligned extent of the cell this QEF
    /// belongs to (in world units).  It is used by the mass-point acceptance
    /// test described below to distinguish "vertex snapped to a genuine sharp
    /// feature" from "QEF drove the vertex out to infinity because the
    /// gradient field lied".  Pass `1.0` if the caller has no meaningful
    /// notion of cell size (e.g. synthetic unit tests).
    ///
    /// Returns a vertex localized within the given cell, and adjusts the solver
    /// to increase the likelihood that the vertex is bounded in the cell.
    ///
    /// Also returns the QEF error as the second item in the tuple
    pub fn solve(&self, cell_extent: f32) -> (CellVertex<3>, f32) {
        // This gets a little tricky; see
        // https://www.mattkeeter.com/projects/qef for a walkthrough of QEF math
        // and references to primary sources.
        let center = self.mass_point.xyz() / self.mass_point.w;
        let atb = self.atb - self.ata * center;

        let svd = nalgebra::linalg::SVD::new(self.ata, true, true);

        // nalgebra doesn't always actually order singular values (?!?)
        // https://github.com/dimforge/nalgebra/issues/1215
        let mut singular_values =
            svd.singular_values.data.0[0].map(ordered_float::OrderedFloat);
        singular_values.sort();
        singular_values.reverse();
        let singular_values = singular_values.map(|o| o.0);

        // Skip any eigenvalues that are small relative to the maximum
        // eigenvalue.  This is very much a tuned value (alas!).  If the value
        // is too small, then we incorrectly pick high-rank solutions, which may
        // shoot vertices out of their cells in near-planar situations.  If the
        // value is too large, then we incorrectly pick low-rank solutions,
        // which makes us less likely to snap to sharp features.
        //
        // For example, our cone test needs to use a rank-3 solver for
        // eigenvalues of [1.5633028, 1.430821, 0.0058764853] (a dynamic range
        // of 2e3); while the bear model needs to use a rank-2 solver for
        // eigenvalues of [2.87, 0.13, 5.64e-7] (a dynamic range of 10^7).  We
        // pick 10^3 here somewhat arbitrarily to be within that range.
        const EIGENVALUE_CUTOFF_RELATIVE: f32 = 1e-3;
        let cutoff = singular_values[0].abs() * EIGENVALUE_CUTOFF_RELATIVE;

        // Intuition about `rank`:
        // 0 => all eigenvalues are invalid (?!), use the center point
        // 1 => the first eigenvalue is valid, this must be planar
        // 2 => the first two eigenvalues are valid, this is a planar or an edge
        // 3 => all eigenvalues are valid, this is a planar, edge, or corner
        let rank = (0..3)
            .find(|i| singular_values[*i].abs() < cutoff)
            .unwrap_or(3);

        let epsilon = singular_values.get(rank).cloned().unwrap_or(0.0);
        let sol = svd.solve(&atb, epsilon);
        let pos = sol.map(|c| c + center).unwrap_or(center);
        // We'll clamp the error to a small > 0 value for ease of comparison
        let err = ((pos.transpose() * self.ata * pos
            - 2.0 * pos.transpose() * self.atb)[0]
            + self.btb)
            .max(1e-6);

        // --- Mass-point acceptance test (Morphé) ------------------------
        // DC/MDC's QEF solver assumes the gradient field is locally
        // Lipschitz-continuous inside a cell: every sampled normal is a
        // reasonable linear approximation of the surface in that cell, and
        // the planes they define intersect near the surface.  Mesh-derived
        // SDFs (and unions/intersections of them) routinely violate this at
        // the medial axis, where the closest-feature swaps between two
        // disparate triangles within a single cell.  The gradients that
        // result point in nearly opposite directions; the QEF then has no
        // well-conditioned intersection and can solve to a vertex tens or
        // hundreds of cell-widths outside the cell, producing the "spike"
        // triangles and overlapping sheets we observe on mesh-derived
        // inputs near sharp edges.
        //
        // libfive handles this with a mass-point acceptance test: compare
        // the solved position to the centroid of the gradient-sample
        // positions and the residual error to a per-plane spread measure;
        // if either is larger than a scale-relative threshold, fall back to
        // the centroid, which is always well-defined and always inside the
        // convex hull of the samples (and hence inside the cell for a
        // convex cell).  The thresholds are expressed in units of the
        // cell's own extent so they rescale automatically when the mesher
        // subdivides — no per-scene tuning.
        //
        // `MASS_POINT_DEVIATION` is allowed to be ≥ 1.0 because a genuine
        // sharp-feature snap can land a vertex near the cell boundary
        // opposite the intersection samples.  We only reject solutions
        // that have escaped the cell by a large multiple of its extent.
        //
        // `PER_PLANE_ERROR` caps the mean squared distance from the solved
        // position to each constraint plane.  For well-conditioned cells
        // on genuine surfaces this is ~1e-6 (the floor).  For medial-axis
        // cells with contradictory normals, residual scales as the plane
        // disagreement times the inter-plane separation squared, and
        // quickly exceeds a fraction of cell_extent².
        const MASS_POINT_DEVIATION: f32 = 1.0;
        const PER_PLANE_ERROR: f32 = 0.25;
        let count = self.mass_point.w;
        if cell_extent > 0.0 && count > 0.0 {
            let deviation = (pos - center).norm();
            let per_plane_err = err / count;
            let dev_limit = MASS_POINT_DEVIATION * cell_extent;
            let err_limit =
                PER_PLANE_ERROR * cell_extent * cell_extent;
            if deviation > dev_limit || per_plane_err > err_limit {
                // Fall back to the mass-point centroid and recompute err
                // at that position so downstream consumers (cell-collapse
                // heuristic) see consistent (pos, err) data.
                let pos = center;
                let err = ((pos.transpose() * self.ata * pos
                    - 2.0 * pos.transpose() * self.atb)[0]
                    + self.btb)
                    .max(1e-6);
                return (CellVertex { pos }, err);
            }
        }

        (CellVertex { pos }, err)
    }
}

#[cfg(test)]
mod test {
    use super::*;
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
        let (_out, err) = q.solve(1.0);
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
        let (out, err) = q.solve(1.0);
        assert_eq!(err, 1e-6);
        let expected = Vector3::new(-0.5, -0.25, 0.5);
        assert!(
            (out.pos - expected).norm() < 1e-3,
            "expected {expected:?}, got {:?}",
            out.pos
        );
    }
}
