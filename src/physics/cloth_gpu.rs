use anyhow::Result;

use crate::compute::harness::{FieldHandle, GpuHarness};
use crate::vulkan::instance::VulkanContext;

pub struct ClothGpuSolver {
    pub num_vertices: u32,
    positions: FieldHandle,
    velocities: FieldHandle,
    old_positions: FieldHandle,
    pinned: FieldHandle,
    corrections: FieldHandle,    // int[num_vertices * 3] fixed-point
    correction_counts: FieldHandle,
    stretch_constraints: FieldHandle,
    bend_constraints: FieldHandle,
    num_stretch: u32,
    num_bend: u32,
    iterations: u32,
}

impl ClothGpuSolver {
    pub fn new(
        harness: &mut GpuHarness,
        ctx: &VulkanContext,
        num_vertices: u32,
        stretch_constraints: &[[u32; 4]],
        bend_constraints: &[[u32; 4]],
        iterations: u32,
    ) -> Result<Self> {
        let nv = num_vertices as usize;
        let positions = harness.create_field::<[f32; 4]>(ctx, nv)?;
        let velocities = harness.create_field::<[f32; 4]>(ctx, nv)?;
        let old_positions = harness.create_field::<[f32; 4]>(ctx, nv)?;
        let pinned = harness.create_field::<u32>(ctx, nv)?;
        let corrections = harness.create_field::<i32>(ctx, nv * 3)?;
        let correction_counts = harness.create_field::<u32>(ctx, nv)?;

        let ns = stretch_constraints.len();
        let nb = bend_constraints.len();
        let stretch_buf = harness.create_field::<[u32; 4]>(ctx, ns.max(1))?;
        let bend_buf = harness.create_field::<[u32; 4]>(ctx, nb.max(1))?;

        if ns > 0 { harness.upload(&stretch_buf, stretch_constraints)?; }
        if nb > 0 { harness.upload(&bend_buf, bend_constraints)?; }

        // Load kernels
        let predict_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/cloth_predict.comp.spv"));
        let solve_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/cloth_solve.comp.spv"));
        let apply_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/cloth_apply.comp.spv"));
        let derive_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/cloth_derive_vel.comp.spv"));

        harness.load_kernel_with_push(ctx, "cloth_predict", predict_spv, 4, 16)?;
        harness.load_kernel_with_push(ctx, "cloth_solve", solve_spv, 5, 16)?;
        harness.load_kernel_with_push(ctx, "cloth_apply", apply_spv, 4, 16)?;
        harness.load_kernel_with_push(ctx, "cloth_derive_vel", derive_spv, 4, 16)?;

        Ok(Self {
            num_vertices,
            positions,
            velocities,
            old_positions,
            pinned,
            corrections,
            correction_counts,
            stretch_constraints: stretch_buf,
            bend_constraints: bend_buf,
            num_stretch: ns as u32,
            num_bend: nb as u32,
            iterations,
        })
    }

    pub fn upload_positions(&self, harness: &GpuHarness, data: &[[f32; 4]]) -> Result<()> {
        harness.upload(&self.positions, data)
    }

    pub fn upload_pinned(&self, harness: &GpuHarness, data: &[u32]) -> Result<()> {
        harness.upload(&self.pinned, data)
    }

    pub fn step(&self, harness: &GpuHarness, ctx: &VulkanContext, dt: f32) -> Result<()> {
        let vert_groups = ((self.num_vertices + 63) / 64).max(1);

        // Predict
        let predict_push = [self.num_vertices, dt.to_bits(), (-9.81f32).to_bits(), 0u32];
        harness.dispatch_push(
            ctx, "cloth_predict",
            &[&self.positions, &self.velocities, &self.old_positions, &self.pinned],
            bytemuck::cast_slice(&predict_push),
            [vert_groups, 1, 1],
        )?;

        // Solve constraints iteratively
        for _ in 0..self.iterations {
            // Stretch constraints (stiffness = 1.0)
            if self.num_stretch > 0 {
                let stretch_groups = ((self.num_stretch + 63) / 64).max(1);
                let solve_push = [self.num_stretch, 1.0f32.to_bits(), 0u32, 0u32];
                harness.dispatch_push(
                    ctx, "cloth_solve",
                    &[&self.positions, &self.corrections, &self.correction_counts,
                      &self.stretch_constraints, &self.pinned],
                    bytemuck::cast_slice(&solve_push),
                    [stretch_groups, 1, 1],
                )?;

                let apply_push = [self.num_vertices, 0u32, 0u32, 0u32];
                harness.dispatch_push(
                    ctx, "cloth_apply",
                    &[&self.positions, &self.corrections, &self.correction_counts, &self.pinned],
                    bytemuck::cast_slice(&apply_push),
                    [vert_groups, 1, 1],
                )?;
            }

            // Bend constraints (stiffness = 0.3)
            if self.num_bend > 0 {
                let bend_groups = ((self.num_bend + 63) / 64).max(1);
                let solve_push = [self.num_bend, 0.3f32.to_bits(), 0u32, 0u32];
                harness.dispatch_push(
                    ctx, "cloth_solve",
                    &[&self.positions, &self.corrections, &self.correction_counts,
                      &self.bend_constraints, &self.pinned],
                    bytemuck::cast_slice(&solve_push),
                    [bend_groups, 1, 1],
                )?;

                let apply_push = [self.num_vertices, 0u32, 0u32, 0u32];
                harness.dispatch_push(
                    ctx, "cloth_apply",
                    &[&self.positions, &self.corrections, &self.correction_counts, &self.pinned],
                    bytemuck::cast_slice(&apply_push),
                    [vert_groups, 1, 1],
                )?;
            }
        }

        // Derive velocity
        let derive_push = [self.num_vertices, dt.to_bits(), 0.99f32.to_bits(), 0u32];
        harness.dispatch_push(
            ctx, "cloth_derive_vel",
            &[&self.positions, &self.old_positions, &self.velocities, &self.pinned],
            bytemuck::cast_slice(&derive_push),
            [vert_groups, 1, 1],
        )?;

        Ok(())
    }

    pub fn download_positions(&self, harness: &GpuHarness) -> Result<Vec<[f32; 4]>> {
        harness.download::<[f32; 4]>(&self.positions, self.num_vertices as usize)
    }
}
