use anyhow::Result;

use crate::compute::harness::{FieldHandle, GpuHarness};
use crate::vulkan::instance::VulkanContext;

const TABLE_SIZE: u32 = 512;
const MAX_PER_CELL: u32 = 16;

pub struct SphGpuSolver {
    pub num_particles: u32,
    positions: FieldHandle,
    velocities: FieldHandle,
    densities: FieldHandle,
    cell_count: FieldHandle,
    cell_entries: FieldHandle,
    kernel_radius: f32,
    cell_size: f32,
    rest_density: f32,
    stiffness: f32,
    viscosity: f32,
    gravity: f32,
    dt: f32,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
}

impl SphGpuSolver {
    pub fn new(
        harness: &mut GpuHarness,
        ctx: &VulkanContext,
        num_particles: u32,
        kernel_radius: f32,
        rest_density: f32,
    ) -> Result<Self> {
        let np = num_particles as usize;
        let positions = harness.create_field::<[f32; 4]>(ctx, np)?;
        let velocities = harness.create_field::<[f32; 4]>(ctx, np)?;
        let densities = harness.create_field::<f32>(ctx, np)?;
        let cell_count = harness.create_field::<u32>(ctx, TABLE_SIZE as usize)?;
        let cell_entries = harness.create_field::<u32>(ctx, (TABLE_SIZE * MAX_PER_CELL) as usize)?;

        let hash_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/sph_hash.comp.spv"));
        let density_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/sph_density.comp.spv"));
        let forces_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/sph_forces.comp.spv"));

        harness.load_kernel_with_push(ctx, "sph_hash", hash_spv, 3, 16)?;
        harness.load_kernel_with_push(ctx, "sph_density", density_spv, 4, 32)?;
        harness.load_kernel_with_push(ctx, "sph_forces", forces_spv, 5, 68)?; // 17 floats rounded to 68

        let h = kernel_radius;
        let poly6_coeff = 315.0 / (64.0 * std::f32::consts::PI * h.powi(9));

        Ok(Self {
            num_particles,
            positions,
            velocities,
            densities,
            cell_count,
            cell_entries,
            kernel_radius,
            cell_size: kernel_radius,
            rest_density,
            stiffness: 50.0,
            viscosity: 0.01,
            gravity: -9.81,
            dt: 0.005,
            bounds_min: [0.0, 0.0, 0.0],
            bounds_max: [10.0, 10.0, 10.0],
        })
    }

    pub fn set_bounds(&mut self, min: [f32; 3], max: [f32; 3]) {
        self.bounds_min = min;
        self.bounds_max = max;
    }

    pub fn upload_positions(&self, harness: &GpuHarness, data: &[[f32; 4]]) -> Result<()> {
        harness.upload(&self.positions, data)
    }

    pub fn step(&self, harness: &GpuHarness, ctx: &VulkanContext) -> Result<()> {
        let groups = ((self.num_particles + 63) / 64).max(1);

        // Clear spatial hash
        harness.clear_field(&self.cell_count)?;

        // Pass 1: Build spatial hash
        let hash_push: [u32; 4] = [
            self.num_particles,
            self.cell_size.to_bits(),
            TABLE_SIZE,
            MAX_PER_CELL,
        ];
        harness.dispatch_push(
            ctx, "sph_hash",
            &[&self.positions, &self.cell_count, &self.cell_entries],
            bytemuck::cast_slice(&hash_push),
            [groups, 1, 1],
        )?;

        // Pass 2: Compute densities
        let h = self.kernel_radius;
        let poly6_coeff = 315.0 / (64.0 * std::f32::consts::PI * h.powi(9));
        let density_push: [u32; 8] = [
            self.num_particles,
            h.to_bits(),
            poly6_coeff.to_bits(),
            TABLE_SIZE,
            MAX_PER_CELL,
            self.cell_size.to_bits(),
            0, // pad
            0, // pad
        ];
        harness.dispatch_push(
            ctx, "sph_density",
            &[&self.positions, &self.densities, &self.cell_count, &self.cell_entries],
            bytemuck::cast_slice(&density_push),
            [groups, 1, 1],
        )?;

        // Pass 3: Compute forces and integrate
        // 17 values * 4 bytes = 68 bytes, but must be aligned to 4
        let forces_push: [u32; 17] = [
            self.num_particles,
            self.kernel_radius.to_bits(),
            self.dt.to_bits(),
            self.stiffness.to_bits(),
            self.viscosity.to_bits(),
            self.gravity.to_bits(),
            self.rest_density.to_bits(),
            TABLE_SIZE,
            MAX_PER_CELL,
            self.cell_size.to_bits(),
            self.bounds_min[0].to_bits(),
            self.bounds_min[1].to_bits(),
            self.bounds_min[2].to_bits(),
            self.bounds_max[0].to_bits(),
            self.bounds_max[1].to_bits(),
            self.bounds_max[2].to_bits(),
            0, // pad
        ];
        harness.dispatch_push(
            ctx, "sph_forces",
            &[&self.positions, &self.velocities, &self.densities, &self.cell_count, &self.cell_entries],
            bytemuck::cast_slice(&forces_push),
            [groups, 1, 1],
        )?;

        Ok(())
    }

    pub fn download_positions(&self, harness: &GpuHarness) -> Result<Vec<[f32; 4]>> {
        harness.download::<[f32; 4]>(&self.positions, self.num_particles as usize)
    }
}
