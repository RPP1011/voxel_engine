use anyhow::Result;

use crate::compute::harness::{FieldHandle, GpuHarness};
use crate::vulkan::instance::VulkanContext;

pub struct SweGpuSolver {
    pub width: u32,
    pub height: u32,
    water: FieldHandle,
    terrain: FieldHandle,
    flux_right: FieldHandle,
    flux_down: FieldHandle,
    sources: FieldHandle,
    gravity_pipe: f32,
}

impl SweGpuSolver {
    pub fn new(
        harness: &mut GpuHarness,
        ctx: &VulkanContext,
        width: u32,
        height: u32,
        gravity: f32,
        pipe_area: f32,
    ) -> Result<Self> {
        let n = (width * height) as usize;
        let water = harness.create_field::<f32>(ctx, n)?;
        let terrain = harness.create_field::<f32>(ctx, n)?;
        let flux_right = harness.create_field::<f32>(ctx, n)?;
        let flux_down = harness.create_field::<f32>(ctx, n)?;
        let sources = harness.create_field::<f32>(ctx, n)?;

        let flux_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/swe_flux.comp.spv"));
        let apply_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/swe_apply.comp.spv"));

        harness.load_kernel_with_push(ctx, "swe_flux", flux_spv, 4, 16)?;
        harness.load_kernel_with_push(ctx, "swe_apply", apply_spv, 4, 16)?;

        Ok(Self { width, height, water, terrain, flux_right, flux_down, sources, gravity_pipe: gravity * pipe_area })
    }

    pub fn upload_water(&self, harness: &GpuHarness, data: &[f32]) -> Result<()> {
        harness.upload(&self.water, data)
    }

    pub fn upload_terrain(&self, harness: &GpuHarness, data: &[f32]) -> Result<()> {
        harness.upload(&self.terrain, data)
    }

    pub fn upload_sources(&self, harness: &GpuHarness, data: &[f32]) -> Result<()> {
        harness.upload(&self.sources, data)
    }

    pub fn step(&self, harness: &GpuHarness, ctx: &VulkanContext, dt: f32) -> Result<()> {
        let groups_x = (self.width + 7) / 8;
        let groups_y = (self.height + 7) / 8;

        // Push constants: width, height, dt, gravity
        let push: [u32; 4] = [
            self.width,
            self.height,
            dt.to_bits(),
            self.gravity_pipe.to_bits(),
        ];
        let push_bytes = bytemuck::cast_slice(&push);

        // Pass 1: compute fluxes
        harness.dispatch_push(
            ctx, "swe_flux",
            &[&self.water, &self.terrain, &self.flux_right, &self.flux_down],
            push_bytes, [groups_x, groups_y, 1],
        )?;

        // Pass 2: apply fluxes + sources
        let push2: [u32; 4] = [
            self.width,
            self.height,
            dt.to_bits(),
            0, // pad
        ];
        let push2_bytes = bytemuck::cast_slice(&push2);

        harness.dispatch_push(
            ctx, "swe_apply",
            &[&self.water, &self.flux_right, &self.flux_down, &self.sources],
            push2_bytes, [groups_x, groups_y, 1],
        )?;

        Ok(())
    }

    pub fn download_water(&self, harness: &GpuHarness) -> Result<Vec<f32>> {
        let n = (self.width * self.height) as usize;
        harness.download::<f32>(&self.water, n)
    }

    pub fn download_flux(&self, harness: &GpuHarness) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = (self.width * self.height) as usize;
        let fr = harness.download::<f32>(&self.flux_right, n)?;
        let fd = harness.download::<f32>(&self.flux_down, n)?;
        Ok((fr, fd))
    }

    pub fn download_terrain(&self, harness: &GpuHarness) -> Result<Vec<f32>> {
        let n = (self.width * self.height) as usize;
        harness.download::<f32>(&self.terrain, n)
    }

    /// Compute flow speed at (x, z) from downloaded flux data.
    pub fn flow_speed(flux_right: &[f32], flux_down: &[f32], x: usize, z: usize, width: usize) -> f32 {
        let i = z * width + x;
        let fr = flux_right.get(i).copied().unwrap_or(0.0);
        let fd = flux_down.get(i).copied().unwrap_or(0.0);
        (fr * fr + fd * fd).sqrt()
    }
}
