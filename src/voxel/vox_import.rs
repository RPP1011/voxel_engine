use anyhow::{Context, Result};
use super::grid::VoxelGrid;
use super::material::{MaterialPalette, MaterialType, PaletteEntry};

/// Result of loading a .vox file.
pub struct VoxImportResult {
    pub grid: VoxelGrid,
    pub palette: MaterialPalette,
}

/// Load a MagicaVoxel .vox file from bytes.
/// `color_to_material` maps palette colors to material types.
pub fn load_vox(
    data: &[u8],
    color_to_material: Option<&dyn Fn(u8, u8, u8) -> MaterialType>,
) -> Result<VoxImportResult> {
    let vox = dot_vox::load_bytes(data)
        .map_err(|e| anyhow::anyhow!("Failed to parse .vox: {e}"))?;

    let model = vox.models.first().context("No models in .vox file")?;
    let size = &model.size;

    // MagicaVoxel uses Z-up; we use Y-up. Swap Y↔Z.
    let mut grid = VoxelGrid::new(size.x, size.z, size.y);

    for v in &model.voxels {
        // .vox palette indices are 1-based (0 = empty in MagicaVoxel)
        grid.set(v.x as u32, v.z as u32, v.y as u32, v.i);
    }

    // Build palette from .vox colors
    let mut palette = MaterialPalette::new();
    let default_material = |r: u8, g: u8, b: u8| -> MaterialType {
        // Simple heuristic: dark = stone, green = foliage, etc.
        MaterialType::Concrete // default
    };
    let material_fn = color_to_material.unwrap_or(&default_material);

    for (i, color) in vox.palette.iter().enumerate() {
        if i == 0 { continue; } // index 0 is air
        if i >= 256 { break; }
        let r = color.r;
        let g = color.g;
        let b = color.b;
        palette.set(i as u8, PaletteEntry {
            r, g, b,
            roughness: 128,
            emissive: 0,
            material_type: material_fn(r, g, b),
        });
    }

    Ok(VoxImportResult { grid, palette })
}

/// Load a .vox file from a filesystem path.
pub fn load_vox_file(
    path: &std::path::Path,
    color_to_material: Option<&dyn Fn(u8, u8, u8) -> MaterialType>,
) -> Result<VoxImportResult> {
    let data = std::fs::read(path).context("Failed to read .vox file")?;
    load_vox(&data, color_to_material)
}
