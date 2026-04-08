use voxel_engine::voxel::material::{MaterialPalette, MaterialType, PaletteEntry};

#[test]
fn set_and_get_palette_entry() {
    let mut palette = MaterialPalette::new();
    palette.set(
        42,
        PaletteEntry {
            r: 180,
            g: 120,
            b: 60,
            roughness: 200,
            emissive: 0,
            material_type: MaterialType::Wood,
        },
    );

    let entry = palette.get(42);
    assert_eq!(entry.r, 180);
    assert_eq!(entry.g, 120);
    assert_eq!(entry.b, 60);
    assert_eq!(entry.material_type, MaterialType::Wood);
}

#[test]
fn entry_zero_is_air() {
    let palette = MaterialPalette::new();
    let air = palette.get(0);
    assert_eq!(air.r, 0);
    assert_eq!(air.g, 0);
    assert_eq!(air.b, 0);
    assert_eq!(air.roughness, 0);
    assert_eq!(air.emissive, 0);
    assert_eq!(air.material_type, MaterialType::Air);
}

#[test]
fn material_density_lookup() {
    assert_eq!(MaterialType::Wood.density_kg_m3(), 500.0);
    assert_eq!(MaterialType::Metal.density_kg_m3(), 7800.0);
    assert_eq!(MaterialType::Concrete.density_kg_m3(), 2400.0);
    assert_eq!(MaterialType::Glass.density_kg_m3(), 2500.0);
    assert_eq!(MaterialType::Brick.density_kg_m3(), 1900.0);
    assert_eq!(MaterialType::Stone.density_kg_m3(), 2600.0);
    assert_eq!(MaterialType::Dirt.density_kg_m3(), 1500.0);
    assert_eq!(MaterialType::Sand.density_kg_m3(), 1600.0);
    assert_eq!(MaterialType::Ice.density_kg_m3(), 917.0);
}

#[test]
fn material_strength_lookup() {
    // Metal should be stronger than wood
    assert!(MaterialType::Metal.strength_pa() > MaterialType::Wood.strength_pa());
    // Concrete stronger than glass in compression
    assert!(MaterialType::Concrete.strength_pa() > MaterialType::Glass.strength_pa());
    // Air has no strength
    assert_eq!(MaterialType::Air.strength_pa(), 0.0);
}

#[test]
fn all_thirteen_material_types_exist() {
    let types = [
        MaterialType::Air,
        MaterialType::Wood,
        MaterialType::Metal,
        MaterialType::Glass,
        MaterialType::Concrete,
        MaterialType::Brick,
        MaterialType::Stone,
        MaterialType::Plite,
        MaterialType::Dirt,
        MaterialType::Sand,
        MaterialType::Plastic,
        MaterialType::Foliage,
        MaterialType::Rubber,
        MaterialType::Ice,
    ];
    // 14 total including Air
    assert_eq!(types.len(), 14);
}
