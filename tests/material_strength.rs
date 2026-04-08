use voxel_engine::voxel::material::MaterialType;

#[test]
fn each_material_has_density_and_strength() {
    let materials = [
        MaterialType::Wood, MaterialType::Metal, MaterialType::Glass,
        MaterialType::Concrete, MaterialType::Brick, MaterialType::Stone,
        MaterialType::Plite, MaterialType::Dirt, MaterialType::Sand,
        MaterialType::Plastic, MaterialType::Foliage, MaterialType::Rubber,
        MaterialType::Ice,
    ];

    for mat in materials {
        assert!(mat.density_kg_m3() > 0.0, "{mat:?} should have positive density");
        assert!(mat.strength_pa() > 0.0, "{mat:?} should have positive strength");
        assert!(mat.compressive_strength_pa() > 0.0, "{mat:?} compressive");
        assert!(mat.tensile_strength_pa() >= 0.0, "{mat:?} tensile");
    }
}

#[test]
fn metal_supports_longer_span_than_wood() {
    // Max cantilever length before self-weight exceeds tensile strength
    let metal_span = max_cantilever_span(MaterialType::Metal);
    let wood_span = max_cantilever_span(MaterialType::Wood);

    assert!(
        metal_span > wood_span,
        "Metal ({metal_span:.1}m) should support longer span than wood ({wood_span:.1}m)"
    );
}

#[test]
fn glass_is_brittle_low_tensile() {
    // Glass has low tensile strength relative to compressive
    let tensile = MaterialType::Glass.tensile_strength_pa();
    let compressive = MaterialType::Glass.compressive_strength_pa();

    assert!(
        tensile < compressive * 0.2,
        "Glass tensile ({tensile}) should be much less than compressive ({compressive})"
    );
}

/// Approximate max cantilever span (m) for a beam of 1m² cross-section
/// before self-weight bending stress exceeds tensile strength.
/// stress = (rho * g * L²) / (2 * h), where h = 1m
fn max_cantilever_span(mat: MaterialType) -> f64 {
    let rho = mat.density_kg_m3() as f64;
    let sigma = mat.tensile_strength_pa() as f64;
    let g = 9.81;
    let h = 1.0;
    // L = sqrt(2 * sigma * h / (rho * g))
    (2.0 * sigma * h / (rho * g)).sqrt()
}
