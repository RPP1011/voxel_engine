pub struct NpcState {
    pub position: [f64; 3],
}

pub struct DestructionOption {
    pub target_position: [f64; 3],
    pub collapsible_mass: f64,
    pub strength: f64,
    pub time_to_destroy: f64,
    pub risk_to_npc: f64,
}

/// Score = (collapsible_mass * proximity * damage_potential) / (time + risk + cost)
pub fn score_destruction(npc: &NpcState, option: &DestructionOption) -> f64 {
    let dx = option.target_position[0] - npc.position[0];
    let dy = option.target_position[1] - npc.position[1];
    let dz = option.target_position[2] - npc.position[2];
    let distance = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);

    let proximity = 1.0 / distance;
    let damage = option.collapsible_mass;
    let cost = option.strength.max(1.0);

    let numerator = damage * proximity;
    let denominator = option.time_to_destroy + option.risk_to_npc * 10.0 + cost * 0.001;

    numerator / denominator.max(0.001)
}
