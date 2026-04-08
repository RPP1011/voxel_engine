use voxel_engine::ai::tactical::{DestructionOption, score_destruction, NpcState};

#[test]
fn near_weak_pillar_scores_higher_than_far_strong() {
    let npc = NpcState { position: [0.0, 0.0, 0.0] };

    let near_weak = DestructionOption {
        target_position: [2.0, 0.0, 0.0],
        collapsible_mass: 500.0,
        strength: 100.0,    // weak
        time_to_destroy: 1.0,
        risk_to_npc: 0.1,
    };

    let far_strong = DestructionOption {
        target_position: [20.0, 0.0, 0.0],
        collapsible_mass: 500.0,
        strength: 10000.0,   // strong
        time_to_destroy: 5.0,
        risk_to_npc: 0.1,
    };

    let score_near = score_destruction(&npc, &near_weak);
    let score_far = score_destruction(&npc, &far_strong);

    assert!(
        score_near > score_far,
        "Near weak pillar ({score_near:.2}) should score higher than far strong ({score_far:.2})"
    );
}

#[test]
fn collapse_endangering_npc_reduces_score() {
    let npc = NpcState { position: [0.0, 0.0, 0.0] };

    let safe = DestructionOption {
        target_position: [10.0, 0.0, 0.0],
        collapsible_mass: 500.0,
        strength: 100.0,
        time_to_destroy: 1.0,
        risk_to_npc: 0.1,   // low risk
    };

    let dangerous = DestructionOption {
        target_position: [10.0, 0.0, 0.0],
        collapsible_mass: 500.0,
        strength: 100.0,
        time_to_destroy: 1.0,
        risk_to_npc: 0.9,   // high risk
    };

    let score_safe = score_destruction(&npc, &safe);
    let score_dangerous = score_destruction(&npc, &dangerous);

    assert!(
        score_safe > score_dangerous,
        "Safe option ({score_safe:.2}) should score higher than dangerous ({score_dangerous:.2})"
    );
}

#[test]
fn npc_picks_best_option() {
    let npc = NpcState { position: [0.0, 0.0, 0.0] };

    let options = vec![
        DestructionOption {
            target_position: [5.0, 0.0, 0.0],
            collapsible_mass: 1000.0,
            strength: 50.0,
            time_to_destroy: 1.0,
            risk_to_npc: 0.1,
        },
        DestructionOption {
            target_position: [30.0, 0.0, 0.0],
            collapsible_mass: 100.0,
            strength: 5000.0,
            time_to_destroy: 10.0,
            risk_to_npc: 0.5,
        },
        DestructionOption {
            target_position: [3.0, 0.0, 0.0],
            collapsible_mass: 2000.0,
            strength: 200.0,
            time_to_destroy: 2.0,
            risk_to_npc: 0.2,
        },
    ];

    let scores: Vec<f64> = options.iter().map(|o| score_destruction(&npc, o)).collect();
    let best = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    // The best option should be either 0 or 2 (both are close, high mass, reasonable risk)
    assert!(
        best == 0 || best == 2,
        "Best option should be one of the good nearby targets, got index {best}"
    );
}
