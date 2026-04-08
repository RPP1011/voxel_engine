#![cfg(feature = "app-harness")]

use voxel_engine::app::AppConfig;

#[test]
fn app_config_default_is_reasonable() {
    let config = AppConfig::default();
    assert_eq!(config.width, 1280);
    assert_eq!(config.height, 720);
    assert!(!config.window_title.is_empty());
    assert!((config.fixed_tick_rate - 10.0).abs() < 1e-3);
}
