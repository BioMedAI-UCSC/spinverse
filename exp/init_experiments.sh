SWEEP_DIR="spinverse_ablation"

python create_sweep_exp_configs.py \
    --baseline $SWEEP_DIR/baseline_config.yaml \
    --mods_yaml $SWEEP_DIR/config_mods.yaml \
    --mesh_dir $SWEEP_DIR/meshes \
    --mesh_pattern "*.pth" \
    --experiments_dir $SWEEP_DIR/experiments \
    --experiments_pattern "*.yaml" \
    --order mesh_mod_experiment \
    --out_dir $SWEEP_DIR/optim \
    --prefix exp_ \
    --pad 4 \
    --start_index 0 \
    --use_absolute_experiment_paths