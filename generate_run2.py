from egmof import EGMOF

egmof = EGMOF(
    load_pretrained_modules=True,
    prop2desc_ckpt_path="checkpoints/prop2desc_ckpt/various_dataset/qmof_bandgap/best.ckpt",
    prop2desc_config_path="config/various_dataset/bandgap.yaml",
    skmodel_ckpt_dir="checkpoints/rf_ckpt/model_bandgap.pickle",
    accelerator="cpu",
    devices=1,
)

df = egmof.generate(
    num_samples=50,
    target_value=1.0,   # bandgap in eV
    build_cif=True,
    cif_dir="./run2/cifs",
    new_bb_dir="./run2/new_bbs",
)

df.to_csv("run2/generated.csv", index=False)
print(df[["filename", "wmse", "pred_value", "cif"]])
print(f"\nCIFs built: {df['cif'].notna().sum()} / {len(df)}")
