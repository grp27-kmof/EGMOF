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
    num_samples=10,
    target_value=1.0,
    build_cif=True,
    cif_dir="./test_output/cifs",
    new_bb_dir="./test_output/new_bbs",
)

print(df[["filename", "wmse", "pred_value", "cif"]])
df.to_csv("test_output/generated_with_cifs.csv", index=False)
