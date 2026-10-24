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
    target_value=1.0,   # bandgap in eV
)

print(df[["filename", "wmse", "pred_value"]])
df.to_csv("generated_test.csv", index=False)
