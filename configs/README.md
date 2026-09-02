# Configs

- `defaults/<DATASET>.yaml`: the untouched baseline settings per dataset. Never edited; copy them.
- `E<NNN>_<short_name>.yaml`: one file per experiment ID, committed, matching the row in
  the experiment log. Copy the default, change one thing, name it.

```
cp configs/defaults/SEGTHOR.yaml configs/E002_wce.yaml   # then edit
python -O main.py --config configs/E002_wce.yaml --dest results/E002 --gpu
python infer.py --config results/E002/config_dump.yaml --weights results/E002/bestweights.pt \
    --img_folder data/SEGTHOR/val/img --dest volumes/E002 \
    --scan_pattern "data/segthor_part1/train/{id_}/{id_}.nii.gz" --gpu
python metrics3d.py --pred_folder volumes/E002/nii \
    --gt_pattern "data/segthor_part1/train/{id_}/GT.nii.gz" \
    --scan_pattern "data/segthor_part1/train/{id_}/{id_}.nii.gz" \
    --class_names background esophagus heart trachea aorta --dest results/E002/metrics3d -p -1
python summarize.py results/E002
```

`results/` and `volumes/` are gitignored; the config and the experiment log row are the record.
