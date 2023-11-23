# Revision of SRGAN
## Updates:
- implementation in pytorch Lightning, including versioning, logging, experiment tracking
- new dataloaders including stratification (by landcover), normalization
## Experiments:
- 2 trainings on CV data - Results: ✅ 
- training on
1. Interpolated SPOT6 data - Results: ✅
2. interpolated SPOT6 data, with blur kernel added - Results: ✅
3. interpolated SPOT6 data, with blur kernel added, histogram matched so Sen2 spectral range - Results: to be seen

## Tracking
tracking via this WandB project: https://wandb.ai/simon-donike/2023_SRGAN

## ToDo
- implement spatial matching (probably best via superglue algo or grid search)
- implement MISR
- implement proper validation procedure to determine metrics for different runs on real Sen2 data
