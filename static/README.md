Small fixed tables used by the daily forecast job, so it does not need the
3 GB IGN BD TOPO or the Corine download. Rebuild them with
`python -m firepredict.communes` and `python -m firepredict.landscape`, then
copy them here from `data/processed/83/`.

- `communes.parquet`: one row per commune (land cover, nearest stations, training fire rate)
- `landscape.parquet`: buildings and roads in or near wildland, from IGN BD TOPO
