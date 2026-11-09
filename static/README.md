Small fixed tables used by the daily forecast job, one folder per department,
so the job does not need the IGN BD TOPO or the Corine download. Rebuild them
with `python -m firepredict.communes` and `python -m firepredict.landscape`
(with FIREPREDICT_DEPARTMENT set), then copy them here from
`data/processed/<department>/`.

- `communes.parquet`: one row per commune (land cover, nearest stations, training fire rate)
- `landscape.parquet`: buildings and roads in or near wildland, from IGN BD TOPO
