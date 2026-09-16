# Real footage runs

The before-and-after measurement in `docs/evaluation.md` §10, as code and as data.

| File | What it is |
|---|---|
| `measure.py` | runs one named config over a real clip through `keepout.service.config_from_params`, the live service's config path, and writes a result plus the evidence frames |
| `run_all.sh` | runs them in sequence, waiting while the machine has under 4 GB free |
| `person_size.py` | detection recall against person height, one pass versus tiled (§10.5) |
| `person-size.json` | its output |
| `results.json` | every run, before and after: frames usable, incidents with their subject boxes, and how each evidence frame was redacted |

## Sources

Each licence was read on the file's own Commons page.

| Clip | Author | Licence |
|---|---|---|
| [Malta - Mdina - Lorenzo Calleja ditch - Il-Foss tal-Imdina (construction) 01](https://commons.wikimedia.org/wiki/File:Malta_-_Mdina_-_Lorenzo_Calleja_ditch_-_Il-Foss_tal-Imdina_(construction)_01_(1)_ies.webm), shot 4 | Frank Vincentz, 2013 | CC BY-SA 3.0 |
| [Prefabricated house construction](https://commons.wikimedia.org/wiki/File:Prefabricated_house_construction.ogv), time-lapse | H. Raab, 2006 | CC BY-SA 3.0 |
| [Amazon warehouse BHX4 loading docks 2](https://commons.wikimedia.org/wiki/File:Amazon_warehouse_BHX4_loading_docks_2.webm) | domdomegg, 2019 | CC BY 4.0 |
| [vtest.avi](https://github.com/opencv/opencv/blob/5.x/samples/data/vtest.avi) (the courtyard clip) | OpenCV project | Apache-2.0 |

The clips are **not** in this repository. They show identifiable workers, and their
licences (CC BY-SA 3.0, CC BY 4.0) ask for attribution that belongs next to the
footage. To reproduce, fetch them from Wikimedia Commons and lay them out like this:

```
$KEEPOUT_REAL_FOOTAGE/clips/malta-mdina-shot4-74.35s-104.6s.mp4   # File:Malta_-_Mdina_-_Lorenzo_Calleja_ditch_-_Il-Foss_tal-Imdina_(construction)_01_(1)_ies.webm, 74.35-104.6 s
$KEEPOUT_REAL_FOOTAGE/clips/prefab-house-shot1-0.6s-47.5s.mp4     # File:Prefabricated_house_construction.ogv, 0.6-47.5 s
$KEEPOUT_REAL_FOOTAGE/clips/courtyard-vtest-40s.mp4               # OpenCV samples/data/vtest.avi, first 40 s
$KEEPOUT_REAL_FOOTAGE/source/Prefabricated_house_construction.ogv
$KEEPOUT_REAL_FOOTAGE/source/Amazon_warehouse_BHX4_loading_docks_2.webm
```

```bash
KEEPOUT_REAL_FOOTAGE=/path/to/footage eval/real_footage/run_all.sh /tmp/after
```

"Before" is commit `ea5ea95` of the monorepo, the parent of this work, with the
reference taken from the first frame, as the service did then
(`--legacy-reference`). Evidence frames are not published here. They show real
workers, and the point of the exercise was to check that their faces are blurred,
not to distribute them.
