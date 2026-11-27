# Information about the HPC server

I am working on a remote HPC server called MareNostrum5.

To access the server I use `ssh mn5`.
For data transfer tasks I need to use another node: `mn5_data_transfer1`.
Nodes don't have internet access.

The project code is `etur166` (use it for `--account` and for `/gpfs/projects/...` and `/gpfs/scratch/...` paths).

## Paths
TODO: where this project's datasets are stored on MN5 (e.g. DGM4plus under `/gpfs/projects/etur166/boga222803/datasets/`), their layout and label format.

Datasets must always be extracted to `$TMPDIR` inside the job, never next to the source code:
```bash
unzip -q /gpfs/projects/etur166/boga222803/datasets/<dataset>.zip -d "$TMPDIR/<dataset>"
```

TODO: confirm the clone path on MN5 (suggested: `/gpfs/home/boga/boga222803/c586-tp-code`, which would put this project at `.../c586-tp-code/HAMMER-sgs`).
This is storage only: never execute jobs there or use it as a job's working directory (see "Where jobs run").

### Getting the source code onto MN5
MN5 can't reach GitHub, so the repo is cloned locally into an `sshfs` mount of the MN5 home directory (git runs locally, the files land on MN5):
```bash
mkdir -p ~/mn5_mount/home
sshfs -o workaround=rename boga222803@transfer1.bsc.es:/gpfs/home/boga/boga222803/ ~/mn5_mount/home
git clone git@github.com:kutayeroglu/c586-tp-code.git ~/mn5_mount/home/c586-tp-code   # first time
git -C ~/mn5_mount/home/c586-tp-code pull                                               # later updates
fusermount -u ~/mn5_mount/home                                                          # unmount when done
```
Only the source goes through git; large files are uploaded separately (TODO: script for uploading weights/models).
Server rules require all large files (weights, images, datasets) to stay in `/gpfs/projects/etur166/boga222803`, never in the source code directory.

Large files in `/gpfs/projects/etur166/boga222803`:
```
TODO: list this project's container image, model weights and datasets
```

## Where jobs run
The server maintainers recommend: "Executing your jobs in your group's /gpfs/projects or /gpfs/scratch directories is recommended."
So jobs run in `/gpfs/scratch/etur166/boga222803/`, never in `/gpfs/home`: both the job's working directory (`--chdir`, where the SLURM `.out`/`.err` logs go) and the folder the code runs in are on scratch.
`/gpfs/scratch/etur166/` is shared with the group; keep everything under `boga222803/`. Scripts create the folders they need with `mkdir -p`; don't leave folder creation as a manual step.

Each job copies only the code it needs from home into its own run folder on scratch and runs there:
```bash
RUN_DIR=/gpfs/scratch/etur166/boga222803/<project>/$SLURM_JOB_ID   # TODO: pick the project folder name
mkdir -p "$RUN_DIR"
rsync -a "$SRC_DIR/<files the job uses>" "$RUN_DIR/"                # SRC_DIR = the project source folder on home
cd "$RUN_DIR"
```
After the job, copy the results to `/gpfs/projects/etur166/boga222803/<results folder>/$SLURM_JOB_ID/` so they are kept, then delete the run folder. A failed job keeps its run folder for debugging.

TODO: how the code finds its weights/models (paths relative to the run folder, env vars, symlinks to `/gpfs/projects/...`).
If the job symlinks to files on `/gpfs`, `/gpfs` must be visible inside the container (`--bind /gpfs`, see below).

## Running the container
MN5 has SingularityCE, not Apptainer. Load it in the job:
```bash
module purge
module load singularity/4.1.5
```
`singularity exec --nv` works on compute nodes but fails on the login node (session directory permission error), so test containers inside a job.

TODO: how this project's `.sif` is built and where it lives. The image is built locally (MN5 has no internet), e.g. converted from the GHCR image built by `.github/workflows/hammer-sgs-cbuild.yml`, then uploaded to `/gpfs/projects/etur166/boga222803/`.
For small additions, prefer layering on top of an existing verified image (`Bootstrap: localimage`) over a full rebuild: a full rebuild re-resolves every unpinned package and could change results.

Run the pipeline from the run folder, pointing the datasets at `$TMPDIR`:
```bash
singularity exec --nv --bind /gpfs \
  --env <DATASET_VAR>="$TMPDIR/<dataset>" \
  /gpfs/projects/etur166/boga222803/<image>.sif <command>   # TODO
```

## GPU selection
Don't hardcode GPU IDs. SLURM sets `CUDA_VISIBLE_DEVICES` for the job; use it.

# Preparing SLURM scripts
TODO: path of this project's MN5 SLURM script; start from it for new jobs.

Recommended pattern: the script is submitted by running it (not with `sbatch`) from anywhere on a login node. It creates the scratch folders, then submits itself with `--chdir` on scratch and passes extra args to `sbatch`:
```bash
bash <path to script>                    # debug run (acc_debug)
bash <path to script> --qos=acc_ehpc     # full run
```
`--chdir` must exist before the job starts, so it can't be created inside the job.

Logs: `/gpfs/scratch/etur166/boga222803/<project>/logs/<jobid>_<name>.{out,err}` (TODO).
Results: `/gpfs/projects/etur166/boga222803/<results folder>/<jobid>/` (TODO).

## QoS
- Debug and test jobs: `--qos=acc_debug`.
- Full runs: `--qos=acc_ehpc`.

TODO: an example MN5 SLURM script from another project to copy from (it lived at the root of the other repo as `mn5_example_slurm_script.sh`; it isn't in this repo).
Reuse its `--account=etur166`, `$TMPDIR` staging and module loading. Don't copy the parts specific to that project, or a `--chdir=.`, which would run the job wherever it was submitted from.
