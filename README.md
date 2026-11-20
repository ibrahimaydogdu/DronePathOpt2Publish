# Code and results: energy-makespan trade-offs in multi-trip drone delivery

Contents
- `03_Code_Execution/drone3d_d2/`: exact (energy, makespan) Pareto method (`fleet.py`), energy and time models (`energy_model.py`, `dtypes.py`), simulated annealing (`heur.py`), real-city layer (`realcity.py`), tests (`tests/`).
- `04_Results/`: run scripts (`run_*.py`, `summarize_*.py`), raw result tables (`*.csv`) and summaries (`*_SUMMARY_*.txt`) behind every number in the paper.
- `DATA_MANIFEST.csv`: file names, sizes and SHA-256 hashes of the raw inputs with their sources. Raw data are not redistributed (Amazon data: CC BY-NC 4.0; building layers: terms of the municipal portals; the Los Angeles layer is marked proprietary by its provider). Download them from the sources listed and check the hashes.

What runs from this repository
- Synthetic experiments and the validation gates: `python -I drone3d_d2/tests/test_core.py` (from `03_Code_Execution/`), `run_phaseA.py`, `run_inventory.py`, `run_mixed.py`. No external data are needed.
- The real-city experiments (`run_realdemand.py`, `run_baselines.py`, `run_realinventory.py`, `run_scalarization*.py`, `run_bridge_real.py`, `run_scale_real.py`, `run_dp_scaling.py`, `run_milp_check.py`, `run_n12_cap.py`, `check_trip_table_perm.py`) call the building-aware leg-cost modules `geometry.py`, `legs.py`, `around.py` and `energy.py` of a related unpublished study, which are available in its anonymised repository (https://anonymous.4open.science/r/DronePathOptPublish-EDB4/, directory `src/drone3d`). Copy these four files into a new directory `03_Code_Execution/core_d1/` and place the raw inputs of `DATA_MANIFEST.csv` under `Evidence/data_buildings/{chicago,los_angeles,austin}/` and `Evidence/data_realdemand/` (file names as in the manifest). The result tables of these experiments are also provided in `04_Results/`, so that every reported number can be recomputed from them with the `summarize_*.py` scripts. The real-city tests in `drone3d_d2/tests/test_realcity_layer.py` skip when the data or the modules are absent.

Setup: Python 3.10, `pip install -r requirements.txt`; run every script with `python -I`. Seeds 1001 to 1030 index the real instances (1001 to 1100 for the synthetic draws).
