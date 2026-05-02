# SD3.0 + FLUX.1-dev q16 top60 7-concept sweep/reproduction

Server: siml-07, launched with nohup.

Main outputs for new sweep:
`/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd3_flux1_q16_7concept_20260430/outputs/{sd3,flux1}/{concept}/...`

Existing best-config reproduction outputs are under:
`/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/launch_0429_i2p_{sd3,flux1}/...`

Concepts: sexual, violence, self-harm, shocking, illegal_activity, harassment, hate.

Sweep cells:
- SD3: ss 15 / tau .45, ss 25 / tau .50
- FLUX.1-dev: ss .75 / tau .45, ss 1.5 / tau .50, ss 2.5 / tau .50
- fixed: hybrid, txt threshold .15, image threshold .10, q16 top60, seed 42 inherited by generator defaults.
