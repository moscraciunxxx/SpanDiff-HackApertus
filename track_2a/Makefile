.PHONY: run run-local predict check

# Docker path. The documented primary grade on this machine is run-local.
run:
	bash run_image.sh

check:
	python3 src/run.py --self-test

run-local: check
	bash grade_dev.sh

predict:
	python3 src/score_split.py --layer 15 --out track_2a/predictions/rerun --order-from track_2a/predictions/dev
