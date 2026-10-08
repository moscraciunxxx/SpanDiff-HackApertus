.PHONY: run run-local predict

run:
	$(MAKE) -C track_2a run

run-local:
	$(MAKE) -C track_2a run-local

predict:
	$(MAKE) -C track_2a predict
