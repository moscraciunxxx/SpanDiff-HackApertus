.PHONY: run run-local predict predict-test

run:
	$(MAKE) -C track_2a run

run-local:
	$(MAKE) -C track_2a run-local

predict:
	$(MAKE) -C track_2a predict

predict-test:
	$(MAKE) -C track_2a predict-test
