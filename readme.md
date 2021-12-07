# The code for DH-VQG.
Please don't distribute since it's under review.
We will polish the code upon paper acceptance.
============


Dependencies
------------
We use pytorch with anaconda (python3.7).
Please refer to the following environments.
* anaconda environment: conda.yaml
* pip environment: pip.txt

How to run
----------

Run with following:

The evaluation code is missing since its' too large.
We will solve this problem upon paper acceptance.
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --rank 0 --world_size 2 --config config/final_vqa_tune_tune.yaml --dist_url tcp://localhost:13379 &
CUDA_VISIBLE_DEVICES=2 python main.py --rank 1 --world_size 2 --config config/final_vqa_tune_tune.yaml --dist_url tcp://localhost:13379
```

