# CS236-HW2

Programming assignment.

Feel free to explore all of the files. The only ones you will need to modify are

1. `codebase/utils.py`
1. `codebase/models/vae.py`
1. `codebase/models/gmvae.py`
1. `codebase/models/ssvae.py`
1. `codebase/models/fsvae.py` (bonus)

Do not modify the other files. All default hyperparameters have been prepared
for you, so please do not change them. If you choose to do the programming
assignment from scratch, please copy the hyperparameters faithfully so that your
results are comparable to what we'd expect.

The models can take a while to run on CPU, so please prepare accordingly. On a
2018 Macbook Pro, it takes ~7 minutes each to run `vae.py` and `gmvae.py`. It
takes ~50 minutes to run `ssvae.py`. It will take a very long time to run
`fsvae.py` to completion; we recommend stopping training the moment your samples
look somewhat decent (maybe around `50000` iterations). Note that we periodically
save models for you.

You are also free to create new files or jupyter notebooks to assist in
answering any of the written assignment questions. For image generation, you
will find the following functions helpful:

1. `codebase.utils.load_model_by_name` (for loading a model. See example usage in `run_vae.py`)
1. The sampling functionalities in `vae.py`/`gmvae.py`/`ssvae.py`/`fsvae.py`
1. `numpy.swapaxes` and/or `torch.permute` (for tiling images when represented as numpy arrays)
1. `matplotlib.pyplot.imshow` (for generating an image from numpy array)

When you introduce new files, do not include them in the codebase.

The following is a checklist of various functions you need to implement in the
codebase, in chronological order:

1. `sample_gaussian` in `utils.py`
1. `negative_elbo_bound` in `vae.py`
1. `log_normal` in `utils.py`
1. `log_normal_mixture` in `utils.py`
1. `negative_elbo_bound` in `gmvae.py`
1. `negative_iwae_bound` in `vae.py`
1. `negative_iwae_bound` in `gmvae.py`
1. `negative_elbo_bound` in `ssvae.py`
1. `negative_elbo_bound` in `fsvae.py` (bonus)

Once you've completed the assignment, run the `make_submission.sh` script and upload `hw2.zip`.

---

### Dependencies

This code was built and tested using the following libraries

```
numpy==1.17.0
torchvision==0.4.1a0+d94043a
torch==1.3.0
tqdm==4.34.0
```

---

### Submission

Use `make_submission.sh` to zip up the appropriate files. This script simply zips the following files to `hw2.zip`:
```
codebase/utils.py 
codebase/models/vae.py 
codebase/models/gmvae.py
codebase/models/ssvae.py 
codebase/models/fsvae.py
```

Note that `hw2.zip` preserves the directories. If you're creating `hw2.zip` manually, you may wish to verify your zip file by running
```
python verify_submission.py
```

To do the inspection manually, simply do
```
unzip hw2.zip -d tmp_dir
```
And then check that `tmp_dir` has the following file structure:
```
tmp_dir/
└── codebase
    ├── models
    │   ├── fsvae.py
    │   ├── gmvae.py
    │   ├── ssvae.py
    │   └── vae.py
    └── utils.py
```
