# Pulling back symmetric Riemannian geometry for data analysis

    [1] W. Diepeveen, A. Bertozzi.  
    Pulling back symmetric Riemannian geometry for data analysis
    arXiv preprint arXiv:xxxx.xxxxx. YYYY MM DD.

Setup
-----

The recommended (and tested) setup is based on MacOS 13.4.1 running Python 3.8. Install the following dependencies with anaconda:

    # Create conda environment
    conda create --name pubariegeo1 python=3.8
    conda activate pubariegeo1

    # Clone source code and install
    git clone https://github.com/wdiepeveen/Pulling-back-symmetric-Riemannian-geometry-for-data-analysis.git
    cd "Pulling-back-symmetric-Riemannian-geometry-for-data-analysis"
    pip install -r requirements.txt


Reproducing the experiments in [1]
----------------------------------

To produce the results in section 6.1 of [1]:
* Run the jupyter notebook `experiment/curvature_effects/evaluate_curvature_effects.ipynb`

To produce the results in section 6.2 of [1]:
* Run the jupyter notebook `experiment/diffeomorphism_effects/evaluate_diffeomorphism_effects.ipynb`
