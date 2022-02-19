FROM jupyter/datascience-notebook

RUN conda install -y --quiet -c conda-forge \
	pandas=1.3.4 \
	numpy=1.20.3 \
	matplotlib=3.4.3 \
	seaborn=0.11.2 \
	scikit-learn=0.24.2
