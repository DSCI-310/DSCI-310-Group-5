FROM jupyter/datascience-notebook:latest

RUN conda install -y --quiet -c conda-forge \
	pandas=1.3.4 \
	numpy=1.20.3 \
	matplotlib=3.4.3 \
	seaborn=0.11.2 \
	scikit-learn=0.24.2 \
	pytest=6.2.4 \
	jupyter-book=0.12.2
