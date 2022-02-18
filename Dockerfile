FROM jupyter/datascience-notebook

RUN conda install -c conda-forge -y --quiet \
	sha=0.40.0 \
	lightgbm=3.3.2 \
	catboost=1.0.4 \
	eli5=0.11.0