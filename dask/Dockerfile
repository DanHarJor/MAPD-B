FROM jpazzini/mapd-b:dask-scheduler
RUN pip install scikit-image
RUN pip install dask-image
RUN mkdir /weights/
CMD dask-worker tcp://dask-scheduler:8786 --local-directory /opt/workspace/dask-worker-space

