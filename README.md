# Docker_TFS

Use Tensorflow Serving via docker to accelarate the prediction of a tf model to meet the demand for high concurrency.

1st,
  Transfer the Keras h5 model to a pb model.
2nd,
  Deploy service on Tensorflow Serving via docker.
3rd,
  Test the interface.
  
Docker makes it easy to build, ship, and run distributed applications and gets rid of many troubles such as setting up environments.
Tensorflow Serving GPU version highly optimizes the performance of a tf model.
