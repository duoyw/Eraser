
We share the code on paper "Eraser: Eliminating Performance Regression on Learned Query Optimizer"

We show the example on how to deploy Eraser on Lero algorithm. Lero is a pairwise model that takes two plans as input and answer which plan is better.

You can know more on Lero by reading the paper "Lero: A Learning-to-Rank Query Optimizer", and its source code is shared in https://github.com/AlibabaIncubator/Lero-on-PostgreSQL.git

The training sqls and corresponding plans is saved in directory "data", and the models that is trained by Lero is saved in directory "model".

All Eraser code is saved in directory "RegressionFramework"

You can reproduce the experiment result by stating the unit test in file "lero_test", then it will leverage the data to build Eraser and protect the lero's model from performance regression

You can modify the related parameters in the file "RegressionFramework.config" for improving eraser's performance.
>>>>>>> master
