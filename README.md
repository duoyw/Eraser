# Eraser: Eliminating Performance Regression on Learned Query Optimizer

This repository contains the source code for the paper titled "Eraser: Eliminating Performance Regression on Learned
Query Optimizer." Eraser is a plugin deployed on top of learned query optimizers like Lero, HyperQo, and PerfGuard. Its
purpose is to assist these algorithms in eliminating performance regression and achieving comparable performance to
PostgreSQL when these algorithms perform worse, while having minimal impact when these algorithms perform well.

Here, we provide a comprehensive and detailed description of two different approaches to reproduce the Eraser.
The first approach allows for a quick reproduction of Eraser. We provide all the trained models and raw data used in
our experiments. This allows anyone to reproduce the most important experiment results using a few simple commands.
The second approach is a comprehensive guide that outlines the steps to reproduce Eraser from scratch. This includes
creating the database, importing data, training Lero, HyperQo, and PerfGuard models, and collecting data for Eraser.

# Reproducing the Experiment Result Quickly

We have provided the necessary resources to quickly reproduce Eraser, including all trained models of learned query
optimizers and the raw data used in our experiments. By using a few simple commands, anyone can reproduce the
experimental results.

**When reproducing the results, we will build an Eraser model from scratch based on the prediction accuracy of ML models
of learned query optimizers. We will then test the performance of Eraser using the collected candidate
plans from the plan exploration algorithms of these learned query optimizers.**

All experiments are finished on Linux, so you should also run codes on a Linux system.

### Download Data

First, you need to download the source code of Eraser by executing the following command:

```
git clone git@github.com:duoyw/Eraser.git
```

Then, you need to download all data including trained models, candidate plans, SQLs, etc. Due to the space limitation of
Github, 
we split all data into four compressed sub-files.
You can download these files using the following commands after switching to the project root 

```
# move to the root of the project
cd ./Eraser
# download all compressed files from the GitHub release
wget https://github.com/duoyw/Eraser/releases/download/v1.1/Eraser_Data_aa
wget https://github.com/duoyw/Eraser/releases/download/v1.2/Eraser_Data_ab
wget https://github.com/duoyw/Eraser/releases/download/v1.3/Eraser_Data_ac
wget https://github.com/duoyw/Eraser/releases/download/v1.4/Eraser_Data_ad
```
After downloading all compressed data, you need to merge all sub-files into a complete file using the following commands.

```
# on the root of the project
rm -rf Data/
cat Eraser_Data* > Data.tar
# The command may need to be executed twice
tar -xvf Data.tar
tar -xvf Data.tar 
```

The complete project is structured as follows:

```
- Eraser: the root of our Eraser project
    - Data: storing all trained models and raw training and test data.
    - Hyperqo: the source code of HyperQO baseline
    - PerfGuard: the source code of PerfGuard baseline
    - RegressionFramework: the source code of our Eraser
    - Spark: the source code of Lero on Spark
    - test_script: the source code of Lero baseline baseline
    - hyperqo_test: it provides a few commands to reproduce the experiment results of HyperQO quickly.
    - lero_test: it provides a few commands to reproduce the experiment results of Lero quickly.
    - perfguard_test: it provides a few commands to reproduce the experiment results of PerfGuard quickly.
    - Others: some auxiliary files or folders, such as README
```

### Configurate Environment

To run Eraser, you need to set up the corresponding Python environment.

First, download Python 3.8 from [https://www.python.org/downloads/]() or download Conda
from [https://docs.conda.io/en/latest/]() to create an independent Python 3.8 environment.

Next, switch to the corresponding 'pip' and install the required libraries used by Eraser using the following command:

```
# switching to the root of project
pip install -r requirements.txt
```

Once you have completed these steps, you can quickly reproduce the experimental results using the provided data.

### Reproduce PostgreSQL Results

To minimize the reproduction overhead, all codes are executed on CPUs. Each experiment's results should be collected
within a few minutes, which is still considered fast.

To reproduce the experiment results shown in Figure 5 of the paper, you can use the following commands.

```
# cd to the root of the project

# lero 
python -m unittest lero_test.LeroTest.test_static_all_job
python -m unittest lero_test.LeroTest.test_static_all_stats
python -m unittest lero_test.LeroTest.test_static_all_tpch

# PerfGuard 
python -m unittest perfguard_test.PerfguardTest.test_static_all_job
python -m unittest perfguard_test.PerfguardTest.test_static_all_stats
python -m unittest perfguard_test.PerfguardTest.test_static_all_tpch

# HyperQO, it is needed to set config manually, please see the description below.
python -m unittest hyperqo_test.HyperqoTest.test_static_all_job
python -m unittest hyperqo_test.HyperqoTest.test_static_all_stats
python -m unittest hyperqo_test.HyperqoTest.test_static_all_tpch
```

The above code utilizes the provided ML models from different learned query optimizers, as well as the training 
data, to build an Eraser system from scratch.
Then, we test the performance of the Eraser on each test query.
Each algorithm will be tested in each
benchmark four times using models trained with 25%, 50%, 75%, and 100% of the training data.

All results will be saved in the "RegressionFramework/fig" directory. You should obtain consistent results with
the paper for the Lero and PerfGuard algorithms. However, due to the high randomness of the HyperQO algorithm, the
results may differ from those in the paper. Nevertheless, Eraser typically eliminates most of the performance
regression.

For example, the figure "RegressionFramework/fig/lero_job_test2" shows the performance of the Lero algorithm
trained with 50% of the data on the IMDB (JOB) benchmark.

![lero_job_test2.png](https://github.com/duoyw/Eraser/blob/main/readmeFig/lero_job_test2.png)

The Figure shows the performance of Lero without Eraser, Lero with Eraser, and PostgreSQL.

The suffixes "1", "2", "3", and "4" of these files indicate that the model was trained on 25%, 50%, 75%, and 100% of the
data, respectively.
Similarly, "lero_stats_test3" indicates the performance of the Lero algorithm trained with 75% of the data on the STATS
benchmark.

For HyperQO, due to limitations in the source code provided by the authors, we were unable to show all the results using
a few simple commands. You need to change the variable "self.database=imdb/stats/tpch" in the "Hyperqo/ImportantConfig.py"
file based on the selected benchmark. For example, when you prepare to execute the "test_static_all_job" command, you
must set "self.database" to "imdb".

Reproducing the experiment results shown in Figure 7 of the paper is easy. When executing the code to collect the result
of Figure 5, Eraser will save the corresponding result of Figure 7 in the "RegressionFramework/fig/" directory.
For example, the figure "lero_job_regression_per_query_2" represents the experiment result of the Lero algorithm trained
with 50% of the data in the IMDB (job) benchmark. Please note that due to slight randomness, the experiment result may
slightly differ from the paper, but Eraser consistently eliminates most of the performance regression.

![lero_job_regression_per_query_2.png](https://github.com/duoyw/Eraser/blob/main/readmeFig/lero_job_regression_per_query_2.png)

To reproduce the experiment results shown in Figure 8 of the paper, you can use the following commands. Please note that
this may consume a significant amount of time and memory due to the for-loop process without releasing memory. However,
this can be optimized by clearing models that are no longer needed.

```
# lero 
python -m unittest lero_test.LeroTest.test_dynamic_job
python -m unittest lero_test.LeroTest.test_dynamic_tpch

# PerfGuard 
python -m unittest perfguard_test.PerfguardTest.test_dynamic_job
python -m unittest perfguard_test.PerfguardTest.test_dynamic_tpch

# HyperQO, it is needed to set config manually, please see the description below.
python -m unittest hyperqo_test.HyperqoTest.test_dynamic_job
python -m unittest hyperqo_test.HyperqoTest.test_dynamic_tpch
```

The above code will build Eraser from scratch based on the provided models and data. All results are also saved in the "
RegressionFramework/fig" directory. For example, the "dynamic_tpch_lero" file shows the dynamic performance of
the Lero algorithm in the TPC-H benchmark. Similarly, you should change the variable of HyperQO algorithms if you want
to execute it.

![dynamic_tpch_lero.png](https://github.com/duoyw/Eraser/blob/main/readmeFig/dynamic_tpch_lero.png)

### Reproducing the Spark Results

To reproduce the experiment results of the paper for Spark, we also provide trained Lero models and candidate plans for
a quick reproduction. It is important to note that we retrained the Lero models due to the loss of raw models. As a
result, there may be slight differences from the results presented in the paper. However, the Eraser is still effective in
eliminating most of the performance regression.

To reproduce Figure 15, you can use the following commands. First, navigate to the Spark root path. Assuming your path
is at the root of the project, execute the command:

```
cd ./Spark
```

Then, run the following commands to reproduce Figure 15(a):

```
python -m unittest spark_lero_test.SparkLeroTest.test_static_all_tpcds
```

Similarly, run the following commands to reproduce Figure 15(b):

```
# for online mode
python -m unittest spark_lero_test.SparkLeroTest.test_dynamic_tpcds
```

The above code will utilize the prepared Lero models, as well as the training and test data, to build an Eraser system
from scratch and test each query in the test set.
All results will be saved in the "RegressionFramework/fig" directory. For example, the file "spark_lero_test_3"
shows the performance of the Lero algorithm with and without Erase where the Lero model is trained on 75% of data.

![spark_lero_test_3.png](https://github.com/duoyw/Eraser/blob/main/readmeFig/spark_lero_test_3.png)

# Reproducing Results from Scratch

Here, we present a comprehensive guide on reproducing the experimental results from scratch, which includes the creation
of the database, data importing, training of Lero, HyperQo, and PerfGuard models, and data collection for Eraser, etc.

It is crucial to note that due to the inherent and diverse randomness in each sub-process, the experiment
results may differ slightly from those outlined in the paper. Nevertheless, Eraser can eliminate most
of the performance regression.

To begin, it is necessary to download PostgreSQL by executing the following commands:

```
wget https://ftp.postgresql.org/pub/source/v13.1/postgresql-13.1.tar.bz2
tar -xvf postgresql-13.1.tar.bz2
```

Next, you should construct the TPC-H, TPC-DS, IMDB, and STATS databases in PostgreSQL, referring to the guidelines
provided in the official documentation. We have included the relevant links below for your convenience:

```
TPCDS:https://www.tpc.org/tpcds/
TPC-H:https://www.tpc.org/tpch/
STATS:https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark
IMDB: https://github.com/gregrahn/join-order-benchmark
```

### Deploying Eraser to Learned Query Optimizers from Scratch

Eraser is a plugin deployed on top of diverse learned query optimizers.
Consequently, if you want to apply Eraser to a new learned query optimizer, you need to install the learned query
optimizer individually.
We introduce the deployment process in a general way such that anyone can apply Eraser to any learned query optimizers,
including Lero, HyperQO, PerfGuard, etc.

To facilitate this process and comply with copyright restrictions, we provide the source code links for all the learned
query optimizers employed by Eraser.

```
Lero: https://github.com/AlibabaIncubator/Lero-on-PostgreSQL
HyperQo：https://github.com/yxfish13/HyperQO
PerfGuard: https://github.com/WoodyBryant/Perfguard
Spark-Lero: https://github.com/Blondig/Lero-on-Spark 
```

Please download each learned query optimizer and configure the respective environments according to their requirements.
Subsequently, you should collect all candidate plans based on their plan exploration strategies and train ML models of
each algorithm, following their documentation.

After completing the data collection and model training, we discuss how to verify the performance of these learned query
optimizers after deploying Eraser. You should provide all the collected candidate plans and trained ML models in the "
Data" and "Data/Model" directories, respectively. The process is elaborated upon in detail below.

### Preparing Data

All training and test SQLs should be stored in the "Data" directory. For instance, let's analyze the IMDB
benchmark data:

```
job1.txt: 25% training SQLs for model training
job2.txt: 50% training SQLs for model training
job3.txt: 75% training SQLs for model training
job4.txt: 100% training SQLs for model training
```

Next, all candidate plans for these SQLs should be saved in files with the appropriate format. 
You need to convert the results of the plan exploration strategies of each algorithm into this format.
For the Lero algorithm and IMDB benchmark, the corresponding files and format are shown below, 

```
lero_job1.log.training: all candidate plans for the 25% training SQLs
lero_job2.log.training: all candidate plans for the 50% training SQLs
lero_job3.log.training: all candidate plans for the 75% training SQLs
lero_job4.log.training: all candidate plans for the 100% training SQLs
job_test: the test data for model testing
```

Each file should adhere to a specific format:

```
qid#####plan1#####plan2#####plan3...
```

In this format, "qid" represents the identification of a SQL query, followed by the concatenation of all candidate
plans, separated by "#####". In addition, the plan1 is generated by the database query optimizer.

You can apply Eraser to other learned query optimizers by preparing the corresponding data.

### Preparing Model

You should train the ML models of the Lero according to their documentation and save them in the "Data/Model"
directory. This ensures that Eraser can access these ML models.

Similarly, you can apply Eraser to other learned query optimizers (e.g., PerfGuard and HyperQo) by preparing the corresponding
trained model.

### Testing Eraser

After completing the aforementioned steps, you have collected ML models for the learned query optimizers, as well as
candidate plans for all queries.
You can now verify the performance of the Eraser using the same commands as those outlined in the "Reproducing the
Experiment Result Quickly" section.
It is important to note that due to the inherent and diverse randomness in each sub-process, the experiment results may
exhibit slight variations compared to the paper. Nevertheless, Eraser can eliminate most performance regressions.
















