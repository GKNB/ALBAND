# An Active Learning-Based Streaming Pipeline for Reduced Data Training of Structure Finding Models in Neutron Diffractometry

This repository contains the code for our recent work submitted to **BigData 2024**, titled "*An Active Learning-Based Streaming Pipeline for Reduced Data Training of Structure Finding Models in Neutron Diffractometry*". The code provides implementations for the two main contributions of the paper:

1. **Active Learning Algorithm**: A smarter data simulation approach based on the performance of a machine learning model, allowing the model to achieve the same accuracy with less data.

2. **Streaming Algorithm**: An improved workflow over the usual serial active learning workflow, utilizing a more resource-friendly streaming pipeline to enhance overall performance.

- The **Active Learning Algorithm** can be tested on any system.

- The **Streaming Algorithm** requires a job scheduling system and is supported only on **Polaris** and **Perlmutter**.

## Prerequisites

### Environment Setup

#### 1. Install Packages for Training and Active Learning Tasks

- **On Perlmutter:**

  ```bash
  module load pytorch/2.0.1

- **On Polaris:**

  ```bash
  module use /soft/modulefiles
  module load conda/2024-04-29

- **On Other systems**

We provide a packages list. A shifter container will be provided shortly to alleviate the need for building environment from scratch. Before that, user can use the following command to build env:

  ```bash
  conda env create -f environment.yml -n <env_name>

#### 2. Install GSAS-II for simulation tasks:

  
  ```bash
  g2="https://github.com/AdvancedPhotonSource/GSAS-II-buildtools/releases/download/v1.0.1/gsas2full-Latest-Linux-x86_64.sh"
  curl -L "$g2" > /tmp/g2.sh
  bash /tmp/g2.sh -b -p ~/g2full

Please note that, if you don't want to install into ~/g2full, please search and replace that string in executable directory and make corresponding changes.


Step 2: Running the baseline

In this step we try to perform baseline experiment and sweep over multiple dataset size. We also run with multiple random number seed for robustness. This will give us the black error bar in Figure 7,8. Note: We will not get exactly the same number, but results will be consistent.

The command for running this experiment will be:

cd workflow
qsub submit_baseline.sh


Before executing this command, modify the script according to the following directions:

Here the script will be executed multiple times, with the following parameter combination:

seed (line 12): 13010, 13110, 13210, 13310, 13410, 13510
num_sample (line 18): 40000, 80000, 120000, 160000, 200000, 240000
In total 6*6=36 experiments shall be done to generate six data point for baseline experiment with error bar

Before running real executable, need to setup env (See step 1). Modify line 9 accordingly.

Setting up the work_dir as the dir where this repo is. Change line 13 accordingly.

Also notice, for num_sample that exceeds 120000, task can not finish within one hour, need to use different queue (like preemptable) on Polaris


Step 3: Running the serial workflow

In this step we try to run the active learning serial workflow. This will output two important data: a). Accuracy performance of Active learning, and b). Running time performance of serial workflow. This will give us the black/red error band in Figure 7,8, data associate with serial workflow in Figure 9,10 and Table III,IV,V. Note: We will not get exactly the same number, but results will be consistent.

The command for running this experiment will be

cd workflow
qsub submit_serial.sh


Before executing this command, modify the script according to the following directions:

Use your own project name (line 7)

To get the red and blue error band in Figure 7,8, data associated with serial workflow in Figure 9,10 and Table III (i.e., Experiment E1), use the following parameter combination:
seed (line 12): 21000, 21100, 21200, 21300, 21400, 21500
num_sample (line 18): 4500
batch_size (line 23): 512

To get data associated with serial workflow in Table IV,V (i.e., Experiment E2), use the following parameter combination:
number_of_nodes (line 2): 1, 2, 4
queue (line 6): preemptable
walltime (line 4): 12:00:00
seed (line 12): 31000
num_sample (line 18): 72000
batch_size (line 23): 2048
