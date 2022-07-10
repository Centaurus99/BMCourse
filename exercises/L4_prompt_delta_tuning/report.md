# L4 Prompt Delta Tuning Report

2020012709 Tong Haixuan

## 1. Your delta checkpoint size comparaed to the backbone model (OPT) size

|                          |Size      |
|-                         |-         |
|Delta checkpoint          |15.75 MB  |
|Backbone model (opt-2.7b) |5057.68 MB|

## 2. The largest size of the OPT model you can use with delta tuning (If you complete the code in a correct way, it will be opt-2.7b) and the run time GPU memory

||GPU memory|
|-|-|
|opt-2.7b|17726 / 24576 MB|

## 3. Use a smaller OPT model, compare the GPU memory with and without delta tuning

### Use opt-350m

|                    |GPU memory     |
|-                   |-              |
|with delta tuning   |5808 / 24576 MB|
|without delta tuning|9210 / 24576 MB|
