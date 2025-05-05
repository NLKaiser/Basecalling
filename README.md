# Development of models and methods for basecalling Oxford Nanopore sequencing data

Download dataset:
https://cdn.oxfordnanoportal.com/software/analysis/bonito/example_data_dna_r10.4.1_v0.zip

Convert numpy dataset to tfrecords: Specify the path to the directory containing the training files in convert_data.py.  
```python convert_data.py```

Training:  
```python bc.py```

Visualise training statistics:  
```python plot_training.py```
