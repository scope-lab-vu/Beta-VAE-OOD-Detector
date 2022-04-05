# LEC training and validation
These scripts are used to train an end-to-end learning DNN to steer the autonomous vehicle. We use NVIDIA's DAVE II DNN model. The CSV files generated in the [data generation](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/tree/main/dave-II-dnn-training) step has the steering values required for training the DNN. Kindly adjust the CSV file and the column numbers while loading them in the training script. The scripts below can be used to train and test the DNN. The performance calculator script can be used to evaluate the trained model. 
```
python3 train.py                       --- train the DNN using the data generated.

python3 test.py                        --- test the DNN's prediction.

python3 performance-calculator.py      --- Measure the performance of the trained model using mean square error.

```

**Note**: Accordingly, split the CSV file generated in the [data generation](https://github.com/scope-lab-vu/Beta-VAE-OOD-Detector/tree/main/dave-II-dnn-training) step to create training and testing data files. 
Dear Editor,

