# LEC training and validation
These scripts are used to train an end-to-end learning DNN to steer the autonomous vehicle. We use the NVIDIA's DAVE II DNN model. The CSV files generated in the [data generation]() step has the steering values required for training the DNN. Kindly adjust the CSV file and the column numbers while loading them in the train script. 

```
python3 train.py                       --- train the DNN using the data generated.

python3 test.py                        --- test the DNN's prediction.

python3 performance-calculator.py      --- Measure the performance of the trained model using mean square error.
```
