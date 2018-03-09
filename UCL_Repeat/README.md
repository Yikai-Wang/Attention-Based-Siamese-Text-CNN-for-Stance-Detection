# UCL's Work Repeat

In this file folder, we repeat the UCL's work.

Here's the repeat guide:

1. Build the environment it needs. Including

   ```
   Python 3.5
   Tensorflow 0.12.1
   Scipy
   Numpy
   ```

2. Run `data_handle.py`.  Note that you can decide the ratio between training set and test set in the line 2.

   You will create new training set and test set after implementing this Python file including:

   ```
   test_bodies.csv
   test_stances_labeled.csv
   train_bodies.csv
   train_stances.csv
   ```

3. Run python `pred.py.` Note that you need to run this Python file on a machine supporting GPU computing. You can modify the GPU number in line 22. Select `load` mode first. All the data set will load in the model

4. Run python `pred.py` again, and choose `train` this time. You will train the model. After the model is trained, it will generate a file named `predictions.csv` .

5. Run `eval.py` . You can get the relative score.

