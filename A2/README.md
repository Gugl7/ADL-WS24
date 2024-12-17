# Assignment 2: Hacking

Since I changed the assignment a bit from the proposal, I will explain the changes here. Using the dataset I provided in assignment 1 was not as planned since it is pretty small and it couldn't actually do the later text classification of receipts I wished to achieve. Since I needed another dataset, I picked SROIE2019 in order to classify the receipt text after OCR was completed. In this assignment I achieved receipt classification based on the text found on it. For the last assignment, the plan is to do OCR on an image and then give it to the trained model in order to extract information such as a company name, date, total amount.

## Error metric

Since it is a classification task, the metrics are F1, precision, recall.

## Target of error metric

Best metric I saw online in an article that tried a simlar approach was:

f1 = 0.9497638471446973
loss = 0.030156854022708204
precision = 0.9317607413647851
recall = 0.968476357267951

But since there was no clear explanation of the method, my results differ a little bit.

## Achieved error metric

|            | precision |   recall | f1-score |  support  |
| --------   | -------   | -------- | -------  | --------  |
| ADDRESS    |   0.91    |   0.93   |    0.92  |    3806   |
|   TOTAL    |   0.22    |  0.08    |   0.11   |    358    |
| COMPANY    |   0.86    |  0.91    |   0.89   |   1457    |
|    DATE    |   0.71    |  0.54    |   0.61   |    409    |
|micro avg    |   0.87    |  0.85    | 0.86     |  6030    |
|macro avg    |   0.84    |  0.85    | 0.84     |  6030    |

f1 = 0.8588007736943908
loss = 0.15838493681936103
precision = 0.8711823920832622
recall = 0.8467661691542289

## The amount spent on each task

- Reformulating the approach: 5h
- Getting a new dataset and preparing the data: 10h
- Training the model: 30h
- Evaluation: 10h

## How to run

Downloading model and dataset from github release should be the first step.
Model should be put into `model/output` and dataset into `data/SROIE2019`.

Then run `install_conda.sh` to prepare the enviroment.

After that run `run_train.sh` and `run_eval.sh` to train and evaluate the model.
