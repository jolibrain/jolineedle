# LARD dataset

The [LARD dataset](https://github.com/deel-ai/LARD),
is a good demonstration of such a sceniario where the goal is to detect
one small object inside a big image.

You can download the dataset here: [https://share.deel.ai/s/H4iLKRmLkdBWqSt?path=%2Flard%2F1.0.0](https://share.deel.ai/s/H4iLKRmLkdBWqSt?path=%2Flard%2F1.0.0).

Make sure to download the `LARD_train.csv` and all zip files. Then you can
unzip everything. Once this is done, you can prepare the dataset for training
by using the following command:

```py
python3 prepare --dataset-path PATH/TO/DATASET
```

where `PATH/TO/DATASET` is the path to the directory containing the
`LARD_train.csv` and the unzipped files.

## Results on LARD dataset

The results are presented for a model trained with the hyperparameters describes
in the section below. Importantly, the model is given only 8 moves to find and
predict the bounding boxes.

Test with all the dataset (2046 examples, max sequence length = 8):

|                 | visited bbox patches | mAP-50 |
|:---------------:|:--------------------:|:------:|
|  1 random start |         92.6%        |  84.6% |

We measure:

- Visited bbox patches: Once the image has been divided into patches, we count
the number of patches countaining a bounding box to predict visited by the model.
This is then reported by the total number of bbox patches in the image.
This measures the decision capacity of the model.
- mAP-50: The standard mAP-50, where the predicted bounding box are compared to all
true bounding boxes of the images (i.e. we also count the bboxes that have not been
visited by the model during its decisions).
This mainly measures the detection capacity of the model, but it is dependent
of the decision quality of the model.

