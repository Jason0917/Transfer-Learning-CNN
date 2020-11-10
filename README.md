# UAlberta-Multimedia-Master-Program---MM811-2020-Assignment-3
This is the coding parts of UAlberta Multimedia Master Program - MM811 2020 Assignment 2. <br>
For coding parts, all my codes are runned with python 3.8.3, pytorch 1.6.0 and torchvision 0.7.0.

## Running
Simply run 
```
python main.py
```

## Parameters Setting of CNN
| Learning rate | Epoches |
| :-: | :-: |
| 0.0001 | 15 |

## Parameters Setting of Transfer Learning
| Learning rate | Epoches |
| :-: | :-: |
| 0.005 | 30 |


## Experiment Results
### CNN
| Training accuracy | Validation accuracy |
| :-: | :-: |
| 89.63% | 66.99% |

### Transfer Learning (VGG_16)
| Training accuracy | Validation accuracy |
| :-: | :-: |
| 84.48% | 80.29% |

### Transfer Learning (ResNet_18)
| Training accuracy | Validation accuracy |
| :-: | :-: |
| 87.42% | 83.84% |

## Comparision and Analysis
Compared with training a CNN model from scratch, our transfer-learning-based model performed much better. Although CNN model achieved an accuracy of 89.63% on training set, its accuracy on validation set significantly dropped to 66.99%, which means the model is poor-generalized. In my opinion, this happened because our dataset is not sufficient enough to train a model from scratch. 
<br>
Therefore, with a dataset containing over 2,000 images, it would be better for us to use transfer learning to train our own model. As is shown in Experiment Results section, both training and validation accuracy of the model using transfer learning is quite high, and the difference between training and validation accuracy is small, which means the models are generalized.
