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
