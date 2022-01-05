## Covid classifier based onCT scan image with resnet
### Dataset preparation
your direcotry tree must be like example below:
```
.
├── data
│   ├── CT_Covid(put covid images here)
│   └── CT_NonCovid(put non covid images here)
├── README.md
└── src
    ├── data.py
    ├── main.py
    ├── model.py
    └── utils.py
```
### Setting up conda env
run `conda create --name myenv --file spec-file.txt` in order to create myenv env

then activate myenv with `conda activate myenv` 

code can be executed with below command:
```
bash run.sh
```
