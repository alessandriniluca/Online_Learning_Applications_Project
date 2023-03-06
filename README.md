# Online_Learning_Applications_Project
Folder of the project of Online Learning Applications (OLA) at Polimi

## Complete report
Please refer to the file OnlineLearningApplications_Report.pdf

## Installation Instruction
Clone this repo

Open the repository folder

Create a Virtual Environment
```
python3 -m venv ./venv
```

Activate it
```
source venv/bin/activate
```

Install the requirements

```
pip3 install -r requirements.txt 
```

## Execution Instructions:
```
python3 run.py <task_number> <options>

<task_number>
    3 -> to execute task 3
        <options>
            UCB -> to use UCB as MAB
            TS  -> to use Thompson Sampling as MAB
    4 -> to execute task 4
        <options>
            UCB -> to use UCB as MAB
            TS  -> to use Thompson Sampling as MAB    
    5 -> to execute task 5
        <options>
            ALL -> to execute it with known alpha functions
            UCB -> to use UCB as MAB
            TS  -> to use Thompson Sampling as MAB       
    6 -> to execute task 6
        <options>
            sliding_window   -> to use sliding window approach
            change_detection -> to use change detection approch and discard 1 arm
            change_detection_discard_all -> to use change detection approach and discard all arms
    7 -> to execute task 7
```


# Contributors
Alessandrini Luca \
Fabris Matteo \
Portanti Mattia \
Portanti Samuele \
Venturini Luca
