import sys

if len(sys.argv) <= 1:
    print("USAGE: ")
    print("python3 run.py <task_num>")
    print(" ")
    print("Example task 2: python3 run.py 2")
    print(" ")
    exit(1)

task = int(sys.argv[1])

if task == 1:
    print("task 1 not implemented yet")
elif task == 2:
    import simulations.task2.base
elif task == 3:
    import simulations.task3.base_ts
else:
    print("no task found...")