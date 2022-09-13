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
    print("task 1 is all the project")
elif task == 2:
    import simulations.task2.base
elif task == 3:
    if len(sys.argv) <= 2:
        print("This task need to specify if USB or TS")
        print(" ")
        print("Example for task 3 UCB use: python3 run.py 3 UCB")
        print(" ")
    type = sys.argv[2]
    if type == "UCB":
        import simulations.task3.base_ucb
    elif type == "TS":
        import simulations.task3.base_ts
    else:
        print("It could only be: \"UCB\" or \"TS\"")
elif task == 4:
    if len(sys.argv) <= 2:
        print("This task need to specify if USB or TS")
        print(" ")
        print("Example for task 4 UCB use: python3 run.py 3 UCB")
        print(" ")
    type = sys.argv[2]
    if type == "UCB":
        import simulations.task4.base_ucb
    elif type == "TS":
        import simulations.task4.base_ts
    else:
        print("It could only be: \"UCB\" or \"TS\"")
elif task == 5:
    if len(sys.argv) <= 2:
        print("This task need to specify if ALL or UCB or TS")
        print(" ")
        print("Example for task 5 UCB use: python3 run.py 5 UCB")
        print(" ")
    type = sys.argv[2]
    if type == "ALL":
        import simulations.task5.base
    elif type == "UCB":
        import simulations.task5.base_ucb
    elif type == "TS":
        import simulations.task5.base_ts
    else:
        print("Error name - errore!!!")
elif task == 6:
    if len(sys.argv) <= 2:
        print("This task need to specify if change_detection or change_detection_discard_all or sliding_window")
        print(" ")
        print("Example for task 6 with Change Detection use: python3 run.py 6 change_detection")
        print(" ")
    type = sys.argv[2]
    if type == "change_detection":
        import simulations.task6.base_change_detection
    elif type == "sliding_window":
        import simulations.task6.base_sliding_window
    elif type == "change_detection_discard_all":
        import simulations.task6.base_change_detection_discard_all
    else:
        print("It could only be: \"change_detection\" or \"change_detection_discard_all\" or \"sliding_window\"")
elif task == 7:
    import simulations.task7.base
else:
    print("no task found...")