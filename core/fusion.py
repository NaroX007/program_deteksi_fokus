def decide(on_task_time, off_task_time):
    total_time = on_task_time + off_task_time

    if total_time == 0:
        return "UNKNOWN", 0

    ratio = on_task_time / total_time

    if ratio >= 0.8:
        return "FOCUSED", ratio
    else:
        return "NOT FOCUSED", ratio
