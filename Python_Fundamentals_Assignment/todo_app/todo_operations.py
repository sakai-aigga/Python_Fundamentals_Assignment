#functions to add, view and delete tasks

tasks=[]

def view_tsk():
    if not tasks:
        print("There are no tasks in the list.")
    else:
        print("To-Do List:")
        i=0
        for i, task in enumerate(tasks):
            print(f"{i+1}. {task}")

def add_tsk(tsk):
    tasks.append(tsk)
    print(f"Task '{tsk}' was added to the list. ")

    
def del_tsk(i):
    view_tsk()
    try:
        tsk = tasks.pop(i - 1)
        print(f"Task '{tsk}' deleted successfully.")
    except IndexError:
        print("Invalid index. Please try again.")

