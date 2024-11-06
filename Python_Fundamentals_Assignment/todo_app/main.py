# to handle user interactions
from todo_operations import  add_tsk, view_tsk, del_tsk
from utils import clr_scr, user_inp

def main():
    while True:
        clr_scr()
        print("------------------------Your TO-DO list------------------------")
        print("Select  an option:")
        print("__________________")
        print("1. Add New Task")
        print("2. Delete Task")
        print("3. View Tasks")
        print("4. Type 'exit' to close the program")
        print("__________________")
        choice = input("Enter choice (1/2/3 or 'exit') ")
        if choice.lower() == 'exit':
            print("Have a great time.")
            break
                
        elif choice == '1':
            tsk = user_inp("Enter a new task:")
            add_tsk(tsk)
            user_inp("\nPress Enter to return to the menu.")
                   
        elif choice == '2':
            view_tsk()
            try:
                i = int(user_inp("Enter the task index to delete: "))
                del_tsk(i)
            except:
                print("Invalid input. Please enter a number.")
            user_inp("\nPress Enter to return to the menu.")

        elif choice == '3':
            view_tsk()
            user_inp("\nPress Enter to return to the menu.")
        else:
            print("Invalid input.")

if __name__=="__main__":
    main()
           