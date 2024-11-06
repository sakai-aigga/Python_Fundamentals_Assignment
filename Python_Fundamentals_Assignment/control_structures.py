def classify_number():
    while True:
        user = input("Enter a number (or type 'exit' to quit): ")
        
        if user.lower() == "exit":
            print("Exiting the program.")
            break
        number = float(user)
        if number > 0:
            print(f"{number} is positive.")
        elif number < 0:
            print(f"{number} is negative.")
        else:
            print("The number is zero.")

classify_number()