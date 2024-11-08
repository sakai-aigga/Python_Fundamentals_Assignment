
def calculator():
        
    print("Select one operation:")
    print("1. Addition:")
    print("2. Subtraction:")
    print("3. Multiplication:")
    print("4. Division:")
    print("5. Modulus:")

    def add(n1,n2):
        return n1+n2
    def sub(n1,n2):
        return n1-n2
    def mult(n1,n2):
        return n1*n2
    def div(n1,n2):
        return n1/n2
    def mod(n1,n2):
        return n1 % n2
    while True:
            choice = input("Enter choice (1/2/3/4/5) or 'exit' to quit: ")
            
            if choice.lower() == 'exit':
                print("Exiting the calculator.")
                break
            
            if choice in ['1', '2', '3', '4','5']:
                    num1 = float(input("Enter first number: "))
                    num2 = float(input("Enter second number: "))
                    
                    if choice == '1':
                        print(f"Result: {add(num1, num2)}")
                    elif choice == '2':
                        print(f"Result: {sub(num1, num2)}")
                    elif choice == '3':
                        print(f"Result: {mult(num1, num2)}")
                    elif choice == '4':
                        print(f"Result: {div(num1, num2)}")
                    elif choice == '5':
                        print(f"Result: {mod(num1, num2)}")
            else:
                print("Invalid choice. Please select a valid operation.")
calculator()
