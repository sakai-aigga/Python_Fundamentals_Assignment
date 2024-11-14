#TEmperature converter: C to F or F to C
# ask user for temperature value and also for unit
class Temperature:

    def __init__(self,val,unit):
        self.v=val
        self.u=unit
        
    def toFahrenheit(self):
            print(f"The temperature in F is {self.v*(9/5)+32}") 
        
    def toCelcius(self):
            print(f"The temperature in F is {(self.v-32)*(5/9)}") 
        
def main(): 
    val=float(input("Enter the temperature value: "))
    unit=input("Enter unit (f/c): ")

    temp=Temperature(val,unit)
    if unit.lower()=='c':
        temp.toFahrenheit()
    elif unit.lower()=='f':
        temp.toCelcius()
    else:
        print("Enter a valid unit (c/f)")



main()
#edited new code

