i1 = int
i2 = int
f1 = float
f2 = float
s1 = str
s2 = str
b1 = bool
b2 = bool


""" Opreartions on the variables """
i1=15
i2= 4
print(f"The sum of {i1} and {i2} is {i1+i2}")

f1=2.53
f2=3.54
print(f"The sum of {f1} and {f2} is {f1+f2}")

s1=" Sakai"
s2="Jin"
print(s2+s1)

b1 = True
b2 = False

and_op= b1 and b2
or_op= b1 or b2


"""Dictionary to store their variables by their types as keys  
    e.g., int[10,20]
"""

dict={
    "int":[i1, i2],
    "float":[f1,f2],
    "str":[s1,s2],
    "bool":[b1,b2,and_op,or_op]
}

print("\nThe data in the dictionary are:")
for data_type, values in dict.items():
    print(f"{data_type}: {values}")
