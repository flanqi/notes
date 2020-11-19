## Coding

**1) FizzBuzz**
```python
for i in range(1,101):
  if i % 3 == 0 and i % 5 == 0:
    print("Fizz Buzz")
  elif i % 3 == 0：
    print("Fizz")
  elif i % 5 == 0:
    print("Buzz)
  else:
    print(i)
```

**2) Remove Duplicates**
```python
def remove_duplicates(lst):
  seen = set()
  result = []
  
  for element in lst:
  
    if element not in seen:
      seen.add(element)
      result.append(element)
  
  return result
```

**3) Power of 2 **. Print numbers between 2 and 16 that are powers of 2 using list comprehension.
```python
[x for x in range(2,17) if (x&(x-1)==0) and (x!=0)]
```

**4) Numbers containing 5**. Print the list of numbers having 5 between 1 and 10001.
```python
def get_num_lst(num):
    lst = []
    
    while(num!=0):
        lst.append(num%10)
        num = num//10
    return lst
    
result = []

for i in range(1,10002):
    if(5 in get_num_lst(i)):
        result.append(i)

result
```
