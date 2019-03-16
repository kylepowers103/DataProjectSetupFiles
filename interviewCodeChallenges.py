#Move zeroes to the end of array
def move_zeros(arr):
    i = j = 0
    while i < len(arr):
        if arr[i] > 0:
            arr[j] = arr[i]
            j += 1
        i += 1

    while j < len(arr):
        arr[j] = 0
        j += 1

    return arr
def fibonacci(n):

    # Write your code here.
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a = 0
        b = 1
        i = 0
        while (i < n):
            a,b =  a + b,a
            print ("Loop",(a,b))
            i = i+1
        #print (a,b)
        return a
fibonacci(5)
