import numpy as np
from timeit import default_timer as timer

def CreateVector(min,max,n):    
    import numpy as np
    np.random.seed(42)
    return(np.random.randint(min,max,n))

def Swap(list,a,b):
    list[a],list[b] = list[b],list[a]

#--------------------------------------
def InsertionSort(vector):
    vector = np.copy(vector)
    t0 = timer()
    for i in range(len(vector)-1):
        
        if vector[i] > vector[i+1]:
            j = i 
            while (vector[j] > vector[j+1]) and j>=0:
                vector[j], vector[j+1] = vector[j+1], vector[j]
                j -= 1
    t = timer()
    delta_t = t-t0
    return([vector,delta_t])

def QuickSort(vector,begin=0,end=-1):
    vector = np.copy(vector)
    if end == -1:
        end = len(vector) - 1
    def Partition(vector,begin,end):
        pivot_index = end 
        pivot = vector[pivot_index] #is it really cheaper to declare the element instead of using its index?
        i = begin
        for j in range(begin,end):
            if vector[j] <= pivot: 
                Swap(vector,i,j)
                i += 1
        Swap(vector,i,pivot_index)
        return i

    def _QuickSort(vector,begin=0,end=-1):
        if len(vector) == 1:
            return vector
        if begin < end:
            pivot_index = Partition(vector, begin, end)
            _QuickSort(vector, begin, pivot_index-1)
            _QuickSort(vector, pivot_index+1, end)  
    t0 = timer()
    _QuickSort(vector)
    t = timer()
    delta_t = t-t0
    return [vector,delta_t]

def CountingSort(vector,low=None,high=None):
    if low == None:
        low = min(vector)
    if high == None:
        high = max(vector)

    t0 = timer()
    n = len(vector)
    count_vector = np.zeros(high-low+1,dtype=int)
    final_vector = np.zeros(n,dtype=int)
    for element in vector:
        count_vector[element-low] += 1
    j = 0
    for i in range(len(count_vector)):
        if(count_vector[i] != 0):
            for temp in range(count_vector[i]):
                final_vector[j] = i + low
                j += 1
    t = timer()
    delta_t = t-t0
    return [final_vector,delta_t]

def CountingSort2(vector,low=None,high=None):
    if low == None:
        low = min(vector)
    if high == None:
        high = max(vector)
    t0 = timer()
    n = len(vector)
    count_vector = np.zeros(high-low +1,dtype=int)
    final_vector = np.zeros(n,dtype=int)
    for element in vector:
        count_vector[element-low] += 1
    for i in range(1,len(count_vector)):
        count_vector[i] += count_vector[i-1]
    for i in range(len(vector)):
        final_vector[ count_vector[(vector[i] - low)] - 1 ] = vector[i] 
        count_vector[vector[i] - low] -= 1     

    t = timer()
    delta_t = t-t0
    return [final_vector, delta_t]
#-------------------------------------
#I copy one code from the internet to make sure nothing is wrong, because i theory Couting should be faster than Quick
#font:https://www.geeksforgeeks.org/counting-sort/
def count_sort(arr):
    t0 = timer()
    max_element = int(max(arr))
    min_element = int(min(arr))
    range_of_elements = max_element - min_element + 1
    # Create a count array to store count of individual
    # elements and initialize count array as 0
    count_arr = [0 for _ in range(range_of_elements)]
    output_arr = [0 for _ in range(len(arr))]
 
    # Store count of each character
    for i in range(0, len(arr)):
        count_arr[arr[i]-min_element] += 1
 
    # Change count_arr[i] so that count_arr[i] now contains actual
    # position of this element in output array
    for i in range(1, len(count_arr)):
        count_arr[i] += count_arr[i-1]
 
    # Build the output character array
    for i in range(len(arr)-1, -1, -1):
        output_arr[count_arr[arr[i] - min_element] - 1] = arr[i]
        count_arr[arr[i] - min_element] -= 1
 
    # Copy the output array to arr, so that arr now
    # contains sorted characters
    for i in range(0, len(arr)):
        arr[i] = output_arr[i]
    t = timer()
    delta_t = t - t0
    return [arr,delta_t]
#font:https://stackabuse.com/counting-sort-in-python/
def countingsort(inputArray):
    t0 = timer()
    # Find the maximum element in the inputArray
    maxElement= max(inputArray)

    countArrayLength = maxElement+1

    # Initialize the countArray with (max+1) zeros
    countArray = [0] * countArrayLength

    # Step 1 -> Traverse the inputArray and increase 
    # the corresponding count for every element by 1
    for el in inputArray: 
        countArray[el] += 1

    # Step 2 -> For each element in the countArray, 
    # sum up its value with the value of the previous 
    # element, and then store that value 
    # as the value of the current element
    for i in range(1, countArrayLength):
        countArray[i] += countArray[i-1] 

    # Step 3 -> Calculate element position
    # based on the countArray values
    outputArray = [0] * len(inputArray)
    i = len(inputArray) - 1
    while i >= 0:
        currentEl = inputArray[i]
        countArray[currentEl] -= 1
        newPosition = countArray[currentEl]
        outputArray[newPosition] = currentEl
        i -= 1
    t = timer()
    delta = t - t0
    return [outputArray,delta]
#--------------------------------------
def CompareSorting():
    v_testing = CreateVector(1,20,10)
    print("Testing the three algorithms with a small vector:",
    "\n Vector (n=10):",v_testing)
    print("InsertionSort: ",InsertionSort(v_testing))
    print("QuickSort: ",QuickSort(v_testing))
    print("CoutingSort1",CountingSort(v_testing,1,20))
    print("CoutingSort2",CountingSort2(v_testing,1,20))

    print("-------------------------------")
    print("Now, comparing the time spent in each case, with two runs for n in [1,5,9]*10^4")
#"Insertion":InsertionSort

algorithms = {"Insertion":InsertionSort}
#algorithms = {"Quick":QuickSort,"Couting1":CountingSort,"Couting2":CountingSort2,"count":count_sort,"count2":countingsort}
#algorithms = {"Insertion":InsertionSort,"Quick":QuickSort,"Couting1":CountingSort,"Couting2":CountingSort2,"count":count_sort,"count2":countingsort}

def CompareTimes(algorithms,vectors):
    #print(list(map(lambda vector:[algorithms[algorithm](vector)[1] for algorithm in algorithms.keys()], vectors)))
    #results = []
    print([algo for algo in algorithms.keys()])
    for vector in vectors:
        #results.append()
        print([algorithms[algorithm](vector)[1] * 1000 for algorithm in algorithms.keys()])


v_10k = CreateVector(1,10000,10000)
v_50k = CreateVector(1,50000,50000)
v_90k = CreateVector(1,90000,90000)
CompareTimes(algorithms,[v_10k,v_10k,v_50k,v_50k,v_90k,v_90k])