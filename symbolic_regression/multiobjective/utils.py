import numpy as np

#BORROWED FROM: https://www.geeksforgeeks.org/counting-inversions/
# Python 3 program to count inversions in an array
#Time Complexity: O(n log n), The algorithm used is divide and conquer, so in each level, one full array traversal is needed
#                             and there are log n levels, so the time complexity is O(n log n).
#Space Complexity: O(n), Temporary array.
 
# Function to Use Inversion Count
def merge_sort_inversions(arr, data_ord_pred):
    # A temp_arr is created to store
    # sorted array in merge function
    temp_arr = [0]*len(arr)
    return _merge_sort(arr, data_ord_pred, temp_arr, 0, len(arr)-1)
 
# This Function will use MergeSort to count inversions
 
def _merge_sort(arr, data_ord_pred, temp_arr, left, right):
    
    # A variable inv_count is used to store
    # inversion counts in each recursive call
 
    inv_count = 0

    squared_error = 0.
 
    # We will make a recursive call if and only if
    # we have more than one elements
 
    if left < right:
 
        # mid is calculated to divide the array into two subarrays
        # Floor division is must in case of python
 
        mid = (left + right)//2
 
        # It will calculate inversion
        # counts in the left subarray
 
        inv, err = _merge_sort(arr, data_ord_pred, temp_arr, left, mid)
        inv_count += inv
        squared_error += err

        # It will calculate inversion
        # counts in right subarray
 
        inv, err = _merge_sort(arr, data_ord_pred, temp_arr, mid + 1, right)
        inv_count += inv
        squared_error += err

        # It will merge two subarrays in
        # a sorted subarray
 
        inv, err = merge(arr, data_ord_pred, temp_arr, left, mid, right)
        inv_count += inv
        squared_error += err
    
    return inv_count, squared_error
 
# This function will merge two subarrays
# in a single sorted subarray
def merge(arr, data_ord_pred, temp_arr, left, mid, right):
    i = left     # Starting index of left subarray
    j = mid + 1 # Starting index of right subarray
    k = left     # Starting index of to be sorted subarray
    inv_count = 0
    squared_error = 0.

    # Conditions are checked to make sure that
    # i and j don't exceed their
    # subarray limits.
 
    while i <= mid and j <= right:
 
        # There will be no inversion if arr[i] <= arr[j]
 
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            k += 1
            i += 1
        else:
            # Inversion will occur.
            temp_arr[k] = arr[j]
            inv_count += (mid-i + 1)

            squared_error += np.sum((data_ord_pred[i:j-1] - data_ord_pred[j])**2)
            
            k += 1
            j += 1
 
    # Copy the remaining elements of left
    # subarray into temporary array
    while i <= mid:
        temp_arr[k] = arr[i]
        k += 1
        i += 1
 
    # Copy the remaining elements of right
    # subarray into temporary array
    while j <= right:
        temp_arr[k] = arr[j]
        k += 1
        j += 1
 
    # Copy the sorted subarray into Original array
    for loop_var in range(left, right + 1):
        arr[loop_var] = temp_arr[loop_var]
         
    return inv_count, squared_error