def sparse_search(data, search_val):
  print("Data: " + str(data))
  print("Search Value: " + str(search_val))
  first = 0
  last = len(data) - 1
  
  while first <= last:
    mid = (first + last) // 2
    if not data[mid]:
      left = mid - 1
      right = mid + 1
      while(True):
        if left < first and right > last:
          "{0} is not in the dataset".format(search_val)
          return
        elif right <= last and data[right]:
          mid = right
          break
        elif left >= first and data[left]:
          mid = left
          break
        right += 1
        left -= 1
    if data[mid] == search_val:
      print("{0} found at position {1}".format(search_val, mid))
    if search_val < data[mid]:
      last = mid - 1
    if search_val > data[mid]:
      first = mid + 1
  print("{0} is not in the dataset".format(search_val))
