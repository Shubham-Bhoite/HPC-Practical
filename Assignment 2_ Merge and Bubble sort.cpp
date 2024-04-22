// Write a program to implement Parallel Bubble Sort and Merge sort using OpenMP. Use existing algorithms and measure the performance of sequential and parallel algorithms


#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

// Function to swap two elements
void swap(int& a, int& b) {
  int temp = a;
  a = b;
  b = temp;
}

// Sequential Bubble Sort
void bubbleSortSequential(vector<int>& arr) {
  int n = arr.size();
  for (int i = 0; i < n - 1; ++i) {
    bool swapped = false;
    for (int j = 0; j < n - i - 1; ++j) {
      if (arr[j] > arr[j + 1]) {
        swap(arr[j], arr[j + 1]);
        swapped = true;
      }
    }
    if (!swapped) {
      break; // Early termination if no swaps occurred
    }
  }
}

// Merge function for Merge Sort
void merge(vector<int>& arr, int left, int mid, int right) {
  int n1 = mid - left + 1;
  int n2 = right - mid;

  vector<int> leftArr(n1);
  vector<int> rightArr(n2);

  for (int i = 0; i < n1; ++i) {
    leftArr[i] = arr[left + i];
  }
  for (int j = 0; j < n2; ++j) {
    rightArr[j] = arr[mid + 1 + j];
  }

  int i = 0, j = 0, k = left;
  while (i < n1 && j < n2) {
    if (leftArr[i] <= rightArr[j]) {
      arr[k] = leftArr[i];
      i++;
    } else {
      arr[k] = rightArr[j];
      j++;
    }
    k++;
  }

  while (i < n1) {
    arr[k] = leftArr[i];
    i++;
    k++;
  }

  while (j < n2) {
    arr[k] = rightArr[j];
    j++;
    k++;
  }
}

// Recursive Merge Sort
void mergeSortRecursive(vector<int>& arr, int left, int right) {
  if (left < right) {
    int mid = left + (right - left) / 2;
    mergeSortRecursive(arr, left, mid);
    mergeSortRecursive(arr, mid + 1, right);
    merge(arr, left, mid, right);
  }
}

// Parallel Merge Sort (uses Merge SortRecursive for base cases)
void mergeSortParallel(vector<int>& arr, int left, int right) {
  if (right - left <= 100) { // Threshold for sequential execution (adjust as needed)
    mergeSortRecursive(arr, left, right);
  } else {
    int mid = left + (right - left) / 2;
    #pragma omp task firstprivate(arr, left, mid)
    mergeSortParallel(arr, left, mid);
    #pragma omp task firstprivate(arr, mid + 1, right)
    mergeSortParallel(arr, mid + 1, right);
    #pragma omp taskwait
    merge(arr, left, mid, right);
  }
}

// Parallel Bubble Sort
void bubbleSortParallel(vector<int>& arr) {
  int n = arr.size();
  for (int i = 0; i < n - 1; ++i) {
    bool swapped = false;
    #pragma omp parallel for num_threads(4)  // Adjust num_threads as needed
    for (int j = 0; j < n - i - 1; ++j) {
      if (arr[j] > arr[j + 1]) {
        swap(arr[j], arr[j + 1]);
        swapped = true;
      }
    }
    #pragma omp barrier // Ensure all threads finish iteration before next
    if (!swapped) {
      break; // Early termination if no swaps occurred
    }
  }
}

int main() {
  int N;

  // Get vector size from user
  cout << "Enter the size of the vector: ";
  cin >> N;

  vector<int> arr(N);

  // Get elements from user
  cout << "Enter the elements of the vector (space-separated): ";
  for (int i = 0; i < N; ++i) {
    cin >> arr[i];
  }

  // Measure performance of sequential bubble sort
  auto start = chrono::high_resolution_clock::now();
  bubbleSortSequential(arr);
  auto end = chrono::high_resolution_clock::now();
  double sequentialBubbleSortTime = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e6;
  cout << "Sequential Bubble Sort Time (ms): " << sequentialBubbleSortTime << endl;

  // Print the sorted vector (optional)
  // cout << "Sorted vector (Sequential Bubble Sort): ";
  // for (int i = 0; i < N; ++i) {
  //   cout << arr[i] << " ";
  // }
  // cout << endl;

  // Reset the vector (optional, uncomment if needed for multiple sorts)
  // arr.assign(N, 0);

  // Measure performance of parallel bubble sort
  start = chrono::high_resolution_clock::now();
  bubbleSortParallel(arr);
  end = chrono::high_resolution_clock::now();
  double parallelBubbleSortTime = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e6;
  cout << "Parallel Bubble Sort Time (ms): " << parallelBubbleSortTime << endl;

  // Print the sorted vector (optional)
  // cout << "Sorted vector (Parallel Bubble Sort): ";
  // for (int i = 0; i < N; ++i) {
  //   cout << arr[i] << " ";
  // }
  // cout << endl;

  // Reset the vector (optional, uncomment if needed for multiple sorts)
  // arr.assign(N, 0);

  // Measure performance of sequential merge sort
  start = chrono::high_resolution_clock::now();
  mergeSortRecursive(arr, 0, N - 1);
  end = chrono::high_resolution_clock::now();
  double sequentialMergeSortTime = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e6;
  cout << "Sequential Merge Sort Time (ms): " << sequentialMergeSortTime << endl;

  // Print the sorted vector (optional)
  // cout << "Sorted vector (Sequential Merge Sort): ";
  // for (int i = 0; i < N; ++i) {
  //   cout << arr[i] << " ";
  // }
  // cout << endl;

  // Reset the vector (optional, uncomment if needed for multiple sorts)
  // arr.assign(N, 0);

  // Measure performance of parallel merge sort
  start = chrono::high_resolution_clock::now();
  mergeSortParallel(arr, 0, N - 1);
  end = chrono::high_resolution_clock::now();
  double parallelMergeSortTime = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e6;
  cout << "Parallel Merge Sort Time (ms): " << parallelMergeSortTime << endl;

  // Print the sorted vector (optional)
  // cout << "Sorted vector (Parallel Merge Sort): ";
  // for (int i = 0; i < N; ++i) {
  //   cout << arr[i] << " ";
  // }
  // cout << endl;

  return 0;
}

