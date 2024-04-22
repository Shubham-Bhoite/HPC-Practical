#include <iostream>
#include <omp.h>
#include <climits>

using namespace std;

// Function to find the minimum value in an array using parallel reduction
void min_reduction(int arr[], int n) {
  int min_value = INT_MAX;  // Initialize min_value to positive infinity

  // Use OpenMP parallel for loop with reduction clause (min)
  #pragma omp parallel for reduction(min: min_value)
  for (int i = 0; i < n; i++) {
    if (arr[i] < min_value) {
      min_value = arr[i];  // Update min_value if a smaller element is found
    }
  }

  cout << "Minimum value: " << min_value << endl;
}

// Function to find the maximum value in an array using parallel reduction
void max_reduction(int arr[], int n) {
  int max_value = INT_MIN;  // Initialize max_value to negative infinity

  // Use OpenMP parallel for loop with reduction clause (max)
  #pragma omp parallel for reduction(max: max_value)
  for (int i = 0; i < n; i++) {
    if (arr[i] > max_value) {
      max_value = arr[i];  // Update max_value if a larger element is found
    }
  }

  cout << "Maximum value: " << max_value << endl;
}

// Function to calculate the sum of elements in an array using parallel reduction
void sum_reduction(int arr[], int n) {
  int sum = 0;

  // Use OpenMP parallel for loop with reduction clause (+)
  #pragma omp parallel for reduction(+: sum)
  for (int i = 0; i < n; i++) {
    sum += arr[i];  // Add each element to the sum
  }

  cout << "Sum: " << sum << endl;
}

// Function to calculate the average of elements in an array using parallel reduction
void average_reduction(int arr[], int n) {
  int sum = 0;

  // Use OpenMP parallel for loop with reduction clause (+)
  #pragma omp parallel for reduction(+: sum)
  for (int i = 0; i < n; i++) {
    sum += arr[i];  // Add each element to the sum
  }

  // Calculate average using the reduced sum (note: consider division by n-1 for unbiased average)
  double average = (double)sum / (n - 1);
  cout << "Average: " << average << endl;
}

int main() {
  int n;
  cout << "\nEnter the total number of elements: ";
  cin >> n;

  int *arr = new int[n]; // Allocate memory for the array

  cout << "\nEnter the elements: ";
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }

  min_reduction(arr, n);
  max_reduction(arr, n);
  sum_reduction(arr, n);
  average_reduction(arr, n);

  delete[] arr;  // Deallocate memory after use
  return 0;
}
