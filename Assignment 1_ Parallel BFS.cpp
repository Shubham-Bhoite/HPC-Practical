#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

struct Node {
  int data;
  vector<Node*> children;
};

void parallel_BFS(Node* root) {
  #pragma omp parallel shared(root)
  {
    queue<Node*> queue;
    queue.push(root);

    while (!queue.empty()) {
      // Single thread to determine level size to avoid race conditions
      int level_size = queue.size();  // Declare level_size here

      // Parallel processing for nodes in the current level
      #pragma omp for nowait
      for (int i = 0; i < level_size; ++i) {
        Node* current = queue.front();
        queue.pop();

        // Visit current node
        cout << current->data << " ";

        // Add unvisited children to queue in parallel
        #pragma omp task shared(queue)
        for (Node* child : current->children) {
          queue.push(child);
        }
      }

      // Wait for all tasks within the loop to finish before continuing
      #pragma omp taskwait
    }
  }
}

int main() {
  // Create a sample tree using initializer list (corrected)
  Node* root = new Node{1, {new Node{2}, new Node{3, {new Node{4}}}}};

  parallel_BFS(root);

  return 0;
}
