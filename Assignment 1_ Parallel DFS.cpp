#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

struct Node {
  int data;
  vector<Node*> neighbors;
};

void parallel_DFS(Node* node, vector<bool>& visited) {
  visited[node->data] = true;
  cout << node->data << " ";

  // Parallel exploration of unvisited neighbors
  #pragma omp parallel for
  for (Node* neighbor : node->neighbors) {
    if (!visited[neighbor->data]) {
      parallel_DFS(neighbor, visited);
    }
  }
}

int main() {
  // Create a sample undirected graph (adjacency list)
  vector<Node> graph(5);

  graph[0].neighbors = {&graph[1], &graph[2]};
  graph[1].neighbors = {&graph[0], &graph[3]};
  graph[2].neighbors = {&graph[0], &graph[4]};
  graph[3].neighbors = {&graph[1]};
  graph[4].neighbors = {&graph[2]};

  vector<bool> visited(graph.size(), false);
  parallel_DFS(&graph[0], visited);

  return 0;
}
