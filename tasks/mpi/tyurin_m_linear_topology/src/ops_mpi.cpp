#include "mpi/tyurin_m_linear_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

bool tyurin_m_linear_topology_mpi::LinearTopologyParallelMPI::validation() {
  internal_order_test();
  int val_rank = world.rank();
  int val_size = world.size();
  if (reinterpret_cast<int*>(taskData->inputs[0]) == nullptr || reinterpret_cast<int*>(taskData->inputs[1]) == nullptr)
    return false;
  int val_sender = *reinterpret_cast<int*>(taskData->inputs[0]);
  int val_target = *reinterpret_cast<int*>(taskData->inputs[1]);
  if (val_sender < 0 || val_target < 0 || std::max(val_sender, val_target) >= val_size || val_sender == val_target) {
    return false;
  }
  if (val_rank == val_sender) {
    return !(reinterpret_cast<int*>(taskData->inputs[2]) == nullptr);
  }
  if (val_rank == val_target) {
    return !(reinterpret_cast<bool*>(taskData->outputs[0]) == nullptr);
  }
  return true;
}

bool tyurin_m_linear_topology_mpi::LinearTopologyParallelMPI::pre_processing() {
  internal_order_test();

  rank = world.rank();

  sender = *reinterpret_cast<int*>(taskData->inputs[0]);

  target = *reinterpret_cast<int*>(taskData->inputs[1]);

  if (rank == sender) {
    data = *reinterpret_cast<int*>(taskData->inputs[2]);
  } else if (rank == target) {
    result_flag = false;
  }

  return true;
}

bool tyurin_m_linear_topology_mpi::LinearTopologyParallelMPI::run() {
  internal_order_test();

  int rank = world.rank();

  if ((sender < target && (rank < sender || rank > target)) || (sender > target && (rank > sender || rank < target))) {
    return true;
  }

  if (rank == sender) {
    if (rank < target) {
      world.send(rank + 1, 0, data);
    } else if (rank > target) {
      world.send(rank - 1, 0, data);
    }
  } else {
    int source = (rank > sender) ? rank - 1 : rank + 1;
    world.recv(source, 0, data);

    if (rank != target) {
      if (rank < target) {
        world.send(rank + 1, 0, data);
      } else {
        world.send(rank - 1, 0, data);
      }
    } else {
      result_flag = true;
    }
  }

  return true;
}

bool tyurin_m_linear_topology_mpi::LinearTopologyParallelMPI::post_processing() {
  internal_order_test();

  if (rank == target) {
    *reinterpret_cast<bool*>(taskData->outputs[0]) = result_flag;
  }

  return true;
}
