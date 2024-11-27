#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace tyurin_m_linear_topology_mpi {

class LinearTopologyParallelMPI : public ppc::core::Task {
 public:
  explicit LinearTopologyParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int rank;
  int sender;
  int target;
  int data;
  bool result_flag;
  boost::mpi::communicator world;
};

}  // namespace tyurin_m_linear_topology_mpi
