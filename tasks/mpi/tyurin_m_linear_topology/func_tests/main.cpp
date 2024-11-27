#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/tyurin_m_linear_topology/include/ops_mpi.hpp"

namespace tyurin_m_linear_topology_mpi {

void run_test(boost::mpi::communicator& world, int sender, int target, int data, bool expected_result = true) {
  if (world.size() <= std::max(sender, target)) expected_result = false;

  bool result_flag = false;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&sender));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target));
  taskDataPar->inputs_count.emplace_back(1);

  if (world.rank() == sender) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&data));
    taskDataPar->inputs_count.emplace_back(1);
  } else if (world.rank() == target) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_flag));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto taskParallel = std::make_shared<tyurin_m_linear_topology_mpi::LinearTopologyParallelMPI>(taskDataPar);
  if (expected_result) {
    ASSERT_TRUE(taskParallel->validation());
    taskParallel->pre_processing();
    taskParallel->run();
    taskParallel->post_processing();

    if (world.rank() == target) {
      EXPECT_TRUE(result_flag);
    }
  } else {
    EXPECT_FALSE(taskParallel->validation());
  }
}

}  // namespace tyurin_m_linear_topology_mpi

TEST(tyurin_m_linear_topology_mpi, task0) {
  boost::mpi::communicator world;
  if (world.size() >= 2) tyurin_m_linear_topology_mpi::run_test(world, 0, 1, 100);
}

TEST(tyurin_m_linear_topology_mpi, rev_task0) {
  boost::mpi::communicator world;
  if (world.size() >= 2) tyurin_m_linear_topology_mpi::run_test(world, 1, 0, 100);
}

TEST(tyurin_m_linear_topology_mpi, task1) {
  boost::mpi::communicator world;
  if (world.size() >= 3) tyurin_m_linear_topology_mpi::run_test(world, 0, 2, 100);
}

TEST(tyurin_m_linear_topology_mpi, rev_task1) {
  boost::mpi::communicator world;
  if (world.size() >= 3) tyurin_m_linear_topology_mpi::run_test(world, 2, 0, 100);
}

TEST(tyurin_m_linear_topology_mpi, task2) {
  boost::mpi::communicator world;
  if (world.size() >= 3) tyurin_m_linear_topology_mpi::run_test(world, 1, 2, 100);
}

TEST(tyurin_m_linear_topology_mpi, rev_task2) {
  boost::mpi::communicator world;
  if (world.size() >= 3) tyurin_m_linear_topology_mpi::run_test(world, 2, 1, 100);
}

TEST(tyurin_m_linear_topology_mpi, task3) {
  boost::mpi::communicator world;
  if (world.size() >= 4) tyurin_m_linear_topology_mpi::run_test(world, 1, 2, 100);
}

TEST(tyurin_m_linear_topology_mpi, rev_task3) {
  boost::mpi::communicator world;
  if (world.size() >= 4) tyurin_m_linear_topology_mpi::run_test(world, 2, 1, 100);
}

TEST(tyurin_m_linear_topology_mpi, val_task0) {
  boost::mpi::communicator world;
  if (world.size() < 3) tyurin_m_linear_topology_mpi::run_test(world, 0, 100, -1, false);
}

TEST(tyurin_m_linear_topology_mpi, val_task1) {
  boost::mpi::communicator world;
  if (world.size() < 3) tyurin_m_linear_topology_mpi::run_test(world, 100, 0, -1, false);
}

TEST(tyurin_m_linear_topology_mpi, val_task2) {
  boost::mpi::communicator world;
  tyurin_m_linear_topology_mpi::run_test(world, 0, 0, -1, false);
}
