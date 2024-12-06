#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/tyurin_m_linear_topology/include/ops_mpi.hpp"

namespace tyurin_m_linear_topology_mpi {

std::vector<int> generate_random_vector(size_t size, int min_value = 0, int max_value = 100) {
  std::mt19937 generator(static_cast<unsigned>(std::time(nullptr)));
  std::uniform_int_distribution<int> distribution(min_value, max_value);

  std::vector<int> random_vector(size);
  for (auto& element : random_vector) {
    element = distribution(generator);
  }

  return random_vector;
}

void run_test(boost::mpi::communicator& world, int sender, int target, size_t vector_size,
              bool expected_result = true) {
  if (world.size() <= std::max(sender, target)) {
    GTEST_SKIP();
    return;
  }

  bool result_flag = false;
  std::vector<int> random_vector;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&sender));
  taskDataPar->inputs_count.emplace_back(1);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target));
  taskDataPar->inputs_count.emplace_back(1);

  if (world.rank() == sender) {
    random_vector = generate_random_vector(vector_size, 0, 100);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&random_vector));
    taskDataPar->inputs_count.emplace_back(random_vector.size());
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
  if (world.size() >= 2) tyurin_m_linear_topology_mpi::run_test(world, 0, 1, 10);
}

TEST(tyurin_m_linear_topology_mpi, rev_task0) {
  boost::mpi::communicator world;
  if (world.size() >= 2) tyurin_m_linear_topology_mpi::run_test(world, 1, 0, 10);
}

TEST(tyurin_m_linear_topology_mpi, task1) {
  boost::mpi::communicator world;
  if (world.size() >= 3) tyurin_m_linear_topology_mpi::run_test(world, 0, 2, 15);
}

TEST(tyurin_m_linear_topology_mpi, rev_task1) {
  boost::mpi::communicator world;
  if (world.size() >= 3) tyurin_m_linear_topology_mpi::run_test(world, 2, 0, 15);
}

TEST(tyurin_m_linear_topology_mpi, task2) {
  boost::mpi::communicator world;
  if (world.size() >= 3) tyurin_m_linear_topology_mpi::run_test(world, 1, 2, 20);
}

TEST(tyurin_m_linear_topology_mpi, rev_task2) {
  boost::mpi::communicator world;
  if (world.size() >= 3) tyurin_m_linear_topology_mpi::run_test(world, 2, 1, 20);
}

TEST(tyurin_m_linear_topology_mpi, task3) {
  boost::mpi::communicator world;
  if (world.size() >= 4) tyurin_m_linear_topology_mpi::run_test(world, 1, 2, 25);
}

TEST(tyurin_m_linear_topology_mpi, rev_task3) {
  boost::mpi::communicator world;
  if (world.size() >= 4) tyurin_m_linear_topology_mpi::run_test(world, 2, 1, 25);
}

TEST(tyurin_m_linear_topology_mpi, val_task0) {
  boost::mpi::communicator world;
  if (world.size() < 3) tyurin_m_linear_topology_mpi::run_test(world, 0, 100, 0, false);
}

TEST(tyurin_m_linear_topology_mpi, val_task1) {
  boost::mpi::communicator world;
  if (world.size() < 3) tyurin_m_linear_topology_mpi::run_test(world, 100, 0, 0, false);
}

TEST(tyurin_m_linear_topology_mpi, val_task2) {
  boost::mpi::communicator world;
  tyurin_m_linear_topology_mpi::run_test(world, 0, 0, 0, false);
}
