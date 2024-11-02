#include "mpi/tyurin_m_count_sentences_in_string/include/ops_mpi.hpp"

#include <algorithm>
#include <thread>

using namespace std::chrono_literals;

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential::pre_processing() {
  internal_order_test();
  input_str_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  sentence_count_ = 0;
  return true;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential::run() {
  internal_order_test();

  bool inside_sentence = false;
  for (char c : input_str_) {
    if (is_sentence_end(c)) {
      if (inside_sentence) {
        sentence_count_++;
        inside_sentence = false;
      }
    } else if (!is_whitespace(c)) {
      inside_sentence = true;
    }
  }
  return true;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = sentence_count_;
  return true;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential::is_sentence_end(char c) {
  return c == '.' || c == '!' || c == '?';
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskSequential::is_whitespace(char c) {
  return c == ' ' || c == '\n' || c == '\t';
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    input_str_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  }

  boost::mpi::broadcast(world, input_str_, 0);

  int chunk_size = input_str_.size() / world.size();
  int start = world.rank() * chunk_size;
  int end = (world.rank() == world.size() - 1) ? input_str_.size() : start + chunk_size;

  local_input_ = input_str_.substr(start, end - start);
  local_sentence_count_ = 0;
  sentence_count_ = 0;
  return true;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel::run() {
  internal_order_test();

  bool inside_sentence = false;

  for (char c : local_input_) {
    if (is_sentence_end(c)) {
      if (inside_sentence || c == local_input_.front()) {
        local_sentence_count_++;
        inside_sentence = false;
      }
    } else if (!is_whitespace(c)) {
      inside_sentence = true;
    }
  }

  if (!local_input_.empty() && is_sentence_end(local_input_.back())) {
    inside_sentence = false;
  }

  boost::mpi::reduce(world, local_sentence_count_, sentence_count_, std::plus<>(), 0);

  return true;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = sentence_count_;
  }

  return true;
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel::is_sentence_end(char c) {
  return c == '.' || c == '!' || c == '?';
}

bool tyurin_m_count_sentences_in_string_mpi::SentenceCountTaskParallel::is_whitespace(char c) {
  return c == ' ' || c == '\n' || c == '\t';
}
