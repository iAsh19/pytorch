#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

#include <vector>

namespace lazy_tensors {

std::vector<int64_t> InversePermutation(
    c10::ArrayRef<int64_t> input_permutation);

bool IsPermutation(c10::ArrayRef<int64_t> permutation);

// Gathers the input using the order specified by the permutation. For each i,
// output[i] = input[permutation[i]]. The given permutation must be the same
// size as the input.
template <typename Container>
static std::vector<typename Container::value_type> Permute(
    c10::ArrayRef<int64_t> permutation, const Container& input) {
  using T = typename Container::value_type;
  TORCH_CHECK(input.size() == permutation.size() &&
                  lazy_tensors::IsPermutation(permutation),
              "Invalid permutation specified");
  std::vector<T> output(input.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    output[i] = input[permutation[i]];
  }
  return output;
}

}  // namespace lazy_tensors
