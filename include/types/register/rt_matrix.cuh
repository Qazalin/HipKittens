/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

 #pragma once

 #include <concepts>
 
 namespace kittens {
 namespace ducks {
 /**
  * @namespace rt_matrix
  * 
  * @brief A namespace for template metaprogramming with register tile layouts.
  */
 namespace rt_matrix {
 
 /**
  * @brief A dummy type used to identify a row-major layout for a register tile.
  */
 struct mfma_32x32x16 {
    static constexpr int tile_size_row_in = 32; // Assumption: row is the non-reduction dimension
    static constexpr int tile_size_col_in = 16; // Assumption: col is the reduction dimension
    static constexpr int tile_size_row_out = 32;
    static constexpr int tile_size_col_out = 32;
 }; // for most matrices
 /**
  * @brief A dummy type used to identify a col-major layout for a register tile.
  */
 struct mfma_16x16x32 {
    static constexpr int tile_size_row_in = 16;
    static constexpr int tile_size_col_in = 32;
    static constexpr int tile_size_row_out = 16;
    static constexpr int tile_size_col_out = 16;
 }; // for the B-matrix of MMA ops.

 
 template<typename T>
 concept all = std::is_same_v<T, mfma_32x32x16> || std::is_same_v<T, mfma_16x16x32>;

 } // namespace rt_matrix
 } // namespace ducks
 } // namespace kittens

