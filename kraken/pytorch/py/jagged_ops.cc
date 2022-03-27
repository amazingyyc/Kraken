#include "pytorch/py/jagged_ops.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace kraken {
namespace py {
namespace jagged {

// v: [row, col]
// offset: [k + 1]
// out: [k, col]
template <typename T>
void SumForwardImpl(const T* v, int64_t col, const int64_t* offsets, int64_t k,
                    T patch_value, T* out) {
  int64_t limit = col / 8 * 8;

#pragma omp parallel for
  for (int64_t i = 0; i < k; ++i) {
    if (offsets[i + 1] <= offsets[i]) {
      T* out_p = out + i * col;

      for (int64_t l = 0; l < col; ++l) {
        out_p[l] = patch_value;
      }
    } else {
      T* out_p = out + i * col;
      const T* v_p = v + offsets[i] * col;

      // Copy first row.
      memcpy(out_p, v_p, sizeof(T) * col);

      // Add other row.
      for (int64_t j = offsets[i] + 1; j < offsets[i + 1]; ++j) {
        v_p = v + j * col;

        int64_t l = 0;
        for (; l < limit; l += 8) {
          out_p[l] += v_p[l];
          out_p[l + 1] += v_p[l + 1];
          out_p[l + 2] += v_p[l + 2];
          out_p[l + 3] += v_p[l + 3];
          out_p[l + 4] += v_p[l + 4];
          out_p[l + 5] += v_p[l + 5];
          out_p[l + 6] += v_p[l + 6];
          out_p[l + 7] += v_p[l + 7];
        }

        for (; l < col; ++l) {
          out_p[l] += v_p[l];
        }
      }
    }
  }
}

#ifdef __AVX2__
template <>
void SumForwardImpl<float>(const float* v, int64_t col, const int64_t* offsets,
                           int64_t k, float patch_value, float* out) {
  int64_t limit = col / 8 * 8;

  __m256 path_value_v = _mm256_set1_ps(patch_value);

#pragma omp parallel for
  for (int64_t i = 0; i < k; ++i) {
    if (offsets[i + 1] <= offsets[i]) {
      // Set path value.
      float* out_p = out + i * col;

      int64_t l = 0;
      for (; l < limit; l += 8) {
        _mm256_store_ps(out_p + l, path_value_v);
      }

      for (; l < col; ++l) {
        out_p[l] = patch_value;
      }
    } else {
      float* out_p = out + i * col;
      const float* v_p = v + offsets[i] * col;

      // Copy first row.
      memcpy(out_p, v_p, sizeof(float) * col);

      // Add left rows.
      for (int64_t j = offsets[i] + 1; j < offsets[i + 1]; ++j) {
        v_p = v + j * col;

        int64_t l = 0;
        for (; l < limit; l += 8) {
          __m256 a = _mm256_load_ps(out_p + l);
          __m256 b = _mm256_load_ps(v_p + l);
          _mm256_store_ps(out_p + l, _mm256_add_ps(a, b));
        }

        for (; l < col; ++l) {
          out_p[l] += v_p[l];
        }
      }
    }
  }
}
#endif

torch::Tensor SumForward(torch::Tensor values, torch::Tensor offsets,
                         float patch_value) {
  TORCH_CHECK(!values.is_cuda() && !offsets.is_cuda());
  TORCH_CHECK(offsets.sizes().size() == 1);

  if (torch::numel(values) == 0) {
    torch::zeros_like(values);
  }

  TORCH_CHECK(offsets.sizes()[0] > 1);

  auto offsets_i64 = offsets.to(torch::kInt64);

  int64_t col = 1;
  for (size_t i = 1; i < values.sizes().size(); ++i) {
    col *= values.sizes()[i];
  }

  int64_t k = offsets_i64.sizes()[0] - 1;

  auto out_dims = values.sizes().vec();
  out_dims[0] = k;

  auto out = torch::empty(out_dims, values.scalar_type());

  if (values.scalar_type() == torch::kFloat32) {
    SumForwardImpl<float>(values.data_ptr<float>(), col,
                          offsets_i64.data_ptr<int64_t>(), k, patch_value,
                          out.data_ptr<float>());
  } else if (values.scalar_type() == torch::kFloat64) {
    SumForwardImpl<double>(values.data_ptr<double>(), col,
                           offsets_i64.data_ptr<int64_t>(), k, patch_value,
                           out.data_ptr<double>());
  } else {
    TORCH_CHECK(false, "Unsupport dtype:", values.scalar_type());
  }

  return out;
}

template <typename T>
void SumBackwardImpl(const int64_t* offsets, int64_t k, const T* grads,
                     int64_t col, T* vgrads) {
#pragma omp parallel for
  for (int64_t i = 0; i < k; ++i) {
    const T* grads_p = grads + i * col;
    T* vgrads_p = vgrads + offsets[i] * col;

    for (int64_t j = offsets[i]; j < offsets[i + 1]; ++j) {
      memcpy(vgrads_p, grads_p, sizeof(T) * col);
    }
  }
}

torch::Tensor SumBackward(torch::Tensor offsets, torch::Tensor grads) {
  TORCH_CHECK(!offsets.is_cuda() && !grads.is_cuda());

  if (torch::numel(grads) == 0) {
    return torch::zeros_like(grads);
  }

  TORCH_CHECK(offsets.sizes().size() == 1);
  TORCH_CHECK(offsets.sizes()[0] > 1);
  TORCH_CHECK(offsets.sizes()[0] == grads.sizes()[0] + 1);

  auto offsets_i64 = offsets.to(torch::kInt64);

  int64_t k = grads.sizes()[0];

  int64_t row = offsets_i64.data_ptr<int64_t>()[k];
  int64_t col = 1;
  for (size_t i = 1; i < grads.sizes().size(); ++i) {
    col *= grads.sizes()[i];
  }

  auto vgrads_dims = grads.sizes().vec();
  vgrads_dims[0] = row;

  auto vgrads = torch::empty(vgrads_dims, grads.scalar_type());

  if (grads.scalar_type() == torch::kFloat32) {
    SumBackwardImpl<float>(offsets_i64.data_ptr<int64_t>(), k,
                           grads.data_ptr<float>(), col,
                           vgrads.data_ptr<float>());
  } else if (grads.scalar_type() == torch::kFloat64) {
    SumBackwardImpl<double>(offsets_i64.data_ptr<int64_t>(), k,
                            grads.data_ptr<double>(), col,
                            vgrads.data_ptr<double>());
  } else {
    TORCH_CHECK(false, "Unsupport dtype:", grads.scalar_type());
  }

  return vgrads;
}

// v: [row, col]
// offset: [k + 1]
// out: [k, col]
template <typename T>
void MeanForwardImpl(const T* v, int64_t col, const int64_t* offsets, int64_t k,
                     T patch_value, T* out) {
  SumForwardImpl<T>(v, col, offsets, k, patch_value, out);

  int64_t limit = col / 8 * 8;

#pragma omp parallel for
  for (int64_t i = 0; i < k; ++i) {
    if (offsets[i + 1] > offsets[i]) {
      T* out_p = out + i * col;

      // Mean
      T ratio = 1.0 / T(offsets[i + 1] - offsets[i]);

      int64_t l = 0;
      for (; l < limit; l += 8) {
        out_p[l] *= ratio;
        out_p[l + 1] *= ratio;
        out_p[l + 2] *= ratio;
        out_p[l + 3] *= ratio;
        out_p[l + 4] *= ratio;
        out_p[l + 5] *= ratio;
        out_p[l + 6] *= ratio;
        out_p[l + 7] *= ratio;
      }

      for (; l < col; ++l) {
        out_p[l] *= ratio;
      }
    }
  }
}

#ifdef __AVX2__
template <>
void MeanForwardImpl<float>(const float* v, int64_t col, const int64_t* offsets,
                            int64_t k, float patch_value, float* out) {
  SumForwardImpl<float>(v, col, offsets, k, patch_value, out);

  int64_t limit = col / 8 * 8;

#pragma omp parallel for
  for (int64_t i = 0; i < k; ++i) {
    if (offsets[i + 1] > offsets[i]) {
      float* out_p = out + i * col;
      float ratio = 1.0 / float(offsets[i + 1] - offsets[i]);

      __m256 ratio_v = _mm256_set1_ps(ratio);

      int64_t l = 0;
      for (; l < limit; l += 8) {
        __m256 a = _mm256_load_ps(out_p + l);
        _mm256_store_ps(out_p + l, _mm256_mul_ps(a, ratio_v));
      }

      for (; l < col; ++l) {
        out_p[l] *= ratio;
      }
    }
  }
}
#endif

torch::Tensor MeanForward(torch::Tensor values, torch::Tensor offsets,
                          float patch_value) {
  TORCH_CHECK(!values.is_cuda() && !offsets.is_cuda());
  TORCH_CHECK(offsets.sizes().size() == 1);

  if (torch::numel(values) == 0) {
    torch::zeros_like(values);
  }

  TORCH_CHECK(offsets.sizes()[0] > 1);

  auto offsets_i64 = offsets.to(torch::kInt64);

  int64_t col = 1;
  for (size_t i = 1; i < values.sizes().size(); ++i) {
    col *= values.sizes()[i];
  }

  int64_t k = offsets_i64.sizes()[0] - 1;

  auto out_dims = values.sizes().vec();
  out_dims[0] = k;

  auto out = torch::empty(out_dims, values.scalar_type());

  if (values.scalar_type() == torch::kFloat32) {
    MeanForwardImpl<float>(values.data_ptr<float>(), col,
                           offsets_i64.data_ptr<int64_t>(), k, patch_value,
                           out.data_ptr<float>());
  } else if (values.scalar_type() == torch::kFloat64) {
    MeanForwardImpl<double>(values.data_ptr<double>(), col,
                            offsets_i64.data_ptr<int64_t>(), k, patch_value,
                            out.data_ptr<double>());

  } else {
    TORCH_CHECK(false, "Unsupport dtype:", values.scalar_type());
  }

  return out;
}

template <typename T>
void MeanBackwardImpl(const int64_t* offsets, int64_t k, const T* grads,
                      int64_t col, T* vgrads) {
  int64_t limit = col / 8 * 8;

#pragma omp parallel for
  for (int64_t i = 0; i < k; ++i) {
    if (offsets[i + 1] > offsets[i]) {
      T ratio = 1.0 / T(offsets[i + 1] - offsets[i]);

      const T* grads_p = grads + i * col;
      T* vgrads_first_p = vgrads + offsets[i] * col;

      int64_t l = 0;
      for (; l < limit; l += 8) {
        vgrads_first_p[l] = grads_p[l] * ratio;
        vgrads_first_p[l + 1] = grads_p[l + 1] * ratio;
        vgrads_first_p[l + 2] = grads_p[l + 2] * ratio;
        vgrads_first_p[l + 3] = grads_p[l + 3] * ratio;
        vgrads_first_p[l + 4] = grads_p[l + 4] * ratio;
        vgrads_first_p[l + 5] = grads_p[l + 5] * ratio;
        vgrads_first_p[l + 6] = grads_p[l + 6] * ratio;
        vgrads_first_p[l + 7] = grads_p[l + 7] * ratio;
      }

      for (; l < col; ++l) {
        vgrads_first_p[l] = grads_p[l] * ratio;
      }

      for (int64_t j = offsets[i] + 1; j < offsets[i + 1]; ++j) {
        T* vgrads_other_p = vgrads + j * col;
        memcpy(vgrads_other_p, vgrads_first_p, sizeof(T) * col);
      }
    }
  }
}

#ifdef __AVX2__
template <>
void MeanBackwardImpl(const int64_t* offsets, int64_t k, const float* grads,
                      int64_t col, float* vgrads) {
  int64_t limit = col / 8 * 8;

#pragma omp parallel for
  for (int64_t i = 0; i < k; ++i) {
    if (offsets[i + 1] > offsets[i]) {
      float ratio = 1.0 / float(offsets[i + 1] - offsets[i]);
      const float* grads_p = grads + i * col;
      float* vgrads_first_p = vgrads + offsets[i] * col;

      __m256 ratio_v = _mm256_set1_ps(ratio);

      int64_t l = 0;
      for (; l < limit; l += 8) {
        __m256 a = _mm256_load_ps(grads_p + l);
        _mm256_store_ps(vgrads_first_p + l, _mm256_mul_ps(a, ratio_v));
      }

      for (; l < col; ++l) {
        vgrads_first_p[l] = grads_p[l] * ratio;
      }

      for (int64_t j = offsets[i] + 1; j < offsets[i + 1]; ++j) {
        float* vgrads_other_p = vgrads + j * col;
        memcpy(vgrads_other_p, vgrads_first_p, sizeof(float) * col);
      }
    }
  }
}
#endif

torch::Tensor MeanBackward(torch::Tensor offsets, torch::Tensor grads) {
  TORCH_CHECK(!offsets.is_cuda() && !grads.is_cuda());

  if (torch::numel(grads) == 0) {
    return torch::zeros_like(grads);
  }

  TORCH_CHECK(offsets.sizes().size() == 1);
  TORCH_CHECK(offsets.sizes()[0] > 1);
  TORCH_CHECK(offsets.sizes()[0] == grads.sizes()[0] + 1);

  auto offsets_i64 = offsets.to(torch::kInt64);

  int64_t k = grads.sizes()[0];

  int64_t row = offsets_i64.data_ptr<int64_t>()[k];
  int64_t col = 1;
  for (size_t i = 1; i < grads.sizes().size(); ++i) {
    col *= grads.sizes()[i];
  }

  auto vgrads_dims = grads.sizes().vec();
  vgrads_dims[0] = row;

  auto vgrads = torch::empty(vgrads_dims, grads.scalar_type());

  if (grads.scalar_type() == torch::kFloat32) {
    MeanBackwardImpl<float>(offsets_i64.data_ptr<int64_t>(), k,
                            grads.data_ptr<float>(), col,
                            vgrads.data_ptr<float>());
  } else if (grads.scalar_type() == torch::kFloat64) {
    MeanBackwardImpl<double>(offsets_i64.data_ptr<int64_t>(), k,
                             grads.data_ptr<double>(), col,
                             vgrads.data_ptr<double>());
  } else {
    TORCH_CHECK(false, "Unsupport dtype:", grads.scalar_type());
  }

  return vgrads;
}

}  // namespace jagged
}  // namespace py
}  // namespace kraken
