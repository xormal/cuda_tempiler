// Разметка ПОДСТАВНОЙ цели (см. phase_fake_target.py). Компилировать не нужно: этот файл читает
// только сканер разметки phaseprof.py -- он берёт отсюда имена и id фаз.
#include "fmha_phase.h"
void fake() {
  float a = 0.f, b = 0.f, c = 0.f;
  FMHA_PHASE(gemm1, 0) { a += 1.f; }
  FMHA_SEAL(a);
  FMHA_PHASE(gemm2, 1) { b += 1.f; }
  FMHA_SEAL(b);
  FMHA_PHASE(softmax, 2) { c += 1.f; }
  FMHA_SEAL(c);
}
