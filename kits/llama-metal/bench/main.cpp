// The llama Metal kit's driver: runs a GGUF model through the candidate's llama.cpp on the Mac's
// GPU, over a fixed English text, and writes what the model predicted at each position for the
// kit's check. The kit owns this file; the candidate's llama.cpp is linked into it (build.sh).
//
// usage: bench <params.txt> <text.txt> <model.gguf> <out-dir>
//
// params.txt holds key=value lines: mode (decode or prefill), offset (the first token of the
// text's window), prompt, and for decode steps, for prefill records, reps and stride. The settings
// are fixed:
// every layer on the GPU, flash attention on, 4 CPU threads, a context of the window's length,
// and no sampling: each record is the argmax of the logits, never a random draw.
//
// - decode: the prompt's tokens in one batch, then the window's next tokens one at a time
//   (a batch of 1) for `steps` steps, each the true token, not the predicted one (teacher
//   forcing). A record after each step. Tokens processed: prompt + steps.
// - prefill: `reps` windows of the text (1 by default), each of `prompt` tokens in one batch from
//   an empty memory; window r starts `stride` tokens after window r-1, so no two are the same.
//   A record at each of the last `records` positions of every window, window by window: records
//   x reps in all. Tokens processed: prompt x reps.
//
// Each record is the argmax token id, the true next token id (both int32) and the true token's
// log-probability under the logits (float32), little-endian, in <out-dir>/records.bin.
// Any llama error exits non-zero.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "llama.h"

namespace {

constexpr int kThreads = 4;
constexpr uint32_t kUbatch = 512;

[[noreturn]] void fail(int code, const std::string& why) {
  std::fprintf(stderr, "bench: %s\n", why.c_str());
  std::fflush(stderr);
  // _Exit, not exit: ggml's Metal device asserts in its destructor while a model is loaded.
  std::_Exit(code);
}

std::map<std::string, std::string> readParams(const char* path) {
  std::ifstream in(path);
  if (!in) fail(2, std::string("cannot read the parameters ") + path);
  std::map<std::string, std::string> p;
  std::string line;
  while (std::getline(in, line)) {
    auto eq = line.find('=');
    if (eq != std::string::npos) p[line.substr(0, eq)] = line.substr(eq + 1);
  }
  return p;
}

long intParam(const std::map<std::string, std::string>& p, const char* key, long fallback) {
  auto it = p.find(key);
  if (it == p.end()) {
    if (fallback < 0) fail(2, std::string("parameter ") + key + " is missing");
    return fallback;
  }
  char* end = nullptr;
  long v = std::strtol(it->second.c_str(), &end, 10);
  if (end == it->second.c_str() || *end != '\0' || v < 0)
    fail(2, std::string("parameter ") + key + " is not a whole number");
  return v;
}

void quietLog(enum ggml_log_level level, const char* text, void*) {
  if (level == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
}

struct Record {
  int32_t argmax;
  int32_t truth;
  float logp;
};

Record record(const float* logits, int nVocab, llama_token truth) {
  int best = 0;
  float max = logits[0];
  for (int i = 1; i < nVocab; i++)
    if (logits[i] > max) {
      max = logits[i];
      best = i;
    }
  double sum = 0;
  for (int i = 0; i < nVocab; i++) sum += std::exp(double(logits[i]) - double(max));
  double logp = double(logits[truth]) - double(max) - std::log(sum);
  return {best, truth, float(logp)};
}

void add(llama_batch& b, llama_token t, llama_pos pos, bool logits) {
  b.token[b.n_tokens] = t;
  b.pos[b.n_tokens] = pos;
  b.n_seq_id[b.n_tokens] = 1;
  b.seq_id[b.n_tokens][0] = 0;
  b.logits[b.n_tokens] = logits ? 1 : 0;
  b.n_tokens++;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) fail(2, "usage: bench <params.txt> <text.txt> <model.gguf> <out-dir>");
  auto params = readParams(argv[1]);
  const std::string mode = params.count("mode") ? params["mode"] : "";
  if (mode != "decode" && mode != "prefill") fail(2, "parameter mode is not decode or prefill");
  const long offset = intParam(params, "offset", 0);
  const long prompt = intParam(params, "prompt", -1);
  const long steps = mode == "decode" ? intParam(params, "steps", -1) : 0;
  const long records = mode == "prefill" ? intParam(params, "records", -1) : steps;
  const long reps = mode == "prefill" ? intParam(params, "reps", 1) : 1;
  const long stride = mode == "prefill" ? intParam(params, "stride", 0) : 0;
  if (prompt < 1 || records < 1 || reps < 1 || (mode == "prefill" && records > prompt))
    fail(2, "the prompt, records and reps must be at least 1, and a prefill's records at most its prompt");
  if (reps > 1 && stride < 1) fail(2, "a prefill of more than 1 rep needs a stride of at least 1");

  std::ifstream textIn(argv[2], std::ios::binary);
  if (!textIn) fail(2, std::string("cannot read the text ") + argv[2]);
  std::stringstream textBuf;
  textBuf << textIn.rdbuf();
  const std::string text = textBuf.str();

  llama_log_set(quietLog, nullptr);
  llama_backend_init();

  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 999;
  llama_model* model = llama_model_load_from_file(argv[3], mp);
  if (!model) fail(3, "the model did not load");
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int nVocab = llama_vocab_n_tokens(vocab);

  std::vector<llama_token> all(text.size() + 16);
  int n = llama_tokenize(vocab, text.data(), int32_t(text.size()), all.data(), int32_t(all.size()),
                         false, false);
  if (n < 0) fail(3, "the text did not tokenize");
  all.resize(n);

  // A window: the tokens from `start`, after the model's BOS when it wants one. Rep r's window
  // starts at offset + r x stride; every one is checked before any work starts.
  const long need = prompt + steps + 1;  // a window's last record needs the token after it
  const bool bos = llama_vocab_get_add_bos(vocab);
  auto window = [&](long start) {
    std::vector<llama_token> w;
    if (bos) w.push_back(llama_vocab_bos(vocab));
    for (long i = start; long(w.size()) < need && i < long(all.size()); i++) w.push_back(all[i]);
    if (long(w.size()) < need)
      fail(2, "the text has " + std::to_string(all.size()) + " tokens, too few for this cell: the window from token " +
                  std::to_string(start) + " needs " + std::to_string(need - (bos ? 1 : 0)) + " tokens from there" +
                  "; lower the offset, stride or reps");
    return w;
  };
  std::vector<std::vector<llama_token>> windows;
  for (long r = 0; r < reps; r++) windows.push_back(window(offset + r * stride));

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = uint32_t(prompt + steps + 1);
  cp.n_batch = uint32_t(prompt);
  cp.n_ubatch = uint32_t(prompt) < kUbatch ? uint32_t(prompt) : kUbatch;
  cp.n_seq_max = 1;
  cp.n_threads = kThreads;
  cp.n_threads_batch = kThreads;
  cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
  cp.no_perf = true;
  llama_context* ctx = llama_init_from_model(model, cp);
  if (!ctx) fail(3, "the context was not made");

  std::vector<Record> out;
  out.reserve(size_t(records * reps));
  llama_batch batch = llama_batch_init(int32_t(prompt), 0, 1);
  const long first = mode == "prefill" ? prompt - records : prompt;  // the first recorded position
  // A prefill runs each rep's own window from an empty memory and records the tail of every one.
  for (long r = 0; r < reps; r++) {
    const auto& w = windows[size_t(r)];
    if (r > 0) llama_memory_clear(llama_get_memory(ctx), true);
    batch.n_tokens = 0;
    for (long i = 0; i < prompt; i++) add(batch, w[i], llama_pos(i), i >= first);
    if (int e = llama_decode(ctx, batch); e != 0)
      fail(4, "llama_decode failed on the prompt: " + std::to_string(e));
    if (mode != "prefill") continue;
    for (long i = first; i < prompt; i++) {
      const float* l = llama_get_logits_ith(ctx, int32_t(i));
      if (!l) fail(4, "no logits at position " + std::to_string(i));
      out.push_back(record(l, nVocab, w[i + 1]));
    }
  }
  if (mode == "decode") {
    const auto& w = windows[0];
    for (long s = 0; s < steps; s++) {
      const long pos = prompt + s;
      batch.n_tokens = 0;
      add(batch, w[pos], llama_pos(pos), true);
      if (int r = llama_decode(ctx, batch); r != 0)
        fail(4, "llama_decode failed at step " + std::to_string(s) + ": " + std::to_string(r));
      const float* l = llama_get_logits_ith(ctx, -1);
      if (!l) fail(4, "no logits at step " + std::to_string(s));
      out.push_back(record(l, nVocab, w[pos + 1]));
    }
  }
  llama_batch_free(batch);

  const std::string path = std::string(argv[4]) + "/records.bin";
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) fail(5, "cannot write " + path);
  if (std::fwrite(out.data(), sizeof(Record), out.size(), f) != out.size() || std::fclose(f) != 0)
    fail(5, "cannot write " + path);

  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
