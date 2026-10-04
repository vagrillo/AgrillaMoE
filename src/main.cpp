// AgrillaMoE v1.0.0 — server di inferenza dedicato a Qwen3.6-35B-A3B (GGUF Unsloth)
//
// Costruito sul fork llama.cpp "moe-expansion" (github.com/vagrillo/llama.cpp,
// branch moe-expansion): collega llama-server-impl e aggiunge:
//
//   1. selezione interattiva del modello quantizzato Unsloth;
//      se nessun modello è già scaricato propone il default in base alla VRAM
//      della GPU, altrimenti sceglie l'utente tra i modelli presenti;
//   2. download automatico via CLI `hf` (huggingface_hub) se il file manca;
//   3. profilo MoE-expansion di default del benchmark RUN1209
//      (Qwen3.6-35B-A3B Q8_0: GPQA-Diamond 84.34% exp vs 81.82% native):
//      --moe-experts 16 --moe-expert-threshold 0.8
//      --moe-expert-layer-start 25 --moe-expert-layer-end 39;
//   4. bind di default su 127.0.0.1:8071 e apertura automatica del browser
//      appena il server è in ascolto.
//
// Tutte le impostazioni iniettate lo sono solo se l'utente non le ha già
// passate: ogni flag di llama-server resta pienamente utilizzabile.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <string>
#include <thread>
#include <vector>

#include <filesystem>
namespace fs = std::filesystem;

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <shellapi.h>
    #include <io.h>
    #define agrilla_popen  _popen
    #define agrilla_pclose _pclose
    #define agrilla_isatty _isatty
    #define agrilla_fileno _fileno
#else
    #include <arpa/inet.h>
    #include <netinet/in.h>
    #include <sys/socket.h>
    #include <unistd.h>
    #define agrilla_popen  popen
    #define agrilla_pclose pclose
    #define agrilla_isatty isatty
    #define agrilla_fileno fileno
#endif

// entry point del server, fornito da llama-server-impl (tools/server/server.cpp)
int llama_server(int argc, char ** argv);

#define AGRILLA_VERSION      "1.0.1"
#define AGRILLA_DEFAULT_HOST "127.0.0.1"
#define AGRILLA_DEFAULT_PORT "8071"
#define AGRILLA_HF_REPO      "unsloth/Qwen3.6-35B-A3B-GGUF"

static const char * AGRILLA_BANNER =
    "    _                    _      _  __  __  _   _ ___ \n"
    "   / \\   __ _  ___ _ __ | |    / \\|  \\/  |/ / | |_ _|\n"
    "  / _ \\ / _` |/ _ \\ '_ \\| |   / _ \\ |\\/| | || | || | \n"
    " / ___ \\ (_| |  __/ | | | |  / ___ \\ |  | | || | || | \n"
    "/_/   \\_\\__, |\\___|_| |_|_| /_/   \\_\\_|  |_| \\_/|___|\n"
    "        |___/  dedicated Qwen3.6-35B-A3B inference server\n";

namespace agrilla {

// ---------------------------------------------------------------- utility --

static std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::string to_lower(const std::string & s) {
    std::string r = s;
    for (char & c : r) c = (char) std::tolower((unsigned char) c);
    return r;
}

static std::string human_gb(unsigned long long bytes) {
    char buf[32];
    snprintf(buf, sizeof buf, "%.1f GB", (double) bytes / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

static bool env_flag_on(const char * name) {
    const char * v = std::getenv(name);
    if (!v || !*v) return false;
    std::string s = to_lower(v);
    return s == "1" || s == "y" || s == "yes" || s == "true" || s == "on";
}

static bool have_flag(int argc, char ** argv, std::initializer_list<const char *> names) {
    // riconosce sia "--flag value" sia "--flag=value"
    for (int i = 1; i < argc; ++i) {
        std::string t = argv[i];
        for (const char * n : names) {
            std::string nm(n);
            if (t == nm || t.rfind(nm + "=", 0) == 0) return true;
        }
    }
    return false;
}

// come have_flag ma sulla lista di argomenti gia' processata (evita doppie
// iniezioni quando una modalita' come --agrilla-streaming ha gia' aggiunto un flag)
static bool args_has(const std::vector<std::string> & args, std::initializer_list<const char *> names) {
    for (const auto & t : args)
        for (const char * n : names) {
            std::string nm(n);
            if (t == nm || t.rfind(nm + "=", 0) == 0) return true;
        }
    return false;
}

static std::string get_flag_value(int argc, char ** argv, std::initializer_list<const char *> names,
                                  const std::string & fallback) {
    for (int i = 1; i < argc; ++i) {
        std::string t = argv[i];
        for (const char * n : names) {
            std::string nm(n);
            if (t == nm && i + 1 < argc) return argv[i + 1];
            if (t.rfind(nm + "=", 0) == 0) return t.substr(nm.size() + 1);
        }
    }
    return fallback;
}

// ------------------------------------------------------------- GPU / VRAM --

struct gpu_info {
    std::string name = "sconosciuta";
    long        vram_mib = 0;   // 0 = non rilevata
};

static gpu_info detect_gpu() {
    gpu_info gi;
    const char * q = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits";
#ifdef _WIN32
    std::string cmd = std::string(q) + " 2>nul";
#else
    std::string cmd = std::string(q) + " 2>/dev/null";
#endif
    FILE * p = agrilla_popen(cmd.c_str(), "r");
    if (!p) return gi;
    char line[512];
    while (fgets(line, sizeof line, p)) {
        std::string l = trim(line);
        size_t comma = l.find(',');
        if (comma == std::string::npos) continue;
        std::string name = trim(l.substr(0, comma));
        long        mib  = strtol(trim(l.substr(comma + 1)).c_str(), nullptr, 10);
        if (mib > gi.vram_mib) {          // multi-GPU: prende la piu' grande
            gi.vram_mib = mib;
            gi.name     = name;
        }
    }
    agrilla_pclose(p);
    return gi;
}

// --------------------------------------------------- catalogo quant Unsloth --

struct quant_entry {
    const char * label;    // nome corto per il menu
    const char * file;     // nome file esatto su unsloth/Qwen3.6-35B-A3B-GGUF
    double       gb;       // dimensione approssimativa
    const char * note;     // nota opzionale
};

// catalogo di unsloth/Qwen3.6-35B-A3B-GGUF (rilevato 2026-10), ordinato per dimensione
static const std::vector<quant_entry> k_catalog = {
    {"UD-IQ1_M",    "Qwen3.6-35B-A3B-UD-IQ1_M.gguf",     9.4,  ""},
    {"UD-IQ2_XXS",  "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",  10.0,  ""},
    {"UD-IQ2_M",    "Qwen3.6-35B-A3B-UD-IQ2_M.gguf",    10.7,  ""},
    {"UD-Q2_K_XL",  "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf",  11.4,  "usato nel benchmark 2-bit (GPQA-D 76.26%)"},
    {"UD-IQ3_XXS",  "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",  12.3,  ""},
    {"UD-IQ3_S",    "Qwen3.6-35B-A3B-UD-IQ3_S.gguf",    12.7,  ""},
    {"UD-Q3_K_S",   "Qwen3.6-35B-A3B-UD-Q3_K_S.gguf",   14.3,  ""},
    {"UD-Q3_K_M",   "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf",   15.5,  ""},
    {"UD-Q3_K_XL",  "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",  15.7,  ""},
    {"UD-IQ4_XS",   "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",   16.5,  ""},
    {"UD-IQ4_NL",   "Qwen3.6-35B-A3B-UD-IQ4_NL.gguf",   16.8,  ""},
    {"UD-IQ4_NL_XL","Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf",18.2,  ""},
    {"UD-Q4_K_S",   "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf",   19.5,  ""},
    {"MXFP4_MOE",   "Qwen3.6-35B-A3B-MXFP4_MOE.gguf",   20.2,  "quant nativo MXFP4"},
    {"UD-Q4_K_M",   "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",   20.6,  ""},
    {"UD-Q4_K_XL",  "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",  20.8,  ""},
    {"UD-Q5_K_S",   "Qwen3.6-35B-A3B-UD-Q5_K_S.gguf",   23.2,  ""},
    {"UD-Q5_K_M",   "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf",   24.6,  ""},
    {"UD-Q5_K_XL",  "Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf",  24.8,  ""},
    {"UD-Q6_K",     "Qwen3.6-35B-A3B-UD-Q6_K.gguf",     27.3,  ""},
    {"UD-Q6_K_XL",  "Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf",  29.7,  ""},
    {"Q8_0",        "Qwen3.6-35B-A3B-Q8_0.gguf",        34.4,  "riferimento benchmark RUN1209: GPQA-D 84.34% (exp) vs 81.82% (native)"},
    {"UD-Q8_K_XL",  "Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf",  35.8,  ""},
};

static unsigned long long gb_to_bytes(double gb) {
    return (unsigned long long) (gb * 1024.0 * 1024.0 * 1024.0);
}

// ------------------------------------------------- modalita' streaming ------

// Variant "streaming" (ispirata a DS4 di antirez): i pesi restano su disco e
// arrivano via mmap; la cache del filesystem fa da staging in RAM e la VRAM
// tiene solo attenzione/KV (--cpu-moe + -ngl). Questo thread legge il file in
// sequenza continua per portare in cache le pagine dei tensore/layer che il
// modello tocchera' prossimi: il layout GGUF e' sequenziale per layer, quindi
// leggere avanti = anticipare i layer successivi. Gli esperti effettivamente
// calcolati restano solo quelli scelti dal router (mul_mat_id non tocca le
// righe degli altri) — la "selezione statistica" la fa gia' il routing del
// modello, qui anticipiamo il caricamento di cio' che sta' arrivando.
static void prefetch_model_loop(const std::string & path) {
    std::vector<char> buf(1 << 20);   // 1 MiB per lettura
    for (;;) {                        // ciclo: i layer si riattraversano a ogni token
#ifdef _WIN32
        HANDLE f = CreateFileA(path.c_str(), GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                               OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (f == INVALID_HANDLE_VALUE) return;
        DWORD rd = 0;
        while (ReadFile(f, buf.data(), (DWORD) buf.size(), &rd, nullptr) && rd > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));   // ~500 MB/s
        }
        CloseHandle(f);
#else
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) return;
        while (fread(buf.data(), 1, buf.size(), f) > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        fclose(f);
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// ----------------------------------------------- lettura metadati GGUF ------

// Legge gli header GGUF (solo sezione chiavi/valori, all'inizio del file) per
// capire se il modello e' un MoE e su quanti livelli: il profilo di espansione
// va iniettato solo sui MoE, perche' il fork rifiuta i modelli densi
// ("moe expert expansion: model is not MoE"). Host little-endian (x86/ARM LE).
static bool gguf_probe_moe(const std::string & path, int64_t & expert_count,
                           int64_t & expert_used, int64_t & block_count) {
    expert_count = 0; expert_used = 0; block_count = 0;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = false;
    do {
        auto rd_u32 = [&](uint32_t & v) { return fread(&v, 4, 1, f) == 1; };
        auto rd_u64 = [&](uint64_t & v) { return fread(&v, 8, 1, f) == 1; };
        // ritorna 1 = letto, 0 = eof/errore, -1 = tipo non scalare (va saltato a parte)
        auto rd_scalar = [&](int type, int64_t & out) -> int {
            uint8_t b8; uint16_t b16; uint32_t b32; uint64_t b64; float f32; double f64;
            switch (type) {
                case 0:  if (fread(&b8,  1, 1, f) != 1) return 0; out = b8;            return 1;
                case 1:  if (fread(&b8,  1, 1, f) != 1) return 0; out = (int8_t) b8;   return 1;
                case 2:  if (fread(&b16, 2, 1, f) != 1) return 0; out = b16;           return 1;
                case 3:  if (fread(&b16, 2, 1, f) != 1) return 0; out = (int16_t) b16; return 1;
                case 4:  if (fread(&b32, 4, 1, f) != 1) return 0; out = b32;           return 1;
                case 5:  if (fread(&b32, 4, 1, f) != 1) return 0; out = (int32_t) b32; return 1;
                case 6:  if (fread(&f32, 4, 1, f) != 1) return 0; out = (int64_t) f32; return 1;
                case 7:  if (fread(&b8,  1, 1, f) != 1) return 0; out = b8 ? 1 : 0;    return 1;
                case 10:
                case 11: if (fread(&b64, 8, 1, f) != 1) return 0; out = (int64_t) b64; return 1;
                case 12: if (fread(&f64, 8, 1, f) != 1) return 0; out = (int64_t) f64; return 1;
                default: return -1;    // 8 = string, 9 = array
            }
        };
        std::function<bool(int)> skip_value = [&](int type) -> bool {
            if (type == 8) {                                   // stringa
                uint64_t n;
                if (!rd_u64(n)) return false;
                return fseek(f, (long) n, SEEK_CUR) == 0;
            }
            if (type == 9) {                                   // array
                uint32_t t; uint64_t n;
                if (!rd_u32(t) || !rd_u64(n)) return false;
                for (uint64_t i = 0; i < n; ++i)
                    if (!skip_value((int) t)) return false;
                return true;
            }
            int64_t v;
            return rd_scalar(type, v) == 1;
        };
        uint32_t magic = 0, version = 0;
        uint64_t n_tensors = 0, n_kv = 0;
        if (!rd_u32(magic) || !rd_u32(version) || !rd_u64(n_tensors) || !rd_u64(n_kv)) break;
        if (magic != 0x46554747u) break;                       // "GGUF" little-endian
        for (uint64_t i = 0; i < n_kv; ++i) {
            uint64_t klen;
            if (!rd_u64(klen) || klen == 0 || klen > 4096) break;
            std::string key(klen, '\0');
            if (fread(&key[0], 1, klen, f) != klen) { klen = (uint64_t) -1; break; }
            uint32_t vt;
            if (!rd_u32(vt)) break;
            auto ends = [&](const char * suffix) {
                std::string s(suffix);
                return key.size() >= s.size() && key.compare(key.size() - s.size(), s.size(), s) == 0;
            };
            bool want_used   = ends(".expert_count_used");
            bool want_count  = !want_used && ends(".expert_count");
            bool want_blocks = ends(".block_count");
            if (want_used || want_count || want_blocks) {
                int64_t v = 0;
                int r = rd_scalar((int) vt, v);
                if (r == 0) break;
                if (r == -1) { if (!skip_value((int) vt)) break; continue; }
                if      (want_used)   expert_used  = v;
                else if (want_count)  expert_count = v;
                else                  block_count  = v;
            } else if (!skip_value((int) vt)) {
                break;
            }
        }
        ok = true;
    } while (false);
    fclose(f);
    return ok;
}

// ------------------------------------------------------------- modelli dir --

static std::vector<std::string> model_dirs(const std::string & extra_dir) {
    std::vector<std::string> dirs;
    if (!extra_dir.empty()) dirs.push_back(extra_dir);
    const char * sep;
#ifdef _WIN32
    const char * home = std::getenv("USERPROFILE");
    sep = ";";
#else
    const char * home = std::getenv("HOME");
    sep = ":";
#endif
    const char * env = std::getenv("AGRILLA_MODELS_DIR");
    if (env && *env) {
        std::string list(env), item;
        size_t pos = 0;
        while ((pos = list.find(sep)) != std::string::npos) {
            item = trim(list.substr(0, pos));
            if (!item.empty()) dirs.push_back(item);
            list.erase(0, pos + 1);
        }
        item = trim(list);
        if (!item.empty()) dirs.push_back(item);
    }
    if (home && *home) dirs.push_back(std::string(home) + "/models");
    dirs.push_back("models");
#ifdef _WIN32
    dirs.push_back("C:\\models");
#else
    dirs.push_back("/mnt/c/models");
#endif
    // dedup, mantieni l'ordine
    std::vector<std::string> out;
    for (auto & d : dirs) {
        std::error_code ec;
        auto abs = fs::absolute(fs::path(d), ec).string();
        if (std::find(out.begin(), out.end(), abs) == out.end()) out.push_back(abs);
    }
    return out;
}

struct local_model {
    std::string          path;
    std::string          name;
    unsigned long long   bytes;
};

static std::vector<local_model> scan_local_models(const std::vector<std::string> & dirs) {
    std::vector<local_model> out;
    for (const auto & d : dirs) {
        std::error_code ec;
        if (!fs::is_directory(d, ec)) continue;
        for (fs::directory_iterator it(d, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code ec2;
            auto p = it->path();
            if (!fs::is_regular_file(p, ec2) || to_lower(p.extension().string()) != ".gguf") continue;
            std::string n  = p.filename().string();
            std::string lo = to_lower(n);
            if (lo.find("qwen") == std::string::npos || lo.find("35b") == std::string::npos) continue;
            unsigned long long sz = fs::file_size(p, ec2);
            if (ec2) sz = 0;
            out.push_back({p.string(), n, sz});
        }
    }
    std::sort(out.begin(), out.end(), [](const local_model & a, const local_model & b) {
        return a.bytes < b.bytes;
    });
    return out;
}

// ---------------------------------------------------- proposta in base VRAM --

// budget caricabile in VRAM: ~92% della memoria (margine per KV cache e overhead)
static unsigned long long vram_budget(const gpu_info & gi) {
    if (gi.vram_mib <= 0) return 0;
    return (unsigned long long) gi.vram_mib * 1024ULL * 1024ULL * 92ULL / 100ULL;
}

// indice del quant consigliato: il piu' grande che entra in VRAM;
// se nessuno entra, il piu' piccolo (offload CPU parziale).
// vram non rilevata -> UD-Q3_K_XL (punto medio del catalogo).
static int suggest_catalog_index(const gpu_info & gi) {
    unsigned long long budget = vram_budget(gi);
    if (budget == 0) {
        for (size_t i = 0; i < k_catalog.size(); ++i)
            if (std::string(k_catalog[i].label) == "UD-Q3_K_XL") return (int) i;
        return 0;
    }
    int best = 0;
    for (size_t i = 0; i < k_catalog.size(); ++i) {
        if (gb_to_bytes(k_catalog[i].gb) <= budget) best = (int) i;
    }
    return best;
}

static int suggest_local_index(const std::vector<local_model> & locals, const gpu_info & gi) {
    unsigned long long budget = vram_budget(gi);
    if (budget == 0) return (int) locals.size() - 1;
    int best = 0;
    for (size_t i = 0; i < locals.size(); ++i)
        if (locals[i].bytes <= budget) best = (int) i;
    return best;
}

// ---------------------------------------------------------------- download --

static bool command_exists(const char * cmd) {
#ifdef _WIN32
    std::string c = std::string("where ") + cmd + " >nul 2>nul";
#else
    std::string c = std::string("command -v ") + cmd + " >/dev/null 2>&1";
#endif
    return std::system(c.c_str()) == 0;
}

static std::string download_dir(const std::vector<std::string> & dirs) {
    for (const auto & d : dirs) {
        std::error_code ec;
        if (fs::is_directory(d, ec)) return d;
    }
    // crea la prima tra $HOME/models
    for (const auto & d : dirs) {
        if (d.find("models") == std::string::npos) continue;
        std::error_code ec;
        if (fs::create_directories(d, ec) || fs::is_directory(d, ec)) return d;
    }
    return ".";
}

static std::string download_model(const quant_entry & q, const std::string & dir, bool assume_yes) {
    std::string target = (fs::path(dir) / q.file).string();
    std::error_code ec;
    if (fs::exists(target, ec)) {
        printf("[AgrillaMoE] %s e' gia' presente in cache: %s\n", q.file, target.c_str());
        return target;
    }
    if (!assume_yes) {
        printf("Scarico %s (~%.1f GB) da %s in\n  %s ? [s/N] ", q.file, q.gb, AGRILLA_HF_REPO, dir.c_str());
        fflush(stdout);
        char buf[64];
        if (!fgets(buf, sizeof buf, stdin)) return "";
        std::string a = to_lower(trim(buf));
        if (a != "s" && a != "si" && a != "y" && a != "yes") return "";
    }
    const char * cli = command_exists("hf") ? "hf" : (command_exists("huggingface-cli") ? "huggingface-cli" : nullptr);
    if (!cli) {
        printf("[AgrillaMoE] ERRORE: nessuna CLI huggingface trovata (hf / huggingface-cli).\n"
               "  Installala con:  pip install -U huggingface_hub\n"
               "  oppure scarica a mano:\n"
               "  hf download %s \"%s\" --local-dir \"%s\"\n", AGRILLA_HF_REPO, q.file, dir.c_str());
        return "";
    }
    printf("[AgrillaMoE] download in corso (%s download %s \"%s\" --local-dir \"%s\")...\n",
           cli, AGRILLA_HF_REPO, q.file, dir.c_str());
    fflush(stdout);
    std::string cmd = std::string(cli) + " download " + AGRILLA_HF_REPO + " \"" + q.file +
                      "\" --local-dir \"" + dir + "\"";
    if (std::system(cmd.c_str()) != 0) {
        printf("[AgrillaMoE] ERRORE: download non riuscito.\n");
        return "";
    }
    if (!fs::exists(target, ec)) {
        printf("[AgrillaMoE] ERRORE: il file %s non risulta presente dopo il download.\n", target.c_str());
        return "";
    }
    printf("[AgrillaMoE] download completato: %s\n", target.c_str());
    return target;
}

// ------------------------------------------------------------------- input --

static std::string prompt_line(const char * prompt) {
    fputs(prompt, stdout);
    fflush(stdout);
    char buf[512];
    if (!fgets(buf, sizeof buf, stdin)) return "";
    return trim(buf);
}

static int parse_choice(const std::string & s, int lo, int hi) {
    if (s.empty()) return -1;
    char * end = nullptr;
    long v = strtol(s.c_str(), &end, 10);
    if (!end || *end != '\0') return -2;              // non numerico
    if (v < lo || v > hi) return -1;
    return (int) v;
}

// ------------------------------------------------------------ selezione ------

struct selection_result {
    std::string path;   // "" = annullato
    bool        cancelled = false;
};

static void print_header(const gpu_info & gi) {
    printf("%s", AGRILLA_BANNER);
    printf("versione %s | server dedicato Qwen3.6-35B-A3B (GGUF Unsloth: %s)\n\n", AGRILLA_VERSION, AGRILLA_HF_REPO);
    if (gi.vram_mib > 0)
        printf("GPU rilevata: %s, %ld MiB VRAM (budget modellino consigliato <= %s)\n",
               gi.name.c_str(), gi.vram_mib, human_gb(vram_budget(gi)).c_str());
    else
        printf("GPU non rilevata (nvidia-smi assente): proposta di default UD-Q3_K_XL\n");
    printf("Profilo MoE-expansion di default (benchmark RUN1209, Q8_0): esperti 16, soglia 0.80, livelli 25-39, decay 0.50, renorm auto\n");
    printf("Contesto di default 142768 (~140k) su 4 slot (35840/slot); temperatura, reasoning e gli altri parametri\nsi passano con i flag di llama-server (--temp, --reasoning-budget N, --reasoning off, --top-p, ...)\n");
    printf("GPU con poca VRAM? Avvia con --agrilla-streaming: esperti streamati da disco via mmap (solo quelli\ninstradati dal router vengono calcolati), attenzione/KV su GPU e prefetch dei layer successivi in RAM\n\n");
    fflush(stdout);
}

static selection_result select_model(const std::string & extra_dir, bool assume_yes) {
    selection_result res;
    auto dirs   = model_dirs(extra_dir);
    auto locals = scan_local_models(dirs);
    gpu_info gi = detect_gpu();
    print_header(gi);

    bool interactive = agrilla_isatty(agrilla_fileno(stdin)) != 0;

    printf("Cartelle modelli cercate:\n");
    for (const auto & d : dirs) printf("  - %s\n", d.c_str());
    printf("\n");

    // ---- nessun modello locale: proposta in base alla VRAM + download ----
    if (locals.empty()) {
        int sug = suggest_catalog_index(gi);
        printf("Nessun modello Qwen3.6-35B-A3B gia' scaricato.\n");
        printf("Quant disponibili su %s:\n\n", AGRILLA_HF_REPO);
        for (size_t i = 0; i < k_catalog.size(); ++i) {
            const auto & q = k_catalog[i];
            std::string tag = (int) i == sug ? "  <== consigliato per la tua VRAM" : "";
            std::string warn;
            if ((int) i == sug && gi.vram_mib > 0 && gb_to_bytes(q.gb) > vram_budget(gi))
                warn = "  [non entra in VRAM: offload CPU parziale]";
            printf("  %2zu) %-14s ~%5.1f GB%s%s\n", i + 1, q.label, q.gb, tag.c_str(), warn.c_str());
            if (q.note[0]) printf("        %s\n", q.note);
        }
        printf("\n");
        if (!interactive) {
            const auto & q = k_catalog[sug];
            printf("[AgrillaMoE] modalita' non interattiva: nessun modello locale e download automatico non confermabile.\n"
                   "Consigliato per la tua GPU: %s (%s, ~%.1f GB).\n"
                   "Avvia in un terminale interattivo, oppure scaricalo con:\n"
                   "  hf download %s \"%s\" --local-dir <dir>\n"
                   "e riavvia, oppure passa -m <percorso/*.gguf>.\n",
                   q.label, q.file, q.gb, AGRILLA_HF_REPO, q.file);
            res.cancelled = false;   // errore, non annullamento utente: exit code 1
            return res;
        }
        while (true) {
            std::string in = prompt_line("Quale quant scarico? [invio = consigliato, x = annulla]: ");
            if (in == "x" || in == "X") { res.cancelled = true; return res; }
            int c = parse_choice(in, 1, (int) k_catalog.size());
            if (c == -1 && in.empty()) c = sug + 1;               // invio -> consigliato
            if (c >= 1) {
                res.path = download_model(k_catalog[c - 1], download_dir(dirs), assume_yes);
                if (!res.path.empty()) return res;
                printf("\n");
                continue;
            }
            printf("Scelta non valida.\n");
        }
    }

    // ---- modelli locali presenti: sceglie l'utente ----
    int sug = suggest_local_index(locals, gi);
    printf("Modelli Qwen3.6-35B-A3B gia' scaricati:\n\n");
    for (size_t i = 0; i < locals.size(); ++i) {
        const auto & m = locals[i];
        std::string tag  = (int) i == sug ? "  <== consigliato per la tua VRAM" : "";
        std::string warn = m.bytes > vram_budget(gi) && gi.vram_mib > 0 ? "  [supera la VRAM: offload CPU parziale]" : "";
        printf("  %2zu) %s  (%s)%s%s\n", i + 1, m.name.c_str(), human_gb(m.bytes).c_str(), tag.c_str(), warn.c_str());
    }
    printf("\n");
    if (!interactive) {
        // scelta automatica non interattiva: best fit
        res.path = locals[sug].path;
        printf("[AgrillaMoE] modalita' non interattiva: uso il best-fit per la VRAM -> %s\n", res.path.c_str());
        return res;
    }
    while (true) {
        std::string in = prompt_line("Quale modello avvio? [invio = consigliato, d = scarica altro quant, x = annulla]: ");
        if (in == "x" || in == "X") { res.cancelled = true; return res; }
        if (in == "d" || in == "D") {
            int sugc = suggest_catalog_index(gi);
            printf("\nQuant disponibili su %s:\n\n", AGRILLA_HF_REPO);
            for (size_t i = 0; i < k_catalog.size(); ++i) {
                const auto & q = k_catalog[i];
                std::string tag = (int) i == sugc ? "  <== consigliato per la tua VRAM" : "";
                printf("  %2zu) %-14s ~%5.1f GB%s\n", i + 1, q.label, q.gb, tag.c_str());
                if (q.note[0]) printf("        %s\n", q.note);
            }
            printf("\n");
            std::string in2 = prompt_line("Quale quant scarico? [invio = consigliato, x = torna indietro]: ");
            if (in2 == "x" || in2 == "X" || in2.empty()) { printf("\n"); continue; }
            int c = parse_choice(in2, 1, (int) k_catalog.size());
            if (c >= 1) {
                res.path = download_model(k_catalog[c - 1], download_dir(dirs), assume_yes);
                if (!res.path.empty()) return res;
                printf("\n");
            }
            continue;
        }
        int c = parse_choice(in, 1, (int) locals.size());
        if (c == -1 && in.empty()) c = sug + 1;
        if (c >= 1) {
            res.path = locals[c - 1].path;
            return res;
        }
        printf("Scelta non valida.\n");
    }
}

// ----------------------------------------------------------------- browser --

static bool tcp_probe(const std::string & host, int port) {
#ifdef _WIN32
    static bool wsa_ok = []() {
        WSADATA d;
        return WSAStartup(MAKEWORD(2, 2), &d) == 0;
    }();
    if (!wsa_ok) return false;
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return false;
    BOOL ok = false;
#else
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return false;
    bool ok = false;
#endif
    sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t) port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) == 1) {
        ok = (connect(s, (sockaddr *) &addr, sizeof addr) == 0);
    }
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
    return ok;
}

static void open_browser(const std::string & url) {
#ifdef _WIN32
    ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
    // in WSL wslview apre il browser Windows; altrimenti i soliti launcher Linux
    for (const char * cmd : {"wslview", "xdg-open", "sensible-browser"}) {
        std::string probe = std::string("command -v ") + cmd + " >/dev/null 2>&1";
        if (std::system(probe.c_str()) == 0) {
            std::string run = std::string(cmd) + " '" + url + "' >/dev/null 2>&1 &";
            std::system(run.c_str());
            return;
        }
    }
#endif
}

static void wait_server_and_open(std::string host, int port) {
    if (host == "0.0.0.0" || host == "::" || host.empty()) host = "127.0.0.1";
    std::string url = "http://" + host + ":" + std::to_string(port);
    // attesa fino a 30 minuti (il caricamento di un 35B puo' essere lento)
    for (int i = 0; i < 60 * 30 * 2; ++i) {
        if (tcp_probe(host, port)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            open_browser(url);
            printf("[AgrillaMoE] browser aperto su %s\n", url.c_str());
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    printf("[AgrillaMoE] timeout attesa server, browser non aperto (%s)\n", url.c_str());
}

} // namespace agrilla

// -------------------------------------------------------------------- main --

int main(int argc, char ** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif
    using namespace agrilla;

    const bool opt_browser_on = !env_flag_on("AGRILLA_NO_BROWSER") &&
                                !have_flag(argc, argv, {"--no-browser", "--agrilla-no-browser"});
    const bool opt_moe_on     = !env_flag_on("AGRILLA_NO_MOE_EXPANSION") &&
                                !have_flag(argc, argv, {"--no-moe-expansion", "--agrilla-no-moe-expansion"});
    const bool opt_assume_yes = env_flag_on("AGRILLA_YES") ||
                                have_flag(argc, argv, {"--agrilla-yes"});
    const bool opt_streaming  = env_flag_on("AGRILLA_STREAMING") ||
                                have_flag(argc, argv, {"--agrilla-streaming"});

    // flag custom AgrillaMoE: consumati qui, tolti da quanto passato a llama_server
    std::string extra_dir = get_flag_value(argc, argv, {"--agrilla-models-dir"}, "");
    bool list_only        = have_flag(argc, argv, {"--agrilla-list-models"});

    std::vector<std::string> clean;
    for (int i = 1; i < argc; ++i) {
        std::string t = argv[i];
        if (t == "--no-browser" || t == "--agrilla-no-browser" ||
            t == "--no-moe-expansion" || t == "--agrilla-no-moe-expansion" ||
            t == "--agrilla-yes" || t == "--agrilla-list-models" ||
            t == "--agrilla-streaming") {
            continue;
        }
        if (t == "--agrilla-models-dir") { ++i; continue; }   // salta anche il valore
        clean.push_back(t);
    }

    if (list_only) {
        gpu_info gi = detect_gpu();
        print_header(gi);
        auto dirs   = model_dirs(extra_dir);
        auto locals = scan_local_models(dirs);
        printf("Modelli locali:\n");
        for (size_t i = 0; i < locals.size(); ++i)
            printf("  %2zu) %s  (%s)\n", i + 1, locals[i].path.c_str(), human_gb(locals[i].bytes).c_str());
        if (locals.empty()) printf("  (nessuno)\n");
        int sug = suggest_catalog_index(gi);
        printf("\nConsigliato per la VRAM rilevata: %s (~%.1f GB)\n", k_catalog[sug].label, k_catalog[sug].gb);
        return 0;
    }

    // fast path: --help/--version passano dritti al server, senza menu ne' default
    if (have_flag(argc, argv, {"-h", "--help", "--version"})) {
        std::vector<char *> hargv;
        hargv.push_back(argv[0]);
        for (auto & s : clean) hargv.push_back(const_cast<char *>(s.c_str()));
        return llama_server((int) hargv.size(), hargv.data());
    }

    // ---- modello: -m vince; altrimenti selezione interattiva ----
    std::string selected_model;
    if (!have_flag(argc, argv, {"-m", "--model"})) {
        selection_result sel = select_model(extra_dir, opt_assume_yes);
        if (sel.cancelled || sel.path.empty()) {
            printf("[AgrillaMoE] nessun modello selezionato, esco.\n");
            return sel.cancelled ? 0 : 1;
        }
        selected_model = sel.path;
        clean.push_back("-m");
        clean.push_back(sel.path);
    }

    // ---- profilo MoE-expansion del benchmark RUN1209 (Q8_0) ----
    // Iniettato solo se l'utente non ha passato flag moe-* / q35-* e solo se
    // il modello e' davvero un MoE: il fork rifiuta i modelli densi
    // ("moe expert expansion: model is not MoE"), e con top-K nativo >= 16
    // il profilo lo poterebbe potare invece di espandere.
    std::string mp = get_flag_value(argc, argv, {"-m", "--model"}, "");
    if (mp.empty()) mp = selected_model;
    std::string moe_summary = "configurazione utente";
    if (!opt_moe_on) {
        moe_summary = "disattivata (--no-moe-expansion)";
    } else if (!have_flag(argc, argv, {
            "--moe-experts", "--q35-experts", "--moe-experts-add",
            "--moe-expert-threshold", "--q35-expert-threshold",
            "--moe-expert-decay-end", "--moe-no-expert-decay", "--q35-no-expert-decay",
            "--moe-expert-renorm", "--moe-expert-layer-start", "--moe-expert-layer-end"})) {
        int64_t ec = 0, eu = 0, bc = 0;
        bool readable = !mp.empty() && gguf_probe_moe(mp, ec, eu, bc);
        if (!readable) {
            printf("[AgrillaMoE] metadati GGUF non leggibili (%s): MoE-expansion non iniettata\n", mp.c_str());
            moe_summary = "non iniettata (GGUF non leggibile)";
        } else if (ec <= 0) {
            printf("[AgrillaMoE] modello non-MoE (expert_count assente): MoE-expansion non iniettata\n");
            moe_summary = "non iniettata (modello non-MoE)";
        } else if (eu >= 16) {
            printf("[AgrillaMoE] MoE con top-K nativo %lld >= 16: il profilo RUN1209 lo poterebbe potare, non iniettato\n",
                   (long long) eu);
            moe_summary = "non iniettata (top-K nativo >= 16)";
        } else if (bc == 40) {
            // firma di Qwen3.6-35B-A3B: 40 livelli -> profilo completo del benchmark
            for (const char * a : {"--moe-experts", "16",
                                   "--moe-expert-threshold", "0.8",
                                   "--moe-expert-layer-start", "25",
                                   "--moe-expert-layer-end", "39"}) {
                clean.push_back(a);
            }
            moe_summary = "esperti 16, soglia 0.80, livelli 25-39 (RUN1209, 40 livelli)";
        } else {
            // altro MoE: espansione base senza il range di livelli calibrato sul 35B
            for (const char * a : {"--moe-experts", "16",
                                   "--moe-expert-threshold", "0.8"}) {
                clean.push_back(a);
            }
            moe_summary = "esperti 16, soglia 0.80 (MoE a " + std::to_string(bc) +
                          " livelli: range 25-39 del 35B non applicato)";
        }
    }

    // ---- modalita' streaming (stile DS4): pesi esperti via mmap da disco,
    //      attenzione/KV su GPU, prefetch sequenziale dei layer in RAM ----
    if (opt_streaming) {
        const std::initializer_list<const char *> moe_placement = {
            "-cmoe", "--cpu-moe", "-ncmoe", "--n-cpu-moe", "-ot", "--override-tensor"};
        if (!have_flag(argc, argv, moe_placement) && !args_has(clean, moe_placement)) {
            clean.push_back("--cpu-moe");
        }
        const std::initializer_list<const char *> ngl_names = {"-ngl", "--gpu-layers", "--n-gpu-layers"};
        if (!have_flag(argc, argv, ngl_names) && !args_has(clean, ngl_names)) {
            clean.push_back("-ngl"); clean.push_back("99");
        }
        if (!mp.empty()) {
            printf("[AgrillaMoE] streaming: esperti su CPU via mmap da disco (calcolati solo quelli instradati),\n"
                   "                     attenzione/KV su GPU, prefetch sequenziale dei layer in RAM attivo\n");
            std::thread(prefetch_model_loop, mp).detach();
        }
    }

    // ---- reasoning budget (flag nativo --reasoning-budget N del fork):
    //      -1 illimitato (default), 0 chiude subito il pensiero, N>0 budget in token.
    //      Iniettabile anche con AGRILLA_REASONING_BUDGET se l'utente non lo passa.
    if (!have_flag(argc, argv, {"--reasoning-budget"})) {
        const char * rb = std::getenv("AGRILLA_REASONING_BUDGET");
        if (rb && *rb) {
            std::string v = trim(rb);
            bool valid = !v.empty() && (v == "-1" || v == "0" ||
                         (v[0] != '-' && v.find_first_not_of("0123456789") == std::string::npos));
            if (valid) { clean.push_back("--reasoning-budget"); clean.push_back(v); }
            else printf("[AgrillaMoE] AGRILLA_REASONING_BUDGET ignorata (valore non valido: '%s')\n", v.c_str());
        }
    }

    // ---- stato reasoning (--reasoning on|off|auto del fork; 'off' disabilita
    //      del tutto il pensiero). Iniettabile anche con AGRILLA_REASONING.
    if (!have_flag(argc, argv, {"--reasoning", "-rea"})) {
        const char * rs = std::getenv("AGRILLA_REASONING");
        if (rs && *rs) {
            std::string v = to_lower(trim(rs));
            if (v == "on" || v == "off" || v == "auto") { clean.push_back("--reasoning"); clean.push_back(v); }
            else printf("[AgrillaMoE] AGRILLA_REASONING ignorata (valori validi: on|off|auto)\n");
        }
    }

    // ---- template chat Qwen (usato in tutti i benchmark) ----
    if (!have_flag(argc, argv, {"--jinja", "--no-jinja"})) clean.push_back("--jinja");

    // ---- contesto e concorrenza di default come nei benchmark RUN1209/Q2:
    //      -c 142768 (~140k) con --parallel 4 -> 35840 token per slot.
    //      fit_params del fork riduce automaticamente il contesto se la VRAM
    //      non basta, quindi e' sicuro anche su GPU piccole.
    if (!have_flag(argc, argv, {"-c", "--ctx-size"}) && !args_has(clean, {"-c", "--ctx-size"})) {
        clean.push_back("-c");
        clean.push_back(opt_streaming ? "8192" : "142768");
    }
    if (!have_flag(argc, argv, {"-np", "--parallel"}) && !args_has(clean, {"-np", "--parallel"})) {
        clean.push_back("--parallel");
        clean.push_back(opt_streaming ? "1" : "4");
    }

    // ---- bind di default 127.0.0.1:8071 ----
    if (!have_flag(argc, argv, {"--host"})) { clean.push_back("--host"); clean.push_back(AGRILLA_DEFAULT_HOST); }
    if (!have_flag(argc, argv, {"--port"})) { clean.push_back("--port"); clean.push_back(AGRILLA_DEFAULT_PORT); }

    // ---- argv finale per llama_server ----
    std::vector<std::string> storage;
    storage.reserve(clean.size() + 1);
    storage.push_back(argv[0]);
    for (auto & s : clean) storage.push_back(s);
    std::vector<char *> fargv;
    fargv.reserve(storage.size());
    for (auto & s : storage) fargv.push_back(const_cast<char *>(s.c_str()));

    // host/port/contesto/concorrenza effettivi (dell'utente o i nostri) per il riepilogo
    std::string bhost = AGRILLA_DEFAULT_HOST;
    std::string bport = AGRILLA_DEFAULT_PORT;
    std::string bctx  = "142768";
    std::string bnp   = "4";
    std::string bbudget  = "-1";
    std::string breason  = "auto";
    for (size_t i = 1; i < storage.size(); ++i) {
        std::string t = storage[i];
        auto eat = [&](const char * name, std::string & out) {
            std::string nm(name);
            if (t == nm && i + 1 < storage.size()) { out = storage[i + 1]; return true; }
            if (t.rfind(nm + "=", 0) == 0) { out = t.substr(nm.size() + 1); return true; }
            return false;
        };
        if (eat("--host", bhost)) continue;
        if (eat("--port", bport)) continue;
        if (eat("-c", bctx)) continue;
        if (eat("--ctx-size", bctx)) continue;
        if (eat("-np", bnp)) continue;
        if (eat("--parallel", bnp)) continue;
        if (eat("--reasoning-budget", bbudget)) continue;
        if (eat("--reasoning", breason) || eat("-rea", breason)) { breason = to_lower(breason); continue; }
    }

    printf("[AgrillaMoE] avvio llama-server con:\n");
    printf("[AgrillaMoE]   MoE expansion : %s\n", moe_summary.c_str());
    {
        long long ctx_ll = atoll(bctx.c_str());
        long long np_ll  = atoll(bnp.c_str());
        printf("[AgrillaMoE]   contesto      : %s token, %s slot da ~%lld (ridotto in automatico se la VRAM non basta)\n",
               bctx.c_str(), bnp.c_str(), np_ll > 0 ? ctx_ll / np_ll : ctx_ll);
    }
    {
        long long rb = atoll(bbudget.c_str());
        if (breason == "off") {
            printf("[AgrillaMoE]   reasoning     : disattivato (--reasoning off)\n");
        } else {
            std::string stato = (breason == "on") ? "attivo (on)" : "attivo (auto: da template)";
            std::string bud   = (rb < 0)  ? "budget illimitato" :
                                (rb == 0) ? "budget 0 (pensiero chiuso subito)" :
                                            "budget " + bbudget + " token";
            printf("[AgrillaMoE]   reasoning     : %s, %s (--reasoning off per disabilitare il pensiero)\n",
                   stato.c_str(), bud.c_str());
        }
    }
    printf("[AgrillaMoE]   endpoint      : http://%s:%s\n", bhost.c_str(), bport.c_str());
    if (opt_streaming) {
        printf("[AgrillaMoE]   streaming     : attivo (--cpu-moe + mmap da disco, attenzione/KV su GPU, prefetch layer in RAM)\n");
    }
    printf("[AgrillaMoE]   browser       : %s\n", opt_browser_on ? "apertura automatica al ready" : "disattivato");
    printf("\n");
    fflush(stdout);

    if (opt_browser_on) {
        std::thread(wait_server_and_open, bhost, atoi(bport.c_str())).detach();
    }

    return llama_server((int) fargv.size(), fargv.data());
}
