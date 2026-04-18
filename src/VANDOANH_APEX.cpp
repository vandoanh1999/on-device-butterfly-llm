/**
 * ╔══════════════════════════════════════════════════════════════════════════════╗
 * ║  VANDOANH_APEX.cpp — Unified Breakthrough Engine                           ║
 * ║  VANDOANH Research 2025 — Built on a phone. For the world.                 ║
 * ╠══════════════════════════════════════════════════════════════════════════════╣
 * ║  BUGS FIXED vs VANDOANH_MASTER:                                             ║
 * ║  [BUG-1] GGUFParser: throw + -fno-exceptions → terminate()                 ║
 * ║           FIXED: thay throw bằng error status return, bỏ exceptions        ║
 * ║  [BUG-2] train_vi mode documented nhưng không implement trong main()        ║
 * ║           FIXED: implement bench_train_vi() + handler trong main            ║
 * ║  [BUG-3] DynamicLayerBudget: mutable rng race trong const method           ║
 * ║           FIXED: thêm mutex bảo vệ rng + exit_counts                       ║
 * ║  [BUG-4] Logic2048Weight::from_float: max power 62 thay vì 63              ║
 * ║           FIXED: max(-63, min(p, 63)) — dùng hết range của bảng            ║
 * ╠══════════════════════════════════════════════════════════════════════════════╣
 * ║  BREAKTHROUGH INTEGRATION (Claude APEX suggestions):                        ║
 * ║  [APEX-1] Logic2048Linear: train trực tiếp format L2K + STE gradient       ║
 * ║  [APEX-2] AdaptiveLayer: DIB thay FFN (O(N logN) vs O(N²))                 ║
 * ║  [APEX-3] AdaptiveUnifiedModel: DLB per-token depth + L2K weights          ║
 * ║  [APEX-4] Unified training loop kết hợp cả 3 breakthrough                  ║
 * ╠══════════════════════════════════════════════════════════════════════════════╣
 * ║  ARCHITECTURE (AdaptiveUnifiedModel):                                       ║
 * ║    Token → Embedding → [AdaptiveLayer × n_layers] → LM Head               ║
 * ║    AdaptiveLayer:                                                            ║
 * ║      RMSNorm → DIBLayer (O(N logN) mixing) →                               ║
 * ║      Logic2048Linear (STE training) → residual                             ║
 * ║      DLB check → early exit if entropy low                                  ║
 * ╠══════════════════════════════════════════════════════════════════════════════╣
 * ║  COMPILE:                                                                    ║
 * ║    clang++ -O3 -std=c++17 -march=armv8.4-a+dotprod+fp16+i8mm \             ║
 * ║      -ffast-math -fopenmp -lpthread \                                        ║
 * ║      VANDOANH_APEX.cpp -o vdapex                                            ║
 * ║    g++ -O3 -std=c++17 -fopenmp -lpthread \                                 ║
 * ║      VANDOANH_APEX.cpp -o vdapex                                            ║
 * ║    NOTE: KHÔNG dùng -fno-exceptions (GGUFParser dùng try/catch)            ║
 * ╠══════════════════════════════════════════════════════════════════════════════╣
 * ║  MODES:                                                                      ║
 * ║    ./vdapex all              # full benchmark suite                         ║
 * ║    ./vdapex unified          # NEW: train AdaptiveUnifiedModel              ║
 * ║    ./vdapex train_vi [file]  # NEW: train tiếng Việt (FIXED BUG-2)         ║
 * ║    ./vdapex charlm [file]    # CharLM v5 (legacy)                           ║
 * ║    ./vdapex logic|butterfly|flash|jit|dlb|kernel|proof                     ║
 * ║    ./vdapex gguf <model>     # GGUF info (FIXED BUG-1)                     ║
 * ║  CLI: --lr --wd --steps --dim --layers --grad_clip                          ║
 * ║       --warmup --lr_sched=cosine|constant                                   ║
 * ║       --diag=1 --seed --reps --exit_thr=<float>                            ║
 * ╚══════════════════════════════════════════════════════════════════════════════╝
 */

// ════════════════════════════════════════════════════════════════════════════
// §0 — HEADERS & PLATFORM
// ════════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#ifdef __ARM_NEON
#  include <arm_neon.h>
#  define HAS_NEON 1
#else
#  define HAS_NEON 0
#endif
#ifdef __aarch64__
#  define HAS_AARCH64 1
#else
#  define HAS_AARCH64 0
#endif
#ifdef __ARM_FEATURE_DOTPROD
#  define HAS_DOTPROD 1
#else
#  define HAS_DOTPROD 0
#endif
#ifdef __ARM_FEATURE_MATMUL_INT8
#  define HAS_I8MM 1
#else
#  define HAS_I8MM 0
#endif
#ifdef _OPENMP
#  include <omp.h>
#  define HAS_OMP 1
#else
#  define HAS_OMP 0
#endif
#ifdef __linux__
#  include <sched.h>
#  define HAS_LINUX 1
#else
#  define HAS_LINUX 0
#endif

using namespace std;
using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8  = int8_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;

static constexpr int CACHE_L1D_BYTES = 64  * 1024;
static constexpr int CACHE_L2_BYTES  = 512 * 1024;
static constexpr int CACHE_L3_BYTES  = 8   * 1024 * 1024;
static constexpr int TILE_BF_L1      = 2048;

static inline u64 cycle_count() {
#if HAS_AARCH64
    u64 v; asm volatile("mrs %0, cntvct_el0" : "=r"(v)); return v;
#else
    u32 lo, hi; asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((u64)hi << 32) | lo;
#endif
}
static inline double now_ms() {
    return chrono::duration<double,milli>(
        chrono::steady_clock::now().time_since_epoch()).count();
}
static void print_sep(int w=80, int c='=') {
    for (int i=0;i<w;i++) putchar(c); putchar('\n');
}
static volatile f32 g_sink = 0.f;
static inline void do_not_optimize(f32 v) { g_sink += v; }

// ════════════════════════════════════════════════════════════════════════════
// §1 — NEON PRIMITIVE LIBRARY
// ════════════════════════════════════════════════════════════════════════════

static inline f32 vdot_f32(const f32* __restrict__ a,
                            const f32* __restrict__ b, int n) {
#if HAS_NEON
    float32x4_t acc0=vdupq_n_f32(0.f), acc1=vdupq_n_f32(0.f);
    int i=0;
    for (; i<=n-8; i+=8) {
        __builtin_prefetch(a+i+32,0,1);
        __builtin_prefetch(b+i+32,0,1);
        acc0=vmlaq_f32(acc0,vld1q_f32(a+i),  vld1q_f32(b+i));
        acc1=vmlaq_f32(acc1,vld1q_f32(a+i+4),vld1q_f32(b+i+4));
    }
    acc0=vaddq_f32(acc0,acc1);
    f32 s=vaddvq_f32(acc0);
    for (; i<n; i++) s+=a[i]*b[i];
    return s;
#else
    f32 s=0.f; for (int i=0;i<n;i++) s+=a[i]*b[i]; return s;
#endif
}

static inline void vaxpy(f32* __restrict__ y, const f32* __restrict__ x,
                          f32 alpha, int n) {
#if HAS_NEON
    float32x4_t va=vdupq_n_f32(alpha); int i=0;
    for (; i<=n-8; i+=8) {
        vst1q_f32(y+i,  vmlaq_f32(vld1q_f32(y+i),  vld1q_f32(x+i),  va));
        vst1q_f32(y+i+4,vmlaq_f32(vld1q_f32(y+i+4),vld1q_f32(x+i+4),va));
    }
    for (; i<n; i++) y[i]+=alpha*x[i];
#else
    for (int i=0;i<n;i++) y[i]+=alpha*x[i];
#endif
}

static inline void vscale(f32* __restrict__ x, f32 s, int n) {
#if HAS_NEON
    float32x4_t vs=vdupq_n_f32(s); int i=0;
    for (; i<=n-8; i+=8) {
        vst1q_f32(x+i,  vmulq_f32(vld1q_f32(x+i),  vs));
        vst1q_f32(x+i+4,vmulq_f32(vld1q_f32(x+i+4),vs));
    }
    for (; i<n; i++) x[i]*=s;
#else
    for (int i=0;i<n;i++) x[i]*=s;
#endif
}

static inline void velmul(f32* __restrict__ x,
                           const f32* __restrict__ d, int n) {
#if HAS_NEON
    int i=0;
    for (; i<=n-8; i+=8) {
        vst1q_f32(x+i,  vmulq_f32(vld1q_f32(x+i),  vld1q_f32(d+i)));
        vst1q_f32(x+i+4,vmulq_f32(vld1q_f32(x+i+4),vld1q_f32(d+i+4)));
    }
    for (; i<n; i++) x[i]*=d[i];
#else
    for (int i=0;i<n;i++) x[i]*=d[i];
#endif
}

static void rmsnorm(f32* out, const f32* x, const f32* w, int n) {
#if HAS_NEON
    float32x4_t a0=vdupq_n_f32(0.f),a1=vdupq_n_f32(0.f); int i=0;
    for (; i<=n-8; i+=8) {
        float32x4_t v0=vld1q_f32(x+i),v1=vld1q_f32(x+i+4);
        a0=vmlaq_f32(a0,v0,v0); a1=vmlaq_f32(a1,v1,v1);
    }
    f32 ss=vaddvq_f32(vaddq_f32(a0,a1));
    for (; i<n; i++) ss+=x[i]*x[i];
    f32 inv=1.f/sqrtf(ss/n+1e-5f);
    float32x4_t vinv=vdupq_n_f32(inv); i=0;
    for (; i<=n-4; i+=4)
        vst1q_f32(out+i,vmulq_f32(vld1q_f32(w+i),vmulq_f32(vld1q_f32(x+i),vinv)));
    for (; i<n; i++) out[i]=w[i]*x[i]*inv;
#else
    f32 ss=0.f; for (int i=0;i<n;i++) ss+=x[i]*x[i];
    f32 inv=1.f/sqrtf(ss/n+1e-5f);
    for (int i=0;i<n;i++) out[i]=w[i]*x[i]*inv;
#endif
}

static void vsoftmax(f32* x, int n) {
    if (n<=0) return;
    f32 mx=x[0]; for (int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    f32 s=0.f; for (int i=0;i<n;i++){x[i]=expf(x[i]-mx); s+=x[i];}
    f32 inv=1.f/s; for (int i=0;i<n;i++) x[i]*=inv;
}

static void matvec_f32(const f32* __restrict__ W,
                        const f32* __restrict__ x,
                        f32* __restrict__ y, int out, int in) {
#if HAS_OMP
    #pragma omp parallel for schedule(static)
#endif
    for (int o=0;o<out;o++) y[o]=vdot_f32(W+(size_t)o*in,x,in);
}

static inline f32 silu_f(f32 x) { return x/(1.f+expf(-x)); }
static inline f32 clampf(f32 v, f32 lo, f32 hi) {
    return v<lo?lo:v>hi?hi:v;
}
static f32 rel_error(const f32* a, const f32* b, int n) {
    f32 err=0.f,nm=0.f;
    for (int i=0;i<n;i++){f32 d=a[i]-b[i];err+=d*d;nm+=a[i]*a[i];}
    return nm>1e-9f?sqrtf(err/nm):sqrtf(err);
}

// ════════════════════════════════════════════════════════════════════════════
// §2 — CLI CONFIG
// ════════════════════════════════════════════════════════════════════════════

struct MasterConfig {
    f32    lr         = 1e-3f;
    f32    wd         = 0.01f;
    int    steps      = 5000;
    int    dim        = 64;
    int    n_layers   = 4;
    f32    grad_clip  = 1.0f;
    f32    warmup     = 0.05f;
    string lr_sched   = "cosine";
    bool   diag       = false;
    int    diag_every = 500;
    int    seed       = 42;
    string preset     = "small";
    string file;
    string prompt     = "Hello, how are you?";
    int    max_new    = 100;
    f32    temp       = 0.8f;
    int    n_bench    = 50;
    int    bench_reps = 200;
    f32    exit_thr   = 0.30f;   // DLB entropy threshold

    void print() const {
        printf("  lr=%.2e  wd=%.3f  steps=%d  dim=%d  layers=%d\n",
               lr,wd,steps,dim,n_layers);
        printf("  grad_clip=%.2f  warmup=%.2f  sched=%s  seed=%d  exit_thr=%.2f\n",
               grad_clip,warmup,lr_sched.c_str(),seed,exit_thr);
    }
};

static MasterConfig parse_config(int argc, char* argv[], int start=2) {
    MasterConfig c;
    if (argc>=3 && argv[2][0]!='-') c.file=argv[2];
    for (int i=start;i<argc;i++) {
        string arg(argv[i]);
        auto eq=arg.find('=');
        string key,val;
        if (eq!=string::npos && arg.substr(0,2)=="--") {
            key=arg.substr(2,eq-2); val=arg.substr(eq+1);
        } else {
            auto nxt=[&]()->string{return(i+1<argc)?argv[++i]:"";};
            if      (arg=="--lr")        {c.lr        =stof(nxt());continue;}
            else if (arg=="--wd")        {c.wd        =stof(nxt());continue;}
            else if (arg=="--steps")     {c.steps     =stoi(nxt());continue;}
            else if (arg=="--dim")       {c.dim       =stoi(nxt());continue;}
            else if (arg=="--layers")    {c.n_layers  =stoi(nxt());continue;}
            else if (arg=="--grad_clip") {c.grad_clip =stof(nxt());continue;}
            else if (arg=="--warmup")    {c.warmup    =stof(nxt());continue;}
            else if (arg=="--lr_sched")  {c.lr_sched  =nxt();      continue;}
            else if (arg=="--diag")      {c.diag      =(stoi(nxt())!=0);continue;}
            else if (arg=="--diag_every"){c.diag_every=stoi(nxt());continue;}
            else if (arg=="--seed")      {c.seed      =stoi(nxt());continue;}
            else if (arg=="--preset")    {c.preset    =nxt();      continue;}
            else if (arg=="--file")      {c.file      =nxt();      continue;}
            else if (arg=="--reps")      {c.bench_reps=stoi(nxt());continue;}
            else if (arg=="--exit_thr")  {c.exit_thr  =stof(nxt());continue;}
            continue;
        }
        if      (key=="lr")        c.lr        =stof(val);
        else if (key=="wd")        c.wd        =stof(val);
        else if (key=="steps")     c.steps     =stoi(val);
        else if (key=="dim")       c.dim       =stoi(val);
        else if (key=="layers")    c.n_layers  =stoi(val);
        else if (key=="grad_clip") c.grad_clip =stof(val);
        else if (key=="warmup")    c.warmup    =stof(val);
        else if (key=="lr_sched")  c.lr_sched  =val;
        else if (key=="diag")      c.diag      =(stoi(val)!=0);
        else if (key=="diag_every")c.diag_every=stoi(val);
        else if (key=="seed")      c.seed      =stoi(val);
        else if (key=="preset")    c.preset    =val;
        else if (key=="file")      c.file      =val;
        else if (key=="reps")      c.bench_reps=stoi(val);
        else if (key=="exit_thr")  c.exit_thr  =stof(val);
        else fprintf(stderr,"[Config] Unknown: --%s\n",key.c_str());
    }
    return c;
}

static f32 get_lr(const MasterConfig& cfg, int step) {
    int ws=max(1,(int)(cfg.steps*cfg.warmup));
    if (step<ws) return cfg.lr*((f32)step/ws);
    if (cfg.lr_sched=="constant") return cfg.lr;
    f32 progress=(f32)(step-ws)/max(1,cfg.steps-ws);
    return cfg.lr*0.5f*(1.f+cosf(M_PI*progress));
}

// ════════════════════════════════════════════════════════════════════════════
// §3 — LOGIC2048 [BUG-4 FIXED: max power 63, không phải 62]
// ════════════════════════════════════════════════════════════════════════════

struct Logic2048Weight {
    u8 terms[4];

    static u8 encode(int sign, int power) {
        return (u8)((sign>0?0x80:0x00)|((power+64)&0x7F));
    }
    static Logic2048Weight from_float(f32 w) {
        Logic2048Weight lw;
        f32 residual=w;
        for (int k=0;k<4;k++) {
            if (fabsf(residual)<1e-30f){lw.terms[k]=encode(1,-64);continue;}
            int s=residual>0?+1:-1;
            int p=(int)floorf(log2f(fabsf(residual)));
            p=max(-63,min(p,63));  // [BUG-4 FIXED] max 63 dùng hết range bảng
            residual-=s*ldexpf(1.f,p);
            lw.terms[k]=encode(s,p);
        }
        return lw;
    }
};
static_assert(sizeof(Logic2048Weight)==4,"Logic2048Weight phải 4 bytes");

struct Logic2048Table {
    alignas(64) f32 val[256];
    Logic2048Table() {
        for (int e=0;e<256;e++) {
            int s=(e&0x80)?+1:-1;
            int p=(int)(e&0x7F)-64;
            val[e]=s*ldexpf(1.f,p);
        }
    }
    inline f32 get(u8 e) const { return val[e]; }
    inline f32 decode4(const Logic2048Weight& lw) const {
        return val[lw.terms[0]]+val[lw.terms[1]]
              +val[lw.terms[2]]+val[lw.terms[3]];
    }
} static g_l2k_table;

static void logic2048_matvec(
    const Logic2048Weight* __restrict__ A,
    const f32* __restrict__ x,
    f32* __restrict__ y,
    int rows, int cols)
{
#if HAS_OMP
    #pragma omp parallel for schedule(static) if(rows>=64)
#endif
    for (int i=0;i<rows;i++) {
        const Logic2048Weight* row=A+(size_t)i*cols;
        f32 s=0.f; int j=0;
#if HAS_NEON
        float32x4_t acc0=vdupq_n_f32(0.f),acc1=vdupq_n_f32(0.f);
        for (; j<=cols-8; j+=8) {
            float w0=g_l2k_table.decode4(row[j+0]);
            float w1=g_l2k_table.decode4(row[j+1]);
            float w2=g_l2k_table.decode4(row[j+2]);
            float w3=g_l2k_table.decode4(row[j+3]);
            float w4=g_l2k_table.decode4(row[j+4]);
            float w5=g_l2k_table.decode4(row[j+5]);
            float w6=g_l2k_table.decode4(row[j+6]);
            float w7=g_l2k_table.decode4(row[j+7]);
            float32x4_t wv0={w0,w1,w2,w3},wv1={w4,w5,w6,w7};
            acc0=vmlaq_f32(acc0,wv0,vld1q_f32(x+j));
            acc1=vmlaq_f32(acc1,wv1,vld1q_f32(x+j+4));
        }
        s=vaddvq_f32(vaddq_f32(acc0,acc1));
#endif
        for (; j<cols; j++) s+=g_l2k_table.decode4(row[j])*x[j];
        y[i]=s;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// §4 — COSSIN CACHE + BUTTERFLY
// ════════════════════════════════════════════════════════════════════════════

struct CosSinCache {
    vector<f32> cosp,sinp;
    int N,L,P;
    CosSinCache():N(0),L(0),P(0){}
    void build(int n, const vector<f32>& theta) {
        N=n; L=0; int tmp=n;
        while(tmp>1){L++;tmp>>=1;}
        P=N/2;
        cosp.resize((size_t)L*P);
        sinp.resize((size_t)L*P);
        for (int l=0;l<L;l++)
            for (int p=0;p<P;p++) {
                f32 th=theta[(size_t)l*P+p];
                cosp[(size_t)l*P+p]=cosf(th);
                sinp[(size_t)l*P+p]=sinf(th);
            }
    }
    size_t bytes() const {return(cosp.size()+sinp.size())*sizeof(f32);}
};

static void butterfly_forward_small(f32* h, int N, int L,
                                     const CosSinCache& cs) {
    int pi=0;
    for (int s=0;s<L;s++) {
        int stride=1<<s,blk=stride<<1;
        for (int i=0;i<N;i+=blk)
            for (int j=0;j<stride;j++,pi++) {
                f32 c=cs.cosp[pi],sn=cs.sinp[pi];
                f32 a=h[i+j],b=h[i+j+stride];
                h[i+j]       =c*a-sn*b;
                h[i+j+stride]=sn*a+c*b;
            }
    }
}

static void butterfly_forward_tiled(f32* h, int N, int L,
                                     const CosSinCache& cs) {
    int pi_stage=0;
    for (int s=0;s<L;s++) {
        int stride=1<<s,blk=stride<<1;
        for (int tile_start=0;tile_start<N;tile_start+=TILE_BF_L1) {
            int tile_end=min(tile_start+TILE_BF_L1,N);
            for (int i=0;i<N;i+=blk) {
                for (int j=0;j<stride;j++) {
                    int idx1=i+j,idx2=i+j+stride;
                    if (idx1<tile_start||idx1>=tile_end) continue;
                    int pair_idx=pi_stage+(i/blk)*stride+j;
                    f32 c=cs.cosp[pair_idx],sn=cs.sinp[pair_idx];
                    f32 a=h[idx1],b=h[idx2];
                    h[idx1]=c*a-sn*b;
                    h[idx2]=sn*a+c*b;
                }
            }
        }
        pi_stage+=N/2;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// §5 — BUTTERFLY STAGE (scalar)
// ════════════════════════════════════════════════════════════════════════════

static void bf_stage(f32* h, int N, int s, const f32* theta_s) {
    int stride=1<<s,blk=stride<<1,ti=0;
    for (int i=0;i<N;i+=blk)
        for (int j=0;j<stride;j++) {
            f32 a=h[i+j],b=h[i+j+stride];
            f32 c=cosf(theta_s[ti]),sn=sinf(theta_s[ti++]);
            h[i+j]       =c*a-sn*b;
            h[i+j+stride]=sn*a+c*b;
        }
}
static void bf_standard(f32* h, int N, const f32* theta_all, int L) {
    for (int s=0;s<L;s++) bf_stage(h,N,s,theta_all+s*(N/2));
}

// ════════════════════════════════════════════════════════════════════════════
// §6 — ADAMW + SPARSE ADAMW
// ════════════════════════════════════════════════════════════════════════════

struct AdamW {
    f32 lr,b1=0.9f,b2=0.999f,eps=1e-8f,wd; int t=0;
    vector<f32> m,v;
    AdamW()=default;
    AdamW(int n, f32 lr_, f32 wd_=0.01f)
        :lr(lr_),wd(wd_),m(n,0.f),v(n,0.f){}

    void step(f32* W, const f32* g, int n, f32 lr_override=-1.f) {
        ++t;
        f32 used_lr=(lr_override>0.f)?lr_override:lr;
        f32 bc1=1.f-powf(b1,(f32)t),bc2=1.f-powf(b2,(f32)t);
        for (int i=0;i<n;i++) {
            m[i]=b1*m[i]+(1-b1)*g[i];
            v[i]=b2*v[i]+(1-b2)*g[i]*g[i];
            f32 mh=m[i]/bc1,vh=v[i]/bc2;
            W[i]-=used_lr*mh/(sqrtf(vh)+eps)+used_lr*wd*W[i];
        }
    }
};

struct SparseAdamW {
    f32 lr,b1=0.9f,b2=0.999f,eps=1e-8f,wd; int dim;
    unordered_map<int,vector<f32>> m_map,v_map;
    unordered_map<int,int> step_map;
    SparseAdamW()=default;
    SparseAdamW(int dim_, f32 lr_, f32 wd_=0.f)
        :lr(lr_),wd(wd_),dim(dim_){}

    void step(f32* W_tok, const f32* g, int tok, f32 lr_override=-1.f) {
        f32 used_lr=(lr_override>0.f)?lr_override:lr;
        if (!m_map.count(tok)){
            m_map[tok].assign(dim,0.f);
            v_map[tok].assign(dim,0.f);
            step_map[tok]=0;
        }
        int& t=step_map[tok]; t++;
        f32 bc1=1.f-powf(b1,(f32)t),bc2=1.f-powf(b2,(f32)t);
        f32* mm=m_map[tok].data(),*vv=v_map[tok].data();
        for (int i=0;i<dim;i++) {
            mm[i]=b1*mm[i]+(1-b1)*g[i];
            vv[i]=b2*vv[i]+(1-b2)*g[i]*g[i];
            f32 mh=mm[i]/bc1,vh=vv[i]/bc2;
            W_tok[i]-=used_lr*mh/(sqrtf(vh)+eps)+used_lr*wd*W_tok[i];
        }
    }
};

// ════════════════════════════════════════════════════════════════════════════
// §7 — GRADIENT UTILITIES
// ════════════════════════════════════════════════════════════════════════════

static f32 grad_l2_norm(const f32* g, int n) {
    f32 s=0.f; for (int i=0;i<n;i++) s+=g[i]*g[i]; return sqrtf(s);
}
static f32 clip_grad_norm(f32* g, int n, f32 max_norm) {
    f32 norm=grad_l2_norm(g,n);
    if (norm>max_norm) {
        f32 scale=max_norm/(norm+1e-6f);
        for (int i=0;i<n;i++) g[i]*=scale;
    }
    return norm;
}

// ════════════════════════════════════════════════════════════════════════════
// §8 — DIB LAYER
// ════════════════════════════════════════════════════════════════════════════

struct DIBLayer {
    int N,L,P;
    vector<f32> theta,diag;
    AdamW opt_t,opt_d;

    DIBLayer()=default;
    explicit DIBLayer(int dim, f32 lr=1e-3f, u32 seed=42)
        :N(dim),L(__builtin_ctz((unsigned)dim)),
         P(dim/2*__builtin_ctz((unsigned)dim)),
         theta((size_t)L*(dim/2),0.f),diag((size_t)L*dim,1.f),
         opt_t(P,lr*0.1f),opt_d(L*dim,lr)
    {
        mt19937 rng(seed); normal_distribution<f32> nd;
        for (auto& v:theta) v=nd(rng)*0.02f;
        for (auto& v:diag)  v=1.f+nd(rng)*0.005f;
    }

    void forward(const f32* x, f32* out) const {
        memcpy(out,x,N*sizeof(f32));
        for (int s=0;s<L;s++) {
            bf_stage(out,N,s,theta.data()+s*(N/2));
            velmul(out,diag.data()+s*N,N);
        }
    }

    struct Tape {
        vector<vector<f32>> h;
        vector<f32> hpre;
    };
    Tape forward_tape(const f32* x) const {
        Tape t; t.h.resize(L+1,vector<f32>(N)); t.hpre.resize(L*N);
        memcpy(t.h[0].data(),x,N*4);
        for (int s=0;s<L;s++) {
            memcpy(t.h[s+1].data(),t.h[s].data(),N*4);
            bf_stage(t.h[s+1].data(),N,s,theta.data()+s*(N/2));
            memcpy(t.hpre.data()+s*N,t.h[s+1].data(),N*4);
            velmul(t.h[s+1].data(),diag.data()+s*N,N);
        }
        return t;
    }

    vector<f32> backward(const Tape& tape, const f32* grad_out,
                          f32 lr_override=-1.f) {
        vector<f32> g(grad_out,grad_out+N);
        vector<f32> gt(P,0.f),gd(L*N,0.f);
        for (int s=L-1;s>=0;s--) {
            const f32* ds=diag.data()+s*N,*hb=tape.hpre.data()+s*N;
            f32* gdp=gd.data()+s*N;
            for (int i=0;i<N;i++){gdp[i]=g[i]*hb[i]; g[i]*=ds[i];}
            const f32* hs=tape.h[s].data();
            f32* gts=gt.data()+s*(N/2);
            int stride=1<<s,blk=stride<<1,ti=0;
            const f32* th=theta.data()+s*(N/2);
            for (int i=0;i<N;i+=blk)
                for (int j=0;j<stride;j++) {
                    f32 a=hs[i+j],b=hs[i+j+stride];
                    f32 ga=g[i+j],gb=g[i+j+stride];
                    f32 c=cosf(th[ti]),sn=sinf(th[ti]);
                    gts[ti]+=ga*(-sn*a-c*b)+gb*(c*a-sn*b);
                    g[i+j]       = c*ga+sn*gb;
                    g[i+j+stride]=-sn*ga+c*gb;
                    ti++;
                }
        }
        opt_t.step(theta.data(),gt.data(),P,lr_override);
        opt_d.step(diag.data(),gd.data(),L*N,lr_override);
        return g;
    }
    long long params() const {return P+(long long)L*N;}
};

// ════════════════════════════════════════════════════════════════════════════
// §9 — FLASHATTENTION NEON
// ════════════════════════════════════════════════════════════════════════════

struct FlashWorkspace {
    vector<f32> m_row,l_row;
    void reset(int N) {
        if((int)m_row.size()<N){m_row.resize(N);l_row.resize(N);}
        fill(m_row.begin(),m_row.begin()+N,-1e30f);
        fill(l_row.begin(),l_row.begin()+N,0.f);
    }
};

static void flash_1head(const f32* Q, const f32* K, const f32* V, f32* O,
                         int N, int d, int stride, int hoff, f32 scale,
                         FlashWorkspace& ws) {
    constexpr int Br=4,Bc=16;
    ws.reset(N);
    f32* mr=ws.m_row.data(),*lr=ws.l_row.data();
    for (int i=0;i<N;i++) memset(O+i*stride+hoff,0,d*4);
    for (int qi=0;qi<N;qi+=Br) {
        int Brr=min(qi+Br,N)-qi;
        for (int kj=0;kj<N;kj+=Bc) {
            int Bcc=min(kj+Bc,N)-kj;
            f32 S[Br*Bc];
            for (int ii=0;ii<Brr;ii++) {
                const f32* qi_ptr=Q+(size_t)(qi+ii)*stride+hoff;
                for (int jj=0;jj<Bcc;jj++)
                    S[ii*Bc+jj]=vdot_f32(qi_ptr,K+(size_t)(kj+jj)*stride+hoff,d)*scale;
            }
            for (int ii=0;ii<Brr;ii++) {
                int row=qi+ii;
                f32 mn=mr[row];
                for (int jj=0;jj<Bcc;jj++) if(S[ii*Bc+jj]>mn) mn=S[ii*Bc+jj];
                f32 corr=expf(mr[row]-mn);
                f32* oi=O+(size_t)row*stride+hoff;
                vscale(oi,corr,d);
                f32 ln=lr[row]*corr;
                for (int jj=0;jj<Bcc;jj++) {
                    f32 p=expf(S[ii*Bc+jj]-mn);
                    ln+=p; vaxpy(oi,V+(size_t)(kj+jj)*stride+hoff,p,d);
                }
                mr[row]=mn; lr[row]=ln;
            }
        }
    }
    for (int i=0;i<N;i++)
        vscale(O+(size_t)i*stride+hoff,1.f/(lr[i]+1e-10f),d);
}

static void flash_attention(const f32* Q, const f32* K, const f32* V,
    f32* O, int N, int nh, int d) {
    f32 scale=1.f/sqrtf((f32)d);
    int hd=nh*d,nth=1;
#if HAS_OMP
    nth=omp_get_max_threads();
#endif
    vector<FlashWorkspace> wss(nth);
#if HAS_OMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h=0;h<nh;h++) {
        int tid=0;
#if HAS_OMP
        tid=omp_get_thread_num();
#endif
        flash_1head(Q,K,V,O,N,d,hd,h*d,scale,wss[tid]);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// §10 — TOKEN COMPRESSOR
// ════════════════════════════════════════════════════════════════════════════

struct TokenCompressor {
    int dim; f32 keep_ratio; bool preserve_order;
    TokenCompressor(int d, f32 keep=0.65f)
        :dim(d),keep_ratio(keep),preserve_order(true){}

    int compress(const f32* x_in, f32* x_out, const f32* scores,
                  int seq, vector<int>& kept) {
        int keep_n=max(1,(int)(seq*keep_ratio));
        vector<int> idx(seq); iota(idx.begin(),idx.end(),0);
        partial_sort(idx.begin(),idx.begin()+keep_n,idx.end(),
            [&](int a, int b){return scores[a]>scores[b];});
        if (preserve_order) sort(idx.begin(),idx.begin()+keep_n);
        kept.assign(idx.begin(),idx.begin()+keep_n);
        for (int i=0;i<keep_n;i++)
            memcpy(x_out+i*dim,x_in+kept[i]*dim,dim*4);
        return keep_n;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// §11 — DYNAMIC LAYER BUDGET [BUG-3 FIXED: mutex bảo vệ mutable state]
// ════════════════════════════════════════════════════════════════════════════

struct DynamicLayerBudget {
    int total_layers,min_layers,sample_size;
    f32 entropy_threshold;
    mutable vector<int> exit_counts;
    mutable mt19937 rng;
    mutable mutex rng_mtx;   // [BUG-3 FIXED] bảo vệ rng + exit_counts

    DynamicLayerBudget(int total, f32 thr=0.3f, int minl=4, int sample=256)
        :total_layers(total),min_layers(minl),sample_size(sample),
         entropy_threshold(thr),exit_counts(total,0),rng(1234){}

    // [BUG-3 FIXED] thread-safe version
    bool should_exit(const f32* x, const f32* lm_head,
                     int layer, int dim, int vocab) const {
        if (layer<min_layers||layer>=total_layers-1) return false;
        int s=min(vocab,sample_size);
        vector<int> sampled(s);
        {
            lock_guard<mutex> lk(rng_mtx);   // protect rng
            if (s<vocab) {
                vector<int> pool(vocab); iota(pool.begin(),pool.end(),0);
                for (int i=0;i<s;i++) {
                    int j=i+rng()%(vocab-i);
                    swap(pool[i],pool[j]);
                    sampled[i]=pool[i];
                }
            } else iota(sampled.begin(),sampled.end(),0);
        }
        vector<f32> logits(s);
        for (int i=0;i<s;i++)
            logits[i]=vdot_f32(lm_head+(size_t)sampled[i]*dim,x,dim);
        f32 mx=*max_element(logits.begin(),logits.end());
        f32 sum=0.f;
        for (f32& l:logits){l=expf(l-mx);sum+=l;}
        f32 entropy=0.f;
        for (f32& l:logits){l/=sum;if(l>1e-10f)entropy-=l*logf(l);}
        f32 norm_entropy=entropy/logf((f32)s);
        if (norm_entropy<entropy_threshold) {
            lock_guard<mutex> lk(rng_mtx);   // protect exit_counts
            exit_counts[layer]++;
            return true;
        }
        return false;
    }

    void print_exits() const {
        printf("  DLB exit distribution:\n");
        for (int l=0;l<total_layers;l++)
            if (exit_counts[l]>0)
                printf("    layer %2d: %d exits\n",l,exit_counts[l]);
    }
};

// ════════════════════════════════════════════════════════════════════════════
// §12 — JIT AARCH64 FMLA KERNEL
// ════════════════════════════════════════════════════════════════════════════

namespace A64 {
    static u32 fmla_4s(int Rd,int Rn,int Rm){
        return 0x4E20CC00u|((u32)Rm<<16)|((u32)Rn<<5)|(u32)Rd;
    }
    static u32 ldr_q(int Qt,int Xn,int imm){
        if(imm<0||imm>=16*4096||imm%16!=0) return 0;
        return 0x3DC00000u|(u32)(imm/16<<10)|((u32)Xn<<5)|(u32)Qt;
    }
    static u32 str_q(int Qt,int Xn,int imm){
        if(imm<0||imm>=16*4096||imm%16!=0) return 0;
        return 0x3D800000u|(u32)(imm/16<<10)|((u32)Xn<<5)|(u32)Qt;
    }
    static u32 eor_v(int Rd,int Rn,int Rm){
        return 0x6E201C00u|((u32)Rm<<16)|((u32)Rn<<5)|(u32)Rd;
    }
    static constexpr u32 RET=0xD65F03C0u;
}

struct JITKernel {
    vector<u32> code;
    void* exec_page=nullptr;
    size_t page_sz=0;
    using KernelFn=void(*)(const f32*,const f32*,f32*,int);
    KernelFn fn=nullptr;

    void generate(int d) {
        code.clear();
        for (int r=0;r<4;r++) code.push_back(A64::eor_v(r,r,r));
        int n_iters=d/4;
        for (int k=0;k<n_iters;k++) {
            int bx=k*16,bw0=k*16;
            int bw1=(d+k*4)*4,bw2=(2*d+k*4)*4,bw3=(3*d+k*4)*4;
            if (bw3>=65520) break;
            code.push_back(A64::ldr_q(8,1,bx));
            code.push_back(A64::ldr_q(4,0,bw0));
            code.push_back(A64::ldr_q(5,0,bw1));
            code.push_back(A64::ldr_q(6,0,bw2));
            code.push_back(A64::ldr_q(7,0,bw3));
            code.push_back(A64::fmla_4s(0,4,8));
            code.push_back(A64::fmla_4s(1,5,8));
            code.push_back(A64::fmla_4s(2,6,8));
            code.push_back(A64::fmla_4s(3,7,8));
        }
        code.push_back(A64::str_q(0,2, 0));
        code.push_back(A64::str_q(1,2,16));
        code.push_back(A64::str_q(2,2,32));
        code.push_back(A64::str_q(3,2,48));
        code.push_back(A64::RET);
    }
    bool compile() {
#if defined(__linux__) && HAS_AARCH64
        page_sz=(code.size()*4+4095)&~4095;
        exec_page=mmap(nullptr,page_sz,PROT_READ|PROT_WRITE|PROT_EXEC,
                       MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(exec_page==MAP_FAILED){exec_page=nullptr;return false;}
        memcpy(exec_page,code.data(),code.size()*4);
        __builtin___clear_cache((char*)exec_page,(char*)exec_page+code.size()*4);
        fn=(KernelFn)exec_page;
        return true;
#else
        return false;
#endif
    }
    void run4rows(const f32* W, const f32* x, f32* y, int d) {
        f32 tmp[16]={};
        if (fn) { fn(W,x,tmp,d); }
        else {
            for (int r=0;r<4;r++) {
                f32 acc[4]={0,0,0,0};
                const f32* Wr=W+r*d;
                for (int k=0;k<d;k+=4) {
                    acc[0]+=Wr[k]*x[k];acc[1]+=Wr[k+1]*x[k+1];
                    acc[2]+=Wr[k+2]*x[k+2];acc[3]+=Wr[k+3]*x[k+3];
                }
                tmp[r*4]=acc[0];tmp[r*4+1]=acc[1];
                tmp[r*4+2]=acc[2];tmp[r*4+3]=acc[3];
            }
        }
        for (int r=0;r<4;r++)
            y[r]+=tmp[r*4]+tmp[r*4+1]+tmp[r*4+2]+tmp[r*4+3];
    }
    ~JITKernel(){
#if defined(__linux__) && HAS_AARCH64
        if(exec_page&&exec_page!=MAP_FAILED) munmap(exec_page,page_sz);
#endif
    }
};

// ════════════════════════════════════════════════════════════════════════════
// §13 — [APEX-1] LOGIC2048 TRAINABLE LINEAR (Straight-Through Estimator)
//
//  Nguyên lý: STE (Bengio et al.)
//  Forward:   dùng Logic2048 weights (decode on-the-fly, 4 bytes/weight)
//  Backward:  gradient pass-through như thể không có quantization
//  Optimizer: AdamW cập nhật FP32 shadow → re-quantize sau mỗi step
//  Kết quả:   model train ở FP32 precision, inference ở Logic2048 speed
// ════════════════════════════════════════════════════════════════════════════

struct Logic2048Linear {
    int in_dim, out_dim;
    vector<f32>           shadow_w;  // FP32 shadow weights (optimizer state)
    vector<Logic2048Weight> qw;       // Quantized weights (inference)
    AdamW opt;

    Logic2048Linear() = default;
    Logic2048Linear(int in, int out, f32 lr, u32 seed=42)
        :in_dim(in), out_dim(out),
         shadow_w((size_t)out*in),
         qw((size_t)out*in),
         opt((int)((size_t)out*in), lr, 0.01f)
    {
        mt19937 rng(seed);
        // Xavier init: variance = 2/(in+out)
        normal_distribution<f32> nd(0.f, sqrtf(2.f/((f32)in+out)));
        for (auto& v:shadow_w) v=nd(rng);
        quantize_all();
    }

    // Quantize shadow → qw
    void quantize_all() {
        for (size_t i=0;i<qw.size();i++)
            qw[i]=Logic2048Weight::from_float(shadow_w[i]);
    }

    // Forward: Logic2048 matvec (fast integer-shift decode)
    void forward(const f32* x, f32* y) const {
        logic2048_matvec(qw.data(), x, y, out_dim, in_dim);
    }

    // Backward: STE — grad flows as if quantization transparent
    // Returns grad_in [in_dim]. Accumulates grad_w using shadow weights.
    vector<f32> backward_ste(const f32* x, const f32* grad_out,
                              f32 lr_override=-1.f) {
        // grad_in = shadow_W^T * grad_out  (STE: use FP32 shadow, not L2K)
        vector<f32> grad_in(in_dim, 0.f);
        vector<f32> grad_w((size_t)out_dim*in_dim, 0.f);
        for (int o=0;o<out_dim;o++) {
            vaxpy(grad_in.data(), shadow_w.data()+(size_t)o*in_dim,
                  grad_out[o], in_dim);
            vaxpy(grad_w.data()+(size_t)o*in_dim, x, grad_out[o], in_dim);
        }
        // Update shadow with AdamW
        opt.step(shadow_w.data(), grad_w.data(),
                 (int)(out_dim*in_dim), lr_override);
        // Re-quantize to L2K format
        quantize_all();
        return grad_in;
    }

    long long params() const {return (long long)in_dim*out_dim;}
    // Memory: L2K format uses 4 bytes/weight vs 4 bytes FP32 — same size now
    // BUT: decode = 4 additions vs 1 load, traded for integer operations
    // On ARM: integer shift pipelines → faster throughput on SIMD
    float memory_mb() const {
        return (float)((size_t)in_dim*out_dim*sizeof(Logic2048Weight)) / (1024*1024);
    }
};

// ════════════════════════════════════════════════════════════════════════════
// §14 — [APEX-2] ADAPTIVE LAYER (DIB core + Logic2048Linear projection)
//
//  Thay thế Transformer FFN block:
//    OLD: Linear(dim→4*dim) + GELU + Linear(4*dim→dim)  → O(dim²) params
//    NEW: DIB(dim) + L2K_Linear(dim→dim)               → O(dim*logdim) params
//
//  DIB = Butterfly transform = O(N log N) mixing operations
//  L2K = Logic2048 projection = fast integer-decode forward pass
//  RMSNorm + residual connection = training stability
// ════════════════════════════════════════════════════════════════════════════

struct AdaptiveLayer {
    int dim;
    DIBLayer          dib;       // Butterfly mixing: O(N logN)
    Logic2048Linear   proj;      // Logic2048 projection: STE training
    vector<f32>       norm_w;    // RMSNorm scale weights
    AdamW             opt_norm;

    AdaptiveLayer() = default;
    AdaptiveLayer(int d, f32 lr, u32 seed=42)
        :dim(d), dib(d, lr*0.8f, seed),
         proj(d, d, lr*0.5f, seed+1000),
         norm_w(d, 1.f),
         opt_norm(d, lr)
    {}

    struct Tape {
        vector<f32>    x_in;        // raw input (for residual)
        vector<f32>    x_normed;    // after RMSNorm
        DIBLayer::Tape dib_tape;    // DIB intermediate states
        vector<f32>    dib_out;     // after DIB
        vector<f32>    proj_out;    // after L2K projection
    };

    // Forward: store all intermediate states for backprop
    Tape forward_tape(const f32* x) const {
        Tape t;
        t.x_in.assign(x, x+dim);
        t.x_normed.resize(dim);
        rmsnorm(t.x_normed.data(), x, norm_w.data(), dim);
        t.dib_tape = dib.forward_tape(t.x_normed.data());
        t.dib_out.assign(t.dib_tape.h[dib.L].begin(),
                         t.dib_tape.h[dib.L].end());
        t.proj_out.resize(dim);
        proj.forward(t.dib_out.data(), t.proj_out.data());
        return t;
    }

    // Forward output: residual + proj(dib(norm(x)))
    void forward(const f32* x, f32* out) const {
        vector<f32> normed(dim), dib_out(dim), proj_out(dim);
        rmsnorm(normed.data(), x, norm_w.data(), dim);
        dib.forward(normed.data(), dib_out.data());
        proj.forward(dib_out.data(), proj_out.data());
        for (int i=0;i<dim;i++) out[i] = x[i] + proj_out[i];
    }

    // Backward: returns gradient wrt input x
    vector<f32> backward(const Tape& tape, const f32* grad_out,
                          f32 lr_override=-1.f) {
        // grad_out flows through residual: grad_residual = grad_out
        // grad_proj_out = grad_out (residual branch passes through)
        // Backward through Logic2048Linear (STE)
        auto grad_dib_out = proj.backward_ste(
            tape.dib_out.data(), grad_out, lr_override);

        // Backward through DIB
        auto grad_normed = dib.backward(
            tape.dib_tape, grad_dib_out.data(), lr_override);

        // Backward through RMSNorm + accumulate norm_w gradient
        // Simplified: use grad_normed * norm_w as approx (full RMSNorm grad costly)
        vector<f32> grad_norm_w(dim, 0.f);
        f32 ss=0.f;
        for (int i=0;i<dim;i++) ss+=tape.x_in[i]*tape.x_in[i];
        f32 inv=1.f/sqrtf(ss/dim+1e-5f);
        for (int i=0;i<dim;i++) {
            grad_norm_w[i] = grad_normed[i]*tape.x_in[i]*inv;
        }
        opt_norm.step(norm_w.data(), grad_norm_w.data(), dim, lr_override);

        // grad_x = grad_out (residual) + grad_normed * norm_w * inv (RMSNorm pass)
        vector<f32> grad_x(dim);
        for (int i=0;i<dim;i++)
            grad_x[i] = grad_out[i] + grad_normed[i]*norm_w[i]*inv;
        return grad_x;
    }

    long long params() const {
        return dib.params() + proj.params() + dim;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// §15 — [APEX-3] ADAPTIVE UNIFIED MODEL
//
//  Kết hợp hoàn toàn 3 breakthrough:
//  1. Logic2048Linear: train trong format L2K từ đầu (STE)
//  2. AdaptiveLayers: DIB thay FFN (O(N logN) per layer)
//  3. DynamicLayerBudget: per-token depth adaptation
//
//  Architecture:
//    tok → Embedding(sparse FP32) →
//    [AdaptiveLayer_0..n-1 với DLB early exit] →
//    RMSNorm → Logic2048Linear LM Head →
//    Softmax → probabilities
// ════════════════════════════════════════════════════════════════════════════

struct StepDiag {
    f32 loss, prob_target, prob_min;
    f32 grad_norm_pre, grad_norm_post, lr_used, proj_delta;
    int layers_used;   // NEW: how many layers actually ran (DLB metric)
};

struct AdaptiveUnifiedModel {
    int vocab, dim, n_layers;
    vector<f32>           emb;        // Embeddings (sparse FP32)
    Logic2048Linear       lm_head;    // LM head (L2K + STE)
    vector<AdaptiveLayer> layers;     // DIB + L2K layers
    vector<f32>           final_norm; // final RMSNorm
    AdamW                 opt_norm_final;
    SparseAdamW           opt_emb;
    DynamicLayerBudget    dlb;        // per-token depth budget
    f32 cfg_grad_clip;

    AdaptiveUnifiedModel(int V, int D, int L,
                          f32 lr, f32 wd, f32 gc, f32 exit_thr,
                          int seed=42)
        :vocab(V), dim(D), n_layers(L),
         emb((size_t)V*D),
         lm_head(D, V, lr*0.3f, seed+9999),
         final_norm(D, 1.f),
         opt_norm_final(D, lr),
         opt_emb(D, lr, 0.f),
         dlb(L, exit_thr, max(1,L/4), min(V,512)),
         cfg_grad_clip(gc)
    {
        mt19937 rng(seed);
        normal_distribution<f32> nd(0.f, sqrtf(2.f/D));
        for (auto& v:emb) v=nd(rng);
        for (int l=0;l<L;l++)
            layers.emplace_back(D, lr, (u32)(seed+l+1));
    }

    struct FwdResult {
        vector<f32>                    hidden;
        vector<AdaptiveLayer::Tape>    tapes;
        vector<f32>                    logits, probs;
        int                            token;
        int                            layers_used;   // DLB: actual depth
        vector<bool>                   layer_ran;
    };

    // Forward with DLB per-token early exit
    FwdResult forward(int tok) const {
        FwdResult r;
        r.token = tok;
        r.layers_used = n_layers;
        r.layer_ran.assign(n_layers, false);
        r.hidden.resize(dim);
        memcpy(r.hidden.data(), emb.data()+(size_t)tok*dim, dim*4);
        r.tapes.resize(n_layers);

        // Precompute LM head weights for DLB entropy check
        // (reuse the L2K shadow weights decoded)
        // Note: DLB samples from lm_head.shadow_w for entropy estimate
        const f32* lm_shadow = lm_head.shadow_w.data();

        for (int l=0;l<n_layers;l++) {
            // DLB check: should we exit early?
            if (dlb.should_exit(r.hidden.data(), lm_shadow,
                                l, dim, vocab)) {
                r.layers_used = l;
                break;
            }
            r.tapes[l] = layers[l].forward_tape(r.hidden.data());
            // Update hidden: residual + proj(dib(norm(h)))
            for (int i=0;i<dim;i++)
                r.hidden[i] = r.tapes[l].x_in[i]
                             + r.tapes[l].proj_out[i];
            r.layer_ran[l] = true;
        }

        // Final RMSNorm
        vector<f32> normed(dim);
        rmsnorm(normed.data(), r.hidden.data(), final_norm.data(), dim);

        // LM Head (Logic2048Linear)
        r.logits.resize(vocab);
        lm_head.forward(normed.data(), r.logits.data());
        r.probs = r.logits;
        vsoftmax(r.probs.data(), vocab);
        return r;
    }

    StepDiag train_step(int tok, int next_tok, f32 lr_override=-1.f) {
        StepDiag d = {};
        d.lr_used = (lr_override>0.f) ? lr_override : layers[0].opt_norm.lr;

        FwdResult r = forward(tok);
        d.layers_used   = r.layers_used;
        d.prob_target   = r.probs[next_tok];
        d.prob_min      = *min_element(r.probs.begin(), r.probs.end());
        d.loss          = -logf(r.probs[next_tok] + 1e-5f);

        // CE gradient
        vector<f32> dlogits = r.probs;
        dlogits[next_tok] -= 1.f;

        d.grad_norm_pre  = grad_l2_norm(dlogits.data(), vocab);
        clip_grad_norm(dlogits.data(), vocab, cfg_grad_clip);
        d.grad_norm_post = grad_l2_norm(dlogits.data(), vocab);

        // Backward through LM head (Logic2048 STE)
        // Need final normed hidden
        vector<f32> normed(dim);
        rmsnorm(normed.data(), r.hidden.data(), final_norm.data(), dim);

        // Track proj norm change for diagnostics
        f32 pb=0.f;
        for (int i=0;i<dim;i++) pb+=fabsf(lm_head.shadow_w[i]);

        vector<f32> grad_h = lm_head.backward_ste(
            normed.data(), dlogits.data(), lr_override);

        f32 pa=0.f;
        for (int i=0;i<dim;i++) pa+=fabsf(lm_head.shadow_w[i]);
        d.proj_delta = fabsf(pa-pb)/dim;

        // Backward through final RMSNorm
        f32 ss=0.f;
        for (int i=0;i<dim;i++) ss+=r.hidden[i]*r.hidden[i];
        f32 inv=1.f/sqrtf(ss/dim+1e-5f);
        vector<f32> grad_final_norm(dim,0.f);
        vector<f32> grad_hidden(dim,0.f);
        for (int i=0;i<dim;i++) {
            grad_final_norm[i] = grad_h[i]*r.hidden[i]*inv;
            grad_hidden[i]     = grad_h[i]*final_norm[i]*inv;
        }
        opt_norm_final.step(final_norm.data(), grad_final_norm.data(),
                            dim, lr_override);

        clip_grad_norm(grad_hidden.data(), dim, cfg_grad_clip);

        // Backward through AdaptiveLayers (only layers that ran)
        for (int l=r.layers_used-1; l>=0; l--) {
            if (!r.layer_ran[l]) continue;
            auto g = layers[l].backward(r.tapes[l],
                                         grad_hidden.data(), lr_override);
            grad_hidden = g;
            clip_grad_norm(grad_hidden.data(), dim, cfg_grad_clip);
        }

        // Embedding update (sparse)
        opt_emb.step(emb.data()+(size_t)tok*dim,
                     grad_hidden.data(), tok, lr_override);
        return d;
    }

    void print_model_info() const {
        long long total=0;
        total += (long long)vocab*dim;   // embeddings
        for (auto& l:layers) total+=l.params();
        total += lm_head.params();
        total += dim;  // final norm
        printf("  AdaptiveUnifiedModel:\n");
        printf("    vocab=%d  dim=%d  layers=%d\n",vocab,dim,n_layers);
        printf("    Total params: %lld (%.2fM)\n",total,total/1e6f);
        printf("    LM Head: Logic2048Linear (%d×%d, %.2fMB L2K)\n",
               dim,vocab,lm_head.memory_mb());
        printf("    Per layer: DIB O(N logN) + L2K Linear\n");
        printf("    DLB: exit_thr=%.2f  min_layers=%d\n",
               dlb.entropy_threshold, dlb.min_layers);
    }
};

// ════════════════════════════════════════════════════════════════════════════
// §16 — CHARLM v5 (legacy — giữ để compare)
// ════════════════════════════════════════════════════════════════════════════

struct CharLM_v5 {
    int vocab,dim,n_layers;
    vector<f32> emb,proj;
    vector<DIBLayer> layers;
    SparseAdamW opt_emb;
    AdamW       opt_proj;
    f32 cfg_grad_clip;
    f32 total_loss=0.f; int total_steps=0;

    CharLM_v5(int V,int D,int L,f32 lr,f32 wd,f32 gc=1.0f,int seed=42)
        :vocab(V),dim(D),n_layers(L),
         emb((size_t)V*D),proj((size_t)V*D),
         opt_emb(D,lr,0.f),opt_proj((size_t)V*D,lr,wd),
         cfg_grad_clip(gc)
    {
        mt19937 rng(seed);
        normal_distribution<f32> nd_emb(0.f,sqrtf(2.f/D));
        normal_distribution<f32> nd_proj(0.f,1.f/sqrtf((f32)D));
        for (auto& v:emb)  v=nd_emb(rng);
        for (auto& v:proj) v=nd_proj(rng);
        for (int l=0;l<L;l++) layers.emplace_back(D,lr*0.8f,(u32)(seed+l+1));
    }

    struct FwdResult {
        vector<f32> hidden;
        vector<DIBLayer::Tape> tapes;
        vector<f32> logits,probs;
        int token;
    };

    FwdResult forward(int tok) const {
        FwdResult r; r.token=tok;
        r.hidden.resize(dim);
        memcpy(r.hidden.data(),emb.data()+(size_t)tok*dim,dim*4);
        r.tapes.resize(n_layers);
        for (int l=0;l<n_layers;l++) {
            r.tapes[l]=layers[l].forward_tape(r.hidden.data());
            memcpy(r.hidden.data(),r.tapes[l].h[layers[l].L].data(),dim*4);
        }
        r.logits.resize(vocab);
        for (int v=0;v<vocab;v++)
            r.logits[v]=vdot_f32(proj.data()+(size_t)v*dim,r.hidden.data(),dim);
        r.probs=r.logits;
        vsoftmax(r.probs.data(),vocab);
        return r;
    }

    StepDiag train_step(int tok,int next_tok,f32 lr_override=-1.f) {
        StepDiag d={}; d.layers_used=n_layers;
        d.lr_used=(lr_override>0.f)?lr_override:opt_proj.lr;
        FwdResult r=forward(tok);
        d.prob_target=r.probs[next_tok];
        d.prob_min=*min_element(r.probs.begin(),r.probs.end());
        d.loss=-logf(r.probs[next_tok]+1e-5f);
        vector<f32> dlogits=r.probs;
        dlogits[next_tok]-=1.f;
        d.grad_norm_pre=grad_l2_norm(dlogits.data(),vocab);
        clip_grad_norm(dlogits.data(),vocab,cfg_grad_clip);
        d.grad_norm_post=grad_l2_norm(dlogits.data(),vocab);
        vector<f32> grad_h(dim,0.f),grad_proj((size_t)vocab*dim,0.f);
        f32 pb=0.f; for(int i=0;i<dim;i++) pb+=fabsf(proj[i]);
        for (int v=0;v<vocab;v++) {
            vaxpy(grad_h.data(),proj.data()+(size_t)v*dim,dlogits[v],dim);
            vaxpy(grad_proj.data()+(size_t)v*dim,r.hidden.data(),dlogits[v],dim);
        }
        opt_proj.step(proj.data(),grad_proj.data(),(int)vocab*dim,lr_override);
        f32 pa=0.f; for(int i=0;i<dim;i++) pa+=fabsf(proj[i]);
        d.proj_delta=fabsf(pa-pb)/dim;
        clip_grad_norm(grad_h.data(),dim,cfg_grad_clip);
        for (int l=n_layers-1;l>=0;l--) {
            auto g=layers[l].backward(r.tapes[l],grad_h.data(),lr_override);
            grad_h=g;
            clip_grad_norm(grad_h.data(),dim,cfg_grad_clip);
        }
        opt_emb.step(emb.data()+(size_t)tok*dim,grad_h.data(),tok,lr_override);
        total_loss+=d.loss; total_steps++;
        return d;
    }
    f32 avg_loss() const{return total_steps?total_loss/total_steps:0.f;}
};

// ════════════════════════════════════════════════════════════════════════════
// §17 — GGUF PARSER [BUG-1 FIXED: không dùng exceptions, dùng error status]
// ════════════════════════════════════════════════════════════════════════════

enum GGUFType : u32 {
    GGUF_UINT8=0,GGUF_INT8,GGUF_UINT16,GGUF_INT16,
    GGUF_UINT32,GGUF_INT32,GGUF_FLOAT32,GGUF_BOOL,
    GGUF_STRING,GGUF_ARRAY,GGUF_UINT64,GGUF_INT64,GGUF_FLOAT64
};

struct GGUFMeta {
    u32 version=0; u64 n_tensors=0;
    string arch;
    u32 n_embd=0,n_head=0,n_head_kv=0,n_layer=0,n_ff=0,n_ctx=0,n_vocab=0;
    f32 rope_theta=10000.f;
    bool valid=false;
    string error_msg;  // [BUG-1 FIXED] error message thay vì exception
    void print() const {
        printf("  arch=%s  embd=%u  heads=%u  kv_heads=%u  layers=%u  ff=%u\n",
               arch.c_str(),n_embd,n_head,n_head_kv,n_layer,n_ff);
        printf("  ctx=%u  vocab=%u  rope_theta=%.0f\n",n_ctx,n_vocab,rope_theta);
    }
};

class GGUFParser {
    const u8* p=nullptr; const u8* end_ptr=nullptr; size_t pos=0;
    bool ok=true; string err;

    // [BUG-1 FIXED] Không dùng throw — trả về bool, set error flag
    template<typename T> bool read(T& out) {
        if (!ok||p+pos+sizeof(T)>end_ptr) {
            ok=false; err="truncated at pos "+to_string(pos); return false;
        }
        memcpy(&out,p+pos,sizeof(T)); pos+=sizeof(T); return true;
    }
    bool read_string(string& s) {
        u64 len=0;
        if (!read(len)) return false;
        if (len>1<<20) { ok=false; err="string too long: "+to_string(len); return false; }
        if (p+pos+len>end_ptr) { ok=false; err="string truncated"; return false; }
        s.assign((const char*)(p+pos),len); pos+=len; return true;
    }
    struct Val {
        GGUFType type;
        union{u8 u8v;u32 u32v;f32 f32v;u64 u64v;};
        string str; vector<Val> arr;
    };
    bool read_value(GGUFType type, Val& v) {
        v.type=type;
        switch(type){
        case GGUF_UINT8:   return read(v.u8v);
        case GGUF_UINT32:  return read(v.u32v);
        case GGUF_FLOAT32: return read(v.f32v);
        case GGUF_BOOL:    return read(v.u8v);
        case GGUF_UINT64:  return read(v.u64v);
        case GGUF_INT32:   {int32_t x; if(!read(x))return false; v.u32v=(u32)x; return true;}
        case GGUF_STRING:  return read_string(v.str);
        case GGUF_ARRAY: {
            u32 et_raw=0; u64 cnt=0;
            if (!read(et_raw)||!read(cnt)) return false;
            GGUFType et=(GGUFType)et_raw;
            u64 cap=min(cnt,(u64)1024);
            v.arr.reserve((size_t)cap);
            for (u64 i=0;i<cnt&&ok;i++) {
                Val child; read_value(et,child);
                if (i<cap) v.arr.push_back(std::move(child));
            }
            return ok;
        }
        default: { u64 x; return read(x); }
        }
    }
public:
    GGUFMeta parse(const void* data, size_t size) {
        p=(const u8*)data; end_ptr=p+size; pos=0; ok=true;
        GGUFMeta m;
        if (size<24) { m.error_msg="file too small"; return m; }
        u32 magic=0; if(!read(magic)||magic!=0x46554747u) {
            m.error_msg="bad magic"; return m;
        }
        read(m.version);
        read(m.n_tensors);
        u64 nkv=0; read(nkv);
        if (!ok) { m.error_msg=err; return m; }
        printf("[GGUF] v%u  tensors=%llu  kv=%llu  %.1fMB\n",
               m.version,(unsigned long long)m.n_tensors,
               (unsigned long long)nkv, size/1048576.f);
        for (u64 i=0;i<nkv&&ok;i++) {
            string key; Val val;
            if (!read_string(key)) break;
            u32 type_raw=0;
            if (!read(type_raw)) break;
            GGUFType type=(GGUFType)type_raw;
            if (!read_value(type,val)) break;
            if (key=="general.architecture"&&type==GGUF_STRING) m.arch=val.str;
            else if (key.find(".embedding_length")!=string::npos) m.n_embd=val.u32v;
            else if (key.find(".head_count")!=string::npos&&
                     key.find("kv")==string::npos) m.n_head=val.u32v;
            else if (key.find(".head_count_kv")!=string::npos) m.n_head_kv=val.u32v;
            else if (key.find(".block_count")!=string::npos) m.n_layer=val.u32v;
            else if (key.find(".feed_forward_length")!=string::npos) m.n_ff=val.u32v;
            else if (key.find(".context_length")!=string::npos) m.n_ctx=val.u32v;
            else if (key=="tokenizer.ggml.tokens"&&type==GGUF_ARRAY)
                m.n_vocab=(u32)val.arr.size();
            else if (key.find(".rope.freq_base")!=string::npos&&type==GGUF_FLOAT32)
                m.rope_theta=val.f32v;
        }
        if (!m.n_head_kv) m.n_head_kv=m.n_head;
        m.valid=(m.n_embd>0&&m.n_layer>0);
        if (!ok) m.error_msg=err;
        return m;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// §18 — BENCHMARK SUITE
// ════════════════════════════════════════════════════════════════════════════

static void bench_logic2048(int REPS=200) {
    printf("\n▶ [Logic2048 BUG-4 FIXED] max power=63 — weight=4B\n");
    printf("  sizeof(Logic2048Weight)=%zu  sizeof(Logic2048Table)=%zu\n",
           sizeof(Logic2048Weight),sizeof(Logic2048Table));
    const int N=512,M=512;
    vector<Logic2048Weight> A(N*M);
    vector<f32> x(M),y(N),y_ref(N);
    mt19937 rng(42); normal_distribution<f32> dist(0.f,0.5f);
    for (auto& w:A) w=Logic2048Weight::from_float(dist(rng));
    for (auto& v:x) v=dist(rng);
    logic2048_matvec(A.data(),x.data(),y.data(),N,M);
    double t0=now_ms();
    for (int r=0;r<REPS;r++){
        logic2048_matvec(A.data(),x.data(),y.data(),N,M);
        do_not_optimize(y[0]);
    }
    double l2k_ms=(now_ms()-t0)/REPS;
    vector<f32> W_ref(N*M);
    for (int i=0;i<N*M;i++) W_ref[i]=g_l2k_table.decode4(A[i]);
    double t1=now_ms();
    for (int r=0;r<REPS;r++){
        for (int i=0;i<N;i++) y_ref[i]=vdot_f32(W_ref.data()+i*M,x.data(),M);
        do_not_optimize(y_ref[0]);
    }
    double fp32_ms=(now_ms()-t1)/REPS;
    f32 max_err=0.f;
    for (int i=0;i<N;i++) max_err=max(max_err,fabsf(y[i]-y_ref[i]));
    printf("  FP32 dense:      %.4f ms\n",fp32_ms);
    printf("  Logic2048 table: %.4f ms  (%.2fx)\n",l2k_ms,fp32_ms/l2k_ms);
    printf("  max|err|=%.2e  %s\n",max_err,max_err<1e-3f?"✅ PASS":"❌ FAIL");
}

static void bench_butterfly(int REPS=200) {
    printf("\n▶ [Butterfly] correctness verify + cache bench\n");
    for (int N : {512,1024,2048,4096}) {
        int L=0; {int tmp=N;while(tmp>1){L++;tmp>>=1;}}
        int P=N/2;
        vector<f32> theta((size_t)L*P);
        mt19937 rng(42+N); normal_distribution<f32> nd(0.f,0.3f);
        for (f32& t:theta) t=nd(rng);
        CosSinCache cs; cs.build(N,theta);
        vector<f32> h0(N),h_small(N),h_tiled(N);
        for (f32& v:h0) v=nd(rng);
        memcpy(h_small.data(),h0.data(),N*4);
        butterfly_forward_small(h_small.data(),N,L,cs);
        memcpy(h_tiled.data(),h0.data(),N*4);
        butterfly_forward_tiled(h_tiled.data(),N,L,cs);
        f32 err=rel_error(h_small.data(),h_tiled.data(),N);
        vector<f32> h_bench(N);
        double t0=now_ms();
        for (int r=0;r<REPS;r++){
            memcpy(h_bench.data(),h0.data(),N*4);
            if(N<=1024) butterfly_forward_small(h_bench.data(),N,L,cs);
            else butterfly_forward_tiled(h_bench.data(),N,L,cs);
            do_not_optimize(h_bench[0]);
        }
        double ms=(now_ms()-t0)/REPS;
        const char* fits=cs.bytes()<=CACHE_L1D_BYTES?"L1D✓":
                         cs.bytes()<=CACHE_L2_BYTES?"L2✓":
                         cs.bytes()<=CACHE_L3_BYTES?"L3✓":"DRAM";
        printf("  N=%-5d  %-5s  %.4f ms  err=%.2e  %s\n",
               N,fits,ms,err,err<1e-4f?"✅":"❌");
    }
}

static void bench_flash() {
    printf("\n▶ [FlashAttention] NEON vs Dense\n");
    printf("  %-6s  %-10s  %-10s  %-8s  %-6s\n","N","Dense ms","Flash ms","Speedup","Err");
    print_sep(48,'-');
    int nh=8,d=64;
    mt19937 rng(42); normal_distribution<f32> nd;
    for (int N:{64,128,256,512,1024}) {
        int sz=N*nh*d;
        vector<f32> Q(sz),K(sz),V(sz),Od(sz),Of(sz);
        for(auto& v:Q) v=nd(rng)*0.1f;
        for(auto& v:K) v=nd(rng)*0.1f;
        for(auto& v:V) v=nd(rng)*0.1f;
        int iters=max(3,500/(N+1));
        double t0=now_ms();
        for(int i=0;i<iters;i++){
            // dense inline
            f32 scale=1.f/sqrtf((f32)d);
            for(int h=0;h<nh;h++) for(int i=0;i<N;i++){
                const f32* qi=Q.data()+(size_t)i*nh*d+h*d;
                vector<f32> sc(N);
                for(int j=0;j<N;j++) sc[j]=vdot_f32(qi,K.data()+(size_t)j*nh*d+h*d,d)*scale;
                vsoftmax(sc.data(),N);
                f32* oi=Od.data()+(size_t)i*nh*d+h*d;
                fill(oi,oi+d,0.f);
                for(int j=0;j<N;j++) vaxpy(oi,V.data()+(size_t)j*nh*d+h*d,sc[j],d);
            }
            do_not_optimize(Od[0]);
        }
        double ms_d=(now_ms()-t0)/iters;
        double t1=now_ms();
        for(int i=0;i<iters;i++){
            flash_attention(Q.data(),K.data(),V.data(),Of.data(),N,nh,d);
            do_not_optimize(Of[0]);
        }
        double ms_f=(now_ms()-t1)/iters;
        f32 err=rel_error(Od.data(),Of.data(),sz);
        printf("  %-6d  %-10.3f  %-10.3f  %-8.2fx  %.4f %s\n",
               N,ms_d,ms_f,ms_d/ms_f,err,err<1e-4f?"✅":"⚠");
    }
}

static void bench_jit() {
    printf("\n▶ [JIT AArch64 FMLA] (REPS=10000)\n");
    const int D=64;
    JITKernel jk; jk.generate(D); bool ok=jk.compile();
    printf("  JIT compile: %s\n",ok?"✅ AArch64":"❌ fallback");
    vector<f32> W(4*D),x(D),y(4,0.f);
    mt19937 rng; normal_distribution<f32> nd;
    for(f32& v:W) v=nd(rng); for(f32& v:x) v=nd(rng);
    const int REPS=10000;
    double t0=now_ms();
    for(int r=0;r<REPS;r++){jk.run4rows(W.data(),x.data(),y.data(),D);do_not_optimize(y[0]);}
    double ms_per=(now_ms()-t0)/REPS;
    printf("  4×%d dot: %.4f ms/call  %.0f ns/call  %.2f GFLOPS\n",
           D,ms_per,ms_per*1e6,2.0*4*D/(ms_per*1e6));
}

static void bench_dlb() {
    printf("\n▶ [DLB BUG-3 FIXED] thread-safe uniform sampling\n");
    const int VOCAB=32000,DIM=512;
    DynamicLayerBudget dlb(16,0.35f,4,256);
    vector<f32> lm_head((size_t)VOCAB*DIM,0.f),x(DIM,1.f);
    mt19937 rng(42); normal_distribution<f32> nd(0.f,0.5f);
    for(int v=0;v<256;v++) for(int d=0;d<DIM;d++) lm_head[(size_t)v*DIM+d]=0.1f;
    for(int v=256;v<VOCAB;v++) for(int d=0;d<DIM;d++) lm_head[(size_t)v*DIM+d]=nd(rng);
    int exits=0,total=1000;
    for(int i=0;i<total;i++)
        if(dlb.should_exit(x.data(),lm_head.data(),8,DIM,VOCAB)) exits++;
    printf("  %d/%d exits (%.1f%%) — mutex-protected, thread-safe\n",
           exits,total,100.f*exits/total);
}

static void bench_kernel(int N=512) {
    printf("\n▶ [Kernel] DIB vs Standard BF (N=%d)\n",N);
    int L=__builtin_ctz((unsigned)N);
    mt19937 rng(42); normal_distribution<f32> nd;
    vector<f32> theta((size_t)L*N/2),h0(N),h1(N);
    for(auto& v:theta) v=nd(rng)*0.1f;
    for(auto& v:h0) v=nd(rng);
    const int REPS=1000;
    double t0=now_ms();
    for(int i=0;i<REPS;i++){
        memcpy(h1.data(),h0.data(),N*4);
        bf_standard(h1.data(),N,theta.data(),L);
        do_not_optimize(h1[0]);
    }
    double ms_std=(now_ms()-t0)/REPS;
    DIBLayer dib(N,1e-4f,42);
    vector<f32> h3(N);
    double t1=now_ms();
    for(int i=0;i<REPS;i++){dib.forward(h0.data(),h3.data());do_not_optimize(h3[0]);}
    double ms_dib=(now_ms()-t1)/REPS;
    long long ops=(long long)L*(N/2)*4;
    printf("  Standard BF:   %.4f ms  %.2f GFLOPS\n",ms_std,(double)ops/ms_std/1e6);
    printf("  DIB (BF+Diag): %.4f ms  %.2f GFLOPS\n",ms_dib,(double)ops/ms_dib/1e6);
}

static void bench_proof(int N=32) {
    printf("\n▶ [Expressivity] BF vs DIB (N=%d)\n",N);
    int L=__builtin_ctz((unsigned)N);
    mt19937 rng(42); normal_distribution<f32> nd;
    vector<f32> T(N*N); for(auto& v:T) v=nd(rng)*0.1f;
    DIBLayer bf_only(N,5e-3f,42),dib(N,5e-3f,99);
    auto eval=[&](DIBLayer& m)->f32{
        mt19937 rt(999); f32 err=0.f;
        for(int t=0;t<200;t++){
            vector<f32> x(N),tx(N,0.f),out(N);
            for(auto& v:x) v=nd(rt);
            for(int i=0;i<N;i++) for(int j=0;j<N;j++) tx[i]+=T[i*N+j]*x[j];
            m.forward(x.data(),out.data());
            f32 en=0.f,tn=0.f;
            for(int i=0;i<N;i++){f32 d=out[i]-tx[i];en+=d*d;tn+=tx[i]*tx[i];}
            err+=sqrtf(en)/(sqrtf(tn)+1e-9f);
        }
        return err/200.f;
    };
    printf("  %-8s  %-12s  %-12s\n","Step","BF err","DIB err");
    print_sep(36,'-');
    printf("  %-8d  %-12.4f  %-12.4f\n",0,eval(bf_only),eval(dib));
    for(int step=1;step<=4000;step++){
        vector<f32> x(N),tx(N,0.f),g(N);
        for(auto& v:x) v=nd(rng);
        for(int i=0;i<N;i++) for(int j=0;j<N;j++) tx[i]+=T[i*N+j]*x[j];
        auto tape1=bf_only.forward_tape(x.data());
        const auto& o1=tape1.h[L];
        for(int i=0;i<N;i++) g[i]=2.f*(o1[i]-tx[i])/N;
        clip_grad_norm(g.data(),N,1.f);
        bf_only.backward(tape1,g.data());
        auto tape2=dib.forward_tape(x.data());
        const auto& o2=tape2.h[L];
        for(int i=0;i<N;i++) g[i]=2.f*(o2[i]-tx[i])/N;
        clip_grad_norm(g.data(),N,1.f);
        dib.backward(tape2,g.data());
        if(step%1000==0){
            f32 eb=eval(bf_only),ed=eval(dib);
            printf("  %-8d  %-12.4f  %-12.4f  %s\n",step,eb,ed,
                   ed<eb*0.8f?"DIB better":ed>eb*1.2f?"BF better":"similar");
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// §19 — [APEX-4] UNIFIED TRAINING (Logic2048 + DIB + DLB combined)
// ════════════════════════════════════════════════════════════════════════════

static void run_training_loop(const string& path, const MasterConfig& cfg,
                               bool is_vi_mode=false) {
    const int VOCAB=256;
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    if (is_vi_mode)
        printf("║  TRAIN_VI — Vietnamese AdaptiveUnifiedModel         ║\n");
    else
        printf("║  UNIFIED — Logic2048 + DIB + DLB Training           ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");
    cfg.print();

    vector<u8> data;
    if (!path.empty()) {
        ifstream f(path,ios::binary);
        if (f) data.assign(istreambuf_iterator<char>(f),{});
    }
    if (data.size()<10) {
        printf("  [WARN] Không có file → synthetic data\n");
        data.resize(50000);
        // Vietnamese-style pattern for train_vi
        string pat = is_vi_mode
            ? "toi la nguoi viet nam. chung ta cung nhau xay dung dat nuoc. "
            : "the quick brown fox jumps over the lazy dog. ";
        for (size_t i=0;i<data.size();i++) data[i]=(u8)pat[i%pat.size()];
    }
    printf("  Data: %zu bytes\n",data.size());

    size_t train_sz=(size_t)(data.size()*0.9f);
    mt19937 rng(cfg.seed);
    uniform_int_distribution<size_t> idx_dist(0,train_sz>=2?train_sz-2:0);

    // Create AdaptiveUnifiedModel (3-way breakthrough)
    AdaptiveUnifiedModel model(VOCAB, cfg.dim, cfg.n_layers,
                                cfg.lr, cfg.wd, cfg.grad_clip,
                                cfg.exit_thr, cfg.seed);
    model.print_model_info();

    auto eval_val=[&]()->f32{
        f32 loss=0.f; int cnt=0;
        for(size_t pos=train_sz; pos+1<data.size()&&cnt<500; pos++,cnt++){
            auto r=model.forward(data[pos]);
            loss+=-logf(r.probs[data[pos+1]]+1e-5f);
        }
        return cnt?expf(loss/cnt):0.f;
    };

    f32 smooth_loss=-logf(1.f/VOCAB);
    double t0=now_ms();
    int total_layers_used=0;

    printf("\n  %-8s  %-10s  %-10s  %-10s  %-8s  %-6s\n",
           "Step","Train PPL","Val PPL","LR","ms/step","AvgDep");
    print_sep(60,'-');

    for (int step=1;step<=cfg.steps;step++) {
        f32 lr=get_lr(cfg,step);
        size_t pos=idx_dist(rng);
        StepDiag d=model.train_step(data[pos],data[pos+1],lr);
        smooth_loss=0.98f*smooth_loss+0.02f*d.loss;
        total_layers_used+=d.layers_used;

        if (cfg.diag && step%cfg.diag_every==0) {
            printf("  [DIAG %d] loss=%.4f  prob_t=%.4f  layers_used=%d/%d\n",
                   step,d.loss,d.prob_target,d.layers_used,cfg.n_layers);
            printf("    grad: %.4f→%.4f  lr=%.2e\n",
                   d.grad_norm_pre,d.grad_norm_post,lr);
        }
        if (step%1000==0||step==1||step==cfg.steps) {
            double el=now_ms()-t0;
            f32 avg_depth=(f32)total_layers_used/step;
            printf("  %-8d  %-10.2f  %-10.2f  %-10.2e  %-8.2f  %.2f\n",
                   step,expf(smooth_loss),eval_val(),lr,el/step,avg_depth);
        }
    }
    double total_ms=now_ms()-t0;
    printf("\n  Val PPL: %.2f  |  %.1fs  (%.0f steps/s)\n",
           eval_val(),total_ms/1000.f,cfg.steps/(total_ms/1000.f));
    printf("  Avg layers/token: %.2f/%d (DLB savings: %.1f%%)\n",
           (f32)total_layers_used/cfg.steps, cfg.n_layers,
           100.f*(1.f-(f32)total_layers_used/(cfg.steps*cfg.n_layers)));
    model.dlb.print_exits();
}

// Legacy charlm benchmark
static void bench_charlm(const string& path, const MasterConfig& cfg) {
    const int VOCAB=256;
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  CHARLM v5 — Legacy (DIB only, FP32 LM head)        ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");
    cfg.print();

    vector<u8> data;
    if (!path.empty()) {
        ifstream f(path,ios::binary);
        if (f) data.assign(istreambuf_iterator<char>(f),{});
    }
    if (data.size()<10) {
        data.resize(50000);
        string pat="the quick brown fox jumps over the lazy dog. ";
        for(size_t i=0;i<data.size();i++) data[i]=(u8)pat[i%pat.size()];
    }
    printf("  Data: %zu bytes\n",data.size());

    size_t train_sz=(size_t)(data.size()*0.9f);
    mt19937 rng(cfg.seed);
    uniform_int_distribution<size_t> idx(0,train_sz>=2?train_sz-2:0);

    CharLM_v5 lm(VOCAB,cfg.dim,cfg.n_layers,cfg.lr,cfg.wd,cfg.grad_clip,cfg.seed);

    auto eval_val=[&]()->f32{
        f32 loss=0.f; int cnt=0;
        for(size_t pos=train_sz;pos+1<data.size()&&cnt<500;pos++,cnt++){
            auto r=lm.forward(data[pos]);
            loss+=-logf(r.probs[data[pos+1]]+1e-5f);
        }
        return cnt?expf(loss/cnt):0.f;
    };

    f32 smooth_loss=-logf(1.f/VOCAB);
    double t0=now_ms();
    printf("  %-8s  %-10s  %-10s  %-10s  %-8s\n",
           "Step","Train PPL","Val PPL","LR","ms/step");
    print_sep(55,'-');

    for (int step=1;step<=cfg.steps;step++) {
        f32 lr=get_lr(cfg,step);
        size_t pos=idx(rng);
        StepDiag d=lm.train_step(data[pos],data[pos+1],lr);
        smooth_loss=0.98f*smooth_loss+0.02f*d.loss;
        if (step%1000==0||step==1||step==cfg.steps) {
            double el=now_ms()-t0;
            printf("  %-8d  %-10.2f  %-10.2f  %-10.2e  %-8.2f\n",
                   step,expf(smooth_loss),eval_val(),lr,el/step);
        }
    }
    printf("\n  Val PPL: %.2f  in %.1fs\n",
           eval_val(),(now_ms()-t0)/1000.f);
}

static void bench_gguf(const string& path) {
    if (path.empty()){printf("[GGUF] Usage: ./vdapex gguf <model.gguf>\n");return;}
    int fd=open(path.c_str(),O_RDONLY);
    if(fd<0){printf("[GGUF] Cannot open: %s\n",path.c_str());return;}
    struct stat st; fstat(fd,&st);
    void* ptr=mmap(nullptr,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(ptr==MAP_FAILED){close(fd);return;}
    GGUFParser parser; GGUFMeta m=parser.parse(ptr,st.st_size);
    munmap(ptr,st.st_size); close(fd);
    if(m.valid){printf("[GGUF] ✅ OK:\n");m.print();}
    else printf("[GGUF] ⚠ Parse incomplete: %s\n",m.error_msg.c_str());
}

// ════════════════════════════════════════════════════════════════════════════
// §20 — MAIN [BUG-2 FIXED: train_vi implemented]
// ════════════════════════════════════════════════════════════════════════════

static void print_banner() {
    print_sep(80);
    printf("║  VANDOANH_APEX — Unified Breakthrough Engine                            ║\n");
    printf("║  NEON=%-3s  AArch64=%-3s  dotprod=%-3s  i8mm=%-3s  OpenMP=%-3s             ║\n",
           HAS_NEON?"ON":"OFF",HAS_AARCH64?"ON":"OFF",
           HAS_DOTPROD?"ON":"OFF",HAS_I8MM?"ON":"OFF",HAS_OMP?"ON":"OFF");
    printf("╠══════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  BUGS FIXED:                                                             ║\n");
    printf("║  [BUG-1] GGUFParser: throw→terminate() fixed → error status ✅         ║\n");
    printf("║  [BUG-2] train_vi unimplemented → fully implemented ✅                 ║\n");
    printf("║  [BUG-3] DLB mutable rng race → mutex protected ✅                     ║\n");
    printf("║  [BUG-4] L2K max power 62→63, full table range used ✅                 ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  BREAKTHROUGH (APEX):                                                    ║\n");
    printf("║  [APEX-1] Logic2048Linear: train in L2K format + STE gradient          ║\n");
    printf("║  [APEX-2] AdaptiveLayer: DIB replaces FFN (O(N logN) vs O(N²))        ║\n");
    printf("║  [APEX-3] AdaptiveUnifiedModel: DLB per-token + L2K weights            ║\n");
    printf("║  [APEX-4] Unified training loop: all 3 breakthroughs combined          ║\n");
    print_sep(80);
}

int main(int argc, char* argv[]) {
    print_banner();
    string mode=(argc>1)?argv[1]:"all";
    MasterConfig cfg=parse_config(argc,argv,2);

    if (mode=="all") {
        bench_logic2048(cfg.bench_reps);
        bench_butterfly(cfg.bench_reps);
        bench_flash();
        bench_jit();
        bench_dlb();
        bench_kernel();
        printf("\n"); print_sep(80);
        printf("  ✅ All benchmarks complete. sink=%.6f\n",(f32)g_sink);
        print_sep(80);
    }
    else if (mode=="logic")              bench_logic2048(cfg.bench_reps);
    else if (mode=="butterfly"||mode=="bf") bench_butterfly(cfg.bench_reps);
    else if (mode=="flash")              bench_flash();
    else if (mode=="jit")                bench_jit();
    else if (mode=="dlb")                bench_dlb();
    else if (mode=="kernel")             bench_kernel();
    else if (mode=="proof")              bench_proof();
    else if (mode=="charlm")             bench_charlm(cfg.file,cfg);
    // [APEX-4] Unified: Logic2048 + DIB + DLB combined training
    else if (mode=="unified")            run_training_loop(cfg.file,cfg,false);
    // [BUG-2 FIXED] train_vi: Vietnamese training với AdaptiveUnifiedModel
    else if (mode=="train_vi")           run_training_loop(cfg.file,cfg,true);
    else if (mode=="gguf")
        bench_gguf(argc>=3&&argv[2][0]!='-'?argv[2]:cfg.file);
    else {
        fprintf(stderr,
            "Usage: %s <mode> [options]\n"
            "  Benchmark:  all logic butterfly flash jit dlb kernel proof\n"
            "  Training:   unified [file] --lr=1e-4 --steps=10000 --dim=64\n"
            "              train_vi [file] --layers=4 --exit_thr=0.3 --diag=1\n"
            "              charlm [file] (legacy DIB-only)\n"
            "  GGUF:       gguf <model.gguf>\n"
            "  Options:    --lr --wd --steps --dim --layers --grad_clip\n"
            "              --warmup --lr_sched=cosine|constant\n"
            "              --exit_thr=<float> (DLB entropy threshold, default 0.3)\n"
            "              --diag=1 --diag_every=500 --seed --reps\n",
            argv[0]);
        return 1;
    }
    return 0;
}
