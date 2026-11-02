// Complete CUDA teaching baseline. Backend compilation/GPU execution not performed by the author.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(err)); } while (0)
constexpr int THREADS = 256;  // Multiple of 32; every warp participates in its collective.
enum Op { SUM, MAXIMUM, DOT };

template<Op op> __device__ float combine(float a, float b) {
    if constexpr (op == MAXIMUM) return fmaxf(a, b);
    return a + b;
}
template<Op op> __device__ float neutral() {
    if constexpr (op == MAXIMUM) return -CUDART_INF_F;
    return 0.0f;
}
template<Op op> __device__ float warp_reduce(float value) {
    for (int offset = 16; offset; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffffu, value, offset);
        value = combine<op>(value, other);
    }
    return value;  // Only lane 0 is consumed as the warp's complete reduction.
}

template<Op op, bool vec4 = false, bool atomic = false>
__global__ void reduce_stage(const float* x, const float* y, float* output, int n) {
    __shared__ float warp_totals[THREADS / 32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int index = (blockIdx.x * THREADS + tid) * (vec4 ? 4 : 1);
    float value = neutral<op>();
    if constexpr (vec4) {
        // Valid only for DOT. Base pointers from cudaMalloc; index is a multiple of 4.
        if (index + 3 < n) {
            float4 a = *reinterpret_cast<const float4*>(x + index);
            float4 b = *reinterpret_cast<const float4*>(y + index);
            value = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
        } else {
            for (int j = 0; j < 4; ++j)
                if (index + j < n) value += x[index+j] * y[index+j];
        }
    } else if (index < n) {
        if constexpr (op == DOT) value = x[index] * y[index];
        else value = x[index];
    }
    // Do not return early for an input tail: every launched warp is full.
    value = warp_reduce<op>(value);
    if (lane == 0) warp_totals[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < THREADS / 32 ? warp_totals[lane] : neutral<op>();
        value = warp_reduce<op>(value);
        if (tid == 0) {
            if constexpr (atomic) atomicAdd(output, value);
            else output[blockIdx.x] = value;
        }
    }
}

struct Buffers {
    float *x=nullptr, *y=nullptr, *a=nullptr, *b=nullptr, *out=nullptr;
    explicit Buffers(int n) {
        auto alloc=[](float** p, size_t count) { CUDA(cudaMalloc(p, std::max(size_t(1), count)*sizeof(float))); };
        alloc(&x,n); alloc(&y,n); alloc(&a,(n+THREADS-1)/THREADS); alloc(&b,(n+THREADS-1)/THREADS); alloc(&out,1);
    }
    ~Buffers() { cudaFree(x); cudaFree(y); cudaFree(a); cudaFree(b); cudaFree(out); }
};

void launch(Buffers& b, int n, Op op, bool vec4, bool atomic) {
    if (n == 0) {
        float identity = op == MAXIMUM ? -std::numeric_limits<float>::infinity() : 0.0f;
        CUDA(cudaMemcpy(b.out,&identity,sizeof(float),cudaMemcpyHostToDevice));
        return;
    }
    int groups = (n + THREADS*(vec4?4:1)-1)/(THREADS*(vec4?4:1));
    float* dst = atomic || groups==1 ? b.out : b.a;
    if (atomic) CUDA(cudaMemsetAsync(b.out,0,sizeof(float))); // Included in timed workflow.
    if (op == MAXIMUM) reduce_stage<MAXIMUM><<<groups,THREADS>>>(b.x,b.y,dst,n);
    else if (op == SUM) {
        if (atomic) reduce_stage<SUM,false,true><<<groups,THREADS>>>(b.x,b.y,dst,n);
        else reduce_stage<SUM><<<groups,THREADS>>>(b.x,b.y,dst,n);
    } else if (vec4) {
        if (atomic) reduce_stage<DOT,true,true><<<groups,THREADS>>>(b.x,b.y,dst,n);
        else reduce_stage<DOT,true><<<groups,THREADS>>>(b.x,b.y,dst,n);
    } else {
        if (atomic) reduce_stage<DOT,false,true><<<groups,THREADS>>>(b.x,b.y,dst,n);
        else reduce_stage<DOT><<<groups,THREADS>>>(b.x,b.y,dst,n);
    }
    CUDA(cudaGetLastError());
    if (atomic) return;
    while (groups > 1) {
        const float* src=dst;
        int length=groups; groups=(length+THREADS-1)/THREADS;
        dst=groups==1 ? b.out : (src==b.a ? b.b : b.a);
        if (op==MAXIMUM) reduce_stage<MAXIMUM><<<groups,THREADS>>>(src,src,dst,length);
        else reduce_stage<SUM><<<groups,THREADS>>>(src,src,dst,length); // DOT partials must be summed.
        CUDA(cudaGetLastError());
    }
}

void run_case(int n, Op op, bool vec4, bool atomic, bool timing) {
    std::vector<float> x(n), y(n);
    double ref=op==MAXIMUM ? -std::numeric_limits<double>::infinity() : 0.0, magnitude=0.0;
    for(int i=0;i<n;++i) {
        x[i]=op==MAXIMUM ? -1.0f-float(i%29)/29 : float((i%23)-11)/11;
        y[i]=float((i%19)-9)/9;
        double v=op==DOT ? double(x[i])*double(y[i]) : x[i];
        if(op==MAXIMUM) ref=std::max(ref,v); else ref+=v;
        magnitude+=std::abs(v);
    }
    Buffers b(n);
    if(n) { CUDA(cudaMemcpy(b.x,x.data(),n*sizeof(float),cudaMemcpyHostToDevice)); CUDA(cudaMemcpy(b.y,y.data(),n*sizeof(float),cudaMemcpyHostToDevice)); }
    float got;
    for(int repeat=0;repeat<2;++repeat) {
        launch(b,n,op,vec4,atomic);
        CUDA(cudaMemcpy(&got,b.out,sizeof(float),cudaMemcpyDeviceToHost));
        bool ok=op==MAXIMUM ? double(got)==ref : std::isfinite(got) && std::abs(double(got)-ref)<=1e-6+2e-6*magnitude;
        if(!ok) throw std::runtime_error("numerical validation failed for n="+std::to_string(n));
    }
    std::cout<<"PASS n="<<n<<" result="<<got<<" fp64_ref="<<ref;
    if(timing && n) {
        for(int i=0;i<5;++i) launch(b,n,op,vec4,atomic);
        CUDA(cudaDeviceSynchronize());
        cudaEvent_t start,end; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
        CUDA(cudaEventRecord(start));
        for(int i=0;i<50;++i) launch(b,n,op,vec4,atomic);
        CUDA(cudaEventRecord(end)); CUDA(cudaEventSynchronize(end));
        float ms; CUDA(cudaEventElapsedTime(&ms,start,end));
        CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
        std::cout<<" complete_workflow_ms="<<ms/50;
    }
    std::cout<<'\n';
}

int main(int argc,char** argv) {
    try {
        int n=argc>1 ? std::stoi(argv[1]) : 1003;
        std::string name=argc>2 ? argv[2] : "sum", method=argc>3 ? argv[3] : "two-pass";
        if(n<0 || n>(1<<26)) throw std::invalid_argument("n must be in [0,2^26]");
        if(name!="sum" && name!="max" && name!="dot" && name!="dot-vec4") throw std::invalid_argument("operation: sum|max|dot|dot-vec4");
        if(method!="two-pass" && method!="atomic") throw std::invalid_argument("method: two-pass|atomic");
        Op op=name=="max" ? MAXIMUM : name=="sum" ? SUM : DOT;
        bool atomic=method=="atomic", vec4=name=="dot-vec4";
        if(atomic && op==MAXIMUM) throw std::invalid_argument("atomic demo supports sum/dot, not max");
        for(int size : {0,1,31,32,33,127,128,129,255,256,257,520,65537}) run_case(size,op,vec4,atomic,false);
        run_case(n,op,vec4,atomic,true);
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
