{ lib }:

let
  inherit (lib) throwIfNot;

  # GPU backends the Colibri Makefile supports on Linux. `cpu` is the
  # default dependency-free build; `hip`/`cuda` compile the same
  # backend_cuda.cu through hipcc/nvcc (one source, two vendors). There is
  # deliberately no `vulkan` here: enabling it is a separate validation
  # decision, never a silent fallback.
  backends = ["cpu" "hip" "cuda"];

  # Engine families whose MoE expert execution the GPU tier covers at the
  # pinned Colibri rev. The GLM engine (colibri.c) compiles COLI_CUDA paths
  # for resident dense tensors only -- its experts stream from disk -- so a
  # `hip`/`cuda` build does NOT make glm53 VRAM inference, and claiming it
  # would disguise hybrid RAM inference as VRAM-only. The module layer
  # rejects non-cpu backends for those engines; this function only builds.
  gpuTierEngines = ["qwen36" "qwen38" "deepseek_v41"];

in
{
  inherit backends gpuTierEngines;

  # Build a Colibri package with an explicit GPU backend.
  #
  # The upstream Makefile honors environment for every GPU knob (`HIP`,
  # `HIP_ARCH`, `ROCM_HOME`, `CUDA`, `CUDA_ARCH`, `CUDA_HOME`,
  # `NVCC_CCBIN` are all `?=`), so no buildPhase copy is needed: the
  # override only adds env vars plus the toolchain. That keeps the
  # expression robust across upstream rev bumps.
  #
  # Arguments:
  #   basePackage: inputs.colibri.packages.${system}.colibri (CPU build).
  #   backend: one of `backends`.
  #   gpuArch: `HIP_ARCH` (e.g. "gfx1100") or `CUDA_ARCH` (e.g. "sm_89"
  #     or "portable"); ignored for cpu.
  #   rocmPackages: ROCm set providing clr (bin/hipcc, libamdhip64) and
  #     rocwmma (matrix-core kernel headers for WMMA-capable archs).
  #   cudaPackages: CUDA redist set (cuda_nvcc, cuda_cudart, libcublas).
  #   symlinkJoin: pkgs.symlinkJoin, for assembling CUDA_HOME.
  #   nvccHostCc: package providing the g++ nvcc drives (CUDA 12.x does
  #     not support gcc 15; the Makefile's NVCC_CCBIN exists for exactly
  #     this, so pass gcc14 rather than -allow-unsupported-compiler).
  mkColibriGpu =
    { basePackage
    , backend
    , gpuArch ? null
    , rocmPackages ? null
    , cudaPackages ? null
    , symlinkJoin ? null
    , nvccHostCc ? null
    }:
    throwIfNot (builtins.elem backend backends)
      "infernix colibri packaging: unknown backend '${backend}' (want one of ${lib.concatStringsSep ", " backends})"
      (throwIfNot (backend == "cpu" || gpuArch != null)
        "infernix colibri packaging: backend '${backend}' needs an explicit gpuArch (e.g. HIP gfx1100, CUDA sm_89)"
        (throwIfNot (backend != "hip" || rocmPackages != null)
          "infernix colibri packaging: backend 'hip' needs rocmPackages (clr)"
          (throwIfNot (backend != "cuda" || (cudaPackages != null && symlinkJoin != null && nvccHostCc != null))
            "infernix colibri packaging: backend 'cuda' needs cudaPackages, symlinkJoin, and nvccHostCc"
            (basePackage.overrideAttrs (prev:
              let
                cudaHome = prev.cudaHome or (symlinkJoin {
                  name = "colibri-cuda-home";
                  paths = with cudaPackages; [cuda_nvcc cuda_cudart libcublas];
                });
              in
              {
                pname = "${prev.pname or "colibri"}-${backend}";
                nativeBuildInputs = (prev.nativeBuildInputs or [])
                  ++ lib.optionals (backend == "hip") [rocmPackages.clr rocmPackages.rocwmma]
                  ++ lib.optionals (backend == "cuda")
                    (with cudaPackages; [cuda_nvcc cuda_cudart libcublas]);
                # The Makefile links -L$(ROCM_HOME)/lib -lamdhip64 and
                # -L$(CUDA_HOME)/lib64 -lcudart -lcublas*; NIX_LDFLAGS from
                # these inputs covers the same dirs if the vars ever drift.
                # rocwmma rides buildInputs so its headers land on the
                # include path: WMMA-capable archs (e.g. gfx1100) fail the
                # build loudly without them (backend_gpu_compat.h), and
                # compiling them out via -DCOLI_HIP_NO_WMMA would silently
                # surrender the tensor-core kernels on capable hardware.
                buildInputs = (prev.buildInputs or [])
                  ++ lib.optionals (backend == "hip") [rocmPackages.clr rocmPackages.rocwmma]
                  ++ lib.optionals (backend == "cuda")
                    (with cudaPackages; [cuda_cudart libcublas]);
              }
              // lib.optionalAttrs (backend == "hip") {
                HIP = "1";
                HIP_ARCH = gpuArch;
                ROCM_HOME = "${rocmPackages.clr}";
              }
              // lib.optionalAttrs (backend == "cuda") {
                CUDA = "1";
                CUDA_ARCH = gpuArch;
                CUDA_HOME = "${cudaHome}";
                NVCC_CCBIN = "${lib.getExe' nvccHostCc "g++"}";
              }
              // {
                passthru = (prev.passthru or {}) // {
                  colibriBackend = backend;
                  colibriGpuArch = gpuArch;
                };
              })))));
}
