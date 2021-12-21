abstract type RunMode end

function setup end
function num_workers end
function array_initializer end
function start_computation end


_array_constructor(type, dims...) = fill(convert(type, NaN), dims)
_sharedarray_constructor(type, dims...) = fill!(SharedArray{type}(dims), NaN)


struct Serial <: RunMode end

setup(::Serial) = nothing
num_workers(::Serial) = 1
array_initializer(::Serial) = _array_constructor

start_computation(f, ::Serial, parameter_chunks) = f(parameter_chunks[1])

struct Threaded <: RunMode
    Nthreads::Int
    function Threaded(Nthreads=Threads.nthreads())
        if Threads.nthreads() < Nthreads
            logmsg("Runmode Threaded: $Nthreads requested but Julia has only $(Threads.nthreads())! Continuing with lower thread count.")
            Nthreads = Threads.nthreads()
        end
        new(Nthreads)
    end
end

setup(::Threaded) = nothing
num_workers(t::Threaded) = t.Nthreads
array_initializer(::Threaded) = _array_constructor

function start_computation(f, ::Threaded, parameter_chunks)
    @sync for chunk in parameter_chunks
        Threads.@spawn begin
            logmsg("Range: $(chunk) on thread #$(Threads.threadid())")
            f(chunk)
        end
    end
end

struct Parallel <: RunMode
    Nprocs::Int
    dosetup::Bool
    NBLASthreads::Int
    useMKL::Bool
    function Parallel(Nprocs, dosetup=true, NBLASthreads=1, useMKL=false)
        new(Nprocs, dosetup, NBLASthreads)
    end
end
function Parallel(; Nprocs=length(Sys.cpu_info()), dosetup=true, NBLASthreads=1, useMKL=false)
    Parallel(Nprocs, dosetup, NBLASthreads, useMKL)
end

setup(p::Parallel) = p.dosetup && _initialize_procs(p.Nprocs, p.NBLASthreads, p.useMKL)
num_workers(p::Parallel) = p.Nprocs
array_initializer(::Parallel) = _sharedarray_constructor

function _initialize_procs(total, num_BLAS_threads, useMKL)
    logmsg("Initialzing worker processes")
    to_add = total
    if workers() != [1]
        to_add = total-length(workers())
    end
    if to_add > 0
        logmsg("Adding $to_add new processes")
        addprocs(to_add; topology=:master_worker)
    end
    @everywhere @eval begin
        import Pkg
        Pkg.activate(".")
        using SimLib, LinearAlgebra
        if $useMKL
            using MKL
        end
        let blas_threads = $num_BLAS_threads
            logmsg("BLAS threads = $blas_threads")
            LinearAlgebra.BLAS.set_num_threads(blas_threads)
        end
    end
end

function start_computation(f, ::Parallel,  parameter_chunks)
    @sync for (p, chunk) in zip(workers(), parameter_chunks)
        @async remotecall_wait(p) do
            logmsg("Range: $(chunk) on process #$(myid())")
            f(chunk)
        end
    end
end
